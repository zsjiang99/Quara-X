"""
Channel Noise Sensitivity Analysis for GradCAM Explainability (IMPROVED VERSION)
=================================================================================
Research Question: How do different channel groups (top/mid/bottom importance)
                   contribute to GradCAM explainability?

Improvements over v1:
1. ✅ Normalized noise strength (relative to activation/weight std)
2. ✅ Multiple runs with averaging to reduce randomness
3. ✅ Support both activation and weight noise injection

Experiment Design:
1. Measure channel importance and divide into 3 groups:
   - Top 20%: Most important channels
   - Mid 60%: Medium importance channels
   - Bottom 20%: Least important channels

2. Test 8 noise injection strategies (2^3 combinations):
   [Top, Mid, Bottom] where each can be Noise/Clean

3. For each strategy, vary noise strength from 0.0 to 1.0
   - Noise is normalized relative to signal std
   - Each point is averaged over multiple runs

4. Plot 8 curves showing how explainability degrades with noise

Author: Experiment Script (Improved)
Date: 2025-10-30
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==========================================================
# Channel Importance Measurement
# ==========================================================
class ChannelImportanceAnalyzer:
    """Analyze channel importance for GradCAM contribution"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input_data, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_importance(self, input_tensor, target_class, method='combined'):
        """Compute channel importance scores"""
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        scores = output[:, target_class]

        # Backward pass
        scores.sum().backward()

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        # Compute GradCAM weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        if method == 'combined':
            # Combined metric: weight * activation magnitude
            weighted_activation = (weights.abs() * activations.abs()).mean(dim=(0, 2, 3))
            importance = weighted_activation  # [C]
        else:
            raise ValueError(f"Unknown importance method: {method}")

        return importance.cpu()


# ==========================================================
# IMPROVED Noise Injection Module
# ==========================================================
class ImprovedNoiseInjectionConv2d(nn.Module):
    """Conv2d layer with normalized noise injection"""

    def __init__(self, conv, noise_mask, noise_strength=0.0,
                 noise_type='activation', normalize=True):
        """
        Args:
            conv: Original Conv2d layer
            noise_mask: Boolean tensor [C] indicating which channels to add noise
            noise_strength: Noise strength (ratio relative to std if normalized)
            noise_type: 'activation' or 'weight'
            normalize: Whether to normalize noise relative to signal std
        """
        super().__init__()
        self.weight = conv.weight.data.clone()
        self.bias = conv.bias.data.clone() if conv.bias is not None else None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.noise_mask = noise_mask  # [C]
        self.noise_strength = noise_strength
        self.noise_type = noise_type
        self.normalize = normalize

        # Pre-compute noisy weights if using weight noise
        if noise_type == 'weight' and noise_strength > 0:
            self.noisy_weight = self._create_noisy_weight()
        else:
            self.noisy_weight = None

    def _create_noisy_weight(self):
        """Create noisy version of weights"""
        noisy_w = self.weight.clone()

        # Add noise to selected channels
        for ch_idx in range(self.weight.shape[0]):
            if self.noise_mask[ch_idx]:
                # Get channel weights
                ch_weights = self.weight[ch_idx]

                # Compute noise
                if self.normalize:
                    # Normalized: noise std = weight_std * noise_strength
                    noise = torch.randn_like(ch_weights) * ch_weights.std() * self.noise_strength
                else:
                    # Absolute noise
                    noise = torch.randn_like(ch_weights) * self.noise_strength

                noisy_w[ch_idx] = ch_weights + noise

        return noisy_w

    def forward(self, x):
        # Select weights
        if self.noisy_weight is not None:
            w = self.noisy_weight
        else:
            w = self.weight

        # Standard convolution
        out = F.conv2d(x, w, bias=self.bias,
                      stride=self.stride, padding=self.padding,
                      dilation=self.dilation, groups=self.groups)

        # Add activation noise if specified
        if self.noise_type == 'activation' and self.noise_strength > 0:
            B, C, H, W = out.shape

            # Create noise tensor (non-inplace operation)
            noise_tensor = torch.zeros_like(out)

            # Fill noise for selected channels
            for ch_idx in range(C):
                if self.noise_mask[ch_idx]:
                    # Get channel activation
                    ch_activation = out[:, ch_idx:ch_idx+1, :, :]

                    # Compute noise
                    if self.normalize:
                        # Normalized: noise std = activation_std * noise_strength
                        act_std = ch_activation.std()
                        if act_std > 0:
                            noise_tensor[:, ch_idx:ch_idx+1, :, :] = \
                                torch.randn_like(ch_activation) * act_std * self.noise_strength
                    else:
                        # Absolute noise
                        noise_tensor[:, ch_idx:ch_idx+1, :, :] = \
                            torch.randn_like(ch_activation) * self.noise_strength

            # Add noise (non-inplace)
            out = out + noise_tensor

        return out


# ==========================================================
# Strategy Configuration
# ==========================================================
def divide_channels_by_importance(importance_scores, top_ratio=0.2, bottom_ratio=0.2):
    """Divide channels into 3 groups based on importance"""
    num_channels = len(importance_scores)
    sorted_indices = torch.argsort(importance_scores, descending=True)

    top_k = int(num_channels * top_ratio)
    bottom_k = int(num_channels * bottom_ratio)

    top_indices = sorted_indices[:top_k]
    bottom_indices = sorted_indices[-bottom_k:]
    mid_indices = sorted_indices[top_k:-bottom_k]

    return top_indices, mid_indices, bottom_indices


def create_noise_mask(num_channels, top_indices, mid_indices, bottom_indices,
                     noise_top, noise_mid, noise_bottom):
    """Create a boolean mask indicating which channels should receive noise"""
    mask = torch.zeros(num_channels, dtype=torch.bool)

    if noise_top:
        mask[top_indices] = True
    if noise_mid:
        mask[mid_indices] = True
    if noise_bottom:
        mask[bottom_indices] = True

    return mask


# ==========================================================
# Model Modification
# ==========================================================
def apply_noise_injection(model_fp32, layer_name, noise_mask, noise_strength,
                         noise_type='activation', normalize=True):
    """Apply noise injection to a specific layer"""
    model_noisy = copy.deepcopy(model_fp32)

    # Navigate to target layer
    parts = layer_name.split('.')
    parent = model_noisy
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Get original conv layer
    original_conv = getattr(parent, parts[-1])

    # Replace with noisy version
    if isinstance(original_conv, nn.Conv2d):
        noisy_conv = ImprovedNoiseInjectionConv2d(
            original_conv, noise_mask, noise_strength,
            noise_type=noise_type, normalize=normalize
        )
        setattr(parent, parts[-1], noisy_conv)

    return model_noisy


# ==========================================================
# GradCAM Implementation
# ==========================================================
class SimpleGradCAM:
    """Simple GradCAM implementation"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input_data, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()

        # Forward
        output = self.model(input_tensor)
        scores = output[:, target_class]

        # Backward
        scores.sum().backward()

        # Compute CAM
        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)

        # Upsample
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                          mode='bilinear', align_corners=False)

        return cam.squeeze(1)


# ==========================================================
# Evaluation Metrics
# ==========================================================
def compute_ssim(heatmap_ref, heatmap_test):
    """Compute SSIM between two heatmaps"""
    from skimage.metrics import structural_similarity

    # SSIM requires 2D arrays
    ssim_value = structural_similarity(heatmap_ref, heatmap_test,
                                       data_range=1.0)
    return ssim_value


def compute_iou(heatmap_ref, heatmap_test, threshold=0.5):
    """Compute IoU between binarized heatmaps"""
    # Binarize at threshold
    mask_ref = (heatmap_ref >= threshold).astype(np.float32)
    mask_test = (heatmap_test >= threshold).astype(np.float32)

    # Compute intersection and union
    intersection = (mask_ref * mask_test).sum()
    union = mask_ref.sum() + mask_test.sum() - intersection

    if union == 0:
        return 1.0

    return intersection / union


def compute_heatmap_similarity(heatmap_ref, heatmap_test):
    """Compute multiple similarity metrics between two heatmaps"""
    # Correlation (original metric)
    corr = np.corrcoef(heatmap_ref.flatten(), heatmap_test.flatten())[0, 1]

    # MAE-based similarity
    mae = np.abs(heatmap_ref - heatmap_test).mean()
    mae_similarity = 1.0 - mae  # Convert to similarity

    # IoU (robust to extreme values) - PRIMARY METRIC
    iou = compute_iou(heatmap_ref, heatmap_test, threshold=0.5)

    return {
        'correlation': corr,
        'mae': mae,
        'mae_similarity': mae_similarity,
        'iou': iou
    }


# ==========================================================
# Data Loading
# ==========================================================
def prepare_tiny_imagenet_val(root):
    """Organize Tiny-ImageNet validation set"""
    val_dir = os.path.join(root, "val")
    img_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(img_dir):
        return

    print("Organizing Tiny-ImageNet validation set...")
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img, label = parts[0], parts[1]
            label_dir = os.path.join(val_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            src = os.path.join(img_dir, img)
            dst = os.path.join(label_dir, img)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)

    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    print("Tiny-ImageNet validation set organized.")


def load_tiny_imagenet(root, batch_size=64, num_workers=4):
    """Load Tiny-ImageNet dataset"""
    prepare_tiny_imagenet_val(root)

    transform = T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageFolder(os.path.join(root, "train"), transform=transform)
    val_set = ImageFolder(os.path.join(root, "val"), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Tiny-ImageNet loaded: {len(train_set)} train / {len(val_set)} val / {len(train_set.classes)} classes")
    return train_loader, val_loader, len(train_set.classes)


# ==========================================================
# Visualization
# ==========================================================
def plot_noise_sensitivity_curves(results_dict, save_path, noise_type='activation'):
    """Plot smooth curves showing how explainability degrades with noise"""
    from scipy.interpolate import make_interp_spline

    # Create figure with better aspect ratio
    fig, ax = plt.subplots(figsize=(12, 7))

    # All 8 strategies with distinct colors
    all_strategies = {
        'Top-Only': {'color': '#E74C3C', 'linewidth': 3.0, 'label': '[N,-,-] Top Only'},
        'Mid-Only': {'color': '#3498DB', 'linewidth': 3.0, 'label': '[-,N,-] Mid Only'},
        'Bottom-Only': {'color': '#2ECC71', 'linewidth': 3.0, 'label': '[-,-,N] Bottom Only'},
        'Top+Mid': {'color': '#F39C12', 'linewidth': 2.5, 'label': '[N,N,-] Top+Mid'},
        'Top+Bottom': {'color': '#8E44AD', 'linewidth': 2.5, 'label': '[N,-,N] Top+Bottom'},
        'Mid+Bottom': {'color': '#1ABC9C', 'linewidth': 2.5, 'label': '[-,N,N] Mid+Bottom'},
        'All-Noise': {'color': '#95A5A6', 'linewidth': 2.5, 'label': '[N,N,N] All Channels'},
        'No-Noise': {'color': '#34495E', 'linewidth': 2.0, 'label': '[-,-,-] Baseline', 'linestyle': '--'},
    }

    for strategy_name, style in all_strategies.items():
        if strategy_name not in results_dict:
            continue

        strategy_results = results_dict[strategy_name]
        noise_levels = np.array(sorted(strategy_results.keys()))
        correlations = np.array([strategy_results[noise]['correlation'] for noise in noise_levels])

        # Create smooth curve using spline interpolation
        if len(noise_levels) > 3 and strategy_name != 'No-Noise':
            # Create denser points for smooth curve
            noise_smooth = np.linspace(noise_levels.min(), noise_levels.max(), 300)

            # Use cubic spline for smoothing
            try:
                spl = make_interp_spline(noise_levels, correlations, k=3)
                corr_smooth = spl(noise_smooth)

                # Plot smooth curve (no shadow)
                ax.plot(noise_smooth, corr_smooth,
                       color=style['color'],
                       linewidth=style['linewidth'],
                       label=style['label'],
                       linestyle=style.get('linestyle', '-'))
            except:
                # Fallback to line plot if spline fails
                ax.plot(noise_levels, correlations,
                       color=style['color'],
                       linewidth=style['linewidth'],
                       label=style['label'],
                       linestyle=style.get('linestyle', '-'))
        else:
            # Plot as line for baseline
            ax.plot(noise_levels, correlations,
                   color=style['color'],
                   linewidth=style['linewidth'],
                   label=style['label'],
                   linestyle=style.get('linestyle', '-'))

    # Styling
    ax.set_xlabel('Noise Strength (σ)', fontsize=16, fontweight='bold')
    ax.set_ylabel('IoU with Baseline', fontsize=16, fontweight='bold')
    ax.set_title('Impact of Channel-wise Noise on GradCAM Explainability (IoU Metric)',
                fontsize=18, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend - two columns for 8 curves
    ax.legend(loc='best', fontsize=11, framealpha=0.95,
             edgecolor='gray', ncol=2)

    # Limits
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([noise_levels.min(), noise_levels.max()])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Clean white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Noise sensitivity curves saved: {save_path}")


# ==========================================================
# Main Experiment
# ==========================================================
def main():
    print("="*80)
    print("Channel Noise Sensitivity Analysis (IMPROVED VERSION)")
    print("="*80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Experiment configuration
    NUM_RUNS = 5  # Number of runs per noise level (reduced since we test multiple images)
    NUM_TEST_IMAGES = 5  # Number of test images to average over
    noise_type = 'activation'  # 'activation' or 'weight'
    normalize = True  # Normalize noise relative to signal std

    print(f"\nExperiment Configuration:")
    print(f"  Noise type: {noise_type}")
    print(f"  Normalized: {normalize}")
    print(f"  Runs per point: {NUM_RUNS}")
    print(f"  Test images: {NUM_TEST_IMAGES}")

    # Load data
    print("\n" + "="*80)
    print("Step 1: Loading Tiny-ImageNet Dataset")
    print("="*80)

    data_root = "/home/zjian137/CAMBoost/gradcam/data/tiny-imagenet"
    if not os.path.exists(data_root):
        print(f"Error: Dataset not found at {data_root}")
        return

    _, val_loader, num_classes = load_tiny_imagenet(root=data_root, batch_size=64, num_workers=4)

    # Load model
    print("\n" + "="*80)
    print("Step 2: Loading ResNet50 Model")
    print("="*80)

    model_fp32 = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, num_classes)
    model_fp32 = model_fp32.to(device)
    model_fp32.eval()

    # Select target layer
    target_layer_name = "layer4.2.conv3"
    target_layer = model_fp32.layer4[2].conv3
    num_channels = target_layer.out_channels

    print(f"Target layer: {target_layer_name}")
    print(f"Number of channels: {num_channels}")

    # Get multiple test images
    print("\n" + "="*80)
    print("Step 3: Selecting Test Images")
    print("="*80)

    test_images = []
    test_labels = []
    test_classes = []

    # Collect images from validation set
    for batch_images, batch_labels in val_loader:
        for i in range(len(batch_images)):
            if len(test_images) >= NUM_TEST_IMAGES:
                break
            test_images.append(batch_images[i:i+1].to(device))
            test_labels.append(batch_labels[i].item())

            # Get predicted class
            with torch.no_grad():
                output = model_fp32(batch_images[i:i+1].to(device))
                pred_class = output.argmax(dim=1).item()
            test_classes.append(pred_class)

        if len(test_images) >= NUM_TEST_IMAGES:
            break

    print(f"Selected {len(test_images)} test images")
    print(f"Sample predictions: {test_classes[:5]}")

    # Define experiment strategies
    strategies = {
        'Top-Only':      (True,  False, False),  # [N, -, -]
        'Mid-Only':      (False, True,  False),  # [-, N, -]
        'Bottom-Only':   (False, False, True),   # [-, -, N]
        'Top+Mid':       (True,  True,  False),  # [N, N, -]
        'Top+Bottom':    (True,  False, True),   # [N, -, N]
        'Mid+Bottom':    (False, True,  True),   # [-, N, N]
        'All-Noise':     (True,  True,  True),   # [N, N, N]
        'No-Noise':      (False, False, False),  # [-, -, -]
    }

    # Define noise levels to test - extended range to show degradation to ~0
    noise_levels = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    # Run experiments across multiple images
    print("\n" + "="*80)
    print("Step 4-6: Running Multi-Image Noise Sensitivity Experiments")
    print("="*80)

    # Storage for results across all images
    all_images_results = {strategy: {noise: [] for noise in noise_levels}
                         for strategy in strategies.keys()}

    # Process each test image
    for img_idx in range(NUM_TEST_IMAGES):
        print(f"\n{'='*80}")
        print(f"Processing Image {img_idx + 1}/{NUM_TEST_IMAGES}")
        print(f"{'='*80}")

        test_image = test_images[img_idx]
        target_class = test_classes[img_idx]

        # Compute channel importance for this image
        analyzer = ChannelImportanceAnalyzer(model_fp32, target_layer)
        with torch.enable_grad():
            importance_scores = analyzer.compute_importance(
                test_image, target_class, method='combined'
            )

        # Divide into groups based on importance
        top_indices, mid_indices, bottom_indices = divide_channels_by_importance(
            importance_scores, top_ratio=0.2, bottom_ratio=0.2
        )

        # Generate baseline heatmap
        cam_baseline = SimpleGradCAM(model_fp32, target_layer)
        with torch.enable_grad():
            heatmap_baseline = cam_baseline(test_image, target_class)
        heatmap_baseline_np = heatmap_baseline.detach().cpu().numpy()[0]

        # Test each strategy
        for strategy_name, (noise_top, noise_mid, noise_bottom) in strategies.items():
            # Create noise mask for this image's channel groups
            noise_mask = create_noise_mask(
                num_channels, top_indices, mid_indices, bottom_indices,
                noise_top, noise_mid, noise_bottom
            )

            # Test each noise level
            for noise_strength in noise_levels:
                # Run multiple times and average
                correlations = []

                for run_idx in range(NUM_RUNS):
                    # Apply noise injection
                    model_noisy = apply_noise_injection(
                        model_fp32, target_layer_name, noise_mask, noise_strength,
                        noise_type=noise_type, normalize=normalize
                    )
                    model_noisy = model_noisy.to(device)

                    # Generate heatmap
                    noisy_layer = model_noisy.layer4[2].conv3
                    cam_noisy = SimpleGradCAM(model_noisy, noisy_layer)

                    with torch.enable_grad():
                        heatmap_noisy = cam_noisy(test_image, target_class)

                    heatmap_noisy_np = heatmap_noisy.detach().cpu().numpy()[0]

                    # Compute metrics (using IoU as primary metric)
                    metrics = compute_heatmap_similarity(heatmap_baseline_np, heatmap_noisy_np)
                    correlations.append(metrics['iou'])  # Use IoU instead of SSIM

                # Store average for this image
                all_images_results[strategy_name][noise_strength].append(np.mean(correlations))

        print(f"  Image {img_idx + 1} complete")

    # Compute final statistics across all images
    print("\n" + "="*80)
    print("Step 7: Computing Statistics Across All Images")
    print("="*80)

    results_dict = {}
    for strategy_name in strategies.keys():
        strategy_results = {}
        for noise_strength in noise_levels:
            # Get all image results for this (strategy, noise) combination
            image_correlations = all_images_results[strategy_name][noise_strength]

            # Compute mean and std across images
            mean_corr = np.mean(image_correlations)
            std_corr = np.std(image_correlations)
            mae = 1.0 - mean_corr  # Approximate MAE

            strategy_results[noise_strength] = {
                'correlation': mean_corr,  # Now using IoU
                'std': std_corr,
                'mae': mae
            }

        results_dict[strategy_name] = strategy_results

        # Print summary for this strategy
        print(f"\n{strategy_name}:")
        for noise_strength in [0.0, 1.0, 5.0, 10.0]:
            if noise_strength in strategy_results:
                iou = strategy_results[noise_strength]['correlation']  # Actually IoU now
                std = strategy_results[noise_strength]['std']
                print(f"  σ={noise_strength:.1f}: IoU={iou:.4f}±{std:.4f}")

    # Visualization
    print("\n" + "="*80)
    print("Step 8: Generating Visualization")
    print("="*80)

    os.makedirs("./results", exist_ok=True)
    plot_noise_sensitivity_curves(
        results_dict,
        save_path="./results/channel_noise_sensitivity_iou.png",
        noise_type=noise_type
    )

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\n📊 Key Findings:")
    print("-" * 80)

    # Compare degradation rates at σ=1.0
    test_noise = 1.0
    print(f"\nIoU at σ={test_noise}:")
    for strategy_name in ['Top-Only', 'Mid-Only', 'Bottom-Only']:
        if strategy_name in results_dict and test_noise in results_dict[strategy_name]:
            iou = results_dict[strategy_name][test_noise]['correlation']  # Actually IoU
            std = results_dict[strategy_name][test_noise]['std']
            print(f"  {strategy_name:<15}: {iou:.4f} ± {std:.4f}")

    # Identify which group is most critical
    top_iou = results_dict['Top-Only'][test_noise]['correlation']
    mid_iou = results_dict['Mid-Only'][test_noise]['correlation']
    bottom_iou = results_dict['Bottom-Only'][test_noise]['correlation']

    top_drop = 1.0 - top_iou
    mid_drop = 1.0 - mid_iou
    bottom_drop = 1.0 - bottom_iou

    print(f"\nIoU drop at σ={test_noise}:")
    print(f"  Top-Only noise:    {top_drop:.4f}")
    print(f"  Mid-Only noise:    {mid_drop:.4f}")
    print(f"  Bottom-Only noise: {bottom_drop:.4f}")

    if top_drop > mid_drop and top_drop > bottom_drop:
        print("\n✅ Conclusion: TOP 20% channels are most critical for explainability!")
        print("   → Prioritize these channels in quantization/compression")
    elif mid_drop > top_drop and mid_drop > bottom_drop:
        print("\n✅ Conclusion: MID 60% channels are most critical for explainability!")
        print("   → Middle channels contribute significantly to heatmap quality")
    else:
        print("\n✅ Conclusion: BOTTOM 20% channels have unexpected importance!")
        print("   → Even low-importance channels affect explainability")

    print("\n" + "="*80)
    print("✅ Experiment Complete!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - ./results/channel_noise_sensitivity_iou.png")
    print("\n")


if __name__ == "__main__":
    main()
