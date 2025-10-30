"""
Channel Noise Sensitivity Analysis using ROAD Metric
=====================================================
This version replaces SSIM with the ROAD (Remove And Debias) metric
to evaluate explainability changes under channel noise.

ROAD evaluates explanation quality by:
1. Removing features according to attribution importance
2. Using noisy linear imputation to fill removed pixels
3. Measuring model performance drop (accuracy/probability)

Author: Experiment Script (ROAD Version)
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
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

# Add ROAD evaluation to path
sys.path.append('/home/zjian137/road_evaluation')
from road import run_road
from road.imputations import NoisyLinearImputer

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
        """
        Compute channel importance scores

        Available methods:
            'gradient': Mean absolute gradient per channel
            'activation': Mean activation magnitude per channel
            'weight': GradCAM weights (α_k) per channel
            'combined': Weight × Activation magnitude (default)
            'gradcam_contribution': Actual contribution to final GradCAM
            'taylor': First-order Taylor approximation
            'variance': Activation variance per channel
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        scores = output[:, target_class]

        # Backward pass
        scores.sum().backward()

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        # Compute GradCAM weights (α_k = global average pooling of gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # ===============================================
        # Method 1: Gradient-based
        # ===============================================
        if method == 'gradient':
            # Importance = mean absolute gradient per channel
            # Measures how sensitive the output is to each channel
            importance = gradients.abs().mean(dim=(0, 2, 3))  # [C]

        # ===============================================
        # Method 2: Activation-based
        # ===============================================
        elif method == 'activation':
            # Importance = mean activation magnitude per channel
            # Measures the average firing strength of each channel
            importance = activations.abs().mean(dim=(0, 2, 3))  # [C]

        # ===============================================
        # Method 3: GradCAM weight-based
        # ===============================================
        elif method == 'weight':
            # Importance = absolute GradCAM weight (α_k) per channel
            # Directly uses the weights from GradCAM formula
            importance = weights.abs().mean(dim=(0, 2, 3))  # [C]

        # ===============================================
        # Method 4: Combined (Weight × Activation)
        # ===============================================
        elif method == 'combined':
            # Combined metric: weight × activation magnitude
            # Measures the actual weighted contribution to GradCAM
            weighted_activation = (weights.abs() * activations.abs()).mean(dim=(0, 2, 3))
            importance = weighted_activation  # [C]

        # ===============================================
        # Method 5: GradCAM Contribution
        # ===============================================
        elif method == 'gradcam_contribution':
            # Actual contribution to the final GradCAM heatmap
            # This computes the per-channel contribution to ReLU(Σ α_k * A_k)
            weighted_activation = weights * activations  # [B, C, H, W]
            cam_contribution = weighted_activation.mean(dim=(2, 3))  # [B, C]
            # Only consider positive contributions (after ReLU)
            importance = F.relu(cam_contribution).mean(dim=0)  # [C]

        # ===============================================
        # Method 6: Taylor Approximation
        # ===============================================
        elif method == 'taylor':
            # First-order Taylor approximation: gradient × activation
            # This measures the approximate change in output if channel is removed
            taylor_importance = (gradients * activations).abs().mean(dim=(0, 2, 3))
            importance = taylor_importance  # [C]

        # ===============================================
        # Method 7: Variance-based
        # ===============================================
        elif method == 'variance':
            # Activation variance per channel
            # High variance = channel is more selective/informative
            mean_act = activations.mean(dim=(2, 3), keepdim=True)
            variance = ((activations - mean_act) ** 2).mean(dim=(0, 2, 3))
            importance = variance  # [C]

        else:
            raise ValueError(
                f"Unknown importance method: {method}. "
                f"Available methods: 'gradient', 'activation', 'weight', "
                f"'combined', 'gradcam_contribution', 'taylor', 'variance'"
            )

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
# Simple Dataset for ROAD Evaluation
# ==========================================================
class SimpleImageDataset(Dataset):
    """Simple dataset wrapper for ROAD evaluation"""

    def __init__(self, images, labels):
        """
        Args:
            images: List of image tensors (already normalized)
            labels: List of label integers
        """
        # Store as CPU tensors to avoid CUDA issues in DataLoader
        self.images = [img.cpu() for img in images]
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return denormalized image for ROAD imputation
        img = self.images[idx]
        label = self.labels[idx]

        # Denormalize from ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        return img, label


# ==========================================================
# ROAD Evaluation Wrapper
# ==========================================================
def evaluate_explanation_with_road(model, test_images, test_labels, heatmaps, device):
    """
    Evaluate explanation quality using ROAD metric

    Args:
        model: The model to evaluate
        test_images: List of test image tensors
        test_labels: List of test labels
        heatmaps: List of heatmap arrays (numpy)
        device: torch device

    Returns:
        road_score: ROAD metric score (higher is better for good explanations)
    """
    # Create simple dataset
    dataset = SimpleImageDataset(test_images, test_labels)

    # Convert heatmaps to attribution format expected by ROAD
    explanations = []
    for heatmap in heatmaps:
        # Heatmap is [H, W], need to expand to [C, H, W] for ROAD
        # Higher values = more important
        attr = np.stack([heatmap] * 3, axis=0)  # [3, H, W]
        explanations.append(attr)

    # Transform to apply after imputation (ImageNet normalization)
    transform_test = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # Run ROAD evaluation with MoRF (Most Relevant First)
    # We test removing 10%, 30%, 50% of pixels
    percentages = [0.1, 0.3, 0.5]

    # Use noisy linear imputer as recommended by ROAD paper
    imputer = NoisyLinearImputer(noise=0.01)

    model.eval()

    try:
        res_acc, prob_acc = run_road(
            model=model,
            dataset_test=dataset,
            explanations_test=explanations,
            transform_test=transform_test,
            percentages=percentages,
            morf=True,  # Most Relevant First
            batch_size=len(test_images),  # Process all at once
            imputation=imputer
        )

        # Compute ROAD score as area under the curve
        # Good explanations should cause larger accuracy drop
        # So we measure the drop: 1.0 - accuracy
        road_score = torch.mean(1.0 - prob_acc).item()

    except Exception as e:
        print(f"ROAD evaluation error: {e}")
        road_score = 0.0

    return road_score


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
        road_scores = np.array([strategy_results[noise]['road'] for noise in noise_levels])

        # Create smooth curve using spline interpolation
        if len(noise_levels) > 3 and strategy_name != 'No-Noise':
            # Create denser points for smooth curve
            noise_smooth = np.linspace(noise_levels.min(), noise_levels.max(), 300)

            # Use cubic spline for smoothing
            try:
                spl = make_interp_spline(noise_levels, road_scores, k=3)
                road_smooth = spl(noise_smooth)

                # Plot smooth curve (no shadow)
                ax.plot(noise_smooth, road_smooth,
                       color=style['color'],
                       linewidth=style['linewidth'],
                       label=style['label'],
                       linestyle=style.get('linestyle', '-'))
            except:
                # Fallback to line plot if spline fails
                ax.plot(noise_levels, road_scores,
                       color=style['color'],
                       linewidth=style['linewidth'],
                       label=style['label'],
                       linestyle=style.get('linestyle', '-'))
        else:
            # Plot as line for baseline
            ax.plot(noise_levels, road_scores,
                   color=style['color'],
                   linewidth=style['linewidth'],
                   label=style['label'],
                   linestyle=style.get('linestyle', '-'))

    # Styling
    ax.set_xlabel('Noise Strength (σ)', fontsize=16, fontweight='bold')
    ax.set_ylabel('ROAD Score (Explanation Quality)', fontsize=16, fontweight='bold')
    ax.set_title('Impact of Channel-wise Noise on GradCAM Explainability (ROAD Metric)',
                fontsize=18, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend - two columns for 8 curves
    ax.legend(loc='best', fontsize=11, framealpha=0.95,
             edgecolor='gray', ncol=2)

    # Limits
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
    print("Channel Noise Sensitivity Analysis (ROAD METRIC VERSION)")
    print("="*80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Experiment configuration
    NUM_RUNS = 2  # Reduced for ROAD (each run is expensive)
    NUM_TEST_IMAGES = 2  # Reduced for computational efficiency
    noise_type = 'activation'
    normalize = True

    # ===================================================================
    # IMPORTANCE CALCULATION METHOD - Choose one of the following:
    # ===================================================================
    # 'gradient'            - Gradient magnitude (sensitivity-based)
    # 'activation'          - Activation magnitude (firing strength)
    # 'weight'              - GradCAM weights α_k (original GradCAM)
    # 'combined'            - Weight × Activation (recommended, default)
    # 'gradcam_contribution'- Actual contribution to final CAM
    # 'taylor'              - Taylor approximation (gradient × activation)
    # 'variance'            - Activation variance (selectivity-based)
    # ===================================================================
    IMPORTANCE_METHOD = 'combined'  # <-- Change this to try different methods

    print(f"\nExperiment Configuration:")
    print(f"  Noise type: {noise_type}")
    print(f"  Normalized: {normalize}")
    print(f"  Runs per point: {NUM_RUNS}")
    print(f"  Test images: {NUM_TEST_IMAGES}")
    print(f"  Metric: ROAD (Remove And Debias)")
    print(f"  Importance method: {IMPORTANCE_METHOD}")  # Show which method is being used

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
    print("Step 2: Loading Trained ResNet50 Model")
    print("="*80)

    # Load pre-trained Tiny-ImageNet model
    model_path = "/home/zjian137/CAMBoost/gradcam/resnet50_tinyimagenet_finetuned.pth"

    model_fp32 = torchvision.models.resnet50(weights=None)
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, num_classes)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model_fp32.load_state_dict(state_dict)

    model_fp32 = model_fp32.to(device)
    model_fp32.eval()

    print(f"Loaded trained model from: {model_path}")

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

    # Define noise levels to test - reduced for faster execution
    noise_levels = [0.0, 1.0, 3.0, 5.0]

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
                test_image, target_class, method=IMPORTANCE_METHOD  # Use configured method
            )

        # Divide into groups based on importance
        top_indices, mid_indices, bottom_indices = divide_channels_by_importance(
            importance_scores, top_ratio=0.2, bottom_ratio=0.2
        )

        # Test each strategy
        for strategy_name, (noise_top, noise_mid, noise_bottom) in strategies.items():
            print(f"\n  Strategy: {strategy_name}")

            # Create noise mask for this image's channel groups
            noise_mask = create_noise_mask(
                num_channels, top_indices, mid_indices, bottom_indices,
                noise_top, noise_mid, noise_bottom
            )

            # Test each noise level
            for noise_strength in noise_levels:
                print(f"    Testing σ={noise_strength}...", end=' ')

                # Run multiple times and average
                road_scores = []

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

                    # Evaluate with ROAD metric
                    road_score = evaluate_explanation_with_road(
                        model_noisy, [test_image[0]], [target_class],
                        [heatmap_noisy_np], device
                    )
                    road_scores.append(road_score)

                # Store average for this image
                avg_road = np.mean(road_scores)
                all_images_results[strategy_name][noise_strength].append(avg_road)
                print(f"ROAD={avg_road:.4f}")

        print(f"\n  Image {img_idx + 1} complete")

    # Compute final statistics across all images
    print("\n" + "="*80)
    print("Step 7: Computing Statistics Across All Images")
    print("="*80)

    results_dict = {}
    for strategy_name in strategies.keys():
        strategy_results = {}
        for noise_strength in noise_levels:
            # Get all image results for this (strategy, noise) combination
            image_road_scores = all_images_results[strategy_name][noise_strength]

            # Compute mean and std across images
            mean_road = np.mean(image_road_scores)
            std_road = np.std(image_road_scores)

            strategy_results[noise_strength] = {
                'road': mean_road,
                'std': std_road,
            }

        results_dict[strategy_name] = strategy_results

        # Print summary for this strategy
        print(f"\n{strategy_name}:")
        for noise_strength in noise_levels:
            road = strategy_results[noise_strength]['road']
            std = strategy_results[noise_strength]['std']
            print(f"  σ={noise_strength:.1f}: ROAD={road:.4f}±{std:.4f}")

    # Visualization
    print("\n" + "="*80)
    print("Step 8: Generating Visualization")
    print("="*80)

    os.makedirs("./results", exist_ok=True)

    # Generate filename with importance method
    filename = f"channel_noise_sensitivity_road_{IMPORTANCE_METHOD}.png"
    save_path = f"./results/{filename}"

    plot_noise_sensitivity_curves(
        results_dict,
        save_path=save_path,
        noise_type=noise_type
    )

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\nKey Findings:")
    print("-" * 80)

    # Compare degradation rates at σ=1.0
    test_noise = 1.0
    print(f"\nROAD Score at σ={test_noise}:")
    print("(Higher = Better explanation quality)")
    for strategy_name in ['Top-Only', 'Mid-Only', 'Bottom-Only']:
        if strategy_name in results_dict and test_noise in results_dict[strategy_name]:
            road = results_dict[strategy_name][test_noise]['road']
            std = results_dict[strategy_name][test_noise]['std']
            print(f"  {strategy_name:<15}: {road:.4f} ± {std:.4f}")

    # Identify which group is most critical
    baseline_road = results_dict['No-Noise'][0.0]['road']
    top_road = results_dict['Top-Only'][test_noise]['road']
    mid_road = results_dict['Mid-Only'][test_noise]['road']
    bottom_road = results_dict['Bottom-Only'][test_noise]['road']

    top_drop = baseline_road - top_road
    mid_drop = baseline_road - mid_road
    bottom_drop = baseline_road - bottom_road

    print(f"\nROAD drop at σ={test_noise}:")
    print(f"  Top-Only noise:    {top_drop:.4f}")
    print(f"  Mid-Only noise:    {mid_drop:.4f}")
    print(f"  Bottom-Only noise: {bottom_drop:.4f}")

    if top_drop > mid_drop and top_drop > bottom_drop:
        print("\nConclusion: TOP 20% channels are most critical for explainability!")
        print("   - Prioritize these channels in quantization/compression")
    elif mid_drop > top_drop and mid_drop > bottom_drop:
        print("\nConclusion: MID 60% channels are most critical for explainability!")
        print("   - Middle channels contribute significantly to explanation quality")
    else:
        print("\nConclusion: BOTTOM 20% channels have unexpected importance!")
        print("   - Even low-importance channels affect explainability")

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Importance method: {IMPORTANCE_METHOD}")
    print(f"  - Noise type: {noise_type}")
    print(f"  - Test images: {NUM_TEST_IMAGES}")
    print(f"  - Runs per noise level: {NUM_RUNS}")
    print(f"\nResults saved to:")
    print(f"  - {save_path}")
    print("\n")


if __name__ == "__main__":
    main()
