"""
Channel Noise Sensitivity Analysis for GradCAM Explainability
==============================================================
Research Question: How do different channel groups (top/mid/bottom importance)
                   contribute to GradCAM explainability?

Experiment Design:
1. Measure channel importance and divide into 3 groups:
   - Top 20%: Most important channels
   - Mid 60%: Medium importance channels
   - Bottom 20%: Least important channels

2. Test 8 noise injection strategies (2^3 combinations):
   [Top, Mid, Bottom] where each can be Noise/Clean

3. For each strategy, vary noise strength from 0.0 to 1.0

4. Plot 8 curves showing how explainability degrades with noise

Expected Results:
- Curves with Top-channel noise should degrade fastest
- Bottom-channel noise should have minimal impact
- This validates differential importance of channels

Author: Experiment Script
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
# Noise Injection Module
# ==========================================================
class NoiseInjectionConv2d(nn.Module):
    """Conv2d layer with selective channel noise injection"""

    def __init__(self, conv, noise_mask, noise_strength=0.0):
        """
        Args:
            conv: Original Conv2d layer
            noise_mask: Boolean tensor [C] indicating which channels to add noise
            noise_strength: Noise standard deviation (relative to activation std)
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

    def forward(self, x):
        # Standard convolution
        out = F.conv2d(x, self.weight, bias=self.bias,
                      stride=self.stride, padding=self.padding,
                      dilation=self.dilation, groups=self.groups)

        # Add noise to selected channels
        if self.noise_strength > 0:
            # Compute per-channel noise
            B, C, H, W = out.shape
            noise = torch.randn_like(out) * self.noise_strength

            # Apply mask: only add noise to specified channels
            mask = self.noise_mask.view(1, C, 1, 1).to(out.device)
            out = out + noise * mask

        return out


# ==========================================================
# Strategy Configuration
# ==========================================================
def divide_channels_by_importance(importance_scores, top_ratio=0.2, bottom_ratio=0.2):
    """
    Divide channels into 3 groups based on importance

    Returns:
        top_indices: Indices of top channels
        mid_indices: Indices of middle channels
        bottom_indices: Indices of bottom channels
    """
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
    """
    Create a boolean mask indicating which channels should receive noise

    Args:
        noise_top/mid/bottom: Boolean, whether to add noise to that group

    Returns:
        mask: Boolean tensor [C]
    """
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
def apply_noise_injection(model_fp32, layer_name, noise_mask, noise_strength):
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
        noisy_conv = NoiseInjectionConv2d(original_conv, noise_mask, noise_strength)
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
def compute_heatmap_correlation(heatmap_ref, heatmap_test):
    """Compute correlation between two heatmaps"""
    corr = np.corrcoef(heatmap_ref.flatten(), heatmap_test.flatten())[0, 1]
    mae = np.abs(heatmap_ref - heatmap_test).mean()
    rmse = np.sqrt(((heatmap_ref - heatmap_test) ** 2).mean())

    return {
        'correlation': corr,
        'mae': mae,
        'rmse': rmse
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
def plot_noise_sensitivity_curves(results_dict, save_path):
    """
    Plot 8 curves showing how explainability degrades with noise

    Args:
        results_dict: {strategy_name: {noise_level: metrics}}
    """
    plt.figure(figsize=(14, 8))

    # Define colors and line styles for different strategies
    strategy_styles = {
        'All-Noise': {'color': 'red', 'linestyle': '-', 'marker': 'o', 'label': '[N,N,N] All'},
        'Top+Mid': {'color': 'orange', 'linestyle': '-', 'marker': 's', 'label': '[N,N,-] Top+Mid'},
        'Top+Bottom': {'color': 'brown', 'linestyle': '--', 'marker': '^', 'label': '[N,-,N] Top+Bottom'},
        'Top-Only': {'color': 'darkred', 'linestyle': '-', 'marker': 'd', 'label': '[N,-,-] Top Only'},
        'Mid+Bottom': {'color': 'blue', 'linestyle': '--', 'marker': 'v', 'label': '[-,N,N] Mid+Bottom'},
        'Mid-Only': {'color': 'cyan', 'linestyle': '-', 'marker': '<', 'label': '[-,N,-] Mid Only'},
        'Bottom-Only': {'color': 'green', 'linestyle': '--', 'marker': '>', 'label': '[-,-,N] Bottom Only'},
        'No-Noise': {'color': 'black', 'linestyle': '-', 'marker': '*', 'label': '[-,-,-] Baseline'},
    }

    for strategy_name, style in strategy_styles.items():
        if strategy_name not in results_dict:
            continue

        strategy_results = results_dict[strategy_name]
        noise_levels = sorted(strategy_results.keys())
        correlations = [strategy_results[noise]['correlation'] for noise in noise_levels]

        plt.plot(noise_levels, correlations,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=2.5,
                markersize=8,
                label=style['label'])

    plt.xlabel('Noise Strength (σ)', fontsize=14, fontweight='bold')
    plt.ylabel('Heatmap Correlation with Baseline', fontsize=14, fontweight='bold')
    plt.title('Channel-wise Noise Sensitivity Analysis\nImpact on GradCAM Explainability',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Noise sensitivity curves saved: {save_path}")


# ==========================================================
# Main Experiment
# ==========================================================
def main():
    print("="*80)
    print("Channel Noise Sensitivity Analysis for GradCAM Explainability")
    print("="*80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

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

    # Get sample image
    print("\n" + "="*80)
    print("Step 3: Selecting Sample Image")
    print("="*80)

    sample_images, sample_labels = next(iter(val_loader))
    sample_image = sample_images[0:1].to(device)

    with torch.no_grad():
        output = model_fp32(sample_image)
        target_class = output.argmax(dim=1).item()

    print(f"Predicted class: {target_class}")

    # Analyze channel importance
    print("\n" + "="*80)
    print("Step 4: Analyzing Channel Importance")
    print("="*80)

    analyzer = ChannelImportanceAnalyzer(model_fp32, target_layer)

    with torch.enable_grad():
        importance_scores = analyzer.compute_importance(
            sample_image, target_class, method='combined'
        )

    # Divide into groups
    top_indices, mid_indices, bottom_indices = divide_channels_by_importance(
        importance_scores, top_ratio=0.2, bottom_ratio=0.2
    )

    print(f"\nChannel groups:")
    print(f"  Top 20%:    {len(top_indices)} channels (most important)")
    print(f"  Mid 60%:    {len(mid_indices)} channels")
    print(f"  Bottom 20%: {len(bottom_indices)} channels (least important)")

    print(f"\nTop 5 most important channels:")
    sorted_indices = torch.argsort(importance_scores, descending=True)
    for i in range(5):
        ch_idx = sorted_indices[i].item()
        score = importance_scores[ch_idx].item()
        print(f"  Channel {ch_idx}: {score:.6f}")

    # Generate baseline heatmap
    print("\n" + "="*80)
    print("Step 5: Generating Baseline Heatmap")
    print("="*80)

    cam_baseline = SimpleGradCAM(model_fp32, target_layer)
    with torch.enable_grad():
        heatmap_baseline = cam_baseline(sample_image, target_class)
    heatmap_baseline_np = heatmap_baseline.detach().cpu().numpy()[0]

    # Define experiment strategies
    strategies = {
        'All-Noise':     (True,  True,  True),   # [N, N, N]
        'Top+Mid':       (True,  True,  False),  # [N, N, -]
        'Top+Bottom':    (True,  False, True),   # [N, -, N]
        'Top-Only':      (True,  False, False),  # [N, -, -]
        'Mid+Bottom':    (False, True,  True),   # [-, N, N]
        'Mid-Only':      (False, True,  False),  # [-, N, -]
        'Bottom-Only':   (False, False, True),   # [-, -, N]
        'No-Noise':      (False, False, False),  # [-, -, -]
    }

    # Define noise levels to test
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    # Run experiments
    print("\n" + "="*80)
    print("Step 6: Running Noise Sensitivity Experiments")
    print("="*80)

    results_dict = {}

    for strategy_name, (noise_top, noise_mid, noise_bottom) in strategies.items():
        print(f"\n{'='*80}")
        print(f"Testing Strategy: {strategy_name}")
        print(f"  Top channels:    {'NOISE' if noise_top else 'clean'}")
        print(f"  Mid channels:    {'NOISE' if noise_mid else 'clean'}")
        print(f"  Bottom channels: {'NOISE' if noise_bottom else 'clean'}")
        print(f"{'='*80}")

        # Create noise mask
        noise_mask = create_noise_mask(
            num_channels, top_indices, mid_indices, bottom_indices,
            noise_top, noise_mid, noise_bottom
        )

        strategy_results = {}

        for noise_strength in noise_levels:
            # Apply noise injection
            model_noisy = apply_noise_injection(
                model_fp32, target_layer_name, noise_mask, noise_strength
            )
            model_noisy = model_noisy.to(device)

            # Generate heatmap
            noisy_layer = model_noisy.layer4[2].conv3
            cam_noisy = SimpleGradCAM(model_noisy, noisy_layer)

            with torch.enable_grad():
                heatmap_noisy = cam_noisy(sample_image, target_class)

            heatmap_noisy_np = heatmap_noisy.detach().cpu().numpy()[0]

            # Compute metrics
            metrics = compute_heatmap_correlation(heatmap_baseline_np, heatmap_noisy_np)
            strategy_results[noise_strength] = metrics

            print(f"  σ={noise_strength:.2f}: Corr={metrics['correlation']:.4f}, "
                  f"MAE={metrics['mae']:.4f}")

        results_dict[strategy_name] = strategy_results

    # Visualization
    print("\n" + "="*80)
    print("Step 7: Generating Visualization")
    print("="*80)

    os.makedirs("./results", exist_ok=True)
    plot_noise_sensitivity_curves(
        results_dict,
        save_path="./results/channel_noise_sensitivity_curves.png"
    )

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\n📊 Key Findings:")
    print("-" * 80)

    # Compare degradation rates at σ=0.5
    print("\nCorrelation at σ=0.5:")
    for strategy_name in ['Top-Only', 'Mid-Only', 'Bottom-Only']:
        if strategy_name in results_dict:
            corr = results_dict[strategy_name][0.5]['correlation']
            print(f"  {strategy_name:<15}: {corr:.4f}")

    # Identify which group is most critical
    top_only_drop = 1.0 - results_dict['Top-Only'][0.5]['correlation']
    mid_only_drop = 1.0 - results_dict['Mid-Only'][0.5]['correlation']
    bottom_only_drop = 1.0 - results_dict['Bottom-Only'][0.5]['correlation']

    print(f"\nCorrelation drop at σ=0.5:")
    print(f"  Top-Only noise:    {top_only_drop:.4f}")
    print(f"  Mid-Only noise:    {mid_only_drop:.4f}")
    print(f"  Bottom-Only noise: {bottom_only_drop:.4f}")

    if top_only_drop > mid_only_drop and top_only_drop > bottom_only_drop:
        print("\n✅ Conclusion: TOP channels are most critical for explainability!")
    elif mid_only_drop > top_only_drop and mid_only_drop > bottom_only_drop:
        print("\n✅ Conclusion: MID channels are most critical for explainability!")
    else:
        print("\n✅ Conclusion: BOTTOM channels have unexpected importance!")

    print("\n" + "="*80)
    print("✅ Experiment Complete!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - ./results/channel_noise_sensitivity_curves.png")
    print("\n")


if __name__ == "__main__":
    main()
