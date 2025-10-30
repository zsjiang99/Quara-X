"""
Channel-wise Quantization Impact on GradCAM Heatmaps
=====================================================
Research Question: Do different channels have different impacts on the final heatmap when quantized?

Experiment Design:
1. Measure channel importance based on multiple metrics (weights, activations, gradients)
2. Apply differential quantization strategies:
   - Important channels → High precision (INT8)
   - Unimportant channels → Low precision (INT4)
   - Reverse strategy (test hypothesis)
   - Uniform baseline (INT4/INT8)
3. Evaluate heatmap quality using correlation with FP32 baseline

Author: Experiment Script
Date: 2025-10-29
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
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# Add parent directory to path to import utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==========================================================
# Quantization Utilities
# ==========================================================
def quantize_tensor_linear(x, bits=8):
    """Symmetric linear quantization"""
    qmax = 2 ** (bits - 1) - 1
    if x.numel() == 0:
        return x, 1.0, 0
    scale = x.abs().max() / qmax if x.abs().max() > 0 else 1.0
    q = (x / scale).round().clamp(-qmax, qmax)
    return q.to(torch.int32), scale, 0


def dequantize_tensor(q, scale):
    """Dequantize tensor"""
    return q.to(torch.float32) * scale


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

        Methods:
            'gradient': Based on gradient magnitude
            'activation': Based on activation magnitude
            'weight': Based on gradient weights (α in GradCAM)
            'combined': Weighted combination of all three
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

        # Compute GradCAM weights (global average pooling of gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Channel importance metrics
        batch_size, num_channels = gradients.shape[0], gradients.shape[1]

        if method == 'gradient':
            # Importance = mean absolute gradient per channel
            importance = gradients.abs().mean(dim=(0, 2, 3))  # [C]

        elif method == 'activation':
            # Importance = mean activation magnitude per channel
            importance = activations.abs().mean(dim=(0, 2, 3))  # [C]

        elif method == 'weight':
            # Importance = absolute GradCAM weight per channel
            importance = weights.abs().mean(dim=(0, 2, 3))  # [C]

        elif method == 'combined':
            # Combined metric: weight * activation magnitude
            # This measures the actual contribution to final CAM
            weighted_activation = (weights.abs() * activations.abs()).mean(dim=(0, 2, 3))
            importance = weighted_activation  # [C]

        else:
            raise ValueError(f"Unknown importance method: {method}")

        return importance.cpu()

    def rank_channels(self, importance_scores):
        """
        Rank channels by importance

        Returns:
            indices: Channel indices sorted by importance (descending)
            scores: Sorted importance scores
        """
        sorted_indices = torch.argsort(importance_scores, descending=True)
        sorted_scores = importance_scores[sorted_indices]
        return sorted_indices, sorted_scores


# ==========================================================
# Channel-wise Quantized Convolution
# ==========================================================
class ChannelWiseQuantizedConv2d(nn.Module):
    """Conv2d with per-channel quantization based on importance"""

    def __init__(self, conv, channel_bits_config):
        """
        Args:
            conv: Original Conv2d layer
            channel_bits_config: List of bits for each output channel [bits_ch0, bits_ch1, ...]
        """
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.bias = conv.bias
        self.channel_bits = channel_bits_config

        # Quantize each channel with its assigned bit width
        w = conv.weight.data.clone()  # [out_ch, in_ch, k, k]
        wq = torch.zeros_like(w)

        for ch, bits in enumerate(channel_bits_config):
            q, scale, _ = quantize_tensor_linear(w[ch:ch+1], bits=bits)
            wq[ch:ch+1] = dequantize_tensor(q, scale)

        self.wq = wq

    def forward(self, x):
        # Quantize activation (use INT8 uniformly)
        qx, scale, _ = quantize_tensor_linear(x, bits=8)
        xq = dequantize_tensor(qx, scale)

        # Perform convolution
        out = F.conv2d(xq, self.wq, bias=self.bias,
                       stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=self.groups)
        return out


# ==========================================================
# Quantization Strategies
# ==========================================================
def create_importance_based_config(importance_scores, strategy='important_high',
                                   high_bits=8, low_bits=4, threshold_percentile=50):
    """
    Create channel-wise bit configuration based on importance

    Strategies:
        'important_high': Important channels → high bits, unimportant → low bits
        'important_low': Important channels → low bits, unimportant → high bits (reverse test)
        'top_k_high': Top k% → high bits, rest → low bits
        'bottom_k_high': Bottom k% → high bits, rest → low bits
    """
    num_channels = len(importance_scores)
    channel_bits = []

    # Compute threshold
    threshold = torch.quantile(importance_scores, threshold_percentile / 100.0)

    for ch_idx in range(num_channels):
        importance = importance_scores[ch_idx]

        if strategy == 'important_high':
            # High importance → high precision
            bits = high_bits if importance >= threshold else low_bits

        elif strategy == 'important_low':
            # High importance → low precision (counter-intuitive test)
            bits = low_bits if importance >= threshold else high_bits

        elif strategy == 'top_k_high':
            # Top k% → high precision
            bits = high_bits if importance >= threshold else low_bits

        elif strategy == 'bottom_k_high':
            # Bottom k% → high precision
            bits = high_bits if importance < threshold else low_bits

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        channel_bits.append(bits)

    return channel_bits


def apply_channel_wise_quantization(model_fp32, layer_name, channel_bits_config):
    """Apply channel-wise quantization to a specific layer"""
    model_quant = copy.deepcopy(model_fp32)

    # Navigate to target layer
    parts = layer_name.split('.')
    parent = model_quant
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Get original conv layer
    original_conv = getattr(parent, parts[-1])

    # Replace with quantized version
    if isinstance(original_conv, nn.Conv2d):
        quantized_conv = ChannelWiseQuantizedConv2d(original_conv, channel_bits_config)
        setattr(parent, parts[-1], quantized_conv)

    return model_quant


def apply_uniform_quantization_to_layer(model_fp32, layer_name, bits):
    """Apply uniform quantization to a specific layer"""
    num_channels = None

    # Get number of channels
    parts = layer_name.split('.')
    parent = model_fp32
    for part in parts[:-1]:
        parent = getattr(parent, part)
    conv_layer = getattr(parent, parts[-1])

    if isinstance(conv_layer, nn.Conv2d):
        num_channels = conv_layer.out_channels

    # Create uniform config
    channel_bits_config = [bits] * num_channels
    return apply_channel_wise_quantization(model_fp32, layer_name, channel_bits_config)


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


def load_tiny_imagenet(root="../CAMBoost/gradcam/data/tiny-imagenet", batch_size=64, num_workers=4):
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
def visualize_channel_importance(importance_scores, channel_bits_config, save_path):
    """Visualize channel importance and bit allocation"""
    num_channels = len(importance_scores)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Channel importance scores
    x = np.arange(num_channels)
    colors = ['red' if bits == 4 else 'green' for bits in channel_bits_config]

    ax1.bar(x, importance_scores.numpy(), color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Channel Index', fontsize=12)
    ax1.set_ylabel('Importance Score', fontsize=12)
    ax1.set_title('Channel Importance Scores (Red=INT4, Green=INT8)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Bit allocation histogram
    bits_values, counts = np.unique(channel_bits_config, return_counts=True)
    ax2.bar(bits_values, counts, color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Bit Width', fontsize=12)
    ax2.set_ylabel('Number of Channels', fontsize=12)
    ax2.set_title('Bit Width Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(bits_values)
    ax2.grid(axis='y', alpha=0.3)

    # Add statistics
    for i, (bits, count) in enumerate(zip(bits_values, counts)):
        percentage = count / num_channels * 100
        ax2.text(bits, count + max(counts)*0.02, f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Channel importance visualization saved: {save_path}")


def visualize_heatmap_comparison(image_tensor, heatmaps_dict, metrics_dict, save_path):
    """Visualize heatmap comparison across strategies"""
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image_tensor.cpu() * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    image_np = image_denorm.permute(1, 2, 0).numpy()

    # Setup grid
    strategies = list(heatmaps_dict.keys())
    n = len(strategies)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        heatmap = heatmaps_dict[strategy]

        # Overlay heatmap
        ax.imshow(image_np)
        im = ax.imshow(heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)

        # Title with metrics
        if strategy in metrics_dict and strategy != 'FP32':
            metrics = metrics_dict[strategy]
            title = f"{strategy}\nCorr: {metrics['correlation']:.4f}, MAE: {metrics['mae']:.4f}"
        else:
            title = f"{strategy}\n(Baseline)"

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap comparison saved: {save_path}")


# ==========================================================
# Main Experiment
# ==========================================================
def main():
    print("="*80)
    print("Channel-wise Quantization Impact on GradCAM Heatmaps - Experiment")
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
        print("Please update the data_root path in the script.")
        return

    _, val_loader, num_classes = load_tiny_imagenet(root=data_root, batch_size=64, num_workers=4)

    # Load model
    print("\n" + "="*80)
    print("Step 2: Loading ResNet50 Model")
    print("="*80)

    model_fp32 = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, num_classes)
    model_fp32 = model_fp32.to(device)

    # Try to load finetuned weights
    checkpoint_paths = [
        "../CAMBoost/resnet50_tinyimagenet_finetuned_improved.pth",
        "../CAMBoost/gradcam/experiment/resnet50_tinyimagenet_finetuned_improved.pth"
    ]

    loaded = False
    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            model_fp32.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"✅ Loaded finetuned model: {ckpt_path}")
            loaded = True
            break

    if not loaded:
        print("⚠️  Finetuned model not found, using ImageNet pretrained weights")
        print("   (Note: Accuracy on Tiny-ImageNet will be lower)")

    model_fp32.eval()

    # Select target layer for analysis
    target_layer_name = "layer4.2.conv3"  # Last conv in layer4
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
    sample_label = sample_labels[0].item()

    with torch.no_grad():
        output = model_fp32(sample_image)
        target_class = output.argmax(dim=1).item()

    print(f"Sample label: {sample_label}, Predicted class: {target_class}")

    # Analyze channel importance
    print("\n" + "="*80)
    print("Step 4: Analyzing Channel Importance")
    print("="*80)

    analyzer = ChannelImportanceAnalyzer(model_fp32, target_layer)

    with torch.enable_grad():
        importance_scores = analyzer.compute_importance(
            sample_image, target_class, method='combined'
        )

    sorted_indices, sorted_scores = analyzer.rank_channels(importance_scores)

    print(f"\nChannel importance statistics:")
    print(f"  Mean: {importance_scores.mean():.6f}")
    print(f"  Std:  {importance_scores.std():.6f}")
    print(f"  Min:  {importance_scores.min():.6f}")
    print(f"  Max:  {importance_scores.max():.6f}")
    print(f"\nTop 5 most important channels:")
    for i in range(5):
        ch_idx = sorted_indices[i].item()
        score = sorted_scores[i].item()
        print(f"  Channel {ch_idx}: {score:.6f}")

    # Define quantization strategies
    print("\n" + "="*80)
    print("Step 5: Testing Quantization Strategies")
    print("="*80)

    strategies = {
        'FP32': None,  # Baseline
        'Uniform_INT8': 'uniform_8',
        'Uniform_INT4': 'uniform_4',
        'Important_High_50%': ('important_high', 50),
        'Important_High_70%': ('important_high', 70),
        'Important_High_30%': ('important_high', 30),
        'Important_Low_50%': ('important_low', 50),  # Reverse test
    }

    heatmaps = {}
    metrics_dict = {}
    bit_configs = {}

    for strategy_name, strategy_config in strategies.items():
        print(f"\n{'='*80}")
        print(f"Testing Strategy: {strategy_name}")
        print(f"{'='*80}")

        if strategy_config is None:
            # FP32 baseline
            test_model = model_fp32
            channel_bits = [32] * num_channels

        elif strategy_config == 'uniform_8':
            # Uniform INT8
            test_model = apply_uniform_quantization_to_layer(
                model_fp32, target_layer_name, bits=8
            )
            channel_bits = [8] * num_channels

        elif strategy_config == 'uniform_4':
            # Uniform INT4
            test_model = apply_uniform_quantization_to_layer(
                model_fp32, target_layer_name, bits=4
            )
            channel_bits = [4] * num_channels

        else:
            # Importance-based strategies
            strategy_type, percentile = strategy_config
            channel_bits = create_importance_based_config(
                importance_scores,
                strategy=strategy_type,
                high_bits=8,
                low_bits=4,
                threshold_percentile=percentile
            )
            test_model = apply_channel_wise_quantization(
                model_fp32, target_layer_name, channel_bits
            )

        bit_configs[strategy_name] = channel_bits

        # Print bit distribution
        unique_bits, counts = np.unique(channel_bits, return_counts=True)
        print(f"Bit distribution:")
        for bits, count in zip(unique_bits, counts):
            percentage = count / num_channels * 100
            print(f"  INT{bits}: {count} channels ({percentage:.1f}%)")

        avg_bits = np.mean(channel_bits)
        print(f"Average bits: {avg_bits:.2f}")

        # Generate GradCAM
        print(f"Generating GradCAM heatmap...")
        test_layer = test_model.layer4[2].conv3
        cam = SimpleGradCAM(test_model, test_layer)

        with torch.enable_grad():
            heatmap = cam(sample_image, target_class)

        heatmap_np = heatmap.detach().cpu().numpy()[0]
        heatmaps[strategy_name] = heatmap_np

        print(f"Heatmap range: [{heatmap_np.min():.3f}, {heatmap_np.max():.3f}]")

    # Compute metrics (relative to FP32)
    print("\n" + "="*80)
    print("Step 6: Computing Heatmap Quality Metrics")
    print("="*80)

    heatmap_fp32 = heatmaps['FP32']

    print(f"\n{'Strategy':<25} {'Correlation':<12} {'MAE':<12} {'RMSE':<12} {'Avg Bits':<10}")
    print("-" * 80)

    for strategy_name in strategies.keys():
        if strategy_name == 'FP32':
            print(f"{'FP32':<25} {'1.0000':<12} {'0.000000':<12} {'0.000000':<12} {'32.00':<10}")
            continue

        heatmap = heatmaps[strategy_name]
        metrics = compute_heatmap_correlation(heatmap_fp32, heatmap)
        metrics_dict[strategy_name] = metrics

        avg_bits = np.mean(bit_configs[strategy_name])

        print(f"{strategy_name:<25} {metrics['correlation']:<12.4f} {metrics['mae']:<12.6f} "
              f"{metrics['rmse']:<12.6f} {avg_bits:<10.2f}")

    # Visualizations
    print("\n" + "="*80)
    print("Step 7: Generating Visualizations")
    print("="*80)

    os.makedirs("./results", exist_ok=True)

    # Visualize channel importance for one strategy
    example_strategy = 'Important_High_50%'
    visualize_channel_importance(
        importance_scores,
        bit_configs[example_strategy],
        save_path="./results/channel_importance_analysis.png"
    )

    # Visualize heatmap comparison
    visualize_heatmap_comparison(
        sample_images[0],
        heatmaps,
        metrics_dict,
        save_path="./results/channel_quantization_heatmap_comparison.png"
    )

    # Summary Report
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\n📊 Key Findings:")
    print("-" * 80)

    # Find best strategy
    best_strategy = None
    best_corr = -1
    for strategy_name, metrics in metrics_dict.items():
        if metrics['correlation'] > best_corr:
            best_corr = metrics['correlation']
            best_strategy = strategy_name

    print(f"\n✅ Best Strategy: {best_strategy}")
    print(f"   Correlation: {metrics_dict[best_strategy]['correlation']:.4f}")
    print(f"   MAE: {metrics_dict[best_strategy]['mae']:.6f}")
    print(f"   Average Bits: {np.mean(bit_configs[best_strategy]):.2f}")

    # Compare important_high vs important_low
    if 'Important_High_50%' in metrics_dict and 'Important_Low_50%' in metrics_dict:
        print("\n🔬 Hypothesis Test: Important channels need higher precision?")
        corr_high = metrics_dict['Important_High_50%']['correlation']
        corr_low = metrics_dict['Important_Low_50%']['correlation']

        print(f"   Important→High precision: Corr = {corr_high:.4f}")
        print(f"   Important→Low precision:  Corr = {corr_low:.4f}")
        print(f"   Difference: {abs(corr_high - corr_low):.4f}")

        if corr_high > corr_low:
            print("   ✅ Hypothesis SUPPORTED: Important channels benefit from higher precision")
        else:
            print("   ❌ Hypothesis REJECTED: No clear benefit from prioritizing important channels")

    print("\n" + "="*80)
    print("✅ Experiment Complete!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - ./results/channel_importance_analysis.png")
    print(f"  - ./results/channel_quantization_heatmap_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()
