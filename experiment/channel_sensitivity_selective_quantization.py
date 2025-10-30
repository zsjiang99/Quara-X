"""
Selective Channel Quantization Sensitivity - GradCAM Explanation Only
=======================================================================
This experiment tests which channel groups are most sensitive to quantization IN THE EXPLANATION PROCESS ONLY.

KEY DESIGN:
  - Model: Always FP32, never modified
  - Forward/Backward: Always clean (no quantization)
  - Quantization: ONLY during GradCAM computation, ONLY on selected interval

DIFFERENCE FROM NOISE:
  - Quantization is DETERMINISTIC (not random)
  - Quantization has STRUCTURED error (small values → 0, large values preserved)
  - Quantization doesn't average out with GAP

EXPERIMENT:
  - Divide channels into 5 intervals by activation magnitude
  - Quantize ONE interval at a time during GradCAM computation (INT8/INT4/INT2)
  - Keep other intervals clean FP32
  - Measure ROAD score degradation

QUESTION:
  "Which channels are most critical for explanation quality under quantization?"

Author: Selective Channel Quantization Analysis
Date: 2025-10-30
Version: 1.0
"""

import os
import sys
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

# Add ROAD evaluation to path
sys.path.append('/home/zjian137/road_evaluation')
from road import run_road
from road.imputations import NoisyLinearImputer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==========================================================
# Quantization Functions
# ==========================================================
def symmetric_quantize(x, bit_width):
    """
    Symmetric quantization: x -> INT-N -> x_quantized

    This is DETERMINISTIC - same input always produces same output
    This is STRUCTURAL - small values get truncated to 0
    """
    if bit_width >= 32:
        return x

    # INT-N range: [-2^(n-1), 2^(n-1)-1]
    n_levels = 2 ** bit_width
    qmax = n_levels // 2 - 1

    # Per-channel scale
    scale = x.abs().max() / qmax

    if scale == 0:
        return torch.zeros_like(x)

    # Quantize: x -> round(x/scale) -> clamp -> dequantize
    x_int = torch.round(x / scale).clamp(-qmax-1, qmax)
    x_quant = x_int * scale

    return x_quant


def per_channel_quantize(tensor, bit_width, channel_mask):
    """
    Quantize ONLY selected channels

    Args:
        tensor: [B, C, H, W]
        bit_width: quantization bits (32=FP32, 8=INT8, 4=INT4, 2=INT2)
        channel_mask: [C] boolean, True = quantize this channel

    Returns:
        Quantized tensor (only selected channels quantized)
    """
    if bit_width >= 32 or channel_mask is None:
        return tensor

    quant_tensor = tensor.clone()
    C = tensor.shape[1]

    for c in range(C):
        if channel_mask[c]:  # Only quantize this channel
            ch_data = tensor[:, c, :, :]  # [B, H, W]
            ch_quant = symmetric_quantize(ch_data, bit_width)
            quant_tensor[:, c, :, :] = ch_quant

    return quant_tensor


# ==========================================================
# Channel Importance Analysis
# ==========================================================
class ChannelImportanceAnalyzer:
    """Measure channel importance based on activation magnitude"""

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

    def compute_importance(self, input_tensor, target_class):
        """Compute channel importance = mean absolute activation"""
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        scores = output[:, target_class]
        scores.sum().backward()

        activations = self.activations
        importance = activations.abs().mean(dim=(0, 2, 3))  # [C]

        return importance.cpu()


# ==========================================================
# Selective Quantized GradCAM
# ==========================================================
class SelectiveQuantizedGradCAM:
    """
    GradCAM with selective channel quantization

    CRITICAL:
    - Model always FP32
    - Forward/Backward always clean
    - Quantization ONLY applied during CAM computation, ONLY to selected channels
    """

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

    def __call__(self, input_tensor, target_class, channel_mask=None, bit_width=32):
        """
        Compute GradCAM with selective channel quantization

        Args:
            input_tensor: input image
            target_class: target class for GradCAM
            channel_mask: [C] boolean mask, True = quantize this channel
            bit_width: quantization bits (32/8/4/2)

        Flow:
            1. Clean forward → clean activation
            2. Clean backward → clean gradient
            3. Quantize ONLY selected channels
            4. Compute CAM with quantized components
        """
        self.model.eval()
        self.model.zero_grad()

        # Step 1 & 2: CLEAN forward and backward
        output = self.model(input_tensor)
        scores = output[:, target_class]
        scores.sum().backward()

        # Get clean activation and gradient
        clean_activations = self.activations.clone()  # [B, C, H, W]
        clean_gradients = self.gradients.clone()      # [B, C, H, W]

        # Step 3: Quantize ONLY selected channels
        quant_activations = per_channel_quantize(
            clean_activations, bit_width, channel_mask
        )
        quant_gradients = per_channel_quantize(
            clean_gradients, bit_width, channel_mask
        )

        # Step 4: Compute GradCAM with quantized components
        weights = quant_gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * quant_activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                          mode='bilinear', align_corners=False)

        return cam.squeeze(1)


# ==========================================================
# Channel Division
# ==========================================================
def divide_channels_into_intervals(importance_scores):
    """
    Divide channels into 5 equal intervals based on importance ranking
    """
    num_channels = len(importance_scores)
    sorted_indices = torch.argsort(importance_scores, descending=True)

    interval_size = num_channels // 5
    intervals = []

    for i in range(5):
        start_idx = i * interval_size
        if i == 4:
            end_idx = num_channels
        else:
            end_idx = (i + 1) * interval_size

        interval_indices = sorted_indices[start_idx:end_idx]
        intervals.append(interval_indices)

    return intervals


def create_channel_mask(num_channels, interval_indices):
    """Create boolean mask for selected channels"""
    mask = torch.zeros(num_channels, dtype=torch.bool)
    mask[interval_indices] = True
    return mask


# ==========================================================
# ROAD Evaluation
# ==========================================================
class SimpleImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = [img.cpu() for img in images]
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        return img, label


def evaluate_explanation_with_road(model, test_images, test_labels, heatmaps, device):
    """Evaluate explanation quality using ROAD metric"""
    dataset = SimpleImageDataset(test_images, test_labels)

    explanations = []
    for heatmap in heatmaps:
        attr = np.stack([heatmap] * 3, axis=0)
        explanations.append(attr)

    transform_test = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    percentages = [0.1, 0.3, 0.5]
    imputer = NoisyLinearImputer(noise=0.1)  # Increased from 0.01 to 0.1 for better sensitivity

    model.eval()

    try:
        res_acc, prob_acc = run_road(
            model=model,
            dataset_test=dataset,
            explanations_test=explanations,
            transform_test=transform_test,
            percentages=percentages,
            morf=True,
            batch_size=len(test_images),
            imputation=imputer
        )
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
def plot_selective_quantization_sensitivity(results_dict, save_path):
    """
    Plot quantization sensitivity for each interval

    X-axis: Bit-width (32, 8, 4, 2)
    Y-axis: ROAD Score
    6 lines: 5 intervals + baseline
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    interval_config = {
        'Interval-1 [0-20%]': {'color': '#E74C3C', 'linewidth': 3.0, 'marker': 'o', 'markersize': 10,
                               'label': '[0-20%] Top (Highest Activation)'},
        'Interval-2 [20-40%]': {'color': '#F39C12', 'linewidth': 2.8, 'marker': 's', 'markersize': 9,
                                'label': '[20-40%] Upper-Mid'},
        'Interval-3 [40-60%]': {'color': '#3498DB', 'linewidth': 2.8, 'marker': '^', 'markersize': 9,
                                'label': '[40-60%] Middle'},
        'Interval-4 [60-80%]': {'color': '#2ECC71', 'linewidth': 2.8, 'marker': 'v', 'markersize': 9,
                                'label': '[60-80%] Lower-Mid'},
        'Interval-5 [80-100%]': {'color': '#8E44AD', 'linewidth': 3.0, 'marker': 'D', 'markersize': 9,
                                 'label': '[80-100%] Bottom (Lowest Activation)'},
        'Baseline': {'color': '#34495E', 'linewidth': 2.5, 'marker': 'x', 'markersize': 10,
                     'label': 'Baseline (No Quantization)', 'linestyle': '--'},
    }

    # Bit-widths to plot
    bit_widths = [32, 8, 4, 2]

    for interval_name, style in interval_config.items():
        if interval_name not in results_dict:
            continue

        interval_results = results_dict[interval_name]
        road_scores = [interval_results[bits]['road'] for bits in bit_widths]

        ax.plot(bit_widths, road_scores,
               color=style['color'],
               linewidth=style['linewidth'],
               marker=style['marker'],
               markersize=style.get('markersize', 8),
               label=style['label'],
               linestyle=style.get('linestyle', '-'),
               alpha=0.9)

    # Styling
    ax.set_xlabel('Quantization Bit-Width (Higher = Better Precision)', fontsize=16, fontweight='bold')
    ax.set_ylabel('ROAD Score (Explanation Quality)', fontsize=16, fontweight='bold')

    title = 'Selective Channel Quantization: Which Channels Need Higher Precision?'
    ax.set_title(title, fontsize=17, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
             edgecolor='gray', ncol=1)

    # X-axis: bit-widths in descending order (32 -> 2)
    ax.set_xticks(bit_widths)
    ax.set_xticklabels(['FP32\n(Baseline)', 'INT8\n(256 levels)', 'INT4\n(16 levels)', 'INT2\n(4 levels)'])
    ax.invert_xaxis()  # Higher precision on left

    ax.tick_params(axis='both', which='major', labelsize=13)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Add annotation
    ax.text(0.98, 0.02,
            'Note: Model always FP32, quantization only in GradCAM computation on selected channels',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {save_path}")


# ==========================================================
# Main Experiment
# ==========================================================
def main():
    print("="*80)
    print("Selective Channel Quantization Sensitivity - GradCAM Explanation Only")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Configuration
    NUM_TEST_IMAGES = 10

    print(f"\nExperiment Configuration:")
    print(f"  Model: Always FP32 (never modified)")
    print(f"  Forward/Backward: Always clean")
    print(f"  Quantization: Only in GradCAM computation, only on selected interval")
    print(f"  Intervals: 5 (20% each, by activation magnitude)")
    print(f"  Bit-widths: FP32, INT8, INT4, INT2")
    print(f"  Test images: {NUM_TEST_IMAGES}")

    # Load data
    print("\n" + "="*80)
    print("Loading Data & Model")
    print("="*80)

    data_root = "/home/zjian137/CAMBoost/gradcam/data/tiny-imagenet"
    _, val_loader, num_classes = load_tiny_imagenet(root=data_root, batch_size=64, num_workers=4)

    model_path = "/home/zjian137/CAMBoost/gradcam/resnet50_tinyimagenet_finetuned.pth"
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    target_layer_name = "layer4.2.conv3"
    target_layer = model.layer4[2].conv3
    num_channels = target_layer.out_channels

    print(f"Target layer: {target_layer_name} ({num_channels} channels)")

    # Get test images
    test_images = []
    test_labels = []
    test_classes = []

    for batch_images, batch_labels in val_loader:
        for i in range(len(batch_images)):
            if len(test_images) >= NUM_TEST_IMAGES:
                break
            test_images.append(batch_images[i:i+1].to(device))
            test_labels.append(batch_labels[i].item())

            with torch.no_grad():
                output = model(batch_images[i:i+1].to(device))
                pred_class = output.argmax(dim=1).item()
            test_classes.append(pred_class)

        if len(test_images) >= NUM_TEST_IMAGES:
            break

    # Define strategies
    interval_names = [
        'Interval-1 [0-20%]',
        'Interval-2 [20-40%]',
        'Interval-3 [40-60%]',
        'Interval-4 [60-80%]',
        'Interval-5 [80-100%]',
        'Baseline'
    ]

    bit_widths = [32, 8, 4, 2]

    # Run experiments
    print("\n" + "="*80)
    print("Running Experiments")
    print("="*80)

    all_images_results = {name: {bits: [] for bits in bit_widths}
                         for name in interval_names}

    for img_idx in range(NUM_TEST_IMAGES):
        print(f"\n{'='*80}")
        print(f"Image {img_idx + 1}/{NUM_TEST_IMAGES}")
        print(f"{'='*80}")

        test_image = test_images[img_idx]
        target_class = test_classes[img_idx]

        # Compute channel importance for this image
        analyzer = ChannelImportanceAnalyzer(model, target_layer)
        with torch.enable_grad():
            importance_scores = analyzer.compute_importance(test_image, target_class)

        intervals = divide_channels_into_intervals(importance_scores)

        # Create GradCAM instance
        gradcam = SelectiveQuantizedGradCAM(model, target_layer)

        # Compute baseline FIRST (FP32, no quantization)
        print(f"\n  Baseline (FP32)")
        print(f"    FP32...", end=' ')

        with torch.enable_grad():
            heatmap = gradcam(test_image, target_class,
                            channel_mask=None, bit_width=32)

        heatmap_np = heatmap.detach().cpu().numpy()[0]

        baseline_road = evaluate_explanation_with_road(
            model, [test_image[0]], [target_class],
            [heatmap_np], device
        )

        all_images_results['Baseline'][32].append(baseline_road)
        print(f"ROAD={baseline_road:.4f}")

        # Fill other bit-widths with same baseline
        for bits in [8, 4, 2]:
            all_images_results['Baseline'][bits].append(baseline_road)

        # Test each interval
        for interval_idx, interval_name in enumerate(interval_names):
            if interval_name == 'Baseline':
                continue

            print(f"\n  {interval_name}")

            # Get interval channels
            interval_indices = intervals[interval_idx]
            channel_mask = create_channel_mask(num_channels, interval_indices)

            # Test each bit-width
            for bit_width in bit_widths:
                if bit_width == 32:
                    # FP32 = baseline
                    all_images_results[interval_name][32].append(baseline_road)
                    continue

                bit_name = f'INT{bit_width}'
                print(f"    {bit_name}...", end=' ')

                # Quantize ONLY this interval
                with torch.enable_grad():
                    heatmap = gradcam(test_image, target_class,
                                    channel_mask=channel_mask,
                                    bit_width=bit_width)

                heatmap_np = heatmap.detach().cpu().numpy()[0]

                road_score = evaluate_explanation_with_road(
                    model, [test_image[0]], [target_class],
                    [heatmap_np], device
                )

                all_images_results[interval_name][bit_width].append(road_score)
                print(f"ROAD={road_score:.4f}")

    # Compute statistics
    print("\n" + "="*80)
    print("Computing Statistics")
    print("="*80)

    results_dict = {}
    for interval_name in interval_names:
        interval_results = {}
        for bit_width in bit_widths:
            image_scores = all_images_results[interval_name][bit_width]
            mean_road = np.mean(image_scores)
            std_road = np.std(image_scores)

            interval_results[bit_width] = {
                'road': mean_road,
                'std': std_road,
            }

        results_dict[interval_name] = interval_results

        print(f"\n{interval_name}:")
        for bit_width in bit_widths:
            road = interval_results[bit_width]['road']
            std = interval_results[bit_width]['std']
            bit_name = 'FP32' if bit_width == 32 else f'INT{bit_width}'
            print(f"  {bit_name}: ROAD={road:.4f}±{std:.4f}")

    # Visualization
    print("\n" + "="*80)
    print("Generating Visualization")
    print("="*80)

    os.makedirs("./results", exist_ok=True)
    save_path = "./results/channel_sensitivity_selective_quantization.png"

    plot_selective_quantization_sensitivity(results_dict, save_path)

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"Results saved to: {save_path}")
    print("\nKey Question Answered:")
    print("- Which channel interval needs higher precision quantization?")
    print("- Can we use INT4 for less important channels?")
    print("- What's the minimum bit-width for each interval?")
    print()


if __name__ == "__main__":
    main()
