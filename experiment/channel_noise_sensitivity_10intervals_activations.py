"""
Channel Noise Sensitivity Analysis - 10-Interval Division (Activation-based)
=============================================================================
This version divides channels into 10 equal intervals based on activation importance:

  IMPORTANT: Channels are sorted by activation in DESCENDING order

  - [0-10%]: Top 10% (HIGHEST activation) - Most important
  - [10%-20%]: 10-20%
  - [20%-30%]: 20-30%
  - [30%-40%]: 30-40%
  - [40%-50%]: 40-50%
  - [50%-60%]: 50-60%
  - [60%-70%]: 60-70%
  - [70%-80%]: 70-80%
  - [80%-90%]: 80-90%
  - [90%-100%]: Bottom 10% (LOWEST activation) - Least important

Each interval is tested separately with noise, plus a baseline (no noise).
Total: 11 curves

Author: Channel Noise Sensitivity Analysis
Date: 2025-10-30
Version: 1.0 (10-Interval Division)
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
# Channel Importance Measurement (Activation-based)
# ==========================================================
class ChannelImportanceAnalyzer:
    """Analyze channel importance based on activation magnitude"""

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

    def compute_importance(self, input_tensor, target_class):
        """Compute channel importance based on activation magnitude"""
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        scores = output[:, target_class]

        # Backward pass (needed for GradCAM later)
        scores.sum().backward()

        activations = self.activations  # [B, C, H, W]

        # Importance = mean activation magnitude per channel
        importance = activations.abs().mean(dim=(0, 2, 3))  # [C]

        return importance.cpu()


# ==========================================================
# Noise Injection Module
# ==========================================================
class ImprovedNoiseInjectionConv2d(nn.Module):
    """Conv2d layer with normalized noise injection"""

    def __init__(self, conv, noise_mask, noise_strength=0.0,
                 noise_type='activation', normalize=True):
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

        for ch_idx in range(self.weight.shape[0]):
            if self.noise_mask[ch_idx]:
                ch_weights = self.weight[ch_idx]
                if self.normalize:
                    noise = torch.randn_like(ch_weights) * ch_weights.std() * self.noise_strength
                else:
                    noise = torch.randn_like(ch_weights) * self.noise_strength
                noisy_w[ch_idx] = ch_weights + noise

        return noisy_w

    def forward(self, x):
        if self.noisy_weight is not None:
            w = self.noisy_weight
        else:
            w = self.weight

        out = F.conv2d(x, w, bias=self.bias,
                      stride=self.stride, padding=self.padding,
                      dilation=self.dilation, groups=self.groups)

        if self.noise_type == 'activation' and self.noise_strength > 0:
            B, C, H, W = out.shape
            noise_tensor = torch.zeros_like(out)

            for ch_idx in range(C):
                if self.noise_mask[ch_idx]:
                    ch_activation = out[:, ch_idx:ch_idx+1, :, :]
                    if self.normalize:
                        act_std = ch_activation.std()
                        if act_std > 0:
                            noise_tensor[:, ch_idx:ch_idx+1, :, :] = \
                                torch.randn_like(ch_activation) * act_std * self.noise_strength
                    else:
                        noise_tensor[:, ch_idx:ch_idx+1, :, :] = \
                            torch.randn_like(ch_activation) * self.noise_strength

            out = out + noise_tensor

        return out


# ==========================================================
# Channel Division into 10 Intervals
# ==========================================================
def divide_channels_into_intervals(importance_scores):
    """
    Divide channels into 10 equal intervals based on importance ranking

    IMPORTANT: Channels are sorted in DESCENDING order
    - Interval 0: indices [0:10%]     = Top 10% (HIGHEST activation)
    - Interval 1: indices [10%:20%]   = 10-20%
    - Interval 2: indices [20%:30%]   = 20-30%
    - Interval 3: indices [30%:40%]   = 30-40%
    - Interval 4: indices [40%:50%]   = 40-50%
    - Interval 5: indices [50%:60%]   = 50-60%
    - Interval 6: indices [60%:70%]   = 60-70%
    - Interval 7: indices [70%:80%]   = 70-80%
    - Interval 8: indices [80%:90%]   = 80-90%
    - Interval 9: indices [90%:100%]  = Bottom 10% (LOWEST activation)

    Returns:
        interval_indices: List of 10 tensors, each containing channel indices
    """
    num_channels = len(importance_scores)
    sorted_indices = torch.argsort(importance_scores, descending=True)

    # Each interval is 10% of channels
    interval_size = num_channels // 10

    intervals = []
    for i in range(10):
        start_idx = i * interval_size
        if i == 9:  # Last interval takes remaining channels
            end_idx = num_channels
        else:
            end_idx = (i + 1) * interval_size

        interval_indices = sorted_indices[start_idx:end_idx]
        intervals.append(interval_indices)

    return intervals


def create_noise_mask_for_interval(num_channels, interval_indices):
    """Create a boolean mask for a specific interval"""
    mask = torch.zeros(num_channels, dtype=torch.bool)
    mask[interval_indices] = True
    return mask


# ==========================================================
# Model Modification
# ==========================================================
def apply_noise_injection(model_fp32, layer_name, noise_mask, noise_strength,
                         noise_type='activation', normalize=True):
    """Apply noise injection to a specific layer"""
    model_noisy = copy.deepcopy(model_fp32)

    parts = layer_name.split('.')
    parent = model_noisy
    for part in parts[:-1]:
        parent = getattr(parent, part)

    original_conv = getattr(parent, parts[-1])

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

        output = self.model(input_tensor)
        scores = output[:, target_class]

        scores.sum().backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)

        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                          mode='bilinear', align_corners=False)

        return cam.squeeze(1)


# ==========================================================
# Simple Dataset for ROAD Evaluation
# ==========================================================
class SimpleImageDataset(Dataset):
    """Simple dataset wrapper for ROAD evaluation"""

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


# ==========================================================
# ROAD Evaluation Wrapper
# ==========================================================
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
    imputer = NoisyLinearImputer(noise=0.01)

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


def compute_model_accuracy(model, test_images, test_labels, device):
    """Compute model accuracy on test images"""
    model.eval()
    correct = 0
    total = len(test_images)

    with torch.no_grad():
        for img, label in zip(test_images, test_labels):
            output = model(img)
            pred = output.argmax(dim=1).item()
            if pred == label:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


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
def plot_noise_sensitivity_curves_10intervals(results_dict, save_path):
    """Plot 11 curves: 10 intervals + baseline"""
    from scipy.interpolate import make_interp_spline

    fig, ax = plt.subplots(figsize=(16, 9))

    # Define colors for 10 intervals using a gradient from red (high importance) to blue (low importance)
    # Red → Orange → Yellow → Green → Cyan → Blue
    strategies_config = {
        'Interval-1 [0-10%]':     {'color': '#8B0000', 'linewidth': 3.0, 'label': '[0-10%] Top (Highest)'},
        'Interval-2 [10-20%]':    {'color': '#E74C3C', 'linewidth': 2.6, 'label': '[10-20%]'},
        'Interval-3 [20-30%]':    {'color': '#F39C12', 'linewidth': 2.4, 'label': '[20-30%]'},
        'Interval-4 [30-40%]':    {'color': '#F1C40F', 'linewidth': 2.2, 'label': '[30-40%]'},
        'Interval-5 [40-50%]':    {'color': '#2ECC71', 'linewidth': 2.2, 'label': '[40-50%]'},
        'Interval-6 [50-60%]':    {'color': '#1ABC9C', 'linewidth': 2.2, 'label': '[50-60%]'},
        'Interval-7 [60-70%]':    {'color': '#3498DB', 'linewidth': 2.2, 'label': '[60-70%]'},
        'Interval-8 [70-80%]':    {'color': '#2980B9', 'linewidth': 2.4, 'label': '[70-80%]'},
        'Interval-9 [80-90%]':    {'color': '#8E44AD', 'linewidth': 2.6, 'label': '[80-90%]'},
        'Interval-10 [90-100%]':  {'color': '#4A235A', 'linewidth': 3.0, 'label': '[90-100%] Bottom (Lowest)'},
        'Baseline':               {'color': '#34495E', 'linewidth': 2.5, 'label': 'Baseline (No Noise)', 'linestyle': '--'},
    }

    for strategy_name, style in strategies_config.items():
        if strategy_name not in results_dict:
            continue

        strategy_results = results_dict[strategy_name]
        noise_levels = np.array(sorted(strategy_results.keys()))
        road_scores = np.array([strategy_results[noise]['road'] for noise in noise_levels])

        # Create smooth curve
        if len(noise_levels) > 3 and strategy_name != 'Baseline':
            noise_smooth = np.linspace(noise_levels.min(), noise_levels.max(), 300)
            try:
                spl = make_interp_spline(noise_levels, road_scores, k=3)
                road_smooth = spl(noise_smooth)
                ax.plot(noise_smooth, road_smooth,
                       color=style['color'],
                       linewidth=style['linewidth'],
                       label=style['label'],
                       linestyle=style.get('linestyle', '-'))
            except:
                ax.plot(noise_levels, road_scores,
                       color=style['color'],
                       linewidth=style['linewidth'],
                       label=style['label'],
                       linestyle=style.get('linestyle', '-'))
        else:
            ax.plot(noise_levels, road_scores,
                   color=style['color'],
                   linewidth=style['linewidth'],
                   label=style['label'],
                   linestyle=style.get('linestyle', '-'))

    # Styling
    ax.set_xlabel('Noise Strength (σ)', fontsize=16, fontweight='bold')
    ax.set_ylabel('ROAD Score (Explanation Quality)', fontsize=16, fontweight='bold')
    ax.set_title('Impact of Channel Noise on Explainability: 10-Interval Analysis (Activation-based)',
                fontsize=17, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
             edgecolor='gray', ncol=2)

    ax.set_xlim([noise_levels.min(), noise_levels.max()])
    ax.tick_params(axis='both', which='major', labelsize=13)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {save_path}")


# ==========================================================
# Main Experiment
# ==========================================================
def main():
    print("="*80)
    print("Channel Noise Sensitivity Analysis - 10-Interval Division")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Experiment configuration
    NUM_RUNS = 2
    NUM_TEST_IMAGES = 2
    noise_type = 'activation'
    normalize = True

    print(f"\nExperiment Configuration:")
    print(f"  Division: 10 intervals (10% each)")
    print(f"  Importance basis: Activation magnitude")
    print(f"  Noise type: {noise_type}")
    print(f"  Normalized: {normalize}")
    print(f"  Runs per point: {NUM_RUNS}")
    print(f"  Test images: {NUM_TEST_IMAGES}")
    print(f"  Metric: ROAD")

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

    model_path = "/home/zjian137/CAMBoost/gradcam/resnet50_tinyimagenet_finetuned.pth"

    model_fp32 = torchvision.models.resnet50(weights=None)
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, num_classes)

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

    # Get test images
    print("\n" + "="*80)
    print("Step 3: Selecting Test Images")
    print("="*80)

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
                output = model_fp32(batch_images[i:i+1].to(device))
                pred_class = output.argmax(dim=1).item()
            test_classes.append(pred_class)

        if len(test_images) >= NUM_TEST_IMAGES:
            break

    print(f"Selected {len(test_images)} test images")
    print(f"Sample predictions: {test_classes[:5]}")

    # Define 10 interval strategies + baseline
    interval_names = [
        'Interval-1 [0-10%]',
        'Interval-2 [10-20%]',
        'Interval-3 [20-30%]',
        'Interval-4 [30-40%]',
        'Interval-5 [40-50%]',
        'Interval-6 [50-60%]',
        'Interval-7 [60-70%]',
        'Interval-8 [70-80%]',
        'Interval-9 [80-90%]',
        'Interval-10 [90-100%]',
        'Baseline'
    ]

    # Define noise levels
    noise_levels = [0.0, 1.0, 3.0, 5.0]

    # Run experiments
    print("\n" + "="*80)
    print("Step 4-6: Running 10-Interval Noise Sensitivity Experiments")
    print("="*80)

    all_images_results = {name: {noise: {'road': [], 'accuracy': []} for noise in noise_levels}
                         for name in interval_names}

    # Process each test image
    for img_idx in range(NUM_TEST_IMAGES):
        print(f"\n{'='*80}")
        print(f"Processing Image {img_idx + 1}/{NUM_TEST_IMAGES}")
        print(f"{'='*80}")

        test_image = test_images[img_idx]
        target_class = test_classes[img_idx]

        # Compute channel importance (activation-based)
        analyzer = ChannelImportanceAnalyzer(model_fp32, target_layer)
        with torch.enable_grad():
            importance_scores = analyzer.compute_importance(test_image, target_class)

        # Divide into 10 intervals
        intervals = divide_channels_into_intervals(importance_scores)

        print(f"  Channel division:")
        for i, interval in enumerate(intervals):
            print(f"    Interval {i+1} [{i*10}-{(i+1)*10}%]: {len(interval)} channels")

        # Test each interval
        for interval_idx, interval_name in enumerate(interval_names):
            print(f"\n  Testing: {interval_name}")

            # Skip baseline for now, will handle separately
            if interval_name == 'Baseline':
                continue

            # Get the interval indices
            interval_indices = intervals[interval_idx]

            # Test each noise level
            for noise_strength in noise_levels:
                print(f"    σ={noise_strength}...", end=' ')

                road_scores = []
                accuracy_scores = []

                for run_idx in range(NUM_RUNS):
                    # Create noise mask for this interval
                    noise_mask = create_noise_mask_for_interval(num_channels, interval_indices)

                    # Apply noise injection
                    model_noisy = apply_noise_injection(
                        model_fp32, target_layer_name, noise_mask, noise_strength,
                        noise_type=noise_type, normalize=normalize
                    )
                    model_noisy = model_noisy.to(device)

                    # Compute accuracy
                    accuracy = compute_model_accuracy(
                        model_noisy, [test_image], [target_class], device
                    )
                    accuracy_scores.append(accuracy)

                    # Generate heatmap
                    noisy_layer = model_noisy.layer4[2].conv3
                    cam_noisy = SimpleGradCAM(model_noisy, noisy_layer)

                    with torch.enable_grad():
                        heatmap_noisy = cam_noisy(test_image, target_class)

                    heatmap_noisy_np = heatmap_noisy.detach().cpu().numpy()[0]

                    # Evaluate with ROAD
                    road_score = evaluate_explanation_with_road(
                        model_noisy, [test_image[0]], [target_class],
                        [heatmap_noisy_np], device
                    )
                    road_scores.append(road_score)

                avg_road = np.mean(road_scores)
                avg_accuracy = np.mean(accuracy_scores)
                all_images_results[interval_name][noise_strength]['road'].append(avg_road)
                all_images_results[interval_name][noise_strength]['accuracy'].append(avg_accuracy)
                print(f"ROAD={avg_road:.4f}, Acc={avg_accuracy:.4f}")

        # Test baseline (no noise)
        print(f"\n  Testing: Baseline (No Noise)")
        for noise_strength in noise_levels:
            print(f"    σ={noise_strength}...", end=' ')

            road_scores = []
            accuracy_scores = []

            for run_idx in range(NUM_RUNS):
                # No noise mask
                noise_mask = torch.zeros(num_channels, dtype=torch.bool)

                model_noisy = apply_noise_injection(
                    model_fp32, target_layer_name, noise_mask, noise_strength,
                    noise_type=noise_type, normalize=normalize
                )
                model_noisy = model_noisy.to(device)

                # Compute accuracy
                accuracy = compute_model_accuracy(
                    model_noisy, [test_image], [target_class], device
                )
                accuracy_scores.append(accuracy)

                noisy_layer = model_noisy.layer4[2].conv3
                cam_noisy = SimpleGradCAM(model_noisy, noisy_layer)

                with torch.enable_grad():
                    heatmap_noisy = cam_noisy(test_image, target_class)

                heatmap_noisy_np = heatmap_noisy.detach().cpu().numpy()[0]

                road_score = evaluate_explanation_with_road(
                    model_noisy, [test_image[0]], [target_class],
                    [heatmap_noisy_np], device
                )
                road_scores.append(road_score)

            avg_road = np.mean(road_scores)
            avg_accuracy = np.mean(accuracy_scores)
            all_images_results['Baseline'][noise_strength]['road'].append(avg_road)
            all_images_results['Baseline'][noise_strength]['accuracy'].append(avg_accuracy)
            print(f"ROAD={avg_road:.4f}, Acc={avg_accuracy:.4f}")

        print(f"\n  Image {img_idx + 1} complete")

    # Compute final statistics
    print("\n" + "="*80)
    print("Step 7: Computing Statistics Across All Images")
    print("="*80)

    results_dict = {}
    for interval_name in interval_names:
        strategy_results = {}
        for noise_strength in noise_levels:
            image_road_scores = all_images_results[interval_name][noise_strength]['road']
            image_accuracy_scores = all_images_results[interval_name][noise_strength]['accuracy']

            mean_road = np.mean(image_road_scores)
            std_road = np.std(image_road_scores)
            mean_accuracy = np.mean(image_accuracy_scores)
            std_accuracy = np.std(image_accuracy_scores)

            strategy_results[noise_strength] = {
                'road': mean_road,
                'road_std': std_road,
                'accuracy': mean_accuracy,
                'accuracy_std': std_accuracy,
            }

        results_dict[interval_name] = strategy_results

        # Print summary
        print(f"\n{interval_name}:")
        for noise_strength in noise_levels:
            road = strategy_results[noise_strength]['road']
            road_std = strategy_results[noise_strength]['road_std']
            accuracy = strategy_results[noise_strength]['accuracy']
            accuracy_std = strategy_results[noise_strength]['accuracy_std']
            print(f"  σ={noise_strength:.1f}: ROAD={road:.4f}±{road_std:.4f}, Acc={accuracy:.4f}±{accuracy_std:.4f}")

    # Visualization
    print("\n" + "="*80)
    print("Step 8: Generating Visualization")
    print("="*80)

    os.makedirs("./results", exist_ok=True)
    save_path = "./results/channel_noise_sensitivity_10intervals_activation.png"

    plot_noise_sensitivity_curves_10intervals(results_dict, save_path)

    # Print interpretation guide
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("\n⚠️  IMPORTANT: Intervals are sorted by activation in DESCENDING order")
    print("\n  Interval Mapping:")
    print("  ─────────────────────────────────────────────────────────")
    print("  🔴 [0-10%]     = Top 10%      (HIGHEST activation)")
    print("  🟠 [10-20%]    = 10-20%")
    print("  🟡 [20-30%]    = 20-30%")
    print("  🟡 [30-40%]    = 30-40%")
    print("  🟢 [40-50%]    = 40-50%")
    print("  🟢 [50-60%]    = 50-60%")
    print("  🔵 [60-70%]    = 60-70%")
    print("  🔵 [70-80%]    = 70-80%")
    print("  🟣 [80-90%]    = 80-90%")
    print("  🟣 [90-100%]   = Bottom 10%   (LOWEST activation)")
    print("  ⚫ Baseline    = No Noise (reference)")
    print("  ─────────────────────────────────────────────────────────")
    print("\n  Expected behavior:")
    print("    - Top 10% noise → ROAD increases (explainability degrades)")
    print("    - Bottom 10% noise → ROAD stable (minimal impact)")
    print("\n" + "="*80)

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\nKey Findings (at σ=5.0):")
    print("-" * 80)

    test_noise = 5.0
    for interval_name in interval_names:
        if interval_name == 'Baseline':
            continue
        road = results_dict[interval_name][test_noise]['road']
        road_std = results_dict[interval_name][test_noise]['road_std']
        accuracy = results_dict[interval_name][test_noise]['accuracy']
        accuracy_std = results_dict[interval_name][test_noise]['accuracy_std']
        print(f"  {interval_name:<30}: ROAD={road:.4f}±{road_std:.4f}, Acc={accuracy:.4f}±{accuracy_std:.4f}")

    baseline_road = results_dict['Baseline'][test_noise]['road']
    baseline_accuracy = results_dict['Baseline'][test_noise]['accuracy']
    print(f"  {'Baseline':<30}: ROAD={baseline_road:.4f}, Acc={baseline_accuracy:.4f}")

    # Find most/least sensitive intervals
    interval_roads = {name: results_dict[name][test_noise]['road']
                     for name in interval_names if name != 'Baseline'}

    most_sensitive = max(interval_roads, key=interval_roads.get)
    least_sensitive = min(interval_roads, key=interval_roads.get)

    print(f"\nMost sensitive interval: {most_sensitive} (ROAD={interval_roads[most_sensitive]:.4f})")
    print(f"  → This should be '[0-10%] Top (Highest Activation)'")
    print(f"\nLeast sensitive interval: {least_sensitive} (ROAD={interval_roads[least_sensitive]:.4f})")
    print(f"  → This should be '[90-100%] Bottom (Lowest Activation)'")

    # Verify correctness
    print("\n" + "="*80)
    print("SANITY CHECK")
    print("="*80)
    if "Interval-1" in most_sensitive:
        print("✅ CORRECT: Top 10% (Interval-1) is most sensitive")
    else:
        print("⚠️  WARNING: Expected Interval-1 to be most sensitive!")

    if "Interval-10" in least_sensitive or "Interval-9" in least_sensitive:
        print("✅ CORRECT: Bottom intervals are least sensitive")
    else:
        print("⚠️  WARNING: Expected bottom intervals to be least sensitive!")

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Division: 10 intervals (10% each)")
    print(f"  - Importance basis: Activation magnitude (DESCENDING order)")
    print(f"  - Test images: {NUM_TEST_IMAGES}")
    print(f"  - Runs per noise level: {NUM_RUNS}")
    print(f"\nResults saved to:")
    print(f"  - {save_path}")
    print(f"\n✅ Fine-grained 10-interval analysis complete!")
    print("\n")


if __name__ == "__main__":
    main()
