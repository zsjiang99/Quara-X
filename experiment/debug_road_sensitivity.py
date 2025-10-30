"""
Enhanced debug: Test ROAD sensitivity with different parameters
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

sys.path.append('/home/zjian137/Quara-X/experiment/road_evaluation')
from road import run_road
from road.imputations import NoisyLinearImputer


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


# Simple GradCAM implementation
class SimpleGradCAM:
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


def evaluate_with_road(model, test_images, test_labels, heatmaps, device,
                       percentages, imputer_noise):
    """Evaluate with ROAD"""
    dataset = SimpleImageDataset(test_images, test_labels)

    explanations = []
    for heatmap in heatmaps:
        attr = np.stack([heatmap] * 3, axis=0)
        explanations.append(attr)

    transform_test = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    imputer = NoisyLinearImputer(noise=imputer_noise)

    model.eval()

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

    return road_score, prob_acc


def main():
    print("="*80)
    print("Testing ROAD Sensitivity with Different Parameters")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    data_root = "/home/zjian137/CAMBoost/gradcam/data/tiny-imagenet"
    transform = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    val_set = ImageFolder(os.path.join(data_root, "val"), transform=transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    state_dict = torch.load("/home/zjian137/CAMBoost/gradcam/resnet50_tinyimagenet_finetuned.pth",
                           map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    target_layer = model.layer4[2].conv3

    # Get multiple test images
    NUM_SAMPLES = 5
    test_images = []
    test_labels = []
    test_classes = []

    for img, label in val_loader:
        if len(test_images) >= NUM_SAMPLES:
            break
        test_images.append(img.to(device))
        test_labels.append(label.item())

        with torch.no_grad():
            output = model(img.to(device))
            pred = output.argmax(dim=1).item()
        test_classes.append(pred)

    print(f"\nCollected {len(test_images)} test samples")

    # Generate real GradCAM heatmaps for all samples
    print("\nGenerating real GradCAM heatmaps...")
    gradcam = SimpleGradCAM(model, target_layer)
    real_heatmaps = []

    for img, pred_class in zip(test_images, test_classes):
        with torch.enable_grad():
            heatmap = gradcam(img, pred_class)
        real_heatmaps.append(heatmap.detach().cpu().numpy()[0])

    # Also create zero heatmaps
    zero_heatmaps = [np.zeros((64, 64), dtype=np.float32) for _ in range(NUM_SAMPLES)]

    # Test different ROAD configurations
    print("\n" + "="*80)
    print("Testing Different ROAD Configurations")
    print("="*80)

    configs = [
        {'percentages': [0.1, 0.3, 0.5], 'imputer_noise': 0.01, 'name': 'Original (p=[0.1,0.3,0.5], noise=0.01)'},
        {'percentages': [0.3, 0.5, 0.7], 'imputer_noise': 0.01, 'name': 'Higher % (p=[0.3,0.5,0.7], noise=0.01)'},
        {'percentages': [0.5, 0.7, 0.9], 'imputer_noise': 0.01, 'name': 'Very High % (p=[0.5,0.7,0.9], noise=0.01)'},
        {'percentages': [0.1, 0.3, 0.5], 'imputer_noise': 0.1, 'name': 'Higher Noise (p=[0.1,0.3,0.5], noise=0.1)'},
        {'percentages': [0.1, 0.3, 0.5], 'imputer_noise': 0.5, 'name': 'Very High Noise (p=[0.1,0.3,0.5], noise=0.5)'},
    ]

    results = []

    for config in configs:
        print(f"\n{config['name']}:")

        # Evaluate real GradCAM
        print("  Real GradCAM...", end=' ')
        road_real, prob_acc_real = evaluate_with_road(
            model,
            [img[0] for img in test_images],
            test_labels,
            real_heatmaps,
            device,
            config['percentages'],
            config['imputer_noise']
        )
        print(f"ROAD={road_real:.6f}")

        # Evaluate zero heatmap
        print("  Zero heatmap...", end=' ')
        road_zero, prob_acc_zero = evaluate_with_road(
            model,
            [img[0] for img in test_images],
            test_labels,
            zero_heatmaps,
            device,
            config['percentages'],
            config['imputer_noise']
        )
        print(f"ROAD={road_zero:.6f}")

        diff = abs(road_real - road_zero)
        print(f"  Difference: {diff:.6f} ({diff/road_zero*100:.1f}% relative)")

        results.append({
            'name': config['name'],
            'road_real': road_real,
            'road_zero': road_zero,
            'diff': diff,
            'prob_acc_real': prob_acc_real.cpu().numpy(),
            'prob_acc_zero': prob_acc_zero.cpu().numpy(),
            'percentages': config['percentages']
        })

    # Visualize results
    fig, axes = plt.subplots(2, len(configs), figsize=(4*len(configs), 8))

    for idx, result in enumerate(results):
        # Top row: prob_acc curves
        ax = axes[0, idx] if len(configs) > 1 else axes[0]
        ax.plot(result['percentages'], result['prob_acc_real'],
                marker='o', label='Real GradCAM', linewidth=2)
        ax.plot(result['percentages'], result['prob_acc_zero'],
                marker='s', label='Zero Heatmap', linewidth=2)
        ax.set_xlabel('Removal %')
        ax.set_ylabel('Prob Accuracy')
        ax.set_title(f"{result['name']}\nROAD Diff: {result['diff']:.6f}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Bottom row: ROAD comparison
        ax2 = axes[1, idx] if len(configs) > 1 else axes[1]
        bars = ax2.bar(['Real\nGradCAM', 'Zero\nHeatmap'],
                       [result['road_real'], result['road_zero']])
        bars[0].set_color('steelblue')
        bars[1].set_color('coral')
        ax2.set_ylabel('ROAD Score')
        ax2.set_title(f"ROAD Comparison", fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/debug_road_sensitivity.png', dpi=200, bbox_inches='tight')
    print(f"\n\nVisualization saved to: ./results/debug_road_sensitivity.png")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nConfiguration Comparison:")
    print(f"{'Config':<50} {'Real ROAD':<12} {'Zero ROAD':<12} {'Difference':<12} {'% Diff':<10}")
    print("-" * 96)
    for result in results:
        pct_diff = result['diff'] / result['road_zero'] * 100 if result['road_zero'] > 0 else 0
        print(f"{result['name']:<50} {result['road_real']:<12.6f} {result['road_zero']:<12.6f} {result['diff']:<12.6f} {pct_diff:<10.1f}%")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    best_config = max(results, key=lambda x: x['diff'])
    print(f"\nBest configuration (highest difference):")
    print(f"  {best_config['name']}")
    print(f"  Difference: {best_config['diff']:.6f}")

    if best_config['diff'] < 0.01:
        print("\n⚠️  WARNING: Even the best configuration shows very small difference!")
        print("   This suggests:")
        print("   1. Model is extremely robust on these samples")
        print("   2. OR imputation quality is too high")
        print("   3. OR ROAD may not be sensitive enough for this task")
    else:
        print("\n✅ Good! This configuration can distinguish real vs zero heatmaps")

    print("="*80)


if __name__ == "__main__":
    main()
