"""
Debug script to understand why zero CAM gives similar ROAD score
"""

import os
import sys
import torch
import torch.nn as nn
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


def evaluate_with_road(model, test_images, test_labels, heatmaps, device):
    """Evaluate with ROAD"""
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
    print("Testing ROAD behavior with different heatmaps")
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

    # Get one test image
    for img, label in val_loader:
        test_img = img.to(device)
        test_label = label.item()
        break

    print(f"\nTest image shape: {test_img.shape}")
    print(f"Test label: {test_label}")

    # Create different heatmaps
    H, W = 64, 64

    # 1. Zero heatmap (all zeros)
    heatmap_zero = np.zeros((H, W), dtype=np.float32)

    # 2. Uniform heatmap (all same value)
    heatmap_uniform = np.ones((H, W), dtype=np.float32) * 0.5

    # 3. Random heatmap
    np.random.seed(42)
    heatmap_random = np.random.rand(H, W).astype(np.float32)

    # 4. Centered heatmap (Gaussian-like, simulating real GradCAM)
    y, x = np.ogrid[:H, :W]
    center_y, center_x = H // 2, W // 2
    heatmap_gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 20**2))
    heatmap_gaussian = heatmap_gaussian.astype(np.float32)

    # 5. Inverse Gaussian (attention on edges)
    heatmap_inverse = 1.0 - heatmap_gaussian

    heatmaps = {
        'Zero': heatmap_zero,
        'Uniform': heatmap_uniform,
        'Random': heatmap_random,
        'Gaussian (Center)': heatmap_gaussian,
        'Inverse (Edges)': heatmap_inverse,
    }

    # Evaluate each heatmap
    print("\n" + "="*80)
    print("Evaluating different heatmaps with ROAD")
    print("="*80)

    results = {}
    for name, heatmap in heatmaps.items():
        print(f"\n{name}:")
        print(f"  Heatmap stats: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")

        road_score, prob_acc = evaluate_with_road(
            model, [test_img[0]], [test_label], [heatmap], device
        )

        results[name] = {
            'road': road_score,
            'prob_acc': prob_acc.cpu().numpy()
        }

        print(f"  ROAD score: {road_score:.6f}")
        print(f"  Prob acc at each %: {prob_acc.cpu().numpy()}")

    # Visualize heatmaps and results
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for idx, (name, heatmap) in enumerate(heatmaps.items()):
        # Top row: heatmaps
        axes[0, idx].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[0, idx].set_title(f'{name}\nROAD={results[name]["road"]:.6f}', fontsize=10)
        axes[0, idx].axis('off')

        # Bottom row: ROAD prob_acc curves
        percentages = [0.1, 0.3, 0.5]
        prob_acc = results[name]['prob_acc']
        axes[1, idx].plot(percentages, prob_acc, marker='o', linewidth=2)
        axes[1, idx].set_xlabel('Removal %')
        axes[1, idx].set_ylabel('Prob Accuracy')
        axes[1, idx].set_title(f'{name}\nCurve')
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_ylim([0, 1.05])

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/debug_zero_cam_road.png', dpi=200, bbox_inches='tight')
    print(f"\n\nVisualization saved to: ./results/debug_zero_cam_road.png")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nROAD Scores (lower = better explanation):")
    for name in heatmaps.keys():
        print(f"  {name:<20}: {results[name]['road']:.6f}")

    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print("If Zero heatmap has similar ROAD to Random/Uniform,")
    print("it means ROAD treats them all as 'no useful explanation'.")
    print("\nIf Gaussian has much HIGHER ROAD, it means:")
    print("  - Removing important regions (center) degrades model performance")
    print("  - This is what we expect from a good explanation")
    print("\nIf all ROAD scores are VERY LOW (~0.003):")
    print("  - Model is very robust to pixel removal")
    print("  - OR the imputation method is too good")
    print("  - OR only 1 test sample is not enough")
    print("="*80)


if __name__ == "__main__":
    main()
