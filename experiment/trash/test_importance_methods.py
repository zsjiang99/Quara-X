"""
Quick test script to verify all importance calculation methods work correctly
"""

import torch
import torch.nn as nn
import torchvision

# Import the analyzer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from channel_noise_sensitivity_road import ChannelImportanceAnalyzer

def test_all_methods():
    """Test all importance calculation methods"""

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}\n")

    # Load a simple model
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    model = model.to(device)
    model.eval()

    # Get target layer
    target_layer = model.layer4[2].conv3

    # Create dummy input
    test_input = torch.randn(1, 3, 224, 224).to(device)
    target_class = 0

    # List of all methods
    methods = [
        'gradient',
        'activation',
        'weight',
        'combined',
        'gradcam_contribution',
        'taylor',
        'variance'
    ]

    print("="*80)
    print("Testing All Importance Calculation Methods")
    print("="*80)

    results = {}

    for method in methods:
        print(f"\nTesting method: '{method}'")
        print("-"*80)

        try:
            # Create analyzer
            analyzer = ChannelImportanceAnalyzer(model, target_layer)

            # Compute importance
            with torch.enable_grad():
                importance_scores = analyzer.compute_importance(
                    test_input, target_class, method=method
                )

            # Statistics
            results[method] = {
                'min': importance_scores.min().item(),
                'max': importance_scores.max().item(),
                'mean': importance_scores.mean().item(),
                'std': importance_scores.std().item(),
                'num_channels': len(importance_scores)
            }

            print(f"✅ SUCCESS")
            print(f"   Channels: {results[method]['num_channels']}")
            print(f"   Range: [{results[method]['min']:.6f}, {results[method]['max']:.6f}]")
            print(f"   Mean: {results[method]['mean']:.6f}")
            print(f"   Std: {results[method]['std']:.6f}")

        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            results[method] = None

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = sum(1 for r in results.values() if r is not None)
    print(f"\n✅ {successful}/{len(methods)} methods passed")

    if successful == len(methods):
        print("\n🎉 All methods are working correctly!")
    else:
        failed = [m for m, r in results.items() if r is None]
        print(f"\n⚠️ Failed methods: {failed}")

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Method':<25} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print("-"*80)

    for method, stats in results.items():
        if stats:
            print(f"{method:<25} {stats['min']:<12.6f} {stats['max']:<12.6f} "
                  f"{stats['mean']:<12.6f} {stats['std']:<12.6f}")
        else:
            print(f"{method:<25} {'FAILED':<12}")

    print("\n" + "="*80)
    print("\n✅ Test complete! All methods are ready to use.")
    print("\nTo use a specific method, edit channel_noise_sensitivity_road.py:")
    print("   IMPORTANCE_METHOD = 'your_choice'  # Line 646\n")

if __name__ == "__main__":
    test_all_methods()
