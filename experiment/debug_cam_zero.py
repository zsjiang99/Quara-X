"""
Debug script to investigate why CAM becomes zero
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from channel_sensitivity_selective_quantization import (
    symmetric_quantize,
    per_channel_quantize
)


def debug_cam_computation(weights, quant_activations):
    """Debug CAM computation step by step"""

    print("\n" + "="*80)
    print("Debugging CAM Computation")
    print("="*80)

    print(f"\n1. Input shapes:")
    print(f"   weights: {weights.shape}")  # [B, C, 1, 1]
    print(f"   quant_activations: {quant_activations.shape}")  # [B, C, H, W]

    print(f"\n2. Value statistics:")
    print(f"   weights - min: {weights.min():.6f}, max: {weights.max():.6f}, mean: {weights.mean():.6f}")
    print(f"   quant_activations - min: {quant_activations.min():.6f}, max: {quant_activations.max():.6f}, mean: {quant_activations.mean():.6f}")

    # Check if weights are all zero
    weights_zero_count = (weights == 0).sum().item()
    weights_total = weights.numel()
    print(f"\n3. Zero values:")
    print(f"   weights: {weights_zero_count}/{weights_total} ({100*weights_zero_count/weights_total:.1f}%)")

    activations_zero_count = (quant_activations == 0).sum().item()
    activations_total = quant_activations.numel()
    print(f"   quant_activations: {activations_zero_count}/{activations_total} ({100*activations_zero_count/activations_total:.1f}%)")

    # Compute weighted activation
    weighted = weights * quant_activations  # [B, C, H, W]
    print(f"\n4. After multiplication (weights * quant_activations):")
    print(f"   Shape: {weighted.shape}")
    print(f"   min: {weighted.min():.6f}, max: {weighted.max():.6f}, mean: {weighted.mean():.6f}")

    # Check per-channel contribution
    print(f"\n5. Per-channel contribution after multiplication:")
    channel_sums = weighted.sum(dim=(0, 2, 3))  # Sum over batch, H, W
    print(f"   Top 5 positive channels: {channel_sums.topk(5).values.cpu().numpy()}")
    print(f"   Top 5 negative channels: {channel_sums.topk(5, largest=False).values.cpu().numpy()}")

    # Sum over channels
    cam = weighted.sum(dim=1, keepdim=True)  # [B, 1, H, W]
    print(f"\n6. After sum over channels:")
    print(f"   Shape: {cam.shape}")
    print(f"   min: {cam.min():.6f}, max: {cam.max():.6f}, mean: {cam.mean():.6f}")
    print(f"   Zero pixels: {(cam == 0).sum().item()}/{cam.numel()}")

    # After ReLU
    cam_relu = F.relu(cam)
    print(f"\n7. After ReLU:")
    print(f"   min: {cam_relu.min():.6f}, max: {cam_relu.max():.6f}, mean: {cam_relu.mean():.6f}")
    print(f"   Zero pixels: {(cam_relu == 0).sum().item()}/{cam_relu.numel()}")

    return cam


def test_quantization_effect():
    """Test how quantization affects gradients and activations"""

    print("\n" + "="*80)
    print("Testing Quantization Effect")
    print("="*80)

    # Create synthetic data
    torch.manual_seed(42)
    B, C, H, W = 1, 2048, 7, 7

    # Simulate gradients (typically small values)
    gradients = torch.randn(B, C, H, W) * 0.01  # Small gradients

    # Simulate activations (typically positive, various magnitudes)
    activations = torch.relu(torch.randn(B, C, H, W))

    print(f"\nOriginal data:")
    print(f"  gradients - min: {gradients.min():.6f}, max: {gradients.max():.6f}, mean: {gradients.mean():.6f}")
    print(f"  activations - min: {activations.min():.6f}, max: {activations.max():.6f}, mean: {activations.mean():.6f}")

    # Test different bit widths
    for bit_width in [8, 4, 2]:
        print(f"\n{'='*60}")
        print(f"Bit-width: INT{bit_width}")
        print(f"{'='*60}")

        # Quantize a subset of channels (first half)
        channel_mask = torch.zeros(C, dtype=torch.bool)
        channel_mask[:C//2] = True

        quant_gradients = per_channel_quantize(gradients, bit_width, channel_mask)
        quant_activations = per_channel_quantize(activations, bit_width, channel_mask)

        print(f"\nAfter quantization:")
        print(f"  quant_gradients - min: {quant_gradients.min():.6f}, max: {quant_gradients.max():.6f}")
        print(f"  quant_activations - min: {quant_activations.min():.6f}, max: {quant_activations.max():.6f}")

        # Count zeros in quantized channels
        quant_grad_zeros = (quant_gradients[:, :C//2, :, :] == 0).sum().item()
        quant_act_zeros = (quant_activations[:, :C//2, :, :] == 0).sum().item()
        total_quant_elements = B * (C//2) * H * W

        print(f"\nZeros in quantized channels:")
        print(f"  gradients: {quant_grad_zeros}/{total_quant_elements} ({100*quant_grad_zeros/total_quant_elements:.1f}%)")
        print(f"  activations: {quant_act_zeros}/{total_quant_elements} ({100*quant_act_zeros/total_quant_elements:.1f}%)")

        # Compute CAM
        weights = quant_gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * quant_activations).sum(dim=1, keepdim=True)

        print(f"\nCAM result:")
        print(f"  min: {cam.min():.6f}, max: {cam.max():.6f}, mean: {cam.mean():.6f}")
        print(f"  All zeros? {(cam == 0).all().item()}")


def test_scale_computation():
    """Test how scale is computed in symmetric quantization"""

    print("\n" + "="*80)
    print("Testing Scale Computation")
    print("="*80)

    # Create channel data with different magnitudes
    test_cases = [
        ("Small values (0.001-0.01)", torch.randn(1, 7, 7) * 0.01),
        ("Medium values (0.1-1.0)", torch.randn(1, 7, 7) * 0.5),
        ("Large values (1.0-10.0)", torch.randn(1, 7, 7) * 5.0),
        ("Very small values (1e-5)", torch.randn(1, 7, 7) * 1e-5),
    ]

    for name, data in test_cases:
        print(f"\n{name}:")
        print(f"  Original - min: {data.min():.8f}, max: {data.max():.8f}")

        for bit_width in [8, 4, 2]:
            quant_data = symmetric_quantize(data, bit_width)
            zeros = (quant_data == 0).sum().item()
            total = quant_data.numel()

            print(f"  INT{bit_width} - min: {quant_data.min():.8f}, max: {quant_data.max():.8f}, zeros: {zeros}/{total} ({100*zeros/total:.1f}%)")


if __name__ == "__main__":
    print("="*80)
    print("CAM Zero Value Debug Tool")
    print("="*80)

    test_scale_computation()
    test_quantization_effect()

    print("\n" + "="*80)
    print("Debug complete!")
    print("="*80)
