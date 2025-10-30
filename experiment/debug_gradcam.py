"""
Enhanced SelectiveQuantizedGradCAM with debugging
"""

import torch
import torch.nn.functional as F


def symmetric_quantize_debug(x, bit_width, name="tensor"):
    """Symmetric quantization with debug output"""
    if bit_width >= 32:
        return x, {"all_zero": False, "zero_ratio": 0.0}

    n_levels = 2 ** bit_width
    qmax = n_levels // 2 - 1

    # Use percentile-based scale
    abs_x = x.abs()
    scale = torch.quantile(abs_x.flatten(), 0.999) / qmax

    if scale < 1e-8:
        scale = abs_x.max() / qmax

    if scale == 0 or scale < 1e-8:
        print(f"  WARNING: {name} has scale=0, returning zeros")
        return torch.zeros_like(x), {"all_zero": True, "zero_ratio": 1.0, "scale": 0}

    # Quantize
    x_int = torch.round(x / scale).clamp(-qmax-1, qmax)
    x_quant = x_int * scale

    # Debug info
    zero_count = (x_quant == 0).sum().item()
    total_count = x_quant.numel()
    zero_ratio = zero_count / total_count

    debug_info = {
        "all_zero": (zero_count == total_count),
        "zero_ratio": zero_ratio,
        "scale": scale.item(),
        "orig_min": x.min().item(),
        "orig_max": x.max().item(),
        "quant_min": x_quant.min().item(),
        "quant_max": x_quant.max().item(),
    }

    print(f"  {name}: scale={scale.item():.6f}, zeros={zero_count}/{total_count} ({100*zero_ratio:.1f}%)")

    return x_quant, debug_info


class DebugSelectiveQuantizedGradCAM:
    """
    GradCAM with debugging output
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
        Compute GradCAM with debugging
        """
        print(f"\n{'='*60}")
        print(f"GradCAM Debug - Bit Width: {bit_width}")
        print(f"{'='*60}")

        self.model.eval()
        self.model.zero_grad()

        # Forward and backward
        output = self.model(input_tensor)
        scores = output[:, target_class]
        scores.sum().backward()

        # Get clean values
        clean_activations = self.activations.clone()
        clean_gradients = self.gradients.clone()

        print(f"\nClean values:")
        print(f"  activations: min={clean_activations.min():.6f}, max={clean_activations.max():.6f}, mean={clean_activations.mean():.6f}")
        print(f"  gradients: min={clean_gradients.min():.6f}, max={clean_gradients.max():.6f}, mean={clean_gradients.mean():.6f}")

        # Apply channel mask if needed
        if channel_mask is not None and bit_width < 32:
            print(f"\nQuantizing {channel_mask.sum().item()} / {len(channel_mask)} channels...")

            quant_activations = clean_activations.clone()
            quant_gradients = clean_gradients.clone()

            # Quantize selected channels
            for c in range(clean_activations.shape[1]):
                if channel_mask[c]:
                    # Quantize activation for this channel
                    ch_act = clean_activations[:, c, :, :]
                    ch_act_quant, _ = symmetric_quantize_debug(ch_act, bit_width, f"act_ch{c}")
                    quant_activations[:, c, :, :] = ch_act_quant

                    # Quantize gradient for this channel
                    ch_grad = clean_gradients[:, c, :, :]
                    ch_grad_quant, _ = symmetric_quantize_debug(ch_grad, bit_width, f"grad_ch{c}")
                    quant_gradients[:, c, :, :] = ch_grad_quant
        else:
            quant_activations = clean_activations
            quant_gradients = clean_gradients

        print(f"\nAfter quantization:")
        print(f"  quant_activations: min={quant_activations.min():.6f}, max={quant_activations.max():.6f}, mean={quant_activations.mean():.6f}")
        print(f"  quant_gradients: min={quant_gradients.min():.6f}, max={quant_gradients.max():.6f}, mean={quant_gradients.mean():.6f}")

        # Compute weights
        weights = quant_gradients.mean(dim=(2, 3), keepdim=True)
        print(f"\nWeights (after spatial mean):")
        print(f"  shape: {weights.shape}")
        print(f"  min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}")
        zero_weights = (weights == 0).sum().item()
        print(f"  zero weights: {zero_weights}/{weights.numel()} ({100*zero_weights/weights.numel():.1f}%)")

        # Weighted activation
        weighted = weights * quant_activations
        print(f"\nWeighted activation (weights * quant_activations):")
        print(f"  min={weighted.min():.6f}, max={weighted.max():.6f}, mean={weighted.mean():.6f}")

        # Sum over channels
        cam = weighted.sum(dim=1, keepdim=True)
        print(f"\nCAM (sum over channels):")
        print(f"  shape: {cam.shape}")
        print(f"  min={cam.min():.6f}, max={cam.max():.6f}, mean={cam.mean():.6f}")
        zero_pixels = (cam == 0).sum().item()
        print(f"  zero pixels: {zero_pixels}/{cam.numel()} ({100*zero_pixels/cam.numel():.1f}%)")

        # Check if all zero
        if cam.abs().max() < 1e-8:
            print(f"\n  ⚠️  WARNING: CAM is essentially all zeros!")

        # ReLU
        cam = F.relu(cam)
        print(f"\nAfter ReLU:")
        print(f"  min={cam.min():.6f}, max={cam.max():.6f}, mean={cam.mean():.6f}")

        # Normalize
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)

        if (cam_max - cam_min).max() < 1e-8:
            print(f"  ⚠️  WARNING: Cannot normalize, CAM has no range!")
            cam = torch.zeros_like(cam)
        else:
            cam = cam - cam_min
            cam = cam / (cam_max - cam_min + 1e-8)

        # Upsample
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                          mode='bilinear', align_corners=False)

        print(f"\nFinal CAM:")
        print(f"  shape: {cam.shape}")
        print(f"  min={cam.min():.6f}, max={cam.max():.6f}, mean={cam.mean():.6f}")

        return cam.squeeze(1)


# Simple test
if __name__ == "__main__":
    print("This is a debug module. Import it in your main script.")
    print("\nUsage:")
    print("  from debug_gradcam import DebugSelectiveQuantizedGradCAM")
    print("  gradcam = DebugSelectiveQuantizedGradCAM(model, target_layer)")
    print("  heatmap = gradcam(image, target_class, channel_mask, bit_width)")
