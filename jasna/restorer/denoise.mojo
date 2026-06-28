# Spatial denoising — bilateral filter on GPU.
# Uses Python interop for torch tensor operations.

from std.python import Python, PythonObject


# ============================================================================
# Denoise enums
# ============================================================================

@fieldwise_init
struct DenoiseStrength:
    """Denoise strength levels."""
    var value: String

    @staticmethod
    def NONE() raises -> DenoiseStrength:
        return DenoiseStrength("none")

    @staticmethod
    def LOW() raises -> DenoiseStrength:
        return DenoiseStrength("low")

    @staticmethod
    def MEDIUM() raises -> DenoiseStrength:
        return DenoiseStrength("medium")

    @staticmethod
    def HIGH() raises -> DenoiseStrength:
        return DenoiseStrength("high")

    def __eq__(self, other: DenoiseStrength) raises -> Bool:
        return self.value == other.value


@fieldwise_init
struct DenoiseStep:
    """When to apply denoising in the pipeline."""
    var value: String

    @staticmethod
    def AFTER_PRIMARY() raises -> DenoiseStep:
        return DenoiseStep("after_primary")

    @staticmethod
    def AFTER_SECONDARY() raises -> DenoiseStep:
        return DenoiseStep("after_secondary")

    def __eq__(self, other: DenoiseStep) raises -> Bool:
        return self.value == other.value


# ============================================================================
# Denoise parameters
# ============================================================================

def _get_denoise_params(strength: DenoiseStrength) raises -> Tuple[Int, Float64, Float64]:
    """Get (kernel_size, sigma_spatial, sigma_range) for a denoise strength."""
    if strength == DenoiseStrength.LOW():
        return (5, 1.0, 0.04)
    elif strength == DenoiseStrength.MEDIUM():
        return (5, 1.5, 0.07)
    elif strength == DenoiseStrength.HIGH():
        return (5, 2.0, 0.09)
    else:
        return (0, 0.0, 0.0)


# ============================================================================
# Spatial bilateral denoise
# ============================================================================

def spatial_denoise(
    frames: PythonObject,
    kernel_size: Int,
    sigma_spatial: Float64,
    sigma_range: Float64,
) raises -> PythonObject:
    """Spatial bilateral filter on GPU (batched over T dimension).
    
    Each pixel is replaced by a weighted average of its spatial neighbours.
    Weights combine spatial proximity (Gaussian on distance) with intensity
    similarity (Gaussian on colour difference), so edges stay sharp while
    flat noisy areas are smoothed. Operates per-frame — no temporal ghosting.
    
    Args:
        frames: [T, C, H, W] float tensor in [0, 1]
        kernel_size: Spatial window side (odd)
        sigma_spatial: Gaussian sigma for spatial distance
        sigma_range: Gaussian sigma for intensity difference (in [0, 1])
    Returns:
        [T, C, H, W] denoised tensor
    """
    var torch = Python.import_module("torch")
    var F = Python.import_module("torch.nn.functional")

    var half = kernel_size // 2

    var offsets = torch.arange(-half, half + 1, dtype=frames.dtype, device=frames.device)
    var gy = offsets.unsqueeze(1).expand(kernel_size, kernel_size)
    var gx = offsets.unsqueeze(0).expand(kernel_size, kernel_size)
    var spatial_weights = torch.exp(-0.5 * (gx * gx + gy * gy) / (sigma_spatial ** 2))

    var padded = F.pad(frames, (half, half, half, half), mode="reflect")

    var range_scale = -0.5 / (sigma_range ** 2)
    var H = Int(frames.shape[2])
    var W = Int(frames.shape[3])

    var result = torch.zeros_like(frames)
    var weight_sum = torch.zeros(
        (Int(frames.shape[0]), 1, H, W),
        dtype=frames.dtype,
        device=frames.device,
    )

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            var neighbor = padded[:, :, dy:dy + H, dx:dx + W]
            var diff_sq = (frames - neighbor).pow(2).mean(dim=1, keepdim=True)
            var w = Float64(spatial_weights[dy, dx]) * torch.exp(diff_sq * range_scale)
            result.addcmul_(neighbor, w.expand_as(neighbor))
            weight_sum.add_(w)

    return result / weight_sum


# ============================================================================
# Apply denoise
# ============================================================================

def apply_denoise(frames: PythonObject, strength: DenoiseStrength) raises -> PythonObject:
    """Apply spatial denoise to float tensor [T, C, H, W] in [0, 1]."""
    if strength == DenoiseStrength.NONE():
        return frames

    var (kernel_size, sigma_spatial, sigma_range) = _get_denoise_params(strength)
    return spatial_denoise(frames, kernel_size, sigma_spatial, sigma_range)


def apply_denoise_u8(frames_u8: PythonObject, strength: DenoiseStrength) raises -> PythonObject:
    """Apply denoise to uint8 tensor. Accepts [T, C, H, W] or [C, H, W].
    
    Returns same shape and dtype.
    """
    var torch = Python.import_module("torch")

    if strength == DenoiseStrength.NONE():
        return frames_u8

    var single = Int(frames_u8.dim()) == 3
    var frames = frames_u8
    if single:
        frames = frames_u8.unsqueeze(0)

    var f = frames.float().div(255.0)
    var denoised = apply_denoise(f, strength)
    var out = denoised.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)

    if single:
        out = out.squeeze(0)

    return out
