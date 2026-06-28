# Blend mask creation for smooth compositing of restored regions.
# Uses Python interop for torch tensor operations (box blur via conv2d).

from std.python import Python, PythonObject


# Blend parameters
comptime BLEND_DILATION_RATIO = 0.028
comptime BLEND_FALLOFF_RATIO = 0.028

# Kernel cache for box blur (stored as Python dict for mutability)
var _kernel_cache: PythonObject = PythonObject()


def _make_odd(n: Int) raises -> Int:
    """Make a number odd by adding 1 if even."""
    return n if n % 2 == 1 else n + 1


def _box_blur(x: PythonObject, kernel_size: Int) raises -> PythonObject:
    """Apply box blur using cached kernel and conv2d.
    
    Args:
        x: torch.Tensor (H, W) float
        kernel_size: Size of square kernel (must be odd)
    Returns:
        torch.Tensor (H, W) blurred
    """
    var torch = Python.import_module("torch")
    var F = Python.import_module("torch.nn.functional")

    var device_str = String(py=x.device.__str__()
    var dtype_str = String(py=x.dtype.__str__()
    var cache_key = device_str + ":" + dtype_str + ":" + String(kernel_size)

    # Initialize cache as Python dict if needed
    if not _kernel_cache:
        _kernel_cache = Python.dict()

    var kernel = PythonObject()
    if _kernel_cache.__contains__(cache_key):
        kernel = _kernel_cache[cache_key]
    else:
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size),
            device=x.device,
            dtype=x.dtype,
        ) / (kernel_size ** 2)
        _kernel_cache[cache_key] = kernel

    var pad = kernel_size // 2
    var x4d = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x4d, kernel).squeeze(0).squeeze(0)


def create_blend_mask(crop_mask: PythonObject, frame_height: Int) raises -> PythonObject:
    """Create blend mask from detection mask with dilation and falloff.
    
    Dilation ensures blend weight=1.0 extends past the mask edge to cover
    any adjacent mosaic blocks the detector missed. Falloff creates a
    smooth transition entirely outside the mosaic area.
    Both are proportional to frame height (~30px each at 1080p).
    
    Args:
        crop_mask: torch.Tensor (H, W) — detection mask
        frame_height: Height of the original frame (for proportional sizing)
    Returns:
        torch.Tensor (H, W) float in [0, 1] — blend mask
    """
    var torch = Python.import_module("torch")

    var mask = crop_mask.squeeze()
    var blend_dtype = mask.dtype if mask.is_floating_point() else torch.get_default_dtype()

    var dilation_px = max(3, round(Float64(frame_height) * BLEND_DILATION_RATIO))
    var falloff_px = max(3, round(Float64(frame_height) * BLEND_FALLOFF_RATIO))

    var dilate_k = _make_odd(Int(dilation_px) * 2 + 1)
    var falloff_k = _make_odd(Int(falloff_px) * 2 + 1)

    var blend = (mask > 0).to(dtype=blend_dtype)
    blend = _box_blur(blend, dilate_k)
    blend = (blend > 0.01).to(dtype=blend_dtype)
    blend = _box_blur(blend, falloff_k)

    return blend.clamp_(0.0, 1.0)
