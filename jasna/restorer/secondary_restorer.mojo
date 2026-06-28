from jasna.py_compat import _hasattr
# Secondary restorer protocol — defines the interface for secondary restoration.
# Secondary restorers upscale or enhance the primary restoration output.

from std.python import Python, PythonObject


# ============================================================================
# Secondary restorer protocol (checked at runtime via Python interop)
# ============================================================================

def is_async_secondary_restorer(obj: PythonObject) raises -> Bool:
    """Check if an object implements the async secondary restorer protocol."""
    return Bool(py=_hasattr(obj, "push_clip")) and Bool(py=_hasattr(obj, "pop_completed"))


def is_secondary_restorer(obj: PythonObject) raises -> Bool:
    """Check if an object implements the secondary restorer protocol."""
    return Bool(py=_hasattr(obj, "restore"))


# ============================================================================
# Helper: create secondary restorer by name
# ============================================================================

def create_secondary_restorer(
    name: String,
    device: PythonObject,
    fp16: Bool,
    tvai_ffmpeg_path: String = "",
    tvai_model: String = "iris-2",
    tvai_scale: Int = 4,
    tvai_args: String = "",
    tvai_workers: Int = 2,
    rtx_scale: Int = 4,
    rtx_quality: String = "high",
    rtx_denoise: String = "medium",
    rtx_deblur: String = "none",
) raises -> PythonObject:
    """Create a secondary restorer by name.
    
    Args:
        name: "none", "unet-4x", "tvai", "rtx-super-res"
        device: torch.device
        fp16: Use FP16
        ... (TVAI and RTX specific params)
    Returns:
        PythonObject (secondary restorer) or None
    """
    var lower_name = name.lower()

    if lower_name == "none":
        return Python()
    elif lower_name == "tvai":
        var TvaiSecondaryRestorer = Python.evaluate("""
def _create_tvai(ffmpeg_path, tvai_args, scale, num_workers) raises:
    from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
    return TvaiSecondaryRestorer(
        ffmpeg_path=ffmpeg_path,
        tvai_args=tvai_args,
        scale=scale,
        num_workers=num_workers,
    )
""")
        var full_tvai_args = tvai_model + ":scale=" + String(tvai_scale) + ":" + tvai_args
        return TvaiSecondaryRestorer(tvai_ffmpeg_path, full_tvai_args, tvai_scale, tvai_workers)

    elif lower_name == "unet-4x":
        var Unet4x = Python.evaluate("""
def _create_unet4x(device, fp16) raises:
    from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer
    return Unet4xSecondaryRestorer(device=device, fp16=fp16)
""")
        return Unet4x(device, fp16)

    elif lower_name == "rtx-super-res":
        var RtxSuperRes = Python.evaluate("""
def _create_rtx(device, scale, quality, denoise, deblur) raises:
    from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer
    return RtxSuperresSecondaryRestorer(
        device=device,
        scale=scale,
        quality=quality,
        denoise=denoise,
        deblur=deblur,
    )
""")
        var dn = PythonObject() if rtx_denoise == "none" else PythonObject(rtx_denoise)
        var db = PythonObject() if rtx_deblur == "none" else PythonObject(rtx_deblur)
        return RtxSuperRes(device, rtx_scale, rtx_quality, dn, db)

    else:
        raise Error("Unsupported secondary restoration: " + name)
