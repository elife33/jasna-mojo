# Video reader factory — creates video readers for different platforms.
# Uses Python interop for PyAV (cross-platform) and NVIDIA NVDEC (CUDA only).

from std.python import Python, PythonObject


# ============================================================================
# Video Reader Factory
# ============================================================================

struct VideoReaderFactory:
    """Factory for creating video readers based on platform and device."""

    @staticmethod
    def create_reader(
        input_video: String,
        batch_size: Int,
        device: PythonObject,
        metadata: PythonObject,
    ) raises -> PythonObject:
        """Create a video reader appropriate for the device.
        
        On NVIDIA CUDA: tries NVDEC reader first, falls back to PyAV.
        On Apple Silicon/CPU: uses PyAV reader.
        
        Args:
            input_video: Path to input video
            batch_size: Batch size for reading
            device: torch.device
            metadata: VideoMetadata
        Returns:
            Video reader context manager
        """
        var device_type = String(py=device.type)

        if device_type == "cuda":
            # Try NVIDIA NVDEC reader
            var try_nvdec = Python.evaluate("""
def _try_nvdec(path, batch_size, device, metadata):
    try:
        from jasna.media.video_nv_decoder import NvVideoReader
        return NvVideoReader(
            video_path=path,
            batch_size=batch_size,
            device=device,
            metadata=metadata,
        )
    except Exception:
        return None
""", file=True)

            var nv_reader = try_nvdec._try_nvdec(input_video, batch_size, device, metadata)
            if nv_reader is not None:
                return nv_reader

        # Fall back to PyAV reader (cross-platform)
        var create_pyav = Python.evaluate("""
def _create_pyav(path, batch_size, device, metadata):
    from jasna.media.video_reader import PyAVVideoReader
    return PyAVVideoReader(
        video_path=path,
        batch_size=batch_size,
        device=device,
        metadata=metadata,
    )
""", file=True)

        return create_pyav._create_pyav(input_video, batch_size, device, metadata)
