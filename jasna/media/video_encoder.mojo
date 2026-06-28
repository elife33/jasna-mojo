# Video encoder factory — creates video encoders for different platforms.
# Uses Python interop for PyAV and platform-specific encoders.

from std.python import Python, PythonObject


# ============================================================================
# Video Encoder Factory
# ============================================================================

struct VideoEncoderFactory:
    """Factory for creating video encoders based on codec and platform."""

    @staticmethod
    def create_encoder(
        output_path: String,
        device: PythonObject,
        metadata: PythonObject,
        codec: String,
        encoder_settings: Dict[String, PythonObject],
        stream_mode: Bool = False,
        working_directory: PythonObject = PythonObject(),
        start_frame: Optional[Int] = None,
        duration_frames: Optional[Int] = None,
    ) raises -> PythonObject:
        """Create a video encoder.
        
        Args:
            output_path: Path to output video
            device: torch.device
            metadata: VideoMetadata
            codec: "h264", "hevc", or "libx264"
            encoder_settings: Encoder parameters
            stream_mode: True for HLS streaming
            working_directory: Directory for temp files
            start_frame: Start frame offset
            duration_frames: Number of frames to encode
        Returns:
            Video encoder context manager
        """
        var create_fn = Python.evaluate("""
def _create_encoder(output_path, device, metadata, codec, encoder_settings,
                    stream_mode, working_directory, start_frame, duration_frames):
    from jasna.media.video_encoder_factory import VideoEncoderFactory as PyVEF
    return PyVEF.create_encoder(
        String(output_path),
        device=device,
        metadata=metadata,
        codec=String(codec),
        encoder_settings=dict(encoder_settings),
        stream_mode=bool(stream_mode),
        working_directory=working_directory,
        start_frame=start_frame,
        duration_frames=duration_frames,
    )
""")
        var sf = PythonObject() if start_frame is None else PythonObject(start_frame.value)
        var df = PythonObject() if duration_frames is None else PythonObject(duration_frames.value)
        return create_fn(
            output_path, device, metadata, codec, encoder_settings,
            stream_mode, working_directory, sf, df,
        )
