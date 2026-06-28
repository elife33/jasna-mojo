# Streaming pipeline — HLS streaming mode for real-time playback.
# Uses Python interop for the HLS server and streaming encoder.

from std.python import Python, PythonObject


def run_streaming(
    pipeline: PythonObject,
    port: Int = 8765,
    segment_duration: Float64 = 4.0,
    hls_server: PythonObject = PythonObject(),
):
    """Run the pipeline in HLS streaming mode.
    
    Args:
        pipeline: Pipeline instance
        port: HTTP port for HLS server
        segment_duration: HLS segment duration in seconds
        hls_server: Optional existing HLS server
    """
    var run_fn = Python.evaluate("""
def _run_streaming(pipeline, port, segment_duration, hls_server) raises:
    from jasna.streaming_pipeline import run_streaming
    run_streaming(pipeline, port=port, segment_duration=segment_duration, hls_server=hls_server)
""")
    run_fn(pipeline, port, segment_duration, hls_server)


# ============================================================================
# HLS Streaming Server
# ============================================================================

struct HlsStreamingServer:
    """HLS streaming server for real-time video playback.
    
    Serves processed video via HTTP with HLS segments.
    Supports seeking and video switching.
    """

    var _server: PythonObject

    def __init__(
        mut self,
        segment_duration: Float64 = 4.0,
        port: Int = 8765,
    ):
        var create_fn = Python.evaluate("""
def _create_hls(segment_duration, port) raises:
    from jasna.streaming import HlsStreamingServer
    return HlsStreamingServer(segment_duration=segment_duration, port=port)
""")
        self._server = create_fn(segment_duration, port)

    def start(self) raises:
        """Start the HLS server."""
        self._server.start()

    def stop(self) raises:
        """Stop the HLS server."""
        self._server.stop()

    def wait_for_video(self) raises -> String:
        """Wait for a video to be selected by the client."""
        var path = self._server.wait_for_video()
        return String(path)

    def unload_video(self) raises:
        """Unload the current video."""
        self._server.unload_video()

    def video_change(self) raises -> Bool:
        """Check if a video change was requested."""
        return Bool(py=self._server.video_change.is_set())

    def consume_seek(self) raises -> Optional[Int]:
        """Consume a seek request if any."""
        var target = self._server.consume_seek()
        if target is None:
            return None
        return Int(target)

    def seek_requested(self) raises -> PythonObject:
        """Get the seek requested event."""
        return self._server.seek_requested
