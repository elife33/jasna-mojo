# VRAM offloader — manages VRAM by moving idle frames to system RAM.
# Uses Python interop for torch tensor operations and threading.

from std.python import Python, PythonObject


struct VramOffloader:
    """Manages VRAM by offloading idle frames to system RAM when VRAM is low.
    
    Monitors VRAM usage and moves frames from blend buffer and crop buffers
    to CPU RAM when the GPU runs low on memory. Frames are moved back when needed.
    """

    var device: PythonObject
    var _blend_buffer: PythonObject
    var _crop_buffers: PythonObject  # dict
    var _crop_lock: PythonObject
    var _clip_queue: PythonObject
    var _secondary_queue: PythonObject
    var _encode_queue: PythonObject
    var _metadata_queue: PythonObject
    var _encode_heartbeat: PythonObject  # list[float]
    var _activity_heartbeat: PythonObject  # list[float]
    var _running: Bool
    var _thread: PythonObject

    def __init__(
        mut self,
        device: PythonObject,
        blend_buffer: PythonObject,
        crop_buffers: PythonObject,
        crop_lock: PythonObject,
    ):
        self.device = device
        self._blend_buffer = blend_buffer
        self._crop_buffers = crop_buffers
        self._crop_lock = crop_lock
        self._clip_queue = PythonObject()
        self._secondary_queue = PythonObject()
        self._encode_queue = PythonObject()
        self._metadata_queue = PythonObject()
        self._encode_heartbeat = PythonObject()
        self._activity_heartbeat = PythonObject()
        self._running = False
        self._thread = PythonObject()

    def set_pipeline_queues(
        mut self,
        clip_queue: PythonObject,
        secondary_queue: PythonObject,
        encode_queue: PythonObject,
        metadata_queue: PythonObject,
    ):
        """Set the pipeline queues for monitoring."""
        self._clip_queue = clip_queue
        self._secondary_queue = secondary_queue
        self._encode_queue = encode_queue
        self._metadata_queue = metadata_queue

    def set_encode_heartbeat(mut self, heartbeat: PythonObject) raises:
        """Set the encode heartbeat for stall detection."""
        self._encode_heartbeat = heartbeat

    def set_pipeline_activity_heartbeat(mut self, heartbeat: PythonObject) raises:
        """Set the pipeline activity heartbeat."""
        self._activity_heartbeat = heartbeat

    def start(mut self) raises:
        """Start the VRAM offloader background thread."""
        if self._running:
            return

        # Only run on CUDA devices
        if String(py=self.device.type) != "cuda":
            return

        self._running = True

        # Create and start the offloader thread via Python interop
        var start_fn = Python.evaluate("""
def _start_offloader(device, blend_buffer, crop_buffers, crop_lock,
                     clip_queue, secondary_queue, encode_queue, metadata_queue,
                     encode_heartbeat, activity_heartbeat):
    from jasna.vram_offloader import VramOffloader as PyVramOffloader
    offloader = PyVramOffloader(
        device=device,
        blend_buffer=blend_buffer,
        crop_buffers=crop_buffers,
        crop_lock=crop_lock,
    )
    offloader.set_pipeline_queues(clip_queue, secondary_queue, encode_queue, metadata_queue)
    if encode_heartbeat:
        offloader.set_encode_heartbeat(encode_heartbeat)
    if activity_heartbeat:
        offloader.set_pipeline_activity_heartbeat(activity_heartbeat)
    offloader.start()
    return offloader
""", file=True)

        self._thread = start_fn._start_offloader(
            self.device, self._blend_buffer, self._crop_buffers, self._crop_lock,
            self._clip_queue, self._secondary_queue, self._encode_queue, self._metadata_queue,
            self._encode_heartbeat, self._activity_heartbeat,
        )

    def stop(mut self) raises:
        """Stop the VRAM offloader."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            var stop_fn = Python.evaluate("""
def _stop_offloader(offloader):
    if offloader is not None:
        offloader.stop()
""", file=True)

            stop_fn._stop_offloader(self._thread)
            self._thread = PythonObject()

    def pause_stall_check(self) raises:
        """Pause the stall check temporarily."""
        if self._thread is not None:
            var pause_fn = Python.evaluate("""
def _pause(offloader):
    if offloader is not None:
        offloader.pause_stall_check()
""", file=True)

            pause_fn._pause(self._thread)
