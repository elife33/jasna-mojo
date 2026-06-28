from jasna.py_compat import _hasattr
# Main pipeline orchestration — coordinates the 4-stage threaded pipeline.
# Stage 1: Decode + Detect → Stage 2: Primary Restore → Stage 3: Secondary Restore → Stage 4: Blend + Encode

from std.python import Python, PythonObject
from jasna.frame_queue import FrameQueue
from jasna.blend_buffer import BlendBuffer
from jasna.vram_offloader import VramOffloader
from jasna.progressbar import Progressbar
from jasna.pipeline_threads import (
    decode_detect_loop,
    primary_restore_loop,
    secondary_restore_loop,
    blend_encode_loop,
)
from jasna.media.video_metadata import get_video_meta_data, UnsupportedColorspaceError
from jasna.media.video_encoder import VideoEncoderFactory
from jasna.media.video_reader import VideoReaderFactory


struct Pipeline:
    """Main processing pipeline for video restoration.
    
    Coordinates 4 threaded stages:
    1. Decode + Detect: Read frames, run mosaic detection, track clips
    2. Primary Restore: Run BasicVSR++ on detected mosaic regions
    3. Secondary Restore: Optional upscaling/denoising
    4. Blend + Encode: Composite restored regions back and encode output
    """

    var input_video: String
    var output_video: String
    var codec: String
    var encoder_settings: Dict[String, PythonObject]
    var batch_size: Int
    var device: PythonObject
    var max_clip_size: Int
    var temporal_overlap: Int
    var enable_crossfade: Bool
    var fp16: Bool
    var detection_model: PythonObject
    var restoration_pipeline: PythonObject
    var disable_progress: Bool
    var progress_callback: PythonObject
    var working_directory: PythonObject
    var start_frame: Optional[Int]
    var duration_frames: Optional[Int]
    var current_frame_shared: List[Int]
    var pause_requested: List[Bool]
    var status_file_path: String

    def __init__(
        mut self,
        input_video: String,
        output_video: String,
        detection_model: PythonObject,
        restoration_pipeline: PythonObject,
        codec: String,
        encoder_settings: Dict[String, PythonObject],
        batch_size: Int,
        device: PythonObject,
        max_clip_size: Int,
        temporal_overlap: Int,
        enable_crossfade: Bool = True,
        fp16: Bool = True,
        disable_progress: Bool = False,
        progress_callback: PythonObject = PythonObject(),
        working_directory: PythonObject = PythonObject(),
        start_frame: Optional[Int] = None,
        duration_frames: Optional[Int] = None,
        current_frame_shared: List[Int] = List[Int](),
        pause_requested: List[Bool] = List[Bool](),
        status_file_path: String = "/tmp/jasna_status.json",
    ):
        self.input_video = input_video
        self.output_video = output_video
        self.detection_model = detection_model
        self.restoration_pipeline = restoration_pipeline
        self.codec = codec
        self.encoder_settings = encoder_settings
        self.batch_size = batch_size
        self.device = device
        self.max_clip_size = max_clip_size
        self.temporal_overlap = temporal_overlap
        self.enable_crossfade = enable_crossfade
        self.fp16 = fp16
        self.disable_progress = disable_progress
        self.progress_callback = progress_callback
        self.working_directory = working_directory
        self.start_frame = start_frame
        self.duration_frames = duration_frames
        self.current_frame_shared = current_frame_shared
        self.pause_requested = pause_requested
        self.status_file_path = status_file_path

        if len(self.current_frame_shared) == 0:
            self.current_frame_shared.append(start_frame.value if start_frame is not None else 0)
        if len(self.pause_requested) == 0:
            self.pause_requested.append(False)

    def close(mut self) raises:
        """Close and release model resources."""
        if self.detection_model is not None:
            if Bool(py=_hasattr(self.detection_model, "close")):
                self.detection_model.close()
            self.detection_model = PythonObject()
        self.restoration_pipeline = PythonObject()

    def run(self) raises:
        """Run the full processing pipeline."""
        var threading = Python.import_module("threading")
        var time = Python.import_module("time")
        var torch = Python.import_module("torch")
        var gc = Python.import_module("gc")
        var json = Python.import_module("json")
        var psutil = Python.import_module("psutil")
        var os_mod = Python.import_module("os")

        # Get video metadata
        var metadata = get_video_meta_data(self.input_video)

        # Check colorspace
        var av = Python.import_module("av.video.reformatter")
        if metadata.color_space != av.Colorspace.ITU709:
            raise UnsupportedColorspaceError(
                "Unsupported color space in " + self.input_video + ". Only BT.709 is supported."
            )

        var secondary_workers = max(1, Int(py=self.restoration_pipeline.secondary_num_workers))

        # Create queues
        var clip_queue = FrameQueue(self.max_clip_size)
        var secondary_queue = FrameQueue(self.max_clip_size * secondary_workers)
        var encode_queue = FrameQueue(self.max_clip_size)
        var metadata_queue = Python.evaluate("from queue import Queue; Queue(maxsize=" + String(self.max_clip_size * 5) + ")")()

        # Create shared state
        var error_holder = List[PythonObject]()
        var blend_buffer = BlendBuffer(self.device)
        var crop_buffers = Python.dict()
        var crop_lock = threading.Lock()
        var primary_idle_event = threading.Event()
        var frame_shape = List[Tuple[Int, Int]]()

        var encode_heartbeat = Python.list([time.monotonic()])
        var activity_heartbeat = Python.list([time.monotonic()])

        # Create VRAM offloader
        var vram_offloader = VramOffloader(
            device=self.device,
            blend_buffer=blend_buffer,
            crop_buffers=crop_buffers,
            crop_lock=crop_lock,
        )
        vram_offloader.set_encode_heartbeat(encode_heartbeat)
        vram_offloader.set_pipeline_activity_heartbeat(activity_heartbeat)
        vram_offloader.set_pipeline_queues(
            PythonObject(clip_queue),
            PythonObject(secondary_queue),
            PythonObject(encode_queue),
            metadata_queue,
        )

        # Calculate progress total
        var progress_total = metadata.num_frames
        if self.duration_frames is not None:
            progress_total = self.duration_frames.value
        elif self.start_frame is not None:
            progress_total = max(0, metadata.num_frames - self.start_frame.value)

        var pb = Progressbar(
            total_frames=progress_total,
            video_fps=metadata.video_fps,
            disable=self.disable_progress,
            callback=self.progress_callback,
        )

        # Create encoder
        var encoder_ctx = VideoEncoderFactory.create_encoder(
            self.output_video,
            device=self.device,
            metadata=PythonObject(metadata),
            codec=self.codec,
            encoder_settings=self.encoder_settings,
            stream_mode=False,
            working_directory=self.working_directory,
            start_frame=self.start_frame,
            duration_frames=self.duration_frames,
        )

        # Create frame writer wrapper
        var frame_writer = Python.evaluate("""
class _OfflineFrameWriter:
    def __init__(self, encoder_ctx, heartbeat):
        self.encoder = encoder_ctx
        self.heartbeat = heartbeat
    def write(self, frame, pts):
        self.encoder.write(frame, pts)
        import time
        self.heartbeat[0] = time.monotonic()
    def after_write(self, count):
        pass
    def close(self):
        self.encoder.close()
""", file=True)
        var frame_writer = frame_writer._OfflineFrameWriter(encoder_ctx, encode_heartbeat)

        var seek_ts = (Float64(py=self.start_frame.value) / metadata.video_fps) if self.start_frame is not None else -1.0

        # Create threads
        var threads = Python.evaluate("""
import threading

def _create_threads(decode_fn, primary_fn, secondary_fn, blend_fn):
    return [
        threading.Thread(target=decode_fn, name="DecodeDetect", daemon=True),
        threading.Thread(target=primary_fn, name="PrimaryRestore", daemon=True),
        threading.Thread(target=secondary_fn, name="SecondaryRestore", daemon=True),
        threading.Thread(target=blend_fn, name="BlendEncode", daemon=True),
    ]
""", file=True)


        # Define thread targets as Python callables that call back into Mojo
        # This uses Python's threading with Mojo function callbacks
        var decode_target = Python.evaluate("""
def _make_decode_target(mojo_decode, args):
    def _target():
        mojo_decode(*args)
    return _target
""", file=True)


        # Build thread arguments
        var decode_args = Python.list([
            PythonObject(self.input_video),
            PythonObject(self.batch_size),
            self.device,
            PythonObject(metadata),
            self.detection_model,
            PythonObject(self.max_clip_size),
            PythonObject(self.temporal_overlap),
            PythonObject(self.enable_crossfade),
            PythonObject(blend_buffer),
            crop_buffers,
            PythonObject(clip_queue),
            metadata_queue,
            PythonObject(error_holder),
            PythonObject(frame_shape),
            Python(),  # cancel_event
            PythonObject(seek_ts),
            PythonObject(pb),
            PythonObject(self.start_frame.value) if self.start_frame is not None else Python(),
            PythonObject(self.duration_frames.value) if self.duration_frames is not None else Python(),
            PythonObject(self.current_frame_shared),
            PythonObject(self.pause_requested),
            PythonObject(activity_heartbeat),
        ])

        # For simplicity, we use Python's threading to run the loops
        # The actual loop functions are defined in Python and call Mojo via interop
        var run_pipeline = Python.evaluate("""
def _run_pipeline(input_video, batch_size, device, metadata, detection_model,
                  max_clip_size, temporal_overlap, enable_crossfade,
                  restoration_pipeline, codec, encoder_settings,
                  output_video, blend_buffer, crop_buffers, clip_queue,
                  secondary_queue, encode_queue, metadata_queue,
                  error_holder, frame_shape, primary_idle_event,
                  encode_heartbeat, activity_heartbeat, vram_offloader,
                  frame_writer, seek_ts, pb, start_frame, duration_frames,
                  current_frame_shared, pause_requested, working_directory):
    import threading
    import time
    import torch
    import gc

    from jasna.pipeline_threads import (
        decode_detect_loop, primary_restore_loop,
        secondary_restore_loop, blend_encode_loop,
    )
    from jasna.vram_offloader import VramOffloader

    sentinel = object()

    # Re-create VRAM offloader with Python objects
    vram = VramOffloader(
        device=device,
        blend_buffer=blend_buffer,
        crop_buffers=crop_buffers,
        crop_lock=threading.Lock(),
    )
    vram.set_pipeline_queues(clip_queue, secondary_queue, encode_queue, metadata_queue)
    vram.set_encode_heartbeat(encode_heartbeat)
    vram.set_pipeline_activity_heartbeat(activity_heartbeat)

    threads = [
        threading.Thread(
            target=decode_detect_loop,
            kwargs=dict(
                input_video=input_video, batch_size=batch_size, device=device,
                metadata=metadata, detection_model=detection_model,
                max_clip_size=max_clip_size, temporal_overlap=temporal_overlap,
                enable_crossfade=enable_crossfade, blend_buffer=blend_buffer,
                crop_buffers=crop_buffers, clip_queue=clip_queue,
                metadata_queue=metadata_queue, error_holder=error_holder,
                frame_shape=frame_shape, seek_ts=seek_ts if seek_ts >= 0 else None,
                progress=pb, start_frame=start_frame, duration_frames=duration_frames,
                current_frame_shared=current_frame_shared,
                pause_requested=pause_requested,
                activity_heartbeat=activity_heartbeat,
            ),
            name="DecodeDetect", daemon=True,
        ),
        threading.Thread(
            target=primary_restore_loop,
            kwargs=dict(
                device=device, restoration_pipeline=restoration_pipeline,
                clip_queue=clip_queue, secondary_queue=secondary_queue,
                error_holder=error_holder, primary_idle_event=primary_idle_event,
                activity_heartbeat=activity_heartbeat,
            ),
            name="PrimaryRestore", daemon=True,
        ),
        threading.Thread(
            target=secondary_restore_loop,
            kwargs=dict(
                device=device, restoration_pipeline=restoration_pipeline,
                secondary_queue=secondary_queue, encode_queue=encode_queue,
                error_holder=error_holder,
                activity_heartbeat=activity_heartbeat,
            ),
            name="SecondaryRestore", daemon=True,
        ),
        threading.Thread(
            target=blend_encode_loop,
            kwargs=dict(
                input_video=input_video, batch_size=batch_size, device=device,
                metadata=metadata, blend_buffer=blend_buffer,
                encode_queue=encode_queue, metadata_queue=metadata_queue,
                error_holder=error_holder, frame_writer=frame_writer,
                seek_ts=seek_ts if seek_ts >= 0 else None,
                vram_offloader=vram,
            ),
            name="BlendEncode", daemon=True,
        ),
    ]

    vram.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    vram.stop()
    frame_writer.close()

    if pause_requested[0]:
        import json
        status_data = {
            "input_video": input_video,
            "output_video": output_video,
            "current_frame": int(current_frame_shared[0]),
            "start_frame": start_frame,
            "duration_frames": duration_frames,
            "timestamp": time.time(),
        }
        from pathlib import Path
        status_path = Path("/tmp/jasna_status.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
        print(f"Pause status saved to {status_path} at frame {current_frame_shared[0]}")

    # Log VRAM/RAM usage
    try:
        free, total = torch.cuda.mem_get_info(device)
        vram_used = total - free
        print(f"VRAM usage at end — {vram_used / (1024 ** 2):.1f} MiB")
    except Exception:
        pass
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        print(f"RAM usage at end — {rss / (1024 ** 2):.1f} MiB")
    except Exception:
        pass

    del clip_queue, secondary_queue, encode_queue, metadata_queue
    del blend_buffer, crop_buffers, error_holder, threads
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats(device)

    if error_holder:
        raise error_holder[0]
""", file=True)


        # Convert encoder_settings to Python dict
        var py_encoder_settings = Python.dict()
        for k in self.encoder_settings.keys():
            py_encoder_settings[k] = self.encoder_settings[k]

        var sf = PythonObject() if self.start_frame is None else PythonObject(self.start_frame.value)
        var df = PythonObject() if self.duration_frames is None else PythonObject(self.duration_frames.value)

        run_pipeline._run_pipeline(
            PythonObject(self.input_video),
            PythonObject(self.batch_size),
            self.device,
            PythonObject(metadata),
            self.detection_model,
            PythonObject(self.max_clip_size),
            PythonObject(self.temporal_overlap),
            PythonObject(self.enable_crossfade),
            self.restoration_pipeline,
            PythonObject(self.codec),
            py_encoder_settings,
            PythonObject(self.output_video),
            PythonObject(blend_buffer),
            crop_buffers,
            PythonObject(clip_queue),
            PythonObject(secondary_queue),
            PythonObject(encode_queue),
            metadata_queue,
            PythonObject(error_holder),
            PythonObject(frame_shape),
            primary_idle_event,
            encode_heartbeat,
            activity_heartbeat,
            Python(),  # vram_offloader (created inside)
            frame_writer,
            PythonObject(seek_ts),
            PythonObject(pb),
            sf,
            df,
            PythonObject(self.current_frame_shared),
            PythonObject(self.pause_requested),
            self.working_directory,
        )

    def run_streaming(
        self,
        port: Int = 8765,
        segment_duration: Float64 = 4.0,
        hls_server: PythonObject = PythonObject(),
    ):
        """Run the pipeline in streaming mode (HLS)."""
        var run_streaming_fn = Python.evaluate("""
def _run_streaming(pipeline, port, segment_duration, hls_server):
    from jasna.streaming_pipeline import run_streaming
    run_streaming(pipeline, port=port, segment_duration=segment_duration, hls_server=hls_server)
""", file=True)

        run_streaming_fn._run_streaming(
            PythonObject(self),
            PythonObject(port),
            PythonObject(segment_duration),
            hls_server,
        )
