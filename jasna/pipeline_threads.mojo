from jasna.py_compat import _hasattr
# Pipeline threads — decode/detect, primary restore, secondary restore, blend/encode loops.
# These run as Python threads (via threading module) for compatibility with Python's GIL
# and torch's threading model. The loop logic is defined in Mojo and executed via Python interop.

from std.python import Python, PythonObject
from jasna.frame_queue import FrameQueue
from jasna.pipeline_items import (
    ClipRestoreItem,
    FrameMeta,
    PrimaryRestoreResult,
    SecondaryRestoreResult,
    Sentinel,
)
from jasna.tracking.clip_tracker import ClipTracker
from jasna.pipeline_processing import (
    process_frame_batch,
    finalize_processing,
)
from jasna.progressbar import Progressbar


# ============================================================================
# Sentinel object (Python-side)
# ============================================================================

def _sentinel() raises -> PythonObject:
    """Get the sentinel object for queue termination."""
    return Python.evaluate("object()")


# ============================================================================
# Decode + Detect Loop
# ============================================================================

def decode_detect_loop(
    input_video: String,
    batch_size: Int,
    device: PythonObject,
    metadata: PythonObject,
    detection_model: PythonObject,
    max_clip_size: Int,
    temporal_overlap: Int,
    enable_crossfade: Bool,
    blend_buffer: PythonObject,
    crop_buffers: Dict[Int, PythonObject],  # Dict of CropBuffer (Python-side)
    clip_queue: FrameQueue,
    metadata_queue: PythonObject,
    error_holder: List[PythonObject],
    frame_shape: List[Tuple[Int, Int]],
    cancel_event: PythonObject = PythonObject(),
    seek_ts: Float64 = -1.0,
    progress: PythonObject = PythonObject(),  # Progressbar or None
    start_frame: Optional[Int] = None,
    duration_frames: Optional[Int] = None,
    current_frame_shared: List[Int] = List[Int](),
    pause_requested: List[Bool] = List[Bool](),
    activity_heartbeat: List[Float64] = List[Float64](),
):
    """Decode frames, run detection, track clips, and emit clips for restoration.
    
    This is the first stage of the pipeline. It reads frames from the input video,
    runs the detection model, updates the clip tracker, extracts crops, and puts
    clips onto the clip_queue for the primary restoration stage.
    """
    var threading = Python.import_module("threading")
    var torch = Python.import_module("torch")
    var time = Python.import_module("time")

    var sentinel = _sentinel()

    try:
        if String(py=device.type) == "cuda":
            torch.cuda.set_device(device)

        var tracker = ClipTracker(max_clip_size, temporal_overlap)
        var discard_margin = temporal_overlap
        var blend_frames = (temporal_overlap // 3) if enable_crossfade else 0

        # Create video reader via Python interop
        var VideoReaderFactory = Python.evaluate("""
def _create_reader(input_video, batch_size, device, metadata):
    from jasna.media.video_reader import VideoReaderFactory
    return VideoReaderFactory.create_reader(input_video, batch_size=batch_size, device=device, metadata=metadata)
""", file=True)


        with torch.inference_mode():
            var reader = VideoReaderFactory._create_reader(input_video, batch_size, device, metadata)

            if progress is not None:
                progress.init()

            var target_hw = (Int(metadata.video_height), Int(metadata.video_width))

            # Determine starting frame
            var frame_idx = 0
            if len(current_frame_shared) > 0:
                frame_idx = current_frame_shared[0]
            elif seek_ts > 0:
                frame_idx = Int(seek_ts * metadata.video_fps)

            var first_batch = seek_ts > 0

            print("Processing " + input_video + ": " + String(metadata.num_frames) +
                  " frames @ " + String(metadata.video_fps) + " fps, " +
                  String(metadata.video_width) + "x" + String(metadata.video_height))

            # Get frame iterator
            var frame_iterator = PythonObject()
            if Bool(py=_hasattr(reader, "frames_with_seek")):
                frame_iterator = reader.frames_with_seek(seek_ts=seek_ts if seek_ts > 0 else None)
            else:
                frame_iterator = reader.frames()

            for batch_tuple in frame_iterator:
                if len(activity_heartbeat) > 0:
                    activity_heartbeat[0] = Float64(time.monotonic())

                if first_batch:
                    pass  # first_batch = False — but Mojo doesn't have mutable closure vars easily

                if cancel_event is not None and Bool(cancel_event.is_set()):
                    break

                var frames = batch_tuple[0]
                var pts_list_py = batch_tuple[1]
                var effective_bs = Int(len(pts_list_py))
                if effective_bs == 0:
                    continue

                # Convert pts to Mojo list
                var pts_list = List[Int]()
                for p in pts_list_py:
                    pts_list.append(Int(p))

                var batch_start_idx = frame_idx
                var batch_end_idx = frame_idx + effective_bs - 1

                # Skip frames before start_frame
                if start_frame is not None and batch_end_idx < start_frame.value:
                    frame_idx += effective_bs
                    continue

                # Check duration limit
                if start_frame is not None and duration_frames is not None:
                    if frame_idx >= start_frame.value + duration_frames.value:
                        break

                if len(frame_shape) == 0:
                    var shape = frames[0].shape
                    frame_shape.append((Int(shape[1]), Int(shape[2])))

                if len(error_holder) > 0:
                    raise error_holder[0]

                # Convert crop_buffers to Python dict for interop
                var py_crop_buffers = Python.dict()
                for k in crop_buffers.keys():
                    py_crop_buffers[k] = crop_buffers[k]

                # Run detection
                var detections = detection_model(frames, target_hw=target_hw)

                # Process each frame in batch
                var result = _process_batch_native(
                    frames=frames,
                    pts_list=pts_list,
                    start_frame_idx=frame_idx,
                    batch_size=batch_size,
                    target_hw=target_hw,
                    detections=detections,
                    tracker=tracker,
                    blend_buffer=blend_buffer,
                    crop_buffers=crop_buffers,
                    clip_queue=clip_queue,
                    metadata_queue=metadata_queue,
                    discard_margin=discard_margin,
                    blend_frames=blend_frames,
                    max_clip_size=max_clip_size,
                )

                if len(activity_heartbeat) > 0:
                    activity_heartbeat[0] = Float64(time.monotonic())

                frame_idx = result.next_frame_idx

                if progress is not None:
                    if start_frame is not None:
                        var effective_start = max(frame_idx - effective_bs, start_frame.value)
                        var effective_end = frame_idx - 1
                        if duration_frames is not None:
                            effective_end = min(effective_end, start_frame.value + duration_frames.value - 1)
                        if effective_end >= effective_start:
                            progress.update(effective_end - effective_start + 1)
                    else:
                        progress.update(effective_bs)

                if len(current_frame_shared) > 0:
                    current_frame_shared[0] = frame_idx

                if len(pause_requested) > 0 and pause_requested[0]:
                    print("Pause requested at frame " + String(frame_idx))
                    break

            # Finalize remaining clips
            if cancel_event is None or not Bool(cancel_event.is_set()):
                var fs = frame_shape[0] if len(frame_shape) > 0 else (Int(metadata.video_height), Int(metadata.video_width))
                finalize_processing(
                    tracker, blend_buffer, crop_buffers, clip_queue,
                    fs[0], fs[1], discard_margin, blend_frames,
                )

    except e:
        if cancel_event is None or not Bool(cancel_event.is_set()):
            print("[decode] thread crashed: " + String(e))
            error_holder.append(e)
    finally:
        clip_queue.put(sentinel)
        metadata_queue.put(sentinel)


# ============================================================================
# Helper: process a batch of frames natively
# ============================================================================

def _process_batch_native(
    frames: PythonObject,
    pts_list: List[Int],
    start_frame_idx: Int,
    batch_size: Int,
    target_hw: Tuple[Int, Int],
    detections: PythonObject,
    tracker: ClipTracker,
    blend_buffer: PythonObject,
    crop_buffers: Dict[Int, PythonObject],
    clip_queue: FrameQueue,
    metadata_queue: PythonObject,
    discard_margin: Int,
    blend_frames: Int,
    max_clip_size: Int,
) raises -> BatchProcessResult:
    """Process a batch of frames through tracking and crop extraction."""
    var effective_bs = len(pts_list)
    if effective_bs == 0:
        return BatchProcessResult(next_frame_idx=start_frame_idx, clips_emitted=0)

    var frame_h = Int(frames[0].shape[1])
    var frame_w = Int(frames[0].shape[2])
    var clips_emitted = 0

    for i in range(effective_bs):
        var current_frame_idx = start_frame_idx + i
        var pts = pts_list[i]
        var frame = frames[i]

        var valid_boxes = detections.boxes_xyxy[i]
        var valid_masks = detections.masks[i]

        var (ended_clips, active_track_ids) = tracker.update(
            current_frame_idx, valid_boxes, valid_masks,
        )

        # Register frame with blend buffer
        var py_track_ids = Python.set([PythonObject(t) for t in active_track_ids])
        blend_buffer.register_frame(current_frame_idx, py_track_ids)

        # Put metadata
        metadata_queue.put(PythonObject(FrameMeta(current_frame_idx, pts)))

        # Extract crops for active tracks
        for track_id in active_track_ids:
            if tracker.active_clips.contains(track_id):
                var clip = tracker.active_clips[track_id]
                if not crop_buffers.contains(track_id):
                    var CropBuffer = Python.evaluate("""
from jasna.crop_buffer import CropBuffer
CropBuffer
""")
                    crop_buffers[track_id] = CropBuffer(track_id=track_id, start_frame=clip.start_frame)

                var extract_fn = Python.evaluate("""
from jasna.crop_buffer import extract_crop
extract_crop
""")
                var raw_crop = extract_fn(frame, clip.bboxes[-1], frame_h, frame_w)
                crop_buffers[track_id].add(raw_crop)

        clips_emitted += len(ended_clips)

        # Process ended clips via Python interop
        var process_ended = Python.evaluate("""
def _process_ended(ended_clips, tracker, discard_margin, blend_frames,
                   max_clip_size, blend_buffer, crop_buffers, clip_queue, frame_h, frame_w):
    from jasna.pipeline_processing import _process_ended_clips
    _process_ended_clips(
        ended_clips=ended_clips,
        discard_margin=discard_margin,
        blend_frames=blend_frames,
        max_clip_size=max_clip_size,
        blend_buffer=blend_buffer,
        crop_buffers=crop_buffers,
        clip_queue=clip_queue,
        frame_shape=(frame_h, frame_w),
    )
""", file=True)

        # Convert ended_clips to Python objects
        var py_ended = Python.list([PythonObject(ec) for ec in ended_clips])
        process_ended._process_ended(
            py_ended, tracker, discard_margin, blend_frames,
            max_clip_size, blend_buffer, crop_buffers, clip_queue,
            frame_h, frame_w,
        )

    return BatchProcessResult(
        next_frame_idx=start_frame_idx + effective_bs,
        clips_emitted=clips_emitted,
    )


# ============================================================================
# Primary Restore Loop
# ============================================================================

def primary_restore_loop(
    device: PythonObject,
    restoration_pipeline: PythonObject,  # RestorationPipeline (Mojo or Python)
    clip_queue: FrameQueue,
    secondary_queue: FrameQueue,
    error_holder: List[PythonObject],
    primary_idle_event: PythonObject,
    cancel_event: PythonObject = PythonObject(),
    activity_heartbeat: List[Float64] = List[Float64](),
):
    """Primary restoration loop — takes clips from clip_queue, runs BasicVSR++,
    puts results on secondary_queue.
    """
    var torch = Python.import_module("torch")
    var time = Python.import_module("time")
    var sentinel = _sentinel()

    try:
        if String(py=device.type) == "cuda":
            torch.cuda.set_device(device)

        while True:
            if cancel_event is not None and Bool(cancel_event.is_set()):
                break

            primary_idle_event.set()

            var item = PythonObject()
            if cancel_event is not None:
                try:
                    item = clip_queue.get(timeout=0.1)
                except:
                    continue
            else:
                item = clip_queue.get()

            primary_idle_event.clear()

            if item is sentinel:
                break

            if len(activity_heartbeat) > 0:
                activity_heartbeat[0] = Float64(time.monotonic())

            # Run primary restoration via Python interop
            var result = Python.evaluate("""
def _run_primary(restoration_pipeline, item):
    return restoration_pipeline.prepare_and_run_primary(
        item.clip,
        item.raw_crops,
        item.frame_shape,
        item.keep_start,
        item.keep_end,
        item.crossfade_weights,
    )
""", file=True)
            result = result._run_primary(restoration_pipeline, item)

            if len(activity_heartbeat) > 0:
                activity_heartbeat[0] = Float64(time.monotonic())

            # Check if secondary prefers CPU input
            var prefers_cpu = Python.evaluate("""
def _check_cpu(restoration_pipeline):
    return bool(getattr(restoration_pipeline, 'secondary_prefers_cpu_input', False))
""", file=True)
            prefers_cpu = prefers_cpu._check_cpu(restoration_pipeline)

            if prefers_cpu:
                result.primary_raw = result.primary_raw.cpu()

            secondary_queue.put(result, frame_count=Int(result.keep_end - result.keep_start))

            # Release device cache
            if String(py=device.type) == "cuda":
                torch.cuda.empty_cache()

    except e:
        if cancel_event is None or not Bool(cancel_event.is_set()):
            print("[primary] thread crashed: " + String(e))
            error_holder.append(e)
    finally:
        secondary_queue.put(sentinel)


# ============================================================================
# Secondary Restore Loop
# ============================================================================

def secondary_restore_loop(
    device: PythonObject,
    restoration_pipeline: PythonObject,
    secondary_queue: FrameQueue,
    encode_queue: FrameQueue,
    error_holder: List[PythonObject],
    cancel_event: PythonObject = PythonObject(),
    activity_heartbeat: List[Float64] = List[Float64](),
):
    """Secondary restoration loop — takes primary results, runs secondary restorer,
    puts final results on encode_queue.
    """
    var torch = Python.import_module("torch")
    var time = Python.import_module("time")
    var sentinel = _sentinel()

    try:
        if String(py=device.type) == "cuda":
            torch.cuda.set_device(device)

        while True:
            if cancel_event is not None and Bool(cancel_event.is_set()):
                break

            var item = PythonObject()
            if cancel_event is not None:
                try:
                    item = secondary_queue.get(timeout=0.1)
                except:
                    continue
            else:
                item = secondary_queue.get()

            if item is sentinel:
                break

            if len(activity_heartbeat) > 0:
                activity_heartbeat[0] = Float64(time.monotonic())

            # Run secondary restoration
            var restored_frames = Python.evaluate("""
def _run_secondary(restoration_pipeline, pr):
    return restoration_pipeline._run_secondary(
        pr.primary_raw, pr.keep_start, pr.keep_end
    )
""", file=True)
            restored_frames = restored_frames._run_secondary(restoration_pipeline, item)

            if len(activity_heartbeat) > 0:
                activity_heartbeat[0] = Float64(time.monotonic())

            # Build secondary result
            var sr = Python.evaluate("""
def _build_sr(restoration_pipeline, pr, restored_frames):
    return restoration_pipeline.build_secondary_result(pr, restored_frames)
""", file=True)
            sr = sr._build_sr(restoration_pipeline, item, restored_frames)

            encode_queue.put(sr, frame_count=Int(sr.keep_end))

            if String(py=device.type) == "cuda":
                torch.cuda.empty_cache()

    except e:
        if cancel_event is None or not Bool(cancel_event.is_set()):
            print("[secondary] thread crashed: " + String(e))
            error_holder.append(e)
    finally:
        encode_queue.put(sentinel)


# ============================================================================
# Blend + Encode Loop
# ============================================================================

def blend_encode_loop(
    input_video: String,
    batch_size: Int,
    device: PythonObject,
    metadata: PythonObject,
    blend_buffer: PythonObject,
    encode_queue: FrameQueue,
    metadata_queue: PythonObject,
    error_holder: List[PythonObject],
    frame_writer: PythonObject,
    cancel_event: PythonObject = PythonObject(),
    seek_ts: Float64 = -1.0,
    vram_offloader: PythonObject = PythonObject(),
):
    """Blend restored crops into original frames and encode the output.
    
    This is the final stage. It reads original frames, blends in the restored
    crops from the encode_queue, and writes the result to the output video.
    """
    var torch = Python.import_module("torch")
    var sentinel = _sentinel()

    try:
        if String(py=device.type) == "cuda":
            torch.cuda.set_device(device)

        var reader_device = torch.device("cpu") if String(py=device.type) == "mps" else device

        # Create second reader for original frames
        var VideoReaderFactory = Python.evaluate("""
def _create_reader(input_video, batch_size, device, metadata):
    from jasna.media.video_reader import VideoReaderFactory
    return VideoReaderFactory.create_reader(input_video, batch_size=batch_size, device=device, metadata=metadata)
""", file=True)


        with VideoReaderFactory._create_reader(input_video, batch_size, reader_device, metadata) as reader2:
            # Create frame generator
            var frame_iterator = PythonObject()
            if Bool(py=_hasattr(reader2, "frames_with_seek")):
                frame_iterator = reader2.frames_with_seek(seek_ts=seek_ts if seek_ts > 0 else None)
            else:
                frame_iterator = reader2.frames()

            # Flatten frame generator
            var frame_gen = Python.evaluate("""
def _flat_frames(frame_iterator):
    for batch, pts in frame_iterator:
        for i in range(len(pts)):
            yield batch[i]
""", file=True)
            frame_gen = frame_gen._flat_frames(frame_iterator)

            var secondary_done = False
            var frames_encoded = 0

            while True:
                if cancel_event is not None and Bool(cancel_event.is_set()):
                    break

                # Drain encode queue
                while not secondary_done:
                    try:
                        var sr_item = encode_queue.get_nowait()
                        if sr_item is sentinel:
                            secondary_done = True
                        else:
                            blend_buffer.add_result(sr_item)
                    except:
                        break

                # Get metadata
                var meta_item = PythonObject()
                try:
                    meta_item = metadata_queue.get(timeout=0.1 if cancel_event is not None else 0.05)
                except:
                    continue

                if meta_item is sentinel:
                    break

                var frame_idx = Int(meta_item.frame_idx)
                var pts = Int(meta_item.pts)
                var original_frame = next(frame_gen)

                # Wait for frame to be ready
                while not Bool(blend_buffer.is_frame_ready(frame_idx)):
                    if cancel_event is not None and Bool(cancel_event.is_set()):
                        break
                    if len(error_holder) > 0:
                        raise error_holder[0]
                    if secondary_done:
                        print("[blend-encode] frame " + String(frame_idx) + " not ready but secondary is done")
                        break
                    try:
                        var sr_item = encode_queue.get(timeout=0.1)
                        if sr_item is sentinel:
                            secondary_done = True
                            continue
                        blend_buffer.add_result(sr_item)
                    except:
                        pass

                var blended = blend_buffer.blend_frame(frame_idx, original_frame)
                frame_writer.write(blended, pts)
                frames_encoded += 1
                frame_writer.after_write(frames_encoded)

                # Memory pressure relief for MPS
                if String(py=reader_device.type) == "cpu" and String(py=device.type) == "mps" and frames_encoded % 256 == 0:
                    var relieve = Python.evaluate("""
def _relieve():
    import gc
    gc.collect()
    import torch
    if _hasattr(torch, 'mps'):
        torch.mps.empty_cache()
""", file=True)

                    relieve._relieve()

            if vram_offloader is not None:
                vram_offloader.pause_stall_check()

    except e:
        if cancel_event is None or not Bool(cancel_event.is_set()):
            print("[blend-encode] thread crashed: " + String(e))
            error_holder.append(e)
