# Pipeline frame batch processing — processes batches of frames through detection and tracking.
# Orchestrates: decode → detect → track → extract crops → emit clips for restoration.

from std.python import Python, PythonObject
from jasna.pipeline_items import (
    BatchProcessResult,
    ClipRestoreItem,
    FrameMeta,
    RawCrop,
)
from jasna.crop_buffer import CropBuffer, extract_crop
from jasna.tracking.clip_tracker import ClipTracker, EndedClip
from jasna.pipeline_overlap import (
    compute_overlap_and_tail_indices,
    compute_keep_range,
    compute_crossfade_weights,
    compute_parent_crossfade_weights,
    merge_crossfade_weights,
)
from jasna.tensor_utils import pad_batch_with_last
from jasna.frame_queue import FrameQueue


# ============================================================================
# Store raw crop (device-specific handling for MPS)
# ============================================================================

def _store_raw_crop(raw_crop: RawCrop, frame_device: PythonObject) raises -> RawCrop:
    """Handle device-specific crop storage (move MPS crops to CPU)."""
    if String(py=frame_device.type) == "mps" and String(py=raw_crop.crop.device.type) == "mps":
        var cpu_crop = raw_crop.crop.cpu()
        return RawCrop(cpu_crop, raw_crop.enlarged_bbox, raw_crop.crop_h, raw_crop.crop_w)
    return raw_crop


# ============================================================================
# Process ended clips
# ============================================================================

def _process_ended_clips(
    ended_clips: List[EndedClip],
    discard_margin: Int,
    blend_frames: Int,
    max_clip_size: Int,
    blend_buffer: PythonObject,  # BlendBuffer (Python-side)
    crop_buffers: Dict[Int, CropBuffer],
    clip_queue: FrameQueue,
    frame_h: Int,
    frame_w: Int,
):
    """Process clips that have ended — emit them for restoration."""
    var bf = min(blend_frames, discard_margin) if discard_margin > 0 else 0
    if bf > 0 and discard_margin > 0:
        var max_bf = max(0, (max_clip_size - 2 * discard_margin) // 2)
        bf = min(bf, max_bf)

    for ended_clip in ended_clips:
        var clip = ended_clip.clip
        var crop_buf = crop_buffers[clip.track_id]
        crop_buffers.remove(clip.track_id)

        if ended_clip.split_due_to_max_size and discard_margin > 0:
            var child_id = ended_clip.continuation_track_id
            if child_id < 0:
                raise Error("split clip is missing continuation_track_id")

            var overlap_len = 2 * discard_margin
            crop_buffers[child_id] = crop_buf.split_overlap(
                overlap_len, child_id, clip.end_frame - overlap_len + 1,
            )

            var (overlap_indices, tail_indices) = compute_overlap_and_tail_indices(
                end_frame=clip.end_frame, discard_margin=discard_margin,
            )
            blend_buffer.add_pending_clip(overlap_indices, child_id)

            if bf > 0:
                var non_crossfade_tail = List[Int]()
                for i in range(clip.end_frame - discard_margin + 1 + bf, clip.end_frame + 1):
                    non_crossfade_tail.append(i)
                if len(non_crossfade_tail) > 0:
                    blend_buffer.remove_pending_clip(non_crossfade_tail, clip.track_id)
            else:
                blend_buffer.remove_pending_clip(tail_indices, clip.track_id)

        var (keep_start, keep_end) = compute_keep_range(
            frame_count=clip.frame_count,
            is_continuation=clip.is_continuation,
            split_due_to_max_size=ended_clip.split_due_to_max_size,
            discard_margin=discard_margin,
            blend_frames=bf,
        )

        var crossfade_weights = Dict[Int, Float64]()
        if clip.is_continuation and bf > 0 and discard_margin > 0:
            crossfade_weights = compute_crossfade_weights(
                discard_margin=discard_margin, blend_frames=bf,
            )
        if ended_clip.split_due_to_max_size and bf > 0 and discard_margin > 0:
            var parent_weights = compute_parent_crossfade_weights(
                frame_count=clip.frame_count,
                discard_margin=discard_margin,
                blend_frames=bf,
            )
            if len(crossfade_weights) == 0:
                crossfade_weights = parent_weights
            else:
                crossfade_weights = merge_crossfade_weights(crossfade_weights, parent_weights)

        var item = ClipRestoreItem(
            clip=clip,
            raw_crops=crop_buf.crops,
            frame_h=frame_h,
            frame_w=frame_w,
            keep_start=keep_start,
            keep_end=keep_end,
        )
        item.crossfade_weights = crossfade_weights
        clip_queue.put(PythonObject(item), frame_count=keep_end - keep_start)


# ============================================================================
# Process frame batch
# ============================================================================

def process_frame_batch(
    frames: PythonObject,
    pts_list: List[Int],
    start_frame_idx: Int,
    batch_size: Int,
    target_hw: Tuple[Int, Int],
    detections_fn: PythonObject,
    tracker: ClipTracker,
    blend_buffer: PythonObject,
    crop_buffers: Dict[Int, CropBuffer],
    clip_queue: FrameQueue,
    metadata_queue: PythonObject,  # Python queue
    discard_margin: Int,
    blend_frames: Int = 0,
) raises -> BatchProcessResult:
    """Process a batch of frames through detection and tracking.
    
    Args:
        frames: torch.Tensor (B, C, H, W) — decoded frames
        pts_list: List of PTS values for each frame
        start_frame_idx: Index of the first frame in this batch
        batch_size: Model batch size (for padding)
        target_hw: (target_h, target_w) for detection
        detections_fn: Detection model callable
        tracker: Clip tracker
        blend_buffer: Blend buffer
        crop_buffers: Crop buffers dict
        clip_queue: Queue for clips to restore
        metadata_queue: Python queue for frame metadata
        discard_margin: Frames to discard at clip boundaries
        blend_frames: Crossfade blend frames
    Returns:
        BatchProcessResult with next_frame_idx and clips_emitted
    """
    var effective_bs = len(pts_list)
    if effective_bs == 0:
        return BatchProcessResult(next_frame_idx=start_frame_idx, clips_emitted=0)

    var torch = Python.import_module("torch")
    var frames_eff = frames[:effective_bs]
    var frames_in = pad_batch_with_last(frames_eff, batch_size=batch_size)

    # Run detection
    var detections = detections_fn(frames_in, target_hw=target_hw)
    var frame_h = Int(frames_eff[0].shape[1])
    var frame_w = Int(frames_eff[0].shape[2])

    var clips_emitted = 0

    for i in range(effective_bs):
        var current_frame_idx = start_frame_idx + i
        var pts = pts_list[i]
        var frame = frames_eff[i]

        var valid_boxes = detections.boxes_xyxy[i]
        var valid_masks = detections.masks[i]

        var (ended_clips, active_track_ids) = tracker.update(
            current_frame_idx, valid_boxes, valid_masks,
        )

        blend_buffer.register_frame(current_frame_idx, active_track_ids)

        # Put metadata
        var meta = FrameMeta(current_frame_idx, pts)
        metadata_queue.put(PythonObject(meta))

        # Extract crops for active tracks
        for track_id in active_track_ids:
            if tracker.active_clips.contains(track_id):
                var clip = tracker.active_clips[track_id]
                if not crop_buffers.contains(track_id):
                    crop_buffers[track_id] = CropBuffer(track_id, clip.start_frame)
                var raw_crop = _store_raw_crop(
                    extract_crop(frame, clip.bboxes[len(clip.bboxes) - 1], frame_h, frame_w),
                    frame.device,
                )
                crop_buffers[track_id].add(raw_crop)

        # Handle ended clips
        for ec in ended_clips:
            var tid = ec.clip.track_id
            if not crop_buffers.contains(tid):
                crop_buffers[tid] = CropBuffer(tid, ec.clip.start_frame)
            if crop_buffers[tid].frame_count < ec.clip.frame_count:
                var raw_crop = _store_raw_crop(
                    extract_crop(frame, ec.clip.bboxes[len(ec.clip.bboxes) - 1], frame_h, frame_w),
                    frame.device,
                )
                crop_buffers[tid].add(raw_crop)

        clips_emitted += len(ended_clips)

        _process_ended_clips(
            ended_clips,
            discard_margin,
            blend_frames,
            tracker.max_clip_size,
            blend_buffer,
            crop_buffers,
            clip_queue,
            frame_h,
            frame_w,
        )

    return BatchProcessResult(
        next_frame_idx=start_frame_idx + effective_bs,
        clips_emitted=clips_emitted,
    )


# ============================================================================
# Finalize processing — flush remaining clips
# ============================================================================

def finalize_processing(
    tracker: ClipTracker,
    blend_buffer: PythonObject,
    crop_buffers: Dict[Int, CropBuffer],
    clip_queue: FrameQueue,
    frame_h: Int,
    frame_w: Int,
    discard_margin: Int,
    blend_frames: Int,
):
    """Flush remaining active clips at end of processing."""
    var ended_clips = tracker.flush()

    for ended_clip in ended_clips:
        var clip = ended_clip.clip
        if not crop_buffers.contains(clip.track_id):
            continue

        var (keep_start, keep_end) = compute_keep_range(
            frame_count=clip.frame_count,
            is_continuation=clip.is_continuation,
            split_due_to_max_size=False,
            discard_margin=discard_margin,
            blend_frames=blend_frames,
        )

        var crossfade_weights = Dict[Int, Float64]()
        if clip.is_continuation and blend_frames > 0 and discard_margin > 0:
            crossfade_weights = compute_crossfade_weights(
                discard_margin=discard_margin, blend_frames=blend_frames,
            )

        var item = ClipRestoreItem(
            clip=clip,
            raw_crops=crop_buffers[clip.track_id].crops,
            frame_h=frame_h,
            frame_w=frame_w,
            keep_start=keep_start,
            keep_end=keep_end,
        )
        item.crossfade_weights = crossfade_weights
        clip_queue.put(PythonObject(item), frame_count=keep_end - keep_start)
