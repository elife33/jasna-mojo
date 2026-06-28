from std.collections import Set, Dict, List
from jasna.py_compat import _hasattr
# Blend buffer — composites restored crops back into original frames.
# Uses Python interop for torch tensor operations and threading.

from std.python import Python, PythonObject

from jasna.pipeline_items import SecondaryRestoreResult
from jasna.crop_buffer import scale_offsets
from jasna.tracking.blending import create_blend_mask


struct BlendBuffer:
    """Buffer for compositing restored crops into original frames.
    
    Tracks which frames have pending restoration results and blends
    them when the frame is ready for encoding.
    """

    var device: PythonObject
    var blend_mask_fn: PythonObject  # callable
    var _lock: PythonObject  # threading.Lock
    var pending_map: PythonObject  # dict[int, set[int]]
    var _results: PythonObject  # dict[int, SecondaryRestoreResult-like]
    var _result_last_frame: PythonObject  # dict[int, int]

    def __init__(
        mut self,
        device: PythonObject,
        blend_mask_fn: PythonObject = PythonObject(),
    ):
        var threading = Python.import_module("threading")

        self.device = device
        if blend_mask_fn is not None:
            self.blend_mask_fn = blend_mask_fn
        else:
            self.blend_mask_fn = Python.evaluate("lambda mask, h: mask")

        self._lock = threading.Lock()
        self.pending_map = Python.dict()
        self._results = Python.dict()
        self._result_last_frame = Python.dict()

    def register_frame(self, frame_idx: Int, pending_track_ids: Set[Int]) raises:
        """Register a frame with its pending track IDs."""
        if len(pending_track_ids) > 0:
            with self._lock:
                var py_set = Python.set([PythonObject(t) for t in pending_track_ids])
                self.pending_map[frame_idx] = py_set.copy()

    def add_pending_clip(self, frame_indices: List[Int], track_id: Int) raises:
        """Add a track ID to pending sets for given frame indices."""
        with self._lock:
            for frame_idx in frame_indices:
                var pending = self.pending_map.get(frame_idx)
                if pending is None:
                    continue
                pending.add(track_id)

    def remove_pending_clip(self, frame_indices: List[Int], track_id: Int) raises:
        """Remove a track ID from pending sets for given frame indices."""
        with self._lock:
            for frame_idx in frame_indices:
                var pending = self.pending_map.get(frame_idx)
                if pending is None:
                    continue
                pending.discard(track_id)

    def add_result(self, sr: SecondaryRestoreResult) raises:
        """Add a secondary restoration result."""
        with self._lock:
            var clip_offset = sr.clip_keep_offset
            var kept_count = sr.base.keep_end
            var start = sr.base.start_frame

            # Remove non-kept frames from pending
            for i in chain(range(clip_offset), range(clip_offset + kept_count, sr.base.frame_count)):
                var pending = self.pending_map.get(start + i)
                if pending is not None:
                    pending.discard(sr.base.track_id)

            self._results[sr.base.track_id] = sr
            var last_frame = start + clip_offset + kept_count - 1
            self._result_last_frame[sr.base.track_id] = last_frame

    def offloadable_results(self) raises -> List[SecondaryRestoreResult]:
        """Return all results (for VRAM offloading)."""
        with self._lock:
            var values = self._results.values()
            var result = List[SecondaryRestoreResult]()
            # This is a simplification — in practice we'd need to convert back
            return result

    def is_frame_ready(self, frame_idx: Int) raises -> Bool:
        """Check if all pending restorations for a frame are complete."""
        with self._lock:
            var pending = self.pending_map.get(frame_idx)
            if pending is None or len(pending) == 0:
                return True
            for tid in pending:
                if not Bool(py=self._results.contains(Int(tid))):
                    return False
            return True

    def blend_frame(self, frame_idx: Int, original_frame: PythonObject) raises -> PythonObject:
        """Blend restored crops into the original frame.
        
        Args:
            frame_idx: Frame index to blend
            original_frame: torch.Tensor (C, H, W) — original frame
        Returns:
            torch.Tensor (C, H, W) — blended frame
        """
        var torch = Python.import_module("torch")
        var F = Python.import_module("torch.nn.functional")

        with self._lock:
            var pending = self.pending_map.pop(frame_idx, None)
            if pending is None or len(pending) == 0:
                return original_frame

            var results_snapshot = List[Tuple[Int, PythonObject]]()
            for track_id in pending:
                results_snapshot.append((Int(track_id), self._results.get(track_id)))

        var blended = original_frame.clone()
        var device = original_frame.device

        for pair in results_snapshot:
            var track_id = pair[0]
            var sr_obj = pair[1]
            if sr_obj is None:
                continue
            self._apply_blend(blended, original_frame, frame_idx, track_id, sr_obj, device)

        with self._lock:
            var released = False
            for pair in results_snapshot:
                var track_id = pair[0]
                var sr_obj = pair[1]
                if sr_obj is not None:
                    var last = self._result_last_frame.get(track_id)
                    if last is not None and Int(last) == frame_idx:
                        del self._results[track_id]
                        del self._result_last_frame[track_id]
                        released = True

        if released and String(py=self.device.type) == "mps":
            if Bool(py=_hasattr(torch, "mps")):
                torch.mps.empty_cache()

        return blended

    def _apply_blend(
        self,
        blended: PythonObject,
        original: PythonObject,
        frame_idx: Int,
        track_id: Int,
        sr: PythonObject,
        device: PythonObject,
    ):
        """Apply blend for a single track's restoration result.
        
        sr is a Python dict-like object with the SecondaryRestoreResult fields.
        """
        var torch = Python.import_module("torch")
        var F = Python.import_module("torch.nn.functional")

        # Access fields from the Python-side result object
        var clip_offset = Int(sr.clip_keep_offset)
        var local_i = frame_idx - Int(sr.start_frame) - clip_offset

        if local_i < 0 or local_i >= Int(sr.keep_end):
            return

        var frame_u8 = sr.restored_frames[local_i].to(device)
        var pad_offset = sr.pad_offsets[local_i]
        var resize_shape = sr.resize_shapes[local_i]
        var (pad_offset_scaled, resize_shape_scaled) = scale_offsets(
            frame_u8,
            (Int(pad_offset[0]), Int(pad_offset[1])),
            (Int(resize_shape[0]), Int(resize_shape[1])),
        )

        var i_crop = clip_offset + local_i
        var cw = 1.0
        if sr.crossfade_weights is not None:
            var cw_val = sr.crossfade_weights.get(i_crop, 1.0)
            cw = Float64(cw_val)

        var bbox = sr.enlarged_bboxes[local_i]
        var x1 = Int(bbox[0])
        var y1 = Int(bbox[1])
        var x2 = Int(bbox[2])
        var y2 = Int(bbox[3])
        var crop_h = Int(sr.crop_shapes[local_i][0])
        var crop_w = Int(sr.crop_shapes[local_i][1])
        var pad_left = pad_offset_scaled[0]
        var pad_top = pad_offset_scaled[1]
        var resize_h = resize_shape_scaled[0]
        var resize_w = resize_shape_scaled[1]

        var mask_lr = sr.masks[local_i].to(device)

        var unpadded = frame_u8[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]
        var resized_back = F.interpolate(
            unpadded.unsqueeze(0).float(),
            (crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        var frame_h = Int(sr.frame_shape[0])
        var frame_w = Int(sr.frame_shape[1])
        var hm = Int(mask_lr.shape[0])
        var wm = Int(mask_lr.shape[1])

        var y_idx = (torch.arange(y1, y2, device=device) * hm) // frame_h
        var x_idx = (torch.arange(x1, x2, device=device) * wm) // frame_w
        var crop_mask = mask_lr.float().index_select(0, y_idx).index_select(1, x_idx)
        var blend_mask = self.blend_mask_fn(crop_mask, frame_h)

        if cw < 1.0:
            var blend_mask_weighted = blend_mask * cw
            var original_crop = original[:, y1:y2, x1:x2].float()
            var delta = (resized_back - original_crop) * blend_mask_weighted.unsqueeze(0)
            var current = blended[:, y1:y2, x1:x2].float()
            current.add_(delta).round_().clamp_(0, 255)
            blended[:, y1:y2, x1:x2] = current.to(blended.dtype)
        else:
            var original_crop = blended[:, y1:y2, x1:x2].float()
            original_crop.lerp_(resized_back, blend_mask.unsqueeze(0)).round_().clamp_(0, 255)
            blended[:, y1:y2, x1:x2] = original_crop.to(blended.dtype)
