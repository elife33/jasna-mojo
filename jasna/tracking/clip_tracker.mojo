from std.collections import Set, Dict, List
# Clip tracking with IoU-based matching.
# Implements multi-object tracking across frames using greedy IoU matching
# and box merging. The core math is native Mojo; numpy/torch operations
# go through Python interop.

from std.python import Python, PythonObject
from jasna.pipeline_items import TrackedClip, EndedClip


# ============================================================================
# IoU Matrix Computation (native Mojo)
# ============================================================================

def compute_iou_matrix_native(
    boxes1: List[Tuple[Float64, Float64, Float64, Float64]],
    boxes2: List[Tuple[Float64, Float64, Float64, Float64]],
) raises -> List[List[Float64]]:
    """Compute IoU matrix between two sets of boxes (native Mojo).
    
    Args:
        boxes1: List of (x1, y1, x2, y2) boxes
        boxes2: List of (x1, y1, x2, y2) boxes
    Returns:
        N x M matrix of IoU values
    """
    var n = len(boxes1)
    var m = len(boxes2)
    var result = List[List[Float64]]()

    if n == 0 or m == 0:
        for i in range(n):
            result.append(List[Float64]())
        return result

    # Precompute areas
    var area1 = List[Float64]()
    for i in range(n):
        var b = boxes1[i]
        area1.append((b[3] - b[1]) * (b[4] - b[2]))

    var area2 = List[Float64]()
    for j in range(m):
        var b = boxes2[j]
        area2.append((b[3] - b[1]) * (b[4] - b[2]))

    for i in range(n):
        var row = List[Float64]()
        var b1 = boxes1[i]
        for j in range(m):
            var b2 = boxes2[j]

            var inter_x1 = max(b1[1], b2[1])
            var inter_y1 = max(b1[2], b2[2])
            var inter_x2 = min(b1[3], b2[3])
            var inter_y2 = min(b1[4], b2[4])

            var inter_w = max(0.0, inter_x2 - inter_x1)
            var inter_h = max(0.0, inter_y2 - inter_y1)
            var inter_area = inter_w * inter_h

            var union_area = area1[i] + area2[j] - inter_area
            var iou = inter_area / max(union_area, 1e-6)
            row.append(iou)

        result.append(row)

    return result


# ============================================================================
# IoU Matrix via numpy (Python interop for batch operations)
# ============================================================================

def compute_iou_matrix_np(
    boxes1: PythonObject,
    boxes2: PythonObject,
) raises -> PythonObject:
    """Compute IoU matrix using numpy for batch efficiency.
    
    Args:
        boxes1: numpy array (N, 4) xyxy
        boxes2: numpy array (M, 4) xyxy
    Returns:
        numpy array (N, M) IoU matrix
    """
    var np = Python.import_module("numpy")
    var n = Int(boxes1.shape[0])
    var m = Int(boxes2.shape[0])
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    var b1 = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    var b2 = boxes2[np.newaxis, :, :]  # (1, M, 4)

    var inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    var inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    var inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    var inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    var inter_w = np.maximum(inter_x2 - inter_x1, 0)
    var inter_h = np.maximum(inter_y2 - inter_y1, 0)
    var inter_area = inter_w * inter_h

    var area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    var area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    var union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


# ============================================================================
# Merge overlapping boxes (Python interop for numpy/torch)
# ============================================================================

def merge_overlapping_boxes(
    bboxes: PythonObject,
    masks: PythonObject,
    iou_threshold: Float64,
) raises -> Tuple[PythonObject, PythonObject]:
    """Merge boxes that overlap above threshold.
    
    Args:
        bboxes: numpy array (K, 4) xyxy
        masks: torch.Tensor (K, Hm, Wm) bool
        iou_threshold: IoU threshold for merging
    Returns:
        (merged_bboxes, merged_masks)
    """
    var np = Python.import_module("numpy")
    var torch = Python.import_module("torch")
    var n = Int(bboxes.shape[0])
    if n == 0 or n == 1:
        return (bboxes, masks)

    var iou_matrix = compute_iou_matrix_np(bboxes, bboxes)
    var adjacency = iou_matrix > iou_threshold

    var labels = List[Int]()
    for i in range(n):
        labels.append(i)

    # Connected component labeling (simple iterative approach)
    var labels_np = np.arange(n)
    for _ in range(n):
        for i in range(n):
            var neighbors = np.where(adjacency[i])[0]
            if len(neighbors) > 0:
                var min_label = labels_np[neighbors].min()
                if min_label < labels_np[i]:
                    labels_np[i] = min_label

    var unique_labels = np.unique(labels_np)
    var merged_bboxes_list = List[PythonObject]()
    var merged_masks_list = List[PythonObject]()

    for label in unique_labels:
        var group_indices = np.where(labels_np == label)[0]
        var group_boxes = bboxes[group_indices]
        var x1 = group_boxes[:, 0].min()
        var y1 = group_boxes[:, 1].min()
        var x2 = group_boxes[:, 2].max()
        var y2 = group_boxes[:, 3].max()
        merged_bboxes_list.append(np.array([x1, y1, x2, y2]))
        merged_masks_list.append(masks[group_indices].any(dim=0))

    var merged_bboxes = np.stack([b for b in merged_bboxes_list])
    var merged_masks = torch.stack([m for m in merged_masks_list])
    return (merged_bboxes, merged_masks)


# ============================================================================
# Clip Tracker
# ============================================================================

struct ClipTracker:
    """Multi-object clip tracker with temporal overlap support.
    
    Tracks mosaic detections across frames, matching by IoU.
    Splits clips when they reach max_clip_size, with overlap for continuity.
    """

    var max_clip_size: Int
    var temporal_overlap: Int
    var iou_threshold: Float64
    var active_clips: Dict[Int, TrackedClip]
    var next_track_id: Int
    var last_frame_boxes: PythonObject  # numpy (T, 4) or None
    var track_ids: List[Int]
    var _has_last_boxes: Bool

    def __init__(
        mut self,
        max_clip_size: Int,
        temporal_overlap: Int = 0,
        iou_threshold: Float64 = 0.3,
    ):
        self.max_clip_size = max_clip_size
        self.temporal_overlap = temporal_overlap
        self.iou_threshold = iou_threshold
        self.active_clips = Dict[Int, TrackedClip]()
        self.next_track_id = 0
        self.last_frame_boxes = PythonObject()
        self.track_ids = List[Int]()
        self._has_last_boxes = False

        if temporal_overlap < 0:
            raise Error("temporal_overlap must be >= 0")
        if temporal_overlap >= max_clip_size:
            raise Error("temporal_overlap must be < max_clip_size")
        if temporal_overlap > 0 and (2 * temporal_overlap) >= max_clip_size:
            raise Error("temporal_overlap must satisfy 2*temporal_overlap < max_clip_size")

    def update(
        mut self,
        frame_idx: Int,
        bboxes: PythonObject,
        masks: PythonObject,
    ) raises -> Tuple[List[EndedClip], Set[Int]]:
        """Update tracker with new detections.
        
        Args:
            frame_idx: Current frame index
            bboxes: numpy array (K, 4) xyxy
            masks: torch.Tensor (K, Hm, Wm) bool
        Returns:
            (ended_clips, active_track_ids)
        """
        var np = Python.import_module("numpy")
        var n_detections = Int(bboxes.shape[0])

        # Merge overlapping boxes if any
        var eff_bboxes = bboxes
        var eff_masks = masks
        if n_detections > 0:
            (eff_bboxes, eff_masks) = merge_overlapping_boxes(
                bboxes, masks, self.iou_threshold
            )

        var ended_clips = List[EndedClip]()
        var active_track_ids = Set[Int]()

        # No detections — end all active clips
        if Int(eff_bboxes.shape[0]) == 0:
            for track_id in self.track_ids:
                if self.active_clips.contains(track_id):
                    var clip = self.active_clips[track_id]
                    ended_clips.append(EndedClip(clip, False))
                    self.active_clips.remove(track_id)
            self._has_last_boxes = False
            self.track_ids = List[Int]()
            return (ended_clips, active_track_ids)

        var actual_n = Int(eff_bboxes.shape[0])
        var matched_det = List[Bool]()
        for i in range(actual_n):
            matched_det.append(False)

        var matched_track_indices = Set[Int]()
        var det_to_track = Dict[Int, Int]()

        # Match detections to existing tracks via IoU
        if self._has_last_boxes and len(self.track_ids) > 0:
            var iou_matrix = compute_iou_matrix_np(eff_bboxes, self.last_frame_boxes)
            var n_tracks = len(self.track_ids)

            for _ in range(min(actual_n, n_tracks)):
                # Build valid mask
                var valid_mask_np = Python.evaluate("~matched_det[:, None] & ~np.array([i in matched_track_indices for i in range(n_tracks)])")
                # Use numpy for masked argmax
                var masked_iou = np.where(
                    np.logical_and(
                        np.logical_not(matched_det_np(actual_n, matched_det)),
                        np.logical_not(track_matched_np(n_tracks, matched_track_indices))
                    ),
                    iou_matrix,
                    0.0
                )
                var max_iou = float(masked_iou.max())
                if max_iou <= self.iou_threshold:
                    break

                var flat_idx = Int(masked_iou.argmax())
                var det_idx = flat_idx // Int(iou_matrix.shape[1])
                var track_idx = flat_idx % Int(iou_matrix.shape[1])

                matched_det[det_idx] = True
                matched_track_indices.add(track_idx)
                det_to_track[det_idx] = track_idx

        # Update matched tracks
        for det_idx in det_to_track.keys():
            var track_idx = det_to_track[det_idx]
            var track_id = self.track_ids[track_idx]
            if not self.active_clips.contains(track_id):
                continue

            var clip = self.active_clips[track_id]
            clip.bboxes.append(eff_bboxes[det_idx])
            clip.masks.append(eff_masks[det_idx])
            active_track_ids.add(track_id)

            if clip.frame_count >= self.max_clip_size:
                if self.temporal_overlap > 0:
                    var overlap_len = 2 * self.temporal_overlap
                    var new_start_frame = clip.end_frame - overlap_len + 1
                    var new_track_id = self.next_track_id
                    self.next_track_id += 1

                    var new_clip = TrackedClip(
                        new_track_id, new_start_frame,
                        clip.mask_h, clip.mask_w
                    )
                    new_clip.is_continuation = True
                    # Copy overlap bboxes and masks
                    var start_copy = len(clip.bboxes) - overlap_len
                    for i in range(start_copy, len(clip.bboxes)):
                        new_clip.bboxes.append(clip.bboxes[i])
                        new_clip.masks.append(clip.masks[i])

                    self.active_clips[new_track_id] = new_clip
                    active_track_ids.add(new_track_id)
                    active_track_ids.remove(track_id)

                    ended_clips.append(EndedClip(clip, True, new_track_id))
                    self.active_clips.remove(track_id)
                else:
                    ended_clips.append(EndedClip(clip, True))
                    self.active_clips.remove(track_id)

        # End unmatched tracks
        for track_idx in range(len(self.track_ids)):
            if not matched_track_indices.contains(track_idx):
                var track_id = self.track_ids[track_idx]
                if self.active_clips.contains(track_id):
                    var clip = self.active_clips[track_id]
                    ended_clips.append(EndedClip(clip, False))
                    self.active_clips.remove(track_id)

        # Create new tracks for unmatched detections
        for det_idx in range(actual_n):
            if not matched_det[det_idx]:
                var track_id = self.next_track_id
                self.next_track_id += 1
                var clip = TrackedClip(
                    track_id, frame_idx,
                    Int(eff_masks.shape[1]), Int(eff_masks.shape[2])
                )
                clip.bboxes.append(eff_bboxes[det_idx])
                clip.masks.append(eff_masks[det_idx])
                self.active_clips[track_id] = clip
                active_track_ids.add(track_id)

        # Update last_frame_boxes
        var new_boxes = List[PythonObject]()
        var new_track_ids = List[Int]()
        for track_id in active_track_ids:
            if self.active_clips.contains(track_id):
                var clip = self.active_clips[track_id]
                new_boxes.append(clip.bboxes[len(clip.bboxes) - 1])
                new_track_ids.append(track_id)

        if len(new_boxes) > 0:
            self.last_frame_boxes = np.stack(new_boxes)
            self.track_ids = new_track_ids
            self._has_last_boxes = True
        else:
            self._has_last_boxes = False
            self.track_ids = List[Int]()

        return (ended_clips, active_track_ids)

    def flush(mut self) raises -> List[EndedClip]:
        """End all active clips and return them."""
        var clips = List[EndedClip]()
        for track_id in self.active_clips.keys():
            var clip = self.active_clips[track_id]
            clips.append(EndedClip(clip, False))
        self.active_clips.clear()
        self._has_last_boxes = False
        self.track_ids = List[Int]()
        return clips


# ============================================================================
# Helpers for numpy interop
# ============================================================================

def matched_det_np(n: Int, matched: List[Bool]) raises -> PythonObject:
    """Convert matched_det list to numpy array."""
    var np = Python.import_module("numpy")
    var arr = List[Bool]()
    for i in range(n):
        arr.append(matched[i])
    # Convert to numpy array
    var py_list = Python.list([Python.bool(b) for b in arr])
    return np.array(py_list)


def track_matched_np(n: Int, matched: Set[Int]) raises -> PythonObject:
    """Convert matched track indices set to numpy array."""
    var np = Python.import_module("numpy")
    var arr = List[Bool]()
    for i in range(n):
        arr.append(matched.contains(i))
    var py_list = Python.list([Python.bool(b) for b in arr])
    return np.array(py_list)
