# Pipeline overlap computation — pure Mojo math, no Python dependencies.
# Computes keep ranges, overlap indices, and crossfade weights for temporal overlap.

from jasna.pipeline_items import BBox


# ============================================================================
# Overlap and tail indices
# ============================================================================

def compute_overlap_and_tail_indices(
    end_frame: Int,
    discard_margin: Int,
) raises -> Tuple[List[Int], List[Int]]:
    """Compute overlap indices and tail indices for clip splitting.
    
    Args:
        end_frame: Last frame index of the clip
        discard_margin: Number of frames to discard at boundaries
    Returns:
        (overlap_indices, tail_indices)
    """
    if discard_margin <= 0:
        return (List[Int](), List[Int]())

    var overlap_len = 2 * discard_margin
    var overlap_start = end_frame - overlap_len + 1

    var overlap_indices = List[Int]()
    for i in range(overlap_start, end_frame + 1):
        overlap_indices.append(i)

    var tail_start = end_frame - discard_margin + 1
    var tail_indices = List[Int]()
    for i in range(tail_start, end_frame + 1):
        tail_indices.append(i)

    return (overlap_indices^, tail_indices^)


# ============================================================================
# Keep range computation
# ============================================================================

def compute_keep_range(
    frame_count: Int,
    is_continuation: Bool,
    split_due_to_max_size: Bool,
    discard_margin: Int,
    blend_frames: Int = 0,
) raises -> Tuple[Int, Int]:
    """Compute the keep range for a clip.
    
    Args:
        frame_count: Total frames in the clip
        is_continuation: True if this clip is a continuation from a split
        split_due_to_max_size: True if clip ended due to max size limit
        discard_margin: Frames to discard at boundaries
        blend_frames: Crossfade blend frames
    Returns:
        (keep_start, keep_end)
    """
    var d = discard_margin
    var bf = min(blend_frames, d) if d > 0 else 0

    var keep_start = (d - bf) if (d > 0 and is_continuation) else 0
    var keep_end = (frame_count - d + bf) if (d > 0 and split_due_to_max_size) else frame_count

    return (keep_start, keep_end)


# ============================================================================
# Crossfade weights
# ============================================================================

def compute_crossfade_weights(
    discard_margin: Int,
    blend_frames: Int,
) raises -> Dict[Int, Float64]:
    """Compute crossfade weights for clip boundary blending.
    
    Args:
        discard_margin: Frames to discard at boundaries
        blend_frames: Number of frames to crossfade
    Returns:
        Dict mapping local frame index to blend weight [0, 1]
    """
    var d = discard_margin
    var bf = min(blend_frames, d) if d > 0 else 0
    if bf <= 0:
        return Dict[Int, Float64]()

    var weights = Dict[Int, Float64]()
    for j in range(2 * bf):
        var local_idx = d - bf + j
        weights[local_idx] = Float64(j + 0.5) / Float64(2 * bf)

    return weights^


def compute_parent_crossfade_weights(
    frame_count: Int,
    discard_margin: Int,
    blend_frames: Int,
) raises -> Dict[Int, Float64]:
    """Compute crossfade weights for the parent side of a split.
    
    Args:
        frame_count: Total frames in the parent clip
        discard_margin: Frames to discard at boundaries
        blend_frames: Number of frames to crossfade
    Returns:
        Dict mapping local frame index to blend weight [0, 1]
    """
    var d = discard_margin
    var bf = min(blend_frames, d) if d > 0 else 0
    if bf <= 0:
        return Dict[Int, Float64]()

    var weights = Dict[Int, Float64]()
    for j in range(2 * bf):
        var local_idx = frame_count - d - bf + j
        weights[local_idx] = 1.0 - Float64(j + 0.5) / Float64(2 * bf)

    return weights^


# ============================================================================
# Helper: merge crossfade weight dicts
# ============================================================================

def merge_crossfade_weights(
    existing: Dict[Int, Float64],
    additions: Dict[Int, Float64],
) raises -> Dict[Int, Float64]:
    """Merge two crossfade weight dictionaries."""
    var result = Dict[Int, Float64]()
    for k in existing.keys():
        result[k] = existing[k]
    for k in additions.keys():
        result[k] = additions[k]
    return result^
