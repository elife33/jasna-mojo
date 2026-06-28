# Tests for jasna-mojo core components.
# Run: mojo test jasna-mojo/tests/

from jasna.pipeline_overlap import (
    compute_overlap_and_tail_indices,
    compute_keep_range,
    compute_crossfade_weights,
    compute_parent_crossfade_weights,
)
from jasna.pipeline_items import BBox, FrameMeta


# ============================================================================
# Test: overlap and tail indices
# ============================================================================

fn test_overlap_indices() -> None:
    let (overlap, tail) = compute_overlap_and_tail_indices(
        end_frame=100, discard_margin=8,
    )
    assert_true(len(overlap) == 16)  # 2 * 8 = 16
    assert_equal(overlap[0], 85)  # 100 - 16 + 1 = 85
    assert_equal(overlap[15], 100)

    assert_true(len(tail) == 8)
    assert_equal(tail[0], 93)  # 100 - 8 + 1 = 93
    assert_equal(tail[7], 100)


fn test_overlap_indices_zero_margin() -> None:
    let (overlap, tail) = compute_overlap_and_tail_indices(
        end_frame=50, discard_margin=0,
    )
    assert_true(len(overlap) == 0)
    assert_true(len(tail) == 0)


# ============================================================================
# Test: keep range computation
# ============================================================================

fn test_keep_range_no_overlap() -> None:
    let (ks, ke) = compute_keep_range(
        frame_count=90, is_continuation=False,
        split_due_to_max_size=False, discard_margin=0,
    )
    assert_equal(ks, 0)
    assert_equal(ke, 90)


fn test_keep_range_with_overlap() -> None:
    let (ks, ke) = compute_keep_range(
        frame_count=90, is_continuation=True,
        split_due_to_max_size=True, discard_margin=8,
        blend_frames=2,
    )
    # keep_start = 8 - 2 = 6
    assert_equal(ks, 6)
    # keep_end = 90 - 8 + 2 = 84
    assert_equal(ke, 84)


fn test_keep_range_continuation_only() -> None:
    let (ks, ke) = compute_keep_range(
        frame_count=90, is_continuation=True,
        split_due_to_max_size=False, discard_margin=8,
        blend_frames=0,
    )
    assert_equal(ks, 8)
    assert_equal(ke, 90)


fn test_keep_range_split_only() -> None:
    let (ks, ke) = compute_keep_range(
        frame_count=90, is_continuation=False,
        split_due_to_max_size=True, discard_margin=8,
        blend_frames=0,
    )
    assert_equal(ks, 0)
    assert_equal(ke, 82)


# ============================================================================
# Test: crossfade weights
# ============================================================================

fn test_crossfade_weights() -> None:
    let weights = compute_crossfade_weights(
        discard_margin=8, blend_frames=3,
    )
    # bf = min(3, 8) = 3
    # 2 * bf = 6 entries
    assert_equal(len(weights), 6)
    # First entry should be 0.5/6 = 1/12
    assert_close(weights[5], 0.5 / 6.0, 1e-6)
    # Last entry should be 5.5/6
    assert_close(weights[10], 5.5 / 6.0, 1e-6)


fn test_crossfade_weights_zero() -> None:
    let weights = compute_crossfade_weights(
        discard_margin=0, blend_frames=0,
    )
    assert_equal(len(weights), 0)


# ============================================================================
# Test: parent crossfade weights
# ============================================================================

fn test_parent_crossfade_weights() -> None:
    let weights = compute_parent_crossfade_weights(
        frame_count=90, discard_margin=8, blend_frames=3,
    )
    # bf = 3, 2*bf = 6 entries
    assert_equal(len(weights), 6)
    # First entry: 1.0 - 0.5/6 = 1 - 1/12 = 11/12
    assert_close(weights[79], 1.0 - 0.5 / 6.0, 1e-6)
    # Last entry: 1.0 - 5.5/6 = 0.5/6
    assert_close(weights[84], 1.0 - 5.5 / 6.0, 1e-6)


# ============================================================================
# Test: BBox
# ============================================================================

fn test_bbox() -> None:
    let bbox = BBox(10, 20, 110, 220)
    assert_equal(bbox.width, 100)
    assert_equal(bbox.height, 200)


# ============================================================================
# Test: FrameMeta
# ============================================================================

fn test_frame_meta() -> None:
    let meta = FrameMeta(42, 12345)
    assert_equal(meta.frame_idx, 42)
    assert_equal(meta.pts, 12345)
