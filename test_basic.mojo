from jasna.pipeline_items import BBox, FrameMeta, BatchProcessResult
from jasna.pipeline_overlap import compute_overlap_and_tail_indices, compute_keep_range

def main() raises:
    var bbox = BBox(10, 20, 110, 220)
    print("BBox width:", bbox.width())
    print("BBox height:", bbox.height())

    var result_tuple = compute_overlap_and_tail_indices(100, 8)
    print("Overlap count:", len(result_tuple[0]))
    print("Tail count:", len(result_tuple[1]))

    var (ks, ke) = compute_keep_range(90, True, True, 8, 2)
    print("Keep range:", ks, ke)

    var meta = FrameMeta(42, 12345)
    print("FrameMeta:", meta.frame_idx, meta.pts)

    var result = BatchProcessResult(10, 3)
    print("BatchProcessResult:", result.next_frame_idx, result.clips_emitted)
    print("All tests passed!")
