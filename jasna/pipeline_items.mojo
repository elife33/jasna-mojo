# Core data structures for the pipeline.
# These are Mojo-native structs that wrap Python objects (torch tensors, numpy arrays)
# via the Python interop layer.

from std.python import Python, PythonObject


# ============================================================================
# Sentinel for queue termination
# ============================================================================

@fieldwise_init
struct Sentinel(Copyable):
    """Marker object used to signal end-of-stream in queues."""
    pass


# ============================================================================
# Frame Metadata
# ============================================================================

@fieldwise_init
struct FrameMeta(Copyable):
    """Metadata for a single decoded frame."""
    var frame_idx: Int
    var pts: Int


# ============================================================================
# Bounding Box (integer pixel coordinates)
# ============================================================================

@fieldwise_init
struct BBox(Copyable):
    """Axis-aligned bounding box in xyxy format."""
    var x1: Int
    var y1: Int
    var x2: Int
    var y2: Int

    def width(self) raises -> Int:
        return self.x2 - self.x1

    def height(self) raises -> Int:
        return self.y2 - self.y1


# ============================================================================
# Raw Crop — extracted region from a frame
# ============================================================================

struct RawCrop:
    """A crop extracted from a frame for restoration."""
    var crop: PythonObject
    var enlarged_bbox: BBox
    var crop_h: Int
    var crop_w: Int

    def __init__(out self, crop: PythonObject, enlarged_bbox: BBox, crop_h: Int, crop_w: Int) raises:
        self.crop = crop
        self.enlarged_bbox = enlarged_bbox
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __init__(out self, *, copy: Self) raises:
        self.crop = copy.crop
        self.enlarged_bbox = copy.enlarged_bbox
        self.crop_h = copy.crop_h
        self.crop_w = copy.crop_w


# ============================================================================
# Tracked Clip — a sequence of detections tracked across frames
# ============================================================================

struct TrackedClip:
    """A tracked clip spanning multiple frames."""
    var track_id: Int
    var start_frame: Int
    var mask_h: Int
    var mask_w: Int
    var bboxes: List[PythonObject]
    var masks: List[PythonObject]
    var is_continuation: Bool

    def __init__(out self, track_id: Int, start_frame: Int, mask_h: Int, mask_w: Int) raises:
        self.track_id = track_id
        self.start_frame = start_frame
        self.mask_h = mask_h
        self.mask_w = mask_w
        self.bboxes = List[PythonObject]()
        self.masks = List[PythonObject]()
        self.is_continuation = False

    def end_frame(self) raises -> Int:
        return self.start_frame + len(self.bboxes) - 1

    def frame_count(self) raises -> Int:
        return len(self.bboxes)

    def frame_indices(self) raises -> List[Int]:
        var indices = List[Int]()
        for i in range(self.frame_count()):
            indices.append(self.start_frame + i)
        return indices


# ============================================================================
# Ended Clip — wrapper for clips that have ended
# ============================================================================

struct EndedClip:
    """Wrapper for ended clips with metadata about why they ended."""
    var clip: TrackedClip
    var split_due_to_max_size: Bool
    var continuation_track_id: Int

    def __init__(out self, clip: TrackedClip, split_due_to_max_size: Bool, continuation_track_id: Int = -1) raises:
        self.clip = clip
        self.split_due_to_max_size = split_due_to_max_size
        self.continuation_track_id = continuation_track_id


# ============================================================================
# Clip Restore Item — item placed on the clip queue for restoration
# ============================================================================

struct ClipRestoreItem:
    """A clip ready for restoration processing."""
    var clip: TrackedClip
    var raw_crops: List[RawCrop]
    var frame_h: Int
    var frame_w: Int
    var keep_start: Int
    var keep_end: Int
    var crossfade_weights: Dict[Int, Float64]

    def __init__(
        out self,
        clip: TrackedClip,
        raw_crops: List[RawCrop],
        frame_h: Int,
        frame_w: Int,
        keep_start: Int,
        keep_end: Int,
    ):
        self.clip = clip
        self.raw_crops = raw_crops
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.keep_start = keep_start
        self.keep_end = keep_end
        self.crossfade_weights = Dict[Int, Float64]()

    def has_crossfade_weights(self) raises -> Bool:
        return len(self.crossfade_weights) > 0


# ============================================================================
# Restore Result Base — common fields for primary/secondary results
# ============================================================================

struct RestoreResultBase:
    """Base fields shared by PrimaryRestoreResult and SecondaryRestoreResult."""
    var track_id: Int
    var start_frame: Int
    var frame_count: Int
    var frame_h: Int
    var frame_w: Int
    var frame_device: PythonObject
    var masks: List[PythonObject]
    var keep_start: Int
    var keep_end: Int
    var crossfade_weights: Dict[Int, Float64]
    var enlarged_bboxes: List[BBox]
    var crop_shapes: List[Tuple[Int, Int]]
    var pad_offsets: List[Tuple[Int, Int]]
    var resize_shapes: List[Tuple[Int, Int]]

    def __init__(
        out self,
        track_id: Int,
        start_frame: Int,
        frame_count: Int,
        frame_h: Int,
        frame_w: Int,
        frame_device: PythonObject,
    ):
        self.track_id = track_id
        self.start_frame = start_frame
        self.frame_count = frame_count
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.frame_device = frame_device
        self.masks = List[PythonObject]()
        self.keep_start = 0
        self.keep_end = 0
        self.crossfade_weights = Dict[Int, Float64]()
        self.enlarged_bboxes = List[BBox]()
        self.crop_shapes = List[Tuple[Int, Int]]()
        self.pad_offsets = List[Tuple[Int, Int]]()
        self.resize_shapes = List[Tuple[Int, Int]]()


# ============================================================================
# Primary Restore Result
# ============================================================================

struct PrimaryRestoreResult:
    """Result from primary restoration (BasicVSR++)."""
    var base: RestoreResultBase
    var primary_raw: PythonObject

    def __init__(out self, base: RestoreResultBase, primary_raw: PythonObject) raises:
        self.base = base
        self.primary_raw = primary_raw


# ============================================================================
# Secondary Restore Result
# ============================================================================

struct SecondaryRestoreResult:
    """Result from secondary restoration (upscaling/denoising)."""
    var base: RestoreResultBase
    var restored_frames: List[PythonObject]
    var clip_keep_offset: Int

    def __init__(
        out self,
        base: RestoreResultBase,
        restored_frames: List[PythonObject],
        clip_keep_offset: Int = 0,
    ):
        self.base = base
        self.restored_frames = restored_frames
        self.clip_keep_offset = clip_keep_offset


# ============================================================================
# Secondary Loop Stats
# ============================================================================

@fieldwise_init
struct SecondaryLoopStats(Copyable):
    """Statistics from the secondary restore loop."""
    var starvation_flushes: Int = 0
    var starvation_seconds: Float64 = 0.0
    var pusher_stall_seconds: Float64 = 0.0
    var clips_pushed: Int = 0
    var clips_popped: Int = 0


# ============================================================================
# Batch Process Result
# ============================================================================

@fieldwise_init
struct BatchProcessResult(Copyable):
    """Result from processing a batch of frames."""
    var next_frame_idx: Int
    var clips_emitted: Int
