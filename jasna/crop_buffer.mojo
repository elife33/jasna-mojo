# Crop buffer — bbox expansion, crop extraction, and preparation for restoration.
# The bbox expansion math is native Mojo; tensor operations use Python interop.

from std.python import Python, PythonObject

from jasna.pipeline_items import RawCrop, BBox


# Constants
comptime RESTORATION_SIZE = 256
comptime BORDER_RATIO = 0.06
comptime MIN_BORDER = 20
comptime MAX_EXPANSION_FACTOR = 1.0


# ============================================================================
# Bounding box expansion (native Mojo math)
# ============================================================================

def expand_bbox(
    x1: Int, y1: Int, x2: Int, y2: Int,
    frame_h: Int, frame_w: Int,
) raises -> BBox:
    """Expand a bounding box to include border and approach restoration size.
    
    The expansion adds a border proportional to the box size, then expands
    further to approach the RESTORATION_SIZE while respecting frame boundaries
    and a maximum expansion factor.
    
    Args:
        x1, y1, x2, y2: Original bounding box
        frame_h, frame_w: Frame dimensions
    Returns:
        Expanded BBox
    """
    var w = x2 - x1
    var h = y2 - y1

    var border = max(MIN_BORDER, Int(Float64(max(w, h)) * BORDER_RATIO)) if BORDER_RATIO > 0.0 else 0
    var x1_exp = max(0, x1 - border)
    var y1_exp = max(0, y1 - border)
    var x2_exp = min(frame_w, x2 + border)
    var y2_exp = min(frame_h, y2 + border)

    w = x2_exp - x1_exp
    h = y2_exp - y1_exp

    var down_scale_factor = min(
        Float64(RESTORATION_SIZE) / Float64(w),
        Float64(RESTORATION_SIZE) / Float64(h),
    ) if w > 0 and h > 0 else 1.0
    var dsf = min(down_scale_factor, 1.0)

    var missing_w = Int((Float64(RESTORATION_SIZE) - (Float64(w) * dsf)) / dsf) if dsf > 0 else 0
    var missing_h = Int((Float64(RESTORATION_SIZE) - (Float64(h) * dsf)) / dsf) if dsf > 0 else 0

    var available_w_l = x1_exp
    var available_w_r = frame_w - x2_exp
    var available_h_t = y1_exp
    var available_h_b = frame_h - y2_exp

    var budget_w = Int(MAX_EXPANSION_FACTOR * Float64(w))
    var budget_h = Int(MAX_EXPANSION_FACTOR * Float64(h))

    var expand_w_lr = min(min(available_w_l, available_w_r), min(missing_w // 2, budget_w))
    var expand_w_l = min(available_w_l - expand_w_lr, min(missing_w - expand_w_lr * 2, budget_w - expand_w_lr))
    var expand_w_r = min(
        available_w_r - expand_w_lr,
        min(missing_w - expand_w_lr * 2 - expand_w_l, budget_w - expand_w_lr - expand_w_l),
    )

    var expand_h_tb = min(min(available_h_t, available_h_b), min(missing_h // 2, budget_h))
    var expand_h_t = min(available_h_t - expand_h_tb, min(missing_h - expand_h_tb * 2, budget_h - expand_h_tb))
    var expand_h_b = min(
        available_h_b - expand_h_tb,
        min(missing_h - expand_h_tb * 2 - expand_h_t, budget_h - expand_h_tb - expand_h_t),
    )

    x1_exp = x1_exp - Int(floor(Float64(expand_w_lr) / 2.0)) - expand_w_l
    x2_exp = x2_exp + Int(ceil(Float64(expand_w_lr) / 2.0)) + expand_w_r
    y1_exp = y1_exp - Int(floor(Float64(expand_h_tb) / 2.0)) - expand_h_t
    y2_exp = y2_exp + Int(ceil(Float64(expand_h_tb) / 2.0)) + expand_h_b

    x1_exp = max(0, min(x1_exp, frame_w))
    x2_exp = max(0, min(x2_exp, frame_w))
    y1_exp = max(0, min(y1_exp, frame_h))
    y2_exp = max(0, min(y2_exp, frame_h))

    return BBox(x1_exp, y1_exp, x2_exp, y2_exp)


# ============================================================================
# Scale offsets — compute pad offsets and resize shapes after restoration
# ============================================================================

def scale_offsets(
    frame_u8: PythonObject,
    pad_offset_256: Tuple[Int, Int],
    resize_shape_256: Tuple[Int, Int],
    restoration_size: Int = RESTORATION_SIZE,
) raises -> Tuple[Tuple[Int, Int], Tuple[Int, Int]]:
    """Compute scaled offsets after restoration.
    
    Args:
        frame_u8: torch.Tensor (C, H, W) — restored frame
        pad_offset_256: (pad_left, pad_top) at 256 resolution
        resize_shape_256: (resize_h, resize_w) at 256 resolution
        restoration_size: Size of restoration input (default 256)
    Returns:
        ((x0, y0), (out_h, out_w)) — offset and shape at output resolution
    """
    var out_h = Int(frame_u8.shape[1])
    var out_w = Int(frame_u8.shape[2])
    var pl = pad_offset_256[0]
    var pt = pad_offset_256[1]
    var rh = resize_shape_256[0]
    var rw = resize_shape_256[1]

    var x0 = Int(round(Float64(pl) * Float64(out_w) / Float64(restoration_size)))
    var x1 = Int(round(Float64(pl + rw) * Float64(out_w) / Float64(restoration_size)))
    var y0 = Int(round(Float64(pt) * Float64(out_h) / Float64(restoration_size)))
    var y1 = Int(round(Float64(pt + rh) * Float64(out_h) / Float64(restoration_size)))

    return ((x0, y0), (y1 - y0, x1 - x0))


# ============================================================================
# Extract crop from frame
# ============================================================================

def extract_crop(
    frame: PythonObject,
    bbox: PythonObject,
    frame_h: Int,
    frame_w: Int,
) raises -> RawCrop:
    """Extract a crop from a frame using expanded bounding box.
    
    Args:
        frame: torch.Tensor (C, H, W) — frame
        bbox: numpy array (4,) xyxy — detection bounding box
        frame_h, frame_w: Frame dimensions
    Returns:
        RawCrop with the extracted region
    """
    var np = Python.import_module("numpy")
    var torch = Python.import_module("torch")

    var x1 = Int(np.floor(bbox[0]))
    var y1 = Int(np.floor(bbox[1]))
    var x2 = Int(np.ceil(bbox[2]))
    var y2 = Int(np.ceil(bbox[3]))

    var cx1 = max(0, min(x1, frame_w))
    var cy1 = max(0, min(y1, frame_h))
    var cx2 = max(0, min(x2, frame_w))
    var cy2 = max(0, min(y2, frame_h))

    var expanded = expand_bbox(cx1, cy1, cx2, cy2, frame_h, frame_w)

    var crop = PythonObject()
    if String(frame.device) == "cpu":
        crop = torch.from_numpy(np.array(frame.numpy()[:, expanded.y1:expanded.y2, expanded.x1:expanded.x2]))
    else:
        crop = frame[:, expanded.y1:expanded.y2, expanded.x1:expanded.x2].clone()

    var crop_h = Int(crop.shape[1])
    var crop_w = Int(crop.shape[2])

    return RawCrop(crop, expanded, crop_h, crop_w)


# ============================================================================
# Crop Buffer — stores crops for a tracked clip
# ============================================================================

struct CropBuffer:
    """Buffer of crops for a single tracked clip."""

    var track_id: Int
    var start_frame: Int
    var crops: List[RawCrop]

    def __init__(out self, track_id: Int, start_frame: Int) raises:
        self.track_id = track_id
        self.start_frame = start_frame
        self.crops = List[RawCrop]()

    def add(mut self, crop: RawCrop) raises:
        self.crops.append(crop)

    def frame_count(self) raises -> Int:
        return len(self.crops)

    def split_overlap(
        mut self,
        overlap_len: Int,
        new_track_id: Int,
        new_start_frame: Int,
    ) raises -> CropBuffer:
        """Split the buffer, keeping the last overlap_len crops for the new buffer."""
        var new_buf = CropBuffer(new_track_id, new_start_frame)
        var start = len(self.crops) - overlap_len
        for i in range(start, len(self.crops)):
            new_buf.crops.append(self.crops[i])
        return new_buf


# ============================================================================
# Prepare crops for restoration
# ============================================================================

def prepare_crops_for_restoration(
    raw_crops: List[RawCrop],
    device: PythonObject,
    restoration_size: Int = RESTORATION_SIZE,
) raises -> Tuple[List[PythonObject], List[Tuple[Int, Int]], List[Tuple[Int, Int]]]:
    """Prepare crops for restoration by resizing and padding to restoration_size.
    
    Args:
        raw_crops: List of RawCrop
        device: torch.device for processing
        restoration_size: Target size (default 256)
    Returns:
        (resized_crops, pad_offsets, resize_shapes)
        - resized_crops: list of (256, 256, C) tensors
        - pad_offsets: list of (pad_left, pad_top)
        - resize_shapes: list of (resize_h, resize_w)
    """
    var torch = Python.import_module("torch")
    var F = Python.import_module("torch.nn.functional")
    var np = Python.import_module("numpy")

    # Find max dimensions
    var max_h = 0
    var max_w = 0
    for crop in raw_crops:
        if crop.crop_h > max_h:
            max_h = crop.crop_h
        if crop.crop_w > max_w:
            max_w = crop.crop_w

    var scale_h = Float64(restoration_size) / Float64(max_h)
    var scale_w = Float64(restoration_size) / Float64(max_w)
    var sh = scale_h
    var sw = scale_w
    if sh > 1.0 and sw > 1.0:
        sh = 1.0
        sw = 1.0

    var resized_crops = List[PythonObject]()
    var resize_shapes = List[Tuple[Int, Int]]()
    var pad_offsets = List[Tuple[Int, Int]]()

    for raw_crop in raw_crops:
        var crop = raw_crop.crop
        if String(crop.device) != String(device):
            crop = crop.to(device, non_blocking=True)

        var new_h = Int(Float64(raw_crop.crop_h) * sh)
        var new_w = Int(Float64(raw_crop.crop_w) * sw)
        resize_shapes.append((new_h, new_w))

        var pad_top = (restoration_size - new_h) // 2
        var pad_left = (restoration_size - new_w) // 2
        var pad_bottom = restoration_size - new_h - pad_top
        var pad_right = restoration_size - new_w - pad_left
        pad_offsets.append((pad_left, pad_top))

        var resized = F.interpolate(
            crop.unsqueeze(0).float(),
            (new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        var padded = _torch_pad_reflect(resized, (pad_left, pad_right, pad_top, pad_bottom))
        resized_crops.append(padded.to(raw_crop.crop.dtype).permute(1, 2, 0))

    return (resized_crops, pad_offsets, resize_shapes)


# ============================================================================
# Reflect padding helper
# ============================================================================

def _torch_pad_reflect(
    image: PythonObject,
    paddings: Tuple[Int, Int, Int, Int],
) raises -> PythonObject:
    """Apply reflect padding iteratively (handles edge cases where padding > image size)."""
    var np = Python.import_module("numpy")
    var F = Python.import_module("torch.nn.functional")

    var img = image
    var remaining = [paddings[0], paddings[1], paddings[2], paddings[3]]

    while remaining[0] > 0 or remaining[1] > 0 or remaining[2] > 0 or remaining[3] > 0:
        var shape = img.shape
        var img_w = Int(shape[2])
        var img_h = Int(shape[1])

        var possible = [
            min(remaining[0], img_w - 1),
            min(remaining[1], img_w - 1),
            min(remaining[2], img_h - 1),
            min(remaining[3], img_h - 1),
        ]

        img = F.pad(img, possible, mode="reflect")

        for i in range(4):
            remaining[i] = remaining[i] - possible[i]

    return img
