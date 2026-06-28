# YOLO mosaic detection model — Ultralytics YOLO for mosaic detection.
# Uses Python interop for ultralytics library.

from std.python import Python, PythonObject
from jasna.mosaic.detections import Detections


struct YoloMosaicDetectionModel:
    """YOLO-based mosaic detection model using Ultralytics.
    
    Supports YOLO models trained for mosaic detection (lada-yolo-v2, v4).
    Runs on GPU (CUDA/MPS) or CPU.
    """

    var model_path: PythonObject
    var batch_size: Int
    var device: PythonObject
    var score_threshold: Float64
    var max_nms: Int
    var fp16: Bool
    var _model: PythonObject  # ultralytics YOLO model

    def DEFAULT_IMGSZ() raises -> Int:
        return 640

    def __init__(
        mut self,
        model_path: PythonObject,
        batch_size: Int,
        device: PythonObject,
        score_threshold: Float64 = 0.25,
        max_nms: Int = 4096,
        fp16: Bool = True,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.score_threshold = score_threshold
        self.max_nms = max_nms
        self.fp16 = fp16
        self._model = PythonObject()

        # Load YOLO model via ultralytics
        var load_fn = Python.evaluate("""
def _load_yolo(path, device, fp16) raises:
    from ultralytics import YOLO
    model = YOLO(String(path))
    return model
""")
        self._model = load_fn(model_path, device, fp16)

    def close(mut self) raises:
        """Close the model."""
        self._model = PythonObject()

    def __call__(
        self,
        frames: PythonObject,
        target_hw: Tuple[Int, Int],
    ) raises -> Detections:
        """Run detection on a batch of frames.
        
        Args:
            frames: torch.Tensor (B, C, H, W) — input frames
            target_hw: (target_h, target_w) — original frame dimensions
        Returns:
            Detections with boxes and masks
        """
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")

        var batch_size = Int(frames.shape[0])
        var target_h = target_hw[0]
        var target_w = target_hw[1]

        # Run YOLO inference via Python interop
        var detect_fn = Python.evaluate("""
def _detect(model, frames, imgsz, score_threshold, device, target_h, target_w) raises:
    import numpy as np
    import torch
    
    # YOLO expects (B, H, W, C) uint8 or similar
    # Convert from (B, C, H, W) float to expected format
    frames_u8 = (frames.clamp(0, 255)).to(torch.uint8)
    # Permute to (B, H, W, C) for ultralytics
    frames_hwc = frames_u8.permute(0, 2, 3, 1).cpu().numpy()
    
    results = model(
        frames_hwc,
        imgsz=imgsz,
        conf=score_threshold,
        device=device,
        verbose=False,
    )
    
    boxes_list = []
    masks_list = []
    
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            boxes_list.append(boxes)
            
            # Get masks if available
            if r.masks is not None:
                masks = r.masks.data.bool()
                # Resize masks to target resolution
                mask_h = target_h // 4
                mask_w = target_w // 4
                if masks.shape[1] != mask_h or masks.shape[2] != mask_w:
                    import torch.nn.functional as F
                    masks = F.interpolate(
                        masks.float().unsqueeze(0),
                        size=(mask_h, mask_w),
                        mode='nearest',
                    ).squeeze(0).bool()
                masks_list.append(masks)
            else:
                # Create rectangular masks from boxes
                mask_h = target_h // 4
                mask_w = target_w // 4
                masks = torch.zeros((len(boxes), mask_h, mask_w), dtype=torch.bool)
                for j, box in enumerate(boxes):
                    x1 = int(box[0] * mask_w / target_w)
                    y1 = int(box[1] * mask_h / target_h)
                    x2 = int(box[2] * mask_w / target_w)
                    y2 = int(box[3] * mask_h / target_h)
                    masks[j, max(0,y1):y2, max(0,x1):x2] = True
                masks_list.append(masks)
        else:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            mask_h = target_h // 4
            mask_w = target_w // 4
            masks_list.append(torch.zeros((0, mask_h, mask_w), dtype=torch.bool))
    
    return boxes_list, masks_list
""")
        var (boxes_list, masks_list) = detect_fn(
            self._model, frames, YoloMosaicDetectionModel.DEFAULT_IMGSZ(),
            self.score_threshold, self.device, target_h, target_w,
        )

        var detections = Detections()
        for i in range(batch_size):
            detections.boxes_xyxy.append(boxes_list[i])
            detections.masks.append(masks_list[i])

        return detections
