# Detection data structures.

from std.python import Python, PythonObject


@fieldwise_init
struct Detections:
    """Detection results for a batch of frames.
    
    Fields:
        boxes_xyxy: List of numpy arrays, each (N_i, 4) xyxy in pixels, CPU
        masks: List of torch.Tensor, each (N_i, Hm, Wm) bool, GPU
    """
    var boxes_xyxy: List[PythonObject]
    var masks: List[PythonObject]

    def __init__(out self) raises:
        self.boxes_xyxy = List[PythonObject]()
        self.masks = List[PythonObject]()
