# Detection model registry — discovers and manages detection models.
# Supports RF-DETR (ONNX) and YOLO (PyTorch) models.

from std.python import Python, PythonObject
from std.collections import Set, Dict, List



# ============================================================================
# Model name sets
# ============================================================================

comptime DEFAULT_DETECTION_MODEL_NAME = "rfdetr-v5"


def _get_rfdetr_names() raises -> Set[String]:
    var s = Set[String]()
    s.add("rfdetr-v2")
    s.add("rfdetr-v3")
    s.add("rfdetr-v4")
    s.add("rfdetr-v5")
    return s^

def _get_yolo_names() raises -> Set[String]:
    var s = Set[String]()
    s.add("lada-yolo-v2")
    s.add("lada-yolo-v4")
    return s^

def _get_yolo_files() raises -> Dict[String, String]:
    var d = Dict[String, String]()
    d["lada-yolo-v2"] = "lada_mosaic_detection_model_v2.pt"
    d["lada-yolo-v4"] = "lada_mosaic_detection_model_v4_fast.pt"
    return d^


# ============================================================================
# Model type checks
# ============================================================================

def is_rfdetr_model(name: String) raises -> Bool:
    """Check if name matches RF-DETR pattern."""
    return name.startswith("rfdetr-")


def is_yolo_model(name: String) raises -> Bool:
    """Check if name is a known YOLO model."""
    return name in _get_yolo_names()


# ============================================================================
# Model discovery
# ============================================================================

def _default_model_weights_dir() raises -> PythonObject:
    """Get the default model weights directory."""
    var pathlib = Python.import_module("pathlib")
    var os_mod = Python.import_module("os")

    var cwd_dir = pathlib.Path("model_weights")
    if Bool(cwd_dir.is_dir()):
        return cwd_dir.resolve()

    var repo_dir = pathlib.Path(os_mod.getcwd()) / "model_weights"
    if Bool(repo_dir.is_dir()):
        return repo_dir.resolve()

    return cwd_dir


def discover_available_detection_models() raises -> List[String]:
    """Discover available detection models from model_weights/ directory."""
    var model_weights_dir = _default_model_weights_dir()

    var rfdetr_names = List[String]()
    var yolo_names = List[String]()

    if Bool(model_weights_dir.is_dir()):
        for f in model_weights_dir.iterdir():
            var suffix = String(f.suffix)
            var stem = String(f.stem)
            var name = String(f.name)

            if suffix == ".onnx" and is_rfdetr_model(stem):
                rfdetr_names.append(stem)

            # Check YOLO files
            var yolo_files = _get_yolo_files()
            var yolo_keys = List[String]()
            for k in yolo_files.keys():
                yolo_keys.append(k)
            for yolo_name in yolo_keys:
                var expected = String(yolo_files[yolo_name])
                if name == expected:
                    yolo_names.append(yolo_name)

    # Sort reverse (latest first)
    _sort_reverse(rfdetr_names)
    _sort_reverse(yolo_names)

    var result = List[String]()
    for n in rfdetr_names:
        result.append(n)
    for n in yolo_names:
        result.append(n)
    return result^


def _sort_reverse(mut lst: List[String]) raises:
    """Sort a list of strings in reverse order (in-place)."""
    # Simple bubble sort
    for i in range(len(lst)):
        for j in range(len(lst) - 1 - i):
            if lst[j] < lst[j + 1]:
                var tmp = lst[j]
                lst[j] = lst[j + 1]
                lst[j + 1] = tmp


# ============================================================================
# Model name coercion
# ============================================================================

def coerce_detection_model_name(name: String) raises -> String:
    """Coerce a model name to a valid detection model name."""
    var lower = name.lower()
    if is_rfdetr_model(lower) or is_yolo_model(lower):
        return lower
    return DEFAULT_DETECTION_MODEL_NAME


# ============================================================================
# Model weights path
# ============================================================================

def detection_model_weights_path(name: String) raises -> PythonObject:
    """Get the path to detection model weights."""
    var coerced = coerce_detection_model_name(name)
    var model_weights_dir = _default_model_weights_dir()

    if is_rfdetr_model(coerced):
        return model_weights_dir / (coerced + ".onnx")
    if is_yolo_model(coerced):
        return model_weights_dir / _get_yolo_files()[coerced]
    return model_weights_dir / (DEFAULT_DETECTION_MODEL_NAME + ".onnx")


# ============================================================================
# Precompile detection engine (TensorRT, NVIDIA only)
# ============================================================================

def precompile_detection_engine(
    detection_model_name: String,
    detection_model_path: PythonObject,
    batch_size: Int,
    device: PythonObject,
    fp16: Bool,
):
    """Precompile detection model to TensorRT engine (NVIDIA only)."""
    if String(py=device.type) != "cuda":
        return

    var det_name = coerce_detection_model_name(detection_model_name)

    if is_rfdetr_model(det_name):
        var compile_fn = Python.evaluate("""
def _compile_rfdetr(path, device, batch_size, fp16):
    from jasna.mosaic.rfdetr import compile_rfdetr_engine
    compile_rfdetr_engine(path, device, batch_size=int(batch_size), fp16=bool(fp16))
""", file=True)

        compile_fn._compile_rfdetr(detection_model_path, device, batch_size, fp16)
    elif is_yolo_model(det_name):
        var compile_fn = Python.evaluate("""
def _compile_yolo(path, batch, fp16, imgsz, device):
    from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine
    from jasna.mosaic.yolo import YoloMosaicDetectionModel
    compile_yolo_to_tensorrt_engine(
        path,
        batch=int(batch),
        fp16=bool(fp16),
        imgsz=YoloMosaicDetectionModel.DEFAULT_IMGSZ,
        device=device,
    )
""", file=True)

        compile_fn._compile_rfdetr(detection_model_path, batch_size, fp16, 640, device)
