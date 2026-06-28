from jasna.py_compat import _hasattr
# Device utilities — device selection, validation, and synchronization.
# Multi-hardware support: CUDA, MPS (Apple Silicon), and CPU.

from std.python import Python, PythonObject




# ============================================================================
# Apple Silicon detection
# ============================================================================

def is_apple_silicon() raises -> Bool:
    """Check if running on Apple Silicon (ARM-based Mac)."""
    var platform = Python.import_module("platform")
    return String(py=platform.system()) == "Darwin" and String(py=platform.machine()) == "arm64"


# ============================================================================
# NVIDIA GPU availability
# ============================================================================

def is_nvidia_gpu_available() raises -> Bool:
    """Check if NVIDIA GPU is available via CUDA."""
    var torch = Python.import_module("torch")
    return Bool(py=torch.cuda.is_available())


# ============================================================================
# Get available device (auto-selection)
# ============================================================================

def get_available_device(preferred: String = "auto") raises -> PythonObject:
    """Get the best available device for processing.
    
    Args:
        preferred: "auto", "cuda", "mps", or "cpu"
    Returns:
        torch.device
    """
    var torch = Python.import_module("torch")

    if preferred == "cuda":
        if not Bool(py=torch.cuda.is_available()):
            raise Error("CUDA requested but not available")
        return torch.device("cuda")
    elif preferred == "mps":
        if not Bool(py=_hasattr(torch.backends, "mps")) or not Bool(py=torch.backends.mps.is_available()):
            raise Error("MPS requested but not available")
        return torch.device("mps")
    elif preferred == "cpu":
        return torch.device("cpu")
    elif preferred == "auto":
        if Bool(py=torch.cuda.is_available()):
            print("CUDA device available, selecting CUDA")
            return torch.device("cuda")
        elif Bool(py=_hasattr(torch.backends, "mps")) and Bool(py=torch.backends.mps.is_available()):
            print("MPS device available, selecting MPS (Apple Silicon)")
            return torch.device("mps")
        else:
            print("No GPU available, falling back to CPU")
            return torch.device("cpu")
    else:
        raise Error("Unknown device preference: " + preferred)


# ============================================================================
# Validate device for processing
# ============================================================================

def validate_device_for_processing(
    device: PythonObject,
    min_compute_capability: Tuple[Int, Int] = (7, 5),
) raises -> Tuple[Bool, String]:
    """Validate if device is suitable for processing.
    
    Args:
        device: torch.device to validate
        min_compute_capability: Minimum (major, minor) compute capability
    Returns:
        (is_valid, message)
    """
    var torch = Python.import_module("torch")
    var device_type = String(py=device.type)

    if device_type == "cuda":
        if not Bool(py=torch.cuda.is_available()):
            return (False, "CUDA not available")

        var cap = torch.cuda.get_device_capability(device.index)
        var cap_major = Int(py=cap[0])
        var cap_minor = Int(py=cap[1])
        var min_major = min_compute_capability[0]
        var min_minor = min_compute_capability[1]

        if cap_major < min_major or (cap_major == min_major and cap_minor < min_minor):
            var msg = "Compute capability " + String(cap_major) + "." + String(cap_minor) +
                " is below minimum required " + String(min_major) + "." + String(min_minor)
            return (False, msg)

        var name = String(torch.cuda.get_device_name(device.index))
        var msg = "NVIDIA GPU: " + name + " (CC " + String(cap_major) + "." + String(cap_minor) + ")"
        return (True, msg)

    elif device_type == "mps":
        if not Bool(py=_hasattr(torch.backends, "mps")) or not Bool(py=torch.backends.mps.is_available()):
            return (False, "MPS not available")
        return (True, "Apple Silicon GPU (Metal Performance Shaders)")

    elif device_type == "cpu":
        return (True, "CPU processing (slower but functional)")

    else:
        return (False, "Unsupported device type: " + device_type)


# ============================================================================
# Synchronize device
# ============================================================================

def synchronize_device(device: PythonObject) raises:
    """Synchronize device operations."""
    var torch = Python.import_module("torch")
    var device_type = String(py=device.type)

    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        if Bool(py=_hasattr(torch, "mps")) and Bool(py=_hasattr(torch.mps, "synchronize")):
            torch.mps.synchronize()


# ============================================================================
# Get device info
# ============================================================================

def get_device_info(device: PythonObject) raises -> Dict[String, PythonObject]:
    """Get information about a device."""
    var torch = Python.import_module("torch")
    var device_type = String(py=device.type)

    var info = Dict[String, PythonObject]()
    info["type"] = PythonObject(device_type)
    info["index"] = PythonObject(0 if device.index is None else device.index)
    info["available"] = PythonObject(False)
    info["name"] = PythonObject("Unknown")

    if device_type == "cuda":
        info["available"] = torch.cuda.is_available()
        if Bool(py=torch.cuda.is_available()):
            info["name"] = torch.cuda.get_device_name(device.index)
            info["capability"] = torch.cuda.get_device_capability(device.index)
            info["memory_total"] = torch.cuda.get_device_properties(device.index).total_memory
    elif device_type == "mps":
        var avail = Bool(py=_hasattr(torch.backends, "mps")) and Bool(py=torch.backends.mps.is_available())
        info["available"] = PythonObject(avail)
        if avail:
            info["name"] = PythonObject("Apple Silicon GPU")
    elif device_type == "cpu":
        info["available"] = PythonObject(True)
        info["name"] = PythonObject("CPU")

    return info
