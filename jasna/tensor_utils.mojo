# Tensor utility functions.

from std.python import Python, PythonObject



def to_device(tensor: PythonObject, device: PythonObject) raises -> PythonObject:
    """Move tensor to device efficiently."""
    var torch = Python.import_module("torch")
    var np = Python.import_module("numpy")

    if String(tensor.device) == String(device):
        return tensor

    if String(tensor.device.type) == "cpu" and not Bool(tensor.is_contiguous()):
        tensor = torch.from_numpy(np.ascontiguousarray(tensor.numpy()))

    var out = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
    out.copy_(tensor, non_blocking=True)
    return out


def pad_batch_with_last(x: PythonObject, batch_size: Int) raises -> PythonObject:
    """Pad a batch tensor by repeating the last element."""
    var torch = Python.import_module("torch")
    var n = Int(x.shape[0])

    if n == batch_size:
        return x

    var pad = x[-1:].expand(batch_size - n, *x.shape[1:])
    return torch.cat([x, pad], dim=0)
