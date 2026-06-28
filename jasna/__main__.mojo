# Jasna-mojo entry point.
# Sets up Python interop, environment variables, and dispatches to CLI.

from std.python import Python, PythonObject

from jasna.py_compat import setup_python_path, ensure_mock_modules
from jasna.main import main_entry


def main() raises:
    """Entry point for jasna-mojo."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")

    os_mod.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    if sys_mod.platform == "win32":
        os_mod.environ.setdefault("OMP_WAIT_POLICY", "passive")

    setup_python_path()
    ensure_mock_modules()

    if sys_mod.platform.startswith("linux"):
        var ctypes = Python.import_module("ctypes")
        try:
            ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
        except _:
            pass
        try:
            ctypes.CDLL("libnvcuvid.so.1", mode=ctypes.RTLD_GLOBAL)
        except _:
            pass

    try:
        main_entry()
    except e:
        var traceback = Python.import_module("traceback")
        traceback.print_exc()
        var err_str = String(e)
        if err_str == "1" or err_str == "0":
            return
        raise e^
