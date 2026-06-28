# OS utilities — executable discovery, version checks, platform-specific helpers.
# Uses Python interop for subprocess, shutil, and platform operations.

from std.python import Python, PythonObject




# Constants
comptime MIN_GPU_COMPUTE_MAJOR = 7
comptime MIN_GPU_COMPUTE_MINOR = 5
comptime MIN_DRIVER_VERSION = 590


# ============================================================================
# Find executable in PATH or bundled
# ============================================================================

def find_executable(name: String) raises -> Optional[String]:
    """Find an executable in PATH. Returns None if not found."""
    var shutil = Python.import_module("shutil")
    var result = shutil.which(name)
    if result is None:
        return None
    return String(result)


def resolve_executable(name: String) raises -> String:
    """Find executable or return the name as-is."""
    var found = find_executable(name)
    if found is not None:
        return found.value()
    return name


# ============================================================================
# Subprocess startup info (Windows)
# ============================================================================

def get_subprocess_startup_info() raises -> PythonObject:
    """Get Windows STARTUPINFO to hide console windows. Returns None on non-Windows."""
    var sys_mod = Python.import_module("sys")
    if sys_mod.platform != "win32":
        return PythonObject()
    var subprocess = Python.import_module("subprocess")
    var startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info


# ============================================================================
# Check NVIDIA GPU
# ============================================================================

def check_nvidia_gpu() raises -> Tuple[Bool, String]:
    """Check NVIDIA GPU availability and compute capability.
    
    Returns:
        (True, gpu_name) or (False, error_message)
    """
    var torch = Python.import_module("torch")
    if not Bool(py=torch.cuda.is_available()):
        return (False, "no_cuda")

    var cap = torch.cuda.get_device_capability(0)
    var cap_major = Int(py=cap[0])
    var cap_minor = Int(py=cap[1])

    if cap_major < MIN_GPU_COMPUTE_MAJOR or (cap_major == MIN_GPU_COMPUTE_MAJOR and cap_minor < MIN_GPU_COMPUTE_MINOR):
        return (False, "compute_too_low")

    var name = String(torch.cuda.get_device_name(0))
    return (True, name)


# ============================================================================
# Check GPU driver version
# ============================================================================

def check_gpu_driver_version() raises -> Tuple[Bool, String]:
    """Check NVIDIA driver version meets minimum requirement."""
    var nvidia_smi = find_executable("nvidia-smi")
    if nvidia_smi is None:
        return (False, "nvidia-smi not found")

    var subprocess = Python.import_module("subprocess")
    var re = Python.import_module("re")

    var cmd = Python.list([nvidia_smi.value(), "--query-gpu=driver_version", "--format=csv,noheader"])
    var result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        startupinfo=get_subprocess_startup_info(),
    )

    if result.returncode != 0:
        return (False, "nvidia-smi exited with code " + String(py=result.returncode))

    var version_str = String(result.stdout.strip().split("\n")[0].strip())
    var m = re.match(r"(\d+)\.(\d+)", version_str)
    if m is None:
        return (False, "Could not parse driver version: " + version_str)

    var major = Int(py=m.group(1))
    if major < MIN_DRIVER_VERSION:
        return (False, version_str + " (requires " + String(MIN_DRIVER_VERSION) + "+)")

    return (True, version_str)


# ============================================================================
# Check required executables (ffmpeg, ffprobe, mkvmerge)
# ============================================================================

def _parse_ffmpeg_major_version(version_output: String) raises -> Int:
    """Parse ffmpeg major version from version output."""
    var re = Python.import_module("re")
    var lines = version_output.split("\n")
    var first_line = String(lines[0]) if len(lines) > 0 else String("")

    var m = re.match(r"^\s*(?:ffmpeg|ffprobe)\s+version\s+(\S+)", PythonObject(first_line))
    if m is None:
        raise Error("Unexpected ffmpeg/ffprobe version output: " + first_line)

    var ver = String(m.group(1))
    if ver.startswith("N-") or ver.startswith("git-") or ver.startswith("GIT-"):
        # Parse from libavutil version
        var libm = re.search(r"(?m)^\s*libavutil\s+(\d+)\.", PythonObject(version_output))
        if libm is None:
            raise Error("Could not parse libavutil version from ffmpeg output")
        var libavutil_major = Int(py=libm.group(1))
        return libavutil_major - 52

    var digit_match = re.search(r"(\d+)", ver)
    if digit_match is None:
        raise Error("Could not parse ffmpeg major version from: " + ver)
    return Int(py=digit_match.group(1))


def check_required_executables(disable_ffmpeg_check: Bool = False) raises:
    """Check that required external tools are available in PATH and callable."""
    var subprocess = Python.import_module("subprocess")

    var missing = List[String]()
    var wrong_version = List[String]()

    var checks = List[Tuple[String, List[String]]]()
    if not disable_ffmpeg_check:
        checks.append(("ffprobe", ["-version"]))
        checks.append(("ffmpeg", ["-version"]))
    checks.append(("mkvmerge", ["--version"]))

    for i in range(len(checks)):
        var exe_name = checks[i][0]
        var exe_args = List[String]()
        for a in checks[i][1]:
            exe_args.append(a)
        var exe_path = find_executable(exe_name)
        if exe_path is None:
            missing.append(exe_name)
            continue

        var cmd = Python.list([PythonObject(exe_path.value())] + [PythonObject(a) for a in exe_args])
        var completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            missing.append(exe_name)
            continue

        if exe_name == "ffprobe" or exe_name == "ffmpeg":
            try:
                var output = String(completed.stdout or "") + String(completed.stderr or "")
                var major = _parse_ffmpeg_major_version(output)
                if major != 8:
                    wrong_version.append(exe_name + " (detected major=" + String(major) + ")")
            except:
                wrong_version.append(exe_name + " (could not detect major version)")

    if len(missing) > 0:
        var msg = "Error: Required executable(s) not found: " + ", ".join(missing)
        print(msg)
        raise Error(msg)

    if len(wrong_version) > 0:
        var msg = "Error: ffmpeg/ffprobe major version must be 8: " + ", ".join(wrong_version)
        print(msg)
        raise Error(msg)


# ============================================================================
# Check ASCII install path
# ============================================================================

def check_ascii_install_path() raises -> Tuple[Bool, String]:
    """Check that the install path contains only ASCII characters."""
    var Path = Python.import_module("pathlib")
    var os_mod = Python.import_module("os")
    var path = Path(os_mod.getcwd())
    var path_str = String(py=path)
    try:
        Python.evaluate("lambda s: s.encode('ascii')")(path_str)
        return (True, path_str)
    except:
        return (False, path_str)


# ============================================================================
# User config directory
# ============================================================================

def get_user_config_dir(app_name: String) raises -> String:
    """Get the user's config directory for an application."""
    var os_mod = Python.import_module("os")
    var pathlib = Python.import_module("pathlib")

    var sys_mod = Python.import_module("sys")
    if sys_mod.platform == "win32":
        var base = os_mod.environ.get("APPDATA") or os_mod.environ.get("LOCALAPPDATA")
        if base is not None:
            return String(py=pathlib.Path(base) / app_name)
        return String(py=pathlib.Path.home() / app_name)

    var xdg = os_mod.environ.get("XDG_CONFIG_HOME")
    if xdg is not None:
        return String(py=pathlib.Path(xdg) / app_name)
    return String(py=pathlib.Path.home() / ".config" / app_name)
