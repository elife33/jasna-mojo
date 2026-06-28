from std.collections import Set, Dict, List
# Video metadata parsing — uses ffprobe via subprocess.
# Extracts video properties: resolution, FPS, codec, color space, frame count, etc.

from std.python import Python, PythonObject
from jasna.os_utils import resolve_executable, get_subprocess_startup_info


# ============================================================================
# Supported encoder settings
# ============================================================================

def supported_encoder_settings() raises -> Set[String]:
    var s = Set[String]()
    s.add("preset")
    s.add("tuning_info")
    s.add("rc")
    s.add("cq")
    s.add("qmin")
    s.add("qmax")
    s.add("nonrefp")
    s.add("gop")
    s.add("maxbitrate")
    s.add("vbvinit")
    s.add("vbvbufsize")
    s.add("temporalaq")
    s.add("lookahead")
    s.add("lookahead_level")
    s.add("aq")
    s.add("initqp")
    s.add("tflevel")
    return s^


# ============================================================================
# Parse encoder settings
# ============================================================================

def _parse_encoder_setting_scalar(value: String) raises -> PythonObject:
    """Parse a scalar value string to bool/int/float/string."""
    var v = value.strip()
    if v.lower() == "true":
        return PythonObject(True)
    if v.lower() == "false":
        return PythonObject(False)
    try:
        return PythonObject(Int(v))
    except:
        pass
    try:
        return PythonObject(Float64(v))
    except:
        pass
    return PythonObject(v)


def parse_encoder_settings(value: String) raises -> Dict[String, PythonObject]:
    """Parse encoder settings from JSON or comma-separated key=value pairs."""
    var v = value.strip()
    if v == "":
        return Dict[String, PythonObject]()

    var json = Python.import_module("json")

    if v.startswith("{"):
        var parsed = json.loads(v)
        var result = Dict[String, PythonObject]()
        for k in parsed.keys():
            result[String(py=k)] = parsed[k]
        return result^

    var settings = Dict[String, PythonObject]()
    for part in v.split(","):
        var p = part.strip()
        if p == "":
            continue
        if not "=" in p:
            raise Error("Invalid encoder-settings item: " + p)
        var eq_idx = p.find("=")
        var k = String(p[byte=0:eq_idx].strip())
        var val = String(p[byte=eq_idx + 1:].strip())
        if k == "":
            raise Error("Invalid encoder-settings item (empty key): " + p)
        settings[k] = _parse_encoder_setting_scalar(val)

    return settings^


def validate_encoder_settings(var settings: Dict[String, PythonObject]) raises -> Dict[String, PythonObject]:
    """Validate that all settings keys are supported."""
    var supported = supported_encoder_settings()
    var invalid = List[String]()
    for k in settings.keys():
        if not k in supported:
            invalid.append(k)

    if len(invalid) > 0:
        raise Error("Unsupported encoder setting(s): " + ", ".join(invalid))

    return settings^


# ============================================================================
# Unsupported colorspace error
# ============================================================================

struct UnsupportedColorspaceError(Exception, Error):
    """Raised when video has an unsupported color space."""
    var msg: String

    def __init__(out self, msg: String) raises:
        self.msg = msg
        super().__init__(msg)


# ============================================================================
# Video metadata struct
# ============================================================================

struct VideoMetadata:
    """Metadata for a video file."""
    var video_file: String
    var video_height: Int
    var video_width: Int
    var video_fps: Float64
    var average_fps: Float64
    var video_fps_exact: PythonObject  # Fraction
    var codec_name: String
    var duration: Float64
    var time_base: PythonObject  # Fraction
    var start_pts: Int
    var color_range: PythonObject  # AvColorRange
    var color_space: PythonObject  # AvColorspace
    var num_frames: Int
    var is_10bit: Bool

    def __init__(out self) raises:
        self.video_file = ""
        self.video_height = 0
        self.video_width = 0
        self.video_fps = 0.0
        self.average_fps = 0.0
        self.video_fps_exact = PythonObject()
        self.codec_name = ""
        self.duration = 0.0
        self.time_base = PythonObject()
        self.start_pts = 0
        self.color_range = PythonObject()
        self.color_space = PythonObject()
        self.num_frames = 0
        self.is_10bit = False


# ============================================================================
# Helpers
# ============================================================================

def _parse_fraction(value: String) raises -> PythonObject:
    """Parse a fraction string like "30000/1001"."""
    var fractions = Python.import_module("fractions")
    if value == "" or value == "None":
        return PythonObject()
    try:
        var parts = value.split("/")
        var num = Int(parts[0])
        var den = Int(parts[1])
        if den == 0 or num <= 0:
            return PythonObject()
        return fractions.Fraction(num, den)
    except:
        return PythonObject()


def is_stream_10bit(json_video_stream: PythonObject) raises -> Bool:
    """Check if video stream is 10-bit."""
    var bprs = json_video_stream.get("bits_per_raw_sample")
    if bprs is not None:
        try:
            if Int(py=bprs) == 10:
                return True
        except:
            pass

    var pix_fmt = String(json_video_stream.get("pix_fmt") or "").lower()
    var markers = ["p10", "p010", "v210", "rgb10", "bgr10", "x2rgb10", "x2bgr10", "yuv10", "gray10"]
    for m in markers:
        if m in pix_fmt:
            return True
    return False


# ============================================================================
# Get video metadata via ffprobe
# ============================================================================

def get_video_meta_data(path: String) raises -> VideoMetadata:
    """Get video metadata using ffprobe.
    
    Args:
        path: Path to video file
    Returns:
        VideoMetadata struct
    """
    var subprocess = Python.import_module("subprocess")
    var json = Python.import_module("json")
    var fractions = Python.import_module("fractions")

    var ffprobe = resolve_executable("ffprobe")
    var cmd = Python.list([
        PythonObject(ffprobe),
        PythonObject("-v"), PythonObject("quiet"),
        PythonObject("-print_format"), PythonObject("json"),
        PythonObject("-select_streams"), PythonObject("v"),
        PythonObject("-show_streams"), PythonObject("-show_format"),
        PythonObject(path),
    ])

    var p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=get_subprocess_startup_info(),
    )
    var comm_result = p.communicate()
    var stdout_data = comm_result[0]
    var stderr_data = comm_result[1]
    if p.returncode != 0:
        raise Error("ffprobe failed: " + String(py=stderr_data.strip()))

    var json_output = json.loads(stdout_data)
    var json_vs = json_output["streams"][0]
    var json_vf = json_output["format"]

    # Parse FPS
    var avg_fps_exact = _parse_fraction(String(json_vs.get("avg_frame_rate") or ""))
    var r_fps_exact = _parse_fraction(String(json_vs.get("r_frame_rate") or ""))
    var fps_exact = avg_fps_exact if avg_fps_exact is not None else (r_fps_exact if r_fps_exact is not None else fractions.Fraction(30, 1))
    var average_fps = Float64(avg_fps_exact) if avg_fps_exact is not None else Float64(fps_exact)
    var fps = Float64(fps_exact)

    # Parse time base
    var tb_str = String(json_vs["time_base"])
    var tb_parts = tb_str.split("/")
    var time_base = fractions.Fraction(Int(tb_parts[0]), Int(tb_parts[1]))

    # Parse color info
    var av = Python.import_module("av.video.reformatter")
    var AvColorspace = av.Colorspace
    var AvColorRange = av.ColorRange

    var cr_str = String(json_vs.get("color_range") or "")
    var color_range = AvColorRange.MPEG
    if cr_str == "jpeg":
        color_range = AvColorRange.JPEG

    var cs_str = String(json_vs.get("color_space") or "")
    var color_space = AvColorspace.ITU709
    if cs_str in ("bt601", "bt470bg", "smpte170m"):
        color_space = AvColorspace.ITU601

    # Parse frame count
    var num_frames = Int(json_vs.get("nb_frames", 0))
    if num_frames == 0:
        # Fallback: count frames with OpenCV
        var cv2 = Python.import_module("cv2")
        var cap = cv2.VideoCapture(path)
        num_frames = Int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    var is_10bit = is_stream_10bit(json_vs)

    var meta = VideoMetadata()
    meta.video_file = path
    meta.video_height = Int(json_vs["height"])
    meta.video_width = Int(json_vs["width"])
    meta.video_fps = fps
    meta.average_fps = average_fps
    meta.video_fps_exact = fps_exact
    meta.codec_name = String(json_vs["codec_name"])
    meta.duration = Float64(json_vs.get("duration", json_vf["duration"]))
    meta.time_base = time_base
    meta.start_pts = Int(json_vs.get("start_pts", 0))
    meta.color_range = color_range
    meta.color_space = color_space
    meta.num_frames = num_frames
    meta.is_10bit = is_10bit

    return meta
