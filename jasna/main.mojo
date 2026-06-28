from jasna.py_compat import _hasattr
# CLI main entry point — argument parsing and pipeline initialization.
# Supports multi-hardware: CUDA, MPS (Apple Silicon), and CPU.

from std.python import Python, PythonObject
from std.collections import Set, Dict, List, Optional
from pathlib import Path





from jasna import VERSION
from jasna.device_utils import (
    get_available_device,
    validate_device_for_processing,
    is_apple_silicon,
)
from jasna.os_utils import (
    check_required_executables,
    check_ascii_install_path,
    check_gpu_driver_version,
    find_executable,
)
from jasna.media.video_metadata import (
    parse_encoder_settings,
    validate_encoder_settings,
    UnsupportedColorspaceError,
)
from jasna.mosaic.detection_registry import (
    coerce_detection_model_name,
    detection_model_weights_path,
    discover_available_detection_models,
    is_rfdetr_model,
    is_yolo_model,
)
from jasna.pipeline import Pipeline


# ============================================================================
# Argument parser
# ============================================================================

@fieldwise_init
struct Args(Movable, Copyable):
    """Parsed command-line arguments."""
    var input: String
    var output: String
    var start_frame: Optional[Int]
    var duration_frames: Optional[Int]
    var status_file: String
    var cont: String
    var batch_size: Int
    var device: String
    var fp16: Bool
    var log_level: String
    var disable_ffmpeg_check: Bool
    var no_progress: Bool
    var apple_silicon_auto_tune: Bool
    var benchmark: Bool
    var benchmark_filter: String
    var benchmark_video: List[String]
    # Restoration
    var restoration_model_name: String
    var restoration_model_path: String
    var compile_basicvsrpp: Bool
    var max_clip_size: Int
    var temporal_overlap: Int
    var enable_crossfade: Bool
    var denoise: String
    var denoise_step: String
    # Secondary restoration
    var secondary_restoration: String
    # RTX Super Res
    var rtx_scale: Int
    var rtx_quality: String
    var rtx_denoise: String
    var rtx_deblur: String
    # TVAI
    var tvai_ffmpeg_path: String
    var tvai_model: String
    var tvai_scale: Int
    var tvai_args: String
    var tvai_workers: Int
    # Detection
    var detection_model: String
    var detection_model_path: String
    var detection_score_threshold: Float64
    var detection_max_candidates: Int
    # Streaming
    var stream: Bool
    var stream_port: Int
    var stream_segment_duration: Float64
    var no_browser: Bool
    # Encoding
    var codec: String
    var encoder_settings: String
    var working_directory: String

    def __init__(out self) raises:
        self.input = ""
        self.output = ""
        self.start_frame = None
        self.duration_frames = None
        self.status_file = "/tmp/jasna_status.json"
        self.cont = ""
        self.batch_size = 4
        self.device = "cuda:0"
        self.fp16 = True
        self.log_level = "error"
        self.disable_ffmpeg_check = False
        self.no_progress = False
        self.apple_silicon_auto_tune = False
        self.benchmark = False
        self.benchmark_filter = ""
        self.benchmark_video = List[String]()
        self.restoration_model_name = "basicvsrpp"
        self.restoration_model_path = ""
        self.compile_basicvsrpp = True
        self.max_clip_size = 90
        self.temporal_overlap = 8
        self.enable_crossfade = True
        self.denoise = "none"
        self.denoise_step = "after_primary"
        self.secondary_restoration = "none"
        self.rtx_scale = 4
        self.rtx_quality = "high"
        self.rtx_denoise = "medium"
        self.rtx_deblur = "none"
        self.tvai_ffmpeg_path = "C:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe"
        self.tvai_model = "iris-2"
        self.tvai_scale = 4
        self.tvai_args = "preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"
        self.tvai_workers = 2
        self.detection_model = "rfdetr-v5"
        self.detection_model_path = ""
        self.detection_score_threshold = 0.25
        self.detection_max_candidates = 4096
        self.stream = False
        self.stream_port = 8765
        self.stream_segment_duration = 4.0
        self.no_browser = False
        self.codec = "h264"
        self.encoder_settings = ""
        self.working_directory = ""


def _default_model_weight_path(filename: String) raises -> String:
    """Get the default path for a model weight file."""
    var pathlib = Python.import_module("pathlib")
    var cwd = pathlib.Path("model_weights")
    if Bool(py=cwd.is_dir()):
        return String(py=cwd.resolve() / filename)
    var os_mod = Python.import_module("os")
    var repo = pathlib.Path(os_mod.getcwd()) / "model_weights"
    if Bool(py=repo.is_dir()):
        return String(py=repo / filename)
    return String(py=cwd / filename)


def parse_args(argv: List[String]) raises -> Args:
    """Parse command-line arguments using Python's argparse."""
    var argparse = Python.import_module("argparse")

    # Build parser in Python for full argparse support
    var build_parser = Python.evaluate("""
def _build_parser(default_model_path):
    import argparse
    parser = argparse.ArgumentParser(prog="jasna")
    parser.add_argument("--version", action="version", version="0.6.0-alpha5")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--duration-frames", type=int, default=None)
    parser.add_argument("--status-file", type=str, default="/tmp/jasna_status.json")
    parser.add_argument("--cont", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--log-level", type=str, default="error", choices=["debug","info","warning","error"])
    parser.add_argument("--disable-ffmpeg-check", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--apple-silicon-auto-tune", default=False, action=argparse.BooleanOptionalAction)

    restoration = parser.add_argument_group("Restoration")
    restoration.add_argument("--restoration-model-name", type=str, default="basicvsrpp", choices=["basicvsrpp"])
    restoration.add_argument("--restoration-model-path", type=str, default=default_model_path)
    restoration.add_argument("--compile-basicvsrpp", default=True, action=argparse.BooleanOptionalAction)
    restoration.add_argument("--max-clip-size", type=int, default=90)
    restoration.add_argument("--temporal-overlap", type=int, default=8)
    restoration.add_argument("--enable-crossfade", default=True, action=argparse.BooleanOptionalAction)
    restoration.add_argument("--denoise", type=str, default="none", choices=["none","low","medium","high"])
    restoration.add_argument("--denoise-step", type=str, default="after_primary", choices=["after_primary","after_secondary"])

    secondary = parser.add_argument_group("2nd restoration")
    secondary.add_argument("--secondary-restoration", type=str, default="none", choices=["none","unet-4x","tvai","rtx-super-res"])

    rtx = parser.add_argument_group("RTX Super Res")
    rtx.add_argument("--rtx-scale", type=int, default=4, choices=[2,4])
    rtx.add_argument("--rtx-quality", type=str, default="high", choices=["low","medium","high","ultra"])
    rtx.add_argument("--rtx-denoise", type=str, default="medium", choices=["none","low","medium","high","ultra"])
    rtx.add_argument("--rtx-deblur", type=str, default="none", choices=["none","low","medium","high","ultra"])

    tvai = parser.add_argument_group("Topaz Video")
    tvai.add_argument("--tvai-ffmpeg-path", type=str, default="C:\\\\Program Files\\\\Topaz Labs LLC\\\\Topaz Video\\\\ffmpeg.exe")
    tvai.add_argument("--tvai-model", type=str, default="iris-2")
    tvai.add_argument("--tvai-scale", type=int, default=4, choices=[1,2,4])
    tvai.add_argument("--tvai-args", type=str, default="preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1")
    tvai.add_argument("--tvai-workers", type=int, default=2)

    detection = parser.add_argument_group("Detection")
    detection.add_argument("--detection-model", type=str, default="rfdetr-v5")
    detection.add_argument("--detection-model-path", type=str, default="")
    detection.add_argument("--detection-score-threshold", type=float, default=0.25)
    detection.add_argument("--detection-max-candidates", type=int, default=4096)

    streaming = parser.add_argument_group("Streaming")
    streaming.add_argument("--stream", action="store_true")
    streaming.add_argument("--stream-port", type=int, default=8765)
    streaming.add_argument("--stream-segment-duration", type=float, default=4.0)
    streaming.add_argument("--no-browser", action="store_true")

    encoding = parser.add_argument_group("Encoding")
    encoding.add_argument("--codec", type=str, default="h264")
    encoding.add_argument("--encoder-settings", type=str, default="")
    encoding.add_argument("--working-directory", type=str, default="")

    benchmark_group = parser.add_argument_group("Benchmark")
    benchmark_group.add_argument("--benchmark-filter", type=str, default=None)
    benchmark_group.add_argument("--benchmark-video", type=str, action="append", default=None)

    return parser
""", file=True)


    var default_model_path = _default_model_weight_path("lada_mosaic_restoration_model_generic_v1.2.pth")
    var parser = build_parser._build_parser(default_model_path)

    # Parse args
    var py_argv = Python.list([PythonObject(a) for a in argv])
    var parsed = parser.parse_args(py_argv)

    # Convert to Mojo Args struct
    var args = Args()
    args.input = String(py=parsed.input or "")
    args.output = String(py=parsed.output or "")
    args.start_frame = Int(py=parsed.start_frame) if parsed.start_frame is not None else None
    args.duration_frames = Int(py=parsed.duration_frames) if parsed.duration_frames is not None else None
    args.status_file = String(py=parsed.status_file or "/tmp/jasna_status.json")
    args.cont = String(py=parsed.cont or "")
    args.batch_size = Int(py=parsed.batch_size)
    args.device = String(py=parsed.device)
    args.fp16 = Bool(py=parsed.fp16)
    args.log_level = String(py=parsed.log_level)
    args.disable_ffmpeg_check = Bool(py=parsed.disable_ffmpeg_check)
    args.no_progress = Bool(py=parsed.no_progress)
    args.apple_silicon_auto_tune = Bool(py=parsed.apple_silicon_auto_tune)
    args.benchmark = Bool(py=parsed.benchmark)
    args.restoration_model_name = String(py=parsed.restoration_model_name)
    args.restoration_model_path = String(py=parsed.restoration_model_path)
    args.compile_basicvsrpp = Bool(py=parsed.compile_basicvsrpp)
    args.max_clip_size = Int(py=parsed.max_clip_size)
    args.temporal_overlap = Int(py=parsed.temporal_overlap)
    args.enable_crossfade = Bool(py=parsed.enable_crossfade)
    args.denoise = String(py=parsed.denoise)
    args.denoise_step = String(py=parsed.denoise_step)
    args.secondary_restoration = String(py=parsed.secondary_restoration)
    args.rtx_scale = Int(py=parsed.rtx_scale)
    args.rtx_quality = String(py=parsed.rtx_quality)
    args.rtx_denoise = String(py=parsed.rtx_denoise)
    args.rtx_deblur = String(py=parsed.rtx_deblur)
    args.tvai_ffmpeg_path = String(py=parsed.tvai_ffmpeg_path)
    args.tvai_model = String(py=parsed.tvai_model)
    args.tvai_scale = Int(py=parsed.tvai_scale)
    args.tvai_args = String(py=parsed.tvai_args)
    args.tvai_workers = Int(py=parsed.tvai_workers)
    args.detection_model = String(py=parsed.detection_model)
    args.detection_model_path = String(py=parsed.detection_model_path or "")
    args.detection_score_threshold = Float64(py=parsed.detection_score_threshold)
    args.detection_max_candidates = Int(py=parsed.detection_max_candidates)
    args.stream = Bool(py=parsed.stream)
    args.stream_port = Int(py=parsed.stream_port)
    args.stream_segment_duration = Float64(py=parsed.stream_segment_duration)
    args.no_browser = Bool(py=parsed.no_browser)
    args.codec = String(py=parsed.codec)
    args.encoder_settings = String(py=parsed.encoder_settings)
    args.working_directory = String(py=parsed.working_directory or "")
    args.benchmark_filter = String(py=parsed.benchmark_filter or "")

    return args^


# ============================================================================
# Apple Silicon auto-tune
# ============================================================================

def _apply_apple_silicon_auto_tune(
    device: PythonObject,
    batch_size: Int,
    max_clip_size: Int,
    temporal_overlap: Int,
    detection_max_candidates: Int,
    enabled: Bool,
) raises -> Tuple[Int, Int, Int, Int]:
    """Auto-tune parameters for Apple Silicon stability."""
    if not enabled or String(py=device.type) != "mps":
        return (batch_size, max_clip_size, temporal_overlap, detection_max_candidates)

    # Conservative defaults for Apple Silicon
    var tuned_batch = min(batch_size, 2)
    var tuned_clip = min(max_clip_size, 60)
    var tuned_overlap = min(temporal_overlap, 6)
    var tuned_candidates = min(detection_max_candidates, 1024)

    print("Apple Silicon auto-tune: batch=" + String(tuned_batch) +
          " clip=" + String(tuned_clip) +
          " overlap=" + String(tuned_overlap))
    return (tuned_batch, tuned_clip, tuned_overlap, tuned_candidates)


# ============================================================================
# Main function
# ============================================================================

def _mojo_dict_to_py(d: Dict[String, PythonObject]) raises -> PythonObject:
    """Convert a Mojo Dict[String, PythonObject] to a Python dict."""
    var py_d = Python.dict()
    var keys = List[String]()
    for k in d.keys():
        keys.append(k)
    for k in keys:
        py_d[k] = d[k]
    return py_d


def _run_cli() raises:
    """Main CLI entry point."""
    var sys_mod = Python.import_module("sys")
    # Mojo's argv includes the program name; sync to Python's sys.argv
    from std.sys import argv as mojo_argv
    var raw_argv = List[String]()
    for a in mojo_argv():
        raw_argv.append(String(a))
    # Skip leading -- separator if present (from `mojo run file -- args`)
    var skip_first = False
    if len(raw_argv) > 1 and raw_argv[1] == "--":
        skip_first = True
    var argv_list = List[String]()
    for i in range(1, len(raw_argv)):
        if skip_first and i == 1:
            continue
        argv_list.append(raw_argv[i])
    # Inject into Python sys.argv so Python code can also access it
    var py_argv = Python.list([PythonObject(raw_argv[0])] + [PythonObject(a) for a in argv_list])
    sys_mod.argv = py_argv

    var args = parse_args(argv_list)

    var pathlib = Python.import_module("pathlib")
    var json = Python.import_module("json")
    var logging = Python.import_module("logging")
    var signal = Python.import_module("signal")

    # Handle --cont (resume from status file)
    var status_file_path = pathlib.Path(args.status_file)
    if args.cont != "":
        var cont_path = pathlib.Path(args.cont)
        if not Bool(cont_path.exists()):
            print("Status file not found: " + args.cont)
            sys_mod.exit(1)
        var status_data = json.loads(cont_path.read_text(encoding="utf-8"))
        args.input = String(py=status_data["input_video"])
        if status_data.get("output_video"):
            args.output = String(py=status_data["output_video"])
        if args.start_frame is None:
            args.start_frame = Int(py=status_data["current_frame"])
        var sf = args.start_frame
        if sf:
            print("Resuming from frame " + String(sf.value()))

    # Handle benchmark mode
    if args.benchmark:
        var run_benchmark = Python.evaluate("""
def _run_benchmark(args):
    from jasna.benchmark import run_benchmark_cli
    run_benchmark_cli(args)
""", file=True)

        var py_args = Python.dict()
        py_args["input"] = args.input
        py_args["output"] = args.output
        py_args["batch_size"] = args.batch_size
        py_args["device"] = args.device
        py_args["fp16"] = args.fp16
        py_args["benchmark_filter"] = args.benchmark_filter
        var py_bv = Python.list([PythonObject(v) for v in args.benchmark_video])
        py_args["benchmark_video"] = py_bv
        run_benchmark._run_benchmark(py_args)
        return

    var is_streaming = args.stream

    # Validate required args
    if args.input == "" and not is_streaming:
        print("--input is required when not using --benchmark or --stream")
        sys_mod.exit(1)
    if args.output == "" and not is_streaming:
        print("--output is required when not using --benchmark or --stream")
        sys_mod.exit(1)

    # Check ASCII path
    var path_result = check_ascii_install_path()
    var path_ok = path_result[0]
    var path_info = path_result[1]
    if not path_ok:
        print("Error: Jasna must be installed in a path with ASCII characters only.")
        print("Current path: " + path_info)
        sys_mod.exit(1)

    # Check executables
    check_required_executables(disable_ffmpeg_check=args.disable_ffmpeg_check)

    # Setup logging
    var builtins = Python.import_module("builtins")
    logging.basicConfig(
        level=builtins.getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noise
    var install_noise = Python.evaluate("""
def _install_noise():
    from jasna._suppress_noise import install
    install()
""", file=True)

    install_noise._install_noise()

    var torch = Python.import_module("torch")

    # Setup signal handlers for pause/resume
    var start_frame_val = 0
    if args.start_frame:
        start_frame_val = args.start_frame.value()
    var pause_requested = Python.evaluate("[False]")
    var current_frame_shared = Python.evaluate("[" + String(start_frame_val) + "]")

    var signal_handler = Python.evaluate("""
def _make_signal_handler(pause_requested):
    def handler(signum, frame):
        import signal as sig
        pause_requested[0] = True
        name = sig.Signals(signum).name
        print(f"\\nReceived {name}; finalizing partial output before exit...")
    return handler
""", file=True)

    var handler = signal_handler._make_signal_handler(pause_requested)
    signal.signal(signal.SIGUSR1, handler)
    signal.signal(signal.SIGINT, handler)

    # Detection model setup
    var detection_model_name = coerce_detection_model_name(args.detection_model)
    var has_explicit_path = String(args.detection_model_path.strip()) != ""
    if not has_explicit_path:
        var available = discover_available_detection_models()
        if len(available) > 0 and not (is_rfdetr_model(detection_model_name) or is_yolo_model(detection_model_name)):
            print("Warning: detection model '" + detection_model_name + "' not found in model_weights/.")

    var det_path = pathlib.Path(args.detection_model_path) if has_explicit_path else detection_model_weights_path(detection_model_name)

    # Validate codec
    var codec = args.codec.lower()
    var valid_codecs = Set[String]()
    valid_codecs.add("hevc")
    valid_codecs.add("h264")
    valid_codecs.add("libx264")
    if not codec in valid_codecs:
        raise Error("Unsupported codec: " + args.codec + " (supported: hevc, h264, libx264)")

    # Parse encoder settings
    var enc_settings = parse_encoder_settings(args.encoder_settings)
    enc_settings = validate_encoder_settings(enc_settings^)

    # Validate parameters
    var batch_size = args.batch_size
    if batch_size <= 0:
        raise Error("--batch-size must be > 0")

    var max_clip_size = args.max_clip_size
    if max_clip_size <= 0:
        raise Error("--max-clip-size must be > 0")

    var temporal_overlap = args.temporal_overlap
    if temporal_overlap < 0:
        raise Error("--temporal-overlap must be >= 0")
    if temporal_overlap >= max_clip_size:
        raise Error("--temporal-overlap must be < --max-clip-size")
    if temporal_overlap > 0 and (2 * temporal_overlap) >= max_clip_size:
        raise Error("--temporal-overlap must satisfy 2*--temporal-overlap < --max-clip-size")

    # Device selection
    var device = get_available_device("auto") if args.device == "auto" else torch.device(args.device)

    var (device_ok, device_info) = validate_device_for_processing(device)
    if not device_ok:
        print("Error: " + device_info)
        sys_mod.exit(1)
    print("Selected device: " + device_info)

    # Apply Apple Silicon auto-tune
    var (tuned_batch, tuned_clip, tuned_overlap, tuned_candidates) = _apply_apple_silicon_auto_tune(
        device, batch_size, max_clip_size, temporal_overlap,
        args.detection_max_candidates, args.apple_silicon_auto_tune,
    )

    # Check detection model path
    var detection_model = PythonObject()
    if not Bool(det_path.exists()):
        print("Warning: Detection model weights not found at " + String(py=det_path) + ". Using mock detection model.")
        var create_mock = Python.evaluate("""
def _create_mock(device):
    from jasna.mock_detection import create_mock_detection_model
    return create_mock_detection_model(device)
""", file=True)

        detection_model = create_mock._create_mock(device)

    # Check restoration model path
    var restoration_model_path = args.restoration_model_path
    var rest_path = pathlib.Path(restoration_model_path)
    if not Bool(rest_path.exists()):
        print("Warning: Restoration model weights not found at " + restoration_model_path + ". Using mock restoration model.")
        restoration_model_path = ""

    # NVIDIA-specific checks (skip on Apple Silicon)
    if not is_apple_silicon() and String(py=device.type) == "cuda":
        var (driver_ok, driver_info) = check_gpu_driver_version()
        if not driver_ok:
            print("Error: GPU driver version check failed: " + driver_info)
            print("Please update your NVIDIA driver to version 590 or newer.")
            sys_mod.exit(1)

    # Validate start frame and duration
    if args.start_frame is not None and args.start_frame.value() < 0:
        print("--start-frame must be >= 0")
        sys_mod.exit(1)
    if args.duration_frames is not None and args.duration_frames.value() <= 0:
        print("--duration-frames must be > 0")
        sys_mod.exit(1)

    var fp16 = args.fp16
    var detection_score_threshold = args.detection_score_threshold
    if detection_score_threshold < 0.0 or detection_score_threshold > 1.0:
        raise Error("--detection-score-threshold must be in [0, 1]")

    if args.restoration_model_name != "basicvsrpp":
        raise Error("Unsupported restoration model: " + args.restoration_model_name)

    # Engine compilation (NVIDIA only)
    var compile_mod = Python.evaluate("""
def _ensure_engines(device, fp16, compile_basicvsrpp, restoration_model_path,
                    max_clip_size, detection_model_name, detection_model_path,
                    batch_size, secondary_name):
    try:
        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        result = ensure_engines_compiled(EngineCompilationRequest(
            device=String(device),
            fp16=fp16,
            basicvsrpp=compile_basicvsrpp,
            basicvsrpp_model_path=restoration_model_path,
            basicvsrpp_max_clip_size=max_clip_size,
            detection=True,
            detection_model_name=detection_model_name,
            detection_model_path=String(detection_model_path),
            detection_batch_size=batch_size,
            unet4x=(secondary_name == "unet-4x"),
        ))
        return result
    except:
        # If compilation fails, continue without TensorRT
        class _MockResult:
            use_basicvsrpp_tensorrt = False
        return _MockResult()
""", file=True)
    var compile_result = compile_mod._ensure_engines(device, fp16, args.compile_basicvsrpp, restoration_model_path,
     tuned_clip, detection_model_name, det_path, tuned_batch,
     args.secondary_restoration)

    var use_tensorrt = Bool(compile_result.use_basicvsrpp_tensorrt)

    # Build restoration pipeline
    var secondary_name = args.secondary_restoration.lower()

    var build_pipeline = Python.evaluate("""
def _build_pipeline(device, fp16, restoration_model_path, max_clip_size,
                    use_tensorrt, secondary_name, denoise, denoise_step,
                    tvai_ffmpeg_path, tvai_model, tvai_scale, tvai_args, tvai_workers,
                    rtx_scale, rtx_quality, rtx_denoise, rtx_deblur):
    import torch
    from contextlib import nullcontext

    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.mock_restoration import MockBasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    # Build secondary restorer
    secondary_restorer = None
    if secondary_name == "tvai":
        from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
        tvai_full_args = f"model={tvai_model}:scale={tvai_scale}:{tvai_args}"
        secondary_restorer = TvaiSecondaryRestorer(
            ffmpeg_path=tvai_ffmpeg_path,
            tvai_args=tvai_full_args,
            scale=tvai_scale,
            num_workers=tvai_workers,
        )
    elif secondary_name == "unet-4x":
        from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer
        secondary_restorer = Unet4xSecondaryRestorer(device=device, fp16=fp16)
    elif secondary_name == "rtx-super-res":
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer
        dn = None if rtx_denoise == "none" else rtx_denoise
        db = None if rtx_deblur == "none" else rtx_deblur
        secondary_restorer = RtxSuperresSecondaryRestorer(
            device=device, scale=rtx_scale, quality=rtx_quality,
            denoise=dn, deblur=db,
        )

    denoise_strength = DenoiseStrength(denoise.lower())
    denoise_step_val = DenoiseStep(denoise_step.lower())

    restorer_cls = BasicvsrppMosaicRestorer if restoration_model_path else MockBasicvsrppMosaicRestorer
    restorer = restorer_cls(
        checkpoint_path=restoration_model_path if restoration_model_path else "mock",
        device=device,
        max_clip_size=max_clip_size,
        use_tensorrt=use_tensorrt,
        fp16=fp16,
    )

    return RestorationPipeline(
        restorer=restorer,
        secondary_restorer=secondary_restorer,
        denoise_strength=denoise_strength,
        denoise_step=denoise_step_val,
    ), restorer, secondary_restorer
""", file=True)


    var pipeline_result = build_pipeline._build_pipeline(
        device, fp16, restoration_model_path, tuned_clip, use_tensorrt,
        secondary_name, args.denoise, args.denoise_step,
        args.tvai_ffmpeg_path, args.tvai_model, args.tvai_scale, args.tvai_args, args.tvai_workers,
        args.rtx_scale, args.rtx_quality, args.rtx_denoise, args.rtx_deblur,
    )
    var restoration_pipeline = pipeline_result[0]
    var restorer = pipeline_result[1]
    var secondary_restorer = pipeline_result[2]

    # Build detection model if not already set
    if detection_model is None:
        var build_detection = Python.evaluate("""
def _build_detection(det_name, det_path, batch_size, device, score_threshold, max_candidates, fp16):
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    from jasna.mosaic.yolo import YoloMosaicDetectionModel
    from jasna.mosaic.detection_registry import is_rfdetr_model, is_yolo_model

    if det_path.exists():
        if is_rfdetr_model(det_name):
            return RfDetrMosaicDetectionModel(
                onnx_path=det_path, batch_size=batch_size, device=device,
                score_threshold=score_threshold, fp16=fp16,
            )
        elif is_yolo_model(det_name):
            return YoloMosaicDetectionModel(
                model_path=det_path, batch_size=batch_size, device=device,
                score_threshold=score_threshold, max_nms=max_candidates, fp16=fp16,
            )
    return None
""", file=True)
        detection_model = build_detection._build_detection(detection_model_name, det_path, tuned_batch, device,
         detection_score_threshold, tuned_candidates, fp16)

        if detection_model is None:
            var create_mock = Python.evaluate("""
def _create_mock(device):
    from jasna.mock_detection import create_mock_detection_model
    return create_mock_detection_model(device)
""", file=True)

            detection_model = create_mock._create_mock(device)

    # Build encoder settings dict
    var enc_dict = Dict[String, PythonObject]()
    var enc_keys = List[String]()
    for k in enc_settings.keys():
        enc_keys.append(k)
    for k in enc_keys:
        enc_dict[k] = enc_settings[k]

    var working_dir = PythonObject() if args.working_directory == "" else pathlib.Path(args.working_directory)

    # Create pipeline
    var make_pipeline = Python.evaluate("""
def _make_pipeline(input_video, output_video, detection_model_name, detection_model_path,
                   detection_score_threshold, detection_max_candidates,
                   restoration_pipeline,
                   codec, encoder_settings, batch_size, device, max_clip_size,
                   temporal_overlap, enable_crossfade, fp16, no_progress,
                   working_directory, start_frame, duration_frames,
                   current_frame_shared, pause_requested, status_file_path):
    from pathlib import Path
    from jasna.pipeline import Pipeline

    input_path = Path(input_video) if input_video else Path("__streaming__")
    output_path = Path(output_video) if output_video else input_path.with_stem(input_path.stem + "_out")

    return Pipeline(
        input_video=input_path,
        output_video=output_path,
        detection_model_name=detection_model_name,
        detection_model_path=detection_model_path,
        detection_score_threshold=detection_score_threshold,
        detection_max_candidates=detection_max_candidates,
        restoration_pipeline=restoration_pipeline,
        codec=codec,
        encoder_settings=encoder_settings,
        batch_size=batch_size,
        device=device,
        max_clip_size=max_clip_size,
        temporal_overlap=temporal_overlap,
        enable_crossfade=enable_crossfade,
        fp16=fp16,
        disable_progress=no_progress,
        working_directory=working_directory,
        start_frame=start_frame,
        duration_frames=duration_frames,
        current_frame_shared=current_frame_shared,
        pause_requested=pause_requested,
        status_file_path=Path(status_file_path),
    )
""", file=True)


    var pipeline = PythonObject()
    try:
        if is_streaming and args.input == "":
            # Streaming mode without input — wait for video selection
            var HlsServer = Python.evaluate("""
def _create_hls(segment_duration, port):
    from jasna.streaming import HlsStreamingServer
    return HlsStreamingServer(segment_duration=segment_duration, port=port)
""", file=True)

            var hls_server = HlsServer._create_hls(args.stream_segment_duration, args.stream_port)
            hls_server.start()

            if not args.no_browser:
                var webbrowser = Python.import_module("webbrowser")
                webbrowser.open("http://localhost:" + String(args.stream_port) + "/")

            try:
                while True:
                    var video_path = hls_server.wait_for_video()
                    pipeline = make_pipeline._make_pipeline(
                        PythonObject(String(video_path)),
                        PythonObject(""),
                        PythonObject(detection_model_name), det_path,
                        PythonObject(detection_score_threshold), PythonObject(tuned_candidates),
                        restoration_pipeline,
                        PythonObject(codec), _mojo_dict_to_py(enc_dict),
                        PythonObject(tuned_batch), device,
                        PythonObject(tuned_clip), PythonObject(tuned_overlap),
                        PythonObject(args.enable_crossfade), PythonObject(fp16),
                        PythonObject(args.no_progress), working_dir,
                        PythonObject(args.start_frame.value()) if args.start_frame is not None else PythonObject(),
                        PythonObject(args.duration_frames.value()) if args.duration_frames is not None else PythonObject(),
                        current_frame_shared, pause_requested,
                        PythonObject(args.status_file),
                    )
                    pipeline.input_video = pathlib.Path(video_path)
                    try:
                        pipeline.run_streaming(
                            hls_server=hls_server,
                            segment_duration=args.stream_segment_duration,
                        )
                    except e:
                        var err_name = String(e)
                        if "UnsupportedColorspace" in err_name:
                            print("Error: " + err_name)
                    hls_server.unload_video()
            except KeyboardInterrupt:
                pass
            finally:
                hls_server.stop()

        elif is_streaming:
            pipeline = make_pipeline._make_pipeline(
                PythonObject(args.input),
                PythonObject(args.output),
                PythonObject(detection_model_name), det_path,
                PythonObject(detection_score_threshold), PythonObject(tuned_candidates),
                restoration_pipeline,
                PythonObject(codec), _mojo_dict_to_py(enc_dict),
                PythonObject(tuned_batch), device,
                PythonObject(tuned_clip), PythonObject(tuned_overlap),
                PythonObject(args.enable_crossfade), PythonObject(fp16),
                PythonObject(args.no_progress), working_dir,
                PythonObject(args.start_frame.value()) if args.start_frame is not None else PythonObject(),
                PythonObject(args.duration_frames.value()) if args.duration_frames is not None else PythonObject(),
                current_frame_shared, pause_requested,
                PythonObject(args.status_file),
            )
            if not args.no_browser:
                var webbrowser = Python.import_module("webbrowser")
                webbrowser.open("http://localhost:" + String(args.stream_port) + "/")
            pipeline.run_streaming(
                port=args.stream_port,
                segment_duration=args.stream_segment_duration,
            )
        else:
            # Normal processing mode
            pipeline = make_pipeline._make_pipeline(
                PythonObject(args.input),
                PythonObject(args.output),
                PythonObject(detection_model_name), det_path,
                PythonObject(detection_score_threshold), PythonObject(tuned_candidates),
                restoration_pipeline,
                PythonObject(codec), _mojo_dict_to_py(enc_dict),
                PythonObject(tuned_batch), device,
                PythonObject(tuned_clip), PythonObject(tuned_overlap),
                PythonObject(args.enable_crossfade), PythonObject(fp16),
                PythonObject(args.no_progress), working_dir,
                PythonObject(args.start_frame.value()) if args.start_frame is not None else PythonObject(),
                PythonObject(args.duration_frames.value()) if args.duration_frames is not None else PythonObject(),
                current_frame_shared, pause_requested,
                PythonObject(args.status_file),
            )
            pipeline.run()

    except e:
        var err_name = String(e)
        if "UnsupportedColorspace" in err_name:
            print("Error: " + err_name)
            sys_mod.exit(1)
        raise e^
    finally:
        if pipeline is not None:
            pipeline.close()
        restorer.close()
        if secondary_restorer is not None and Bool(py=_hasattr(secondary_restorer, "close")):
            secondary_restorer.close()


# ============================================================================
# Entry point
# ============================================================================

def main_entry() raises:
    """Entry point that handles multiprocessing and bootstrap."""
    var os_mod = Python.import_module("os")
    var sys_mod = Python.import_module("sys")
    var multiprocessing = Python.import_module("multiprocessing")

    # Set environment defaults
    os_mod.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    if sys_mod.platform == "win32":
        os_mod.environ.setdefault("OMP_WAIT_POLICY", "passive")

    # Handle compile-engines subcommand
    from std.sys import argv as mojo_argv
    var raw_argv = List[String]()
    for a in mojo_argv():
        raw_argv.append(String(a))
    # Skip leading -- separator
    var start_idx = 1
    if len(raw_argv) > 1 and raw_argv[1] == "--":
        start_idx = 2
    var cli_argv = List[String]()
    for i in range(start_idx, len(raw_argv)):
        cli_argv.append(raw_argv[i])
    var py_argv = Python.list([PythonObject(raw_argv[0])] + [PythonObject(a) for a in cli_argv])
    sys_mod.argv = py_argv
    var argv = sys_mod.argv
    if len(argv) >= 3 and argv[1] == "--compile-engines":
        var compile_fn = Python.evaluate("""
def _subprocess_compile(json_str):
    from jasna.engine_compiler import EngineCompilationRequest, _subprocess_compile
    _subprocess_compile(EngineCompilationRequest.from_json(json_str))
""", file=True)

        compile_fn._subprocess_compile(argv[2])
        sys_mod.exit(0)

    # Multiprocessing guard
    var main_pid = os_mod.environ.get("JASNA_MAIN_PID")
    if main_pid and String(py=os_mod.getpid()) != String(py=main_pid):
        if len(argv) < 2 or argv[1] != "--multiprocessing-fork":
            sys_mod.exit(0)

    if multiprocessing.parent_process() is not None:
        sys_mod.exit(0)

    os_mod.environ["JASNA_MAIN_PID"] = String(py=os_mod.getpid())

    # Bootstrap
    var bootstrap = Python.evaluate("""
def _bootstrap():
    from jasna.bootstrap import sanitize_sys_path_for_local_dev
    from pathlib import Path
    import os
    if not getattr(__import__('sys'), 'frozen', False):
        sanitize_sys_path_for_local_dev(Path(os.getcwd()).resolve())
""", file=True)

    bootstrap._bootstrap()

    multiprocessing.freeze_support()
    _run_cli()
