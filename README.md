# Jasna-Mojo

Mojo implementation of [Jasna](https://github.com/Kruk2/jasna) — JAV model restoration tool.

This is a Mojo port of the Python-based Jasna project, designed to run on **multiple hardware platforms** (NVIDIA CUDA, Apple Silicon MPS, and CPU).

## Architecture

- **Native Mojo** — Core pipeline logic, data structures, tracking, blending math, queue management, CLI
- **Python interop** — PyTorch (ML models), PyAV (video I/O), OpenCV, ffprobe/mkvmerge (external tools)

This hybrid approach gives us Mojo's performance and type safety for orchestration while leveraging the mature Python ML/video ecosystem.

## Multi-Hardware Support

Unlike the original Jasna (NVIDIA-only), jasna-mojo supports:
- **NVIDIA CUDA** — Full GPU acceleration (TensorRT optional)
- **Apple Silicon MPS** — Metal Performance Shaders via PyTorch
- **CPU** — Fallback for any platform

Device selection is automatic (`--device auto`) or manual (`--device cuda:0`, `--device mps`, `--device cpu`).

## Requirements

- [Mojo](https://docs.modular.com/mojo/) compiler
- Python 3.9+
- `torch>=2.1.0`, `torchvision>=0.15.0`
- `av>=8.0.0` (PyAV)
- `numpy`, `opencv-python`, `tqdm`
- `ffmpeg` + `ffprobe` (major version 8)
- `mkvmerge` (MKVToolNix)
- Model weights in a local `model_weights/` directory, or paths supplied with
  `--restoration-model-path` and `--detection-model-path`

Optional (NVIDIA only):
- TensorRT, `python_vali`, `PyNvVideoCodec`

## Building

```bash
mojo build jasna/__main__.mojo -o jasna_bin
```

If Mojo or Python are installed outside your `PATH`, set:

```bash
MOJO_BIN=/path/to/mojo PYTHON_BIN=/path/to/python3 ./run.sh --help
```

Python import paths can be extended with `JASNA_PYTHON_PATH`. If you need to
reuse modules from the original Python Jasna checkout, set
`JASNA_ORIGINAL_JASNA_PATH=/path/to/jasna`. Only point these variables at
directories you trust; Python code imported from those paths runs with your
user permissions.

## Usage

```bash
# Process a video
./jasna_bin --input video.mp4 --output restored.mp4

# Auto device selection
./jasna_bin --input video.mp4 --output restored.mp4 --device auto

# Apple Silicon
./jasna_bin --input video.mp4 --output restored.mp4 --device mps

# Streaming mode
./jasna_bin --stream

# See all options
./jasna_bin --help
```

Missing model weights are treated as an error by default. For smoke tests only,
you can pass `--allow-mock-models` to run placeholder detection/restoration
models.

## Benchmark

Reference run on a 30 second 1280x720 H.264 sample:

- Machine: Mac mini with Apple M4 Pro, 12-core CPU, 16-core GPU, 64 GB memory
- OS: macOS 26.5.1
- Device selected by `--device auto`: Apple Silicon GPU via PyTorch MPS
- Detection model: `lada-yolo-v4`
- Restoration model: BasicVSR++ generic v1.2
- Command options: `--log-level info --codec h264`
- Input: 24 fps, 30.083 seconds, 769 reported source frames
- Output: 24 fps, 30.000 seconds, 720 frames
- Wall time: 55.90 seconds
- End-of-run throughput: 18.3 fps
- Reported memory: MPS VRAM max 1566 MiB, average 1410 MiB; RAM max 4991 MiB,
  average 1858 MiB

Benchmark numbers depend on model weights, PyTorch/torchvision builds, decoder
and encoder availability, clip content, and thermal state.

## Security and Privacy

Jasna-Mojo runs locally and does not include telemetry or cloud upload code.
External tools are invoked with argument lists rather than through a shell.

Model weights are executable trust boundaries. PyTorch `.pt` / `.pth`
checkpoints and TensorRT engines can execute code or native kernels when loaded
by the underlying Python stack, so use only weights from sources you trust.

The current working directory is not added to the front of Python's import path.
Additional Python module paths must be configured explicitly with
`JASNA_PYTHON_PATH` or `JASNA_ORIGINAL_JASNA_PATH`, and those paths should be
treated as trusted code.

Video inputs, restored outputs, HLS segments, working directories, and
`--status-file` contents may contain sensitive filenames or video data. Store
them on trusted local disks and delete temporary working data when processing is
complete. Streaming mode opens a local web UI by default; use `--no-browser` if
you do not want the browser opened automatically, and avoid exposing the stream
port on untrusted networks.

## Testing

The current Mojo toolchain used for this project does not expose a `mojo test`
subcommand. Use the build command above as the minimum verification step, and
add executable smoke tests as the toolchain support matures.

## Project Structure

```
jasna-mojo/
├── jasna/
│   ├── __init__.mojo          — Version info
│   ├── main.mojo              — CLI entry point & argument parsing
│   ├── pipeline.mojo          — Main pipeline orchestration
│   ├── pipeline_items.mojo    — Data structures (FrameMeta, ClipRestoreItem, etc.)
│   ├── pipeline_overlap.mojo  — Overlap/crossfade math (native Mojo)
│   ├── pipeline_processing.mojo — Frame batch processing
│   ├── pipeline_threads.mojo  — Threaded pipeline loops
│   ├── frame_queue.mojo       — Thread-safe frame queue (native Mojo)
│   ├── blend_buffer.mojo      — Blend buffer for restoration compositing
│   ├── crop_buffer.mojo       — Crop extraction & bbox expansion
│   ├── tensor_utils.mojo      — Tensor utility functions
│   ├── device_utils.mojo      — Device selection & validation
│   ├── os_utils.mojo          — OS/executable utilities
│   ├── progressbar.mojo       — Progress bar
│   ├── vram_offloader.mojo    — VRAM management
│   ├── tracking/
│   │   ├── clip_tracker.mojo  — Clip tracking with IoU matching (native Mojo)
│   │   └── blending.mojo      — Blend mask creation
│   ├── mosaic/
│   │   ├── detections.mojo    — Detection data structures
│   │   ├── detection_registry.mojo — Model discovery
│   │   ├── rfdetr.mojo        — RF-DETR detection model
│   │   └── yolo.mojo          — YOLO detection model
│   ├── restorer/
│   │   ├── restoration_pipeline.mojo — Restoration orchestration
│   │   ├── basicvsrpp.mojo    — BasicVSR++ restorer
│   │   ├── denoise.mojo       — Spatial denoising
│   │   └── secondary_restorer.mojo — Secondary restoration protocol
│   ├── media/
│   │   ├── video_metadata.mojo — Video metadata parsing
│   │   ├── video_reader.mojo   — Video reader factory
│   │   └── video_encoder.mojo  — Video encoder factory
│   └── streaming/
│       └── streaming_pipeline.mojo — HLS streaming
└── tests/
    └── ...
```

## License

AGPL-3.0, matching the upstream Jasna project.
