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

Optional (NVIDIA only):
- TensorRT, `python_vali`, `PyNvVideoCodec`

## Building

```bash
mojo build jasna-mojo -o jasna
```

## Usage

```bash
# Process a video
jasna --input video.mp4 --output restored.mp4

# Auto device selection
jasna --input video.mp4 --output restored.mp4 --device auto

# Apple Silicon
jasna --input video.mp4 --output restored.mp4 --device mps

# Streaming mode
jasna --stream

# See all options
jasna --help
```

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

Same as original Jasna project.
