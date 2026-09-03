# Family-Safe Video Checker

A function-based local utility that samples long videos, processes frames in bounded chunks, and writes JSON timestamps for potentially explicit segments detected by NudeNet.

## Setup

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an RTX 2050, install a compatible NVIDIA driver and CUDA/cuDNN runtime for the installed ONNX Runtime GPU package. The script falls back to CPU if the CUDA execution provider cannot initialize.

## Run

```powershell
.\.venv\Scripts\python.exe video_screen.py input.mp4 --output report.json
```

Adjust sampling and memory use for long videos:

```powershell
.\.venv\Scripts\python.exe video_screen.py input.mp4 --sample-seconds 1.5 --chunk-size 8 --threshold 0.70
```

The JSON report contains merged `start_seconds`, `end_seconds`, `duration_seconds`, detected classes, and maximum confidence. Detection is probabilistic and should be reviewed manually.

## Docker

Place a video at `input/video.mp4`, then run:

```powershell
docker compose run --rm video-screen
```

The report is written to `output/report.json`. The default service uses CPU and works on macOS. The application also falls back to CPU if the CUDA provider is unavailable inside the container.

The default Compose file is portable and works on macOS using CPU. On a Linux or Windows host with NVIDIA Container Toolkit, enable the GPU override:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm video-screen
```

The `Dockerfile` includes CUDA libraries, while `compose.gpu.yaml` is the only file that requests an NVIDIA device.

To use a different input file or settings, override the command:

```powershell
docker compose run --rm video-screen /app/input/video.mp4 --output /app/output/report.json --sample-seconds 1.5 --chunk-size 8 --threshold 0.70
```
