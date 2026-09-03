---
name: adult-video-screening
description: Process videos in chunks with NudeNet to identify potentially explicit segments and write timestamped JSON reports. Use when building, running, or improving a local video content-screening workflow with CUDA, confidence thresholds, frame sampling, or family-safe review timestamps.
---

# Adult Video Screening

## Workflow

1. Use the project virtual environment in `.venv`.
2. Validate the input path and open the video with OpenCV.
3. Sample frames at a configurable interval instead of decoding every frame.
4. Process sampled frames in bounded chunks to control memory use on long videos.
5. Run NudeNet with CUDA when `CUDAExecutionProvider` is available, otherwise use CPU.
6. Keep only configured explicit classes above the confidence threshold.
7. Merge nearby detections into intervals and write JSON containing timestamps, duration, classes, and confidence.
8. Treat the output as a review aid: model misses and false positives are possible.

## Commands

Create or refresh the environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run a scan:

```powershell
.\.venv\Scripts\python.exe video_screen.py input.mp4 --output report.json
```

Useful options include `--sample-seconds`, `--chunk-size`, `--threshold`, and repeated `--class-name` values. Use `--list-classes` to inspect NudeNet labels exposed by the installed model.

## Implementation rules

- Keep the program function-based: separate configuration, video metadata, frame sampling, inference, interval merging, and report writing.
- Do not load a whole video into memory.
- Preserve source timestamps by basing time on frame index and FPS.
- Include model/provider details in the report for reproducibility.
- Fail with an actionable message when OpenCV cannot open the video or CUDA was requested but unavailable.
- Avoid logging image contents or writing extracted frames unless explicitly requested.

## Output contract

The report is JSON with `video`, `settings`, `provider`, and `segments` fields. Each segment has `start_seconds`, `end_seconds`, `duration_seconds`, `classes`, and `max_confidence`.
