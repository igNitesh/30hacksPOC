"""Create timestamped review intervals for potentially explicit video content."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import onnxruntime
from nudenet import NudeDetector
from nudenet.nudenet import _postprocess, _read_image


DEFAULT_EXPLICIT_CLASSES = (
    "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    # NudeNet does not currently emit KISSING, but keeping it here allows a
    # replacement detector with that class to use the same configuration.
    "KISSING",
)


@dataclass(frozen=True)
class Settings:
    sample_seconds: float
    chunk_size: int
    threshold: float
    merge_gap_seconds: float
    explicit_classes: tuple[str, ...]
    use_cuda: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find potentially explicit video intervals with NudeNet."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=Path("screening_report.json"))
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--merge-gap-seconds", type=float, default=2.0)
    parser.add_argument("--class-name", action="append", dest="class_names")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA provider")
    parser.add_argument("--list-classes", action="store_true")
    return parser.parse_args()


def build_settings(arguments: argparse.Namespace) -> Settings:
    if arguments.sample_seconds <= 0:
        raise ValueError("--sample-seconds must be greater than zero")
    if arguments.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")
    if not 0 < arguments.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    return Settings(
        sample_seconds=arguments.sample_seconds,
        chunk_size=arguments.chunk_size,
        threshold=arguments.threshold,
        merge_gap_seconds=max(0.0, arguments.merge_gap_seconds),
        explicit_classes=tuple(arguments.class_names or DEFAULT_EXPLICIT_CLASSES),
        use_cuda=not arguments.cpu,
    )


def read_video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0:
        raise RuntimeError("Video does not expose a valid FPS value")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps if frame_count else None,
    }


def iter_frame_chunks(
    video_path: Path, fps: float, settings: Settings
) -> Iterable[list[tuple[float, Any]]]:
    capture = cv2.VideoCapture(str(video_path))
    frame_step = max(1, round(fps * settings.sample_seconds))
    chunk: list[tuple[float, Any]] = []
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                chunk.append((frame_index / fps, frame))
                if len(chunk) == settings.chunk_size:
                    yield chunk
                    chunk = []
            frame_index += 1
        if chunk:
            yield chunk
    finally:
        capture.release()


def create_detector(use_cuda: bool) -> tuple[NudeDetector, str]:
    available = onnxruntime.get_available_providers()
    provider = "CUDAExecutionProvider" if use_cuda and "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    if use_cuda and provider == "CPUExecutionProvider":
        print("CUDA provider unavailable; using CPU.")

    nudenet_module = import_module("nudenet.nudenet")
    model_path = Path(nudenet_module.__file__).with_name("320n.onnx")
    detector = NudeDetector.__new__(NudeDetector)
    detector.onnx_session = onnxruntime.InferenceSession(
        str(model_path),
        providers=[provider],
    )
    model_inputs = detector.onnx_session.get_inputs()
    detector.input_width = 320
    detector.input_height = 320
    detector.input_name = model_inputs[0].name
    return detector, provider


def detect_frame(detector: NudeDetector, frame: Any) -> list[dict[str, Any]]:
    rgba_frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGBA)
    preprocessed, x_ratio, y_ratio, x_pad, y_pad, width, height = _read_image(
        rgba_frame, detector.input_width
    )
    outputs = detector.onnx_session.run(None, {detector.input_name: preprocessed})
    return _postprocess(
        outputs,
        x_pad,
        y_pad,
        x_ratio,
        y_ratio,
        width,
        height,
        detector.input_width,
        detector.input_height,
    )


def find_detections(
    video_path: Path, metadata: dict[str, Any], settings: Settings, detector: NudeDetector
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for chunk in iter_frame_chunks(video_path, metadata["fps"], settings):
        for timestamp, frame in chunk:
            matches = [
                item
                for item in detect_frame(detector, frame)
                if item.get("class") in settings.explicit_classes
                and float(item.get("score", 0)) >= settings.threshold
            ]
            if matches:
                detections.append(
                    {
                        "timestamp": timestamp,
                        "classes": sorted({item["class"] for item in matches}),
                        "confidence": max(float(item["score"]) for item in matches),
                    }
                )
    return detections


def merge_detections(
    detections: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for detection in detections:
        start = detection["timestamp"]
        end = start + settings.sample_seconds
        if segments and start <= segments[-1]["end_seconds"] + settings.merge_gap_seconds:
            segment = segments[-1]
            segment["end_seconds"] = max(segment["end_seconds"], end)
            segment["classes"] = sorted(set(segment["classes"]) | set(detection["classes"]))
            segment["max_confidence"] = max(segment["max_confidence"], detection["confidence"])
        else:
            segments.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "classes": detection["classes"],
                    "max_confidence": detection["confidence"],
                }
            )
    for segment in segments:
        segment["duration_seconds"] = segment["end_seconds"] - segment["start_seconds"]
    return segments


def write_report(
    output_path: Path,
    video_path: Path,
    metadata: dict[str, Any],
    settings: Settings,
    provider: str,
    segments: list[dict[str, Any]],
) -> None:
    report = {
        "video": {"path": str(video_path), **metadata},
        "settings": {
            "sample_seconds": settings.sample_seconds,
            "chunk_size": settings.chunk_size,
            "threshold": settings.threshold,
            "merge_gap_seconds": settings.merge_gap_seconds,
            "explicit_classes": list(settings.explicit_classes),
        },
        "provider": provider,
        "segments": segments,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    if not arguments.video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {arguments.video}")
    settings = build_settings(arguments)
    detector, provider = create_detector(settings.use_cuda)
    if arguments.list_classes:
        print("Use NudeNet's model labels with --class-name to filter detections.")
        return
    metadata = read_video_metadata(arguments.video)
    detections = find_detections(arguments.video, metadata, settings, detector)
    segments = merge_detections(detections, settings)
    write_report(arguments.output, arguments.video, metadata, settings, provider, segments)
    print(f"Wrote {len(segments)} segment(s) to {arguments.output}")


if __name__ == "__main__":
    main()
