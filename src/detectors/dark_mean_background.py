"""Dark rolling mean-background subtraction detector."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    detect_dark_against_reference,
    expand_cutoffs,
    image_bytes,
    load_gray,
    same_capture_session,
)

MODEL_NAME = "dark_mean_background"


def mean_background(images: list[bytes]) -> bytes:
    pixel_count = len(images[0])
    return bytes(round(sum(image[index] for image in images) / len(images)) for index in range(pixel_count))


def evaluate(
    frames: list[LabeledFrame],
    thresholds: list[int],
    cutoffs: list[float],
    windows: list[int],
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path = DEFAULT_FOREGROUND_ROOT,
) -> list[DetectorResult]:
    del blur_radius
    results: list[DetectorResult] = []

    for threshold in thresholds:
        for window in windows:
            history: deque[bytes] = deque(maxlen=window)
            previous_frame: LabeledFrame | None = None

            for frame in frames:
                if previous_frame is not None and not same_capture_session(previous_frame, frame):
                    history.clear()

                image = load_gray(frame.masked_path)
                current = image_bytes(image)
                width, height = image.size

                if len(history) == window:
                    features = detect_dark_against_reference(
                        MODEL_NAME,
                        frame,
                        current,
                        mean_background(list(history)),
                        width,
                        height,
                        threshold,
                        window,
                        min_blob_area,
                        max_shift_pixels,
                        0,
                        foreground_root,
                    )
                    results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, window, features))

                history.append(current)
                previous_frame = frame

    return results
