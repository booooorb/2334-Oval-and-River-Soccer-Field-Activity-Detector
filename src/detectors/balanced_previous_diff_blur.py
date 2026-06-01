"""Balanced blurred previous-frame subtraction detector."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageFilter

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    detect_against_reference,
    expand_cutoffs,
    image_bytes,
    load_gray,
    same_capture_session,
)

MODEL_NAME = "balanced_previous_diff_blur"


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
    del windows
    results: list[DetectorResult] = []

    for threshold in thresholds:
        inactive_reference: bytes | None = None
        inactive_reference_frame: LabeledFrame | None = None

        for frame in frames:
            image = load_gray(frame.masked_path).filter(ImageFilter.GaussianBlur(radius=blur_radius))
            current = image_bytes(image)
            width, height = image.size

            if inactive_reference is not None and inactive_reference_frame is not None and same_capture_session(inactive_reference_frame, frame):
                features = detect_against_reference(
                    MODEL_NAME,
                    frame,
                    current,
                    inactive_reference,
                    width,
                    height,
                    threshold,
                    1,
                    min_blob_area,
                    max_shift_pixels,
                    blur_radius,
                    foreground_root,
                )
                results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, 1, features))

            if frame.label == "inactive":
                inactive_reference = current
                inactive_reference_frame = frame

    return results
