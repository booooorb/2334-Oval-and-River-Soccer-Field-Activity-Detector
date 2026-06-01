"""Balanced sample-based background detector inspired by PBAS."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    aligned_sample_references,
    expand_cutoffs,
    features_from_blob_mask,
    image_bytes,
    load_gray,
    same_capture_session,
)

MODEL_NAME = "balanced_pbas"


def sample_background_mask(
    current: bytes,
    samples: list[bytes],
    valid_mask: bytes,
    threshold: int,
) -> list[bool]:
    close_radius = max(8, round(threshold * 0.75))
    change_threshold = max(6, round(threshold * 0.5))
    min_matches = 2 if len(samples) >= 3 else 1
    midpoint = len(samples) // 2
    mask = []

    for index, pixel in enumerate(current):
        if not valid_mask[index]:
            mask.append(False)
            continue

        values = sorted(sample[index] for sample in samples)
        matches = sum(1 for value in values if abs(pixel - value) <= close_radius)
        background_level = values[midpoint]
        mask.append(matches < min_matches and abs(background_level - pixel) > change_threshold)

    return mask


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
                    samples, valid_mask, shift_x, shift_y = aligned_sample_references(
                        current,
                        list(history),
                        width,
                        height,
                        max_shift_pixels,
                    )
                    mask = sample_background_mask(current, samples, valid_mask, threshold)
                    features = features_from_blob_mask(
                        MODEL_NAME,
                        frame,
                        mask,
                        width,
                        height,
                        threshold,
                        window,
                        min_blob_area,
                        max_shift_pixels,
                        0,
                        foreground_root,
                        shift_x,
                        shift_y,
                    )
                    results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, window, features))

                history.append(current)
                previous_frame = frame

    return results
