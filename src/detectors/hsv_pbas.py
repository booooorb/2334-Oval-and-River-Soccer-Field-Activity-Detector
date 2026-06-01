"""HSV sample-based background detector inspired by PBAS."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    MASK_BLACK_THRESHOLD,
    expand_cutoffs,
    features_from_hsv_mask,
    field_mask_bytes,
    same_capture_session,
    suppress_black_mask_pixels_hsv,
)
from src.detectors.hsv_blue_diff import hue_distance
from src.detectors.hsv_previous_diff import load_hsv
from src.detectors.hsv_previous_diff import HUE_WEIGHT, SATURATION_WEIGHT, VALUE_WEIGHT
from src.detectors.hsv_previous_diff import HUE_SATURATION_MIN

MODEL_NAME = "hsv_pbas"


def hsv_distance_squared(first_hsv: bytes, second_hsv: bytes, offset: int) -> int:
    hue_confidence = 1 if max(first_hsv[offset + 1], second_hsv[offset + 1]) >= HUE_SATURATION_MIN else 0
    hue_delta = HUE_WEIGHT * hue_confidence * (hue_distance(first_hsv[offset], second_hsv[offset]) / 128)
    saturation_delta = SATURATION_WEIGHT * (abs(first_hsv[offset + 1] - second_hsv[offset + 1]) / 255)
    value_delta = VALUE_WEIGHT * (abs(first_hsv[offset + 2] - second_hsv[offset + 2]) / 255)
    return (hue_delta * hue_delta) + (saturation_delta * saturation_delta) + (value_delta * value_delta)


def hsv_sample_background_mask(
    current_hsv: bytes,
    samples: list[bytes],
    valid_mask: bytes,
    threshold: int,
) -> list[bool]:
    close_radius = max(0.12, threshold / 100)
    close_radius_squared = close_radius * close_radius
    min_matches = 2 if len(samples) >= 3 else 1
    mask = []

    for index, valid in enumerate(valid_mask):
        if not valid:
            mask.append(False)
            continue

        offset = index * 3
        if any(sample[offset + 2] <= MASK_BLACK_THRESHOLD for sample in samples):
            mask.append(False)
            continue

        matches = sum(
            1
            for sample in samples
            if hsv_distance_squared(current_hsv, sample, offset) <= close_radius_squared
        )
        mask.append(matches < min_matches)

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
    del max_shift_pixels
    del blur_radius
    results: list[DetectorResult] = []

    for threshold in thresholds:
        for window in windows:
            history: deque[bytes] = deque(maxlen=window)
            previous_frame: LabeledFrame | None = None

            for frame in frames:
                if previous_frame is not None and not same_capture_session(previous_frame, frame):
                    history.clear()

                current_hsv, width, height = load_hsv(frame.masked_path)

                if len(history) == window:
                    valid_mask = field_mask_bytes(width, height)
                    valid_mask = suppress_black_mask_pixels_hsv(current_hsv, current_hsv, valid_mask)
                    mask = hsv_sample_background_mask(current_hsv, list(history), valid_mask, threshold)
                    features = features_from_hsv_mask(
                        MODEL_NAME,
                        frame,
                        mask,
                        width,
                        height,
                        threshold,
                        window,
                        min_blob_area,
                        0,
                        0,
                        foreground_root,
                        0,
                        0,
                    )
                    results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, window, features))

                if frame.label == "inactive":
                    history.append(current_hsv)
                previous_frame = frame

    return results
