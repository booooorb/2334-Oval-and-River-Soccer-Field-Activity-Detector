"""HSV running Gaussian background model using trusted inactive updates."""

from __future__ import annotations

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
from src.detectors.hsv_previous_diff import (
    HUE_SATURATION_MIN,
    HUE_WEIGHT,
    SATURATION_WEIGHT,
    VALUE_WEIGHT,
    load_hsv,
)

MODEL_NAME = "hsv_running_gaussian"
DEFAULT_VARIANCE = 0.025
MIN_VARIANCE = 0.0025


def learning_rate(window: int) -> float:
    return max(0.01, min(0.2, 2 / (window + 1)))


def initialize_mean(sample: bytes) -> list[float]:
    return [float(value) for value in sample]


def initialize_variance(pixel_count: int) -> list[float]:
    return [DEFAULT_VARIANCE] * pixel_count


def hsv_distance_to_mean_squared(current_hsv: bytes, mean: list[float], offset: int) -> float:
    mean_hue = mean[offset] % 256
    hue_confidence = 1 if max(current_hsv[offset + 1], mean[offset + 1]) >= HUE_SATURATION_MIN else 0
    hue_delta = HUE_WEIGHT * hue_confidence * (hue_distance(current_hsv[offset], round(mean_hue)) / 128)
    saturation_delta = SATURATION_WEIGHT * (abs(current_hsv[offset + 1] - mean[offset + 1]) / 255)
    value_delta = VALUE_WEIGHT * (abs(current_hsv[offset + 2] - mean[offset + 2]) / 255)
    return (hue_delta * hue_delta) + (saturation_delta * saturation_delta) + (value_delta * value_delta)


def update_hue_mean(previous: float, current: int, rho: float) -> float:
    forward_delta = (current - previous + 128) % 256 - 128
    return (previous + (rho * forward_delta)) % 256


def update_background(mean: list[float], variance: list[float], sample: bytes, valid_mask: bytes, rho: float) -> None:
    for index, valid in enumerate(valid_mask):
        if not valid:
            continue

        offset = index * 3
        if sample[offset + 2] <= MASK_BLACK_THRESHOLD:
            continue

        distance_squared = hsv_distance_to_mean_squared(sample, mean, offset)
        variance[index] = max(MIN_VARIANCE, (rho * distance_squared) + ((1 - rho) * variance[index]))
        mean[offset] = update_hue_mean(mean[offset], sample[offset], rho)
        mean[offset + 1] = (rho * sample[offset + 1]) + ((1 - rho) * mean[offset + 1])
        mean[offset + 2] = (rho * sample[offset + 2]) + ((1 - rho) * mean[offset + 2])


def gaussian_foreground_mask(
    current_hsv: bytes,
    mean: list[float],
    variance: list[float],
    valid_mask: bytes,
    threshold: int,
) -> list[bool]:
    k = max(1.5, threshold / 4)
    mask = []

    for index, valid in enumerate(valid_mask):
        if not valid:
            mask.append(False)
            continue

        offset = index * 3
        if current_hsv[offset + 2] <= MASK_BLACK_THRESHOLD or mean[offset + 2] <= MASK_BLACK_THRESHOLD:
            mask.append(False)
            continue

        distance = hsv_distance_to_mean_squared(current_hsv, mean, offset) ** 0.5
        sigma = variance[index] ** 0.5
        mask.append((distance / sigma) > k)

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
            mean: list[float] | None = None
            variance: list[float] | None = None
            previous_frame: LabeledFrame | None = None
            rho = learning_rate(window)

            for frame in frames:
                if previous_frame is not None and not same_capture_session(previous_frame, frame):
                    mean = None
                    variance = None

                current_hsv, width, height = load_hsv(frame.masked_path)
                valid_mask = field_mask_bytes(width, height)
                valid_mask = suppress_black_mask_pixels_hsv(current_hsv, current_hsv, valid_mask)

                if mean is not None and variance is not None:
                    mask = gaussian_foreground_mask(current_hsv, mean, variance, valid_mask, threshold)
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
                    if mean is None or variance is None:
                        mean = initialize_mean(current_hsv)
                        variance = initialize_variance(width * height)
                    else:
                        update_background(mean, variance, current_hsv, valid_mask, rho)

                previous_frame = frame

    return results
