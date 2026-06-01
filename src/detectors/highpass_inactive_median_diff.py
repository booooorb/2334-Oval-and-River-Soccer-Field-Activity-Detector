"""High-pass inactive-median background subtraction detector."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageFilter

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    aligned_sample_references,
    diff_mask,
    expand_cutoffs,
    features_from_blob_mask,
    highpass_diff_mask,
    image_bytes,
    load_gray,
    median_bytes,
    same_capture_session,
)

MODEL_NAME = "highpass_inactive_median_diff"
MIN_HIGHPASS_RADIUS = 8.0
LIGHTING_PENALTY_WEIGHT = 0.5
LIGHTING_HIGHPASS_MULTIPLIER = 3.0


def blurred_bytes(pixels: bytes, width: int, height: int, radius: float) -> bytes:
    image = Image.frombytes("L", (width, height), pixels)
    return image.filter(ImageFilter.GaussianBlur(radius=radius)).tobytes()


def changed_ratio(mask: list[bool], valid_mask: bytes) -> float:
    valid_pixels = sum(1 for pixel in valid_mask if pixel)
    if not valid_pixels:
        return 0
    return sum(mask) / valid_pixels


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
    results: list[DetectorResult] = []
    highpass_radius = max(MIN_HIGHPASS_RADIUS, blur_radius)

    for threshold in thresholds:
        for window in windows:
            inactive_history: deque[bytes] = deque(maxlen=window)
            previous_frame: LabeledFrame | None = None

            for frame in frames:
                if previous_frame is not None and not same_capture_session(previous_frame, frame):
                    inactive_history.clear()

                image = load_gray(frame.masked_path)
                current = image_bytes(image)
                width, height = image.size

                if len(inactive_history) == window:
                    samples, valid_mask, shift_x, shift_y = aligned_sample_references(
                        current,
                        list(inactive_history),
                        width,
                        height,
                        max_shift_pixels,
                    )
                    background = median_bytes(samples)
                    current_smooth = blurred_bytes(current, width, height, highpass_radius)
                    background_smooth = blurred_bytes(background, width, height, highpass_radius)
                    mask = highpass_diff_mask(
                        current,
                        current_smooth,
                        background,
                        background_smooth,
                        threshold,
                        valid_mask,
                    )
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
                        highpass_radius,
                        foreground_root,
                        shift_x,
                        shift_y,
                    )

                    raw_mask = diff_mask(current, background, threshold, valid_mask)
                    raw_change_ratio = changed_ratio(raw_mask, valid_mask)
                    lighting_score = max(
                        0.0,
                        raw_change_ratio - (LIGHTING_HIGHPASS_MULTIPLIER * features.changed_ratio),
                    )
                    adjusted_score = max(0.0, features.score - (LIGHTING_PENALTY_WEIGHT * lighting_score))
                    features = replace(features, score=adjusted_score)
                    results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, window, features))

                if frame.label == "inactive":
                    inactive_history.append(current)

                previous_frame = frame

    return results
