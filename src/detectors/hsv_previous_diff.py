"""No-blur HSV previous-frame differencing detector."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    expand_cutoffs,
    features_from_hsv_mask,
    field_mask_bytes,
    image_bytes,
    same_capture_session,
    suppress_black_mask_pixels_hsv,
)
from src.detectors.hsv_blue_diff import hue_distance

MODEL_NAME = "hsv_previous_diff"
HUE_WEIGHT = 2.0
SATURATION_WEIGHT = 1.5
VALUE_WEIGHT = 0.35
HUE_SATURATION_MIN = 30


def load_hsv(path: Path) -> tuple[bytes, int, int]:
    image = Image.open(path).convert("HSV")
    return image_bytes(image), image.width, image.height


def hsv_diff_mask(
    current_hsv: bytes,
    reference_hsv: bytes,
    valid_mask: bytes,
    threshold: int,
) -> list[bool]:
    threshold_squared = (threshold / 100) ** 2
    mask = []

    for index, valid in enumerate(valid_mask):
        if not valid:
            mask.append(False)
            continue

        offset = index * 3
        hue_confidence = 1 if max(current_hsv[offset + 1], reference_hsv[offset + 1]) >= HUE_SATURATION_MIN else 0
        hue_delta = HUE_WEIGHT * hue_confidence * (hue_distance(current_hsv[offset], reference_hsv[offset]) / 128)
        saturation_delta = SATURATION_WEIGHT * (abs(current_hsv[offset + 1] - reference_hsv[offset + 1]) / 255)
        value_delta = VALUE_WEIGHT * (abs(current_hsv[offset + 2] - reference_hsv[offset + 2]) / 255)
        distance_squared = (hue_delta * hue_delta) + (saturation_delta * saturation_delta) + (value_delta * value_delta)
        mask.append(distance_squared > threshold_squared)

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
    del windows
    del max_shift_pixels
    del blur_radius
    results: list[DetectorResult] = []

    for threshold in thresholds:
        inactive_reference_hsv: bytes | None = None
        inactive_reference_frame: LabeledFrame | None = None

        for frame in frames:
            current_hsv, width, height = load_hsv(frame.masked_path)

            if inactive_reference_hsv is not None and inactive_reference_frame is not None and same_capture_session(inactive_reference_frame, frame):
                valid_mask = field_mask_bytes(width, height)
                valid_mask = suppress_black_mask_pixels_hsv(current_hsv, inactive_reference_hsv, valid_mask)
                mask = hsv_diff_mask(current_hsv, inactive_reference_hsv, valid_mask, threshold)
                features = features_from_hsv_mask(
                    MODEL_NAME,
                    frame,
                    mask,
                    width,
                    height,
                    threshold,
                    1,
                    min_blob_area,
                    0,
                    0,
                    foreground_root,
                    0,
                    0,
                )
                results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, 1, features))

            if frame.label == "inactive":
                inactive_reference_hsv = current_hsv
                inactive_reference_frame = frame

    return results
