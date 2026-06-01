"""HSV balanced blurred previous-frame subtraction detector."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    aligned_reference_and_valid_mask,
    expand_cutoffs,
    features_from_hsv_mask,
    image_bytes,
    same_capture_session,
)
from src.detectors.hsv_blue_diff import (
    BLUE_SATURATION_THRESHOLD,
    HUE_DISTANCE_THRESHOLD,
    hue_distance,
    median_field_hue,
    shift_hsv_reference,
)

MODEL_NAME = "hsv_balanced_previous_diff_blur"
MIN_COLOR_SATURATION = 18
SATURATION_MULTIPLIER = 1.0
VALUE_MULTIPLIER = 4.0


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def hsv_balanced_mask(
    current_hsv: bytes,
    reference_hsv: bytes,
    valid_mask: bytes,
    threshold: int,
) -> list[bool]:
    field_hue = median_field_hue(reference_hsv, valid_mask)
    saturation_threshold = max(12, round(threshold * SATURATION_MULTIPLIER))
    value_threshold = max(32, round(threshold * VALUE_MULTIPLIER))
    mask = []

    for index, valid in enumerate(valid_mask):
        if not valid:
            mask.append(False)
            continue

        offset = index * 3
        current_hue = current_hsv[offset]
        current_saturation = current_hsv[offset + 1]
        current_value = current_hsv[offset + 2]
        reference_hue = reference_hsv[offset]
        reference_saturation = reference_hsv[offset + 1]
        reference_value = reference_hsv[offset + 2]

        current_is_blue = (
            current_saturation >= BLUE_SATURATION_THRESHOLD
            and hue_distance(current_hue, field_hue) <= HUE_DISTANCE_THRESHOLD
        )
        reference_is_blue = (
            reference_saturation >= BLUE_SATURATION_THRESHOLD
            and hue_distance(reference_hue, field_hue) <= HUE_DISTANCE_THRESHOLD
        )
        if current_is_blue and reference_is_blue:
            mask.append(False)
            continue

        hue_changed = (
            max(current_saturation, reference_saturation) >= MIN_COLOR_SATURATION
            and hue_distance(current_hue, reference_hue) > threshold
        )
        saturation_changed = abs(current_saturation - reference_saturation) > saturation_threshold
        value_changed = abs(current_value - reference_value) > value_threshold
        blue_to_nonblue = reference_is_blue and not current_is_blue

        mask.append(blue_to_nonblue or hue_changed or saturation_changed or value_changed)

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
    results: list[DetectorResult] = []

    for threshold in thresholds:
        inactive_reference_rgb: Image.Image | None = None
        inactive_reference_gray: bytes | None = None
        inactive_reference_frame: LabeledFrame | None = None

        for frame in frames:
            rgb = load_rgb(frame.masked_path)
            blurred_rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            gray_image = blurred_rgb.convert("L")
            current_gray = image_bytes(gray_image)
            current_hsv = image_bytes(blurred_rgb.convert("HSV"))
            width, height = rgb.size

            if inactive_reference_rgb is not None and inactive_reference_gray is not None and inactive_reference_frame is not None and same_capture_session(inactive_reference_frame, frame):
                _reference_gray, valid_mask, shift_x, shift_y = aligned_reference_and_valid_mask(
                    current_gray,
                    inactive_reference_gray,
                    width,
                    height,
                    max_shift_pixels,
                )
                previous_hsv = image_bytes(inactive_reference_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius)).convert("HSV"))
                reference_hsv = shift_hsv_reference(current_hsv, previous_hsv, width, height, shift_x, shift_y)
                mask = hsv_balanced_mask(current_hsv, reference_hsv, valid_mask, threshold)
                features = features_from_hsv_mask(
                    MODEL_NAME,
                    frame,
                    mask,
                    width,
                    height,
                    threshold,
                    1,
                    min_blob_area,
                    max_shift_pixels,
                    blur_radius,
                    foreground_root,
                    shift_x,
                    shift_y,
                )
                results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, 1, features))

            if frame.label == "inactive":
                inactive_reference_rgb = rgb
                inactive_reference_gray = current_gray
                inactive_reference_frame = frame

    return results
