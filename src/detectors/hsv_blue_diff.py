"""HSV blue-field filtered previous-frame subtraction detector."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    aligned_reference_and_valid_mask,
    diff_mask,
    expand_cutoffs,
    features_from_hsv_mask,
    image_bytes,
    same_capture_session,
)

MODEL_NAME = "hsv_blue_diff"
HUE_DISTANCE_THRESHOLD = 32
BLUE_SATURATION_THRESHOLD = 25
SIGNAL_SATURATION_THRESHOLD = 25
MIN_HUE_SAMPLE_SATURATION = 35
MIN_HUE_SAMPLE_VALUE = 25


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def hue_distance(first: int, second: int) -> int:
    distance = abs(first - second)
    return min(distance, 256 - distance)


def median_field_hue(hsv: bytes, valid_mask: bytes) -> int:
    hues = []

    for index, valid in enumerate(valid_mask):
        if not valid:
            continue

        offset = index * 3
        hue = hsv[offset]
        saturation = hsv[offset + 1]
        value = hsv[offset + 2]
        if saturation >= MIN_HUE_SAMPLE_SATURATION and value >= MIN_HUE_SAMPLE_VALUE:
            hues.append(hue)

    if len(hues) < 100:
        hues = [hsv[index * 3] for index, valid in enumerate(valid_mask) if valid]

    if not hues:
        return 0

    hues.sort()
    return hues[len(hues) // 2]


def shift_hsv_reference(
    current_hsv: bytes,
    reference_hsv: bytes,
    width: int,
    height: int,
    dx: int,
    dy: int,
) -> bytes:
    if dx == 0 and dy == 0:
        return reference_hsv

    aligned = bytearray(current_hsv)
    current_x_start = max(0, dx)
    current_y_start = max(0, dy)
    current_x_end = min(width, width + dx)
    current_y_end = min(height, height + dy)

    for y in range(current_y_start, current_y_end):
        reference_y = y - dy
        current_row = y * width
        reference_row = reference_y * width
        for x in range(current_x_start, current_x_end):
            reference_x = x - dx
            current_offset = (current_row + x) * 3
            reference_offset = (reference_row + reference_x) * 3
            aligned[current_offset : current_offset + 3] = reference_hsv[reference_offset : reference_offset + 3]

    return bytes(aligned)


def blue_filtered_mask(
    gray_diff: list[bool],
    current_hsv: bytes,
    reference_hsv: bytes,
    valid_mask: bytes,
) -> list[bool]:
    field_hue = median_field_hue(current_hsv, valid_mask)
    mask = []

    for index, changed in enumerate(gray_diff):
        if not valid_mask[index]:
            mask.append(False)
            continue

        offset = index * 3
        current_hue = current_hsv[offset]
        current_saturation = current_hsv[offset + 1]
        reference_hue = reference_hsv[offset]
        reference_saturation = reference_hsv[offset + 1]

        current_is_blue = (
            current_saturation >= BLUE_SATURATION_THRESHOLD
            and hue_distance(current_hue, field_hue) <= HUE_DISTANCE_THRESHOLD
        )
        reference_is_blue = (
            reference_saturation >= BLUE_SATURATION_THRESHOLD
            and hue_distance(reference_hue, field_hue) <= HUE_DISTANCE_THRESHOLD
        )
        colored_nonblue = (
            current_saturation >= SIGNAL_SATURATION_THRESHOLD
            and hue_distance(current_hue, field_hue) > HUE_DISTANCE_THRESHOLD
        )
        blue_to_nonblue = reference_is_blue and not current_is_blue
        blue_to_blue_lighting = current_is_blue and reference_is_blue

        mask.append(blue_to_nonblue or (changed and colored_nonblue and not blue_to_blue_lighting))

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
            gray_image = rgb.convert("L").filter(ImageFilter.GaussianBlur(radius=blur_radius))
            current_gray = image_bytes(gray_image)
            current_hsv = image_bytes(rgb.convert("HSV"))
            width, height = rgb.size

            if inactive_reference_rgb is not None and inactive_reference_gray is not None and inactive_reference_frame is not None and same_capture_session(inactive_reference_frame, frame):
                reference_gray, valid_mask, shift_x, shift_y = aligned_reference_and_valid_mask(
                    current_gray,
                    inactive_reference_gray,
                    width,
                    height,
                    max_shift_pixels,
                )
                previous_hsv = image_bytes(inactive_reference_rgb.convert("HSV"))
                reference_hsv = shift_hsv_reference(current_hsv, previous_hsv, width, height, shift_x, shift_y)
                changed = diff_mask(current_gray, reference_gray, threshold, valid_mask)
                mask = blue_filtered_mask(changed, current_hsv, reference_hsv, valid_mask)
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
