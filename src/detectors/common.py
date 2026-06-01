"""Shared image-difference detector utilities."""

from __future__ import annotations

import colorsys
import csv
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.roi_preprocess import DEFAULT_CONFIG_PATH, build_field_mask, load_roi_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_PATH = REPO_ROOT / "labels" / "labels.csv"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_FOREGROUND_ROOT = REPO_ROOT / "data" / "foreground"
EVAL_LABELS = {"active", "inactive", "discard"}
MAX_FRAME_GAP_SECONDS = 14 * 24 * 60 * 60
MASK_BLACK_THRESHOLD = 3
DISCARD_GRAY_RGB = (130, 130, 130)
DISCARD_GRAY_TOLERANCE = 2
DISCARD_GRAY_RATIO_THRESHOLD = 0.20
DISCARD_PARTIAL_GRAY_RATIO_THRESHOLD = 0.08
DISCARD_SMALL_GRAY_RATIO_THRESHOLD = 0.02
DISCARD_LOW_BLUE_RATIO_THRESHOLD = 0.10
DISCARD_DARK_RATIO_THRESHOLD = 0.70
DISCARD_VIVID_RATIO_THRESHOLD = 0.10


@dataclass(frozen=True)
class LabeledFrame:
    image_id: str
    timestamp: str
    masked_path: Path
    label: str


@dataclass(frozen=True)
class DetectorResult:
    model: str
    threshold: int
    cutoff: float
    window: int
    blur_radius: float
    image_id: str
    timestamp: str
    label: str
    prediction: str
    score: float
    changed_ratio: float
    largest_blob_ratio: float
    blob_count: int
    camera_shift_x: int
    camera_shift_y: int
    foreground_path: str
    confidence: float = 0.0
    config_label: str = ""


@dataclass(frozen=True)
class DetectionFeatures:
    image_id: str
    timestamp: str
    label: str
    score: float
    blur_radius: float
    changed_ratio: float
    largest_blob_ratio: float
    blob_count: int
    camera_shift_x: int
    camera_shift_y: int
    foreground_path: str


def repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    path.relative_to(REPO_ROOT.resolve())
    return path


def repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def load_frames(labels_path: Path) -> list[LabeledFrame]:
    with labels_path.open("r", newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("masked_path")]

    frames = []
    for row in sorted(rows, key=lambda item: (item.get("timestamp_utc", item.get("timestamp", "")), item["image_id"])):
        label = row.get("label", "")
        if label not in EVAL_LABELS:
            continue

        masked_path = repo_path(row["masked_path"])
        if not masked_path.exists():
            continue

        frames.append(
            LabeledFrame(
                image_id=row["image_id"],
                timestamp=row.get("timestamp", ""),
                masked_path=masked_path,
                label=label,
            )
        )

    return frames


def parse_frame_time(frame: LabeledFrame) -> datetime | None:
    if not frame.timestamp:
        return None
    try:
        return datetime.fromisoformat(frame.timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def same_capture_session(previous: LabeledFrame, current: LabeledFrame) -> bool:
    previous_time = parse_frame_time(previous)
    current_time = parse_frame_time(current)
    if previous_time is None or current_time is None:
        return previous.image_id.split("/", 1)[0] == current.image_id.split("/", 1)[0]
    gap = (current_time - previous_time).total_seconds()
    return 0 < gap <= MAX_FRAME_GAP_SECONDS


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def image_bytes(image: Image.Image) -> bytes:
    return image.tobytes()


def discard_gray_mask(path: Path) -> tuple[list[bool], int, int, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        get_pixels = getattr(image, "get_flattened_data", image.getdata)
        pixels = list(get_pixels())

    target_r, target_g, target_b = DISCARD_GRAY_RGB
    mask = []
    valid_pixels = 0
    gray_pixels = 0
    for red, green, blue in pixels:
        is_gray = (
            abs(red - target_r) <= DISCARD_GRAY_TOLERANCE
            and abs(green - target_g) <= DISCARD_GRAY_TOLERANCE
            and abs(blue - target_b) <= DISCARD_GRAY_TOLERANCE
        )
        is_valid_field_pixel = max(red, green, blue) > MASK_BLACK_THRESHOLD
        mask.append(is_gray)
        if is_valid_field_pixel:
            valid_pixels += 1
            if is_gray:
                gray_pixels += 1

    ratio = gray_pixels / valid_pixels if valid_pixels else 0.0
    return mask, width, height, ratio


def discard_artifact_mask(path: Path) -> tuple[list[bool], int, int, float, bool]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        get_pixels = getattr(image, "get_flattened_data", image.getdata)
        pixels = list(get_pixels())

    target_r, target_g, target_b = DISCARD_GRAY_RGB
    gray_mask: list[bool] = []
    no_color_mask: list[bool] = []
    valid_pixels = 0
    gray_pixels = 0
    low_saturation_pixels = 0
    dark_pixels = 0
    vivid_pixels = 0
    blue_pixels = 0

    for red, green, blue in pixels:
        max_channel = max(red, green, blue)
        min_channel = min(red, green, blue)
        is_valid_field_pixel = max_channel > MASK_BLACK_THRESHOLD
        is_gray = (
            abs(red - target_r) <= DISCARD_GRAY_TOLERANCE
            and abs(green - target_g) <= DISCARD_GRAY_TOLERANCE
            and abs(blue - target_b) <= DISCARD_GRAY_TOLERANCE
        )

        if is_valid_field_pixel:
            valid_pixels += 1
            hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
            saturation_255 = saturation * 255
            value_255 = value * 255
            is_low_saturation = saturation_255 < 35
            is_dark = value_255 < 55
            is_vivid = saturation_255 > 45 and value_255 > 55
            is_blue_field = 0.55 <= hue <= 0.78 and saturation_255 > 35 and value_255 > 45

            if is_gray:
                gray_pixels += 1
            if is_low_saturation:
                low_saturation_pixels += 1
            if is_dark:
                dark_pixels += 1
            if is_vivid:
                vivid_pixels += 1
            if is_blue_field:
                blue_pixels += 1
            no_color_mask.append(is_dark or is_low_saturation)
        else:
            no_color_mask.append(False)

        gray_mask.append(is_gray)

    if not valid_pixels:
        return [False for _pixel in pixels], width, height, 1.0, True

    gray_ratio = gray_pixels / valid_pixels
    low_saturation_ratio = low_saturation_pixels / valid_pixels
    dark_ratio = dark_pixels / valid_pixels
    vivid_ratio = vivid_pixels / valid_pixels
    blue_ratio = blue_pixels / valid_pixels

    gray_discard = gray_ratio >= DISCARD_GRAY_RATIO_THRESHOLD
    partial_gray_discard = (
        gray_ratio >= DISCARD_PARTIAL_GRAY_RATIO_THRESHOLD
        and low_saturation_ratio >= 0.10
    )
    small_gray_discard = (
        gray_ratio >= DISCARD_SMALL_GRAY_RATIO_THRESHOLD
        and low_saturation_ratio >= 0.05
    )
    no_color_discard = (
        blue_ratio <= DISCARD_LOW_BLUE_RATIO_THRESHOLD
        and (dark_ratio >= DISCARD_DARK_RATIO_THRESHOLD or vivid_ratio <= DISCARD_VIVID_RATIO_THRESHOLD)
    )
    is_discard = gray_discard or partial_gray_discard or small_gray_discard or no_color_discard

    if no_color_discard and not gray_discard and not partial_gray_discard and not small_gray_discard:
        mask = no_color_mask
        score = max(dark_ratio, 1 - blue_ratio, 1 - vivid_ratio)
    else:
        mask = gray_mask
        score = max(
            gray_ratio,
            low_saturation_ratio if partial_gray_discard else 0.0,
            gray_ratio if small_gray_discard else 0.0,
        )

    return mask, width, height, score, is_discard


def discard_gray_ratio(path: Path) -> float:
    _mask, _width, _height, ratio = discard_gray_mask(path)
    return ratio


def is_discard_gray_frame(path: Path) -> bool:
    return discard_gray_ratio(path) > DISCARD_GRAY_RATIO_THRESHOLD


def discard_artifact_score(path: Path) -> float:
    _mask, _width, _height, score, _is_discard = discard_artifact_mask(path)
    return score


def is_discard_artifact_frame(path: Path) -> bool:
    _mask, _width, _height, _score, is_discard = discard_artifact_mask(path)
    return is_discard


def mean_bytes(images: list[bytes]) -> bytes:
    pixel_count = len(images[0])
    return bytes(round(sum(image[index] for image in images) / len(images)) for index in range(pixel_count))


def median_bytes(images: list[bytes]) -> bytes:
    pixel_count = len(images[0])
    midpoint = len(images) // 2
    medians = []

    for index in range(pixel_count):
        values = sorted(image[index] for image in images)
        if len(values) % 2:
            medians.append(values[midpoint])
        else:
            medians.append(round((values[midpoint - 1] + values[midpoint]) / 2))

    return bytes(medians)


@lru_cache(maxsize=8)
def field_mask_bytes(width: int, height: int) -> bytes:
    config = load_roi_config(DEFAULT_CONFIG_PATH)
    mask = build_field_mask(config)
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return mask.tobytes()


def clamp_pixel(value: float) -> int:
    return max(0, min(255, round(value)))


def brightness_matched_reference(current: bytes, reference: bytes, valid_mask: bytes) -> bytes:
    current_values = []
    reference_values = []

    for pixel, base, valid in zip(current, reference, valid_mask):
        if valid:
            current_values.append(pixel)
            reference_values.append(base)

    if len(current_values) < 100:
        return reference

    current_mean = sum(current_values) / len(current_values)
    reference_mean = sum(reference_values) / len(reference_values)
    current_variance = sum((pixel - current_mean) ** 2 for pixel in current_values) / len(current_values)
    reference_variance = sum((base - reference_mean) ** 2 for base in reference_values) / len(reference_values)
    current_std = current_variance ** 0.5
    reference_std = reference_variance ** 0.5

    if reference_std < 1:
        scale = 1
    else:
        scale = current_std / reference_std
    offset = current_mean - (scale * reference_mean)

    return bytes(
        clamp_pixel((scale * base) + offset) if valid else base
        for base, valid in zip(reference, valid_mask)
    )


def diff_mask(current: bytes, background: bytes, threshold: int, valid_mask: bytes) -> list[bool]:
    return [
        bool(valid) and abs(pixel - base) > threshold
        for pixel, base, valid in zip(current, background, valid_mask)
    ]


def highpass_diff_mask(
    current: bytes,
    current_smooth: bytes,
    background: bytes,
    background_smooth: bytes,
    threshold: int,
    valid_mask: bytes,
) -> list[bool]:
    return [
        bool(valid) and abs((pixel - smooth_pixel) - (base - smooth_base)) > threshold
        for pixel, smooth_pixel, base, smooth_base, valid in zip(
            current,
            current_smooth,
            background,
            background_smooth,
            valid_mask,
        )
    ]


def dark_diff_mask(current: bytes, background: bytes, threshold: int, valid_mask: bytes) -> list[bool]:
    return [
        bool(valid) and (base - pixel) > threshold
        for pixel, base, valid in zip(current, background, valid_mask)
    ]


def suppress_masked_pixels(mask: bytes, suppress: bytes) -> bytes:
    return bytes(255 if valid and not blocked else 0 for valid, blocked in zip(mask, suppress))


def dilate_byte_mask(mask: bytes, width: int, height: int, radius: int) -> bytes:
    if radius <= 0:
        return mask

    dilated = bytearray(width * height)
    active = [index for index, pixel in enumerate(mask) if pixel]
    for index in active:
        x = index % width
        y = index // width
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            row = yy * width
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                dilated[row + xx] = 255
    return bytes(dilated)


def static_edge_mask(reference: bytes, width: int, height: int, valid_mask: bytes, threshold: int = 22) -> bytes:
    edges = bytearray(width * height)

    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            index = row + x
            if not valid_mask[index]:
                continue
            center = reference[index]
            contrast = max(
                abs(center - reference[index - 1]),
                abs(center - reference[index + 1]),
                abs(center - reference[index - width]),
                abs(center - reference[index + width]),
            )
            if contrast >= threshold:
                edges[index] = 255

    return dilate_byte_mask(bytes(edges), width, height, 1)


def sampled_alignment_error(
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    dx: int,
    dy: int,
    valid_mask: bytes,
    sample_step: int = 4,
) -> float:
    current_x_start = max(0, dx)
    current_y_start = max(0, dy)
    current_x_end = min(width, width + dx)
    current_y_end = min(height, height + dy)

    total = 0
    count = 0
    for y in range(current_y_start, current_y_end, sample_step):
        reference_y = y - dy
        current_row = y * width
        reference_row = reference_y * width
        for x in range(current_x_start, current_x_end, sample_step):
            reference_x = x - dx
            current_index = current_row + x
            reference_index = reference_row + reference_x
            if not valid_mask[current_index] or not valid_mask[reference_index]:
                continue
            current_pixel = current[current_row + x]
            reference_pixel = reference[reference_row + reference_x]
            total += abs(current_pixel - reference_pixel)
            count += 1

    if count < 100:
        return float("inf")
    return total / count


def array_alignment_error(
    current: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    dx: int,
    dy: int,
    sample_step: int,
) -> float:
    height, width = current.shape
    current_x_start = max(0, dx)
    current_y_start = max(0, dy)
    current_x_end = min(width, width + dx)
    current_y_end = min(height, height + dy)

    if current_x_end <= current_x_start or current_y_end <= current_y_start:
        return float("inf")

    current_view = current[current_y_start:current_y_end:sample_step, current_x_start:current_x_end:sample_step]
    reference_view = reference[
        current_y_start - dy:current_y_end - dy:sample_step,
        current_x_start - dx:current_x_end - dx:sample_step,
    ]
    current_valid = valid_mask[current_y_start:current_y_end:sample_step, current_x_start:current_x_end:sample_step]
    reference_valid = valid_mask[
        current_y_start - dy:current_y_end - dy:sample_step,
        current_x_start - dx:current_x_end - dx:sample_step,
    ]
    valid = current_valid & reference_valid
    valid_count = int(valid.sum())
    if valid_count < 100:
        return float("inf")

    diff = np.abs(current_view.astype(np.int16) - reference_view.astype(np.int16))
    return float(diff[valid].mean())


def estimate_shift(
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    max_shift_pixels: int,
    valid_mask: bytes,
) -> tuple[int, int]:
    if max_shift_pixels <= 0:
        return 0, 0

    current_array = np.frombuffer(current, dtype=np.uint8).reshape((height, width))
    reference_array = np.frombuffer(reference, dtype=np.uint8).reshape((height, width))
    valid_array = np.frombuffer(valid_mask, dtype=np.uint8).reshape((height, width)) > 0

    coarse_errors: list[tuple[float, int, int]] = []
    for dy in range(-max_shift_pixels, max_shift_pixels + 1, 2):
        for dx in range(-max_shift_pixels, max_shift_pixels + 1, 2):
            error = array_alignment_error(current_array, reference_array, valid_array, dx, dy, sample_step=4)
            coarse_errors.append((error, dx, dy))

    fine_candidates: set[tuple[int, int]] = {(0, 0)}
    for _error, coarse_dx, coarse_dy in sorted(coarse_errors)[:8]:
        for dy in range(max(-max_shift_pixels, coarse_dy - 2), min(max_shift_pixels, coarse_dy + 2) + 1):
            for dx in range(max(-max_shift_pixels, coarse_dx - 2), min(max_shift_pixels, coarse_dx + 2) + 1):
                fine_candidates.add((dx, dy))

    best_dx = 0
    best_dy = 0
    best_error = float("inf")
    for dx, dy in sorted(fine_candidates):
        error = array_alignment_error(current_array, reference_array, valid_array, dx, dy, sample_step=1)
        if error < best_error:
            best_dx = dx
            best_dy = dy
            best_error = error

    return best_dx, best_dy


def shifted_reference(
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    dx: int,
    dy: int,
) -> bytes:
    if dx == 0 and dy == 0:
        return reference

    aligned = bytearray(current)
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
            aligned[current_row + x] = reference[reference_row + reference_x]

    return bytes(aligned)


def shifted_valid_mask(
    valid_mask: bytes,
    width: int,
    height: int,
    dx: int,
    dy: int,
) -> bytes:
    if dx == 0 and dy == 0:
        return valid_mask

    aligned = bytearray(width * height)
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
            aligned[current_row + x] = valid_mask[reference_row + reference_x]

    return bytes(aligned)


def combined_valid_mask(current_valid: bytes, reference_valid: bytes) -> bytes:
    return bytes(255 if current and reference else 0 for current, reference in zip(current_valid, reference_valid))


def suppress_black_mask_pixels_gray(current: bytes, reference: bytes, valid_mask: bytes) -> bytes:
    return bytes(
        255 if valid and pixel > MASK_BLACK_THRESHOLD and base > MASK_BLACK_THRESHOLD else 0
        for pixel, base, valid in zip(current, reference, valid_mask)
    )


def suppress_black_mask_pixels_hsv(current_hsv: bytes, reference_hsv: bytes, valid_mask: bytes) -> bytes:
    filtered = bytearray(len(valid_mask))
    for index, valid in enumerate(valid_mask):
        if not valid:
            continue
        offset = index * 3
        if current_hsv[offset + 2] > MASK_BLACK_THRESHOLD and reference_hsv[offset + 2] > MASK_BLACK_THRESHOLD:
            filtered[index] = 255
    return bytes(filtered)


def aligned_diff_mask(
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    threshold: int,
    max_shift_pixels: int,
) -> tuple[list[bool], int, int]:
    matched_reference, valid_mask, dx, dy = aligned_reference_and_valid_mask(
        current,
        reference,
        width,
        height,
        max_shift_pixels,
    )
    return diff_mask(current, matched_reference, threshold, valid_mask), dx, dy


def aligned_dark_diff_mask(
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    threshold: int,
    max_shift_pixels: int,
) -> tuple[list[bool], int, int]:
    matched_reference, valid_mask, dx, dy = aligned_reference_and_valid_mask(
        current,
        reference,
        width,
        height,
        max_shift_pixels,
    )
    return dark_diff_mask(current, matched_reference, threshold, valid_mask), dx, dy


def aligned_reference_and_valid_mask(
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    max_shift_pixels: int,
) -> tuple[bytes, bytes, int, int]:
    current_valid = field_mask_bytes(width, height)
    dx, dy = estimate_shift(current, reference, width, height, max_shift_pixels, current_valid)
    aligned_reference = shifted_reference(current, reference, width, height, dx, dy)
    reference_valid = shifted_valid_mask(current_valid, width, height, dx, dy)
    valid_mask = combined_valid_mask(current_valid, reference_valid)
    valid_mask = suppress_black_mask_pixels_gray(current, aligned_reference, valid_mask)
    matched_reference = brightness_matched_reference(current, aligned_reference, valid_mask)
    valid_mask = suppress_masked_pixels(valid_mask, static_edge_mask(matched_reference, width, height, valid_mask))
    return matched_reference, valid_mask, dx, dy


def aligned_sample_references(
    current: bytes,
    samples: list[bytes],
    width: int,
    height: int,
    max_shift_pixels: int,
) -> tuple[list[bytes], bytes, int, int]:
    if not samples:
        raise ValueError("need at least one sample")

    current_valid = field_mask_bytes(width, height)
    dx, dy = estimate_shift(current, samples[-1], width, height, max_shift_pixels, current_valid)
    reference_valid = shifted_valid_mask(current_valid, width, height, dx, dy)
    valid_mask = combined_valid_mask(current_valid, reference_valid)

    shifted_samples = [shifted_reference(current, sample, width, height, dx, dy) for sample in samples]
    for shifted_sample in shifted_samples:
        valid_mask = suppress_black_mask_pixels_gray(current, shifted_sample, valid_mask)

    aligned_samples = [
        brightness_matched_reference(current, shifted_sample, valid_mask)
        for shifted_sample in shifted_samples
    ]
    edge_reference = mean_bytes(aligned_samples)
    valid_mask = suppress_masked_pixels(valid_mask, static_edge_mask(edge_reference, width, height, valid_mask))
    return aligned_samples, valid_mask, dx, dy


def blob_features(
    mask: list[bool],
    width: int,
    height: int,
    min_blob_area: int,
    total_pixels: int,
) -> tuple[float, float, int]:
    visited = bytearray(width * height)
    changed = sum(mask)
    largest = 0
    blob_count = 0

    for start, is_foreground in enumerate(mask):
        if not is_foreground or visited[start]:
            continue

        area = 0
        stack = [start]
        visited[start] = 1

        while stack:
            index = stack.pop()
            area += 1
            x = index % width
            y = index // width

            neighbors = []
            if x > 0:
                neighbors.append(index - 1)
            if x < width - 1:
                neighbors.append(index + 1)
            if y > 0:
                neighbors.append(index - width)
            if y < height - 1:
                neighbors.append(index + width)

            for neighbor in neighbors:
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)

        if area >= min_blob_area:
            blob_count += 1
            largest = max(largest, area)

    return changed / total_pixels, largest / total_pixels, blob_count


def compact_blob_mask(mask: list[bool], width: int, height: int, min_blob_area: int) -> list[bool]:
    visited = bytearray(width * height)
    filtered = [False] * (width * height)

    for start, is_foreground in enumerate(mask):
        if not is_foreground or visited[start]:
            continue

        pixels = []
        stack = [start]
        visited[start] = 1
        min_x = max_x = start % width
        min_y = max_y = start // width

        while stack:
            index = stack.pop()
            pixels.append(index)
            x = index % width
            y = index // width
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            if x > 0:
                neighbor = index - 1
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if x < width - 1:
                neighbor = index + 1
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if y > 0:
                neighbor = index - width
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if y < height - 1:
                neighbor = index + width
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)

        area = len(pixels)
        if area < min_blob_area:
            continue

        blob_width = max_x - min_x + 1
        blob_height = max_y - min_y + 1
        fill_ratio = area / (blob_width * blob_height)
        aspect_ratio = max(blob_width / blob_height, blob_height / blob_width)

        if blob_width < 3 or blob_height < 4:
            continue
        if fill_ratio < 0.25:
            continue
        if aspect_ratio > 3.2:
            continue

        for pixel in pixels:
            filtered[pixel] = True

    return filtered


def hsv_person_blob_mask(mask: list[bool], width: int, height: int, min_blob_area: int) -> list[bool]:
    del min_blob_area
    visited = bytearray(width * height)
    filtered = [False] * (width * height)

    for start, is_foreground in enumerate(mask):
        if not is_foreground or visited[start]:
            continue

        pixels = []
        stack = [start]
        visited[start] = 1
        min_x = max_x = start % width
        min_y = max_y = start // width

        while stack:
            index = stack.pop()
            pixels.append(index)
            x = index % width
            y = index // width
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            if x > 0:
                neighbor = index - 1
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if x < width - 1:
                neighbor = index + 1
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if y > 0:
                neighbor = index - width
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if y < height - 1:
                neighbor = index + width
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)

        area = len(pixels)
        blob_width = max_x - min_x + 1
        blob_height = max_y - min_y + 1
        fill_ratio = area / (blob_width * blob_height)
        aspect_ratio = max(blob_width / blob_height, blob_height / blob_width)
        tiny_compact = 4 <= area <= 80 and blob_width >= 2 and blob_height >= 2 and fill_ratio >= 0.25 and aspect_ratio <= 3.5
        medium_compact = 80 < area <= 220 and fill_ratio >= 0.25 and aspect_ratio <= 3.0

        if not tiny_compact and not medium_compact:
            continue

        for pixel in pixels:
            filtered[pixel] = True

    return filtered


def hsv_person_blob_features(
    mask: list[bool],
    width: int,
    height: int,
    total_pixels: int,
) -> tuple[float, float, int, int, float]:
    visited = bytearray(width * height)
    changed = sum(mask)
    largest = 0
    person_blob_count = 0
    tiny_blob_area = 0
    large_artifact_area = 0

    for start, is_foreground in enumerate(mask):
        if not is_foreground or visited[start]:
            continue

        area = 0
        stack = [start]
        visited[start] = 1
        min_x = max_x = start % width
        min_y = max_y = start // width

        while stack:
            index = stack.pop()
            area += 1
            x = index % width
            y = index // width
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            if x > 0:
                neighbor = index - 1
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if x < width - 1:
                neighbor = index + 1
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if y > 0:
                neighbor = index - width
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
            if y < height - 1:
                neighbor = index + width
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)

        blob_width = max_x - min_x + 1
        blob_height = max_y - min_y + 1
        fill_ratio = area / (blob_width * blob_height)
        aspect_ratio = max(blob_width / blob_height, blob_height / blob_width)
        person_like = 4 <= area <= 80 and blob_width >= 2 and blob_height >= 2 and fill_ratio >= 0.25 and aspect_ratio <= 3.5

        if person_like:
            person_blob_count += 1
            tiny_blob_area += area

        if area > 220 or aspect_ratio > 4.0:
            large_artifact_area += area

        largest = max(largest, area)

    return (
        changed / total_pixels,
        largest / total_pixels,
        person_blob_count,
        tiny_blob_area,
        large_artifact_area / total_pixels,
    )


def score_hsv_person_mask(mask: list[bool], width: int, height: int) -> tuple[float, float, float, int]:
    total_pixels = sum(1 for pixel in field_mask_bytes(width, height) if pixel)
    changed_ratio, largest_blob_ratio, person_blob_count, tiny_blob_area, large_artifact_ratio = hsv_person_blob_features(
        mask,
        width,
        height,
        total_pixels,
    )
    tiny_blob_area_ratio = tiny_blob_area / total_pixels
    score = (
        0.15 * changed_ratio
        + 4.0 * tiny_blob_area_ratio
        + 0.01 * min(person_blob_count, 12)
        + 0.25 * min(largest_blob_ratio, 0.01)
        - 0.75 * large_artifact_ratio
    )
    return max(0.0, score), changed_ratio, largest_blob_ratio, person_blob_count


def score_mask(mask: list[bool], width: int, height: int, min_blob_area: int) -> tuple[float, float, float, int]:
    total_pixels = sum(1 for pixel in field_mask_bytes(width, height) if pixel)
    changed_ratio, largest_blob_ratio, blob_count = blob_features(mask, width, height, min_blob_area, total_pixels)
    score = changed_ratio + (0 * largest_blob_ratio) + (0.001 * min(blob_count, 20))
    return score, changed_ratio, largest_blob_ratio, blob_count


def score_dark_mask(mask: list[bool], width: int, height: int, min_blob_area: int) -> tuple[float, float, float, int]:
    total_pixels = sum(1 for pixel in field_mask_bytes(width, height) if pixel)
    changed_ratio, largest_blob_ratio, blob_count = blob_features(mask, width, height, min_blob_area, total_pixels)
    score = changed_ratio + (3 * largest_blob_ratio) + (0.003 * min(blob_count, 20))
    return score, changed_ratio, largest_blob_ratio, blob_count


def score_blob_mask(mask: list[bool], width: int, height: int, min_blob_area: int) -> tuple[float, float, float, int]:
    total_pixels = sum(1 for pixel in field_mask_bytes(width, height) if pixel)
    changed_ratio, largest_blob_ratio, blob_count = blob_features(mask, width, height, min_blob_area, total_pixels)
    score = changed_ratio + (3 * largest_blob_ratio) + (0.003 * min(blob_count, 20))
    return score, changed_ratio, largest_blob_ratio, blob_count


def foreground_path_for(
    model: str,
    threshold: int,
    window: int,
    max_shift_pixels: int,
    blur_radius: float,
    frame: LabeledFrame,
    foreground_root: Path,
) -> Path:
    date_part = frame.image_id.split("/", 1)[0]
    image_name = Path(frame.image_id).name
    blur_part = f"{blur_radius:g}".replace(".", "p")
    setting = f"t{threshold}_w{window}_s{max_shift_pixels}_b{blur_part}"
    return foreground_root / model / setting / date_part / f"{image_name}.png"


def save_foreground_mask(mask: list[bool], width: int, height: int, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (width, height))
    image.putdata([255 if pixel else 0 for pixel in mask])
    image.save(output_path)
    return repo_relative(output_path)


def detect_against_reference(
    model: str,
    frame: LabeledFrame,
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    threshold: int,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
) -> DetectionFeatures:
    mask, shift_x, shift_y = aligned_diff_mask(current, reference, width, height, threshold, max_shift_pixels)
    return features_from_blob_mask(
        model,
        frame,
        mask,
        width,
        height,
        threshold,
        window,
        min_blob_area,
        max_shift_pixels,
        blur_radius,
        foreground_root,
        shift_x,
        shift_y,
    )


def features_from_blob_mask(
    model: str,
    frame: LabeledFrame,
    mask: list[bool],
    width: int,
    height: int,
    threshold: int,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    shift_x: int,
    shift_y: int,
) -> DetectionFeatures:
    mask = compact_blob_mask(mask, width, height, min_blob_area)
    score, changed_ratio, largest_blob_ratio, blob_count = score_blob_mask(mask, width, height, min_blob_area)
    foreground_path = save_foreground_mask(
        mask,
        width,
        height,
        foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
    )
    return DetectionFeatures(
        image_id=frame.image_id,
        timestamp=frame.timestamp,
        label=frame.label,
        score=score,
        blur_radius=blur_radius,
        changed_ratio=changed_ratio,
        largest_blob_ratio=largest_blob_ratio,
        blob_count=blob_count,
        camera_shift_x=shift_x,
        camera_shift_y=shift_y,
        foreground_path=foreground_path,
    )


def features_from_hsv_mask(
    model: str,
    frame: LabeledFrame,
    mask: list[bool],
    width: int,
    height: int,
    threshold: int,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    shift_x: int,
    shift_y: int,
) -> DetectionFeatures:
    mask = hsv_person_blob_mask(mask, width, height, min_blob_area)
    score, changed_ratio, largest_blob_ratio, blob_count = score_hsv_person_mask(mask, width, height)
    foreground_path = save_foreground_mask(
        mask,
        width,
        height,
        foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
    )
    return DetectionFeatures(
        image_id=frame.image_id,
        timestamp=frame.timestamp,
        label=frame.label,
        score=score,
        blur_radius=blur_radius,
        changed_ratio=changed_ratio,
        largest_blob_ratio=largest_blob_ratio,
        blob_count=blob_count,
        camera_shift_x=shift_x,
        camera_shift_y=shift_y,
        foreground_path=foreground_path,
    )


def detect_dark_against_reference(
    model: str,
    frame: LabeledFrame,
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    threshold: int,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
) -> DetectionFeatures:
    mask, shift_x, shift_y = aligned_dark_diff_mask(current, reference, width, height, threshold, max_shift_pixels)
    mask = compact_blob_mask(mask, width, height, min_blob_area)
    score, changed_ratio, largest_blob_ratio, blob_count = score_dark_mask(mask, width, height, min_blob_area)
    foreground_path = save_foreground_mask(
        mask,
        width,
        height,
        foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
    )
    return DetectionFeatures(
        image_id=frame.image_id,
        timestamp=frame.timestamp,
        label=frame.label,
        score=score,
        blur_radius=blur_radius,
        changed_ratio=changed_ratio,
        largest_blob_ratio=largest_blob_ratio,
        blob_count=blob_count,
        camera_shift_x=shift_x,
        camera_shift_y=shift_y,
        foreground_path=foreground_path,
    )


def features_from_dark_mask(
    model: str,
    frame: LabeledFrame,
    mask: list[bool],
    width: int,
    height: int,
    threshold: int,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    shift_x: int,
    shift_y: int,
) -> DetectionFeatures:
    mask = compact_blob_mask(mask, width, height, min_blob_area)
    score, changed_ratio, largest_blob_ratio, blob_count = score_dark_mask(mask, width, height, min_blob_area)
    foreground_path = save_foreground_mask(
        mask,
        width,
        height,
        foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
    )
    return DetectionFeatures(
        image_id=frame.image_id,
        timestamp=frame.timestamp,
        label=frame.label,
        score=score,
        blur_radius=blur_radius,
        changed_ratio=changed_ratio,
        largest_blob_ratio=largest_blob_ratio,
        blob_count=blob_count,
        camera_shift_x=shift_x,
        camera_shift_y=shift_y,
        foreground_path=foreground_path,
    )


def predict(score: float, cutoff: float) -> str:
    return "active" if score >= cutoff else "inactive"


def prediction_confidence(score: float, cutoff: float) -> float:
    """Return a simple confidence for threshold-based legacy detectors."""
    if score >= cutoff:
        return min(1.0, 0.5 + (score - cutoff))
    return min(1.0, 0.5 + (cutoff - score))


def expand_cutoffs(
    model: str,
    threshold: int,
    cutoffs: list[float],
    window: int,
    features: DetectionFeatures,
) -> list[DetectorResult]:
    results = []
    for cutoff in cutoffs:
        results.append(
            DetectorResult(
                model=model,
                threshold=threshold,
                cutoff=cutoff,
                window=window,
                blur_radius=features.blur_radius,
                image_id=features.image_id,
                timestamp=features.timestamp,
                label=features.label,
                prediction=predict(features.score, cutoff),
                score=features.score,
                changed_ratio=features.changed_ratio,
                largest_blob_ratio=features.largest_blob_ratio,
                blob_count=features.blob_count,
                camera_shift_x=features.camera_shift_x,
                camera_shift_y=features.camera_shift_y,
                foreground_path=features.foreground_path,
                confidence=prediction_confidence(features.score, cutoff),
            )
        )
    return results
