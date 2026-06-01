"""HSV previous-frame differencing with local support and patch energy."""

from __future__ import annotations

from pathlib import Path

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    expand_cutoffs,
    features_from_hsv_mask,
    field_mask_bytes,
    same_capture_session,
    suppress_black_mask_pixels_hsv,
)
from src.detectors.hsv_pbas import hsv_distance_squared
from src.detectors.hsv_previous_diff import load_hsv

MODEL_NAME = "hsv_local_support"
SUPPORT_RADIUS = 2
MIN_SUPPORT_COUNT = 5
PATCH_RADIUS = 4
MIN_PATCH_ENERGY = 0.045


def hsv_distance_map(current_hsv: bytes, reference_hsv: bytes, valid_mask: bytes) -> list[float]:
    distances = []
    for index, valid in enumerate(valid_mask):
        if not valid:
            distances.append(0.0)
            continue
        distances.append(hsv_distance_squared(current_hsv, reference_hsv, index * 3) ** 0.5)
    return distances


def local_support_count(weak_mask: list[bool], width: int, height: int, x: int, y: int) -> int:
    count = 0
    for yy in range(max(0, y - SUPPORT_RADIUS), min(height, y + SUPPORT_RADIUS + 1)):
        row = yy * width
        for xx in range(max(0, x - SUPPORT_RADIUS), min(width, x + SUPPORT_RADIUS + 1)):
            if weak_mask[row + xx]:
                count += 1
    return count


def patch_energy(distances: list[float], valid_mask: bytes, width: int, height: int, x: int, y: int) -> float:
    total = 0.0
    count = 0
    for yy in range(max(0, y - PATCH_RADIUS), min(height, y + PATCH_RADIUS + 1)):
        row = yy * width
        for xx in range(max(0, x - PATCH_RADIUS), min(width, x + PATCH_RADIUS + 1)):
            index = row + xx
            if valid_mask[index]:
                total += distances[index]
                count += 1
    return total / count if count else 0.0


def local_support_mask(
    current_hsv: bytes,
    reference_hsv: bytes,
    valid_mask: bytes,
    width: int,
    height: int,
    threshold: int,
) -> list[bool]:
    distances = hsv_distance_map(current_hsv, reference_hsv, valid_mask)
    weak_threshold = max(0.07, threshold / 100)
    weak_mask = [bool(valid) and distance > weak_threshold for distance, valid in zip(distances, valid_mask)]
    mask = []

    for index, weak in enumerate(weak_mask):
        if not weak:
            mask.append(False)
            continue

        x = index % width
        y = index // width
        supported = local_support_count(weak_mask, width, height, x, y) >= MIN_SUPPORT_COUNT
        energetic = patch_energy(distances, valid_mask, width, height, x, y) >= MIN_PATCH_ENERGY
        mask.append(supported and energetic)

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
                mask = local_support_mask(current_hsv, inactive_reference_hsv, valid_mask, width, height, threshold)
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
