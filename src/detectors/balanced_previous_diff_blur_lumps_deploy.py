"""Deployment-faithful lump detector with learned scoring and causal references."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - fallback for minimal installs
    ndimage = None

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    MASK_BLACK_THRESHOLD,
    DetectorResult,
    LabeledFrame,
    aligned_reference_and_valid_mask,
    estimate_shift,
    field_mask_bytes,
    foreground_path_for,
    repo_relative,
    same_capture_session,
    shifted_reference,
    shifted_valid_mask,
)

MODEL_NAME = "balanced_previous_diff_blur_lumps_deploy"
FEATURE_NAMES = (
    "diff_strength",
    "changed_ratio",
    "largest_blob_ratio",
    "blob_count",
    "artifact_ratio",
    "shift_pixels",
    "time_sin",
    "time_cos",
    "rgb_color_support",
)
ARTIFACT_RATIO_INDEX = FEATURE_NAMES.index("artifact_ratio")
SHIFT_PIXELS_INDEX = FEATURE_NAMES.index("shift_pixels")
TIME_SIN_INDEX = FEATURE_NAMES.index("time_sin")
TIME_COS_INDEX = FEATURE_NAMES.index("time_cos")


@dataclass(frozen=True)
class FramePixels:
    gray: bytes
    rgb: bytes
    width: int
    height: int


@dataclass(frozen=True)
class ReferenceFrame:
    frame: LabeledFrame
    pixels: FramePixels


@dataclass(frozen=True)
class LumpOptions:
    min_lump_area: int
    max_lump_area: int
    min_density: float
    max_aspect_ratio: float
    artifact_penalty: float
    reference_update_mode: str
    reference_strategy: str
    high_confidence_inactive: float
    hysteresis_margin: float
    rgb_color_weight: float
    logistic_iterations: int
    logistic_learning_rate: float
    logistic_l2: float
    auto_tune_lumps: bool


@dataclass(frozen=True)
class LumpFeatures:
    values: tuple[float, ...]
    legacy_score: float
    diff_strength: float
    changed_ratio: float
    largest_blob_ratio: float
    blob_count: int
    artifact_ratio: float
    shift_x: int
    shift_y: int
    rgb_color_support: float
    clean_mask: np.ndarray


@dataclass(frozen=True)
class CachedFrameFeatures:
    frame: LabeledFrame
    current: FramePixels
    features: LumpFeatures | None


@dataclass(frozen=True)
class LogisticModel:
    weights: np.ndarray
    bias: float
    means: np.ndarray
    scales: np.ndarray
    fallback_weights: np.ndarray

    def probability(self, features: tuple[float, ...]) -> float:
        values = np.asarray(features, dtype=np.float64)
        normalized = (values - self.means) / self.scales
        logit = float(np.dot(normalized, self.weights) + self.bias)
        if not np.any(self.weights):
            logit = float(np.dot(values, self.fallback_weights))
        return sigmoid(logit)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def bool_option(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def options_from(raw_options: dict[str, Any] | None, min_blob_area: int) -> LumpOptions:
    raw = raw_options or {}
    default_min_lump_area = max(3, min(8, round(min_blob_area * 0.35)))
    default_max_lump_area = max(120, min_blob_area * 6)
    reference_update_mode = str(raw.get("reference_update_mode", "manual_inactive"))
    reference_strategy = str(raw.get("reference_strategy", "latest"))
    if reference_update_mode not in {"predicted_inactive", "manual_inactive"}:
        reference_update_mode = "predicted_inactive"
    if reference_strategy not in {"latest", "nearest_past", "rolling_median", "same_time_of_day"}:
        reference_strategy = "rolling_median"

    return LumpOptions(
        min_lump_area=max(1, int(raw.get("min_lump_area", default_min_lump_area))),
        max_lump_area=max(1, int(raw.get("max_lump_area", default_max_lump_area))),
        min_density=max(0.01, float(raw.get("min_density", 0.20))),
        max_aspect_ratio=max(1.0, float(raw.get("max_aspect_ratio", 5.0))),
        artifact_penalty=max(0.0, float(raw.get("artifact_penalty", 1.5))),
        reference_update_mode=reference_update_mode,
        reference_strategy=reference_strategy,
        high_confidence_inactive=min(1.0, max(0.5, float(raw.get("high_confidence_inactive", 0.75)))),
        hysteresis_margin=max(0.0, float(raw.get("hysteresis_margin", 0.10))),
        rgb_color_weight=max(0.0, float(raw.get("rgb_color_weight", 0.0))),
        logistic_iterations=max(50, int(raw.get("logistic_iterations", 350))),
        logistic_learning_rate=max(0.001, float(raw.get("logistic_learning_rate", 0.18))),
        logistic_l2=max(0.0, float(raw.get("logistic_l2", 0.02))),
        auto_tune_lumps=bool_option(raw.get("auto_tune_lumps"), False),
    )


def compact_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def options_label(options: LumpOptions) -> str:
    return (
        f"area{options.min_lump_area}-{options.max_lump_area}_"
        f"d{compact_float(options.min_density)}_"
        f"as{compact_float(options.max_aspect_ratio)}_"
        f"pen{compact_float(options.artifact_penalty)}_"
        f"rgb{compact_float(options.rgb_color_weight)}_"
        f"{options.reference_strategy}_"
        f"{options.reference_update_mode}"
    )


def candidate_option_profiles(options: LumpOptions) -> list[LumpOptions]:
    candidates = [
        options,
        replace(
            options,
            min_lump_area=7,
            max_lump_area=120,
            min_density=0.20,
            max_aspect_ratio=5.0,
            artifact_penalty=1.5,
            rgb_color_weight=0.0,
        ),
        replace(
            options,
            min_lump_area=7,
            max_lump_area=180,
            min_density=0.18,
            max_aspect_ratio=6.0,
            artifact_penalty=1.8,
            rgb_color_weight=0.0,
        ),
        replace(
            options,
            min_lump_area=7,
            max_lump_area=240,
            min_density=0.16,
            max_aspect_ratio=6.5,
            artifact_penalty=2.2,
            rgb_color_weight=0.20,
        ),
        replace(
            options,
            min_lump_area=5,
            max_lump_area=260,
            min_density=0.14,
            max_aspect_ratio=8.0,
            artifact_penalty=2.8,
            rgb_color_weight=0.25,
        ),
        replace(
            options,
            min_lump_area=10,
            max_lump_area=180,
            min_density=0.22,
            max_aspect_ratio=5.0,
            artifact_penalty=3.0,
            rgb_color_weight=0.25,
        ),
        replace(
            options,
            min_lump_area=12,
            max_lump_area=140,
            min_density=0.28,
            max_aspect_ratio=4.5,
            artifact_penalty=3.5,
            rgb_color_weight=0.10,
        ),
        replace(
            options,
            min_lump_area=4,
            max_lump_area=220,
            min_density=0.16,
            max_aspect_ratio=6.0,
            artifact_penalty=2.5,
            rgb_color_weight=0.20,
        ),
        replace(
            options,
            min_lump_area=7,
            max_lump_area=320,
            min_density=0.18,
            max_aspect_ratio=7.0,
            artifact_penalty=2.5,
            rgb_color_weight=0.40,
        ),
    ]
    unique: dict[str, LumpOptions] = {}
    for candidate in candidates:
        unique.setdefault(options_label(candidate), candidate)
    return list(unique.values())


@lru_cache(maxsize=512)
def load_pixels(path: str, blur_radius: float) -> FramePixels:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        if blur_radius > 0:
            rgb_image = rgb_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        gray_image = rgb_image.convert("L")
        width, height = rgb_image.size
        return FramePixels(
            gray=gray_image.tobytes(),
            rgb=rgb_image.tobytes(),
            width=width,
            height=height,
        )


def frame_pixels(frame: LabeledFrame, blur_radius: float) -> FramePixels:
    return load_pixels(str(frame.masked_path), float(blur_radius))


def parse_time(frame: LabeledFrame) -> datetime | None:
    if not frame.timestamp:
        return None
    try:
        return datetime.fromisoformat(frame.timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def minute_of_day(frame: LabeledFrame) -> int | None:
    parsed = parse_time(frame)
    if parsed is None:
        return None
    return parsed.hour * 60 + parsed.minute


def time_features(frame: LabeledFrame) -> tuple[float, float]:
    minute = minute_of_day(frame)
    if minute is None:
        return 0.0, 1.0
    angle = (2 * math.pi * minute) / (24 * 60)
    return math.sin(angle), math.cos(angle)


def filtered_references(references: list[ReferenceFrame], frame: LabeledFrame) -> list[ReferenceFrame]:
    return [reference for reference in references if same_capture_session(reference.frame, frame)]


def minute_distance(first: int, second: int) -> int:
    distance = abs(first - second)
    return min(distance, (24 * 60) - distance)


def median_pixels(frame: LabeledFrame, references: list[ReferenceFrame]) -> ReferenceFrame:
    if len(references) == 1:
        return references[0]

    width = references[-1].pixels.width
    height = references[-1].pixels.height
    gray_stack = np.stack(
        [np.frombuffer(reference.pixels.gray, dtype=np.uint8) for reference in references],
        axis=0,
    )
    rgb_stack = np.stack(
        [np.frombuffer(reference.pixels.rgb, dtype=np.uint8) for reference in references],
        axis=0,
    )
    gray = np.median(gray_stack, axis=0).astype(np.uint8).tobytes()
    rgb = np.median(rgb_stack, axis=0).astype(np.uint8).tobytes()
    return ReferenceFrame(frame=references[-1].frame, pixels=FramePixels(gray=gray, rgb=rgb, width=width, height=height))


def select_reference(
    frame: LabeledFrame,
    references: list[ReferenceFrame],
    strategy: str,
    window: int,
) -> ReferenceFrame | None:
    candidates = filtered_references(references, frame)
    if not candidates:
        return None

    if strategy == "rolling_median":
        count = max(1, window)
        return median_pixels(frame, candidates[-count:])

    if strategy == "same_time_of_day":
        target_minute = minute_of_day(frame)
        if target_minute is not None:
            return min(
                candidates,
                key=lambda reference: minute_distance(minute_of_day(reference.frame) or target_minute, target_minute),
            )

    return candidates[-1]


def shift_rgb_reference(
    current_rgb: bytes,
    reference_rgb: bytes,
    width: int,
    height: int,
    dx: int,
    dy: int,
) -> bytes:
    if dx == 0 and dy == 0:
        return reference_rgb

    aligned = bytearray(current_rgb)
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
            aligned[current_offset : current_offset + 3] = reference_rgb[reference_offset : reference_offset + 3]

    return bytes(aligned)


def supported_change_mask(mask: np.ndarray, min_support: int = 2) -> np.ndarray:
    support = local_support_count(mask)
    return mask & (support >= min_support)


def local_support_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    support = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return support


def fill_small_gaps(mask: np.ndarray) -> np.ndarray:
    """Fill one-pixel holes/gaps without growing isolated speckles."""
    support = local_support_count(mask)
    return mask | (support >= 5)


def component_core_pixels(component: list[int], component_set: set[int], width: int, height: int) -> int:
    core_pixels = 0
    for index in component:
        x = index % width
        y = index // width
        support = 0
        for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
            row = neighbor_y * width
            for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                if row + neighbor_x in component_set:
                    support += 1
        if support >= 5:
            core_pixels += 1
    return core_pixels


def scipy_lump_mask(
    mask: np.ndarray,
    width: int,
    height: int,
    options: LumpOptions,
) -> tuple[np.ndarray, list[int], int]:
    labeled, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labeled)
    kept = np.zeros((height, width), dtype=bool)
    kept_areas: list[int] = []
    artifact_area = 0

    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        component = labeled[y_slice, x_slice] == component_id
        area = int(np.count_nonzero(component))
        if area == 0:
            continue

        blob_height, blob_width = component.shape
        bbox_area = blob_width * blob_height
        density = area / bbox_area
        aspect_ratio = max(blob_width, blob_height) / max(1, min(blob_width, blob_height))
        core_pixels = int(np.count_nonzero(local_support_count(component) >= 5))

        person_sized = options.min_lump_area <= area <= options.max_lump_area
        compact = density >= options.min_density and aspect_ratio <= options.max_aspect_ratio
        thick_enough = core_pixels >= 1 or (area <= 12 and density >= 0.35 and blob_width >= 2 and blob_height >= 2)

        if person_sized and compact and thick_enough:
            kept[y_slice, x_slice] |= component
            kept_areas.append(area)
        elif area > options.max_lump_area or aspect_ratio > options.max_aspect_ratio * 1.2 or density < options.min_density * 0.9:
            artifact_area += area

    del component_count
    return kept, kept_areas, artifact_area


def lump_mask(
    mask: np.ndarray,
    width: int,
    height: int,
    options: LumpOptions,
) -> tuple[np.ndarray, list[int], int]:
    if ndimage is not None:
        return scipy_lump_mask(mask, width, height, options)

    flat_mask = mask.reshape(-1)
    visited = bytearray(width * height)
    kept = np.zeros(width * height, dtype=bool)
    kept_areas: list[int] = []
    artifact_area = 0

    for start, is_foreground in enumerate(flat_mask):
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

            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                row = neighbor_y * width
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row + neighbor_x
                    if neighbor == index:
                        continue
                    if flat_mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)

        area = len(pixels)
        blob_width = max_x - min_x + 1
        blob_height = max_y - min_y + 1
        bbox_area = blob_width * blob_height
        density = area / bbox_area
        aspect_ratio = max(blob_width, blob_height) / max(1, min(blob_width, blob_height))
        component_set = set(pixels)
        core_pixels = component_core_pixels(pixels, component_set, width, height)

        person_sized = options.min_lump_area <= area <= options.max_lump_area
        compact = density >= options.min_density and aspect_ratio <= options.max_aspect_ratio
        thick_enough = core_pixels >= 1 or (area <= 12 and density >= 0.35 and blob_width >= 2 and blob_height >= 2)

        if person_sized and compact and thick_enough:
            kept[pixels] = True
            kept_areas.append(area)
        elif area > options.max_lump_area or aspect_ratio > options.max_aspect_ratio * 1.2 or density < options.min_density * 0.9:
            artifact_area += area

    return kept.reshape((height, width)), kept_areas, artifact_area


def rgb_field_like(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, 0].astype(np.int16)
    green = rgb[:, 1].astype(np.int16)
    blue = rgb[:, 2].astype(np.int16)
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)
    spread = max_channel - min_channel
    blue_green_dominant = (blue >= red + 8) | (green >= red + 8)
    return (max_channel > 35) & (spread > 12) & blue_green_dominant


def rgb_candidate_mask(
    current_rgb: bytes,
    reference_rgb: bytes,
    diff_values: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find likely people/equipment pixels that grayscale can miss.

    The field is mostly blue/green. This mask only promotes pixels where the
    reference looked field-like and the current pixel now looks non-field-like,
    with explicit support for white and black clothing/equipment.
    """
    current = np.frombuffer(current_rgb, dtype=np.uint8).reshape((-1, 3))
    reference = np.frombuffer(reference_rgb, dtype=np.uint8).reshape((-1, 3))
    current_i = current.astype(np.int16)
    reference_i = reference.astype(np.int16)
    valid_flat = valid_mask.reshape(-1)

    current_field = rgb_field_like(current)
    reference_field = rgb_field_like(reference)
    color_delta = np.max(np.abs(current_i - reference_i), axis=1)
    gray_delta = diff_values
    current_max = np.max(current_i, axis=1)
    current_min = np.min(current_i, axis=1)
    current_spread = current_max - current_min

    strong_color_change = color_delta >= max(22, threshold * 1.8)
    enough_gray_change = gray_delta >= max(5, threshold * 0.45)
    non_field_on_field = reference_field & ~current_field

    white_object = (
        reference_field
        & (current_min >= 105)
        & (current_max >= 145)
        & (current_spread <= 70)
        & enough_gray_change
    )
    black_object = (
        reference_field
        & (current_max <= 90)
        & (gray_delta >= max(8, threshold * 0.60))
    )
    colored_object = non_field_on_field & (strong_color_change | enough_gray_change)
    color_candidate = valid_flat & (white_object | black_object | colored_object)

    field_to_field_shimmer = (
        valid_flat
        & current_field
        & reference_field
        & (color_delta < max(18, threshold * 1.5))
        & (gray_delta < max(18, threshold * 1.5))
    )
    return color_candidate.reshape((height, width)), field_to_field_shimmer.reshape((height, width))


def rgb_color_support(
    current_rgb: bytes,
    reference_rgb: bytes,
    clean_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    total_pixels = int(np.count_nonzero(valid_mask))
    if total_pixels == 0 or not np.any(clean_mask):
        return 0.0

    current = np.frombuffer(current_rgb, dtype=np.uint8).reshape((-1, 3)).astype(np.float32)
    reference = np.frombuffer(reference_rgb, dtype=np.uint8).reshape((-1, 3)).astype(np.float32)
    flat_clean = clean_mask.reshape(-1)
    current_clean = current[flat_clean]
    reference_clean = reference[flat_clean]
    if current_clean.size == 0:
        return 0.0

    color_distance = np.linalg.norm(current_clean - reference_clean, axis=1) / (255.0 * math.sqrt(3))
    current_non_field = ~rgb_field_like(current_clean.astype(np.uint8))
    reference_field = rgb_field_like(reference_clean.astype(np.uint8))
    person_color_ratio = float(np.count_nonzero(current_non_field & reference_field)) / total_pixels
    color_strength = float(color_distance.sum()) / total_pixels
    return color_strength + person_color_ratio


def save_mask(mask: np.ndarray, width: int, height: int, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (width, height))
    image.putdata([255 if value else 0 for value in mask.reshape(-1)])
    image.save(output_path)
    return repo_relative(output_path)


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

    current_array = np.frombuffer(current, dtype=np.uint8)
    reference_array = np.frombuffer(aligned_reference, dtype=np.uint8)
    valid = (
        (np.frombuffer(current_valid, dtype=np.uint8) > 0)
        & (np.frombuffer(reference_valid, dtype=np.uint8) > 0)
        & (current_array > MASK_BLACK_THRESHOLD)
        & (reference_array > MASK_BLACK_THRESHOLD)
    )

    if int(np.count_nonzero(valid)) >= 100:
        current_values = current_array[valid].astype(np.float32)
        reference_values = reference_array[valid].astype(np.float32)
        current_mean = float(current_values.mean())
        reference_mean = float(reference_values.mean())
        current_std = float(current_values.std())
        reference_std = float(reference_values.std())
        scale = 1.0 if reference_std < 1 else current_std / reference_std
        offset = current_mean - (scale * reference_mean)
        matched = reference_array.astype(np.float32)
        matched[valid] = np.clip((scale * matched[valid]) + offset, 0, 255)
        matched_reference = np.rint(matched).astype(np.uint8)
    else:
        matched_reference = reference_array.copy()

    valid_2d = valid.reshape((height, width))
    matched_2d = matched_reference.reshape((height, width)).astype(np.int16)
    center = matched_2d[1:-1, 1:-1]
    contrast = np.maximum.reduce(
        [
            np.abs(center - matched_2d[1:-1, :-2]),
            np.abs(center - matched_2d[1:-1, 2:]),
            np.abs(center - matched_2d[:-2, 1:-1]),
            np.abs(center - matched_2d[2:, 1:-1]),
        ]
    )
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[1:-1, 1:-1] = valid_2d[1:-1, 1:-1] & (contrast >= 22)
    if ndimage is not None:
        edge_mask = ndimage.binary_dilation(edge_mask, structure=np.ones((3, 3), dtype=bool))
    else:
        edge_mask = fill_small_gaps(edge_mask)
    valid_2d = valid_2d & ~edge_mask
    return matched_reference.tobytes(), (valid_2d.astype(np.uint8) * 255).tobytes(), dx, dy


def extract_features(
    frame: LabeledFrame,
    current: FramePixels,
    reference: ReferenceFrame,
    threshold: float,
    max_shift_pixels: int,
    options: LumpOptions,
) -> LumpFeatures:
    width, height = current.width, current.height
    matched_reference, valid_mask_bytes, shift_x, shift_y = aligned_reference_and_valid_mask(
        current.gray,
        reference.pixels.gray,
        width,
        height,
        max_shift_pixels,
    )
    current_values = np.frombuffer(current.gray, dtype=np.uint8).astype(np.int16)
    reference_values = np.frombuffer(matched_reference, dtype=np.uint8).astype(np.int16)
    valid_mask = np.frombuffer(valid_mask_bytes, dtype=np.uint8).reshape((height, width)) > 0
    diff_values = np.where(valid_mask.reshape(-1), np.abs(current_values - reference_values), 0)
    aligned_rgb_reference = shift_rgb_reference(current.rgb, reference.pixels.rgb, width, height, shift_x, shift_y)
    gray_mask = diff_values.reshape((height, width)) > threshold
    if options.rgb_color_weight > 0:
        color_mask, field_shimmer_mask = rgb_candidate_mask(
            current.rgb,
            aligned_rgb_reference,
            diff_values,
            valid_mask,
            threshold,
            width,
            height,
        )
        raw_mask = (gray_mask | color_mask) & ~field_shimmer_mask
    else:
        raw_mask = gray_mask
    supported_mask = supported_change_mask(raw_mask)
    supported_mask = supported_change_mask(fill_small_gaps(supported_mask))
    clean_mask, component_areas, artifact_area = lump_mask(supported_mask, width, height, options)

    total_pixels = int(np.count_nonzero(valid_mask))
    if total_pixels == 0:
        values = (0.0, 0.0, 0.0, 0.0, 0.0, float(abs(shift_x) + abs(shift_y)), *time_features(frame), 0.0)
        return LumpFeatures(values, 0.0, 0.0, 0.0, 0.0, 0, 0.0, shift_x, shift_y, 0.0, clean_mask)

    clean_flat = clean_mask.reshape(-1)
    changed_pixels = int(np.count_nonzero(clean_flat))
    largest_blob = max(component_areas, default=0)
    diff_strength = float(diff_values[clean_flat].sum()) / (255.0 * total_pixels)
    changed_ratio = changed_pixels / total_pixels
    largest_blob_ratio = largest_blob / total_pixels
    artifact_ratio = artifact_area / total_pixels
    color_support = rgb_color_support(current.rgb, aligned_rgb_reference, clean_mask, valid_mask)
    positive_score = (
        diff_strength
        + (0.40 * changed_ratio)
        + (0.0015 * min(len(component_areas), 10))
        + (0.25 * min(largest_blob_ratio, 0.01))
        + (options.rgb_color_weight * color_support)
    )
    legacy_score = positive_score / (1 + (options.artifact_penalty * artifact_ratio))
    time_sin, time_cos = time_features(frame)
    shift_pixels = abs(shift_x) + abs(shift_y)
    values = (
        diff_strength,
        changed_ratio,
        largest_blob_ratio,
        float(len(component_areas)),
        artifact_ratio,
        float(shift_pixels),
        time_sin,
        time_cos,
        options.rgb_color_weight * color_support,
    )
    return LumpFeatures(
        values=values,
        legacy_score=legacy_score,
        diff_strength=diff_strength,
        changed_ratio=changed_ratio,
        largest_blob_ratio=largest_blob_ratio,
        blob_count=len(component_areas),
        artifact_ratio=artifact_ratio,
        shift_x=shift_x,
        shift_y=shift_y,
        rgb_color_support=color_support,
        clean_mask=clean_mask,
    )


def fallback_weights(options: LumpOptions) -> np.ndarray:
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    weights[0] = 18.0
    weights[1] = 8.0
    weights[2] = 10.0
    weights[3] = 0.15
    weights[4] = -options.artifact_penalty
    weights[5] = -0.04
    weights[8] = 4.0
    return weights


def fit_logistic_model(samples: list[tuple[LumpFeatures, int]], options: LumpOptions) -> LogisticModel:
    fallback = fallback_weights(options)
    if len(samples) < 4 or len({label for _features, label in samples}) < 2:
        means = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        scales = np.ones(len(FEATURE_NAMES), dtype=np.float64)
        return LogisticModel(
            weights=np.zeros(len(FEATURE_NAMES), dtype=np.float64),
            bias=0.0,
            means=means,
            scales=scales,
            fallback_weights=fallback,
        )

    x = np.asarray([features.values for features, _label in samples], dtype=np.float64)
    y = np.asarray([label for _features, label in samples], dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    x_norm = (x - means) / scales

    weights = np.zeros(x_norm.shape[1], dtype=np.float64)
    bias = 0.0
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    sample_weights = np.where(y > 0, len(y) / (2 * positives), len(y) / (2 * negatives))
    sample_weights = sample_weights / sample_weights.mean()

    for _iteration in range(options.logistic_iterations):
        logits = np.clip(x_norm @ weights + bias, -40, 40)
        predictions = 1 / (1 + np.exp(-logits))
        errors = (predictions - y) * sample_weights
        gradient = (x_norm.T @ errors) / len(y) + (options.logistic_l2 * weights)
        bias_gradient = float(errors.mean())
        weights -= options.logistic_learning_rate * gradient
        bias -= options.logistic_learning_rate * bias_gradient

    # Keep learned scoring deployment-safe: artifacts and alignment instability
    # may correlate with activity in a small train split, but they should never
    # become positive evidence by themselves.
    weights[ARTIFACT_RATIO_INDEX] = min(weights[ARTIFACT_RATIO_INDEX], -0.35 * options.artifact_penalty)
    weights[SHIFT_PIXELS_INDEX] = min(weights[SHIFT_PIXELS_INDEX], 0.0)
    weights[TIME_SIN_INDEX] *= 0.25
    weights[TIME_COS_INDEX] *= 0.25

    return LogisticModel(weights=weights, bias=bias, means=means, scales=scales, fallback_weights=fallback)


def build_training_samples(
    frames: list[LabeledFrame],
    threshold: float,
    window: int,
    max_shift_pixels: int,
    blur_radius: float,
    options: LumpOptions,
    training_image_ids: set[str] | None,
) -> list[tuple[LumpFeatures, int]]:
    references: list[ReferenceFrame] = []
    samples: list[tuple[LumpFeatures, int]] = []
    train_ids = training_image_ids or {frame.image_id for frame in frames}

    for frame in frames:
        current = frame_pixels(frame, blur_radius)
        reference = select_reference(frame, references, options.reference_strategy, window)
        if frame.image_id in train_ids and reference is not None and frame.label in {"active", "inactive"}:
            features = extract_features(frame, current, reference, threshold, max_shift_pixels, options)
            samples.append((features, 1 if frame.label == "active" else 0))

        if frame.image_id in train_ids and frame.label == "inactive":
            references.append(ReferenceFrame(frame=frame, pixels=current))

    return samples


def build_manual_feature_cache(
    frames: list[LabeledFrame],
    threshold: float,
    window: int,
    max_shift_pixels: int,
    blur_radius: float,
    options: LumpOptions,
    reference_image_ids: set[str] | None = None,
) -> list[CachedFrameFeatures]:
    references: list[ReferenceFrame] = []
    cached: list[CachedFrameFeatures] = []

    for frame in frames:
        current = frame_pixels(frame, blur_radius)
        reference = select_reference(frame, references, options.reference_strategy, window)
        features = None
        if reference is not None:
            features = extract_features(frame, current, reference, threshold, max_shift_pixels, options)
        cached.append(CachedFrameFeatures(frame=frame, current=current, features=features))

        if frame.label == "inactive" and (reference_image_ids is None or frame.image_id in reference_image_ids):
            references.append(ReferenceFrame(frame=frame, pixels=current))

    return cached


def samples_from_cache(
    cached: list[CachedFrameFeatures],
    training_image_ids: set[str] | None,
) -> list[tuple[LumpFeatures, int]]:
    train_ids = training_image_ids or {item.frame.image_id for item in cached}
    samples: list[tuple[LumpFeatures, int]] = []
    for item in cached:
        if item.features is None:
            continue
        if item.frame.image_id not in train_ids or item.frame.label not in {"active", "inactive"}:
            continue
        samples.append((item.features, 1 if item.frame.label == "active" else 0))
    return samples


def active_metric_tuple(results: list[DetectorResult]) -> tuple[float, float, float, float, float]:
    active_or_inactive = [result for result in results if result.label in {"active", "inactive"}]
    if not active_or_inactive:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    true_active = sum(result.label == "active" and result.prediction == "active" for result in active_or_inactive)
    false_active = sum(result.label == "inactive" and result.prediction == "active" for result in active_or_inactive)
    false_inactive = sum(result.label == "active" and result.prediction == "inactive" for result in active_or_inactive)
    correct = sum(result.label == result.prediction for result in active_or_inactive)
    true_inactive = sum(result.label == "inactive" and result.prediction == "inactive" for result in active_or_inactive)

    precision = true_active / (true_active + false_active) if true_active + false_active else 0.0
    recall = true_active / (true_active + false_inactive) if true_active + false_inactive else 0.0
    inactive_recall = true_inactive / (true_inactive + false_active) if true_inactive + false_active else 0.0
    balanced_activity = (recall + inactive_recall) / 2
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = correct / len(active_or_inactive)
    return balanced_activity, f1, precision, recall, accuracy


def tune_options(
    frames: list[LabeledFrame],
    threshold: float,
    cutoffs: list[float],
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    options: LumpOptions,
    training_image_ids: set[str] | None,
) -> LumpOptions:
    train_ids = training_image_ids or {frame.image_id for frame in frames}
    training_frames = [
        frame
        for frame in frames
        if frame.image_id in train_ids and frame.label in {"active", "inactive"}
    ]
    if len(training_frames) < 4 or len({frame.label for frame in training_frames}) < 2:
        return options

    candidate_cutoffs = cutoffs or [0.03]
    best_options = options
    best_metrics = (-1.0, -1.0, -1.0, -1.0, -1.0)
    for candidate in candidate_option_profiles(options):
        if candidate.reference_update_mode == "manual_inactive":
            cached = build_manual_feature_cache(
                training_frames,
                threshold,
                window,
                max_shift_pixels,
                blur_radius,
                candidate,
            )
            samples = samples_from_cache(cached, train_ids)
        else:
            cached = []
            samples = build_training_samples(
                training_frames,
                threshold,
                window,
                max_shift_pixels,
                blur_radius,
                candidate,
                train_ids,
            )
        model = fit_logistic_model(samples, candidate)
        candidate_metrics = (-1.0, -1.0, -1.0, -1.0, -1.0)
        for cutoff in candidate_cutoffs:
            trial_results = (
                evaluate_cached_cutoff(
                    cached,
                    threshold,
                    cutoff,
                    window,
                    max_shift_pixels,
                    blur_radius,
                    foreground_root,
                    candidate,
                    model,
                    save_foreground=False,
                    config_label=options_label(candidate),
                )
                if candidate.reference_update_mode == "manual_inactive"
                else evaluate_one_cutoff(
                    training_frames,
                    threshold,
                    cutoff,
                    window,
                    min_blob_area,
                    max_shift_pixels,
                    blur_radius,
                    foreground_root,
                    candidate,
                    model,
                    train_ids,
                    save_foreground=False,
                    config_label=options_label(candidate),
                )
            )
            candidate_metrics = max(candidate_metrics, active_metric_tuple(trial_results))

        if candidate_metrics > best_metrics:
            best_metrics = candidate_metrics
            best_options = candidate

    return best_options


def inactive_confidence(active_probability: float) -> float:
    return 1.0 - active_probability


def prediction_for_probability(active_probability: float, cutoff: float, was_active: bool, hysteresis_margin: float) -> str:
    if was_active:
        off_cutoff = max(0.0, cutoff - hysteresis_margin)
        return "active" if active_probability >= off_cutoff else "inactive"
    return "active" if active_probability >= cutoff else "inactive"


def is_legacy_cutoff(cutoff: float) -> bool:
    return cutoff <= 0.30


def prediction_for_score(score: float, cutoff: float, was_active: bool, hysteresis_margin: float) -> str:
    if was_active:
        score_margin = min(hysteresis_margin, max(0.001, cutoff * 0.25))
        return "active" if score >= max(0.0, cutoff - score_margin) else "inactive"
    return "active" if score >= cutoff else "inactive"


def confidence_for_score(score: float, cutoff: float, prediction: str) -> float:
    scale = max(0.001, cutoff)
    distance = abs(score - cutoff) / scale
    confidence = min(1.0, 0.5 + (0.25 * distance))
    if prediction == "inactive" and score == 0:
        return max(confidence, 0.75)
    return confidence


def should_add_reference(
    frame: LabeledFrame,
    prediction: str,
    confidence: float,
    current: FramePixels,
    references: list[ReferenceFrame],
    options: LumpOptions,
    training_image_ids: set[str] | None,
) -> bool:
    if options.reference_update_mode == "manual_inactive":
        return frame.label == "inactive"

    if not references and frame.label == "inactive" and (training_image_ids is None or frame.image_id in training_image_ids):
        return True

    del current
    return prediction == "inactive" and confidence >= options.high_confidence_inactive


def unavailable_reference_result(
    frame: LabeledFrame,
    model: str,
    threshold: float,
    cutoff: float,
    window: int,
    blur_radius: float,
    max_shift_pixels: int,
    foreground_root: Path,
    save_foreground: bool = True,
    config_label: str = "",
) -> DetectorResult:
    current = frame_pixels(frame, blur_radius)
    if save_foreground:
        foreground_path = save_mask(
            np.zeros((current.height, current.width), dtype=bool),
            current.width,
            current.height,
            foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
        )
    else:
        foreground_path = ""
    return DetectorResult(
        model=model,
        threshold=threshold,
        cutoff=cutoff,
        window=window,
        blur_radius=blur_radius,
        image_id=frame.image_id,
        timestamp=frame.timestamp,
        label=frame.label,
        prediction="inactive",
        score=0.0,
        changed_ratio=0.0,
        largest_blob_ratio=0.0,
        blob_count=0,
        camera_shift_x=0,
        camera_shift_y=0,
        foreground_path=foreground_path,
        confidence=0.5,
        config_label=config_label,
    )


def evaluate_cached_cutoff(
    cached: list[CachedFrameFeatures],
    threshold: float,
    cutoff: float,
    window: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    options: LumpOptions,
    model: LogisticModel,
    save_foreground: bool = True,
    config_label: str = "",
    foreground_paths: dict[str, str] | None = None,
) -> list[DetectorResult]:
    results: list[DetectorResult] = []
    was_active = False
    saved_paths = foreground_paths if foreground_paths is not None else {}

    for item in cached:
        frame = item.frame
        current = item.current
        cache_key = frame.image_id
        if item.features is None:
            foreground_path = ""
            if save_foreground:
                foreground_path = saved_paths.get(cache_key, "")
                if not foreground_path:
                    foreground_path = save_mask(
                        np.zeros((current.height, current.width), dtype=bool),
                        current.width,
                        current.height,
                        foreground_path_for(MODEL_NAME, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
                    )
                    saved_paths[cache_key] = foreground_path
            results.append(
                DetectorResult(
                    model=MODEL_NAME,
                    threshold=threshold,
                    cutoff=cutoff,
                    window=window,
                    blur_radius=blur_radius,
                    image_id=frame.image_id,
                    timestamp=frame.timestamp,
                    label=frame.label,
                    prediction="inactive",
                    score=0.0,
                    changed_ratio=0.0,
                    largest_blob_ratio=0.0,
                    blob_count=0,
                    camera_shift_x=0,
                    camera_shift_y=0,
                    foreground_path=foreground_path,
                    confidence=0.5,
                    config_label=config_label,
                )
            )
            continue

        features = item.features
        probability = model.probability(features.values)
        if is_legacy_cutoff(cutoff):
            score = features.legacy_score
            prediction = prediction_for_score(score, cutoff, was_active, options.hysteresis_margin)
            confidence = confidence_for_score(score, cutoff, prediction)
        else:
            score = probability
            prediction = prediction_for_probability(probability, cutoff, was_active, options.hysteresis_margin)
            confidence = probability if prediction == "active" else inactive_confidence(probability)
        was_active = prediction == "active"

        foreground_path = ""
        if save_foreground:
            foreground_path = saved_paths.get(cache_key, "")
            if not foreground_path:
                foreground_path = save_mask(
                    features.clean_mask,
                    current.width,
                    current.height,
                    foreground_path_for(MODEL_NAME, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
                )
                saved_paths[cache_key] = foreground_path

        results.append(
            DetectorResult(
                model=MODEL_NAME,
                threshold=threshold,
                cutoff=cutoff,
                window=window,
                blur_radius=blur_radius,
                image_id=frame.image_id,
                timestamp=frame.timestamp,
                label=frame.label,
                prediction=prediction,
                score=score,
                changed_ratio=features.changed_ratio,
                largest_blob_ratio=features.largest_blob_ratio,
                blob_count=features.blob_count,
                camera_shift_x=features.shift_x,
                camera_shift_y=features.shift_y,
                foreground_path=foreground_path,
                confidence=confidence,
                config_label=config_label,
            )
        )

    return results


def evaluate_one_cutoff(
    frames: list[LabeledFrame],
    threshold: float,
    cutoff: float,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    options: LumpOptions,
    model: LogisticModel,
    training_image_ids: set[str] | None,
    save_foreground: bool = True,
    config_label: str = "",
) -> list[DetectorResult]:
    del min_blob_area
    references: list[ReferenceFrame] = []
    results: list[DetectorResult] = []
    was_active = False

    for frame in frames:
        current = frame_pixels(frame, blur_radius)
        reference = select_reference(frame, references, options.reference_strategy, window)
        if reference is None:
            result = unavailable_reference_result(
                frame,
                MODEL_NAME,
                threshold,
                cutoff,
                window,
                blur_radius,
                max_shift_pixels,
                foreground_root,
                save_foreground,
                config_label,
            )
            results.append(result)
            prediction = result.prediction
            confidence = result.confidence
        else:
            features = extract_features(frame, current, reference, threshold, max_shift_pixels, options)
            probability = model.probability(features.values)
            if is_legacy_cutoff(cutoff):
                score = features.legacy_score
                prediction = prediction_for_score(score, cutoff, was_active, options.hysteresis_margin)
                confidence = confidence_for_score(score, cutoff, prediction)
            else:
                score = probability
                prediction = prediction_for_probability(probability, cutoff, was_active, options.hysteresis_margin)
                confidence = probability if prediction == "active" else inactive_confidence(probability)
            was_active = prediction == "active"
            if save_foreground:
                foreground_path = save_mask(
                    features.clean_mask,
                    current.width,
                    current.height,
                    foreground_path_for(MODEL_NAME, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
                )
            else:
                foreground_path = ""
            results.append(
                DetectorResult(
                    model=MODEL_NAME,
                    threshold=threshold,
                    cutoff=cutoff,
                    window=window,
                    blur_radius=blur_radius,
                    image_id=frame.image_id,
                    timestamp=frame.timestamp,
                    label=frame.label,
                    prediction=prediction,
                    score=score,
                    changed_ratio=features.changed_ratio,
                    largest_blob_ratio=features.largest_blob_ratio,
                    blob_count=features.blob_count,
                    camera_shift_x=features.shift_x,
                    camera_shift_y=features.shift_y,
                    foreground_path=foreground_path,
                    confidence=confidence,
                    config_label=config_label,
                )
            )

        if should_add_reference(frame, prediction, confidence, current, references, options, training_image_ids):
            references.append(ReferenceFrame(frame=frame, pixels=current))

    return results


def evaluate(
    frames: list[LabeledFrame],
    thresholds: list[int | float],
    cutoffs: list[float],
    windows: list[int],
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path = DEFAULT_FOREGROUND_ROOT,
    detector_options: dict[str, Any] | None = None,
    training_image_ids: set[str] | None = None,
) -> list[DetectorResult]:
    results: list[DetectorResult] = []
    options = options_from(detector_options, min_blob_area)
    active_windows = windows or [5]

    for threshold in thresholds:
        for window in active_windows:
            active_options = (
                tune_options(
                    frames,
                    float(threshold),
                    cutoffs,
                    window,
                    min_blob_area,
                    max_shift_pixels,
                    blur_radius,
                    foreground_root,
                    options,
                    training_image_ids,
                )
                if options.auto_tune_lumps
                else options
            )
            config_label = options_label(active_options)
            if active_options.reference_update_mode == "manual_inactive":
                sample_cache = build_manual_feature_cache(
                    frames,
                    float(threshold),
                    window,
                    max_shift_pixels,
                    blur_radius,
                    active_options,
                    training_image_ids,
                )
                samples = samples_from_cache(sample_cache, training_image_ids)
                eval_cache = (
                    sample_cache
                    if training_image_ids is None
                    else build_manual_feature_cache(
                        frames,
                        float(threshold),
                        window,
                        max_shift_pixels,
                        blur_radius,
                        active_options,
                    )
                )
            else:
                eval_cache = []
                samples = build_training_samples(
                    frames,
                    float(threshold),
                    window,
                    max_shift_pixels,
                    blur_radius,
                    active_options,
                    training_image_ids,
                )
            model = fit_logistic_model(samples, active_options)
            foreground_paths: dict[str, str] = {}
            for cutoff in cutoffs:
                if active_options.reference_update_mode == "manual_inactive":
                    results.extend(
                        evaluate_cached_cutoff(
                            eval_cache,
                            float(threshold),
                            cutoff,
                            window,
                            max_shift_pixels,
                            blur_radius,
                            foreground_root,
                            active_options,
                            model,
                            config_label=config_label,
                            foreground_paths=foreground_paths,
                        )
                    )
                else:
                    results.extend(
                        evaluate_one_cutoff(
                            frames,
                            float(threshold),
                            cutoff,
                            window,
                            min_blob_area,
                            max_shift_pixels,
                            blur_radius,
                            foreground_root,
                            active_options,
                            model,
                            training_image_ids,
                            config_label=config_label,
                        )
                    )

    return results
