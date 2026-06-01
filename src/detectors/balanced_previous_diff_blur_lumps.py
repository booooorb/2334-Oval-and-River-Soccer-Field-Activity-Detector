"""Lump-focused balanced blurred previous-frame subtraction detector."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageFilter

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectionFeatures,
    DetectorResult,
    LabeledFrame,
    aligned_reference_and_valid_mask,
    expand_cutoffs,
    foreground_path_for,
    image_bytes,
    load_gray,
    same_capture_session,
    save_foreground_mask,
)

MODEL_NAME = "balanced_previous_diff_blur_lumps"


def supported_change_mask(mask: list[bool], width: int, height: int, min_support: int = 2) -> list[bool]:
    supported = [False] * len(mask)
    for index, is_changed in enumerate(mask):
        if not is_changed:
            continue

        x = index % width
        y = index // width
        support = 0
        for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
            row = neighbor_y * width
            for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                if mask[row + neighbor_x]:
                    support += 1

        if support >= min_support:
            supported[index] = True

    return supported


def component_support_count(component: list[int], component_set: set[int], width: int, height: int) -> int:
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


def lump_mask(
    mask: list[bool],
    width: int,
    height: int,
    min_blob_area: int,
) -> tuple[list[bool], list[list[int]], int]:
    min_lump_area = max(3, min(8, round(min_blob_area * 0.35)))
    max_lump_area = max(120, min_blob_area * 6)
    visited = bytearray(width * height)
    kept = [False] * (width * height)
    kept_components: list[list[int]] = []
    artifact_area = 0

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

            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                row = neighbor_y * width
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row + neighbor_x
                    if neighbor == index:
                        continue
                    if mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)

        area = len(pixels)
        blob_width = max_x - min_x + 1
        blob_height = max_y - min_y + 1
        bbox_area = blob_width * blob_height
        density = area / bbox_area
        aspect_ratio = max(blob_width, blob_height) / max(1, min(blob_width, blob_height))
        component_set = set(pixels)
        core_pixels = component_support_count(pixels, component_set, width, height)

        person_sized = min_lump_area <= area <= max_lump_area
        compact = density >= 0.20 and aspect_ratio <= 5.0
        thick_enough = core_pixels >= 1 or (area <= 12 and density >= 0.35 and blob_width >= 2 and blob_height >= 2)

        if person_sized and compact and thick_enough:
            for pixel in pixels:
                kept[pixel] = True
            kept_components.append(pixels)
        elif area > max_lump_area or aspect_ratio > 6.0 or density < 0.18:
            artifact_area += area

    return kept, kept_components, artifact_area


def lump_score(
    mask: list[bool],
    components: list[list[int]],
    artifact_area: int,
    diff_values: list[int],
    valid_mask: bytes,
) -> tuple[float, float, float, int]:
    total_pixels = sum(1 for value in valid_mask if value)
    if not total_pixels:
        return 0.0, 0.0, 0.0, 0

    changed_pixels = sum(mask)
    largest_blob = max((len(component) for component in components), default=0)
    diff_strength = sum(diff_values[index] for index, is_changed in enumerate(mask) if is_changed) / (255 * total_pixels)
    changed_ratio = changed_pixels / total_pixels
    largest_blob_ratio = largest_blob / total_pixels
    artifact_ratio = artifact_area / total_pixels
    blob_count = len(components)
    positive_score = (
        diff_strength
        + (0.40 * changed_ratio)
        + (0.0015 * min(blob_count, 10))
        + (0.25 * min(largest_blob_ratio, 0.01))
    )
    score = positive_score / (1 + (1.5 * artifact_ratio))
    return score, changed_ratio, largest_blob_ratio, blob_count


def detect_lumps_against_reference(
    frame: LabeledFrame,
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    threshold: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
) -> DetectionFeatures:
    matched_reference, valid_mask, shift_x, shift_y = aligned_reference_and_valid_mask(
        current,
        reference,
        width,
        height,
        max_shift_pixels,
    )
    diff_values = [
        abs(pixel - base) if valid else 0
        for pixel, base, valid in zip(current, matched_reference, valid_mask)
    ]
    raw_mask = [value > threshold for value in diff_values]
    supported_mask = supported_change_mask(raw_mask, width, height)
    clean_mask, components, artifact_area = lump_mask(supported_mask, width, height, min_blob_area)
    score, changed_ratio, largest_blob_ratio, blob_count = lump_score(
        clean_mask,
        components,
        artifact_area,
        diff_values,
        valid_mask,
    )
    foreground_path = save_foreground_mask(
        clean_mask,
        width,
        height,
        foreground_path_for(MODEL_NAME, threshold, 1, max_shift_pixels, blur_radius, frame, foreground_root),
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
        inactive_reference: bytes | None = None
        inactive_reference_frame: LabeledFrame | None = None

        for frame in frames:
            image = load_gray(frame.masked_path).filter(ImageFilter.GaussianBlur(radius=blur_radius))
            current = image_bytes(image)
            width, height = image.size

            if inactive_reference is not None and inactive_reference_frame is not None and same_capture_session(inactive_reference_frame, frame):
                features = detect_lumps_against_reference(
                    frame,
                    current,
                    inactive_reference,
                    width,
                    height,
                    int(threshold),
                    min_blob_area,
                    max_shift_pixels,
                    blur_radius,
                    foreground_root,
                )
                results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, 1, features))

            if frame.label == "inactive":
                inactive_reference = current
                inactive_reference_frame = frame

    return results
