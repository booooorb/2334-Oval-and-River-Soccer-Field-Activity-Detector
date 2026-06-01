"""MOG2/GMM foreground detector gated by previous-frame motion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectionFeatures,
    DetectorResult,
    LabeledFrame,
    MASK_BLACK_THRESHOLD,
    brightness_matched_reference,
    combined_valid_mask,
    estimate_shift,
    expand_cutoffs,
    field_mask_bytes,
    foreground_path_for,
    repo_relative,
    shifted_reference,
    shifted_valid_mask,
)
from src.detectors.lab_bilateral_previous_diff import cv2_module, read_bgr, save_array_image

MODEL_NAME = "gmm_mog2_foreground_motion"
BACKGROUND_IMAGE_NAME = "background_reference.png"
FOREGROUND_MEMORY_LENGTH = 3
FOREGROUND_MEMORY_DILATION_RADIUS = 2
STATIC_FOREGROUND_RATIO_THRESHOLD = 0.10
STATIC_FOREGROUND_DILATION_RADIUS = 2


def train_mog2_background(
    images: list[np.ndarray],
    *,
    history: int,
    var_threshold: float,
) -> tuple[object, np.ndarray]:
    cv2 = cv2_module()
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=max(2, history),
        varThreshold=var_threshold,
        detectShadows=False,
    )
    for image in images:
        subtractor.apply(image, learningRate=-1)

    background = subtractor.getBackgroundImage()
    if background is None:
        background = np.zeros_like(images[0])
    return subtractor, background


def training_images_for_background(frames: list[LabeledFrame], images: list[np.ndarray]) -> list[np.ndarray]:
    inactive_images = [image for frame, image in zip(frames, images) if frame.label == "inactive"]
    return inactive_images or images


def shift_binary_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return mask

    height, width = mask.shape
    shifted = np.zeros_like(mask)
    current_x_start = max(0, dx)
    current_y_start = max(0, dy)
    current_x_end = min(width, width + dx)
    current_y_end = min(height, height + dy)

    reference_x_start = current_x_start - dx
    reference_y_start = current_y_start - dy
    reference_x_end = current_x_end - dx
    reference_y_end = current_y_end - dy

    shifted[current_y_start:current_y_end, current_x_start:current_x_end] = mask[
        reference_y_start:reference_y_end,
        reference_x_start:reference_x_end,
    ]
    return shifted


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    cv2 = cv2_module()
    kernel_size = (radius * 2) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1).astype(bool)


def foreground_memory_suppression_mask(
    previous_fg: np.ndarray | None,
    foreground_history: list[np.ndarray],
    shift_x: int,
    shift_y: int,
) -> np.ndarray | None:
    """Return recent foreground pixels in the current frame's coordinates."""
    memory_masks: list[np.ndarray] = []
    if previous_fg is not None:
        memory_masks.append(shift_binary_mask(previous_fg, shift_x, shift_y))
    memory_masks.extend(foreground_history[-FOREGROUND_MEMORY_LENGTH:])

    if not memory_masks:
        return None

    memory = np.zeros_like(memory_masks[0], dtype=bool)
    for mask in memory_masks:
        memory |= mask
    return dilate_mask(memory, FOREGROUND_MEMORY_DILATION_RADIUS)


def align_foreground_history(
    foreground_history: list[np.ndarray],
    shift_x: int,
    shift_y: int,
) -> list[np.ndarray]:
    return [
        shift_binary_mask(mask, shift_x, shift_y)
        for mask in foreground_history[-FOREGROUND_MEMORY_LENGTH:]
    ]


def static_foreground_prior(foregrounds: list[np.ndarray]) -> np.ndarray:
    if not foregrounds:
        raise ValueError("need at least one foreground mask")
    frequency = np.mean(np.stack(foregrounds, axis=0), axis=0)
    static_mask = frequency >= STATIC_FOREGROUND_RATIO_THRESHOLD
    return dilate_mask(static_mask, STATIC_FOREGROUND_DILATION_RADIUS)


def keep_thick_lumps(
    mask: np.ndarray,
    min_area: int,
    min_core_radius: float = 1.25,
    min_core_pixels: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    cv2 = cv2_module()
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
    clean = np.zeros(mask.shape, dtype=np.uint8)
    components: list[np.ndarray] = []

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        density = area / max(1, width * height)
        aspect = max(width, height) / max(1, min(width, height))
        if density < 0.18 or aspect > 6.0:
            continue

        component = labels == component_index
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
        if float(distance.max()) < min_core_radius:
            continue
        if int(np.count_nonzero(distance >= min_core_radius)) < min_core_pixels:
            continue

        clean[component] = 255
        components.append(component)

    return clean.astype(bool), components


def score_motion_lumps(
    clean_mask: np.ndarray,
    components: list[np.ndarray],
    diff_values: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float, float, int]:
    valid_pixels = int(np.count_nonzero(valid_mask))
    if valid_pixels == 0:
        return 0.0, 0.0, 0.0, 0

    changed_pixels = int(np.count_nonzero(clean_mask))
    changed_ratio = changed_pixels / valid_pixels
    largest_blob = max((int(np.count_nonzero(component)) for component in components), default=0)
    largest_blob_ratio = largest_blob / valid_pixels
    blob_count = len(components)
    diff_strength = float(np.sum(diff_values[clean_mask]) / (255 * valid_pixels))
    score = (
        diff_strength
        + (0.40 * changed_ratio)
        + (0.0015 * min(blob_count, 10))
        + (0.20 * min(largest_blob_ratio, 0.01))
    )
    return score, changed_ratio, largest_blob_ratio, blob_count


def save_debug_images(
    foreground_path: Path,
    background_bgr: np.ndarray,
    gmm_fg: np.ndarray,
    foreground_memory: np.ndarray,
    static_prior: np.ndarray,
    fg_minus_previous: np.ndarray,
    motion_mask: np.ndarray,
    final_raw: np.ndarray,
    clean_mask: np.ndarray,
) -> str:
    cv2 = cv2_module()
    debug_dir = foreground_path.parent / f"{foreground_path.stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    background_path = foreground_path.parent.parent / BACKGROUND_IMAGE_NAME
    cv2.imwrite(str(background_path), background_bgr)
    save_array_image(debug_dir / "gmm_fg.png", gmm_fg.astype(np.uint8) * 255)
    save_array_image(debug_dir / "static_foreground_prior.png", static_prior.astype(np.uint8) * 255)
    save_array_image(debug_dir / "foreground_memory_suppression.png", foreground_memory.astype(np.uint8) * 255)
    save_array_image(debug_dir / "fg_minus_previous_fg.png", fg_minus_previous.astype(np.uint8) * 255)
    save_array_image(debug_dir / "previous_motion.png", motion_mask.astype(np.uint8) * 255)
    save_array_image(debug_dir / "gmm_fg_and_motion.png", (gmm_fg & motion_mask).astype(np.uint8) * 255)
    save_array_image(debug_dir / "final_raw_new_fg_and_motion.png", final_raw.astype(np.uint8) * 255)
    save_array_image(debug_dir / "cleaned_mask.png", clean_mask.astype(np.uint8) * 255)
    return repo_relative(background_path)


def gmm_motion_features(
    frame: LabeledFrame,
    current_bgr: np.ndarray,
    current_gray: bytes,
    gmm_fg: np.ndarray,
    previous_gray: bytes | None,
    previous_fg: np.ndarray | None,
    foreground_history: list[np.ndarray],
    static_prior: np.ndarray,
    background_bgr: np.ndarray,
    threshold: float,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    foreground_root: Path,
) -> DetectionFeatures:
    height, width = gmm_fg.shape
    current_valid = field_mask_bytes(width, height)
    shift_x = 0
    shift_y = 0

    if previous_gray is None or previous_fg is None:
        valid_mask = np.frombuffer(current_valid, dtype=np.uint8).reshape((height, width)) > 0
        motion_mask = np.zeros((height, width), dtype=bool)
        foreground_memory = np.zeros((height, width), dtype=bool)
        fg_minus_previous = gmm_fg & ~static_prior & valid_mask
        diff_values = np.zeros((height, width), dtype=np.uint8)
    else:
        shift_x, shift_y = estimate_shift(
            current_gray,
            previous_gray,
            width,
            height,
            max_shift_pixels,
            current_valid,
        )
        previous_aligned = shifted_reference(current_gray, previous_gray, width, height, shift_x, shift_y)
        previous_valid = shifted_valid_mask(current_valid, width, height, shift_x, shift_y)
        valid_mask_bytes = combined_valid_mask(current_valid, previous_valid)
        previous_aligned = brightness_matched_reference(current_gray, previous_aligned, valid_mask_bytes)
        valid_mask = np.frombuffer(valid_mask_bytes, dtype=np.uint8).reshape((height, width)) > 0
        previous_aligned_array = np.frombuffer(previous_aligned, dtype=np.uint8).reshape((height, width))
        current_array = np.frombuffer(current_gray, dtype=np.uint8).reshape((height, width))
        valid_mask &= current_array > MASK_BLACK_THRESHOLD
        valid_mask &= previous_aligned_array > MASK_BLACK_THRESHOLD
        diff_values = np.abs(current_array.astype(np.int16) - previous_aligned_array.astype(np.int16)).astype(np.uint8)
        motion_mask = (diff_values > threshold) & valid_mask
        foreground_history[:] = align_foreground_history(foreground_history, shift_x, shift_y)
        foreground_memory = foreground_memory_suppression_mask(
            previous_fg,
            foreground_history,
            shift_x,
            shift_y,
        )
        if foreground_memory is None:
            foreground_memory = np.zeros((height, width), dtype=bool)
            fg_minus_previous = gmm_fg & ~static_prior & valid_mask
        else:
            fg_minus_previous = gmm_fg & ~foreground_memory & ~static_prior & valid_mask

    final_raw = fg_minus_previous & motion_mask & valid_mask
    clean_mask, components = keep_thick_lumps(final_raw, min_blob_area)
    score, changed_ratio, largest_blob_ratio, blob_count = score_motion_lumps(
        clean_mask,
        components,
        diff_values,
        valid_mask,
    )

    foreground_path = foreground_path_for(MODEL_NAME, threshold, window, max_shift_pixels, 0, frame, foreground_root)
    foreground_relative = save_array_image(foreground_path, clean_mask.astype(np.uint8) * 255)
    save_debug_images(
        foreground_path,
        background_bgr,
        gmm_fg,
        foreground_memory,
        static_prior,
        fg_minus_previous,
        motion_mask,
        final_raw,
        clean_mask,
    )
    foreground_history.append(gmm_fg & valid_mask)
    foreground_history[:] = foreground_history[-FOREGROUND_MEMORY_LENGTH:]

    return DetectionFeatures(
        image_id=frame.image_id,
        timestamp=frame.timestamp,
        label=frame.label,
        score=score,
        blur_radius=0.0,
        changed_ratio=changed_ratio,
        largest_blob_ratio=largest_blob_ratio,
        blob_count=blob_count,
        camera_shift_x=shift_x,
        camera_shift_y=shift_y,
        foreground_path=foreground_relative,
    )


def evaluate(
    frames: list[LabeledFrame],
    thresholds: list[int | float],
    cutoffs: list[float],
    windows: list[int],
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path = DEFAULT_FOREGROUND_ROOT,
) -> list[DetectorResult]:
    del blur_radius
    if not frames:
        return []

    images = [read_bgr(frame.masked_path) for frame in frames]
    gray_images = [cv2_module().cvtColor(image, cv2_module().COLOR_BGR2GRAY).tobytes() for image in images]
    background_training_images = training_images_for_background(frames, images)
    results: list[DetectorResult] = []

    for threshold in [float(value) for value in thresholds]:
        for window in windows:
            history = max(2, len(background_training_images), int(window))
            subtractor, background = train_mog2_background(
                background_training_images,
                history=history,
                var_threshold=threshold,
            )
            gmm_foregrounds = [subtractor.apply(image, learningRate=0) > 0 for image in images]
            inactive_foregrounds = [
                foreground
                for frame, foreground in zip(frames, gmm_foregrounds)
                if frame.label == "inactive"
            ]
            static_prior = static_foreground_prior(inactive_foregrounds or gmm_foregrounds)

            previous_gray: bytes | None = None
            previous_fg: np.ndarray | None = None
            foreground_history: list[np.ndarray] = []
            for frame, image, current_gray, gmm_fg in zip(frames, images, gray_images, gmm_foregrounds):
                features = gmm_motion_features(
                    frame,
                    image,
                    current_gray,
                    gmm_fg,
                    previous_gray,
                    previous_fg,
                    foreground_history,
                    static_prior,
                    background,
                    threshold,
                    window,
                    min_blob_area,
                    max_shift_pixels,
                    foreground_root,
                )
                results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, window, features))
                previous_gray = current_gray
                previous_fg = gmm_fg

    return results
