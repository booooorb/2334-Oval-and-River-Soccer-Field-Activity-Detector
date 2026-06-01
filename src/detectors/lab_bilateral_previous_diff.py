"""Lab bilateral previous-frame subtraction detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    DetectionFeatures,
    LabeledFrame,
    MASK_BLACK_THRESHOLD,
    combined_valid_mask,
    estimate_shift,
    expand_cutoffs,
    field_mask_bytes,
    foreground_path_for,
    repo_relative,
    same_capture_session,
    shifted_valid_mask,
)

MODEL_NAME = "lab_bilateral_previous_diff"
MODEL_NAME_COLOR = "lab_bilateral_previous_diff_color"
MODEL_NAME_COLOR_STRONG = "lab_bilateral_previous_diff_color_strong"

BILATERAL_PRESETS = {
    3: {"d": 3, "sigmaColor": 15, "sigmaSpace": 3},
    5: {"d": 5, "sigmaColor": 20, "sigmaSpace": 5},
    7: {"d": 7, "sigmaColor": 35, "sigmaSpace": 7},
}


@dataclass(frozen=True)
class LabDiffResult:
    clean_mask: np.ndarray
    d_final: np.ndarray
    d_l: np.ndarray
    d_color: np.ndarray
    filtered_l_previous: np.ndarray
    filtered_l_current: np.ndarray
    changed_mask: np.ndarray
    closed_mask: np.ndarray
    valid_mask: np.ndarray


def cv2_module() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "lab_bilateral_previous_diff requires OpenCV. Install it with: "
            "& 'D:\\Downloads\\a5_code_data\\python-embed\\python.exe' -m pip install opencv-python"
        ) from error
    return cv2


def read_bgr(path: Path) -> np.ndarray:
    cv2 = cv2_module()
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def robust_balance(channel: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    values = channel[field_mask].astype(np.float32)
    if values.size < 100:
        return channel.astype(np.float32)

    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    epsilon = 1e-6
    return (channel.astype(np.float32) - median) / (iqr + epsilon)


def lab_bilateral_preprocess(img_bgr: np.ndarray, d: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
    cv2 = cv2_module()
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return cv2.bilateralFilter(lab, d, sigmaColor, sigmaSpace)


def shift_bgr_reference(current_bgr: np.ndarray, previous_bgr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return previous_bgr

    height, width = current_bgr.shape[:2]
    aligned = current_bgr.copy()
    current_x_start = max(0, dx)
    current_y_start = max(0, dy)
    current_x_end = min(width, width + dx)
    current_y_end = min(height, height + dy)

    reference_x_start = current_x_start - dx
    reference_y_start = current_y_start - dy
    reference_x_end = current_x_end - dx
    reference_y_end = current_y_end - dy

    aligned[current_y_start:current_y_end, current_x_start:current_x_end] = previous_bgr[
        reference_y_start:reference_y_end,
        reference_x_start:reference_x_end,
    ]
    return aligned


def bilateral_params_for_window(window: int) -> dict[str, int | float]:
    if window in BILATERAL_PRESETS:
        return BILATERAL_PRESETS[window]
    closest = min(BILATERAL_PRESETS, key=lambda preset: abs(preset - window))
    return BILATERAL_PRESETS[closest]


def suppress_black_mask_pixels_bgr(current_bgr: np.ndarray, reference_bgr: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    current_nonblack = np.any(current_bgr > MASK_BLACK_THRESHOLD, axis=2)
    reference_nonblack = np.any(reference_bgr > MASK_BLACK_THRESHOLD, axis=2)
    return field_mask & current_nonblack & reference_nonblack


def lab_bilateral_previous_diff(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    field_mask: np.ndarray,
    params: dict[str, Any],
) -> LabDiffResult:
    d = int(params.get("d", 5))
    sigma_color = float(params.get("sigmaColor", 20))
    sigma_space = float(params.get("sigmaSpace", 5))
    threshold = float(params.get("threshold", 0.4))
    min_area = int(params.get("min_area", 4))
    color_weight = float(params.get("color_weight", 0.0))

    previous_lab = lab_bilateral_preprocess(prev_bgr, d, sigma_color, sigma_space)
    current_lab = lab_bilateral_preprocess(curr_bgr, d, sigma_color, sigma_space)

    prev_l, prev_a, prev_b = previous_lab[:, :, 0], previous_lab[:, :, 1], previous_lab[:, :, 2]
    curr_l, curr_a, curr_b = current_lab[:, :, 0], current_lab[:, :, 1], current_lab[:, :, 2]

    prev_l_bal = robust_balance(prev_l, field_mask)
    curr_l_bal = robust_balance(curr_l, field_mask)
    d_l = np.abs(curr_l_bal - prev_l_bal).astype(np.float32)

    d_color_raw = np.sqrt(
        (curr_a.astype(np.float32) - prev_a.astype(np.float32)) ** 2
        + (curr_b.astype(np.float32) - prev_b.astype(np.float32)) ** 2
    )
    color_median = float(np.median(d_color_raw[field_mask])) if np.any(field_mask) else 0.0
    d_color = np.maximum(d_color_raw - color_median, 0) / 255.0

    d_final = d_l + (color_weight * d_color)
    d_final = np.where(field_mask, d_final, 0)

    changed_mask = d_final > threshold
    clean = clean_mask(changed_mask, min_area)
    closed = close_mask(changed_mask)
    return LabDiffResult(
        clean_mask=clean,
        d_final=d_final,
        d_l=np.where(field_mask, d_l, 0),
        d_color=np.where(field_mask, d_color, 0),
        filtered_l_previous=np.where(field_mask, prev_l, 0).astype(np.uint8),
        filtered_l_current=np.where(field_mask, curr_l, 0).astype(np.uint8),
        changed_mask=np.where(field_mask, changed_mask, False),
        closed_mask=np.where(field_mask, closed, False),
        valid_mask=field_mask,
    )


def close_mask(mask: np.ndarray) -> np.ndarray:
    cv2 = cv2_module()
    kernel = np.ones((3, 3), dtype=np.uint8)
    raw = mask.astype(np.uint8) * 255
    return cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel).astype(bool)


def component_shape_ok(stats: np.ndarray, index: int, min_density: float = 0.20, max_aspect: float = 6.0) -> bool:
    cv2 = cv2_module()
    area = float(stats[index, cv2.CC_STAT_AREA])
    width = float(stats[index, cv2.CC_STAT_WIDTH])
    height = float(stats[index, cv2.CC_STAT_HEIGHT])
    density = area / ((width * height) + 1e-6)
    aspect = max(width, height) / (min(width, height) + 1e-6)
    return density >= min_density and aspect <= max_aspect


def keep_thick_lumps(
    mask: np.ndarray,
    min_area: int,
    min_core_radius: float = 1.5,
    min_core_pixels: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv2 = cv2_module()
    raw = mask.astype(np.uint8) * 255
    closed = close_mask(mask).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
    clean = np.zeros_like(raw)

    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area >= min_area:
            if not component_shape_ok(stats, component):
                continue

            component_mask = (labels == component).astype(np.uint8)
            distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 3)
            max_radius = float(distance.max())
            core_pixels = int(np.sum(distance >= min_core_radius))

            if max_radius < min_core_radius:
                continue
            if core_pixels < min_core_pixels:
                continue

            clean[labels == component] = 255

    return clean.astype(bool), raw.astype(bool), closed.astype(bool)


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    clean, _raw, _closed = keep_thick_lumps(mask, min_area)
    return clean


def compute_activity_features(
    d_final: np.ndarray,
    clean_mask_array: np.ndarray,
    field_mask: np.ndarray,
) -> tuple[float, float, float, int]:
    cv2 = cv2_module()
    valid_pixels = int(np.count_nonzero(field_mask))
    if valid_pixels == 0:
        return 0.0, 0.0, 0.0, 0

    changed_pixels = int(np.count_nonzero(clean_mask_array))
    changed_ratio = changed_pixels / valid_pixels
    diff_strength = float(np.sum(d_final[clean_mask_array]) / valid_pixels)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(clean_mask_array.astype(np.uint8), connectivity=4)
    blob_count = max(0, component_count - 1)
    largest = 0
    if blob_count:
        largest = int(max(stats[1:, cv2.CC_STAT_AREA]))
    largest_blob_ratio = largest / valid_pixels
    score = diff_strength
    return score, changed_ratio, largest_blob_ratio, blob_count


def to_display_image(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    high = float(np.percentile(finite, 99))
    if high <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values / high) * 255, 0, 255).astype(np.uint8)


def save_array_image(path: Path, values: np.ndarray) -> str:
    cv2 = cv2_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = values.astype(np.uint8)
    cv2.imwrite(str(path), image)
    return repo_relative(path)


def save_debug_images(foreground_path: Path, result: LabDiffResult) -> None:
    debug_dir = foreground_path.parent / f"{foreground_path.stem}_debug"
    save_array_image(debug_dir / "filtered_l_previous.png", result.filtered_l_previous)
    save_array_image(debug_dir / "filtered_l_current.png", result.filtered_l_current)
    save_array_image(debug_dir / "d_l.png", to_display_image(result.d_l))
    save_array_image(debug_dir / "d_color.png", to_display_image(result.d_color))
    save_array_image(debug_dir / "d_final.png", to_display_image(result.d_final))
    save_array_image(debug_dir / "thresholded_mask.png", result.changed_mask.astype(np.uint8) * 255)
    save_array_image(debug_dir / "closed_mask.png", result.closed_mask.astype(np.uint8) * 255)
    save_array_image(debug_dir / "cleaned_mask.png", result.clean_mask.astype(np.uint8) * 255)


def features_from_lab_result(
    model: str,
    frame: LabeledFrame,
    result: LabDiffResult,
    width: int,
    height: int,
    threshold: float,
    window: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    shift_x: int,
    shift_y: int,
    save_debug: bool,
) -> DetectionFeatures:
    del min_blob_area
    score, changed_ratio, largest_blob_ratio, blob_count = compute_activity_features(
        result.d_final,
        result.clean_mask,
        result.valid_mask,
    )
    foreground_path = foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root)
    foreground_relative = save_array_image(foreground_path, result.clean_mask.astype(np.uint8) * 255)
    if save_debug:
        save_debug_images(foreground_path, result)
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
        foreground_path=foreground_relative,
    )


def evaluate_with_color_weight(
    model: str,
    color_weight: float,
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
    results: list[DetectorResult] = []
    lab_thresholds = [float(threshold) for threshold in thresholds]
    debug_threshold = lab_thresholds[0] if lab_thresholds else None
    debug_window = windows[0] if windows else None

    for threshold in lab_thresholds:
        inactive_reference_image: np.ndarray | None = None
        inactive_reference_frame: LabeledFrame | None = None

        for frame in frames:
            current_image = read_bgr(frame.masked_path)
            height, width = current_image.shape[:2]

            if inactive_reference_image is not None and inactive_reference_frame is not None and same_capture_session(inactive_reference_frame, frame):
                current_gray = cv2_module().cvtColor(current_image, cv2_module().COLOR_BGR2GRAY).tobytes()
                previous_gray = cv2_module().cvtColor(inactive_reference_image, cv2_module().COLOR_BGR2GRAY).tobytes()
                current_valid = field_mask_bytes(width, height)
                shift_x, shift_y = estimate_shift(
                    current_gray,
                    previous_gray,
                    width,
                    height,
                    max_shift_pixels,
                    current_valid,
                )
                previous_aligned = shift_bgr_reference(current_image, inactive_reference_image, shift_x, shift_y)
                reference_valid = shifted_valid_mask(current_valid, width, height, shift_x, shift_y)
                valid_mask_bytes = combined_valid_mask(current_valid, reference_valid)
                valid_mask = np.frombuffer(valid_mask_bytes, dtype=np.uint8).reshape((height, width)) > 0
                valid_mask = suppress_black_mask_pixels_bgr(current_image, previous_aligned, valid_mask)

                for window in windows:
                    bilateral_params = bilateral_params_for_window(window)
                    params = {
                        **bilateral_params,
                        "threshold": threshold,
                        "min_area": min_blob_area,
                        "color_weight": color_weight,
                    }
                    lab_result = lab_bilateral_previous_diff(previous_aligned, current_image, valid_mask, params)
                    features = features_from_lab_result(
                        model,
                        frame,
                        lab_result,
                        width,
                        height,
                        threshold,
                        window,
                        min_blob_area,
                        max_shift_pixels,
                        0,
                        foreground_root,
                        shift_x,
                        shift_y,
                        model == MODEL_NAME and threshold == debug_threshold and window == debug_window,
                    )
                    results.extend(expand_cutoffs(model, threshold, cutoffs, window, features))

            if frame.label == "inactive":
                inactive_reference_image = current_image
                inactive_reference_frame = frame

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
) -> list[DetectorResult]:
    return evaluate_with_color_weight(
        MODEL_NAME,
        0.0,
        frames,
        thresholds,
        cutoffs,
        windows,
        min_blob_area,
        max_shift_pixels,
        blur_radius,
        foreground_root,
    )


def evaluate_color(
    frames: list[LabeledFrame],
    thresholds: list[int | float],
    cutoffs: list[float],
    windows: list[int],
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path = DEFAULT_FOREGROUND_ROOT,
) -> list[DetectorResult]:
    return evaluate_with_color_weight(
        MODEL_NAME_COLOR,
        0.25,
        frames,
        thresholds,
        cutoffs,
        windows,
        min_blob_area,
        max_shift_pixels,
        blur_radius,
        foreground_root,
    )


def evaluate_color_strong(
    frames: list[LabeledFrame],
    thresholds: list[int | float],
    cutoffs: list[float],
    windows: list[int],
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path = DEFAULT_FOREGROUND_ROOT,
) -> list[DetectorResult]:
    return evaluate_with_color_weight(
        MODEL_NAME_COLOR_STRONG,
        0.5,
        frames,
        thresholds,
        cutoffs,
        windows,
        min_blob_area,
        max_shift_pixels,
        blur_radius,
        foreground_root,
    )
