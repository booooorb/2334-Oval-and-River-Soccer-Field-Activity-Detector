"""Blur-stabilized Lab L previous-frame subtraction detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectionFeatures,
    DetectorResult,
    LabeledFrame,
    combined_valid_mask,
    estimate_shift,
    expand_cutoffs,
    field_mask_bytes,
    foreground_path_for,
    same_capture_session,
    shifted_valid_mask,
)
from src.detectors.lab_bilateral_previous_diff import (
    clean_mask,
    cv2_module,
    read_bgr,
    robust_balance,
    save_array_image,
    shift_bgr_reference,
    suppress_black_mask_pixels_bgr,
)

MODEL_NAME = "blur_stabilized_balanced_previous_diff"
BLUR_RATIO_TRIGGER = 1.6
BLUR_PENALTY_WEIGHT = 0.45


def lab_l_channel(image_bgr: np.ndarray) -> np.ndarray:
    cv2 = cv2_module()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def blur_score(channel: np.ndarray, field_mask: np.ndarray) -> float:
    cv2 = cv2_module()
    laplacian = cv2.Laplacian(channel, cv2.CV_32F, ksize=3)
    values = laplacian[field_mask]
    if values.size < 100:
        return 0.0
    return float(np.var(values))


def blur_ratio(previous_score: float, current_score: float) -> float:
    epsilon = 1e-6
    return max(previous_score, current_score, epsilon) / max(min(previous_score, current_score), epsilon)


def adaptive_blur_sigma(ratio: float, base_sigma: float) -> float:
    if ratio < BLUR_RATIO_TRIGGER:
        return base_sigma
    extra_sigma = min(2.5, 0.75 + (0.45 * (ratio - BLUR_RATIO_TRIGGER)))
    return max(base_sigma, extra_sigma)


def gaussian_blur(channel: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return channel
    cv2 = cv2_module()
    return cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)


def blur_penalty(ratio: float) -> float:
    if ratio <= BLUR_RATIO_TRIGGER:
        return 1.0
    return 1.0 + (BLUR_PENALTY_WEIGHT * (ratio - BLUR_RATIO_TRIGGER))


def blur_stabilized_previous_diff(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    field_mask: np.ndarray,
    threshold: float,
    min_area: int,
    base_blur_sigma: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    previous_l = lab_l_channel(previous_bgr)
    current_l = lab_l_channel(current_bgr)

    previous_blur_score = blur_score(previous_l, field_mask)
    current_blur_score = blur_score(current_l, field_mask)
    ratio = blur_ratio(previous_blur_score, current_blur_score)
    sigma = adaptive_blur_sigma(ratio, base_blur_sigma)

    previous_l = gaussian_blur(previous_l, sigma)
    current_l = gaussian_blur(current_l, sigma)

    previous_balanced = robust_balance(previous_l, field_mask)
    current_balanced = robust_balance(current_l, field_mask)
    diff = np.abs(current_balanced - previous_balanced).astype(np.float32)
    diff = np.where(field_mask, diff, 0)

    thresholded = diff > threshold
    cleaned = clean_mask(thresholded, min_area)
    return diff, cleaned, ratio, sigma, blur_penalty(ratio)


def compute_blur_stabilized_features(
    diff: np.ndarray,
    cleaned: np.ndarray,
    field_mask: np.ndarray,
    penalty: float,
) -> tuple[float, float, float, int]:
    cv2 = cv2_module()
    valid_pixels = int(np.count_nonzero(field_mask))
    if valid_pixels == 0:
        return 0.0, 0.0, 0.0, 0

    changed_ratio = int(np.count_nonzero(cleaned)) / valid_pixels
    raw_activity_score = float(np.sum(diff[cleaned]) / valid_pixels)
    final_score = raw_activity_score / penalty

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=4)
    blob_count = max(0, component_count - 1)
    largest = int(max(stats[1:, cv2.CC_STAT_AREA])) if blob_count else 0
    return final_score, changed_ratio, largest / valid_pixels, blob_count


def features_from_blur_stabilized(
    frame: LabeledFrame,
    diff: np.ndarray,
    cleaned: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path,
    shift_x: int,
    shift_y: int,
    penalty: float,
) -> DetectionFeatures:
    del min_blob_area
    height, width = cleaned.shape
    score, changed_ratio, largest_blob_ratio, blob_count = compute_blur_stabilized_features(
        diff,
        cleaned,
        valid_mask,
        penalty,
    )
    foreground_path = foreground_path_for(MODEL_NAME, threshold, 1, max_shift_pixels, blur_radius, frame, foreground_root)
    foreground_relative = save_array_image(foreground_path, cleaned.astype(np.uint8) * 255)
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
    del windows
    results: list[DetectorResult] = []

    for threshold in [float(value) for value in thresholds]:
        inactive_reference_image: np.ndarray | None = None
        inactive_reference_frame: LabeledFrame | None = None

        for frame in frames:
            current_image = read_bgr(frame.masked_path)
            height, width = current_image.shape[:2]

            if inactive_reference_image is not None and inactive_reference_frame is not None and same_capture_session(inactive_reference_frame, frame):
                cv2 = cv2_module()
                current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY).tobytes()
                previous_gray = cv2.cvtColor(inactive_reference_image, cv2.COLOR_BGR2GRAY).tobytes()
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

                diff, cleaned, _ratio, _sigma, penalty = blur_stabilized_previous_diff(
                    previous_aligned,
                    current_image,
                    valid_mask,
                    threshold,
                    min_blob_area,
                    blur_radius,
                )
                features = features_from_blur_stabilized(
                    frame,
                    diff,
                    cleaned,
                    valid_mask,
                    threshold,
                    min_blob_area,
                    max_shift_pixels,
                    blur_radius,
                    foreground_root,
                    shift_x,
                    shift_y,
                    penalty,
                )
                results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, 1, features))

            if frame.label == "inactive":
                inactive_reference_image = current_image
                inactive_reference_frame = frame

    return results
