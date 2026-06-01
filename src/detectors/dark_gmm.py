"""Dark adaptive multi-mode background detector inspired by GMM."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    aligned_sample_references,
    expand_cutoffs,
    features_from_dark_mask,
    image_bytes,
    load_gray,
    same_capture_session,
)

MODEL_NAME = "dark_gmm"
MAX_COMPONENTS = 3
SIGMA_MATCH = 2.5


def gaussian_components(values: list[int], cluster_radius: int) -> list[tuple[float, float, int]]:
    clusters: list[dict[str, float]] = []

    for value in sorted(values):
        best = None
        best_distance = float("inf")
        for cluster in clusters:
            mean = cluster["sum"] / cluster["count"]
            distance = abs(value - mean)
            if distance < best_distance:
                best = cluster
                best_distance = distance

        if best is not None and best_distance <= cluster_radius:
            best["sum"] += value
            best["sum_sq"] += value * value
            best["count"] += 1
        else:
            clusters.append({"sum": value, "sum_sq": value * value, "count": 1})

    clusters = sorted(clusters, key=lambda cluster: cluster["count"], reverse=True)[:MAX_COMPONENTS]
    components = []
    for cluster in clusters:
        count = int(cluster["count"])
        mean = cluster["sum"] / count
        variance = max(1.0, (cluster["sum_sq"] / count) - (mean * mean))
        components.append((mean, variance**0.5, count))
    return components


def gmm_background_mask(
    current: bytes,
    samples: list[bytes],
    valid_mask: bytes,
    threshold: int,
) -> list[bool]:
    cluster_radius = max(4, round(threshold * 0.5))
    dark_floor = max(6, round(threshold * 0.5))
    mask = []

    for index, pixel in enumerate(current):
        if not valid_mask[index]:
            mask.append(False)
            continue

        values = [sample[index] for sample in samples]
        components = gaussian_components(values, cluster_radius)
        matched = False
        darker_than_all_modes = True

        for mean, sigma, _count in components:
            adaptive_radius = max(dark_floor, SIGMA_MATCH * sigma)
            if abs(pixel - mean) <= adaptive_radius:
                matched = True
            if pixel >= mean - adaptive_radius:
                darker_than_all_modes = False

        mask.append((not matched) and darker_than_all_modes)

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
    del blur_radius
    results: list[DetectorResult] = []

    for threshold in thresholds:
        for window in windows:
            history: deque[bytes] = deque(maxlen=window)
            previous_frame: LabeledFrame | None = None

            for frame in frames:
                if previous_frame is not None and not same_capture_session(previous_frame, frame):
                    history.clear()

                image = load_gray(frame.masked_path)
                current = image_bytes(image)
                width, height = image.size

                if len(history) == window:
                    samples, valid_mask, shift_x, shift_y = aligned_sample_references(
                        current,
                        list(history),
                        width,
                        height,
                        max_shift_pixels,
                    )
                    mask = gmm_background_mask(current, samples, valid_mask, threshold)
                    features = features_from_dark_mask(
                        MODEL_NAME,
                        frame,
                        mask,
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
                    )
                    results.extend(expand_cutoffs(MODEL_NAME, threshold, cutoffs, window, features))

                history.append(current)
                previous_frame = frame

    return results
