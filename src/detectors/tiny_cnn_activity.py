"""Tiny convolutional activity classifier trained on the current split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DetectorResult,
    LabeledFrame,
    blob_features,
    foreground_path_for,
    same_capture_session,
    save_foreground_mask,
)

MODEL_NAME = "tiny_cnn_activity"
CNN_WIDTH = 96
CNN_HEIGHT = 36
POOL_ROWS = 4
POOL_COLS = 8
PROBABILITY_CUTOFFS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.65]


@dataclass(frozen=True)
class CnnExample:
    frame: LabeledFrame
    features: np.ndarray
    label: int | None
    changed_ratio: float
    largest_blob_ratio: float
    blob_count: int
    foreground_mask: list[bool]


@dataclass(frozen=True)
class CnnModel:
    estimator: Any | None
    constant_probability: float

    def probability(self, features: np.ndarray) -> float:
        if self.estimator is None:
            return self.constant_probability
        return float(self.estimator.predict_proba(features.reshape(1, -1))[0, 1])


def image_array(path: Path, blur_radius: float) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((CNN_WIDTH, CNN_HEIGHT), Image.Resampling.BILINEAR)
        if blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return np.asarray(image, dtype=np.float32) / 255.0


def gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0]) + (0.587 * rgb[:, :, 1]) + (0.114 * rgb[:, :, 2])


def field_like(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)
    spread = max_channel - min_channel
    return (max_channel > 0.14) & (spread > 0.05) & ((blue > red + 0.03) | (green > red + 0.03))


def color_value(rgb: np.ndarray) -> np.ndarray:
    return np.max(rgb, axis=2)


def color_saturation(rgb: np.ndarray) -> np.ndarray:
    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    return (max_channel - min_channel) / np.maximum(max_channel, 1e-4)


def pool_grid(channel: np.ndarray) -> list[float]:
    features: list[float] = []
    row_edges = np.linspace(0, channel.shape[0], POOL_ROWS + 1, dtype=int)
    col_edges = np.linspace(0, channel.shape[1], POOL_COLS + 1, dtype=int)
    for row in range(POOL_ROWS):
        for col in range(POOL_COLS):
            patch = channel[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]]
            if patch.size == 0:
                features.extend([0.0, 0.0])
            else:
                features.extend([float(patch.mean()), float(patch.max())])
    return features


def conv_features(channel: np.ndarray) -> list[np.ndarray]:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
    laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    box = np.ones((5, 5), dtype=np.float32) / 25
    return [
        channel,
        np.maximum(0, ndimage.convolve(channel, sobel_x, mode="nearest")),
        np.maximum(0, -ndimage.convolve(channel, sobel_x, mode="nearest")),
        np.maximum(0, ndimage.convolve(channel, sobel_y, mode="nearest")),
        np.maximum(0, -ndimage.convolve(channel, sobel_y, mode="nearest")),
        np.abs(ndimage.convolve(channel, laplace, mode="nearest")),
        ndimage.convolve(channel, box, mode="nearest"),
    ]


def foreground_from_arrays(current: np.ndarray, reference: np.ndarray, threshold: float) -> tuple[list[bool], float, float, int]:
    current_gray = gray(current)
    reference_gray = gray(reference)
    diff = np.abs(current_gray - reference_gray)
    valid = (current.max(axis=2) > 0.03) & (reference.max(axis=2) > 0.03)
    mask = (diff > (threshold / 255.0)) & valid
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((2, 2), dtype=bool))
    flat = [bool(value) for value in mask.reshape(-1)]
    changed_ratio, largest_blob_ratio, blob_count = blob_features(
        flat,
        CNN_WIDTH,
        CNN_HEIGHT,
        min_blob_area=4,
        total_pixels=max(1, int(np.count_nonzero(valid))),
    )
    return flat, changed_ratio, largest_blob_ratio, blob_count


def extract_features(current: np.ndarray, reference: np.ndarray, threshold: float) -> tuple[np.ndarray, list[bool], float, float, int]:
    current_gray = gray(current)
    reference_gray = gray(reference)
    diff = current_gray - reference_gray
    abs_diff = np.abs(diff)
    rgb_delta = current - reference
    rgb_abs_delta = np.abs(rgb_delta)
    current_value = color_value(current)
    reference_value = color_value(reference)
    current_saturation = color_saturation(current)
    reference_saturation = color_saturation(reference)
    saturation_delta = np.abs(current_saturation - reference_saturation)
    value_delta = current_value - reference_value
    current_field = field_like(current).astype(np.float32)
    reference_field = field_like(reference).astype(np.float32)
    field_to_nonfield = reference_field * (1.0 - current_field)
    field_gain = current_field * (1.0 - reference_field)
    color_distance = np.linalg.norm(current - reference, axis=2) / np.sqrt(3)
    valid = ((current_value > 0.03) & (reference_value > 0.03)).astype(np.float32)
    raw_foreground = ((abs_diff > (threshold / 255.0)) & (valid > 0)).astype(np.float32)
    white_support = ((current.min(axis=2) > 0.42) & (reference_field > 0) & (abs_diff > 0.02)).astype(np.float32)
    black_support = ((current.max(axis=2) < 0.35) & (reference_field > 0) & (abs_diff > 0.02)).astype(np.float32)
    gray_object_support = (
        (current_saturation < 0.22)
        & (current_value > 0.20)
        & (reference_field > 0)
        & (abs_diff > 0.02)
    ).astype(np.float32)
    warm_object_support = (
        (current[:, :, 0] > current[:, :, 2] + 0.05)
        & (current[:, :, 0] > current[:, :, 1] - 0.02)
        & (reference_field > 0)
        & (color_distance > 0.04)
    ).astype(np.float32)
    blue_green_loss = np.maximum(0.0, (reference[:, :, 1] + reference[:, :, 2]) - (current[:, :, 1] + current[:, :, 2]))
    red_gain = np.maximum(0.0, rgb_delta[:, :, 0])
    bright_change = np.maximum(0.0, value_delta)
    dark_change = np.maximum(0.0, -value_delta)

    channels = [
        current_gray,
        reference_gray,
        abs_diff,
        np.maximum(0, diff),
        np.maximum(0, -diff),
        rgb_abs_delta[:, :, 0],
        rgb_abs_delta[:, :, 1],
        rgb_abs_delta[:, :, 2],
        red_gain,
        np.maximum(0.0, rgb_delta[:, :, 1]),
        np.maximum(0.0, rgb_delta[:, :, 2]),
        color_distance,
        current_saturation,
        reference_saturation,
        saturation_delta,
        current_value,
        reference_value,
        bright_change,
        dark_change,
        current_field,
        reference_field,
        field_to_nonfield,
        field_gain,
        raw_foreground,
        white_support,
        black_support,
        gray_object_support,
        warm_object_support,
        blue_green_loss,
    ]
    feature_values: list[float] = []
    for channel in channels:
        for response in conv_features(channel.astype(np.float32)):
            feature_values.extend(pool_grid(response))

    foreground_mask, changed_ratio, largest_blob_ratio, blob_count = foreground_from_arrays(current, reference, threshold)
    feature_values.extend(
        [
            float(abs_diff.mean()),
            float(abs_diff.max()),
            float(color_distance.mean()),
            float(field_to_nonfield.mean()),
            float(field_gain.mean()),
            float(white_support.mean()),
            float(black_support.mean()),
            float(gray_object_support.mean()),
            float(warm_object_support.mean()),
            float(raw_foreground.mean()),
            float(saturation_delta.mean()),
            float(bright_change.mean()),
            float(dark_change.mean()),
            float(blue_green_loss.mean()),
            changed_ratio,
            largest_blob_ratio,
            float(blob_count),
        ]
    )
    return np.asarray(feature_values, dtype=np.float32), foreground_mask, changed_ratio, largest_blob_ratio, blob_count


def build_examples(
    frames: list[LabeledFrame],
    threshold: float,
    blur_radius: float,
    reference_image_ids: set[str] | None,
) -> list[CnnExample]:
    references: list[tuple[LabeledFrame, np.ndarray]] = []
    examples: list[CnnExample] = []

    for frame in frames:
        current = image_array(frame.masked_path, blur_radius)
        candidates = [
            (reference_frame, pixels)
            for reference_frame, pixels in references
            if same_capture_session(reference_frame, frame)
        ]
        if candidates:
            _reference_frame, reference = candidates[-1]
            features, foreground_mask, changed_ratio, largest_blob_ratio, blob_count = extract_features(
                current,
                reference,
                threshold,
            )
            label = None
            if frame.label == "active":
                label = 1
            elif frame.label == "inactive":
                label = 0
            examples.append(
                CnnExample(
                    frame=frame,
                    features=features,
                    label=label,
                    changed_ratio=changed_ratio,
                    largest_blob_ratio=largest_blob_ratio,
                    blob_count=blob_count,
                    foreground_mask=foreground_mask,
                )
            )

        if frame.label == "inactive" and (reference_image_ids is None or frame.image_id in reference_image_ids):
            references.append((frame, current))

    return examples


def fit_model(examples: list[CnnExample], training_image_ids: set[str] | None) -> CnnModel:
    train_ids = training_image_ids or {example.frame.image_id for example in examples}
    train_examples = [
        example
        for example in examples
        if example.frame.image_id in train_ids and example.label is not None
    ]
    if not train_examples:
        return CnnModel(estimator=None, constant_probability=0.0)

    labels = np.asarray([example.label for example in train_examples], dtype=np.int32)
    if len(set(labels.tolist())) < 2:
        return CnnModel(estimator=None, constant_probability=float(labels.mean()))

    features = np.stack([example.features for example in train_examples])
    component_count = min(8, max(1, features.shape[0] - 1), features.shape[1])
    estimator = make_pipeline(
        StandardScaler(),
        PCA(n_components=component_count, random_state=17),
        LogisticRegression(
            C=0.08,
            class_weight="balanced",
            max_iter=1000,
            random_state=17,
            solver="liblinear",
        ),
    )
    estimator.fit(features, labels)
    return CnnModel(estimator=estimator, constant_probability=float(labels.mean()))


def active_cutoffs(cutoffs: list[float]) -> list[float]:
    if not cutoffs or max(cutoffs) <= 0.30:
        return PROBABILITY_CUTOFFS
    return cutoffs


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
    del windows, min_blob_area, max_shift_pixels, detector_options
    results: list[DetectorResult] = []
    cnn_cutoffs = active_cutoffs(cutoffs)

    for threshold in thresholds:
        train_examples = build_examples(frames, float(threshold), blur_radius, training_image_ids)
        model = fit_model(train_examples, training_image_ids)
        eval_examples = (
            train_examples
            if training_image_ids is None
            else build_examples(frames, float(threshold), blur_radius, reference_image_ids=None)
        )
        foreground_paths: dict[str, str] = {}

        for cutoff in cnn_cutoffs:
            for example in eval_examples:
                probability = model.probability(example.features)
                prediction = "active" if probability >= cutoff else "inactive"
                confidence = probability if prediction == "active" else 1.0 - probability
                foreground_path = foreground_paths.get(example.frame.image_id)
                if foreground_path is None:
                    foreground_path = save_foreground_mask(
                        example.foreground_mask,
                        CNN_WIDTH,
                        CNN_HEIGHT,
                        foreground_path_for(
                            MODEL_NAME,
                            float(threshold),
                            1,
                            0,
                            blur_radius,
                            example.frame,
                            foreground_root,
                        ),
                    )
                    foreground_paths[example.frame.image_id] = foreground_path

                results.append(
                    DetectorResult(
                        model=MODEL_NAME,
                        threshold=float(threshold),
                        cutoff=cutoff,
                        window=1,
                        blur_radius=blur_radius,
                        image_id=example.frame.image_id,
                        timestamp=example.frame.timestamp,
                        label=example.frame.label,
                        prediction=prediction,
                        score=probability,
                        changed_ratio=example.changed_ratio,
                        largest_blob_ratio=example.largest_blob_ratio,
                        blob_count=example.blob_count,
                        camera_shift_x=0,
                        camera_shift_y=0,
                        foreground_path=foreground_path,
                        confidence=confidence,
                        config_label="tiny_cnn_96x36_29ch_pca8_logreg",
                    )
                )

    return results
