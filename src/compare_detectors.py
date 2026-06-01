"""Compare subtraction-based activity detectors against labels."""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors import MODEL_EVALUATORS
from src.detectors.common import (
    DEFAULT_FOREGROUND_ROOT,
    DEFAULT_LABELS_PATH,
    DEFAULT_REPORTS_DIR,
    DetectorResult,
    LabeledFrame,
    discard_artifact_mask,
    load_frames,
    foreground_path_for,
    save_foreground_mask,
)


def summarize(results: list[DetectorResult]) -> dict[str, str]:
    numeric = summarize_numeric(results)
    first = results[0]
    return {
        "model": first.model,
        "config": first.config_label,
        "threshold": str(first.threshold),
        "cutoff": f"{first.cutoff:.3f}",
        "window": str(first.window),
        "blur_radius": f"{first.blur_radius:.2f}",
        "samples": str(numeric["samples"]),
        "accuracy": f"{numeric['accuracy']:.3f}",
        "precision_active": f"{numeric['precision_active']:.3f}",
        "recall_active": f"{numeric['recall_active']:.3f}",
        "f1_active": f"{numeric['f1_active']:.3f}",
        "recall_inactive": f"{numeric['recall_inactive']:.3f}",
        "balanced_activity": f"{numeric['balanced_activity']:.3f}",
        "precision_discard": f"{numeric['precision_discard']:.3f}",
        "recall_discard": f"{numeric['recall_discard']:.3f}",
        "f1_discard": f"{numeric['f1_discard']:.3f}",
        "true_active": str(numeric["true_active"]),
        "false_active": str(numeric["false_active"]),
        "false_inactive": str(numeric["false_inactive"]),
        "true_inactive": str(numeric["true_inactive"]),
        "true_discard": str(numeric["true_discard"]),
        "false_discard": str(numeric["false_discard"]),
        "missed_discard": str(numeric["missed_discard"]),
    }


def summarize_numeric(results: list[DetectorResult]) -> dict[str, int | float]:
    total = len(results)
    correct = sum(result.label == result.prediction for result in results)
    true_active = sum(result.label == "active" and result.prediction == "active" for result in results)
    false_active = sum(result.label == "inactive" and result.prediction == "active" for result in results)
    false_inactive = sum(result.label == "active" and result.prediction == "inactive" for result in results)
    true_inactive = sum(result.label == "inactive" and result.prediction == "inactive" for result in results)
    true_discard = sum(result.label == "discard" and result.prediction == "discard" for result in results)
    false_discard = sum(result.label != "discard" and result.prediction == "discard" for result in results)
    missed_discard = sum(result.label == "discard" and result.prediction != "discard" for result in results)

    precision_denominator = true_active + false_active
    recall_denominator = true_active + false_inactive
    precision = true_active / precision_denominator if precision_denominator else 0
    recall = true_active / recall_denominator if recall_denominator else 0
    inactive_recall_denominator = true_inactive + false_active
    inactive_recall = true_inactive / inactive_recall_denominator if inactive_recall_denominator else 0
    balanced_activity = (recall + inactive_recall) / 2 if recall_denominator or inactive_recall_denominator else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0
    discard_precision_denominator = true_discard + false_discard
    discard_recall_denominator = true_discard + missed_discard
    discard_precision = true_discard / discard_precision_denominator if discard_precision_denominator else 0
    discard_recall = true_discard / discard_recall_denominator if discard_recall_denominator else 0
    discard_f1 = (
        2 * discard_precision * discard_recall / (discard_precision + discard_recall)
        if discard_precision + discard_recall
        else 0
    )

    return {
        "samples": total,
        "accuracy": correct / total if total else 0,
        "precision_active": precision,
        "recall_active": recall,
        "f1_active": f1,
        "recall_inactive": inactive_recall,
        "balanced_activity": balanced_activity,
        "precision_discard": discard_precision,
        "recall_discard": discard_recall,
        "f1_discard": discard_f1,
        "true_active": true_active,
        "false_active": false_active,
        "false_inactive": false_inactive,
        "true_inactive": true_inactive,
        "true_discard": true_discard,
        "false_discard": false_discard,
        "missed_discard": missed_discard,
    }


def result_to_dict(result: DetectorResult) -> dict[str, str | int | float | bool]:
    return {
        "model": result.model,
        "threshold": result.threshold,
        "cutoff": result.cutoff,
        "window": result.window,
        "blur_radius": result.blur_radius,
        "config": result.config_label,
        "image_id": result.image_id,
        "timestamp": result.timestamp,
        "label": result.label,
        "prediction": result.prediction,
        "correct": result.label == result.prediction,
        "score": result.score,
        "confidence": result.confidence,
        "changed_ratio": result.changed_ratio,
        "largest_blob_ratio": result.largest_blob_ratio,
        "blob_count": result.blob_count,
        "camera_shift_x": result.camera_shift_x,
        "camera_shift_y": result.camera_shift_y,
        "camera_shift_pixels": abs(result.camera_shift_x) + abs(result.camera_shift_y),
        "foreground_path": result.foreground_path,
    }


def discard_prediction_for(
    model: str,
    frame: LabeledFrame,
    threshold: float,
    cutoff: float,
    window: int,
    blur_radius: float,
    max_shift_pixels: int,
    foreground_root: Path,
    config_label: str = "",
) -> DetectorResult | None:
    mask, width, height, discard_score, is_discard = discard_artifact_mask(frame.masked_path)
    if not is_discard:
        return None

    foreground_path = save_foreground_mask(
        mask,
        width,
        height,
        foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
    )
    return DetectorResult(
        model=model,
        threshold=threshold,
        cutoff=cutoff,
        window=window,
        blur_radius=blur_radius,
        image_id=frame.image_id,
        timestamp=frame.timestamp,
        label=frame.label,
        prediction="discard",
        score=discard_score,
        changed_ratio=discard_score,
        largest_blob_ratio=discard_score,
        blob_count=1,
        camera_shift_x=0,
        camera_shift_y=0,
        foreground_path=foreground_path,
        confidence=1.0,
        config_label=config_label,
    )


def append_discard_predictions(
    model: str,
    model_results: list[DetectorResult],
    frames: list[LabeledFrame],
    max_shift_pixels: int,
    foreground_root: Path,
) -> list[DetectorResult]:
    if not model_results:
        return model_results

    keys = sorted(
        {
            (result.threshold, result.cutoff, result.window, result.blur_radius, result.config_label)
            for result in model_results
        }
    )
    discard_results: list[DetectorResult] = []
    for threshold, cutoff, window, blur_radius, config_label in keys:
        for frame in frames:
            result = discard_prediction_for(
                model,
                frame,
                threshold,
                cutoff,
                window,
                blur_radius,
                max_shift_pixels,
                foreground_root,
                config_label,
            )
            if result is not None:
                discard_results.append(result)

    return model_results + discard_results


def inactive_reference_unavailable_prediction(
    model: str,
    frame: LabeledFrame,
    threshold: float,
    cutoff: float,
    window: int,
    blur_radius: float,
    max_shift_pixels: int,
    foreground_root: Path,
    config_label: str = "",
) -> DetectorResult:
    with Image.open(frame.masked_path) as image:
        width, height = image.size

    foreground_path = save_foreground_mask(
        [False] * (width * height),
        width,
        height,
        foreground_path_for(model, threshold, window, max_shift_pixels, blur_radius, frame, foreground_root),
    )
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


def append_missing_reference_predictions(
    model: str,
    model_results: list[DetectorResult],
    frames: list[LabeledFrame],
    max_shift_pixels: int,
    foreground_root: Path,
) -> list[DetectorResult]:
    keys = sorted(
        {
            (result.threshold, result.cutoff, result.window, result.blur_radius, result.config_label)
            for result in model_results
        }
    )
    if not keys:
        return model_results

    missing_results: list[DetectorResult] = []
    for threshold, cutoff, window, blur_radius, config_label in keys:
        existing_ids = {
            result.image_id
            for result in model_results
            if (
                result.threshold == threshold
                and result.cutoff == cutoff
                and result.window == window
                and result.blur_radius == blur_radius
                and result.config_label == config_label
            )
        }
        for frame in frames:
            if frame.image_id in existing_ids:
                continue
            discard_result = discard_prediction_for(
                model,
                frame,
                threshold,
                cutoff,
                window,
                blur_radius,
                max_shift_pixels,
                foreground_root,
                config_label,
            )
            if discard_result is not None:
                missing_results.append(discard_result)
                continue
            missing_results.append(
                inactive_reference_unavailable_prediction(
                    model,
                    frame,
                    threshold,
                    cutoff,
                    window,
                    blur_radius,
                    max_shift_pixels,
                    foreground_root,
                    config_label,
                )
            )

    return model_results + missing_results


def non_discard_gray_frames(frames: list[LabeledFrame]) -> list[LabeledFrame]:
    return [
        frame
        for frame in frames
        if not discard_artifact_mask(frame.masked_path)[4]
    ]


def evaluate_selected_models(
    frames: list[LabeledFrame],
    models: list[str],
    thresholds: list[int],
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

    for model in models:
        evaluator = MODEL_EVALUATORS.get(model)
        if evaluator is None:
            continue
        evaluator_kwargs: dict[str, Any] = {
            "frames": non_discard_gray_frames(frames),
            "thresholds": thresholds,
            "cutoffs": cutoffs,
            "windows": windows,
            "min_blob_area": min_blob_area,
            "max_shift_pixels": max_shift_pixels,
            "blur_radius": blur_radius,
            "foreground_root": foreground_root,
        }
        evaluator_parameters = inspect.signature(evaluator).parameters
        if "detector_options" in evaluator_parameters:
            evaluator_kwargs["detector_options"] = detector_options
        if "training_image_ids" in evaluator_parameters:
            evaluator_kwargs["training_image_ids"] = training_image_ids

        model_results = evaluator(**evaluator_kwargs)
        model_results = append_discard_predictions(
            model,
            model_results,
            frames,
            max_shift_pixels,
            foreground_root,
        )
        results.extend(
            append_missing_reference_predictions(
                model,
                model_results,
                frames,
                max_shift_pixels,
                foreground_root,
            )
        )

    return results


def grouped_results(results: list[DetectorResult]) -> dict[tuple[str, int, float, int, float, str], list[DetectorResult]]:
    grouped: dict[tuple[str, int, float, int, float, str], list[DetectorResult]] = {}
    for result in results:
        grouped.setdefault(
            (result.model, result.threshold, result.cutoff, result.window, result.blur_radius, result.config_label),
            [],
        ).append(result)
    return grouped


def train_validate_models(
    frames: list[LabeledFrame],
    models: list[str],
    thresholds: list[int],
    cutoffs: list[float],
    windows: list[int],
    validation_percent: int,
    min_blob_area: int,
    max_shift_pixels: int,
    blur_radius: float,
    foreground_root: Path = DEFAULT_FOREGROUND_ROOT,
    detector_options: dict[str, Any] | None = None,
) -> dict[str, object]:
    if len(frames) < 4:
        raise ValueError("need at least 4 labeled active/inactive/discard frames")

    validation_percent = min(100, max(1, validation_percent))
    validation_count = max(1, round(len(frames) * validation_percent / 100))
    validation_count = min(validation_count, len(frames))
    train_ids = {frame.image_id for frame in frames[:-validation_count]}
    validation_ids = {frame.image_id for frame in frames[-validation_count:]}

    results = evaluate_selected_models(
        frames,
        models,
        thresholds,
        cutoffs,
        windows,
        min_blob_area,
        max_shift_pixels,
        blur_radius,
        foreground_root,
        detector_options,
        train_ids,
    )

    model_reports = []
    for key, group in grouped_results(results).items():
        train_results = [result for result in group if result.image_id in train_ids]
        validation_results = [result for result in group if result.image_id in validation_ids]
        if not validation_results:
            continue

        model, threshold, cutoff, window, blur_radius, config_label = key
        model_reports.append(
            {
                "model": model,
                "config": config_label,
                "threshold": threshold,
                "cutoff": cutoff,
                "window": window,
                "blur_radius": blur_radius,
                "train": summarize_numeric(train_results),
                "validation": summarize_numeric(validation_results),
                "predictions": [result_to_dict(result) for result in validation_results],
            }
        )

    model_reports.sort(
        key=lambda report: (
            report["validation"]["balanced_activity"],
            report["train"]["balanced_activity"],
            report["validation"]["accuracy"],
            report["validation"]["f1_active"],
            report["validation"]["precision_active"],
            report["validation"]["recall_active"],
        ),
        reverse=True,
    )

    return {
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "models": model_reports,
    }


def write_detail(path: Path, results: list[DetectorResult]) -> None:
    fields = [
        "model",
        "threshold",
        "cutoff",
        "window",
        "blur_radius",
        "config",
        "image_id",
        "timestamp",
        "label",
        "prediction",
        "score",
        "changed_ratio",
        "largest_blob_ratio",
        "blob_count",
        "camera_shift_x",
        "camera_shift_y",
        "confidence",
        "foreground_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.model,
                    "threshold": result.threshold,
                    "cutoff": f"{result.cutoff:.3f}",
                    "window": result.window,
                    "blur_radius": result.blur_radius,
                    "config": result.config_label,
                    "image_id": result.image_id,
                    "timestamp": result.timestamp,
                    "label": result.label,
                    "prediction": result.prediction,
                    "score": f"{result.score:.6f}",
                    "changed_ratio": f"{result.changed_ratio:.6f}",
                    "largest_blob_ratio": f"{result.largest_blob_ratio:.6f}",
                    "blob_count": result.blob_count,
                    "camera_shift_x": result.camera_shift_x,
                    "camera_shift_y": result.camera_shift_y,
                    "confidence": f"{result.confidence:.3f}",
                    "foreground_path": result.foreground_path,
                }
            )


def write_summary(path: Path, summaries: list[dict[str, str]]) -> None:
    fields = [
        "model",
        "threshold",
        "cutoff",
        "window",
        "blur_radius",
        "config",
        "samples",
        "accuracy",
        "precision_active",
        "recall_active",
        "f1_active",
        "recall_inactive",
        "balanced_activity",
        "precision_discard",
        "recall_discard",
        "f1_discard",
        "true_active",
        "false_active",
        "false_inactive",
        "true_inactive",
        "true_discard",
        "false_discard",
        "missed_discard",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def run_comparison(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[DetectorResult]]:
    frames = load_frames(args.labels)
    detector_options = detector_options_from_args(args)
    results = evaluate_selected_models(
        frames,
        args.models,
        args.thresholds,
        args.cutoffs,
        args.windows,
        args.min_blob_area,
        args.max_shift_pixels,
        args.blur_radius,
        args.foreground_root,
        detector_options,
    )

    summaries = [summarize(results_for_model) for results_for_model in grouped_results(results).values() if results_for_model]
    summaries.sort(
        key=lambda row: (
            float(row["balanced_activity"]),
            float(row.get("train_balanced_activity", row["balanced_activity"])),
            float(row["accuracy"]),
            float(row["f1_active"]),
            float(row["precision_active"]),
        ),
        reverse=True,
    )
    return summaries, results


def detector_options_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "reference_update_mode": args.reference_update_mode,
        "reference_strategy": args.reference_strategy,
        "min_lump_area": args.min_lump_area,
        "max_lump_area": args.max_lump_area,
        "min_density": args.min_lump_density,
        "max_aspect_ratio": args.max_lump_aspect,
        "artifact_penalty": args.artifact_penalty,
        "high_confidence_inactive": args.high_confidence_inactive,
        "hysteresis_margin": args.hysteresis_margin,
        "rgb_color_weight": args.rgb_color_weight,
        "auto_tune_lumps": args.auto_tune_lumps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare subtraction-based activity detectors.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--foreground-root", type=Path, default=DEFAULT_FOREGROUND_ROOT)
    parser.add_argument("--models", nargs="+", default=list(MODEL_EVALUATORS))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[35])
    parser.add_argument("--windows", type=int, nargs="+", default=[5])
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.015, 0.02, 0.025, 0.03])
    parser.add_argument("--min-blob-area", type=int, default=20)
    parser.add_argument("--max-shift-pixels", type=int, default=0)
    parser.add_argument("--blur-radius", type=float, default=0.0)
    parser.add_argument("--reference-update-mode", choices=["predicted_inactive", "manual_inactive"], default="manual_inactive")
    parser.add_argument(
        "--reference-strategy",
        choices=["latest", "nearest_past", "rolling_median", "same_time_of_day"],
        default="latest",
    )
    parser.add_argument("--min-lump-area", type=int, default=7)
    parser.add_argument("--max-lump-area", type=int, default=120)
    parser.add_argument("--min-lump-density", type=float, default=0.20)
    parser.add_argument("--max-lump-aspect", type=float, default=5.0)
    parser.add_argument("--artifact-penalty", type=float, default=1.5)
    parser.add_argument("--high-confidence-inactive", type=float, default=0.75)
    parser.add_argument("--hysteresis-margin", type=float, default=0.10)
    parser.add_argument("--rgb-color-weight", type=float, default=0.0)
    parser.add_argument("--auto-tune-lumps", action="store_true", dest="auto_tune_lumps", default=False)
    parser.add_argument("--no-auto-tune-lumps", action="store_false", dest="auto_tune_lumps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summaries, results = run_comparison(args)

    summary_path = args.reports_dir / "detector_summary.csv"
    detail_path = args.reports_dir / "detector_predictions.csv"
    write_summary(summary_path, summaries)
    write_detail(detail_path, results)

    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote foreground masks under {args.foreground_root}")
    print("Top detector summaries:")
    for summary in summaries[:5]:
        print(
            f"{summary['model']} threshold={summary['threshold']} cutoff={summary['cutoff']} "
            f"window={summary['window']} "
            f"accuracy={summary['accuracy']} precision={summary['precision_active']} "
            f"recall={summary['recall_active']} samples={summary['samples']}"
        )


if __name__ == "__main__":
    main()
