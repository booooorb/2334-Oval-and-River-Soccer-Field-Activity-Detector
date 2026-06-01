"""Batch-process local camera images and maintain the label manifest."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.roi_preprocess import (
    DEFAULT_CONFIG_PATH,
    apply_stabilization_shift,
    estimate_image_shift,
    fixed_reference_path,
    load_roi_config,
    prepare_detector_image,
)
from src.detectors.common import DISCARD_GRAY_RGB, DISCARD_GRAY_RATIO_THRESHOLD, is_discard_artifact_frame

DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_ROI_ROOT = REPO_ROOT / "data" / "roi"
DEFAULT_MASKED_ROOT = REPO_ROOT / "data" / "masked"
DEFAULT_LABELS_PATH = REPO_ROOT / "labels" / "labels.csv"
LOCAL_TIME_ZONE = ZoneInfo("America/Vancouver")
CORRUPT_GRAY_NOTE = "auto-discard: corrupt gray frame"

LABEL_FIELDS = [
    "image_id",
    "timestamp",
    "timestamp_utc",
    "timestamp_local",
    "previous_image_id",
    "previous_masked_path",
    "raw_path",
    "masked_path",
    "label",
    "notes",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def image_id_for(raw_path: Path, raw_root: Path) -> str:
    return raw_path.relative_to(raw_root).with_suffix("").as_posix()


def format_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_local(timestamp: datetime) -> str:
    return timestamp.astimezone(LOCAL_TIME_ZONE).isoformat(timespec="seconds")


def parsed_timestamp_for(raw_path: Path) -> datetime | None:
    raw_timestamp = raw_path.stem.rsplit("_", 1)[-1]
    timestamp_formats = [
        ("%Y%m%dT%H%M%S%z", None),
        ("%Y%m%dT%H%M%SZ", timezone.utc),
    ]
    for timestamp_format, fallback_zone in timestamp_formats:
        try:
            parsed = datetime.strptime(raw_timestamp, timestamp_format)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=fallback_zone or timezone.utc)
        return parsed
    return None


def timestamp_for(raw_path: Path) -> str:
    parsed = parsed_timestamp_for(raw_path)
    return format_local(parsed) if parsed else ""


def timestamp_utc_for(raw_path: Path) -> str:
    parsed = parsed_timestamp_for(raw_path)
    return format_utc(parsed) if parsed else ""


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["image_id"]: row for row in csv.DictReader(file) if row.get("image_id")}


def write_labels(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for image_id in sorted(rows, key=lambda key: (rows[key].get("timestamp_utc", ""), key)):
            writer.writerow({field: rows[image_id].get(field, "") for field in LABEL_FIELDS})


CORRUPT_GRAY_RGB = DISCARD_GRAY_RGB
CORRUPT_GRAY_DISCARD_RATIO = DISCARD_GRAY_RATIO_THRESHOLD


def apply_auto_discard_label(row: dict[str, str], is_corrupt: bool) -> None:
    if is_corrupt:
        row["label"] = "discard"
        row["notes"] = CORRUPT_GRAY_NOTE
        return

    if row.get("notes") != CORRUPT_GRAY_NOTE:
        return

    if row.get("label") == "discard":
        row["label"] = ""
    row["notes"] = ""


def clamp_shift(shift: tuple[int, int], limit: int) -> tuple[int, int]:
    if limit <= 0:
        return shift
    dx, dy = shift
    return max(-limit, min(limit, dx)), max(-limit, min(limit, dy))


def continuity_adjusted_shift(
    direct_shift: tuple[int, int],
    previous_shift: tuple[int, int] | None,
    relative_previous_shift: tuple[int, int] | None,
    *,
    max_shift_pixels: int,
    max_jump_pixels: int,
) -> tuple[int, int]:
    """Reject direct-reference shifts that jump away from previous-frame motion."""
    if previous_shift is None or relative_previous_shift is None or max_jump_pixels <= 0:
        return direct_shift

    if (
        abs(relative_previous_shift[0]) > max_jump_pixels
        or abs(relative_previous_shift[1]) > max_jump_pixels
    ):
        return direct_shift

    predicted_shift = (
        previous_shift[0] + relative_previous_shift[0],
        previous_shift[1] + relative_previous_shift[1],
    )
    predicted_shift = clamp_shift(predicted_shift, max_shift_pixels)

    if (
        abs(direct_shift[0] - predicted_shift[0]) > max_jump_pixels
        or abs(direct_shift[1] - predicted_shift[1]) > max_jump_pixels
    ):
        return predicted_shift

    return direct_shift


def process_dataset(
    raw_root: Path,
    roi_root: Path,
    masked_root: Path,
    labels_path: Path,
    config_path: Path,
    *,
    update_labels: bool = True,
) -> tuple[int, int]:
    config = load_roi_config(config_path)
    labels = read_labels(labels_path) if update_labels else {}
    processed = 0
    added_labels = 0

    raw_paths = sorted(
        raw_root.rglob("*.jpg"),
        key=lambda path: (timestamp_utc_for(path), image_id_for(path, raw_root)),
    )

    fixed_reference: Image.Image | None = None
    if raw_paths and config.stabilization.reference_strategy == "fixed":
        fixed_reference_image_path = fixed_reference_path(raw_root, config.stabilization.reference_image_id)
        if fixed_reference_image_path is None:
            raise FileNotFoundError(
                f"stabilization reference not found: {config.stabilization.reference_image_id}"
            )
        with Image.open(fixed_reference_image_path) as reference:
            fixed_reference = reference.convert("RGB")

    previous_image_id = ""
    previous_masked_path = ""
    stabilization_reference: Image.Image | None = None
    stabilization_reference_date = ""
    previous_stabilization_image: Image.Image | None = None
    previous_stabilization_shift: tuple[int, int] | None = None
    for raw_path in raw_paths:
        image_id = image_id_for(raw_path, raw_root)
        timestamp = timestamp_for(raw_path)
        timestamp_utc = timestamp_utc_for(raw_path)
        capture_date = raw_path.parent.name
        relative = raw_path.relative_to(raw_root)
        roi_path = roi_root / relative
        masked_path = masked_root / relative

        roi_path.parent.mkdir(parents=True, exist_ok=True)
        masked_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(raw_path) as image:
            image = image.convert("RGB")
            if (
                config.stabilization.reference_strategy == "first_per_day"
                and stabilization_reference_date
                and capture_date != stabilization_reference_date
            ):
                stabilization_reference = None
                stabilization_reference_date = ""

            reference_image = fixed_reference if config.stabilization.reference_strategy == "fixed" else stabilization_reference
            shift = estimate_image_shift(image, reference_image, config)
            if (
                config.stabilization.reference_strategy == "fixed"
                and config.stabilization.continuity_enabled
                and previous_stabilization_image is not None
            ):
                relative_previous_shift = estimate_image_shift(image, previous_stabilization_image, config)
                shift = continuity_adjusted_shift(
                    shift,
                    previous_stabilization_shift,
                    relative_previous_shift,
                    max_shift_pixels=config.stabilization.max_shift_pixels,
                    max_jump_pixels=config.stabilization.continuity_max_jump_pixels,
                )

            stabilized_image = apply_stabilization_shift(image, shift[0], shift[1], config)
            detector_image = prepare_detector_image(stabilized_image, config, downsample=False)
            stabilized_image.crop(config.crop_box).save(roi_path, quality=95)
            detector_image.save(masked_path, quality=95)
            is_corrupt = is_discard_artifact_frame(masked_path)
            if not is_corrupt:
                if config.stabilization.reference_strategy in {"first_per_day", "first_dataset"}:
                    if stabilization_reference is None:
                        stabilization_reference = image.copy()
                        stabilization_reference_date = capture_date
                elif config.stabilization.reference_strategy != "fixed":
                    stabilization_reference = image.copy()
                    stabilization_reference_date = capture_date
                previous_stabilization_image = image.copy()
                previous_stabilization_shift = shift

        if not update_labels:
            processed += 1
            previous_image_id = image_id
            previous_masked_path = rel(masked_path)
            continue

        if image_id not in labels:
            labels[image_id] = {
                "image_id": image_id,
                "timestamp": timestamp,
                "timestamp_utc": timestamp_utc,
                "timestamp_local": timestamp,
                "previous_image_id": previous_image_id,
                "previous_masked_path": previous_masked_path,
                "raw_path": rel(raw_path),
                "masked_path": rel(masked_path),
                "label": "",
                "notes": "",
            }
            apply_auto_discard_label(labels[image_id], is_corrupt)
            added_labels += 1
        else:
            labels[image_id]["timestamp"] = timestamp
            labels[image_id]["timestamp_utc"] = timestamp_utc
            labels[image_id]["timestamp_local"] = timestamp
            labels[image_id]["previous_image_id"] = previous_image_id
            labels[image_id]["previous_masked_path"] = previous_masked_path
            labels[image_id]["raw_path"] = rel(raw_path)
            labels[image_id]["masked_path"] = rel(masked_path)
            apply_auto_discard_label(labels[image_id], is_corrupt)

        processed += 1
        previous_image_id = image_id
        previous_masked_path = rel(masked_path)

    if update_labels:
        write_labels(labels_path, labels)
    return processed, added_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process local raw camera data and update labels.csv.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--roi-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--masked-root", type=Path, default=DEFAULT_MASKED_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Regenerate roi/masked images without reading or writing labels.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed, added_labels = process_dataset(
        args.raw_root,
        args.roi_root,
        args.masked_root,
        args.labels,
        args.config,
        update_labels=not args.images_only,
    )
    print(f"Processed {processed} raw image(s)")
    if args.images_only:
        print("Labels unchanged")
    else:
        print(f"Added {added_labels} new label row(s)")


if __name__ == "__main__":
    main()
