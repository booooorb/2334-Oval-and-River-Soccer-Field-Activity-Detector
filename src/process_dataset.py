"""Batch-process local camera images and maintain the label manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.roi_preprocess import DEFAULT_CONFIG_PATH, load_roi_config, prepare_detector_image

DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_ROI_ROOT = REPO_ROOT / "data" / "roi"
DEFAULT_MASKED_ROOT = REPO_ROOT / "data" / "masked"
DEFAULT_LABELS_PATH = REPO_ROOT / "labels" / "labels.csv"

LABEL_FIELDS = ["image_id", "raw_path", "masked_path", "label", "notes"]


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def image_id_for(raw_path: Path, raw_root: Path) -> str:
    return raw_path.relative_to(raw_root).with_suffix("").as_posix()


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
        for image_id in sorted(rows):
            writer.writerow({field: rows[image_id].get(field, "") for field in LABEL_FIELDS})


def process_dataset(
    raw_root: Path,
    roi_root: Path,
    masked_root: Path,
    labels_path: Path,
    config_path: Path,
) -> tuple[int, int]:
    config = load_roi_config(config_path)
    labels = read_labels(labels_path)
    processed = 0
    added_labels = 0

    for raw_path in sorted(raw_root.rglob("*.jpg")):
        image_id = image_id_for(raw_path, raw_root)
        relative = raw_path.relative_to(raw_root)
        roi_path = roi_root / relative
        masked_path = masked_root / relative

        roi_path.parent.mkdir(parents=True, exist_ok=True)
        masked_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(raw_path) as image:
            image = image.convert("RGB")
            image.crop(config.crop_box).save(roi_path, quality=95)
            prepare_detector_image(image, config, downsample=False).save(masked_path, quality=95)

        if image_id not in labels:
            labels[image_id] = {
                "image_id": image_id,
                "raw_path": rel(raw_path),
                "masked_path": rel(masked_path),
                "label": "",
                "notes": "",
            }
            added_labels += 1
        else:
            labels[image_id]["raw_path"] = rel(raw_path)
            labels[image_id]["masked_path"] = rel(masked_path)

        processed += 1

    write_labels(labels_path, labels)
    return processed, added_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process local raw camera data and update labels.csv.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--roi-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--masked-root", type=Path, default=DEFAULT_MASKED_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed, added_labels = process_dataset(
        args.raw_root,
        args.roi_root,
        args.masked_root,
        args.labels,
        args.config,
    )
    print(f"Processed {processed} raw image(s)")
    print(f"Added {added_labels} new label row(s)")


if __name__ == "__main__":
    main()
