"""Prepare traffic camera frames for the soccer field detector."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageDraw


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "field_roi.json"


@dataclass(frozen=True)
class RoiConfig:
    source_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    exclude_polygons: Mapping[str, tuple[tuple[int, int], ...]]
    downsample_size: tuple[int, int] | None
    mask_fill_rgb: tuple[int, int, int]


def _pair(value: Iterable[int], name: str) -> tuple[int, int]:
    items = tuple(int(item) for item in value)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly 2 integers")
    return items


def _box(value: Iterable[int], name: str) -> tuple[int, int, int, int]:
    items = tuple(int(item) for item in value)
    if len(items) != 4:
        raise ValueError(f"{name} must contain exactly 4 integers")
    left, top, right, bottom = items
    if right <= left or bottom <= top:
        raise ValueError(f"{name} must be [left, top, right, bottom]")
    return items


def load_roi_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RoiConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)

    downsample = raw.get("downsample_size")
    return RoiConfig(
        source_size=_pair(raw["source_size"], "source_size"),
        crop_box=_box(raw["crop_box"], "crop_box"),
        exclude_polygons={
            name: tuple(_pair(point, f"exclude_polygons.{name} point") for point in polygon)
            for name, polygon in raw["exclude_polygons"].items()
        },
        downsample_size=None if downsample is None else _pair(downsample, "downsample_size"),
        mask_fill_rgb=tuple(int(channel) for channel in raw.get("mask_fill_rgb", [0, 0, 0])),
    )


def build_field_mask(config: RoiConfig) -> Image.Image:
    crop_width = config.crop_box[2] - config.crop_box[0]
    crop_height = config.crop_box[3] - config.crop_box[1]
    mask = Image.new("L", (crop_width, crop_height), 255)
    draw = ImageDraw.Draw(mask)
    for polygon in config.exclude_polygons.values():
        draw.polygon(polygon, fill=0)
    return mask


def prepare_detector_image(
    image: Image.Image,
    config: RoiConfig,
    *,
    downsample: bool = True,
) -> Image.Image:
    """Crop, mask, and optionally downsample one full camera frame."""
    if image.size != config.source_size:
        raise ValueError(f"expected image size {config.source_size}, got {image.size}")

    cropped = image.convert("RGB").crop(config.crop_box)
    mask = build_field_mask(config)
    background = Image.new("RGB", cropped.size, config.mask_fill_rgb)
    masked = Image.composite(cropped, background, mask)

    if downsample and config.downsample_size is not None:
        masked = masked.resize(config.downsample_size, Image.Resampling.LANCZOS)

    return masked


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    downsample: bool = True,
) -> Path:
    config = load_roi_config(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        prepared = prepare_detector_image(image, config, downsample=downsample)
        prepared.save(output)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the masked detector input from a raw camera image.")
    parser.add_argument("input", type=Path, help="Path to the raw 1280x720 camera image.")
    parser.add_argument("output", type=Path, help="Path where the processed detector image should be written.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ROI configuration JSON.",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Keep the masked 540x200 crop instead of resizing to the configured detector size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = preprocess_file(
        args.input,
        args.output,
        args.config,
        downsample=not args.no_downsample,
    )
    print(output)


if __name__ == "__main__":
    main()
