"""Prepare traffic camera frames for the soccer field detector."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "field_roi.json"


@dataclass(frozen=True)
class StabilizationConfig:
    enabled: bool
    max_shift_pixels: int
    reference_strategy: str
    reference_image_id: str | None
    continuity_enabled: bool
    continuity_max_jump_pixels: int
    fill_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class RoiConfig:
    source_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    exclude_polygons: Mapping[str, tuple[tuple[int, int], ...]]
    downsample_size: tuple[int, int] | None
    mask_fill_rgb: tuple[int, int, int]
    stabilization: StabilizationConfig


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
    stabilization = raw.get("stabilization", {})
    continuity = stabilization.get("continuity", {})
    fill_rgb = tuple(int(channel) for channel in stabilization.get("fill_rgb", raw.get("mask_fill_rgb", [0, 0, 0])))
    reference_image_id = stabilization.get("reference_image_id")
    return RoiConfig(
        source_size=_pair(raw["source_size"], "source_size"),
        crop_box=_box(raw["crop_box"], "crop_box"),
        exclude_polygons={
            name: tuple(_pair(point, f"exclude_polygons.{name} point") for point in polygon)
            for name, polygon in raw["exclude_polygons"].items()
        },
        downsample_size=None if downsample is None else _pair(downsample, "downsample_size"),
        mask_fill_rgb=tuple(int(channel) for channel in raw.get("mask_fill_rgb", [0, 0, 0])),
        stabilization=StabilizationConfig(
            enabled=bool(stabilization.get("enabled", False)),
            max_shift_pixels=int(stabilization.get("max_shift_pixels", 0)),
            reference_strategy=str(stabilization.get("reference_strategy", "previous")),
            reference_image_id=None if reference_image_id is None else str(reference_image_id),
            continuity_enabled=bool(continuity.get("enabled", False)),
            continuity_max_jump_pixels=int(continuity.get("max_jump_pixels", 6)),
            fill_rgb=fill_rgb,
        ),
    )


def build_field_mask(config: RoiConfig) -> Image.Image:
    crop_width = config.crop_box[2] - config.crop_box[0]
    crop_height = config.crop_box[3] - config.crop_box[1]
    mask = Image.new("L", (crop_width, crop_height), 255)
    draw = ImageDraw.Draw(mask)
    for polygon in config.exclude_polygons.values():
        draw.polygon(polygon, fill=0)
    return mask


def _sampled_alignment_error(
    current: bytes,
    reference: bytes,
    width: int,
    height: int,
    dx: int,
    dy: int,
    valid_mask: bytes,
    sample_step: int = 4,
) -> float:
    current_x_start = max(0, dx)
    current_y_start = max(0, dy)
    current_x_end = min(width, width + dx)
    current_y_end = min(height, height + dy)

    total = 0
    count = 0
    for y in range(current_y_start, current_y_end, sample_step):
        reference_y = y - dy
        current_row = y * width
        reference_row = reference_y * width
        for x in range(current_x_start, current_x_end, sample_step):
            reference_x = x - dx
            current_index = current_row + x
            reference_index = reference_row + reference_x
            if not valid_mask[current_index] or not valid_mask[reference_index]:
                continue
            total += abs(current[current_index] - reference[reference_index])
            count += 1

    if count < 100:
        return float("inf")
    return total / count


def _array_alignment_error(
    current: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    dx: int,
    dy: int,
    sample_step: int,
) -> float:
    height, width = current.shape
    current_x_start = max(0, dx)
    current_y_start = max(0, dy)
    current_x_end = min(width, width + dx)
    current_y_end = min(height, height + dy)

    if current_x_end <= current_x_start or current_y_end <= current_y_start:
        return float("inf")

    current_view = current[current_y_start:current_y_end:sample_step, current_x_start:current_x_end:sample_step]
    reference_view = reference[
        current_y_start - dy:current_y_end - dy:sample_step,
        current_x_start - dx:current_x_end - dx:sample_step,
    ]
    current_valid = valid_mask[current_y_start:current_y_end:sample_step, current_x_start:current_x_end:sample_step]
    reference_valid = valid_mask[
        current_y_start - dy:current_y_end - dy:sample_step,
        current_x_start - dx:current_x_end - dx:sample_step,
    ]
    valid = current_valid & reference_valid
    valid_count = int(valid.sum())
    if valid_count < 100:
        return float("inf")

    diff = np.abs(current_view.astype(np.int16) - reference_view.astype(np.int16))
    return float(diff[valid].mean())


def _best_shift_from_candidates(
    current: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    candidates: set[tuple[int, int]],
    sample_step: int,
) -> tuple[int, int, float]:
    best_dx = 0
    best_dy = 0
    best_error = float("inf")
    for dx, dy in sorted(candidates):
        error = _array_alignment_error(current, reference, valid_mask, dx, dy, sample_step)
        if error < best_error:
            best_dx = dx
            best_dy = dy
            best_error = error
    return best_dx, best_dy, best_error


def estimate_crop_shift(current_crop: Image.Image, reference_crop: Image.Image, config: RoiConfig) -> tuple[int, int]:
    """Return the reference-to-current translation for two cropped frames."""
    max_shift_pixels = config.stabilization.max_shift_pixels
    if max_shift_pixels <= 0:
        return 0, 0

    current = np.asarray(current_crop.convert("L"))
    reference = np.asarray(reference_crop.convert("L"))
    valid_mask = np.asarray(build_field_mask(config)) > 0

    coarse_errors: list[tuple[float, int, int]] = []
    for dy in range(-max_shift_pixels, max_shift_pixels + 1, 2):
        for dx in range(-max_shift_pixels, max_shift_pixels + 1, 2):
            error = _array_alignment_error(current, reference, valid_mask, dx, dy, sample_step=4)
            coarse_errors.append((error, dx, dy))

    fine_candidates: set[tuple[int, int]] = {(0, 0)}
    for _error, coarse_dx, coarse_dy in sorted(coarse_errors)[:8]:
        for dy in range(max(-max_shift_pixels, coarse_dy - 2), min(max_shift_pixels, coarse_dy + 2) + 1):
            for dx in range(max(-max_shift_pixels, coarse_dx - 2), min(max_shift_pixels, coarse_dx + 2) + 1):
                fine_candidates.add((dx, dy))

    best_dx, best_dy, _best_error = _best_shift_from_candidates(
        current,
        reference,
        valid_mask,
        fine_candidates,
        sample_step=1,
    )
    return best_dx, best_dy


def translate_image(image: Image.Image, dx: int, dy: int, fill_rgb: tuple[int, int, int]) -> Image.Image:
    """Translate an image and fill newly exposed pixels with black/configured fill."""
    if dx == 0 and dy == 0:
        return image.copy()

    width, height = image.size
    translated = Image.new("RGB", image.size, fill_rgb)

    source_left = max(0, -dx)
    source_top = max(0, -dy)
    source_right = min(width, width - dx)
    source_bottom = min(height, height - dy)

    if source_right <= source_left or source_bottom <= source_top:
        return translated

    patch = image.crop((source_left, source_top, source_right, source_bottom))
    translated.paste(patch, (max(0, dx), max(0, dy)))
    return translated


def estimate_image_shift(
    image: Image.Image,
    reference_image: Image.Image | None,
    config: RoiConfig,
) -> tuple[int, int]:
    """Estimate the reference-to-current shift for a full raw frame."""
    if not config.stabilization.enabled or reference_image is None:
        return 0, 0
    if image.size != config.source_size:
        raise ValueError(f"expected image size {config.source_size}, got {image.size}")
    if reference_image.size != config.source_size:
        raise ValueError(f"expected reference image size {config.source_size}, got {reference_image.size}")

    current_crop = image.convert("RGB").crop(config.crop_box)
    reference_crop = reference_image.convert("RGB").crop(config.crop_box)
    return estimate_crop_shift(current_crop, reference_crop, config)


def apply_stabilization_shift(image: Image.Image, dx: int, dy: int, config: RoiConfig) -> Image.Image:
    """Move a frame into the reference coordinate system."""
    if not config.stabilization.enabled:
        return image.copy()
    return translate_image(image.convert("RGB"), -dx, -dy, config.stabilization.fill_rgb)


def stabilize_image(
    image: Image.Image,
    reference_image: Image.Image | None,
    config: RoiConfig,
) -> tuple[Image.Image, int, int]:
    """Align a full raw frame to a reference before masking.

    The estimated shift is the reference-to-current camera shift. The returned
    image applies the opposite translation so the current frame lands in the
    reference coordinate system.
    """
    dx, dy = estimate_image_shift(image, reference_image, config)
    stabilized = apply_stabilization_shift(image, dx, dy, config)
    return stabilized, dx, dy


def previous_reference_path(input_path: Path, reference_root: Path) -> Path | None:
    """Find the closest earlier jpg in the same raw-data tree."""
    input_path = input_path.resolve()
    reference_root = reference_root.resolve()
    try:
        input_key = input_path.relative_to(reference_root).as_posix()
    except ValueError:
        input_key = input_path.as_posix()
    candidates = sorted(
        (path.resolve() for path in reference_root.rglob("*.jpg")),
        key=lambda path: path.relative_to(reference_root).as_posix(),
    )
    previous: Path | None = None
    for candidate in candidates:
        candidate_key = candidate.relative_to(reference_root).as_posix()
        if candidate == input_path:
            return previous
        if candidate_key < input_key:
            previous = candidate
    return previous


def daily_reference_path(input_path: Path, reference_root: Path) -> Path | None:
    """Find the first earlier jpg from the same date folder."""
    input_path = input_path.resolve()
    reference_root = reference_root.resolve()
    try:
        input_key = input_path.relative_to(reference_root).as_posix()
    except ValueError:
        input_key = input_path.as_posix()

    candidates = sorted(
        (path.resolve() for path in input_path.parent.glob("*.jpg")),
        key=lambda path: path.name,
    )
    for candidate in candidates:
        try:
            candidate_key = candidate.relative_to(reference_root).as_posix()
        except ValueError:
            candidate_key = candidate.as_posix()
        if candidate == input_path:
            return None
        if candidate_key < input_key:
            return candidate
    return None


def dataset_reference_path(input_path: Path, reference_root: Path) -> Path | None:
    """Find the first earlier jpg from the whole raw-data tree."""
    input_path = input_path.resolve()
    reference_root = reference_root.resolve()
    try:
        input_key = input_path.relative_to(reference_root).as_posix()
    except ValueError:
        input_key = input_path.as_posix()

    for candidate in sorted(
        (path.resolve() for path in reference_root.rglob("*.jpg")),
        key=lambda path: path.relative_to(reference_root).as_posix(),
    ):
        candidate_key = candidate.relative_to(reference_root).as_posix()
        if candidate == input_path:
            return None
        if candidate_key < input_key:
            return candidate
    return None


def fixed_reference_path(reference_root: Path, reference_image_id: str | None) -> Path | None:
    """Find a configured reference image inside a raw-data tree."""
    if not reference_image_id:
        return None

    reference_root = reference_root.resolve()
    normalized_id = reference_image_id.replace("\\", "/")
    direct = reference_root / normalized_id
    candidates = [direct]
    if direct.suffix.lower() != ".jpg":
        candidates.append(direct.with_suffix(".jpg"))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    wanted = normalized_id.removesuffix(".jpg")
    for candidate in sorted(
        (path.resolve() for path in reference_root.rglob("*.jpg")),
        key=lambda path: path.relative_to(reference_root).as_posix(),
    ):
        key = candidate.relative_to(reference_root).with_suffix("").as_posix()
        if key == wanted or candidate.stem == wanted:
            return candidate
    return None


def reference_path_for(input_path: Path, reference_root: Path, config: RoiConfig) -> Path | None:
    strategy = config.stabilization.reference_strategy
    if strategy == "fixed":
        return fixed_reference_path(reference_root, config.stabilization.reference_image_id)
    if strategy == "first_dataset":
        return dataset_reference_path(input_path, reference_root)
    if strategy == "first_per_day":
        return daily_reference_path(input_path, reference_root)
    return previous_reference_path(input_path, reference_root)


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
    reference_path: str | Path | None = None,
    reference_root: str | Path | None = None,
    stabilize: bool = True,
) -> Path:
    config = load_roi_config(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    resolved_reference_path = Path(reference_path) if reference_path is not None else None
    if resolved_reference_path is None and reference_root is not None:
        resolved_reference_path = reference_path_for(
            Path(input_path),
            Path(reference_root),
            config,
        )

    with Image.open(input_path) as image:
        image = image.convert("RGB")
        reference_image = None
        if stabilize and resolved_reference_path is not None and resolved_reference_path.exists():
            with Image.open(resolved_reference_path) as reference:
                reference_image = reference.convert("RGB")
        if stabilize:
            image, _shift_x, _shift_y = stabilize_image(image, reference_image, config)
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
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional raw reference image used to stabilize the frame before masking.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Optional raw image tree used to find a stabilization reference.",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable camera-shift stabilization even if the ROI config enables it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = preprocess_file(
        args.input,
        args.output,
        args.config,
        downsample=not args.no_downsample,
        reference_path=args.reference,
        reference_root=args.reference_root,
        stabilize=not args.no_stabilize,
    )
    print(output)


if __name__ == "__main__":
    main()
