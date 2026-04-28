"""One-command local data update.

Fetches raw camera images from the GitHub data branch, copies them into the
local ignored data cache, regenerates processed images, and preserves labels.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.process_dataset import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_MASKED_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_ROI_ROOT,
    process_dataset,
)
from src.roi_preprocess import DEFAULT_CONFIG_PATH  # noqa: E402


DEFAULT_CACHE_ROOT = REPO_ROOT / ".cache" / "data-branch"


def run_git(args: list[str]) -> None:
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True)


def safe_extract_tar(archive_file: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_file, "r") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination)):
                raise ValueError(f"refusing to extract unsafe path: {member.name}")
        archive.extractall(destination, filter="data")


def sync_raw_images(remote: str, branch: str, raw_root: Path, cache_root: Path) -> int:
    archive_dir = cache_root / "archive"
    archive_file = cache_root / "data.tar"
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    run_git(["fetch", remote, branch])

    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)

    if archive_file.exists():
        archive_file.unlink()

    run_git(["archive", "--format=tar", f"--output={archive_file}", f"{remote}/{branch}", "data"])
    safe_extract_tar(archive_file, archive_dir)

    source_root = archive_dir / "data"
    copied = 0
    for source in sorted(source_root.rglob("*.jpg")):
        relative = source.relative_to(source_root)
        parts = relative.parts

        if len(parts) >= 3 and parts[1] == "raw":
            date = parts[0]
            file_name = parts[-1]
        elif len(parts) == 2:
            date = parts[0]
            file_name = parts[-1]
        else:
            continue

        target_dir = raw_root / date
        target = target_dir / file_name
        target_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists() or source.stat().st_size != target.stat().st_size:
            shutil.copy2(source, target)
            copied += 1

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync and process local camera data.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="data")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--roi-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--masked-root", type=Path, default=DEFAULT_MASKED_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synced = sync_raw_images(args.remote, args.branch, args.raw_root, args.cache_root)
    processed, added_labels = process_dataset(
        args.raw_root,
        args.roi_root,
        args.masked_root,
        args.labels,
        args.config,
    )
    print(f"Synced {synced} new or changed raw image(s)")
    print(f"Processed {processed} raw image(s)")
    print(f"Added {added_labels} new label row(s)")
    print(f"Labels: {args.labels}")


if __name__ == "__main__":
    main()
