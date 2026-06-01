"""Small local web UI for labeling masked camera images."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compare_detectors import load_frames, train_validate_models  # noqa: E402
from src.detectors import MODEL_NAMES  # noqa: E402
from src.process_dataset import (  # noqa: E402
    DEFAULT_MASKED_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_ROI_ROOT,
    process_dataset,
)
from src.roi_preprocess import DEFAULT_CONFIG_PATH  # noqa: E402
from src.update_data import DEFAULT_CACHE_ROOT, sync_raw_images  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = PACKAGE_ROOT / "templates" / "index.html"
STATIC_ROOT = PACKAGE_ROOT / "static"
LABELS_PATH = REPO_ROOT / "labels" / "labels.csv"
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
VALID_LABELS = {"", "active", "inactive", "discard"}
VALID_MODELS = set(MODEL_NAMES)
MODEL_MAX_SHIFT_PIXELS = 0


def safe_child(root: Path, requested: str) -> Path:
    path = (root / requested).resolve()
    path.relative_to(root.resolve())
    return path


def repo_path(relative_path: str) -> Path:
    return safe_child(REPO_ROOT, relative_path)


def read_rows() -> list[dict[str, str]]:
    if not LABELS_PATH.exists():
        return []

    with LABELS_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = []
        for row in csv.DictReader(file):
            if row.get("image_id"):
                rows.append({field: row.get(field, "") for field in LABEL_FIELDS})
        return sorted(rows, key=lambda row: (row.get("timestamp_utc", ""), row["image_id"]))


def write_rows(rows: list[dict[str, str]]) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.get("timestamp_utc", ""), item["image_id"])):
            writer.writerow({field: row.get(field, "") for field in LABEL_FIELDS})


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {label: 0 for label in ["unlabeled", "active", "inactive", "discard"]}
    for row in rows:
        label = row.get("label", "")
        counts[label if label else "unlabeled"] += 1
    return counts


def render_page() -> bytes:
    rows = read_rows()
    page = TEMPLATE_PATH.read_text(encoding="utf-8")
    page = page.replace("__ROWS_JSON__", json.dumps(rows))
    page = page.replace("__COUNTS_JSON__", json.dumps(label_counts(rows)))
    return page.encode("utf-8")


def value_items(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: object, default: list[int], field_name: str) -> list[int]:
    if value is None or value == "":
        return default

    parsed = []
    for item in value_items(value):
        try:
            parsed.append(int(item))
        except ValueError as error:
            raise ValueError(
                f"{field_name} must be whole numbers like 15,20,25. "
                "Decimals like 0.03 belong in Activity cutoffs."
            ) from error

    return parsed


def parse_float_list(value: object, default: list[float], field_name: str) -> list[float]:
    if value is None or value == "":
        return default

    parsed = []
    for item in value_items(value):
        try:
            parsed.append(float(item))
        except ValueError as error:
            raise ValueError(f"{field_name} must be numbers like 0.005,0.01,0.03.") from error

    return parsed


def parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def parse_detector_options(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("detector_options")
    options = raw if isinstance(raw, dict) else {}
    return {
        "reference_update_mode": str(options.get("reference_update_mode", "manual_inactive")),
        "reference_strategy": str(options.get("reference_strategy", "latest")),
        "min_lump_area": int(options.get("min_lump_area", 7)),
        "max_lump_area": int(options.get("max_lump_area", 120)),
        "min_density": float(options.get("min_density", 0.20)),
        "max_aspect_ratio": float(options.get("max_aspect_ratio", 5.0)),
        "artifact_penalty": float(options.get("artifact_penalty", 1.5)),
        "high_confidence_inactive": float(options.get("high_confidence_inactive", 0.75)),
        "hysteresis_margin": float(options.get("hysteresis_margin", 0.10)),
        "rgb_color_weight": float(options.get("rgb_color_weight", 0.0)),
        "auto_tune_lumps": parse_bool(options.get("auto_tune_lumps"), False),
    }


class LabelHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: int = 200,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: object, status: int = 200) -> None:
        self.send_bytes(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(render_page(), "text/html; charset=utf-8")
            return

        if parsed.path.startswith("/static/"):
            requested = unquote(parsed.path.removeprefix("/static/"))
            try:
                path = safe_child(STATIC_ROOT, requested)
            except ValueError:
                self.send_bytes(b"bad path", "text/plain; charset=utf-8", 400)
                return

            if not path.exists() or not path.is_file():
                self.send_bytes(b"not found", "text/plain; charset=utf-8", 404)
                return

            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_bytes(
                path.read_bytes(),
                content_type,
                extra_headers={
                    "Cache-Control": "no-store, max-age=0",
                    "Pragma": "no-cache",
                },
            )
            return

        if parsed.path == "/image":
            query = parse_qs(parsed.query)
            requested = unquote(query.get("path", [""])[0])
            try:
                path = repo_path(requested)
            except ValueError:
                self.send_bytes(b"bad path", "text/plain; charset=utf-8", 400)
                return

            if not path.exists() or not path.is_file():
                self.send_bytes(b"image not found", "text/plain; charset=utf-8", 404)
                return

            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_bytes(
                path.read_bytes(),
                content_type,
                extra_headers={
                    "Cache-Control": "no-store, max-age=0",
                    "Pragma": "no-cache",
                },
            )
            return

        self.send_bytes(b"not found", "text/plain; charset=utf-8", 404)

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/api/update-data":
            self.update_data()
            return

        if parsed_path == "/api/run-models":
            self.run_models()
            return

        if parsed_path != "/label":
            self.send_bytes(b"not found", "text/plain; charset=utf-8", 404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        image_id = str(payload.get("image_id", ""))
        label = str(payload.get("label", ""))
        notes = str(payload.get("notes", ""))

        if label not in VALID_LABELS:
            self.send_bytes(b"invalid label", "text/plain; charset=utf-8", 400)
            return

        rows = read_rows()
        for row in rows:
            if row["image_id"] == image_id:
                row["label"] = label
                row["notes"] = notes
                write_rows(rows)
                self.send_bytes(b"ok", "text/plain; charset=utf-8")
                return

        self.send_bytes(b"image id not found", "text/plain; charset=utf-8", 404)

    def run_models(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            requested_models = payload.get("models") or [
                "hsv_previous_diff",
                "hsv_pbas",
                "hsv_local_support",
                "hsv_running_gaussian",
                "hsv_balanced_previous_diff_blur",
                "hsv_blue_diff",
                "balanced_previous_diff_blur",
                "balanced_previous_diff_blur_lumps",
                "balanced_previous_diff_blur_lumps_deploy",
                "tiny_cnn_activity",
                "blur_stabilized_balanced_previous_diff",
                "gmm_mog2_foreground_motion",
                "lab_bilateral_previous_diff",
                "lab_bilateral_previous_diff_color",
                "lab_bilateral_previous_diff_color_strong",
            ]
            models = [str(model).strip() for model in requested_models if str(model).strip() in VALID_MODELS]
            if not models:
                invalid_models = ", ".join(str(model) for model in requested_models) or "none"
                valid_models = ", ".join(sorted(VALID_MODELS))
                raise ValueError(
                    "select at least one detector. "
                    f"Received: {invalid_models}. Valid on this server: {valid_models}"
                )

            validation_percent = int(payload.get("validation_percent", 30))
            validation_percent = min(100, max(1, validation_percent))
            thresholds = parse_float_list(payload.get("thresholds"), [35], "Thresholds")
            cutoffs = parse_float_list(payload.get("cutoffs"), [0.015, 0.02, 0.025, 0.03], "Activity cutoffs")
            windows = parse_int_list(payload.get("windows"), [5], "Background windows")
            min_blob_area = int(payload.get("min_blob_area", 20))
            max_shift_pixels = int(payload.get("max_shift_pixels", MODEL_MAX_SHIFT_PIXELS))
            blur_radius = float(payload.get("blur_radius", 0.0))
            detector_options = parse_detector_options(payload)

            frames = load_frames(LABELS_PATH)
            report = train_validate_models(
                frames=frames,
                models=models,
                thresholds=thresholds,
                cutoffs=cutoffs,
                windows=windows,
                validation_percent=validation_percent,
                min_blob_area=min_blob_area,
                max_shift_pixels=max_shift_pixels,
                blur_radius=blur_radius,
                detector_options=detector_options,
            )
            self.send_json(report)
        except Exception as error:  # noqa: BLE001
            self.send_json({"error": str(error)}, status=400)

    def update_data(self) -> None:
        try:
            synced = sync_raw_images("origin", "data", DEFAULT_RAW_ROOT, DEFAULT_CACHE_ROOT)
            processed, added_labels = process_dataset(
                DEFAULT_RAW_ROOT,
                DEFAULT_ROI_ROOT,
                DEFAULT_MASKED_ROOT,
                LABELS_PATH,
                DEFAULT_CONFIG_PATH,
            )
            self.send_json(
                {
                    "synced": synced,
                    "processed": processed,
                    "added_labels": added_labels,
                }
            )
        except Exception as error:  # noqa: BLE001
            self.send_json({"error": str(error)}, status=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local image labeling UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), LabelHandler)
    try:
        print(f"Labeler running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop.")
    except OSError:
        pass
    server.serve_forever()


if __name__ == "__main__":
    main()
