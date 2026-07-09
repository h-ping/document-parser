from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

from .config import ConfigError, RuntimeConfig
from .image_compare import (
    ImageCompareError,
    SUPPORTED_IMAGE_SUFFIXES,
    build_ocr_quality_report,
    compare_standard_to_ocr,
    load_standard_artifacts,
    normalize_ppocr_fixture_page,
)
from .image_compare_html import write_image_compare_html
from .models import PageInfo, to_jsonable
from .ocr import PPOCRV6OcrClient, RecordedOcrClient
from .utils import sha256_file, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare package image printed text against structured standard artifacts.")
    parser.add_argument("--standard-dir", required=True, type=Path, help="Directory containing standard_items.json and related artifacts.")
    parser.add_argument("--image", required=True, type=Path, help="Input PNG/JPG package image.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output artifact directory.")
    parser.add_argument("--ocr-fixture", type=Path, help="Recorded PP-OCR response fixture for offline tests.")
    args = parser.parse_args(argv)

    try:
        run_compare_package_image(
            standard_dir=args.standard_dir,
            image_path=args.image,
            output_dir=args.output_dir,
            ocr_fixture_path=args.ocr_fixture,
        )
    except (ConfigError, ImageCompareError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "failure_result.json",
            {
                "status": "failed",
                "stage": "package_image_comparison",
                "error_type": exc.__class__.__name__,
                "reason": str(exc),
            },
        )
        _write_artifact_index(args.output_dir)
        print(f"compare-package-image failed: {exc}", file=sys.stderr)
        return 1
    return 0


def run_compare_package_image(
    *,
    standard_dir: Path,
    image_path: Path,
    output_dir: Path,
    ocr_fixture_path: Path | None = None,
) -> dict[str, Any]:
    standard_dir = standard_dir.resolve()
    image_path = image_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ImageCompareError(f"Unsupported package image type: {image_path.suffix or '<none>'}")
    if not image_path.exists():
        raise ImageCompareError(f"Package image does not exist: {image_path}")
    artifacts = load_standard_artifacts(standard_dir)
    input_image_width, input_image_height = image_size(image_path)
    if ocr_fixture_path:
        ocr_page_width, ocr_page_height = normalize_ppocr_fixture_page(ocr_fixture_path, input_image_width, input_image_height)
        ocr_client = RecordedOcrClient(ocr_fixture_path)
        ocr_mode = "recorded_fixture"
    else:
        ocr_page_width, ocr_page_height = input_image_width, input_image_height
        config = RuntimeConfig.from_env(require_secrets=True, required_env_vars=["PPOCRV6_API_KEY"])
        ocr_client = PPOCRV6OcrClient(config)
        ocr_mode = "ppocrv6"
    page = PageInfo(page=1, width=ocr_page_width, height=ocr_page_height)
    ocr_lines = ocr_client.recognize_image(image_path, page)
    ocr_quality_report = build_ocr_quality_report(
        fixture_path=ocr_fixture_path,
        input_image_width=input_image_width,
        input_image_height=input_image_height,
        ocr_page_width=ocr_page_width,
        ocr_page_height=ocr_page_height,
        ocr_lines=ocr_lines,
    )
    comparison = compare_standard_to_ocr(artifacts, ocr_lines, image_path)
    package_ocr_lines = [to_jsonable(line) for line in ocr_lines]

    copied_image = output_dir / f"package_image{image_path.suffix.lower()}"
    if copied_image.resolve() != image_path:
        shutil.copy2(image_path, copied_image)

    runtime_policy = {
        "artifact_version": "package_image_compare_runtime_policy_v0.1",
        "source": "compare-package-image",
        "input_format": image_path.suffix.lower().lstrip("."),
        "ocr": {
            "mode": ocr_mode,
            "fixture_path": str(ocr_fixture_path.resolve()) if ocr_fixture_path else None,
            "line_count": len(ocr_lines),
            "bbox_available_count": sum(1 for line in ocr_lines if line.bbox_normalized is not None),
            "ocr_page_size": {"width": ocr_page_width, "height": ocr_page_height},
            "bbox_overlay_status": ocr_quality_report.get("bbox_overlay_status"),
        },
        "llm_agent": {"enabled": False, "mode": "not_applicable"},
    }

    write_json(output_dir / "runtime_policy.json", runtime_policy)
    write_json(output_dir / "package_ocr_quality_report.json", ocr_quality_report)
    write_json(output_dir / "standard_targets.json", comparison["standard_targets"])
    write_json(output_dir / "package_ocr_lines.json", package_ocr_lines)
    write_json(output_dir / "package_layout.json", comparison["package_layout"])
    write_json(output_dir / "package_candidates.json", comparison["package_candidates"])
    write_json(output_dir / "package_extracted_items.json", comparison["package_extracted_items"])
    write_json(output_dir / "unmatched_print_text.json", comparison["unmatched_print_text"])
    write_json(output_dir / "comparison_result.json", comparison["comparison_result"])
    write_json(
        output_dir / "00_inputs" / "file_inventory.json",
        {
            "files": [
                _file_inventory_item("standard_dir", standard_dir),
                _file_inventory_item("package_image", image_path),
                *([_file_inventory_item("ocr_fixture", ocr_fixture_path.resolve())] if ocr_fixture_path else []),
            ]
        },
    )
    write_image_compare_html(
        output_dir / "result_preview.html",
        image_path=copied_image,
        image_width=input_image_width,
        image_height=input_image_height,
        package_ocr_lines=package_ocr_lines,
        comparison_result=comparison["comparison_result"],
        standard_targets=comparison["standard_targets"],
        unmatched_print_text=comparison["unmatched_print_text"],
        ocr_quality_report=ocr_quality_report,
    )
    _write_artifact_index(output_dir)
    return comparison["comparison_result"]


def image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
        if signature.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", signature[16:24])
        if signature[:2] == b"\xff\xd8":
            return _jpeg_size(handle)
    raise ImageCompareError(f"Unsupported or unreadable image header: {path}")


def _jpeg_size(handle: Any) -> tuple[int, int]:
    handle.seek(2)
    while True:
        marker_start = handle.read(1)
        if not marker_start:
            break
        if marker_start != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if marker in {b"\xd8", b"\xd9"}:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            break
        length = struct.unpack(">H", length_bytes)[0]
        if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
            data = handle.read(5)
            if len(data) != 5:
                break
            height, width = struct.unpack(">HH", data[1:5])
            return width, height
        handle.seek(length - 2, 1)
    raise ImageCompareError("Could not read JPEG dimensions.")


def _file_inventory_item(role: str, path: Path) -> dict[str, Any]:
    is_file = path.exists() and path.is_file()
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if is_file else None,
        "size_bytes": path.stat().st_size if is_file else None,
    }


def _write_artifact_index(output_dir: Path) -> None:
    index_path = output_dir / "artifacts" / "index.json"
    suffixes = {".json", ".html", ".png", ".jpg", ".jpeg", ".md"}
    artifacts = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.suffix.lower() in suffixes):
        if path == index_path:
            continue
        artifacts.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(index_path, {"artifact_count": len(artifacts), "artifacts": artifacts})


if __name__ == "__main__":
    raise SystemExit(main())
