from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Literal

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
from .llm import LlmClient, OpenAICompatibleLlmClient
from .models import OcrLine, PageInfo, to_jsonable
from .ocr import GLMOcrClient, PPOCRV6Client, RecordedOcrClient
from .package_structure import LlmMode, run_package_structure_stage, write_package_structure_artifacts
from .utils import sha256_file, write_json


OcrMode = Literal["hybrid", "ppocr", "glm"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare package image printed text against structured standard artifacts.")
    parser.add_argument("--standard-dir", required=True, type=Path, help="Directory containing standard_items.json and related artifacts.")
    parser.add_argument("--image", required=True, type=Path, help="Input PNG/JPG package image.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output artifact directory.")
    parser.add_argument("--ocr-fixture", type=Path, help="Recorded PP-OCR response fixture for offline tests.")
    parser.add_argument("--ppocr-fixture", type=Path, help="Recorded PP-OCR response fixture for offline hybrid tests.")
    parser.add_argument("--glm-ocr-fixture", type=Path, help="Recorded GLM-OCR response fixture for offline hybrid tests.")
    parser.add_argument("--ocr-mode", choices=["hybrid", "ppocr", "glm"], default="hybrid", help="OCR engine routing mode.")
    parser.add_argument("--llm-mode", choices=["auto", "disabled", "required"], default="auto", help="Use LLM to structure GLM-OCR text before rule comparison.")
    args = parser.parse_args(argv)

    try:
        run_compare_package_image(
            standard_dir=args.standard_dir,
            image_path=args.image,
            output_dir=args.output_dir,
            ocr_fixture_path=args.ocr_fixture,
            ppocr_fixture_path=args.ppocr_fixture,
            glm_ocr_fixture_path=args.glm_ocr_fixture,
            ocr_mode=args.ocr_mode,
            llm_mode=args.llm_mode,
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
    ppocr_fixture_path: Path | None = None,
    glm_ocr_fixture_path: Path | None = None,
    ocr_mode: OcrMode = "hybrid",
    llm_mode: LlmMode = "auto",
    llm_client: LlmClient | None = None,
) -> dict[str, Any]:
    standard_dir = standard_dir.resolve()
    image_path = image_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ImageCompareError(f"Unsupported package image type: {image_path.suffix or '<none>'}")
    if not image_path.exists():
        raise ImageCompareError(f"Package image does not exist: {image_path}")
    legacy_fixture_only = ocr_fixture_path is not None and ppocr_fixture_path is None and glm_ocr_fixture_path is None
    ppocr_fixture_path = ppocr_fixture_path or ocr_fixture_path
    artifacts = load_standard_artifacts(standard_dir)
    input_image_width, input_image_height = image_size(image_path)
    config = RuntimeConfig.from_env(require_secrets=False)
    ocr_run = _recognize_for_mode(
        image_path=image_path,
        input_image_width=input_image_width,
        input_image_height=input_image_height,
        config=config,
        ocr_mode=ocr_mode,
        ppocr_fixture_path=ppocr_fixture_path,
        glm_ocr_fixture_path=glm_ocr_fixture_path,
        legacy_fixture_only=legacy_fixture_only,
    )
    ocr_lines = ocr_run["comparison_lines"]
    ppocr_lines = ocr_run["ppocr_lines"]
    glm_lines = ocr_run["glm_lines"]
    ocr_quality_report = build_ocr_quality_report(
        fixture_path=ocr_run.get("quality_fixture_path"),
        input_image_width=input_image_width,
        input_image_height=input_image_height,
        ocr_page_width=int(ocr_run["comparison_page_size"]["width"]),
        ocr_page_height=int(ocr_run["comparison_page_size"]["height"]),
        ocr_lines=ocr_lines,
    )
    ocr_quality_report["sources"] = ocr_run["sources"]
    active_llm_client = llm_client or _llm_client_for_mode(llm_mode, config)
    structure_run = run_package_structure_stage(
        artifacts=artifacts,
        ocr_lines=ocr_run["structure_lines"],
        llm_mode=llm_mode,
        llm_client=active_llm_client,
    )
    package_structure = structure_run.package_structured_items if structure_run.package_structured_items.get("enabled") else None
    package_structure_scope = "all"
    comparison = compare_standard_to_ocr(
        artifacts,
        ocr_lines,
        image_path,
        package_structure=package_structure,
        package_structure_scope=package_structure_scope,
    )
    package_ocr_lines = [to_jsonable(line) for line in ocr_lines]
    package_ppocr_lines = [to_jsonable(line) for line in ppocr_lines]
    package_glm_lines = [to_jsonable(line) for line in glm_lines]
    package_overlay_lines = [to_jsonable(line) for line in ocr_run["overlay_lines"]]
    fusion_evidence = _fusion_evidence(ocr_run, structure_run, package_structure_scope)
    fusion_quality = _fusion_quality_report(ocr_run, fusion_evidence)

    copied_image = output_dir / f"package_image{image_path.suffix.lower()}"
    if copied_image.resolve() != image_path:
        shutil.copy2(image_path, copied_image)

    runtime_policy = {
        "artifact_version": "package_image_compare_runtime_policy_v0.1",
        "source": "compare-package-image",
        "input_format": image_path.suffix.lower().lstrip("."),
        "ocr": {
            "mode": ocr_run["effective_mode"],
            "requested_mode": ocr_mode,
            "fixture_path": str(ocr_fixture_path.resolve()) if ocr_fixture_path else None,
            "ppocr_fixture_path": str(ppocr_fixture_path.resolve()) if ppocr_fixture_path else None,
            "glm_ocr_fixture_path": str(glm_ocr_fixture_path.resolve()) if glm_ocr_fixture_path else None,
            "line_count": len(ocr_lines),
            "bbox_available_count": sum(1 for line in ocr_lines if line.bbox_normalized is not None),
            "ocr_page_size": ocr_run["comparison_page_size"],
            "bbox_overlay_status": ocr_quality_report.get("bbox_overlay_status"),
        },
        "llm_agent": {"enabled": False, "mode": "not_applicable"},
        "llm_structure": {
            **structure_run.runtime,
            "requested_mode": llm_mode,
            "model": config.llm_model if structure_run.runtime.get("enabled") else None,
            "base_url_configured": bool(config.llm_base_url),
            "values_redacted": True,
        },
    }

    write_json(output_dir / "runtime_policy.json", runtime_policy)
    write_json(output_dir / "package_ocr_quality_report.json", ocr_quality_report)
    write_package_structure_artifacts(output_dir, structure_run)
    write_json(output_dir / "standard_targets.json", comparison["standard_targets"])
    write_json(output_dir / "package_ocr_lines.json", package_ocr_lines)
    write_json(output_dir / "package_overlay_lines.json", package_overlay_lines)
    write_json(output_dir / "package_ppocr_lines.json", package_ppocr_lines)
    write_json(output_dir / "package_glm_lines.json", package_glm_lines)
    write_json(output_dir / "package_fusion_evidence.json", fusion_evidence)
    write_json(output_dir / "package_fusion_quality_report.json", fusion_quality)
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
                *([_file_inventory_item("ppocr_fixture", ppocr_fixture_path.resolve())] if ppocr_fixture_path else []),
                *([_file_inventory_item("glm_ocr_fixture", glm_ocr_fixture_path.resolve())] if glm_ocr_fixture_path else []),
            ]
        },
    )
    write_image_compare_html(
        output_dir / "result_preview.html",
        image_path=copied_image,
        image_width=input_image_width,
        image_height=input_image_height,
        package_ocr_lines=package_overlay_lines,
        comparison_result=comparison["comparison_result"],
        standard_targets=comparison["standard_targets"],
        unmatched_print_text=comparison["unmatched_print_text"],
        ocr_quality_report=ocr_quality_report,
    )
    _write_artifact_index(output_dir)
    return comparison["comparison_result"]


def _llm_client_for_mode(llm_mode: LlmMode, config: RuntimeConfig) -> LlmClient | None:
    if llm_mode == "disabled":
        return None
    configured = bool(config.llm_api_key and config.llm_base_url and config.llm_model)
    if not configured:
        if llm_mode == "required":
            raise ConfigError("LLM package structure requires LLM_API_KEY, LLM_BASE_URL and LLM_MODEL.")
        return None
    return OpenAICompatibleLlmClient(config)


def _recognize_for_mode(
    *,
    image_path: Path,
    input_image_width: int,
    input_image_height: int,
    config: RuntimeConfig,
    ocr_mode: OcrMode,
    ppocr_fixture_path: Path | None,
    glm_ocr_fixture_path: Path | None,
    legacy_fixture_only: bool,
) -> dict[str, Any]:
    effective_mode = _effective_ocr_mode(ocr_mode, config, ppocr_fixture_path, glm_ocr_fixture_path, legacy_fixture_only)
    raw_ppocr_lines: list[OcrLine] = []
    raw_glm_lines: list[OcrLine] = []
    ppocr_page_size = {"width": input_image_width, "height": input_image_height}
    glm_page_size = {"width": input_image_width, "height": input_image_height}

    if effective_mode in {"hybrid", "ppocr"}:
        ppocr_page_size, raw_ppocr_lines = _recognize_ppocr(
            image_path=image_path,
            input_image_width=input_image_width,
            input_image_height=input_image_height,
            config=config,
            fixture_path=ppocr_fixture_path,
        )
    if effective_mode in {"hybrid", "glm"}:
        glm_page_size, raw_glm_lines = _recognize_glm(
            image_path=image_path,
            input_image_width=input_image_width,
            input_image_height=input_image_height,
            config=config,
            fixture_path=glm_ocr_fixture_path,
        )

    ppocr_lines = _prefix_ocr_lines(raw_ppocr_lines, "ppocr") if effective_mode == "hybrid" else raw_ppocr_lines
    glm_lines = _prefix_ocr_lines(raw_glm_lines, "glm") if effective_mode == "hybrid" else raw_glm_lines
    comparison_lines = ppocr_lines if ppocr_lines else glm_lines
    structure_lines = glm_lines if glm_lines else comparison_lines
    return {
        "requested_mode": ocr_mode,
        "effective_mode": effective_mode,
        "ppocr_lines": ppocr_lines,
        "glm_lines": glm_lines,
        "comparison_lines": comparison_lines,
        "structure_lines": structure_lines,
        "overlay_lines": [*ppocr_lines, *glm_lines] if effective_mode == "hybrid" else comparison_lines,
        "comparison_page_size": ppocr_page_size if ppocr_lines else glm_page_size,
        "quality_fixture_path": ppocr_fixture_path if ppocr_lines else glm_ocr_fixture_path,
        "sources": {
            "ppocr": _source_summary(ppocr_lines, ppocr_fixture_path, ppocr_page_size),
            "glm_ocr": _source_summary(glm_lines, glm_ocr_fixture_path, glm_page_size),
        },
    }


def _effective_ocr_mode(
    requested: OcrMode,
    config: RuntimeConfig,
    ppocr_fixture_path: Path | None,
    glm_ocr_fixture_path: Path | None,
    legacy_fixture_only: bool,
) -> OcrMode:
    if requested != "hybrid":
        return requested
    if legacy_fixture_only:
        return "ppocr"
    ppocr_available = ppocr_fixture_path is not None or bool(config.ppocrv6_api_key)
    glm_available = glm_ocr_fixture_path is not None or bool(config.glm_ocr_api_key)
    if ppocr_available and glm_available:
        return "hybrid"
    if ppocr_fixture_path is not None and not glm_available:
        return "ppocr"
    if glm_ocr_fixture_path is not None and not ppocr_available:
        return "glm"
    missing = []
    if not ppocr_available:
        missing.append("PPOCRV6_API_KEY/PPOCRV6_TOKEN")
    if not glm_available:
        missing.append("GLM_OCR_API_KEY")
    raise ConfigError(f"Missing required environment variables for hybrid OCR: {', '.join(missing)}")


def _recognize_ppocr(
    *,
    image_path: Path,
    input_image_width: int,
    input_image_height: int,
    config: RuntimeConfig,
    fixture_path: Path | None,
) -> tuple[dict[str, int], list[OcrLine]]:
    if fixture_path:
        width, height = normalize_ppocr_fixture_page(fixture_path, input_image_width, input_image_height)
        client = RecordedOcrClient(fixture_path)
    else:
        width, height = input_image_width, input_image_height
        if not config.ppocrv6_api_key:
            raise ConfigError("Missing required environment variables: PPOCRV6_API_KEY/PPOCRV6_TOKEN")
        client = PPOCRV6Client(config)
    lines = client.recognize_image(image_path, PageInfo(page=1, width=width, height=height))
    return {"width": width, "height": height}, lines


def _recognize_glm(
    *,
    image_path: Path,
    input_image_width: int,
    input_image_height: int,
    config: RuntimeConfig,
    fixture_path: Path | None,
) -> tuple[dict[str, int], list[OcrLine]]:
    if fixture_path:
        width, height = normalize_ppocr_fixture_page(fixture_path, input_image_width, input_image_height)
        client = RecordedOcrClient(fixture_path)
    else:
        width, height = input_image_width, input_image_height
        if not config.glm_ocr_api_key:
            raise ConfigError("Missing required environment variables: GLM_OCR_API_KEY")
        client = GLMOcrClient(config)
    lines = client.recognize_image(image_path, PageInfo(page=1, width=width, height=height))
    return {"width": width, "height": height}, lines


def _prefix_ocr_lines(lines: list[OcrLine], prefix: str) -> list[OcrLine]:
    return [
        OcrLine(
            ocr_line_id=f"{prefix}_{line.ocr_line_id}",
            page=line.page,
            text=line.text,
            confidence=line.confidence,
            bbox_pdf=line.bbox_pdf,
            bbox_normalized=line.bbox_normalized,
            block_id=f"{prefix}_{line.block_id}" if line.block_id else None,
            tokens=line.tokens,
            metadata={**line.metadata, "original_ocr_line_id": line.ocr_line_id, "provider": line.metadata.get("provider") or prefix},
        )
        for line in lines
    ]


def _source_summary(lines: list[OcrLine], fixture_path: Path | None, page_size: dict[str, int]) -> dict[str, Any]:
    return {
        "line_count": len(lines),
        "bbox_available_count": sum(1 for line in lines if line.bbox_normalized is not None),
        "fixture_path": str(fixture_path.resolve()) if fixture_path else None,
        "page_size": page_size,
    }


def _fusion_evidence(ocr_run: dict[str, Any], structure_run: Any, package_structure_scope: str) -> dict[str, Any]:
    effective_mode = str(ocr_run.get("effective_mode") or "")
    structure_enabled = bool(structure_run.package_structured_items.get("enabled"))
    if effective_mode == "hybrid" and structure_enabled and package_structure_scope == "all":
        field_text_source = "llm_structured_with_ppocr_fallback"
    else:
        field_text_source = "ppocr" if ocr_run.get("ppocr_lines") else "glm_ocr"
    return {
        "artifact_version": "package_fusion_evidence_v0.1",
        "mode": effective_mode,
        "field_text_source": field_text_source,
        "nutrition_table_source": "glm_ocr" if structure_enabled and ocr_run.get("glm_lines") else "ppocr",
        "package_structure_scope": package_structure_scope,
        "ppocr_line_count": len(ocr_run.get("ppocr_lines") or []),
        "glm_line_count": len(ocr_run.get("glm_lines") or []),
        "llm_structure_enabled": structure_enabled,
    }


def _fusion_quality_report(ocr_run: dict[str, Any], fusion_evidence: dict[str, Any]) -> dict[str, Any]:
    flags = []
    if fusion_evidence.get("mode") == "hybrid" and not fusion_evidence.get("llm_structure_enabled"):
        flags.append("glm_structure_disabled")
    return {
        "artifact_version": "package_fusion_quality_report_v0.1",
        "status": "pass" if not flags else "review_required",
        "mode": ocr_run.get("effective_mode"),
        "quality_flags": flags,
        "field_text_source": fusion_evidence.get("field_text_source"),
        "nutrition_table_source": fusion_evidence.get("nutrition_table_source"),
    }


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
