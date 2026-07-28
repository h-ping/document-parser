from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import json
import re
from pathlib import Path
from typing import Any, Literal

from .llm import LlmClient, LlmError
from .models import OcrLine, to_jsonable
from .utils import stable_id, write_json


LlmMode = Literal["auto", "disabled", "required"]
PACKAGE_STRUCTURE_LLM_MAX_TOKENS = 32768


class PackageStructureError(RuntimeError):
    pass


PACKAGE_STRUCTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fields": {
            "type": "array",
            "maxItems": 240,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "semantic_key": {"type": "string"},
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "string"}},
                    "group_id": {"type": ["string", "null"]},
                    "table_id": {"type": ["string", "null"]},
                    "row_key": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "review_required": {"type": "boolean"},
                },
                "required": ["semantic_key", "label", "text", "source_ids", "confidence", "review_required"],
            },
        },
        "field_groups": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "group_id": {"type": "string"},
                    "group_type": {"type": "string"},
                    "label": {"type": "string"},
                    "source_ids": {"type": "array", "maxItems": 20, "items": {"type": "string"}},
                    "fields": {
                        "type": "array",
                        "maxItems": 30,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "semantic_key": {"type": "string"},
                                "label": {"type": "string"},
                                "text": {"type": "string"},
                                "source_ids": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "string"}},
                                "confidence": {"type": "number"},
                                "review_required": {"type": "boolean"},
                            },
                            "required": ["semantic_key", "label", "text", "source_ids", "confidence", "review_required"],
                        },
                    },
                },
                "required": ["group_id", "group_type", "label", "fields"],
            },
        },
        "content_items": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "group_id": {"type": "string"},
                    "content_index": {"type": "integer"},
                    "source_ids": {"type": "array", "maxItems": 20, "items": {"type": "string"}},
                    "name": {"type": ["string", "null"]},
                    "net_content": {"type": ["string", "null"]},
                    "product_classification": {"type": ["string", "null"]},
                    "ingredients": {"type": ["string", "null"]},
                    "field_source_ids": {
                        "type": "object",
                        "additionalProperties": {"type": "array", "maxItems": 20, "items": {"type": "string"}},
                    },
                    "confidence": {"type": "number"},
                    "review_required": {"type": "boolean"},
                },
                "required": ["group_id", "content_index", "confidence", "review_required"],
            },
        },
        "nutrition_tables": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "table_id": {"type": ["string", "null"]},
                    "title": {"type": "string"},
                    "source_ids": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "string"}},
                    "columns": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "maxItems": 80,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "row_key": {"type": "string"},
                                "cells": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
                                "source_ids": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "string"}},
                                "confidence": {"type": "number"},
                                "review_required": {"type": "boolean"},
                            },
                            "required": ["row_key", "cells", "source_ids", "confidence", "review_required"],
                        },
                    },
                    "footnotes": {
                        "type": "array",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "source_ids": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "string"}},
                                "confidence": {"type": "number"},
                                "review_required": {"type": "boolean"},
                            },
                            "required": ["text", "source_ids", "confidence", "review_required"],
                        },
                    },
                    "confidence": {"type": "number"},
                    "review_required": {"type": "boolean"},
                },
                "required": ["title", "source_ids", "columns", "rows", "footnotes", "confidence", "review_required"],
            },
        },
        "other_text": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "review_required": {"type": "boolean"},
                },
                "required": ["label", "text", "source_ids", "confidence", "review_required"],
            },
        },
        "warnings": {"type": "array", "maxItems": 80, "items": {"type": "string"}},
    },
    "required": ["fields", "field_groups", "content_items", "nutrition_tables", "other_text", "warnings"],
}


@dataclass(frozen=True)
class StructureRun:
    package_glm_blocks: dict[str, Any]
    package_ppocr_blocks: dict[str, Any]
    package_llm_structure_input: dict[str, Any]
    package_llm_structure_chunks: dict[str, Any]
    package_llm_structure_output: dict[str, Any]
    package_structured_items: dict[str, Any]
    package_structure_quality_report: dict[str, Any]
    package_fusion_structure_quality_report: dict[str, Any]
    runtime: dict[str, Any]


def run_package_structure_stage(
    *,
    artifacts: dict[str, Any],
    ocr_lines: list[OcrLine] | None = None,
    ppocr_lines: list[OcrLine] | None = None,
    glm_lines: list[OcrLine] | None = None,
    llm_mode: LlmMode,
    llm_client: LlmClient | None,
) -> StructureRun:
    split_lines = _split_ocr_lines_for_structure(ocr_lines or [])
    resolved_glm_lines = glm_lines if glm_lines is not None else split_lines["glm"]
    resolved_ppocr_lines = ppocr_lines if ppocr_lines is not None else split_lines["ppocr"]
    package_glm_blocks = build_package_glm_blocks(resolved_glm_lines)
    package_ppocr_blocks = build_package_ppocr_blocks(resolved_ppocr_lines)
    evidence_bundle = _evidence_bundle(package_glm_blocks, package_ppocr_blocks)
    structure_input = build_package_llm_structure_input(artifacts, package_glm_blocks, package_ppocr_blocks)
    if llm_mode == "disabled":
        return _disabled_run(package_glm_blocks, package_ppocr_blocks, structure_input, "llm_mode_disabled")
    if not structure_input["evidence_chunks"]:
        if llm_mode == "required":
            raise PackageStructureError("LLM package structure requires OCR evidence chunks, but none were found.")
        return _disabled_run(package_glm_blocks, package_ppocr_blocks, structure_input, "no_ocr_evidence_chunks")
    if llm_client is None:
        if llm_mode == "required":
            raise PackageStructureError("LLM package structure requires LLM_API_KEY, LLM_BASE_URL and LLM_MODEL.")
        return _disabled_run(package_glm_blocks, package_ppocr_blocks, structure_input, "llm_env_not_configured")

    try:
        chunk_outputs = []
        for chunk in _as_list(structure_input.get("evidence_chunks")):
            chunk_input = _chunk_structure_input(structure_input, _as_dict(chunk))
            chunk_outputs.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "body": llm_client.structured_json_with_max_tokens(
                        _llm_system_prompt(),
                        json.dumps(chunk_input, ensure_ascii=False),
                        PACKAGE_STRUCTURE_SCHEMA,
                        PACKAGE_STRUCTURE_LLM_MAX_TOKENS,
                    ),
                }
            )
    except LlmError as exc:
        if llm_mode == "required":
            raise PackageStructureError(f"LLM package structure failed: {exc}") from exc
        return _disabled_run(package_glm_blocks, package_ppocr_blocks, structure_input, f"llm_error:{exc.__class__.__name__}")

    llm_output = _merge_llm_chunk_outputs([_as_dict(item.get("body")) for item in chunk_outputs])
    package_structured_items, quality_report = normalize_package_structure_output(llm_output, evidence_bundle)
    chunk_artifact = _chunk_artifact(structure_input, chunk_outputs)
    return StructureRun(
        package_glm_blocks=package_glm_blocks,
        package_ppocr_blocks=package_ppocr_blocks,
        package_llm_structure_input=structure_input,
        package_llm_structure_chunks=chunk_artifact,
        package_llm_structure_output={
            "enabled": True,
            "mode": "llm_fusion_ocr_structure",
            "body": llm_output,
            "chunk_count": len(chunk_outputs),
        },
        package_structured_items=package_structured_items,
        package_structure_quality_report=quality_report,
        package_fusion_structure_quality_report=_fusion_structure_quality_report(
            package_glm_blocks,
            package_ppocr_blocks,
            quality_report,
            len(chunk_outputs),
        ),
        runtime={
            "enabled": True,
            "mode": "llm_fusion_ocr_structure",
            "disabled_reason": None,
            "final_decision_owner": "rules",
        },
    )


def write_package_structure_artifacts(output_dir: Path, run: StructureRun) -> None:
    write_json(output_dir / "package_glm_blocks.json", run.package_glm_blocks)
    write_json(output_dir / "package_ppocr_blocks.json", run.package_ppocr_blocks)
    write_json(output_dir / "package_llm_structure_input.json", run.package_llm_structure_input)
    write_json(output_dir / "package_llm_structure_chunks.json", run.package_llm_structure_chunks)
    write_json(output_dir / "package_llm_structure_output.json", run.package_llm_structure_output)
    write_json(output_dir / "package_structured_items.json", run.package_structured_items)
    write_json(output_dir / "package_structure_quality_report.json", run.package_structure_quality_report)
    write_json(output_dir / "package_fusion_structure_quality_report.json", run.package_fusion_structure_quality_report)


def build_package_glm_blocks(ocr_lines: list[OcrLine]) -> dict[str, Any]:
    grouped: dict[str, list[OcrLine]] = {}
    for line in ocr_lines:
        if not _is_glm_line(line):
            continue
        raw_block_id = str(line.block_id or line.metadata.get("block_id") or stable_id("glm_block", len(grouped) + 1))
        grouped.setdefault(raw_block_id, []).append(line)

    blocks = []
    for order, (raw_block_id, lines) in enumerate(grouped.items(), start=1):
        block_id = _source_id("glm", "block", raw_block_id)
        label = str(lines[0].metadata.get("detail_label") or "")
        raw_lines = [line.text for line in lines if line.text.strip()]
        cleaned_lines = [clean_glm_text(line) for line in raw_lines]
        cleaned_lines = [line for line in cleaned_lines if line]
        table_rows = _html_table_rows("\n".join(raw_lines))
        line_entries = [
            {
                "line_id": _source_id("glm", "line", line.ocr_line_id),
                "ocr_line_id": line.ocr_line_id,
                "text": clean_glm_text(line.text),
                "bbox_normalized": to_jsonable(line.bbox_normalized),
                "confidence": line.confidence,
            }
            for line in lines
            if line.text.strip()
        ]
        block = {
            "block_id": block_id,
            "raw_block_id": raw_block_id,
            "order": order,
            "label": label,
            "detail_index": lines[0].metadata.get("detail_index"),
            "source_ocr_line_ids": [line.ocr_line_id for line in lines],
            "lines": line_entries,
            "raw_text": "\n".join(raw_lines),
            "cleaned_text": "\n".join(cleaned_lines),
            "bbox_normalized": _union_line_bbox(lines),
            "table": _table_block(block_id, table_rows) if table_rows else None,
        }
        blocks.append(block)

    provider_features = {
        "provider": "glm_ocr" if blocks else "not_glm_ocr",
        "block_count": len(blocks),
        "table_block_count": sum(1 for block in blocks if block.get("table")),
        "html_table_count": sum(1 for block in blocks if "<table" in str(block.get("raw_text") or "").lower()),
        "markup_cleaned_count": sum(1 for block in blocks if block.get("raw_text") != block.get("cleaned_text")),
    }
    return {"artifact_version": "package_glm_blocks_v0.1", "provider_features": provider_features, "blocks": blocks}


def build_package_ppocr_blocks(ocr_lines: list[OcrLine]) -> dict[str, Any]:
    ppocr_lines = [line for line in ocr_lines if _is_ppocr_line(line) and line.text.strip()]
    ordered = sorted(ppocr_lines, key=lambda line: (_line_x_band(line), _line_y1(line), _line_x1(line)))
    paragraphs: list[list[OcrLine]] = []
    current: list[OcrLine] = []
    for line in ordered:
        if not current or _same_ppocr_paragraph(current[-1], line, len(current)):
            current.append(line)
        else:
            paragraphs.append(current)
            current = [line]
    if current:
        paragraphs.append(current)

    blocks = []
    for order, lines in enumerate(paragraphs, start=1):
        block_id = _source_id("pp", "block", f"{order:03d}")
        line_entries = [
            {
                "line_id": _source_id("pp", "line", line.ocr_line_id),
                "ocr_line_id": line.ocr_line_id,
                "text": clean_glm_text(line.text),
                "bbox_normalized": to_jsonable(line.bbox_normalized),
                "confidence": line.confidence,
            }
            for line in lines
        ]
        cleaned_lines = [entry["text"] for entry in line_entries if str(entry["text"]).strip()]
        blocks.append(
            {
                "block_id": block_id,
                "order": order,
                "label": "paragraph",
                "source_ocr_line_ids": [line.ocr_line_id for line in lines],
                "lines": line_entries,
                "raw_text": "\n".join(line.text for line in lines if line.text.strip()),
                "cleaned_text": "\n".join(str(line) for line in cleaned_lines),
                "bbox_normalized": _union_line_bbox(lines),
                "confidence": sum(line.confidence for line in lines) / len(lines),
            }
        )

    provider_features = {
        "provider": "ppocrv6" if blocks else "not_ppocr",
        "block_count": len(blocks),
        "line_count": len(ppocr_lines),
        "paragraph_block_count": len(blocks),
    }
    return {"artifact_version": "package_ppocr_blocks_v0.1", "provider_features": provider_features, "blocks": blocks}


def clean_glm_text(text: str) -> str:
    value = unescape(text).strip()
    value = re.sub(r"</?div[^>]*>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value)
    value = re.sub(r"^\s*[*+-]\s+", "", value)
    value = value.replace("\\geqslant", "≥").replace("\\ge", "≥").replace("\\leqslant", "≤").replace("\\le", "≤")
    value = value.replace("\\%", "%").replace("$", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def build_package_llm_structure_input(
    artifacts: dict[str, Any],
    package_glm_blocks: dict[str, Any],
    package_ppocr_blocks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_ppocr_blocks = package_ppocr_blocks or {"blocks": []}
    chunks = _build_evidence_chunks(artifacts, package_glm_blocks, resolved_ppocr_blocks)
    return {
        "artifact_version": "package_llm_structure_input_v0.2",
        "task": "structure_package_text_from_chunked_ocr_evidence",
        "rules": [
            "Only use text present in the provided chunk GLM-OCR blocks, PP-OCR blocks, lines, or table cells.",
            "Return package-side structured fields only; do not compare with the standard and do not decide pass/fail.",
            "Keep printed field prefixes such as 产品名称：, 配料表：, 地址：, 内容物2： when they appear in the source.",
            "Do not include Markdown markers or HTML tags as printed text.",
            "Use source ids exactly as provided, such as glm:block_*, glm:line_*, glm:cell_*, pp:block_*, or pp:line_*.",
            "For nutrition facts tables, prefer GLM table cells over PP-OCR text when both are present.",
            "The JSON root must contain exactly these keys: fields, field_groups, content_items, nutrition_tables, other_text, warnings.",
            "Do not return wrapper keys such as structured_items, result, data, or markdown.",
        ],
        "output_contract": _package_structure_output_contract(),
        "standard_context": {
            "standard_items": _compact_standard_items(artifacts.get("standard_items")),
            "field_groups": _compact_groups(artifacts.get("field_groups")),
            "tables": _compact_tables(artifacts.get("tables")),
        },
        "chunking": {
            "strategy": "standard_field_anchor_overlap",
            "chunk_count": len(chunks),
            "glm_block_count": len(_as_list(package_glm_blocks.get("blocks"))),
            "ppocr_block_count": len(_as_list(resolved_ppocr_blocks.get("blocks"))),
        },
        "evidence_chunks": chunks,
    }


def normalize_package_structure_output(body: dict[str, Any], package_glm_blocks: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_index = _source_index(package_glm_blocks)
    errors: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    fields = []
    tables = []

    raw_fields = list(_as_list(body.get("fields")))
    for group in _as_list(body.get("field_groups")):
        group_dict = _as_dict(group)
        for field in _as_list(group_dict.get("fields")):
            field_dict = {**_as_dict(field), "group_id": group_dict.get("group_id")}
            raw_fields.append(field_dict)
    for content_item in _as_list(body.get("content_items")):
        raw_fields.extend(_fields_from_content_item(_as_dict(content_item)))
    for other in _as_list(body.get("other_text")):
        other_dict = _as_dict(other)
        raw_fields.append(
            {
                "semantic_key": "custom.other_label_text",
                "label": other_dict.get("label") or "其他标签文字",
                "text": other_dict.get("text"),
                "source_ids": other_dict.get("source_ids"),
                "confidence": other_dict.get("confidence", 0.0),
                "review_required": other_dict.get("review_required", True),
            }
        )

    for index, item in enumerate(raw_fields, start=1):
        item_dict = _as_dict(item)
        if _empty_structured_placeholder(item_dict):
            continue
        normalized, item_errors = _normalize_structured_field(index, item_dict, source_index)
        errors.extend(item_errors)
        if normalized is None:
            rejected.append({"item": item, "errors": item_errors})
        else:
            fields.append(normalized)

    for index, table in enumerate(_as_list(body.get("nutrition_tables")), start=1):
        tables.append(_normalize_nutrition_table(index, _as_dict(table), source_index, errors))

    status = "pass" if not errors else "review_required"
    package_structured_items = {
        "artifact_version": "package_structured_items_v0.1",
        "enabled": True,
        "source": "llm_fusion_ocr_structure",
        "fields": fields,
        "field_groups": _as_list(body.get("field_groups")),
        "content_items": _as_list(body.get("content_items")),
        "nutrition_tables": tables,
        "other_text": _as_list(body.get("other_text")),
        "rejected_items": rejected,
    }
    quality_report = {
        "artifact_version": "package_structure_quality_report_v0.1",
        "status": status,
        "enabled": True,
        "error_count": len(errors),
        "errors": errors,
        "rejected_item_count": len(rejected),
        "field_count": len(fields),
        "nutrition_table_count": len(tables),
    }
    return package_structured_items, quality_report


def _disabled_run(
    package_glm_blocks: dict[str, Any],
    package_ppocr_blocks: dict[str, Any],
    structure_input: dict[str, Any],
    reason: str,
) -> StructureRun:
    fallback_tables = _glm_nutrition_table_fallback(package_glm_blocks)
    fallback_enabled = bool(fallback_tables)
    quality_report = {
        "artifact_version": "package_structure_quality_report_v0.1",
        "status": "review_required" if fallback_enabled else "disabled",
        "enabled": fallback_enabled,
        "disabled_reason": reason,
        "error_count": 0,
        "errors": [],
        "field_count": 0,
        "nutrition_table_count": len(fallback_tables),
    }
    return StructureRun(
        package_glm_blocks=package_glm_blocks,
        package_ppocr_blocks=package_ppocr_blocks,
        package_llm_structure_input=structure_input,
        package_llm_structure_chunks=_chunk_artifact(structure_input, []),
        package_llm_structure_output={"enabled": False, "mode": "disabled", "disabled_reason": reason, "body": {}},
        package_structured_items={
            "artifact_version": "package_structured_items_v0.1",
            "enabled": fallback_enabled,
            "source": "glm_table_fallback" if fallback_enabled else "disabled",
            "fields": [],
            "field_groups": [],
            "content_items": [],
            "nutrition_tables": fallback_tables,
            "other_text": [],
            "rejected_items": [],
        },
        package_structure_quality_report=quality_report,
        package_fusion_structure_quality_report=_fusion_structure_quality_report(
            package_glm_blocks,
            package_ppocr_blocks,
            quality_report,
            0,
        ),
        runtime={
            "enabled": False,
            "mode": "disabled",
            "disabled_reason": reason,
            "final_decision_owner": "rules",
        },
    )


def _glm_nutrition_table_fallback(package_glm_blocks: dict[str, Any]) -> list[dict[str, Any]]:
    tables = []
    for block_index, block in enumerate(_as_list(package_glm_blocks.get("blocks")), start=1):
        block_dict = _as_dict(block)
        table = _as_dict(block_dict.get("table"))
        rows = _as_list(table.get("rows"))
        text_rows = [_table_row_texts(row) for row in rows]
        if not _looks_like_nutrition_table(text_rows):
            continue
        header_index = _nutrition_header_index(text_rows)
        header_start = header_index if header_index is not None else 0
        header = text_rows[header_start]
        source_ocr_line_ids = [str(line_id) for line_id in _as_list(block_dict.get("source_ocr_line_ids"))]
        data_rows = []
        for row in rows[header_start + 1 :]:
            row_dict = _as_dict(row)
            cells = _as_list(row_dict.get("cells"))
            cell_texts = [clean_glm_text(str(_as_dict(cell).get("text") or "")) for cell in cells]
            cell_texts = [text for text in cell_texts if text]
            if not cell_texts:
                continue
            data_rows.append(
                {
                    "row_key": cell_texts[0],
                    "cells": cell_texts,
                    "source_ids": [str(_as_dict(cell).get("cell_id") or "") for cell in cells if _as_dict(cell).get("cell_id")],
                    "source_ocr_line_ids": source_ocr_line_ids,
                    "bbox_normalized": block_dict.get("bbox_normalized"),
                    "confidence": 1.0,
                    "review_required": False,
                }
            )
        if not data_rows:
            continue
        tables.append(
            {
                "structured_table_id": stable_id("pkg_glm_table", len(tables) + 1),
                "table_id": stable_id("glm_nutrition", block_index),
                "title": "营养成分表",
                "source_ids": [str(block_dict.get("block_id") or "")],
                "source_ocr_line_ids": source_ocr_line_ids,
                "bbox_normalized": block_dict.get("bbox_normalized"),
                "columns": header,
                "rows": data_rows,
                "footnotes": [],
                "confidence": 1.0,
                "review_required": False,
                "metadata": {"source": "glm_table_fallback", "source_provider": "glm"},
            }
        )
    return tables


def _table_row_texts(row: Any) -> list[str]:
    return [
        text
        for text in (clean_glm_text(str(_as_dict(cell).get("text") or "")) for cell in _as_list(_as_dict(row).get("cells")))
        if text
    ]


def _looks_like_nutrition_table(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    flattened = "".join("".join(row) for row in rows)
    nutrient_labels = ("能量", "蛋白质", "脂肪", "碳水化合物", "钠")
    return "项目" in flattened and any(label in flattened for label in nutrient_labels)


def _nutrition_header_index(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows):
        text = "".join(row)
        if "项目" in text and ("营养素参考值" in text or "NRV" in text or "nrv" in text.lower()):
            return index
    return None


def _empty_structured_placeholder(item: dict[str, Any]) -> bool:
    return not any(
        str(item.get(key) or "").strip()
        for key in ("semantic_key", "label", "text", "source_ids", "source_ocr_line_ids")
    )


def _llm_system_prompt() -> str:
    return (
        "You structure package artwork text from chunked GLM-OCR and PP-OCR evidence. "
        "You are not a consistency checker. Do not judge pass/fail. "
        "Return exactly one JSON object matching the output_contract/schema. "
        "Do not return wrapper keys such as structured_items, result, or data. "
        "Every text value must be copied from provided source text after removing OCR markup only. "
        "Every item must cite source_ids from the input."
    )


def _package_structure_output_contract() -> dict[str, Any]:
    return {
        "root_required_keys": ["fields", "field_groups", "content_items", "nutrition_tables", "other_text", "warnings"],
        "field": ["semantic_key", "label", "text", "source_ids", "group_id", "table_id", "row_key", "confidence", "review_required"],
        "field_group": ["group_id", "group_type", "label", "source_ids", "fields"],
        "content_item": [
            "group_id",
            "content_index",
            "source_ids",
            "name",
            "net_content",
            "product_classification",
            "ingredients",
            "field_source_ids",
            "confidence",
            "review_required",
        ],
        "nutrition_table": ["table_id", "title", "source_ids", "columns", "rows", "footnotes", "confidence", "review_required"],
        "nutrition_row": ["row_key", "cells", "source_ids", "confidence", "review_required"],
        "footnote": ["text", "source_ids", "confidence", "review_required"],
        "other_text": ["label", "text", "source_ids", "confidence", "review_required"],
    }


def _table_block(block_id: str, rows: list[list[str]]) -> dict[str, Any]:
    return {
        "rows": [
            {
                "row_index": row_index,
                "cells": [
                    {"cell_id": _source_id("glm", "cell", f"{block_id}_r{row_index}c{cell_index}"), "text": cell}
                    for cell_index, cell in enumerate(row, start=1)
                ],
            }
            for row_index, row in enumerate(rows, start=1)
        ]
    }


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(clean_glm_text("".join(self._current_cell)))
            self._current_cell = None
        elif tag.lower() == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def _html_table_rows(text: str) -> list[list[str]]:
    if "<table" not in text.lower():
        return []
    parser = _TableParser()
    parser.feed(text)
    return parser.rows


def _split_ocr_lines_for_structure(ocr_lines: list[OcrLine]) -> dict[str, list[OcrLine]]:
    return {
        "glm": [line for line in ocr_lines if _is_glm_line(line)],
        "ppocr": [line for line in ocr_lines if _is_ppocr_line(line)],
    }


def _is_glm_line(line: OcrLine) -> bool:
    return str(line.metadata.get("provider") or "") == "glm_ocr"


def _is_ppocr_line(line: OcrLine) -> bool:
    return not _is_glm_line(line)


def _source_id(provider: str, source_type: str, value: str) -> str:
    token = _source_token(value)
    return f"{provider}:{token}" if token.startswith(f"{source_type}_") else f"{provider}:{source_type}_{token}"


def _source_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", str(value)).strip("_")
    return token or "source"


def _line_x1(line: OcrLine) -> float:
    return float(line.bbox_normalized.x1) if line.bbox_normalized else 0.0


def _line_y1(line: OcrLine) -> float:
    return float(line.bbox_normalized.y1) if line.bbox_normalized else 0.0


def _line_y2(line: OcrLine) -> float:
    return float(line.bbox_normalized.y2) if line.bbox_normalized else 0.0


def _line_x_band(line: OcrLine) -> int:
    return int(_line_x1(line) / 0.12)


def _same_ppocr_paragraph(previous: OcrLine, current: OcrLine, current_size: int) -> bool:
    if current_size >= 8:
        return False
    if _line_x_band(previous) != _line_x_band(current):
        return False
    y_gap = _line_y1(current) - _line_y2(previous)
    return -0.01 <= y_gap <= 0.035


def _union_line_bbox(lines: list[OcrLine]) -> dict[str, float] | None:
    boxes = [line.bbox_normalized for line in lines if line.bbox_normalized is not None]
    if not boxes:
        return None
    return {
        "x1": min(box.x1 for box in boxes),
        "y1": min(box.y1 for box in boxes),
        "x2": max(box.x2 for box in boxes),
        "y2": max(box.y2 for box in boxes),
    }


def _evidence_bundle(package_glm_blocks: dict[str, Any], package_ppocr_blocks: dict[str, Any]) -> dict[str, Any]:
    return {"glm_blocks": package_glm_blocks, "ppocr_blocks": package_ppocr_blocks}


def _build_evidence_chunks(
    artifacts: dict[str, Any],
    package_glm_blocks: dict[str, Any],
    package_ppocr_blocks: dict[str, Any],
) -> list[dict[str, Any]]:
    chunks = []
    item_groups = _standard_items_by_chunk_type(_as_list(artifacts.get("standard_items")))
    for chunk_type, label in [
        ("main_label", "标签主文字"),
        ("enterprise", "企业信息"),
        ("content", "多内容物组合装"),
        ("other_text", "其他标签文字"),
    ]:
        items = item_groups.get(chunk_type, [])
        if not items:
            continue
        chunks.append(_evidence_chunk(len(chunks) + 1, chunk_type, label, items, [], package_glm_blocks, package_ppocr_blocks))
    tables = _as_list(artifacts.get("tables"))
    if tables:
        chunks.append(_evidence_chunk(len(chunks) + 1, "nutrition", "营养成分表", [], tables, package_glm_blocks, package_ppocr_blocks))
    if not chunks and (_as_list(package_glm_blocks.get("blocks")) or _as_list(package_ppocr_blocks.get("blocks"))):
        chunks.append(_evidence_chunk(1, "fallback", "全部文字", [], [], package_glm_blocks, package_ppocr_blocks))
    return chunks


def _standard_items_by_chunk_type(items: list[Any]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"main_label": [], "enterprise": [], "content": [], "other_text": []}
    for item in items:
        item_dict = _as_dict(item)
        key = str(item_dict.get("semantic_key") or "")
        label = str(item_dict.get("label") or "")
        if key == "custom.other_label_text":
            grouped["other_text"].append(item_dict)
        elif key.startswith("content_item."):
            grouped["content"].append(item_dict)
        elif key.startswith("manufacturer.") or key.startswith("enterprise.") or _enterprise_label(label):
            grouped["enterprise"].append(item_dict)
        elif key != "product.nutrition_table":
            grouped["main_label"].append(item_dict)
    return grouped


def _enterprise_label(label: str) -> bool:
    return any(token in label for token in ("企业", "委托", "受托", "生产者", "地址", "产地", "许可证", "邮编", "网站", "电话"))


def _evidence_chunk(
    chunk_index: int,
    chunk_type: str,
    label: str,
    standard_items: list[dict[str, Any]],
    tables: list[Any],
    package_glm_blocks: dict[str, Any],
    package_ppocr_blocks: dict[str, Any],
) -> dict[str, Any]:
    anchors = _chunk_anchors(standard_items, tables, chunk_type)
    glm_blocks = [_compact_glm_block_for_llm(block) for block in _select_blocks(_as_list(package_glm_blocks.get("blocks")), anchors, 14)]
    ppocr_blocks = [_compact_ppocr_block_for_llm(block) for block in _select_blocks(_as_list(package_ppocr_blocks.get("blocks")), anchors, 22)]
    return {
        "chunk_id": stable_id("pkg_chunk", chunk_index),
        "chunk_type": chunk_type,
        "label": label,
        "standard_items": _compact_standard_items(standard_items),
        "tables": _compact_tables(tables),
        "anchors": anchors[:40],
        "glm_blocks": glm_blocks,
        "ppocr_blocks": ppocr_blocks,
    }


def _compact_glm_block_for_llm(block: dict[str, Any]) -> dict[str, Any]:
    block_dict = _as_dict(block)
    table = _as_dict(block_dict.get("table"))
    compact = {
        "block_id": block_dict.get("block_id"),
        "order": block_dict.get("order"),
        "label": block_dict.get("label"),
        "cleaned_text": block_dict.get("cleaned_text"),
        "lines": [
            {"line_id": _as_dict(line).get("line_id"), "text": _as_dict(line).get("text")}
            for line in _as_list(block_dict.get("lines"))
        ],
    }
    if table:
        compact["table"] = table
    return compact


def _compact_ppocr_block_for_llm(block: dict[str, Any]) -> dict[str, Any]:
    block_dict = _as_dict(block)
    return {
        "block_id": block_dict.get("block_id"),
        "order": block_dict.get("order"),
        "cleaned_text": block_dict.get("cleaned_text"),
        "confidence": block_dict.get("confidence"),
        "lines": [
            {"line_id": _as_dict(line).get("line_id"), "text": _as_dict(line).get("text"), "confidence": _as_dict(line).get("confidence")}
            for line in _as_list(block_dict.get("lines"))
        ],
    }


def _chunk_anchors(standard_items: list[dict[str, Any]], tables: list[Any], chunk_type: str) -> list[str]:
    anchors = [
        "产品名称",
        "净含量",
        "保质期",
        "贮存条件",
        "配料",
        "致敏",
        "地址",
        "生产者",
        "营养成分表",
    ]
    if chunk_type == "nutrition":
        anchors.extend(["项目", "每100", "营养素参考值", "NRV", "能量", "蛋白质", "脂肪", "钠"])
    for item in standard_items:
        anchors.extend(_anchors_from_text(str(item.get("label") or "")))
        anchors.extend(_anchors_from_text(str(item.get("text") or "")))
    for table in tables:
        table_dict = _as_dict(table)
        anchors.extend(_anchors_from_text(str(table_dict.get("title") or "")))
        for row in _as_list(table_dict.get("rows")):
            anchors.extend(_anchors_from_text(str(_as_dict(row).get("row_key") or "")))
    return _dedupe_anchors(anchors)


def _anchors_from_text(text: str) -> list[str]:
    cleaned = clean_glm_text(text)
    prefix = re.split(r"[:：]", cleaned, maxsplit=1)[0]
    anchors = [prefix] if 1 < len(prefix) <= 16 else []
    anchors.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9®™]{2,18}", cleaned))
    anchors.extend(re.findall(r"\d+(?:\.\d+)?\s*(?:克|毫克|微克|千焦|%|个月|年|月|日)", cleaned))
    return anchors


def _dedupe_anchors(anchors: list[str]) -> list[str]:
    seen = set()
    output = []
    for anchor in anchors:
        cleaned = clean_glm_text(anchor)
        norm = _evidence_norm(cleaned)
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        output.append(cleaned)
    return output


def _select_blocks(blocks: list[Any], anchors: list[str], limit: int) -> list[dict[str, Any]]:
    block_dicts = [_as_dict(block) for block in blocks]
    selected_indexes = set()
    for index, block in enumerate(block_dicts):
        if _text_matches_anchors(str(block.get("cleaned_text") or block.get("raw_text") or ""), anchors):
            selected_indexes.update({index - 1, index, index + 1})
    if not selected_indexes and block_dicts:
        selected_indexes.update(range(min(8, len(block_dicts))))
    selected = [block_dicts[index] for index in sorted(selected_indexes) if 0 <= index < len(block_dicts)]
    return selected[:limit]


def _text_matches_anchors(text: str, anchors: list[str]) -> bool:
    norm_text = _evidence_norm(text)
    return any(_evidence_norm(anchor) in norm_text for anchor in anchors if len(_evidence_norm(anchor)) >= 2)


def _chunk_structure_input(structure_input: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_version": structure_input.get("artifact_version"),
        "task": structure_input.get("task"),
        "rules": structure_input.get("rules"),
        "output_contract": structure_input.get("output_contract"),
        "standard_context": {
            "standard_items": chunk.get("standard_items", []),
            "field_groups": _as_dict(structure_input.get("standard_context")).get("field_groups", []),
            "tables": chunk.get("tables", []),
        },
        "evidence_chunk": chunk,
    }


def _merge_llm_chunk_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "fields": [],
        "field_groups": [],
        "content_items": [],
        "nutrition_tables": [],
        "other_text": [],
        "warnings": [],
    }
    for body in outputs:
        for key in merged:
            merged[key].extend(_as_list(body.get(key)))
    return merged


def _chunk_artifact(structure_input: dict[str, Any], chunk_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    output_by_id = {str(item.get("chunk_id")): item for item in chunk_outputs}
    chunks = []
    for chunk in _as_list(structure_input.get("evidence_chunks")):
        chunk_dict = _as_dict(chunk)
        chunk_id = str(chunk_dict.get("chunk_id") or "")
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": chunk_dict.get("chunk_type"),
                "standard_item_count": len(_as_list(chunk_dict.get("standard_items"))),
                "table_count": len(_as_list(chunk_dict.get("tables"))),
                "glm_block_count": len(_as_list(chunk_dict.get("glm_blocks"))),
                "ppocr_block_count": len(_as_list(chunk_dict.get("ppocr_blocks"))),
                "llm_output_enabled": bool(output_by_id.get(chunk_id)),
            }
        )
    return {
        "artifact_version": "package_llm_structure_chunks_v0.1",
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def _fusion_structure_quality_report(
    package_glm_blocks: dict[str, Any],
    package_ppocr_blocks: dict[str, Any],
    quality_report: dict[str, Any],
    llm_chunk_count: int,
) -> dict[str, Any]:
    return {
        "artifact_version": "package_fusion_structure_quality_report_v0.1",
        "status": quality_report.get("status"),
        "glm_block_count": len(_as_list(package_glm_blocks.get("blocks"))),
        "ppocr_block_count": len(_as_list(package_ppocr_blocks.get("blocks"))),
        "llm_chunk_count": llm_chunk_count,
        "final_decision_owner": "rules",
    }


def _source_index(package_glm_blocks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if "glm_blocks" in package_glm_blocks or "ppocr_blocks" in package_glm_blocks:
        glm_blocks = _as_dict(package_glm_blocks.get("glm_blocks"))
        ppocr_blocks = _as_dict(package_glm_blocks.get("ppocr_blocks"))
    else:
        glm_blocks = package_glm_blocks
        ppocr_blocks = {"blocks": []}
    for block in _as_list(glm_blocks.get("blocks")):
        block_dict = _as_dict(block)
        block_id = str(block_dict.get("block_id") or "")
        if block_id:
            index[block_id] = {
                "text": str(block_dict.get("cleaned_text") or block_dict.get("raw_text") or ""),
                "bbox_normalized": block_dict.get("bbox_normalized"),
                "ocr_line_ids": [str(line_id) for line_id in _as_list(block_dict.get("source_ocr_line_ids"))],
                "provider": "glm",
                "confidence": 1.0,
            }
        for line in _as_list(block_dict.get("lines")):
            line_dict = _as_dict(line)
            line_id = str(line_dict.get("line_id") or "")
            if line_id:
                index[line_id] = {
                    "text": str(line_dict.get("text") or ""),
                    "bbox_normalized": line_dict.get("bbox_normalized"),
                    "ocr_line_ids": [str(line_dict.get("ocr_line_id") or "")],
                    "provider": "glm",
                    "confidence": _confidence(line_dict),
                }
        for line_id in _as_list(block_dict.get("source_ocr_line_ids")):
            index.setdefault(
                str(line_id),
                {
                    "text": str(block_dict.get("cleaned_text") or block_dict.get("raw_text") or ""),
                    "bbox_normalized": block_dict.get("bbox_normalized"),
                    "ocr_line_ids": [str(line_id)],
                    "provider": "glm",
                    "confidence": 1.0,
                },
            )
        raw_block_id = str(block_dict.get("raw_block_id") or "")
        if raw_block_id and block_id:
            index.setdefault(raw_block_id, index[block_id])
        table = _as_dict(block_dict.get("table"))
        for row in _as_list(table.get("rows")):
            for cell in _as_list(_as_dict(row).get("cells")):
                cell_dict = _as_dict(cell)
                cell_id = str(cell_dict.get("cell_id") or "")
                if cell_id:
                    index[cell_id] = {
                        "text": str(cell_dict.get("text") or ""),
                        "bbox_normalized": block_dict.get("bbox_normalized"),
                        "ocr_line_ids": [str(line_id) for line_id in _as_list(block_dict.get("source_ocr_line_ids"))],
                        "provider": "glm",
                        "confidence": 1.0,
                    }
    for block in _as_list(ppocr_blocks.get("blocks")):
        block_dict = _as_dict(block)
        block_id = str(block_dict.get("block_id") or "")
        if block_id:
            index[block_id] = {
                "text": str(block_dict.get("cleaned_text") or block_dict.get("raw_text") or ""),
                "bbox_normalized": block_dict.get("bbox_normalized"),
                "ocr_line_ids": [str(line_id) for line_id in _as_list(block_dict.get("source_ocr_line_ids"))],
                "provider": "ppocr",
                "confidence": _confidence(block_dict),
            }
        for line in _as_list(block_dict.get("lines")):
            line_dict = _as_dict(line)
            line_id = str(line_dict.get("line_id") or "")
            if line_id:
                index[line_id] = {
                    "text": str(line_dict.get("text") or ""),
                    "bbox_normalized": line_dict.get("bbox_normalized"),
                    "ocr_line_ids": [str(line_dict.get("ocr_line_id") or "")],
                    "provider": "ppocr",
                    "confidence": _confidence(line_dict),
                }
    return index


def _normalize_structured_field(index: int, item: dict[str, Any], source_index: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    source_ids = [str(source_id) for source_id in _as_list(item.get("source_ids")) if str(source_id).strip()]
    errors = _source_errors(source_ids, source_index)
    text = clean_glm_text(str(item.get("text") or ""))
    if not text:
        errors.append({"type": "empty_text", "source_ids": source_ids})
    review_required = bool(item.get("review_required")) or bool(errors)
    if not source_ids or any(error["type"] in {"empty_text", "unknown_source_id"} for error in errors):
        return None, errors
    if text and not _text_supported_by_sources(text, source_ids, source_index):
        errors.append({"type": "text_not_supported_by_sources", "text": text, "source_ids": source_ids})
        return None, errors
    field = {
        "structured_item_id": stable_id("pkg_structured_field", index),
        "semantic_key": str(item.get("semantic_key") or ""),
        "label": str(item.get("label") or item.get("semantic_key") or ""),
        "text": text,
        "source_ids": source_ids,
        "source_ocr_line_ids": _source_ocr_line_ids(source_ids, source_index),
        "bbox_normalized": _union_source_bbox(source_ids, source_index),
        "group_id": item.get("group_id"),
        "table_id": item.get("table_id"),
        "row_key": item.get("row_key"),
        "confidence": _confidence(item),
        "review_required": review_required,
        "metadata": {
            "source": "llm_fusion_ocr_structure",
            "source_provider": _source_provider(source_ids, source_index),
            "source_confidence": _source_confidence(source_ids, source_index),
        },
    }
    return field, errors


def _normalize_nutrition_table(index: int, table: dict[str, Any], source_index: dict[str, dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = [str(source_id) for source_id in _as_list(table.get("source_ids")) if str(source_id).strip()]
    errors.extend(_source_errors(source_ids, source_index))
    return {
        "structured_table_id": stable_id("pkg_structured_table", index),
        "table_id": table.get("table_id"),
        "title": clean_glm_text(str(table.get("title") or "")),
        "source_ids": source_ids,
        "source_ocr_line_ids": _source_ocr_line_ids(source_ids, source_index),
        "bbox_normalized": _union_source_bbox(source_ids, source_index),
        "columns": [clean_glm_text(str(column)) for column in _as_list(table.get("columns"))],
        "rows": [
            {
                "row_key": clean_glm_text(str(_as_dict(row).get("row_key") or "")),
                "cells": [clean_glm_text(str(cell)) for cell in _as_list(_as_dict(row).get("cells"))],
                "source_ids": [str(source_id) for source_id in _as_list(_as_dict(row).get("source_ids"))],
                "source_ocr_line_ids": _source_ocr_line_ids([str(source_id) for source_id in _as_list(_as_dict(row).get("source_ids"))], source_index),
                "bbox_normalized": _union_source_bbox([str(source_id) for source_id in _as_list(_as_dict(row).get("source_ids"))], source_index),
                "confidence": _confidence(_as_dict(row)),
                "review_required": bool(_as_dict(row).get("review_required")),
            }
            for row in _as_list(table.get("rows"))
        ],
        "footnotes": [
            {
                "text": clean_glm_text(str(_as_dict(footnote).get("text") or "")),
                "source_ids": [str(source_id) for source_id in _as_list(_as_dict(footnote).get("source_ids"))],
                "source_ocr_line_ids": _source_ocr_line_ids([str(source_id) for source_id in _as_list(_as_dict(footnote).get("source_ids"))], source_index),
                "bbox_normalized": _union_source_bbox([str(source_id) for source_id in _as_list(_as_dict(footnote).get("source_ids"))], source_index),
                "confidence": _confidence(_as_dict(footnote)),
                "review_required": bool(_as_dict(footnote).get("review_required")),
            }
            for footnote in _as_list(table.get("footnotes"))
        ],
        "confidence": _confidence(table),
        "review_required": bool(table.get("review_required")),
        "metadata": {"source": "llm_fusion_ocr_structure", "source_provider": "glm"},
    }


def _fields_from_content_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    mappings = [
        ("name", "content_item.name", "内容物名称"),
        ("net_content", "content_item.net_content", "净含量/数量"),
        ("product_classification", "content_item.product_classification", "产品分类"),
        ("ingredients", "content_item.ingredients", "配料原文"),
    ]
    source_map = _as_dict(item.get("field_source_ids"))
    fields = []
    for key, semantic_key, label in mappings:
        text = item.get(key)
        if not text:
            continue
        fields.append(
            {
                "semantic_key": semantic_key,
                "label": label,
                "text": text,
                "source_ids": source_map.get(key) or item.get("source_ids") or [],
                "group_id": item.get("group_id"),
                "confidence": item.get("confidence", 0.0),
                "review_required": item.get("review_required", True),
            }
        )
    return fields


def _source_errors(source_ids: list[str], source_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not source_ids:
        return [{"type": "missing_source_ids"}]
    return [{"type": "unknown_source_id", "source_id": source_id} for source_id in source_ids if source_id not in source_index]


def _text_supported_by_sources(text: str, source_ids: list[str], source_index: dict[str, dict[str, Any]]) -> bool:
    expected = _evidence_norm(text)
    if not expected:
        return False
    source_text = "\n".join(str(source_index.get(source_id, {}).get("text") or "") for source_id in source_ids)
    return expected in _evidence_norm(source_text)


def _evidence_norm(text: str) -> str:
    value = clean_glm_text(text).lower()
    value = value.replace("×", "x")
    return re.sub(r"\s+", "", value)


def _union_source_bbox(source_ids: list[str], source_index: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    boxes = [_as_dict(source_index.get(source_id, {}).get("bbox_normalized")) for source_id in source_ids]
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    return {
        "x1": min(float(box.get("x1", 0)) for box in boxes),
        "y1": min(float(box.get("y1", 0)) for box in boxes),
        "x2": max(float(box.get("x2", 0)) for box in boxes),
        "y2": max(float(box.get("y2", 0)) for box in boxes),
    }


def _source_ocr_line_ids(source_ids: list[str], source_index: dict[str, dict[str, Any]]) -> list[str]:
    seen = set()
    line_ids = []
    for source_id in source_ids:
        for line_id in _as_list(source_index.get(source_id, {}).get("ocr_line_ids")):
            line_id_text = str(line_id)
            if line_id_text and line_id_text not in seen:
                seen.add(line_id_text)
                line_ids.append(line_id_text)
    return line_ids


def _source_provider(source_ids: list[str], source_index: dict[str, dict[str, Any]]) -> str:
    providers = {
        str(source_index.get(source_id, {}).get("provider") or "")
        for source_id in source_ids
        if source_index.get(source_id, {}).get("provider")
    }
    if not providers:
        return "unknown"
    if providers == {"glm"}:
        return "glm"
    if providers == {"ppocr"}:
        return "ppocr"
    return "fusion"


def _source_confidence(source_ids: list[str], source_index: dict[str, dict[str, Any]]) -> float:
    values = []
    for source_id in source_ids:
        try:
            values.append(float(source_index.get(source_id, {}).get("confidence", 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return sum(values) / len(values) if values else 0.0


def _confidence(item: dict[str, Any]) -> float:
    try:
        value = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _compact_standard_items(value: Any) -> list[dict[str, Any]]:
    items = []
    for item in _as_list(value):
        item_dict = _as_dict(item)
        items.append(
            {
                "semantic_key": item_dict.get("semantic_key"),
                "label": item_dict.get("label"),
                "group_id": item_dict.get("group_id"),
                "table_id": item_dict.get("table_id"),
                "row_key": item_dict.get("row_key"),
                "text": item_dict.get("text"),
            }
        )
    return items


def _compact_groups(value: Any) -> list[dict[str, Any]]:
    return [
        {
            "group_id": _as_dict(item).get("group_id"),
            "group_type": _as_dict(item).get("group_type"),
            "label": _as_dict(item).get("label"),
        }
        for item in _as_list(value)
    ]


def _compact_tables(value: Any) -> list[dict[str, Any]]:
    tables = []
    for table in _as_list(value):
        table_dict = _as_dict(table)
        tables.append(
            {
                "table_id": table_dict.get("table_id"),
                "title": table_dict.get("title"),
                "columns": table_dict.get("columns"),
                "row_keys": [_as_dict(row).get("row_key") for row in _as_list(table_dict.get("rows"))],
            }
        )
    return tables


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
