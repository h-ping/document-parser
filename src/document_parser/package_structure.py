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
    package_llm_structure_input: dict[str, Any]
    package_llm_structure_output: dict[str, Any]
    package_structured_items: dict[str, Any]
    package_structure_quality_report: dict[str, Any]
    runtime: dict[str, Any]


def run_package_structure_stage(
    *,
    artifacts: dict[str, Any],
    ocr_lines: list[OcrLine],
    llm_mode: LlmMode,
    llm_client: LlmClient | None,
) -> StructureRun:
    package_glm_blocks = build_package_glm_blocks(ocr_lines)
    structure_input = build_package_llm_structure_input(artifacts, package_glm_blocks)
    if llm_mode == "disabled":
        return _disabled_run(package_glm_blocks, structure_input, "llm_mode_disabled")
    if not package_glm_blocks["blocks"]:
        if llm_mode == "required":
            raise PackageStructureError("LLM package structure requires GLM-OCR blocks, but none were found.")
        return _disabled_run(package_glm_blocks, structure_input, "no_glm_ocr_blocks")
    if llm_client is None:
        if llm_mode == "required":
            raise PackageStructureError("LLM package structure requires LLM_API_KEY, LLM_BASE_URL and LLM_MODEL.")
        return _disabled_run(package_glm_blocks, structure_input, "llm_env_not_configured")

    try:
        llm_output = llm_client.structured_json_with_max_tokens(
            _llm_system_prompt(),
            json.dumps(structure_input, ensure_ascii=False),
            PACKAGE_STRUCTURE_SCHEMA,
            PACKAGE_STRUCTURE_LLM_MAX_TOKENS,
        )
    except LlmError as exc:
        if llm_mode == "required":
            raise PackageStructureError(f"LLM package structure failed: {exc}") from exc
        return _disabled_run(package_glm_blocks, structure_input, f"llm_error:{exc.__class__.__name__}")

    package_structured_items, quality_report = normalize_package_structure_output(llm_output, package_glm_blocks)
    return StructureRun(
        package_glm_blocks=package_glm_blocks,
        package_llm_structure_input=structure_input,
        package_llm_structure_output={"enabled": True, "mode": "llm_glm_ocr_structure", "body": llm_output},
        package_structured_items=package_structured_items,
        package_structure_quality_report=quality_report,
        runtime={
            "enabled": True,
            "mode": "llm_glm_ocr_structure",
            "disabled_reason": None,
            "final_decision_owner": "rules",
        },
    )


def write_package_structure_artifacts(output_dir: Path, run: StructureRun) -> None:
    write_json(output_dir / "package_glm_blocks.json", run.package_glm_blocks)
    write_json(output_dir / "package_llm_structure_input.json", run.package_llm_structure_input)
    write_json(output_dir / "package_llm_structure_output.json", run.package_llm_structure_output)
    write_json(output_dir / "package_structured_items.json", run.package_structured_items)
    write_json(output_dir / "package_structure_quality_report.json", run.package_structure_quality_report)


def build_package_glm_blocks(ocr_lines: list[OcrLine]) -> dict[str, Any]:
    grouped: dict[str, list[OcrLine]] = {}
    for line in ocr_lines:
        if line.metadata.get("provider") != "glm_ocr":
            continue
        block_id = str(line.block_id or line.metadata.get("block_id") or stable_id("glm_block", len(grouped) + 1))
        grouped.setdefault(block_id, []).append(line)

    blocks = []
    for order, (block_id, lines) in enumerate(grouped.items(), start=1):
        label = str(lines[0].metadata.get("detail_label") or "")
        raw_lines = [line.text for line in lines if line.text.strip()]
        cleaned_lines = [clean_glm_text(line) for line in raw_lines]
        cleaned_lines = [line for line in cleaned_lines if line]
        table_rows = _html_table_rows("\n".join(raw_lines))
        block = {
            "block_id": block_id,
            "order": order,
            "label": label,
            "detail_index": lines[0].metadata.get("detail_index"),
            "source_ocr_line_ids": [line.ocr_line_id for line in lines],
            "raw_text": "\n".join(raw_lines),
            "cleaned_text": "\n".join(cleaned_lines),
            "bbox_normalized": to_jsonable(lines[0].bbox_normalized),
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


def build_package_llm_structure_input(artifacts: dict[str, Any], package_glm_blocks: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_version": "package_llm_structure_input_v0.1",
        "task": "structure_package_text_from_glm_ocr",
        "rules": [
            "Only use text present in the provided GLM-OCR blocks, lines, or table cells.",
            "Return package-side structured fields only; do not compare with the standard and do not decide pass/fail.",
            "Keep printed field prefixes such as 产品名称：, 配料表：, 地址：, 内容物2： when they appear in the source.",
            "Do not include Markdown markers or HTML tags as printed text.",
            "The JSON root must contain exactly these keys: fields, field_groups, content_items, nutrition_tables, other_text, warnings.",
            "Do not return wrapper keys such as structured_items, result, data, or markdown.",
        ],
        "output_contract": _package_structure_output_contract(),
        "standard_context": {
            "standard_items": _compact_standard_items(artifacts.get("standard_items")),
            "field_groups": _compact_groups(artifacts.get("field_groups")),
            "tables": _compact_tables(artifacts.get("tables")),
        },
        "glm_blocks": package_glm_blocks.get("blocks", []),
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
        normalized, item_errors = _normalize_structured_field(index, _as_dict(item), source_index)
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
        "source": "llm_glm_ocr_structure",
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


def _disabled_run(package_glm_blocks: dict[str, Any], structure_input: dict[str, Any], reason: str) -> StructureRun:
    return StructureRun(
        package_glm_blocks=package_glm_blocks,
        package_llm_structure_input=structure_input,
        package_llm_structure_output={"enabled": False, "mode": "disabled", "disabled_reason": reason, "body": {}},
        package_structured_items={
            "artifact_version": "package_structured_items_v0.1",
            "enabled": False,
            "source": "disabled",
            "fields": [],
            "field_groups": [],
            "content_items": [],
            "nutrition_tables": [],
            "other_text": [],
            "rejected_items": [],
        },
        package_structure_quality_report={
            "artifact_version": "package_structure_quality_report_v0.1",
            "status": "disabled",
            "enabled": False,
            "disabled_reason": reason,
            "error_count": 0,
            "errors": [],
            "field_count": 0,
            "nutrition_table_count": 0,
        },
        runtime={
            "enabled": False,
            "mode": "disabled",
            "disabled_reason": reason,
            "final_decision_owner": "rules",
        },
    )


def _llm_system_prompt() -> str:
    return (
        "You structure package artwork text from GLM-OCR output. "
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
                    {"cell_id": f"{block_id}:r{row_index}c{cell_index}", "text": cell}
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


def _source_index(package_glm_blocks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for block in _as_list(package_glm_blocks.get("blocks")):
        block_dict = _as_dict(block)
        block_id = str(block_dict.get("block_id") or "")
        if block_id:
            index[block_id] = {
                "text": str(block_dict.get("cleaned_text") or block_dict.get("raw_text") or ""),
                "bbox_normalized": block_dict.get("bbox_normalized"),
                "ocr_line_ids": [str(line_id) for line_id in _as_list(block_dict.get("source_ocr_line_ids"))],
            }
        for line_id in _as_list(block_dict.get("source_ocr_line_ids")):
            index[str(line_id)] = {
                "text": str(block_dict.get("cleaned_text") or block_dict.get("raw_text") or ""),
                "bbox_normalized": block_dict.get("bbox_normalized"),
                "ocr_line_ids": [str(line_id)],
            }
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
                    }
    return index


def _normalize_structured_field(index: int, item: dict[str, Any], source_index: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    source_ids = [str(source_id) for source_id in _as_list(item.get("source_ids")) if str(source_id).strip()]
    errors = _source_errors(source_ids, source_index)
    text = clean_glm_text(str(item.get("text") or ""))
    if not text:
        errors.append({"type": "empty_text", "source_ids": source_ids})
    review_required = bool(item.get("review_required")) or bool(errors)
    if not source_ids or any(error["type"] == "unknown_source_id" for error in errors):
        return None, errors
    if text and not _text_supported_by_sources(text, source_ids, source_index):
        errors.append({"type": "text_not_supported_by_sources", "text": text, "source_ids": source_ids})
        review_required = True
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
        "metadata": {"source": "llm_glm_ocr_structure"},
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
        "metadata": {"source": "llm_glm_ocr_structure"},
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
