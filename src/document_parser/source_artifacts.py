from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from .models import CompiledField, Evidence, OcrLine, TextSpan, to_jsonable


FIELD_ANCHOR_RE = re.compile(
    r"(品名|产品名称|配料表?|净含量|规格|产品标准|执行标准|许可证编号|生产者|生产商|生产厂商|地址|产地|"
    r"保质期|贮存条件|储存条件|商品条码|外箱条码|营养成分表|内容物\s*\d+)"
)
CJK_INTERNAL_SPACING_RE = re.compile(r"[\u3400-\u9fff]\s+[\u3400-\u9fff]")


def build_source_layers(
    perception: Any,
    ocr_lines: list[OcrLine],
    spans: list[TextSpan],
    table_layers: dict[str, Any],
    ocr_error: str | None = None,
) -> dict[str, Any]:
    page_summaries = []
    issues: list[dict[str, Any]] = []
    pdf_text_available = bool(getattr(perception, "text_layer_available", False))
    total_chars = sum(len(span.text) for span in spans)
    control_char_count = sum(_control_char_count(span.text) for span in spans)
    cjk_spacing_count = sum(1 for span in spans if CJK_INTERNAL_SPACING_RE.search(span.text))
    multi_anchor_line_count = sum(1 for span in spans if len(FIELD_ANCHOR_RE.findall(span.text)) >= 3)
    bbox_available_count = sum(1 for span in spans if span.bbox_pdf is not None)

    for page in perception.pages:
        page_spans = [span for span in spans if span.page == page.page]
        page_ocr_lines = [line for line in ocr_lines if line.page == page.page]
        page_summaries.append(
            {
                "page": page.page,
                "width": page.width,
                "height": page.height,
                "text_span_count": len(page_spans),
                "ocr_line_count": len(page_ocr_lines),
                "bbox_available_count": sum(1 for span in page_spans if span.bbox_pdf is not None),
            }
        )

    if not spans:
        issues.append(_issue("source_text_missing", "high", "No PDF text or OCR spans are available.", None))
    if spans and not pdf_text_available and ocr_lines:
        issues.append(
            _issue(
                "pdf_text_missing_ocr_used",
                "info",
                "PDF text layer is unavailable; OCR spans are used as source text.",
                {"ocr_line_count": len(ocr_lines)},
            )
        )
    if spans and bbox_available_count < len(spans):
        issues.append(
            _issue(
                "source_bbox_missing",
                "medium",
                "Some source spans are missing bbox and cannot be highlighted reliably.",
                {"missing_bbox_count": len(spans) - bbox_available_count},
            )
        )
    if cjk_spacing_count:
        issues.append(
            _issue(
                "source_cjk_spacing_noise",
                "medium",
                "CJK text contains internal spacing that may indicate degraded source extraction.",
                {"affected_span_count": cjk_spacing_count},
            )
        )
    if multi_anchor_line_count:
        issues.append(
            _issue(
                "source_multi_anchor_line",
                "low",
                "Some source lines contain multiple field anchors and require boundary validation.",
                {"affected_span_count": multi_anchor_line_count},
            )
        )
    for warning in getattr(perception, "warnings", []):
        issues.append(_issue("source_extraction_warning", "medium", warning, None))

    return {
        "status": "review_required" if any(issue["severity"] in {"high", "medium"} for issue in issues) else "pass",
        "source_mode": _source_mode(pdf_text_available, ocr_lines, spans),
        "pages": page_summaries,
        "layers": {
            "pdf_text": {
                "available": pdf_text_available,
                "span_count": sum(1 for span in spans if span.source == "pdf_text"),
                "bbox_available_count": sum(1 for span in spans if span.source == "pdf_text" and span.bbox_pdf is not None),
                "warnings": list(getattr(perception, "warnings", [])),
            },
            "ocr": {
                "provider": "ppocrv6",
                "status": "failed" if ocr_error else "pass",
                "error": ocr_error,
                "fallback_used": bool(ocr_error and pdf_text_available),
                "line_count": len(ocr_lines),
                "block_count": len({line.block_id or stable_block_id(line.page) for line in ocr_lines}),
                "token_count": sum(len(line.tokens) for line in ocr_lines),
                "bbox_available_count": sum(1 for line in ocr_lines if line.bbox_pdf is not None),
                "confidence_min": min((line.confidence for line in ocr_lines), default=None),
                "confidence_avg": round(sum(line.confidence for line in ocr_lines) / len(ocr_lines), 4) if ocr_lines else None,
                "blocks": _ocr_blocks(ocr_lines),
                "lines": [_ocr_line_artifact(line) for line in ocr_lines],
            },
            "tables": {
                "parsers": table_layers.get("parsers", []),
                "table_count": len(table_layers.get("tables", [])),
                "parser_issue_count": len(table_layers.get("parser_issues", [])),
            },
        },
        "text_quality": {
            "total_text_span_count": len(spans),
            "total_char_count": total_chars,
            "bbox_coverage_rate": round(bbox_available_count / len(spans), 4) if spans else 0.0,
            "control_char_count": control_char_count,
            "cjk_internal_spacing_span_count": cjk_spacing_count,
            "multi_anchor_line_count": multi_anchor_line_count,
        },
        "spans": [
            {
                "span_id": span.span_id,
                "page": span.page,
                "source": span.source,
                "text": span.text,
                "char_count": len(span.text),
                "bbox_status": "available" if span.bbox_pdf else "missing",
                "bbox_pdf": to_jsonable(span.bbox_pdf),
                "bbox_normalized": to_jsonable(span.bbox_normalized),
                "confidence": span.confidence,
            }
            for span in spans
        ],
        "source_issues": issues,
        "source_issue_count": len(issues),
    }


def _ocr_blocks(ocr_lines: list[OcrLine]) -> list[dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for line in ocr_lines:
        block_id = line.block_id or stable_block_id(line.page)
        block = blocks.setdefault(
            block_id,
            {
                "block_id": block_id,
                "page": line.page,
                "line_ids": [],
                "line_count": 0,
                "token_count": 0,
            },
        )
        block["line_ids"].append(line.ocr_line_id)
        block["line_count"] += 1
        block["token_count"] += len(line.tokens)
    return list(blocks.values())


def _ocr_line_artifact(line: OcrLine) -> dict[str, Any]:
    return {
        "ocr_line_id": line.ocr_line_id,
        "block_id": line.block_id or stable_block_id(line.page),
        "page": line.page,
        "text": line.text,
        "confidence": line.confidence,
        "bbox_status": "available" if line.bbox_pdf else "missing",
        "bbox_pdf": to_jsonable(line.bbox_pdf),
        "bbox_normalized": to_jsonable(line.bbox_normalized),
        "tokens": to_jsonable(line.tokens),
        "metadata": to_jsonable(line.metadata),
    }


def stable_block_id(page: int) -> str:
    return f"ocr_p{page}_block_0001"


def build_coverage_map(
    spans: list[TextSpan],
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    tables: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    anchor_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    coverage_by_span: dict[str, dict[str, Any]] = {
        span.span_id: {
            "span_id": span.span_id,
            "page": span.page,
            "source": span.source,
            "text": span.text,
            "mappings": [],
        }
        for span in spans
    }

    for field in compiled_fields.values():
        for evidence_ref in field.evidence_refs:
            source_evidence = evidence_by_id.get(evidence_ref)
            if not source_evidence:
                continue
            for span_id in source_evidence.source_node_ids:
                _add_mapping(
                    coverage_by_span,
                    span_id,
                    {
                        "target_type": "field",
                        "target_id": field.field_id,
                        "semantic_key": field.semantic_key,
                        "evidence_ref": evidence_ref,
                    },
                )

    for table in tables:
        for span_id in table.get("source_span_ids", []):
            _add_mapping(
                coverage_by_span,
                span_id,
                {
                    "target_type": "table",
                    "target_id": table.get("table_id"),
                    "table_type": table.get("table_type"),
                    "table_layer_id": table.get("table_layer_id"),
                },
            )

    for requirement in requirements:
        for evidence_ref in requirement.get("evidence_refs", []):
            source_evidence = evidence_by_id.get(evidence_ref)
            if not source_evidence:
                continue
            for span_id in source_evidence.source_node_ids:
                _add_mapping(
                    coverage_by_span,
                    span_id,
                    {
                        "target_type": "requirement",
                        "target_id": requirement.get("requirement_id"),
                        "evidence_ref": evidence_ref,
                    },
                )

    for region in regions:
        for span_id in region.get("source_span_ids", []):
            _add_mapping(
                coverage_by_span,
                span_id,
                {
                    "target_type": "region",
                    "target_id": region.get("region_id"),
                    "region_type": region.get("region_type"),
                },
            )

    anchors = []
    for anchor in anchor_inventory:
        mappings = coverage_by_span.get(anchor["span_id"], {}).get("mappings", [])
        anchors.append(
            {
                **anchor,
                "coverage_status": "covered" if mappings else "missing",
                "mappings": mappings,
            }
        )

    duplicate_coverage_issues = _duplicate_coverage_issues(coverage_by_span)
    covered_span_count = sum(1 for item in coverage_by_span.values() if item["mappings"])
    return {
        "span_count": len(spans),
        "covered_span_count": covered_span_count,
        "unassigned_span_count": len(spans) - covered_span_count,
        "span_coverage_rate": round(covered_span_count / len(spans), 4) if spans else 0.0,
        "unassigned_span_ids": [item["span_id"] for item in coverage_by_span.values() if not item["mappings"]],
        "span_coverage": list(coverage_by_span.values()),
        "anchors": anchors,
        "anchor_count": len(anchors),
        "covered_anchor_count": sum(1 for anchor in anchors if anchor["coverage_status"] == "covered"),
        "missing_anchor_count": sum(1 for anchor in anchors if anchor["coverage_status"] == "missing"),
        "duplicate_coverage_issues": duplicate_coverage_issues,
        "duplicate_coverage_issue_count": len(duplicate_coverage_issues),
    }


def _issue(issue_type: str, severity: str, message: str, detail: dict[str, Any] | None) -> dict[str, Any]:
    digest = hashlib.sha256(f"{issue_type}:{message}".encode("utf-8")).hexdigest()[:8]
    return {
        "issue_id": f"source_issue_{digest}",
        "issue_type": issue_type,
        "severity": severity,
        "message": message,
        "detail": detail or {},
    }


def _control_char_count(text: str) -> int:
    return sum(1 for char in text if unicodedata.category(char).startswith("C") and char not in "\n\r\t")


def _source_mode(pdf_text_available: bool, ocr_lines: list[OcrLine], spans: list[TextSpan]) -> str:
    if not spans:
        return "empty"
    if pdf_text_available and ocr_lines:
        return "pdf_text_plus_ocr"
    if pdf_text_available:
        return "pdf_text_only"
    if ocr_lines:
        return "ocr_only"
    return "unknown_text_source"


def _add_mapping(coverage_by_span: dict[str, dict[str, Any]], span_id: str, mapping: dict[str, Any]) -> None:
    if span_id not in coverage_by_span:
        return
    if mapping not in coverage_by_span[span_id]["mappings"]:
        coverage_by_span[span_id]["mappings"].append(mapping)


def _duplicate_coverage_issues(coverage_by_span: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []
    for item in coverage_by_span.values():
        seen: set[tuple[Any, Any, Any]] = set()
        duplicates = []
        for mapping in item["mappings"]:
            key = (mapping.get("target_type"), mapping.get("target_id"), mapping.get("semantic_key"))
            if key in seen:
                duplicates.append(mapping)
            seen.add(key)
        if duplicates:
            issues.append(
                {
                    "expected": "each source span maps once to the same target",
                    "actual": duplicates,
                    "source": {"span_id": item["span_id"], "text": item["text"]},
                    "repair_hint": "Deduplicate repeated coverage references while preserving distinct fields/tables.",
                }
            )
    return issues
