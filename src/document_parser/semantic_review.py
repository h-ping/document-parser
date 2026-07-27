from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from typing import Any

from .models import CompiledField, Evidence, ExtractionPlan, TextSpan, to_jsonable


_TRUNCATED_ENDINGS = ("：", ":", "（", "(", "、", "/")
_BALANCED_PAIRS = (("（", "）"), ("(", ")"), ("【", "】"), ("[", "]"))


def build_semantic_review_input(
    blocks: dict[str, Any],
    plan: ExtractionPlan,
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    spans: list[TextSpan],
) -> dict[str, Any]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    span_by_id = {span.span_id: span for span in spans}
    fields_by_block: dict[str, list[dict[str, Any]]] = {}
    for index, field_plan in enumerate(plan.fields, start=1):
        block_id = str(field_plan.boundary.get("agent_block_id") or "")
        if not block_id:
            continue
        compiled = compiled_fields.get(f"fld_{index:04d}")
        if compiled is None:
            continue
        fields_by_block.setdefault(block_id, []).append(
            {
                "field_id": compiled.field_id,
                "semantic_key": compiled.semantic_key,
                "entity_id": compiled.entity_id,
                "raw_value": compiled.raw_value,
                "evidence_refs": compiled.evidence_refs,
                "source_span_ids": [
                    source_id
                    for ref in compiled.evidence_refs
                    if ref in evidence_by_id
                    for source_id in evidence_by_id[ref].source_node_ids
                ],
            }
        )
    review_blocks = []
    for block in blocks.get("blocks", []):
        source_ids = [str(value) for value in block.get("context_span_ids", []) if str(value) in span_by_id]
        review_blocks.append(
            {
                "block_id": block.get("block_id"),
                "block_type": block.get("block_type"),
                "source_spans": [
                    {"span_id": span_id, "page": span_by_id[span_id].page, "text": span_by_id[span_id].text}
                    for span_id in source_ids
                ],
                "compiled_fields": fields_by_block.get(str(block.get("block_id")), []),
                "planned_tables": [
                    to_jsonable(table)
                    for table in plan.tables
                    if isinstance(table, dict) and str(table.get("agent_block_id") or "") == str(block.get("block_id"))
                ],
            }
        )
    return {
        "artifact_version": "semantic_review_input_v0.1",
        "policy": "review_only_no_final_values",
        "blocks": review_blocks,
    }


def deterministic_semantic_findings(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for block in review_input.get("blocks", []):
        block_id = str(block.get("block_id") or "")
        source_spans = [item for item in block.get("source_spans", []) if isinstance(item, dict)]
        fields = [item for item in block.get("compiled_fields", []) if isinstance(item, dict)]
        planned_tables = [item for item in block.get("planned_tables", []) if isinstance(item, dict)]
        consumed = {str(span_id) for field in fields for span_id in field.get("source_span_ids", [])}
        for field in fields:
            value = str(field.get("raw_value") or "").strip()
            semantic_key = str(field.get("semantic_key") or "")
            issue_type = _field_issue_type(semantic_key, value)
            if issue_type:
                findings.append(
                    _finding(
                        issue_type=issue_type,
                        block_id=block_id,
                        target_id=str(field.get("field_id") or "document"),
                        source_span_ids=[str(value) for value in field.get("source_span_ids", [])],
                        message=f"Field {semantic_key} appears truncated or structurally incomplete.",
                    )
                )
        for span in source_spans:
            text = str(span.get("text") or "")
            span_id = str(span.get("span_id") or "")
            if span_id in consumed or not _important_anchor(text) or _anchor_covered(text, fields):
                continue
            findings.append(
                _finding(
                    issue_type="important_anchor_unconsumed",
                    block_id=block_id,
                    target_id="document",
                    source_span_ids=[span_id],
                    message="An important label anchor remains unconsumed in this block.",
                )
            )
        if block.get("block_type") == "nutrition_table":
            if not planned_tables:
                findings.append(
                    _finding(
                        issue_type="nutrition_table_missing",
                        block_id=block_id,
                        target_id="document",
                        source_span_ids=[str(item.get("span_id")) for item in source_spans if item.get("span_id")],
                        message="A nutrition layout block has no evidence-bound table plan.",
                    )
                )
            for table in planned_tables:
                findings.extend(_nutrition_table_findings(table, block_id))
    return _dedupe_findings(findings)


def validate_agent_review_findings(
    body: dict[str, Any] | None,
    review_input: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_blocks = {str(block.get("block_id")): block for block in review_input.get("blocks", [])}
    valid_spans = {
        str(span.get("span_id"))
        for block in valid_blocks.values()
        for span in block.get("source_spans", [])
        if isinstance(span, dict)
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate((body or {}).get("findings", []), start=1):
        if not isinstance(item, dict):
            rejected.append({"item_index": index, "reason": "finding_not_object"})
            continue
        block_id = str(item.get("block_id") or "")
        source_ids = [str(value) for value in item.get("source_span_ids", [])]
        if block_id not in valid_blocks:
            rejected.append({"item_index": index, "reason": "unknown_block_id", "block_id": block_id})
            continue
        missing = [span_id for span_id in source_ids if span_id not in valid_spans]
        if missing:
            rejected.append({"item_index": index, "reason": "unknown_source_span_id", "source_span_ids": missing})
            continue
        accepted.append(
            _finding(
                issue_type=str(item.get("issue_type") or "semantic_review_issue"),
                block_id=block_id,
                target_id=str(item.get("target_id") or "document"),
                source_span_ids=source_ids,
                message=str(item.get("message") or "Semantic review found an unresolved issue."),
                severity=str(item.get("severity") or "high"),
                repair_required=bool(item.get("repair_required", True)),
            )
        )
    return _dedupe_findings(accepted), rejected


def call_semantic_review_agent(agent: Any, review_input: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    findings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    blocks = [block for block in review_input.get("blocks", []) if isinstance(block, dict)]
    batches: list[list[dict[str, Any]]] = []
    for block in blocks:
        block_size = sum(len(str(span.get("text") or "")) for span in block.get("source_spans", []) if isinstance(span, dict))
        current_size = sum(
            len(str(span.get("text") or ""))
            for current in (batches[-1] if batches else [])
            for span in current.get("source_spans", [])
            if isinstance(span, dict)
        )
        if not batches or len(batches[-1]) >= 8 or current_size + block_size > 12_000:
            batches.append([])
        batches[-1].append(block)
    for batch in batches:
        local_input = {**review_input, "blocks": batch}
        block_ids = [str(block.get("block_id") or "") for block in batch]
        try:
            body = agent.review_compiled_blocks(local_input)
        except Exception as exc:
            raise RuntimeError(f"Semantic review Agent call failed for blocks={block_ids}: {exc}") from exc
        local_findings, local_rejected = validate_agent_review_findings(body, local_input)
        findings.extend(local_findings)
        rejected.extend(local_rejected)
    return _dedupe_findings(findings), rejected, len(batches)


def merge_semantic_findings(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_findings([item for group in groups for item in group])


def replace_block_fields(
    plan: ExtractionPlan,
    repair_plan: ExtractionPlan,
    block_id: str,
    spans: list[TextSpan] | None = None,
    allowed_new_source_ids: set[str] | None = None,
) -> ExtractionPlan:
    span_by_id = {span.span_id: span for span in spans or []}
    fields = list(plan.fields)
    for proposed in repair_plan.fields:
        if proposed.semantic_key.startswith("nutrition_table."):
            continue
        repaired = replace(proposed, boundary={**proposed.boundary, "agent_block_id": block_id, "semantic_repair": True})
        matching_indexes = [
            index
            for index, existing in enumerate(fields)
            if str(existing.boundary.get("agent_block_id") or "") == block_id
            and existing.semantic_key == repaired.semantic_key
            and (not repaired.entity_id or not existing.entity_id or existing.entity_id == repaired.entity_id)
        ]
        if matching_indexes:
            best_index = max(matching_indexes, key=lambda index: _field_range_length(fields[index]))
            if _repair_field_is_better(fields[best_index], repaired, span_by_id):
                fields[best_index] = repaired
            continue
        repair_source_ids = {item.span_id for item in repaired.value_source.ranges}
        if allowed_new_source_ids is not None and not repair_source_ids.intersection(allowed_new_source_ids):
            continue
        if _field_uses_mixed_source_layers(repaired, span_by_id):
            continue
        fields.append(repaired)
    return replace(plan, fields=fields)


def replace_block_tables(plan: ExtractionPlan, repair_plan: ExtractionPlan, block_id: str) -> ExtractionPlan:
    retained = [
        table
        for table in plan.tables
        if not isinstance(table, dict) or str(table.get("agent_block_id") or "") != block_id
    ]
    repaired = [
        {**table, "agent_block_id": block_id, "semantic_repair": True}
        for table in repair_plan.tables
        if isinstance(table, dict)
    ]
    return replace(plan, tables=[*retained, *repaired])


def semantic_state_hash(findings: list[dict[str, Any]], compiled_fields: dict[str, CompiledField]) -> str:
    payload = {
        "issues": sorted(
            (item.get("issue_type"), item.get("block_id"), tuple(item.get("source_span_ids", [])))
            for item in findings
        ),
        "values": sorted((field.semantic_key, field.entity_id or "", field.value_hash) for field in compiled_fields.values()),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_semantic_review_report(
    review_input: dict[str, Any],
    findings: list[dict[str, Any]],
    rejected_findings: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    return {
        "artifact_version": "semantic_review_report_v0.1",
        "status": "disabled" if not enabled else "pass" if not findings else "review_required",
        "review_agent_independent": enabled,
        "review_input_block_count": len(review_input.get("blocks", [])),
        "finding_count": len(findings),
        "high_risk_finding_count": sum(1 for item in findings if item.get("severity") == "high"),
        "findings": findings,
        "rejected_agent_findings": rejected_findings,
        "repair_round_count": len(rounds),
        "repair_rounds": rounds,
    }


def semantic_review_validation_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("status") == "disabled":
        return []
    return [
        {
            "validation_id": "val_semantic_review_001",
            "target_id": "document",
            "check_type": "semantic_review",
            "result": "passed" if not report.get("findings") else "failed",
            "severity": "high" if report.get("high_risk_finding_count") else "medium",
            "message": "Independent semantic review passed." if not report.get("findings") else "Independent semantic review has unresolved findings.",
            "issues": report.get("findings", []),
        }
    ]


def _field_issue_type(semantic_key: str, value: str) -> str | None:
    if not value or value.endswith(_TRUNCATED_ENDINGS):
        return "field_value_truncated"
    if any(value.count(left) != value.count(right) for left, right in _BALANCED_PAIRS):
        return "unbalanced_brackets"
    digits = re.sub(r"\D", "", value)
    if "barcode" in semantic_key and len(digits) not in {8, 12, 13, 14}:
        return "barcode_incomplete"
    if "license" in semantic_key and "SC" in value.upper() and not re.search(r"SC\s*\d{14}\b", value.upper()):
        return "license_number_incomplete"
    if "standard" in semantic_key and len(value) < 5:
        return "standard_code_incomplete"
    return None


def _nutrition_table_findings(table: dict[str, Any], block_id: str) -> list[dict[str, Any]]:
    rows = [row for row in table.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return [
            _finding(
                issue_type="nutrition_rows_missing",
                block_id=block_id,
                target_id="document",
                source_span_ids=[],
                message="Nutrition table plan has no rows.",
            )
        ]
    findings = []
    data_widths = []
    for row in rows:
        cells = [cell for cell in row.get("cells", []) if isinstance(cell, dict)]
        texts = [str(cell.get("text") or cell.get("raw_value") or "").strip() for cell in cells]
        source_ids = [span_id for cell in cells for span_id in _cell_span_ids(cell)]
        if texts and re.fullmatch(r"[-—]{1,3}", texts[0]):
            findings.append(
                _finding(
                    issue_type="nutrition_marker_separate_cell",
                    block_id=block_id,
                    target_id="document",
                    source_span_ids=source_ids,
                    message="A nutrition row marker is emitted as an independent cell.",
                )
            )
        if texts and re.sub(r"^[-—]+", "", texts[0]) in {"能量", "蛋白质", "脂肪", "饱和脂肪", "反式脂肪酸", "碳水化合物", "糖", "钠"}:
            data_widths.append(len(cells))
    if data_widths and len(set(data_widths)) > 1:
        findings.append(
            _finding(
                issue_type="nutrition_column_count_inconsistent",
                block_id=block_id,
                target_id="document",
                source_span_ids=[],
                message="Nutrition data rows do not use a consistent cell count.",
            )
        )
    return findings


def _cell_span_ids(cell: dict[str, Any]) -> list[str]:
    values = cell.get("source_span_ids")
    if isinstance(values, list):
        return [str(value) for value in values]
    source_id = cell.get("span_id") or cell.get("source_span_id")
    return [str(source_id)] if source_id else []


def _repair_field_is_better(existing: Any, proposed: Any, span_by_id: dict[str, TextSpan]) -> bool:
    if _field_uses_mixed_source_layers(proposed, span_by_id):
        return False
    return _field_range_length(proposed) >= _field_range_length(existing)


def _field_range_length(field: Any) -> int:
    return sum(item.end_offset - item.start_offset for item in field.value_source.ranges)


def _field_uses_mixed_source_layers(field: Any, span_by_id: dict[str, TextSpan]) -> bool:
    if not span_by_id:
        return False
    layers = {
        "ocr" if span_by_id[item.span_id].source.startswith("ocr") else "pdf"
        for item in field.value_source.ranges
        if item.span_id in span_by_id
    }
    return len(layers) > 1


def _important_anchor(text: str) -> bool:
    return bool(re.search(r"(?:产品名称|品名|内容物|配料|净含量|保质期|贮存条件|产品标准|委托方|受托方|生产者|地址|许可证|条码)\s*[:：]", text))


def _anchor_covered(text: str, fields: list[dict[str, Any]]) -> bool:
    semantic_keys = {str(field.get("semantic_key") or "") for field in fields}
    anchors = (
        (("产品名称", "品名"), ("product.name", "content_item.name")),
        (("内容物",), ("content_item.name", "product.name")),
        (("配料",), ("product.ingredients", "content_item.ingredients")),
        (("净含量",), ("product.net_content", "content_item.net_content")),
        (("保质期",), ("product.shelf_life", "content_item.shelf_life")),
        (("贮存条件",), ("product.storage_condition",)),
        (("产品标准",), ("product.standard_code",)),
        (("委托方", "受托方", "生产者"), ("manufacturer.name",)),
        (("地址",), ("manufacturer.address",)),
        (("许可证",), ("manufacturer.license_number",)),
        (("条码",), ("barcode.commodity", "barcode.outer_box")),
    )
    for labels, expected_keys in anchors:
        if any(label in text for label in labels):
            return any(key in semantic_keys or any(value.endswith(key.rsplit(".", 1)[-1]) for value in semantic_keys) for key in expected_keys)
    return False


def _finding(
    *,
    issue_type: str,
    block_id: str,
    target_id: str,
    source_span_ids: list[str],
    message: str,
    severity: str = "high",
    repair_required: bool = True,
) -> dict[str, Any]:
    return {
        "issue_type": issue_type,
        "block_id": block_id,
        "target_type": "field" if target_id.startswith("fld_") else "document",
        "target_id": target_id,
        "source_span_ids": source_span_ids,
        "message": message,
        "severity": severity if severity in {"high", "medium", "low"} else "high",
        "repair_required": repair_required,
    }


def _dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[tuple[Any, ...]] = set()
    for finding in findings:
        marker = (
            finding.get("issue_type"),
            finding.get("block_id"),
            finding.get("target_id"),
            tuple(finding.get("source_span_ids", [])),
        )
        if marker in seen:
            continue
        seen.add(marker)
        result.append(finding)
    return result
