from __future__ import annotations

import re
from typing import Any

from .models import CompiledField, Evidence, TextSpan
from .utils import stable_id


PANEL_RE = re.compile(r"^第[一二三四五六七八九十]+唛$")
CONTENT_ITEM_RE = re.compile(r"^内容物\s*(\d+)\s*[:：]\s*(.+)$")
NUTRITION_ROW_LABELS = (
    "能量",
    "蛋白质",
    "蛋⽩质",
    "脂肪",
    "饱和脂肪",
    "--饱和脂肪",
    "—饱和脂肪",
    "碳水化合物",
    "碳⽔化合物",
    "糖",
    "--糖",
    "—糖",
    "钠",
    "钙",
)


def detect_regions(spans: list[TextSpan]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for span in spans:
        text = span.text.strip()
        if not text:
            continue
        if PANEL_RE.match(text):
            regions.append(_region("package_panel", text, span))
        if "更改前" in text:
            regions.append(_region("revision_before", "更改前", span))
        if "更改后" in text or "修改后" in text:
            regions.append(_region("revision_after", "更改后", span))
        if "营养成分表" in text:
            regions.append(_region("nutrition_table_area", text, span))
        if CONTENT_ITEM_RE.match(text):
            regions.append(_region("content_item_block", text, span))
        if any(label in text for label in ("文字要求", "其它要求", "其他要求", "设计注意", "推广注意", "日期喷印注意", "变化说明")):
            regions.append(_region("requirements", text, span))
        if any(label in text for label in ("商品条码", "外箱条码")):
            regions.append(_region("barcode_area", text, span))
        if any(label in text for label in ("生产厂商", "生产商", "生产者", "食品生产许可证编号", "受托方", "委托方")):
            regions.append(_region("manufacturer_info", text, span))

    detected = []
    for index, region in enumerate(regions, start=1):
        region_id = stable_id("reg", index)
        item = {**region, "region_id": region_id}
        if item["region_type"] == "package_panel":
            item["panel_id"] = region_id
            item["panel_name"] = item["display_name"]
            item["panel_type"] = item["region_type"]
        detected.append(item)
    return detected


def attach_region_evidence(
    regions: list[dict[str, Any]],
    spans: list[TextSpan],
    existing_evidence_count: int,
) -> tuple[list[dict[str, Any]], list[Evidence]]:
    span_by_id = {span.span_id: span for span in spans}
    evidence: list[Evidence] = []
    enriched_regions: list[dict[str, Any]] = []
    for region in regions:
        enriched = dict(region)
        evidence_refs = []
        for span_id in _unique_refs(enriched.get("source_span_ids", [])):
            span = span_by_id.get(span_id)
            if not span:
                continue
            ev_id = stable_id("ev", existing_evidence_count + len(evidence) + 1)
            evidence.append(
                Evidence(
                    evidence_id=ev_id,
                    source_text=span.text,
                    page=span.page,
                    extraction_methods=[span.source, "region_detector"],
                    bbox_status="available" if span.bbox_pdf else "missing",
                    source_node_ids=[span.span_id],
                    bbox_pdf=span.bbox_pdf,
                    bbox_normalized=span.bbox_normalized,
                )
            )
            evidence_refs.append(ev_id)
        enriched["evidence_refs"] = evidence_refs
        enriched_regions.append(enriched)
    return enriched_regions, evidence


def assign_region_memberships(
    regions: list[dict[str, Any]],
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    tables: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    spans: list[TextSpan],
) -> list[dict[str, Any]]:
    span_order = {span.span_id: index for index, span in enumerate(spans)}
    span_page = {span.span_id: span.page for span in spans}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    panel_bounds = _panel_bounds(regions, spans)
    assigned: list[dict[str, Any]] = []
    for region in regions:
        enriched = dict(region)
        enriched.setdefault("fields", [])
        enriched.setdefault("tables", [])
        enriched.setdefault("entities", [])
        if enriched.get("region_type") != "package_panel":
            enriched["assignment_status"] = "not_applicable"
            assigned.append(enriched)
            continue

        bounds = panel_bounds.get(str(enriched.get("region_id")))
        if bounds:
            field_ids = _field_ids_in_bounds(compiled_fields, evidence_by_id, span_order, span_page, bounds)
            table_ids = _table_ids_in_bounds(tables, span_order, span_page, bounds)
            entity_ids = _entity_ids_for_region(entities, field_ids, table_ids)
        else:
            field_ids = []
            table_ids = []
            entity_ids = []

        enriched["fields"] = field_ids
        enriched["tables"] = table_ids
        enriched["entities"] = entity_ids
        if field_ids or table_ids:
            enriched["assignment_status"] = "assigned"
        else:
            enriched["assignment_status"] = "uncertain"
            enriched["status"] = "uncertain"
            enriched["risk_level"] = "high"
            enriched["review_required"] = True
        assigned.append(enriched)
    return assigned


def content_item_names(spans: list[TextSpan]) -> dict[str, str]:
    names: dict[str, str] = {}
    for span in spans:
        match = CONTENT_ITEM_RE.match(span.text.strip())
        if not match:
            continue
        index = int(match.group(1))
        names[f"content_item_{index:03d}"] = match.group(2).strip()
    return names


def extract_nutrition_tables(
    spans: list[TextSpan],
    content_names: dict[str, str],
    existing_evidence_count: int,
) -> tuple[list[dict[str, Any]], list[Evidence]]:
    tables: list[dict[str, Any]] = []
    evidence: list[Evidence] = []
    title_indexes = [index for index, span in enumerate(spans) if "营养成分表" in span.text]

    for table_index, span_index in enumerate(title_indexes, start=1):
        title_span = spans[span_index]
        row_spans = _nutrition_row_spans(spans, span_index)
        source_spans = [title_span] + row_spans
        evidence_refs = []
        for source_span in source_spans:
            ev_id = stable_id("ev", existing_evidence_count + len(evidence) + 1)
            evidence.append(_evidence(ev_id, source_span))
            evidence_refs.append(ev_id)

        columns = _nutrition_columns(row_spans)
        rows = []
        for row_index, row_span in enumerate(row_spans, start=1):
            if "项目" in row_span.text and not rows:
                continue
            if not _looks_like_nutrition_row(row_span.text):
                continue
            row_ev_ref = evidence_refs[source_spans.index(row_span)]
            item, amount, nrv = _split_nutrition_row(row_span.text)
            rows.append(
                {
                    "row_id": stable_id("row", row_index),
                    "row_key": _nutrition_row_key(item),
                    "evidence_refs": [row_ev_ref],
                    "cells": [
                        {
                            "column_id": "col_001",
                            "raw_value": item,
                            "normalized_value": item,
                            "evidence_refs": [row_ev_ref],
                        },
                        {
                            "column_id": "col_002",
                            "raw_value": amount,
                            "normalized_value": amount,
                            "evidence_refs": [row_ev_ref],
                        },
                        {
                            "column_id": "col_003",
                            "raw_value": nrv,
                            "normalized_value": nrv,
                            "evidence_refs": [row_ev_ref],
                        },
                    ],
                }
            )

        table_id = stable_id("tbl", table_index)
        bbox_metadata = _table_bbox_metadata_from_spans(source_spans)
        confidence = {
            "table_structure_confidence": 0.96 if rows else 0.50,
            "evidence_confidence": 1.0 if bbox_metadata["bbox_status"] == "available" else 0.80,
        }
        review_required = not bool(rows) or _minimum_confidence(confidence) < 0.95
        tables.append(
            {
                "table_id": table_id,
                "table_type": "nutrition_facts",
                "title": title_span.text,
                "linked_entity_id": _linked_content_entity(title_span.text, content_names),
                "columns": columns,
                "rows": rows,
                "status": "manual_review_required" if review_required else "verified",
                **bbox_metadata,
                "confidence": confidence,
                "criticality": "critical",
                "risk_level": "high" if review_required else "low",
                "review_required": review_required,
                "evidence_refs": evidence_refs,
                "source_span_ids": [source_span.span_id for source_span in source_spans],
            }
        )

    return tables, evidence


def extract_nutrition_tables_from_layers(
    table_layers: dict[str, Any],
    content_names: dict[str, str],
    existing_evidence_count: int,
) -> tuple[list[dict[str, Any]], list[Evidence]]:
    tables: list[dict[str, Any]] = []
    evidence: list[Evidence] = []

    primary_layers = [table for table in table_layers.get("tables", []) if table.get("parser") == "text_span_nutrition"]
    for table_index, table_layer in enumerate(primary_layers, start=1):
        evidence_refs = []
        source_span_ids = table_layer.get("source_span_ids", [])
        title_text = str(table_layer.get("title", "")).strip()
        bbox_metadata = _table_bbox_metadata_from_layer(table_layer)
        evidence_bbox_status = bbox_metadata["bbox_status"]
        if title_text:
            ev_id = stable_id("ev", existing_evidence_count + len(evidence) + 1)
            evidence.append(
                Evidence(
                    evidence_id=ev_id,
                    source_text=title_text,
                    page=int(table_layer.get("page", 1)),
                    extraction_methods=[table_layer.get("parser", "table_parser")],
                    bbox_status=evidence_bbox_status,
                    source_node_ids=source_span_ids[:1],
                    bbox_pdf=bbox_metadata.get("bbox_pdf"),
                    bbox_normalized=bbox_metadata.get("bbox_normalized"),
                )
            )
            evidence_refs.append(ev_id)

        data_evidence_refs = []
        for row in table_layer.get("rows", []):
            if row.get("row_type") != "data":
                continue
            row_text = " ".join(cell.get("text", "") for cell in row.get("cells", [])).strip()
            ev_id = stable_id("ev", existing_evidence_count + len(evidence) + 1)
            evidence.append(
                Evidence(
                    evidence_id=ev_id,
                    source_text=row_text,
                    page=int(table_layer.get("page", 1)),
                    extraction_methods=[table_layer.get("parser", "table_parser")],
                    bbox_status=evidence_bbox_status,
                    source_node_ids=row.get("source_span_ids", []),
                    bbox_pdf=bbox_metadata.get("bbox_pdf"),
                    bbox_normalized=bbox_metadata.get("bbox_normalized"),
                )
            )
            evidence_refs.append(ev_id)
            data_evidence_refs.append(ev_id)

        final_rows = []
        data_rows = [row for row in table_layer.get("rows", []) if row.get("row_type") == "data"]
        for row_index, row in enumerate(data_rows, start=1):
            row_ev_ref = data_evidence_refs[row_index - 1] if row_index - 1 < len(data_evidence_refs) else None
            cells = []
            for cell in row.get("cells", []):
                column_id = f"col_{int(cell.get('col_index', 0)) + 1:03d}"
                cells.append(
                    {
                        "column_id": column_id,
                        "raw_value": cell.get("text", ""),
                        "normalized_value": cell.get("text", ""),
                        "evidence_refs": [row_ev_ref] if row_ev_ref else [],
                    }
                )
            final_rows.append(
                {
                    "row_id": stable_id("row", row_index),
                    "row_key": row.get("row_key") or "unknown",
                    "evidence_refs": [row_ev_ref] if row_ev_ref else [],
                    "cells": cells,
                }
            )

        table_id = stable_id("tbl", table_index)
        rows_available = bool(final_rows)
        confidence = {
            "table_structure_confidence": table_layer.get("confidence", 0.50),
            "evidence_confidence": 1.0 if bbox_metadata["bbox_status"] == "available" else 0.80,
        }
        review_required = not rows_available or _minimum_confidence(confidence) < 0.95
        tables.append(
            {
                "table_id": table_id,
                "table_layer_id": table_layer.get("table_layer_id"),
                "table_type": table_layer.get("table_type", "nutrition_facts"),
                "page": int(table_layer.get("page", 1)),
                "title": table_layer.get("title", "营养成分表"),
                "linked_entity_id": _linked_content_entity(table_layer.get("title", ""), content_names),
                "columns": table_layer.get("columns", []),
                "rows": final_rows,
                "status": "manual_review_required" if review_required else "verified",
                **bbox_metadata,
                "confidence": confidence,
                "criticality": "critical",
                "risk_level": "high" if review_required else "low",
                "review_required": review_required,
                "evidence_refs": evidence_refs,
                "source_span_ids": source_span_ids,
            }
        )

    return tables, evidence


def build_entities(
    compiled_fields: dict[str, CompiledField],
    tables: list[dict[str, Any]],
    entity_plans: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    planned_types = {
        str(item.get("entity_id")): _canonical_entity_type(str(item.get("entity_type") or item.get("type") or ""))
        for item in entity_plans or []
        if isinstance(item, dict) and item.get("entity_id")
    }
    entities: dict[str, dict[str, Any]] = {
        "product_001": {
            "entity_id": "product_001",
            "entity_type": "product",
            "index": 1,
            "fields": {},
            "linked_table_ids": [],
            "evidence_refs": [],
            "status": "verified",
            "confidence": {"entity_linking_confidence": 1.0, "overall": 1.0},
            "risk_level": "info",
            "review_required": False,
        }
    }

    for field in compiled_fields.values():
        if not field.entity_id:
            continue
        entity = entities.setdefault(
            field.entity_id,
            {
                "entity_id": field.entity_id,
                "entity_type": _entity_type_for_field(
                    field.semantic_key,
                    planned_types.get(field.entity_id) or _entity_type(field.entity_id),
                ),
                "index": _entity_index(field.entity_id),
                "fields": {},
                "linked_table_ids": [],
                "evidence_refs": [],
                "status": "verified",
                "confidence": {"entity_linking_confidence": 1.0, "overall": 1.0},
                "risk_level": "info",
                "review_required": False,
            },
        )
        entity["fields"][_field_slot(field.semantic_key)] = _entity_field(field)
        entity["evidence_refs"] = _unique_refs([*entity.get("evidence_refs", []), *field.evidence_refs])
        _merge_entity_confidence(entity, field)

    for table in tables:
        linked_entity_id = table.get("linked_entity_id") or "product_001"
        entity = entities.setdefault(
            linked_entity_id,
            {
                "entity_id": linked_entity_id,
                "entity_type": planned_types.get(linked_entity_id) or _entity_type(linked_entity_id),
                "index": _entity_index(linked_entity_id),
                "fields": {},
                "linked_table_ids": [],
                "evidence_refs": [],
                "status": "verified",
                "confidence": {"entity_linking_confidence": 1.0, "overall": 1.0},
                "risk_level": "info",
                "review_required": False,
            },
        )
        entity["linked_table_ids"].append(table["table_id"])
        entity["evidence_refs"] = _unique_refs([*entity.get("evidence_refs", []), *table.get("evidence_refs", [])])
        _merge_table_confidence(entity, table)

    return entities


def _entity_field(field: CompiledField) -> dict[str, Any]:
    return {
        "field_id": field.field_id,
        "semantic_key": field.semantic_key,
        "value": field.raw_value,
        "normalized_value": field.normalized_value,
        "status": field.status,
        "criticality": field.criticality,
        "confidence": field.confidence,
        "risk_level": field.risk_level,
        "review_required": field.review_required,
        "evidence_refs": field.evidence_refs,
    }


def _merge_entity_confidence(entity: dict[str, Any], field: CompiledField) -> None:
    confidence = dict(entity.get("confidence", {}))
    current_linking = float(confidence.get("entity_linking_confidence") or 1.0)
    field_linking = field.confidence.get("entity_linking_confidence")
    if isinstance(field_linking, (int, float)) and not isinstance(field_linking, bool):
        confidence["entity_linking_confidence"] = min(current_linking, float(field_linking))
    confidence["overall"] = min(
        value
        for value in confidence.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    entity["confidence"] = confidence
    if field.review_required or float(confidence.get("entity_linking_confidence") or 1.0) < 0.90:
        entity["status"] = "uncertain"
        entity["risk_level"] = "medium"
        entity["review_required"] = True


def _merge_table_confidence(entity: dict[str, Any], table: dict[str, Any]) -> None:
    confidence = dict(entity.get("confidence", {}))
    table_confidence = table.get("confidence", {})
    if isinstance(table_confidence, dict):
        table_linking = table_confidence.get("entity_linking_confidence")
        if isinstance(table_linking, (int, float)) and not isinstance(table_linking, bool):
            confidence["entity_linking_confidence"] = min(
                float(confidence.get("entity_linking_confidence") or 1.0),
                float(table_linking),
            )
    confidence["overall"] = min(
        value
        for value in confidence.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    entity["confidence"] = confidence
    if table.get("review_required") or float(confidence.get("entity_linking_confidence") or 1.0) < 0.90:
        entity["status"] = "uncertain"
        entity["risk_level"] = "medium"
        entity["review_required"] = True


def build_requirements(compiled_fields: dict[str, CompiledField]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for field in compiled_fields.values():
        if field.field_type != "requirement":
            continue
        requirements.append(
            {
                "requirement_id": stable_id("req", len(requirements) + 1),
                "requirement_type": _requirement_type(field.raw_value),
                "target": _requirement_target(field.raw_value),
                "requirement_text": field.raw_value,
                "status": "extracted",
                "confidence": field.confidence,
                "verification_status": "not_verified_in_mvp",
                "risk_level": "info" if not field.review_required else field.risk_level,
                "review_required": field.review_required,
                "evidence_refs": field.evidence_refs,
            }
        )
    return requirements


def _requirement_type(text: str) -> str:
    if _contains_any(text, ("日期喷印", "生产日期", "保质期到期日", "喷码日期")) and _contains_any(
        text,
        ("喷印", "喷码", "打码", "打印"),
    ):
        return "date_printing_requirement"
    if "条码" in text:
        return "barcode_requirement"
    if _contains_any(text, ("推广", "宣传", "广告", "宣称", "夸大")):
        return "advertising_claim_restriction"
    if _contains_any(text, ("变化说明", "变更说明", "更改说明", "修改说明")):
        return "change_note"
    if _contains_any(text, ("字高", "字号", "字体", "文字高度", "字符高度")):
        return "text_size"
    if _contains_any(
        text,
        ("主视面", "版面", "位置", "布局", "排版", "居中", "左侧", "右侧", "上方", "下方"),
    ):
        return "layout_requirement"
    if _contains_any(text, ("回收", "可回收", "循环标志", "环保标志")):
        return "recycling_mark_requirement"
    if _contains_any(text, ("印刷", "喷印", "喷码", "打码", "打印")):
        return "printing_requirement"
    if _contains_any(text, ("设计注意", "设计要求")):
        return "design_note"
    return "other"


def _requirement_target(text: str) -> str | None:
    for target in (
        "净含量",
        "商品条码",
        "外箱条码",
        "条码",
        "生产日期",
        "保质期到期日",
        "喷码日期",
        "主视面",
        "营养成分表",
        "回收标志",
        "循环标志",
        "印刷",
        "推广",
        "宣传",
    ):
        if target in text:
            return target
    return None


def build_revision_blocks(
    regions: list[dict[str, Any]],
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence] | None = None,
    spans: list[TextSpan] | None = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    revision_regions = [region for region in regions if region["region_type"] in {"revision_before", "revision_after"}]
    assignments = _revision_field_assignments(revision_regions, compiled_fields, evidence or [], spans or [])
    for region in revision_regions:
        role = "before" if region["region_type"] == "revision_before" else "after"
        assigned_fields = assignments.get(region.get("region_id", ""), [])
        assignment_pending = not bool(assigned_fields)
        blocks.append(
            {
                "revision_block_id": f"revision_{role}",
                "region_id": region.get("region_id"),
                "revision_role": role,
                "revision_status": "current_standard" if role == "after" else "historical_reference",
                "display_name": region["display_name"],
                "fields": assigned_fields,
                "is_current_standard": role == "after",
                "status": "uncertain" if assignment_pending else "verified",
                "risk_level": "high" if assignment_pending else "info",
                "review_required": assignment_pending,
                "evidence_refs": region.get("evidence_refs", []),
                "source_span_ids": region["source_span_ids"],
                "assignment_status": "region_detected_field_assignment_pending" if assignment_pending else "assigned_by_span_order",
                "assignment_method": None if assignment_pending else "source_span_order_between_revision_markers",
            }
        )
    return blocks


def build_structure_audit(
    spans: list[TextSpan],
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    tables: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    anchors = _anchor_inventory(spans, regions)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    assigned_span_ids = set()
    for field in compiled_fields.values():
        for evidence_ref in field.evidence_refs:
            source_evidence = evidence_by_id.get(evidence_ref)
            if source_evidence:
                assigned_span_ids.update(source_evidence.source_node_ids)
    assigned_span_ids.update(
        source_id
        for table in tables
        for source_id in table.get("source_span_ids", [])
    )

    missing_anchor_issues = [
        {
            "expected": anchor["anchor_type"],
            "actual": None,
            "source": anchor,
            "repair_hint": "Create an evidence-bound field/table/region assignment for this anchor.",
        }
        for anchor in anchors
        if anchor["span_id"] not in assigned_span_ids and anchor["anchor_type"] not in {"revision", "package_panel"}
    ]
    required_prefix_issues = _required_prefix_issues(compiled_fields)
    group_issues = _group_issues(entities)
    table_issues = _table_issues(tables)
    anchor_coverage = 1.0 if not anchors else round((len(anchors) - len(missing_anchor_issues)) / len(anchors), 4)

    return {
        "anchor_inventory": anchors,
        "anchor_coverage": anchor_coverage,
        "missing_anchor_issues": missing_anchor_issues,
        "missing_anchor_count": len(missing_anchor_issues),
        "sequence_gap_count": _content_sequence_gap_count(anchors),
        "group_issues": group_issues,
        "group_issue_count": len(group_issues),
        "table_issues": table_issues,
        "table_issue_count": len(table_issues),
        "required_prefix_issues": required_prefix_issues,
        "required_prefix_issue_count": len(required_prefix_issues),
        "container_duplicate_issues": [],
        "container_duplicate_issue_count": 0,
        "agent_override_issues": [],
        "agent_override_issue_count": 0,
        "duplicate_coverage_issues": [],
        "duplicate_coverage_issue_count": 0,
    }


def build_repair_plan(
    risks: list[Any],
    audit_findings: list[dict[str, Any]],
    structure_audit: dict[str, Any],
    max_rounds: int,
    review_items: list[dict[str, Any]] | None = None,
    rejected_agent_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for risk in risks:
        actions.append(
            {
                "action_id": stable_id("repair_action", len(actions) + 1),
                "target_type": risk.target_type,
                "target_id": risk.target_id,
                "issue_type": risk.risk_type,
                "recommended_agent": _repair_agent_for_issue(risk.risk_type),
                "expected_output": "span-grounded extraction plan patch",
                "acceptance_gate": "compiler_and_validation_pass",
            }
        )
    for finding in audit_findings:
        actions.append(
            {
                "action_id": stable_id("repair_action", len(actions) + 1),
                "target_type": finding["target_type"],
                "target_id": finding["target_id"],
                "issue_type": finding["finding_type"],
                "recommended_agent": _repair_agent_for_issue(finding["finding_type"]),
                "expected_output": "span-grounded extraction plan patch",
                "acceptance_gate": "compiler_and_validation_pass",
            }
        )
    for issue in structure_audit.get("missing_anchor_issues", []):
        actions.append(
            {
                "action_id": stable_id("repair_action", len(actions) + 1),
                "target_type": "anchor",
                "target_id": issue["source"]["span_id"],
                "issue_type": "missing_anchor",
                "recommended_agent": "anchor_boundary_agent",
                "expected_output": "field/table assignment using existing source span",
                "acceptance_gate": "structure_audit_anchor_coverage_pass",
            }
        )
    if structure_audit.get("sequence_gap_count", 0):
        actions.append(
            {
                "action_id": stable_id("repair_action", len(actions) + 1),
                "target_type": "anchor_sequence",
                "target_id": "content_item_sequence",
                "issue_type": "content_sequence_gap",
                "recommended_agent": "anchor_agent",
                "expected_output": "continuous content item anchor assignments",
                "acceptance_gate": "structure_audit_sequence_gap_zero",
            }
        )
    for key, issue_type, target_type, expected_output, acceptance_gate in (
        (
            "required_prefix_issues",
            "required_prefix_issue",
            "field",
            "content item name field preserving the original source prefix",
            "structure_audit_required_prefix_pass",
        ),
        (
            "container_duplicate_issues",
            "container_duplicate_issue",
            "field_group",
            "container text moved to field_groups without duplicate comparison item",
            "structure_audit_container_duplicate_pass",
        ),
        (
            "agent_override_issues",
            "agent_override_issue",
            "agent_candidate",
            "agent candidate merge that preserves stronger rule-grounded fields",
            "structure_audit_agent_override_pass",
        ),
        (
            "duplicate_coverage_issues",
            "duplicate_coverage_issue",
            "anchor",
            "deduplicated source coverage mapping",
            "structure_audit_duplicate_coverage_pass",
        ),
    ):
        for issue_index, issue in enumerate(structure_audit.get(key, []), start=1):
            actions.append(
                {
                    "action_id": stable_id("repair_action", len(actions) + 1),
                    "target_type": target_type,
                    "target_id": _structure_issue_target_id(issue, key, issue_index),
                    "issue_type": issue_type,
                    "recommended_agent": _repair_agent_for_issue(issue_type),
                    "expected_output": expected_output,
                    "acceptance_gate": acceptance_gate,
                    "source_issue": issue,
                }
            )
    for item in review_items or []:
        issue_type = str(item.get("reason") or "agent_candidate_review_required")
        actions.append(
            {
                "action_id": stable_id("repair_action", len(actions) + 1),
                "target_type": "agent_candidate",
                "target_id": _agent_candidate_target_id(item, "review_item"),
                "issue_type": issue_type,
                "recommended_agent": _repair_agent_for_issue(issue_type),
                "expected_output": "confirmed or corrected span-grounded extraction candidate",
                "acceptance_gate": "agent_candidate_span_validation_and_compiler_pass",
                "input_artifact": "review_items.json",
                "source_item": item,
            }
        )
    for item in rejected_agent_items or []:
        issue_type = str(item.get("reason") or "agent_candidate_rejected")
        actions.append(
            {
                "action_id": stable_id("repair_action", len(actions) + 1),
                "target_type": "agent_candidate",
                "target_id": _agent_candidate_target_id(item, "rejected_item"),
                "issue_type": issue_type,
                "recommended_agent": _repair_agent_for_issue(issue_type),
                "expected_output": "corrected span-grounded candidate that passes merge validation",
                "acceptance_gate": "agent_candidate_span_validation_and_compiler_pass",
                "input_artifact": "rejected_agent_items.json",
                "source_item": item,
            }
        )

    return {
        "repair_mode": "execute_plan",
        "max_repair_rounds": max_rounds,
        "status": "review_required" if actions else "pass",
        "action_count": len(actions),
        "actions": actions,
    }


def _revision_field_assignments(
    revision_regions: list[dict[str, Any]],
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    spans: list[TextSpan],
) -> dict[str, list[dict[str, Any]]]:
    if not revision_regions or not evidence or not spans:
        return {}

    span_order = {span.span_id: index for index, span in enumerate(spans)}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    markers = []
    for region in revision_regions:
        role = "before" if region["region_type"] == "revision_before" else "after"
        marker_indexes = [span_order[span_id] for span_id in region.get("source_span_ids", []) if span_id in span_order]
        if not marker_indexes:
            continue
        markers.append(
            {
                "region_id": region.get("region_id", ""),
                "role": role,
                "span_index": min(marker_indexes),
                "source_span_ids": region.get("source_span_ids", []),
            }
        )
    markers.sort(key=lambda item: item["span_index"])
    if not markers:
        return {}

    assignments: dict[str, list[dict[str, Any]]] = {
        str(marker["region_id"]): [] for marker in markers if marker.get("region_id")
    }
    for field in compiled_fields.values():
        field_span_ids = _field_source_span_ids(field, evidence_by_id)
        field_indexes = [span_order[span_id] for span_id in field_span_ids if span_id in span_order]
        if not field_indexes:
            continue
        first_field_index = min(field_indexes)
        previous_markers = [marker for marker in markers if marker["span_index"] < first_field_index]
        if not previous_markers:
            continue
        active_marker = previous_markers[-1]
        region_id = str(active_marker.get("region_id") or "")
        if not region_id:
            continue
        role = active_marker["role"]
        assignments.setdefault(region_id, []).append(
            {
                "field_id": field.field_id,
                "semantic_key": field.semantic_key,
                "display_name": field.display_name,
                "evidence_refs": field.evidence_refs,
                "source_span_ids": field_span_ids,
                "assignment_confidence": 0.90,
                "assignment_reason": f"field_source_span_after_revision_{role}_marker",
            }
        )

    return assignments


def _field_source_span_ids(field: CompiledField, evidence_by_id: dict[str, Evidence]) -> list[str]:
    span_ids: list[str] = []
    for evidence_ref in field.evidence_refs:
        source_evidence = evidence_by_id.get(evidence_ref)
        if not source_evidence:
            continue
        for span_id in source_evidence.source_node_ids:
            if span_id not in span_ids:
                span_ids.append(span_id)
    return span_ids


def _region(region_type: str, display_name: str, span: TextSpan) -> dict[str, Any]:
    return {
        "region_type": region_type,
        "display_name": display_name,
        "page": span.page,
        "source_span_ids": [span.span_id],
        "bbox_status": "available" if span.bbox_pdf else "missing",
        "bbox_pdf": span.bbox_pdf,
        "bbox_normalized": span.bbox_normalized,
        "confidence": 0.90,
        "status": "verified",
        "risk_level": "info",
        "review_required": False,
        "evidence_refs": [],
        "fields": [],
        "tables": [],
        "entities": [],
        "assignment_status": "not_applicable" if region_type != "package_panel" else "pending",
    }


def _panel_bounds(regions: list[dict[str, Any]], spans: list[TextSpan]) -> dict[str, dict[str, Any]]:
    span_order = {span.span_id: index for index, span in enumerate(spans)}
    span_page = {span.span_id: span.page for span in spans}
    panels = []
    for region in regions:
        if region.get("region_type") != "package_panel":
            continue
        indexes = [span_order[span_id] for span_id in region.get("source_span_ids", []) if span_id in span_order]
        if not indexes:
            continue
        start = min(indexes)
        first_span_id = next((span_id for span_id in region.get("source_span_ids", []) if span_id in span_page), None)
        if not first_span_id:
            continue
        panels.append({"region_id": region.get("region_id"), "page": span_page[first_span_id], "start": start})

    bounds: dict[str, dict[str, Any]] = {}
    for index, panel in enumerate(panels):
        same_page_next_starts = [
            other["start"]
            for other in panels[index + 1 :]
            if other["page"] == panel["page"] and other["start"] > panel["start"]
        ]
        bounds[str(panel["region_id"])] = {
            "page": panel["page"],
            "start": panel["start"],
            "end": min(same_page_next_starts) if same_page_next_starts else None,
        }
    return bounds


def _field_ids_in_bounds(
    compiled_fields: dict[str, CompiledField],
    evidence_by_id: dict[str, Evidence],
    span_order: dict[str, int],
    span_page: dict[str, int],
    bounds: dict[str, Any],
) -> list[str]:
    field_ids = []
    for field in compiled_fields.values():
        source_span_ids = [
            span_id
            for evidence_ref in field.evidence_refs
            for span_id in getattr(evidence_by_id.get(evidence_ref), "source_node_ids", [])
        ]
        if _span_ids_in_bounds(source_span_ids, span_order, span_page, bounds):
            field_ids.append(field.field_id)
    return sorted(set(field_ids))


def _table_ids_in_bounds(
    tables: list[dict[str, Any]],
    span_order: dict[str, int],
    span_page: dict[str, int],
    bounds: dict[str, Any],
) -> list[str]:
    table_ids = []
    for table in tables:
        if _span_ids_in_bounds(table.get("source_span_ids", []), span_order, span_page, bounds):
            table_ids.append(str(table.get("table_id")))
    return sorted(set(table_ids))


def _span_ids_in_bounds(
    source_span_ids: list[Any],
    span_order: dict[str, int],
    span_page: dict[str, int],
    bounds: dict[str, Any],
) -> bool:
    for span_id in source_span_ids:
        value = str(span_id)
        order = span_order.get(value)
        if order is None or span_page.get(value) != bounds["page"]:
            continue
        if order <= bounds["start"]:
            continue
        if bounds["end"] is not None and order >= bounds["end"]:
            continue
        return True
    return False


def _entity_ids_for_region(
    entities: dict[str, dict[str, Any]],
    field_ids: list[str],
    table_ids: list[str],
) -> list[str]:
    field_id_set = set(field_ids)
    table_id_set = set(table_ids)
    entity_ids = []
    for entity in entities.values():
        entity_field_ids = {
            field.get("field_id")
            for field in entity.get("fields", {}).values()
            if isinstance(field, dict)
        }
        entity_table_ids = set(entity.get("linked_table_ids", []))
        if entity_field_ids & field_id_set or entity_table_ids & table_id_set:
            entity_ids.append(str(entity.get("entity_id")))
    return sorted(set(entity_ids))


def _table_bbox_metadata_from_spans(spans: list[TextSpan]) -> dict[str, Any]:
    bbox_pdf = _union_bbox_pdf([span.bbox_pdf for span in spans])
    bbox_normalized = _union_bbox_normalized([span.bbox_normalized for span in spans])
    if not bbox_pdf and not bbox_normalized:
        return {"bbox_status": "missing"}
    metadata: dict[str, Any] = {"bbox_status": "available"}
    if bbox_pdf:
        metadata["bbox_pdf"] = bbox_pdf
    if bbox_normalized:
        metadata["bbox_normalized"] = bbox_normalized
    return metadata


def _table_bbox_metadata_from_layer(table_layer: dict[str, Any]) -> dict[str, Any]:
    bbox_pdf = table_layer.get("bbox_pdf")
    bbox_normalized = table_layer.get("bbox_normalized")
    if not bbox_pdf:
        bbox_pdf = _union_bbox_pdf(
            [
                cell.get("bbox_pdf")
                for row in table_layer.get("rows", [])
                if isinstance(row, dict)
                for cell in row.get("cells", [])
                if isinstance(cell, dict)
            ]
        )
    if not bbox_normalized:
        bbox_normalized = _union_bbox_normalized(
            [
                cell.get("bbox_normalized")
                for row in table_layer.get("rows", [])
                if isinstance(row, dict)
                for cell in row.get("cells", [])
                if isinstance(cell, dict)
            ]
        )
    if not bbox_pdf and not bbox_normalized:
        return {"bbox_status": "missing"}
    metadata: dict[str, Any] = {"bbox_status": "available"}
    if bbox_pdf:
        metadata["bbox_pdf"] = bbox_pdf
    if bbox_normalized:
        metadata["bbox_normalized"] = bbox_normalized
    return metadata


def _union_bbox_pdf(bboxes: list[Any]) -> dict[str, Any] | None:
    present = [bbox for bbox in bboxes if bbox]
    if len(present) != len(bboxes) or not present:
        return None
    x1 = min(float(_bbox_value(bbox, "x") or 0) for bbox in present)
    y1 = min(float(_bbox_value(bbox, "y") or 0) for bbox in present)
    x2 = max(float(_bbox_value(bbox, "x") or 0) + float(_bbox_value(bbox, "width") or 0) for bbox in present)
    y2 = max(float(_bbox_value(bbox, "y") or 0) + float(_bbox_value(bbox, "height") or 0) for bbox in present)
    first = present[0]
    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "page_width": _bbox_value(first, "page_width"),
        "page_height": _bbox_value(first, "page_height"),
        "unit": _bbox_value(first, "unit") or "pt",
        "origin": _bbox_value(first, "origin") or "top_left",
    }


def _union_bbox_normalized(bboxes: list[Any]) -> dict[str, float] | None:
    present = [bbox for bbox in bboxes if bbox]
    if len(present) != len(bboxes) or not present:
        return None
    return {
        "x1": min(float(_bbox_value(bbox, "x1") or 0) for bbox in present),
        "y1": min(float(_bbox_value(bbox, "y1") or 0) for bbox in present),
        "x2": max(float(_bbox_value(bbox, "x2") or 0) for bbox in present),
        "y2": max(float(_bbox_value(bbox, "y2") or 0) for bbox in present),
    }


def _bbox_value(bbox: Any, key: str) -> Any:
    if isinstance(bbox, dict):
        return bbox.get(key)
    return getattr(bbox, key, None)


def _minimum_confidence(confidence: dict[str, Any]) -> float:
    values = [float(value) for value in confidence.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return min(values) if values else 0.0


def _unique_refs(refs: list[Any]) -> list[str]:
    unique: list[str] = []
    for ref in refs:
        value = str(ref)
        if value not in unique:
            unique.append(value)
    return unique


def _nutrition_row_spans(spans: list[TextSpan], title_index: int) -> list[TextSpan]:
    title_page = spans[title_index].page
    rows: list[TextSpan] = []
    for span in spans[title_index + 1 :]:
        text = span.text.strip()
        if span.page != title_page:
            break
        if "营养成分表" in text:
            break
        if not rows and "项目" not in text and not _looks_like_nutrition_row(text):
            continue
        if rows and not _looks_like_nutrition_row(text) and "项目" not in text:
            break
        rows.append(span)
    return rows


def _nutrition_columns(row_spans: list[TextSpan]) -> list[dict[str, Any]]:
    for span in row_spans:
        if "项目" in span.text:
            parts = [part for part in re.split(r"\s{1,}", span.text.strip()) if part]
            if len(parts) >= 3:
                return [
                    {"column_id": "col_001", "name": parts[0]},
                    {"column_id": "col_002", "name": " ".join(parts[1:-1])},
                    {"column_id": "col_003", "name": parts[-1]},
                ]
    return [
        {"column_id": "col_001", "name": "项目"},
        {"column_id": "col_002", "name": "含量"},
        {"column_id": "col_003", "name": "NRV%"},
    ]


def _looks_like_nutrition_row(text: str) -> bool:
    return any(text.startswith(label) for label in NUTRITION_ROW_LABELS)


def _split_nutrition_row(text: str) -> tuple[str, str, str]:
    parts = [part for part in re.split(r"\s+", text.strip()) if part]
    if len(parts) <= 1:
        return text.strip(), "", ""
    nrv = parts[-1] if re.search(r"(%|-)$", parts[-1]) else ""
    amount_parts = parts[1:-1] if nrv else parts[1:]
    return parts[0], " ".join(amount_parts), nrv


def _nutrition_row_key(item: str) -> str:
    mapping = {
        "能量": "energy",
        "蛋白质": "protein",
        "蛋⽩质": "protein",
        "脂肪": "fat",
        "碳水化合物": "carbohydrate",
        "碳⽔化合物": "carbohydrate",
        "钠": "sodium",
        "钙": "calcium",
    }
    normalized = item.strip("—- ")
    return mapping.get(normalized, re.sub(r"\W+", "_", normalized).strip("_") or "unknown")


def _linked_content_entity(title: str, content_names: dict[str, str]) -> str | None:
    for entity_id, name in content_names.items():
        if name and name in title:
            return entity_id
    return None


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _evidence(evidence_id: str, span: TextSpan) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_text=span.text,
        page=span.page,
        extraction_methods=[span.source],
        bbox_status="available" if span.bbox_pdf else "missing",
        source_node_ids=[span.span_id],
        bbox_pdf=span.bbox_pdf,
        bbox_normalized=span.bbox_normalized,
    )


def _entity_type(entity_id: str) -> str:
    if entity_id.startswith("content_item_"):
        return "content_item"
    if entity_id.startswith("manufacturer_"):
        return "manufacturer"
    if entity_id.startswith("barcode_"):
        return "barcode"
    if entity_id.startswith("requirement_"):
        return "requirement"
    return "product"


def _canonical_entity_type(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    return {
        "business_operator": "manufacturer",
        "producer": "manufacturer",
        "manufacturer": "manufacturer",
        "principal": "principal",
        "entrusting_party": "principal",
        "content_item": "content_item",
        "product": "product",
        "barcode": "barcode",
        "requirement": "requirement",
    }.get(normalized, normalized or "product")


def _entity_type_for_field(semantic_key: str, fallback: str) -> str:
    prefix = semantic_key.split(".", 1)[0]
    if prefix in {"principal", "manufacturer", "content_item", "barcode", "requirement", "product"}:
        return prefix
    return fallback


def _entity_index(entity_id: str) -> int:
    try:
        return int(entity_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 1


def _field_slot(semantic_key: str) -> str:
    return semantic_key.split(".")[-1]


def _anchor_inventory(spans: list[TextSpan], regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    region_span_ids = {
        source_span_id: region["region_type"]
        for region in regions
        for source_span_id in region["source_span_ids"]
    }
    for span in spans:
        anchor_type = region_span_ids.get(span.span_id)
        if not anchor_type:
            if _looks_like_nutrition_row(span.text):
                anchor_type = "nutrition_row"
            elif any(label in span.text for label in ("品名", "配料", "条码", "许可证编号", "保质期", "贮存条件")):
                anchor_type = "field_anchor"
        if anchor_type:
            anchors.append(
                {
                    "span_id": span.span_id,
                    "anchor_type": anchor_type,
                    "page": span.page,
                    "text": span.text,
                }
            )
    return anchors


def _required_prefix_issues(compiled_fields: dict[str, CompiledField]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for field in compiled_fields.values():
        if field.semantic_key != "content_item.name":
            continue
        if not field.raw_value.startswith("内容物"):
            issues.append(
                {
                    "expected": "content_name keeps original 内容物 N prefix",
                    "actual": field.raw_value,
                    "source": {"field_id": field.field_id},
                    "repair_hint": "Use the full source span as content_item.name.",
                }
            )
    return issues


def _group_issues(entities: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for entity in entities.values():
        if entity["entity_type"] != "manufacturer":
            continue
        fields = entity.get("fields", {})
        if "license_number" not in fields and ("name" in fields or "address" in fields):
            issues.append(
                {
                    "expected": "manufacturer group includes license_number when present in source",
                    "actual": sorted(fields),
                    "source": {"entity_id": entity["entity_id"]},
                    "repair_hint": "Run Group Agent to bind producer/address/license into the same group.",
                }
            )
    return issues


def _table_issues(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "expected": "nutrition table has at least one structured nutrient row",
            "actual": len(table.get("rows", [])),
            "source": {"table_id": table["table_id"], "title": table["title"]},
            "repair_hint": "Run Table/List Agent to recover rows from source spans.",
        }
        for table in tables
        if not table.get("rows")
    ]


def _content_sequence_gap_count(anchors: list[dict[str, Any]]) -> int:
    indexes = []
    for anchor in anchors:
        match = CONTENT_ITEM_RE.match(anchor["text"])
        if match:
            indexes.append(int(match.group(1)))
    if not indexes:
        return 0
    expected = set(range(min(indexes), max(indexes) + 1))
    return len(expected.difference(indexes))


def _structure_issue_target_id(issue: dict[str, Any], fallback_prefix: str, index: int) -> str:
    source = issue.get("source", {})
    if isinstance(source, dict):
        for key in ("field_id", "entity_id", "table_id", "span_id", "region_id"):
            value = source.get(key)
            if value:
                return str(value)
    return stable_id(fallback_prefix, index)


def _repair_agent_for_issue(issue_type: str) -> str:
    if "agent_candidate" in issue_type or "span" in issue_type or "offset" in issue_type:
        return "boundary_agent"
    if "duplicate" in issue_type:
        return "dedupe_agent"
    if "table" in issue_type:
        return "table_list_agent"
    if "group" in issue_type or "manufacturer" in issue_type:
        return "group_agent"
    if "sequence" in issue_type:
        return "anchor_agent"
    if "format" in issue_type:
        return "boundary_validation_agent"
    if "adhesion" in issue_type or "truncation" in issue_type or "bbox" in issue_type:
        return "boundary_agent"
    if "anchor" in issue_type or "missing" in issue_type:
        return "anchor_agent"
    return "audit_repair_agent"


def _agent_candidate_target_id(item: dict[str, Any], prefix: str) -> str:
    index = item.get("item_index")
    try:
        return stable_id(prefix, int(index))
    except (TypeError, ValueError):
        return stable_id(prefix, 1)
