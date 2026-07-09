from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import CompiledField, Evidence, GeneratedSchema, Risk, TextSpan, to_jsonable
from .utils import stable_id


STANDARD_FIELD_NAMES = {
    "product.name": "product_name",
    "product.ingredients": "ingredients",
    "product.net_content": "net_content",
    "product.product_type": "product_type",
    "product.standard_code": "standard_code",
    "product.shelf_life": "shelf_life",
    "product.storage_condition": "storage",
    "product.nutrition_table": "nutrition_table",
    "product.date_marking": "date_marking",
    "product.directions": "directions",
    "product.warning": "warning",
    "product.origin": "origin",
    "content_item.name": "content_name",
    "content_item.net_content": "net_content",
    "content_item.product_category": "product_type",
    "content_item.ingredients": "ingredients",
    "principal.name": "principal",
    "principal.address": "principal_address",
    "principal.origin": "principal_origin",
    "principal.license_number": "principal_license",
    "principal.contact": "principal_contact",
    "principal.postal_code": "principal_postal_code",
    "principal.website": "principal_website",
    "manufacturer.name": "manufacturer",
    "manufacturer.address": "address",
    "manufacturer.origin": "origin",
    "manufacturer.license_number": "license",
    "manufacturer.contact": "manufacturer_contact",
    "manufacturer.postal_code": "manufacturer_postal_code",
    "manufacturer.website": "manufacturer_website",
    "barcode.commodity": "barcode",
    "barcode.outer_case": "outer_barcode",
    "requirement.text": "requirement",
}


def build_standard_items(
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    spans: list[TextSpan],
    source_path: Path,
    revision_blocks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    span_by_id = {span.span_id: span for span in spans}
    revision_by_field_id = _revision_by_field_id(revision_blocks or [])
    instance_counts: dict[tuple[str, str | None], int] = {}
    items = []

    for item_index, field in enumerate(compiled_fields.values(), start=1):
        standard_field = STANDARD_FIELD_NAMES.get(field.semantic_key, field.semantic_key)
        instance_key = (field.semantic_key, field.entity_id)
        instance_counts[instance_key] = instance_counts.get(instance_key, 0) + 1
        field_evidence = [evidence_by_id[ref] for ref in field.evidence_refs if ref in evidence_by_id]
        first_source = _source_from_evidence(field_evidence[0], span_by_id, source_path, field.section_id) if field_evidence else None
        revision = revision_by_field_id.get(field.field_id)
        item = {
            "id": stable_id("std", item_index),
            "field_id": field.field_id,
            "field": standard_field,
            "semantic_key": field.semantic_key,
            "label": field.display_name,
            "text": field.raw_value,
            "normalized_text": field.normalized_value,
            "value_hash": field.value_hash,
            "source": first_source or {"path": str(source_path), "section": field.section_id, "char_start": None, "char_end": None},
            "sources": [_source_from_evidence(item, span_by_id, source_path, field.section_id) for item in field_evidence],
            "group_id": field.entity_id,
            "table_id": field.table_id,
            "row_key": field.row_key,
            "instance_index": instance_counts[instance_key],
            "extraction_method": _extraction_method(field_evidence),
            "confidence": field.confidence.get("overall"),
            "confidence_breakdown": field.confidence,
            "quality_flags": _quality_flags(field),
            "comparison_required": field.field_type != "requirement"
            and not (revision and revision["revision_role"] == "historical_reference"),
            "review_required": field.review_required,
            "status": field.status,
            "evidence_refs": field.evidence_refs,
        }
        item["comparison_profile"] = _comparison_profile(field, item["source"])
        if revision:
            item.update(revision)
        items.append(item)
    return items


def build_quality_report(
    quality: dict[str, Any],
    risks: list[Risk],
    validation: list[dict[str, Any]],
    schema_audit: dict[str, Any],
    structure_audit: dict[str, Any],
    source_layers: dict[str, Any],
    table_quality_report: dict[str, Any],
    repair_plan: dict[str, Any],
    repair_attempts_artifact: dict[str, Any] | None = None,
    repair_trace: dict[str, Any] | None = None,
    repair_agent_candidates: dict[str, Any] | None = None,
    vdg_quality_report: dict[str, Any] | None = None,
    label_text_scope_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_checks = _quality_gate_checks(quality, validation, schema_audit, structure_audit, source_layers, table_quality_report, vdg_quality_report, label_text_scope_report)
    pass_gate = all(check["result"] == "passed" for check in gate_checks)
    issues = _quality_issues(risks, validation, schema_audit, structure_audit, source_layers, table_quality_report)
    status = "pass" if pass_gate and not issues else "review_required"
    return {
        "status": status,
        "downstream_allowed": status == "pass",
        "quality_gate": "strict",
        "overall_status": quality.get("overall_status"),
        "auto_ingest_allowed": quality.get("auto_ingest_allowed", False) and status == "pass",
        "gate_checks": gate_checks,
        "issues": issues,
        "issue_count": len(issues),
        "risk_counts": {
            "high": quality.get("high_risk_count", 0),
            "medium": quality.get("medium_risk_count", 0),
            "low": quality.get("low_risk_count", 0),
        },
        "repair_summary": _repair_summary(
            repair_plan,
            repair_attempts_artifact,
            repair_trace,
            repair_agent_candidates,
        ),
    }


def _repair_summary(
    repair_plan: dict[str, Any],
    repair_attempts_artifact: dict[str, Any] | None = None,
    repair_trace: dict[str, Any] | None = None,
    repair_agent_candidates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actions = repair_plan.get("actions", [])
    action_count = len(actions)
    attempt_status = (repair_attempts_artifact or {}).get("status")
    trace_status = (repair_trace or {}).get("status")
    candidate_status = (repair_agent_candidates or {}).get("status")
    downstream_blocking_reason = "repair_plan_pending" if repair_plan.get("status") != "pass" and action_count else None
    return {
        "repair_mode": repair_plan.get("repair_mode", "execute_plan"),
        "status": repair_plan.get("status"),
        "plan_status": repair_plan.get("status"),
        "action_count": action_count,
        "max_repair_rounds": repair_plan.get("max_repair_rounds"),
        "attempt_status": attempt_status,
        "attempt_count": (repair_attempts_artifact or {}).get("attempt_count", 0),
        "trace_status": trace_status,
        "round_count": (repair_trace or {}).get("round_count", 0),
        "applied_attempt_count": (repair_trace or {}).get("applied_attempt_count", 0),
        "final_audit_finding_count": (repair_trace or {}).get("final_audit_finding_count", 0),
        "agent_candidate_status": candidate_status,
        "agent_candidate_count": (repair_agent_candidates or {}).get("candidate_count", 0),
        "downstream_blocking_reason": downstream_blocking_reason,
    }


def build_field_groups(
    entities: dict[str, dict[str, Any]],
    compiled_fields: dict[str, CompiledField],
    standard_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    item_by_field_id = {item["evidence_refs"][0]: item for item in standard_items if item.get("evidence_refs")}
    groups = []
    for entity in entities.values():
        entity_id = entity.get("entity_id")
        if entity_id == "product_001":
            continue
        entity_fields = [field for field in compiled_fields.values() if field.entity_id == entity_id]
        field_items = []
        for field in entity_fields:
            item = _standard_item_for_field(field, standard_items, item_by_field_id)
            field_items.append(
                {
                    "field_id": field.field_id,
                    "semantic_key": field.semantic_key,
                    "standard_item_id": item.get("id") if item else None,
                    "text": field.raw_value,
                }
            )
        groups.append(
            {
                "group_id": entity_id,
                "group_type": entity.get("entity_type"),
                "instance_index": entity.get("index"),
                "fields": field_items,
                "linked_table_ids": entity.get("linked_table_ids", []),
                "container_text": _container_text(entity_fields),
            }
        )
    return groups


def build_tables_artifact(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return tables


def build_lists_artifact(field_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content_items = [group for group in field_groups if group.get("group_type") == "content_item"]
    if not content_items:
        return []
    return [
        {
            "list_id": "content_items_001",
            "list_type": "content_items",
            "item_count": len(content_items),
            "items": [
                {
                    "index": item.get("instance_index"),
                    "group_id": item.get("group_id"),
                    "text": item.get("container_text"),
                }
                for item in content_items
            ],
        }
    ]


def build_structured_document(
    document: dict[str, Any],
    source_layers: dict[str, Any],
    schema: GeneratedSchema,
    regions: list[dict[str, Any]],
    standard_items: list[dict[str, Any]],
    field_groups: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    lists: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "document": document,
        "sections": schema.sections,
        "regions": to_jsonable(regions),
        "source_text": [
            {
                "span_id": span["span_id"],
                "page": span["page"],
                "source": span["source"],
                "text": span["text"],
                "bbox_status": span["bbox_status"],
            }
            for span in source_layers.get("spans", [])
        ],
        "field_count": len(standard_items),
        "field_group_count": len(field_groups),
        "table_count": len(tables),
        "list_count": len(lists),
        "standard_item_ids": [item["id"] for item in standard_items],
    }


def build_taxonomy_proposals(compiled_fields: dict[str, CompiledField], evidence: list[Evidence]) -> list[dict[str, Any]]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    proposals = []
    for field in compiled_fields.values():
        if not (field.semantic_key.startswith("custom.") or field.semantic_key.startswith("proposed.")):
            continue
        first_evidence = evidence_by_id.get(field.evidence_refs[0]) if field.evidence_refs else None
        proposals.append(
            {
                "field": field.semantic_key,
                "label": field.display_name,
                "example_text": field.raw_value,
                "source": to_jsonable(first_evidence) if first_evidence else None,
                "reason": "field_not_in_approved_packaging_label_taxonomy",
                "required_review": True,
            }
        )
    return proposals


def _source_from_evidence(evidence: Evidence, span_by_id: dict[str, TextSpan], source_path: Path, section_id: str | None) -> dict[str, Any]:
    span = span_by_id.get(evidence.source_node_ids[0]) if evidence.source_node_ids else None
    char_start = None
    char_end = None
    if span:
        found = span.text.find(evidence.source_text)
        if found >= 0:
            char_start = found
            char_end = found + len(evidence.source_text)
    return {
        "path": str(source_path),
        "section": section_id,
        "char_start": char_start,
        "char_end": char_end,
        "page": evidence.page,
        "span_id": span.span_id if span else None,
        "bbox": _bbox_list(evidence),
        "bbox_pdf": to_jsonable(evidence.bbox_pdf),
        "bbox_normalized": to_jsonable(evidence.bbox_normalized),
        "source_text": evidence.source_text,
    }


def _bbox_list(evidence: Evidence) -> list[float] | None:
    if not evidence.bbox_pdf:
        return None
    bbox = evidence.bbox_pdf
    if isinstance(bbox, dict):
        return [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
    return [bbox.x, bbox.y, bbox.width, bbox.height]


def _extraction_method(field_evidence: list[Evidence]) -> str:
    methods = []
    for item in field_evidence:
        methods.extend(item.extraction_methods)
    return "+".join(sorted(set(methods))) if methods else "unknown"


def _quality_flags(field: CompiledField) -> list[str]:
    flags = []
    if field.review_required:
        flags.append("manual_review_required")
    if field.reason:
        flags.append(field.reason)
    if field.normalization:
        flags.extend(field.normalization)
    return flags


def _comparison_profile(field: CompiledField, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_key": field.semantic_key,
        "normalized_value": field.normalized_value,
        "value_hash": field.value_hash,
        "section_id": field.section_id,
        "entity_id": field.entity_id,
        "table_id": field.table_id,
        "row_key": field.row_key,
        "bbox_normalized": source.get("bbox_normalized"),
        "evidence_refs": field.evidence_refs,
    }


def _quality_gate_checks(
    quality: dict[str, Any],
    validation: list[dict[str, Any]],
    schema_audit: dict[str, Any],
    structure_audit: dict[str, Any],
    source_layers: dict[str, Any],
    table_quality_report: dict[str, Any],
    vdg_quality_report: dict[str, Any] | None = None,
    label_text_scope_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    failed_validation_count = sum(1 for check in validation if check.get("result") != "passed")
    vdg_quality = vdg_quality_report or {"status": "not_available"}
    label_text_scope = label_text_scope_report or {"status": "pass", "extracted_out_of_scope_count": 0}
    return [
        _gate("no_high_risk", quality.get("high_risk_count", 0) == 0, quality.get("high_risk_count", 0)),
        _gate("validation_checks_pass", failed_validation_count == 0, failed_validation_count),
        _gate("vdg_quality_pass", vdg_quality.get("status") != "fail", vdg_quality.get("status")),
        _gate("label_text_scope_pass", label_text_scope.get("status") != "fail", label_text_scope.get("status")),
        _gate("label_text_scope_out_of_scope_zero", label_text_scope.get("extracted_out_of_scope_count") == 0, label_text_scope.get("extracted_out_of_scope_count")),
        _gate("schema_audit_pass", schema_audit.get("status") == "pass", schema_audit.get("status")),
        _gate("anchor_coverage_full", structure_audit.get("anchor_coverage") == 1.0, structure_audit.get("anchor_coverage")),
        _gate("missing_anchor_count_zero", structure_audit.get("missing_anchor_count", 0) == 0, structure_audit.get("missing_anchor_count", 0)),
        _gate("sequence_gap_count_zero", structure_audit.get("sequence_gap_count", 0) == 0, structure_audit.get("sequence_gap_count", 0)),
        _gate("group_issue_count_zero", structure_audit.get("group_issue_count", 0) == 0, structure_audit.get("group_issue_count", 0)),
        _gate("table_issue_count_zero", structure_audit.get("table_issue_count", 0) == 0, structure_audit.get("table_issue_count", 0)),
        _gate("required_prefix_issue_count_zero", structure_audit.get("required_prefix_issue_count", 0) == 0, structure_audit.get("required_prefix_issue_count", 0)),
        _gate("container_duplicate_issue_count_zero", structure_audit.get("container_duplicate_issue_count", 0) == 0, structure_audit.get("container_duplicate_issue_count", 0)),
        _gate("agent_override_issue_count_zero", structure_audit.get("agent_override_issue_count", 0) == 0, structure_audit.get("agent_override_issue_count", 0)),
        _gate("duplicate_coverage_issue_count_zero", structure_audit.get("duplicate_coverage_issue_count", 0) == 0, structure_audit.get("duplicate_coverage_issue_count", 0)),
        _gate("source_layers_pass", source_layers.get("status") == "pass", source_layers.get("status")),
        _gate("table_quality_pass", table_quality_report.get("status") == "pass", table_quality_report.get("status")),
    ]


def _gate(name: str, passed: bool, actual: Any) -> dict[str, Any]:
    return {"check": name, "result": "passed" if passed else "failed", "actual": actual}


def _quality_issues(
    risks: list[Risk],
    validation: list[dict[str, Any]],
    schema_audit: dict[str, Any],
    structure_audit: dict[str, Any],
    source_layers: dict[str, Any],
    table_quality_report: dict[str, Any],
) -> list[dict[str, Any]]:
    issues = [
        {
            "issue_id": risk.risk_id,
            "issue_type": risk.risk_type,
            "severity": risk.risk_level,
            "target_type": risk.target_type,
            "target_id": risk.target_id,
            "message": risk.message,
            "evidence_refs": risk.evidence_refs,
        }
        for risk in risks
    ]
    for check in validation:
        if check.get("result") != "passed":
            issues.append(
                {
                    "issue_type": check.get("check_type", "validation_failed"),
                    "severity": check.get("severity", "high"),
                    "target_id": check.get("target_id"),
                    "detail": check,
                }
            )
    for item in schema_audit.get("issues", []):
        issues.append({"issue_type": item.get("issue_type"), "severity": item.get("severity", "high"), "detail": item})
    sequence_gap_count = structure_audit.get("sequence_gap_count", 0)
    if sequence_gap_count:
        issues.append(
            {
                "issue_type": "content_sequence_gap",
                "severity": "high",
                "detail": {
                    "expected": "content item indexes are continuous",
                    "actual": sequence_gap_count,
                    "source": {"anchor_inventory": structure_audit.get("anchor_inventory", [])},
                    "repair_hint": "Run Anchor/Boundary Agent to recover missing 内容物 sequence anchors.",
                },
            }
        )
    for key in (
        "missing_anchor_issues",
        "group_issues",
        "table_issues",
        "required_prefix_issues",
        "container_duplicate_issues",
        "agent_override_issues",
        "duplicate_coverage_issues",
    ):
        for item in structure_audit.get(key, []):
            issues.append({"issue_type": key[:-1], "severity": "high", "detail": item})
    for item in source_layers.get("source_issues", []):
        issues.append({"issue_type": item.get("issue_type"), "severity": item.get("severity"), "detail": item})
    for item in table_quality_report.get("issues", []):
        issues.append({"issue_type": item.get("issue_type"), "severity": item.get("severity"), "detail": item})
    return issues


def _revision_by_field_id(revision_blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    revision_by_field_id: dict[str, dict[str, Any]] = {}
    for block in revision_blocks:
        role = block.get("revision_role")
        if role not in {"before", "after"}:
            continue
        revision_role = "historical_reference" if role == "before" else "current_standard"
        for field_ref in block.get("fields", []):
            if not isinstance(field_ref, dict) or not field_ref.get("field_id"):
                continue
            revision_by_field_id[str(field_ref["field_id"])] = {
                "revision_role": revision_role,
                "revision_block_role": role,
                "is_current_standard": role == "after",
            }
    return revision_by_field_id


def _standard_item_for_field(
    field: CompiledField,
    standard_items: list[dict[str, Any]],
    item_by_first_evidence_ref: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if field.evidence_refs:
        item = item_by_first_evidence_ref.get(field.evidence_refs[0])
        if item and item.get("semantic_key") == field.semantic_key:
            return item
    for item in standard_items:
        if item.get("semantic_key") == field.semantic_key and item.get("group_id") == field.entity_id:
            return item
    return None


def _container_text(entity_fields: list[CompiledField]) -> str | None:
    name_field = next((field for field in entity_fields if field.semantic_key == "content_item.name"), None)
    if name_field:
        return name_field.raw_value
    return None
