from __future__ import annotations

from typing import Any

from .models import CompiledField, Evidence, ExtractionPlan, GeneratedSchema, Risk, ReviewTask


def build_mvp_acceptance_metrics(
    *,
    schema: GeneratedSchema,
    plan: ExtractionPlan,
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    validation: list[dict[str, Any]],
    risks: list[Risk],
    review_tasks: list[ReviewTask],
    coverage: dict[str, Any],
    coverage_map: dict[str, Any],
    schema_audit: dict[str, Any],
    structure_audit: dict[str, Any],
    source_layers: dict[str, Any],
    table_quality_report: dict[str, Any],
    repair_trace: dict[str, Any],
    output_contract_validation_report: dict[str, Any],
    missing_item_report: dict[str, Any],
) -> dict[str, Any]:
    critical_fields = [field for field in compiled_fields.values() if field.criticality == "critical"]
    long_text_fields = [field for field in compiled_fields.values() if field.field_type == "long_text"]
    field_count = len(compiled_fields)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    fields_with_evidence = [field for field in compiled_fields.values() if field.evidence_refs]
    bbox_available_evidence = [item for item in evidence if item.bbox_status == "available" and item.bbox_pdf is not None]
    critical_without_bbox = [
        field
        for field in critical_fields
        if not field.evidence_refs or any(evidence_by_id.get(ref) is None or evidence_by_id[ref].bbox_status == "missing" for ref in field.evidence_refs)
    ]
    verified_without_evidence = [
        field
        for field in compiled_fields.values()
        if field.status in {"verified", "normalized", "extracted", "compiled"} and not field.evidence_refs
    ]
    no_guessing_failures = _failed_checks(validation, "no_guessing")
    bbox_failures = _failed_checks(validation, "bbox_integrity")
    audit_findings = _repair_trace_findings(repair_trace)
    adhesion_findings = [finding for finding in audit_findings if finding.get("finding_type") == "possible_field_adhesion"]
    truncation_findings = [finding for finding in audit_findings if "truncation" in str(finding.get("finding_type", ""))]
    high_risks = [risk for risk in risks if risk.risk_level == "high"]
    output_contract_failed_count = int(output_contract_validation_report.get("failed_count", 0) or 0)
    structure_blocking_issue_count = _structure_blocking_issue_count(structure_audit, coverage_map)
    blocking_issue_count = (
        len(high_risks)
        + output_contract_failed_count
        + int(schema_audit.get("blocking_issue_count", 0) or 0)
        + structure_blocking_issue_count
    )

    return {
        "metrics_version": "mvp_acceptance_metrics_v0.1",
        "status": "pass" if blocking_issue_count == 0 else "review_required",
        "blocking_issue_count": blocking_issue_count,
        "schema": {
            "field_definition_count": len(schema.field_definitions),
            "table_definition_count": len(schema.table_definitions),
            "requirement_definition_count": len(schema.requirement_definitions),
            "schema_audit_status": schema_audit.get("status"),
            "schema_audit_issue_count": schema_audit.get("issue_count", 0),
            "schema_field_recall": _not_available("requires benchmark expected schema fields"),
            "critical_field_schema_recall": _not_available("requires benchmark expected critical fields"),
            "dynamic_field_discovery_rate": _rate(len(schema.field_definitions), max(len(plan.fields), len(schema.field_definitions))),
            "schema_overmerge_count": _issue_count(schema_audit, "schema_overmerged_field"),
            "schema_oversplit_count": _issue_count(schema_audit, "schema_oversplit_field"),
        },
        "boundary": {
            "long_text_field_count": len(long_text_fields),
            "possible_adhesion_finding_count": len(adhesion_findings),
            "possible_truncation_finding_count": len(truncation_findings),
            "adhesion_rate": _rate(len(adhesion_findings), len(long_text_fields)),
            "truncation_rate": _rate(len(truncation_findings), len(long_text_fields)),
            "boundary_confidence_avg": _confidence_avg(compiled_fields.values(), "boundary_confidence"),
            "long_text_boundary_f1": _not_available("requires benchmark expected char ranges"),
        },
        "evidence": {
            "field_count": field_count,
            "fields_with_evidence_count": len(fields_with_evidence),
            "evidence_count": len(evidence),
            "evidence_coverage_rate": _rate(len(fields_with_evidence), field_count),
            "bbox_available_evidence_count": len(bbox_available_evidence),
            "bbox_available_rate": _rate(len(bbox_available_evidence), len(evidence)),
            "critical_field_count": len(critical_fields),
            "critical_field_without_bbox_count": len(critical_without_bbox),
            "verified_without_evidence_count": len(verified_without_evidence),
            "ungrounded_value_count": len(no_guessing_failures),
            "bbox_integrity_failure_count": len(bbox_failures),
        },
        "coverage": {
            "section_text_coverage_rate": coverage.get("text_block_coverage_rate"),
            "important_region_coverage_rate": coverage.get("important_region_coverage_rate"),
            "span_coverage_rate": coverage_map.get("span_coverage_rate"),
            "anchor_coverage": structure_audit.get("anchor_coverage"),
            "anchor_count": coverage_map.get("anchor_count", 0),
            "missing_anchor_count": structure_audit.get("missing_anchor_count", coverage_map.get("missing_anchor_count", 0)),
            "sequence_gap_count": structure_audit.get("sequence_gap_count", 0),
            "group_issue_count": structure_audit.get("group_issue_count", 0),
            "table_issue_count": structure_audit.get("table_issue_count", 0),
            "required_prefix_issue_count": structure_audit.get("required_prefix_issue_count", 0),
            "container_duplicate_issue_count": structure_audit.get("container_duplicate_issue_count", 0),
            "agent_override_issue_count": structure_audit.get("agent_override_issue_count", 0),
            "duplicate_coverage_issue_count": structure_audit.get("duplicate_coverage_issue_count", coverage_map.get("duplicate_coverage_issue_count", 0)),
            "structure_blocking_issue_count": structure_blocking_issue_count,
        },
        "table": {
            "table_quality_status": table_quality_report.get("status"),
            "table_count": table_quality_report.get("table_count", 0),
            "issue_count": table_quality_report.get("issue_count", 0),
            "parser_agreement": table_quality_report.get("parser_agreement"),
        },
        "risk": {
            "high_risk_count": len(high_risks),
            "medium_risk_count": sum(1 for risk in risks if risk.risk_level == "medium"),
            "low_risk_count": sum(1 for risk in risks if risk.risk_level == "low"),
            "review_task_count": len(review_tasks),
            "high_risk_review_task_count": sum(1 for task in review_tasks if task.risk_level == "high" and task.required),
            "missing_item_count": missing_item_report.get("missing_count", 0),
            "output_contract_status": output_contract_validation_report.get("status"),
            "output_contract_failed_count": output_contract_failed_count,
        },
        "source": {
            "source_layer_status": source_layers.get("status"),
            "source_mode": source_layers.get("source_mode"),
            "source_issue_count": source_layers.get("source_issue_count", 0),
            "text_span_count": source_layers.get("text_quality", {}).get("total_text_span_count", 0),
            "source_bbox_coverage_rate": source_layers.get("text_quality", {}).get("bbox_coverage_rate"),
        },
        "repair": {
            "repair_status": repair_trace.get("status"),
            "round_count": repair_trace.get("round_count", 0),
            "attempt_count": repair_trace.get("attempt_count", 0),
            "applied_attempt_count": repair_trace.get("applied_attempt_count", 0),
            "final_audit_finding_count": repair_trace.get("final_audit_finding_count", 0),
            "recompiled_after_applied_repair": _repair_recompiled_when_applied(repair_trace),
        },
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _structure_blocking_issue_count(structure_audit: dict[str, Any], coverage_map: dict[str, Any]) -> int:
    return sum(
        int(value or 0)
        for value in (
            structure_audit.get("missing_anchor_count", coverage_map.get("missing_anchor_count", 0)),
            structure_audit.get("sequence_gap_count", 0),
            structure_audit.get("group_issue_count", 0),
            structure_audit.get("table_issue_count", 0),
            structure_audit.get("required_prefix_issue_count", 0),
            structure_audit.get("container_duplicate_issue_count", 0),
            structure_audit.get("agent_override_issue_count", 0),
            structure_audit.get("duplicate_coverage_issue_count", coverage_map.get("duplicate_coverage_issue_count", 0)),
        )
    )


def _confidence_avg(fields: Any, key: str) -> float | None:
    values = [
        float(value)
        for field in fields
        for value in [field.confidence.get(key)]
        if value is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _failed_checks(validation: list[dict[str, Any]], check_type: str) -> list[dict[str, Any]]:
    return [check for check in validation if check.get("check_type") == check_type and check.get("result") == "failed"]


def _issue_count(report: dict[str, Any], issue_type: str) -> int:
    return sum(1 for issue in report.get("issues", []) if issue.get("issue_type") == issue_type)


def _not_available(reason: str) -> dict[str, str]:
    return {
        "status": "not_available",
        "reason": reason,
    }


def _repair_trace_findings(repair_trace: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for round_record in repair_trace.get("rounds", []):
        if not isinstance(round_record, dict):
            continue
        for finding in round_record.get("audit_findings", []):
            if isinstance(finding, dict):
                findings.append(finding)
    return findings


def _repair_recompiled_when_applied(repair_trace: dict[str, Any]) -> bool:
    for round_record in repair_trace.get("rounds", []):
        attempts = round_record.get("attempts", []) if isinstance(round_record, dict) else []
        if any(isinstance(attempt, dict) and attempt.get("status") == "applied" for attempt in attempts):
            if not round_record.get("compiled_after_repair"):
                return False
    return True
