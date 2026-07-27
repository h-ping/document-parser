from __future__ import annotations

from typing import Any

from .agent_candidates import agent_field_items


class GlobalReconciliationError(RuntimeError):
    pass


def validate_and_finalize_reconciliation(
    proposals: dict[str, Any],
    reconciled: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal_fields = agent_field_items(proposals)
    reconciled_fields = agent_field_items(reconciled)
    allowed_ranges_by_key: dict[str, set[tuple[str, int, int, str]]] = {}
    for field in proposal_fields:
        semantic_key = str(field.get("semantic_key") or "")
        allowed_ranges_by_key.setdefault(semantic_key, set()).update(_field_ranges(field))
    invalid_fields: list[dict[str, Any]] = []
    for field in reconciled_fields:
        semantic_key = str(field.get("semantic_key") or "")
        field_ranges = _field_ranges(field)
        allowed_ranges = allowed_ranges_by_key.get(semantic_key, set())
        invalid_ranges = [signature for signature in field_ranges if signature not in allowed_ranges]
        if not field_ranges or invalid_ranges:
            invalid_fields.append(
                {
                    "semantic_key": semantic_key,
                    "reason": "semantic_key_or_range_not_in_proposals",
                    "invalid_ranges": [list(signature) for signature in invalid_ranges],
                }
            )
    if invalid_fields:
        raise GlobalReconciliationError(
            f"Global reconciliation produced {len(invalid_fields)} field(s) outside the proposal evidence envelope."
        )

    finalized = {
        **reconciled,
        "tables": proposals.get("tables", []),
        "requirements": proposals.get("requirements", []),
        "layout_candidate_decisions": proposals.get("layout_candidate_decisions", []),
        "node_scope_decisions": proposals.get("node_scope_decisions", []),
        "ignored_nodes": _unique([*proposals.get("ignored_nodes", []), *reconciled.get("ignored_nodes", [])]),
        "unknown_nodes": _unique([*proposals.get("unknown_nodes", []), *reconciled.get("unknown_nodes", [])]),
    }
    report = {
        "artifact_version": "global_reconciliation_report_v0.1",
        "status": "applied",
        "input_field_count": len(proposal_fields),
        "output_field_count": len(reconciled_fields),
        "removed_field_count": max(len(proposal_fields) - len(reconciled_fields), 0),
        "input_entity_count": len(proposals.get("entities", [])),
        "output_entity_count": len(finalized.get("entities", [])),
        "evidence_envelope_valid": True,
        "tables_preserved": True,
    }
    return finalized, report


def disabled_reconciliation_report(reason: str, proposals: dict[str, Any] | None = None) -> dict[str, Any]:
    body = proposals or {}
    return {
        "artifact_version": "global_reconciliation_report_v0.1",
        "status": "disabled",
        "reason": reason,
        "input_field_count": len(body.get("fields", [])),
        "output_field_count": len(body.get("fields", [])),
        "removed_field_count": 0,
        "input_entity_count": len(body.get("entities", [])),
        "output_entity_count": len(body.get("entities", [])),
        "evidence_envelope_valid": True,
        "tables_preserved": True,
    }


def _field_ranges(field: dict[str, Any]) -> list[tuple[str, int, int, str]]:
    ranges = field.get("ranges")
    if isinstance(ranges, list) and ranges:
        return [
            (
                str(span_range.get("span_id") or ""),
                int(span_range.get("start_offset", -1)),
                int(span_range.get("end_offset", -1)),
                str(span_range.get("text") or ""),
            )
            for span_range in ranges
            if isinstance(span_range, dict)
        ]
    if field.get("span_id"):
        return [
            (
                str(field.get("span_id")),
                int(field.get("start_offset", -1)),
                int(field.get("end_offset", -1)),
                str(field.get("text") or ""),
            )
        ]
    return []


def _unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
