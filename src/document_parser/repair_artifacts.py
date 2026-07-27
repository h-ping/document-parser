from __future__ import annotations

from typing import Any

from .models import ExtractionPlan, to_jsonable
from .utils import stable_id


def build_repair_trace_artifact(rounds: list[dict[str, Any]], max_repair_rounds: int) -> dict[str, Any]:
    final_round = rounds[-1] if rounds else {}
    attempts = [
        attempt
        for round_record in rounds
        for attempt in round_record.get("attempts", [])
        if isinstance(attempt, dict)
    ]
    applied_attempt_count = sum(1 for attempt in attempts if attempt.get("status") == "applied")
    skipped_attempt_count = sum(1 for attempt in attempts if attempt.get("status") == "skipped")
    final_audit_finding_count = int(final_round.get("audit_finding_count", 0)) if rounds else 0
    return {
        "status": "pass" if final_audit_finding_count == 0 else "review_required",
        "max_repair_rounds": max_repair_rounds,
        "round_count": len(rounds),
        "attempt_count": len(attempts),
        "applied_attempt_count": applied_attempt_count,
        "skipped_attempt_count": skipped_attempt_count,
        "final_audit_finding_count": final_audit_finding_count,
        "final_round_status": final_round.get("status"),
        "validation_after_repair": "final_pipeline_validation",
        "rounds": rounds,
    }


def build_repair_attempts_artifact(repair_attempts: list[dict[str, Any]], repair_plan: dict[str, Any]) -> dict[str, Any]:
    actions = repair_plan.get("actions", [])
    if repair_attempts:
        status = "attempted"
        attempts = repair_attempts
    elif actions:
        status = "pending_agent_repair"
        attempts = [
            {
                "attempt_id": stable_id("repair_attempt", index),
                "action_id": action.get("action_id"),
                "issue_type": action.get("issue_type"),
                "target_type": action.get("target_type"),
                "target_id": action.get("target_id"),
                "status": "skipped",
                "reason": "requires_agent_patch_not_available_in_current_round",
                "acceptance_gate": action.get("acceptance_gate"),
            }
            for index, action in enumerate(actions, start=1)
        ]
    else:
        status = "not_needed"
        attempts = []

    return {
        "repair_mode": repair_plan.get("repair_mode", "execute_plan"),
        "status": status,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }


def build_repair_agent_candidates(repair_plan: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for index, action in enumerate(repair_plan.get("actions", []), start=1):
        candidates.append(
            {
                "candidate_id": stable_id("repair_candidate", index),
                "action_id": action.get("action_id"),
                "target_type": action.get("target_type"),
                "target_id": action.get("target_id"),
                "issue_type": action.get("issue_type"),
                "recommended_agent": action.get("recommended_agent"),
                "expected_output": action.get("expected_output"),
                "acceptance_gate": action.get("acceptance_gate"),
                "candidate_template": {
                    "items": [],
                    "groups": [],
                    "tables": [],
                    "lists": [],
                },
                "constraints": [
                    "values_must_be_copied_from_source_spans",
                    "include_span_id_and_char_offsets",
                    "do_not_overwrite_rule_candidates_without_evidence",
                ],
            }
        )

    return {
        "status": "pending_agent_fill" if candidates else "not_needed",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_repaired_source_layers(source_layers: dict[str, Any], repair_attempts: dict[str, Any]) -> dict[str, Any]:
    repair_applied = any(attempt.get("status") == "applied" for attempt in repair_attempts.get("attempts", []))
    repair_attempted = repair_attempts.get("status") == "attempted"
    return {
        "status": "source_layers_not_modified_plan_repair_applied" if repair_applied else "not_modified",
        "repair_applied": repair_applied,
        "reason": "extraction_plan_repaired_and_recompiled" if repair_applied else "no_source_layer_patch_available",
        "repair_attempted": repair_attempted,
        "base_source_layer_status": source_layers.get("status"),
        "base_source_issue_count": source_layers.get("source_issue_count", 0),
        "source_issues": source_layers.get("source_issues", []),
    }


def build_repair_plan_patches(
    before_plan: ExtractionPlan,
    after_plan: ExtractionPlan,
    repair_attempts: dict[str, Any],
) -> dict[str, Any]:
    before_fields = {field.field_plan_id: field for field in before_plan.fields}
    after_fields = {field.field_plan_id: field for field in after_plan.fields}
    attempts_by_field_plan_id = _attempts_by_field_plan_id(repair_attempts)
    patches: list[dict[str, Any]] = []

    for field_plan_id, before_field in before_fields.items():
        after_field = after_fields.get(field_plan_id)
        if after_field is None:
            patches.append(
                {
                    "patch_id": stable_id("plan_patch", len(patches) + 1),
                    "operation": "remove_field_plan",
                    "field_plan_id": field_plan_id,
                    "semantic_key": before_field.semantic_key,
                    "previous_value_source": to_jsonable(before_field.value_source),
                    "new_value_source": None,
                    "reason": "field_plan_removed_by_repair",
                    "attempts": attempts_by_field_plan_id.get(field_plan_id, []),
                }
            )
            continue
        if before_field.value_source != after_field.value_source:
            patches.append(
                {
                    "patch_id": stable_id("plan_patch", len(patches) + 1),
                    "operation": "adjust_field_boundary",
                    "field_plan_id": field_plan_id,
                    "semantic_key": before_field.semantic_key,
                    "display_name": before_field.display_name,
                    "previous_value_source": to_jsonable(before_field.value_source),
                    "new_value_source": to_jsonable(after_field.value_source),
                    "previous_boundary": to_jsonable(before_field.boundary),
                    "new_boundary": to_jsonable(after_field.boundary),
                    "reason": after_field.boundary.get("repair") or "value_source_changed_by_repair",
                    "attempts": attempts_by_field_plan_id.get(field_plan_id, []),
                }
            )

    for field_plan_id, after_field in after_fields.items():
        if field_plan_id in before_fields:
            continue
        patches.append(
            {
                "patch_id": stable_id("plan_patch", len(patches) + 1),
                "operation": "add_missing_field",
                "field_plan_id": field_plan_id,
                "semantic_key": after_field.semantic_key,
                "display_name": after_field.display_name,
                "previous_value_source": None,
                "new_value_source": to_jsonable(after_field.value_source),
                "reason": "field_plan_added_by_repair",
                "attempts": attempts_by_field_plan_id.get(field_plan_id, []),
            }
        )

    return {
        "artifact_version": "repair_plan_patches_v0.1",
        "status": "applied" if patches else "no_plan_change",
        "source_plan_id": before_plan.plan_id,
        "target_plan_id": after_plan.plan_id,
        "patch_count": len(patches),
        "patches": patches,
        "constraints": [
            "patches_modify_extraction_plan_only",
            "final_values_are_recompiled_by_deterministic_compiler",
            "patch_values_must_reference_source_spans_or_table_cells",
        ],
    }


def _attempts_by_field_plan_id(repair_attempts: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for attempt in repair_attempts.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        field_plan_id = attempt.get("details", {}).get("target_field_plan_id")
        if isinstance(field_plan_id, str):
            grouped.setdefault(field_plan_id, []).append(attempt)
    return grouped
