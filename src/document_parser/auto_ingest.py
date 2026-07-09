from __future__ import annotations

from typing import Any


def build_auto_ingest_candidates(
    standard_items: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    risks: list[Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    failed_checks_by_target = _failed_checks_by_target(validation)
    risks_by_target = _risks_by_target(risks)
    candidates = []
    blocked_items = []

    for item in standard_items:
        block_reasons = _block_reasons(item, failed_checks_by_target, risks_by_target)
        if block_reasons:
            blocked_items.append(
                {
                    "standard_item_id": item.get("id"),
                    "field_id": item.get("field_id"),
                    "semantic_key": item.get("semantic_key"),
                    "field": item.get("field"),
                    "label": item.get("label"),
                    "status": "blocked",
                    "block_reasons": block_reasons,
                }
            )
            continue
        candidates.append(
            {
                "candidate_id": f"auto_ingest_{len(candidates) + 1:04d}",
                "standard_item_id": item.get("id"),
                "field_id": item.get("field_id"),
                "semantic_key": item.get("semantic_key"),
                "field": item.get("field"),
                "label": item.get("label"),
                "text": item.get("text"),
                "normalized_text": item.get("normalized_text"),
                "confidence": item.get("confidence"),
                "source": item.get("source"),
                "evidence_refs": item.get("evidence_refs", []),
                "ingest_status": "candidate",
                "reason": "field_passed_item_level_checks",
            }
        )

    document_allowed = bool(quality.get("auto_ingest_allowed")) and int(quality.get("high_risk_count", 0)) == 0
    return {
        "status": "ready" if document_allowed else "blocked_by_document_quality",
        "document_auto_ingest_allowed": document_allowed,
        "candidate_count": len(candidates),
        "blocked_count": len(blocked_items),
        "quality_snapshot": {
            "overall_status": quality.get("overall_status"),
            "high_risk_count": quality.get("high_risk_count", 0),
            "medium_risk_count": quality.get("medium_risk_count", 0),
            "low_risk_count": quality.get("low_risk_count", 0),
            "auto_ingest_allowed": quality.get("auto_ingest_allowed", False),
        },
        "candidates": candidates,
        "blocked_items": blocked_items,
    }


def _block_reasons(
    item: dict[str, Any],
    failed_checks_by_target: dict[str, list[dict[str, Any]]],
    risks_by_target: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    reasons = []
    field_id = str(item.get("field_id") or "")
    if not item.get("comparison_required", True):
        reasons.append({"reason": "comparison_not_required"})
    if item.get("review_required"):
        reasons.append({"reason": "review_required"})
    if item.get("status") not in {"verified", "normalized", "compiled"}:
        reasons.append({"reason": "field_status_not_ingestable", "status": item.get("status")})
    if not item.get("evidence_refs"):
        reasons.append({"reason": "missing_evidence_refs"})
    if _confidence(item) < 0.95:
        reasons.append({"reason": "confidence_below_threshold", "confidence": item.get("confidence"), "threshold": 0.95})
    for check in failed_checks_by_target.get(field_id, []):
        reasons.append({"reason": "validation_failed", "check_type": check.get("check_type"), "validation_id": check.get("validation_id")})
    for risk in risks_by_target.get(field_id, []):
        reasons.append({"reason": "risk_present", "risk_type": getattr(risk, "risk_type", None), "risk_level": getattr(risk, "risk_level", None)})
    return reasons


def _confidence(item: dict[str, Any]) -> float:
    try:
        return float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _failed_checks_by_target(validation: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for check in validation:
        if check.get("result") == "passed":
            continue
        target_id = check.get("target_id")
        if target_id:
            grouped.setdefault(str(target_id), []).append(check)
    return grouped


def _risks_by_target(risks: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for risk in risks:
        target_type = getattr(risk, "target_type", "")
        if target_type != "field" or getattr(risk, "risk_level", "") not in {"high", "medium"}:
            continue
        grouped.setdefault(str(getattr(risk, "target_id", "")), []).append(risk)
    return grouped
