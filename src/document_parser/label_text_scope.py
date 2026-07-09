from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import ExtractionPlan, FieldPlan, TextSpan, to_jsonable
from .utils import stable_id


REFERENCE_RESOURCE = "label_text_scope_reference_v0.1.json"
IN_SCOPE_STATUS = "in_scope_label_text"
OUT_OF_SCOPE_STATUS = "out_of_scope_noise"
UNKNOWN_SCOPE_STATUS = "unknown_scope"


def load_label_text_scope_reference() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "resources" / REFERENCE_RESOURCE
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_label_text_scope_agent_context(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_version": "label_text_scope_agent_context_v0.1",
        "reference_version": reference.get("reference_version"),
        "primary_rule": reference.get("scope_policy", {}).get("primary_rule"),
        "primary_rule_zh": reference.get("scope_policy", {}).get("primary_rule_zh"),
        "reference_is_not_evidence": bool(reference.get("scope_policy", {}).get("reference_is_not_evidence")),
        "template_placeholders_are_not_values": bool(reference.get("scope_policy", {}).get("template_placeholders_are_not_values")),
        "non_template_printed_text_policy": reference.get("scope_policy", {}).get("non_template_printed_text_policy"),
        "in_scope_categories": _category_summaries(reference.get("in_scope_categories")),
        "out_of_scope_categories": _category_summaries(reference.get("out_of_scope_categories")),
        "field_catalog": _catalog_summaries(reference.get("field_catalog"), ["semantic_key", "display_name", "category_id", "aliases", "criticality"]),
        "entity_catalog": _catalog_summaries(reference.get("entity_catalog"), ["entity_type", "display_name", "category_id", "roles", "fields"]),
        "table_catalog": _catalog_summaries(reference.get("table_catalog"), ["table_type", "display_name", "category_id", "columns", "canonical_rows"]),
        "scope_instructions": [
            "Only final printed packaging label text can become extracted_data.",
            "Do not treat this reference as evidence.",
            "Do not emit template placeholders as final PDF values.",
            "If a text is not covered by the template but looks printed on packaging, keep it as other_printed_label_text or mark unknown_scope.",
            "If unsure whether a node is printed label text, put its node_id in unknown_nodes.",
        ],
    }


def build_label_text_scope_report(
    *,
    reference: dict[str, Any],
    node_scope_decisions: list[dict[str, Any]],
    rejected_items: list[dict[str, Any]],
    validation_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    ignored_noise = [item for item in node_scope_decisions if item.get("scope_status") == OUT_OF_SCOPE_STATUS]
    unknown_scope = [item for item in node_scope_decisions if item.get("scope_status") == UNKNOWN_SCOPE_STATUS]
    failed_checks = [check for check in validation_checks if check.get("result") == "failed"]
    extracted_out_of_scope_count = 0
    status = "review_required" if rejected_items or unknown_scope or failed_checks else "pass"
    return {
        "report_version": "label_text_scope_report_v0.1",
        "reference_version": reference.get("reference_version"),
        "status": status,
        "extracted_out_of_scope_count": extracted_out_of_scope_count,
        "ignored_noise_node_count": len(ignored_noise),
        "unknown_scope_node_count": len(unknown_scope),
        "scope_gate_rejected_count": len(rejected_items),
        "node_scope_decision_count": len(node_scope_decisions),
        "node_scope_decisions": node_scope_decisions[:200],
        "ignored_noise_nodes": ignored_noise[:100],
        "unknown_scope_nodes": unknown_scope[:100],
        "rejected_items": rejected_items,
        "checks": validation_checks,
    }


def apply_label_text_scope_gate(
    plan: ExtractionPlan,
    spans: list[TextSpan],
    reference: dict[str, Any],
    agent_bodies: list[dict[str, Any] | None] | None = None,
) -> tuple[ExtractionPlan, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    span_by_id = {span.span_id: span for span in spans}
    agent_decisions = _agent_scope_decisions(agent_bodies or [], span_by_id)
    decisions_by_node = {str(item.get("node_id")): item for item in agent_decisions if item.get("node_id")}
    rejected: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    kept_fields: list[FieldPlan] = []
    node_scope_decisions: list[dict[str, Any]] = list(agent_decisions)
    decision_keys = {str(item.get("node_id")) for item in node_scope_decisions if item.get("node_id")}
    ignored_nodes = set(plan.ignored_nodes)
    unknown_nodes = set(plan.unknown_nodes)
    ignored_node_reasons = dict(getattr(plan, "ignored_node_reasons", {}) or {})

    for node_id in sorted(ignored_nodes):
        if node_id in decision_keys:
            continue
        span = span_by_id.get(node_id)
        decision = _node_decision(
            node_id=node_id,
            span=span,
            scope_status=OUT_OF_SCOPE_STATUS if span and _text_out_of_scope(span.text, reference) else "ignored",
            scope_category=_scope_category_for_text(span.text if span else "", reference),
            reason=ignored_node_reasons.get(node_id, "agent_marked_ignored"),
            confidence=0.75,
            decided_by="agent" if node_id in plan.ignored_nodes else "rule",
        )
        node_scope_decisions.append(decision)
        decision_keys.add(node_id)
        ignored_node_reasons.setdefault(node_id, decision["reason"])

    for node_id in sorted(unknown_nodes):
        if node_id in decision_keys:
            continue
        span = span_by_id.get(node_id)
        node_scope_decisions.append(
            _node_decision(
                node_id=node_id,
                span=span,
                scope_status=UNKNOWN_SCOPE_STATUS,
                scope_category=_scope_category_for_text(span.text if span else "", reference),
                reason="agent_marked_unknown_scope",
                confidence=0.5,
                decided_by="agent",
            )
        )
        decision_keys.add(node_id)

    for field in plan.fields:
        field_issues: list[dict[str, Any]] = []
        field_unknown = False
        hard_reject = False
        for span_range in field.value_source.ranges:
            span = span_by_id.get(span_range.span_id)
            if span is None:
                continue
            extracted_text = span.text[span_range.start_offset : span_range.end_offset]
            agent_decision = decisions_by_node.get(span_range.span_id)
            rule_status = _classify_text(extracted_text, reference)
            scope_status = str(agent_decision.get("scope_status")) if agent_decision else rule_status
            scope_category = str((agent_decision or {}).get("scope_category") or _scope_category_for_text(extracted_text, reference))
            if scope_status == OUT_OF_SCOPE_STATUS or rule_status == OUT_OF_SCOPE_STATUS:
                hard_reject = True
                issue = {
                    "issue_type": "label_text_scope_out_of_scope_candidate",
                    "span_id": span_range.span_id,
                    "scope_status": OUT_OF_SCOPE_STATUS,
                    "scope_category": scope_category,
                    "text": extracted_text,
                }
                field_issues.append(issue)
                if span_range.span_id not in decision_keys:
                    node_scope_decisions.append(
                        _node_decision(
                            node_id=span_range.span_id,
                            span=span,
                            scope_status=OUT_OF_SCOPE_STATUS,
                            scope_category=scope_category,
                            reason="scope_gate_rejected_out_of_scope_candidate",
                            confidence=0.95,
                            decided_by="rule",
                        )
                    )
                    decision_keys.add(span_range.span_id)
                ignored_nodes.add(span_range.span_id)
                ignored_node_reasons[span_range.span_id] = "scope_gate_rejected_out_of_scope_candidate"
                continue
            if scope_status == UNKNOWN_SCOPE_STATUS or rule_status == UNKNOWN_SCOPE_STATUS:
                field_unknown = True
                field_issues.append(
                    {
                        "issue_type": "label_text_scope_unknown",
                        "span_id": span_range.span_id,
                        "scope_status": UNKNOWN_SCOPE_STATUS,
                        "scope_category": scope_category,
                        "text": extracted_text,
                    }
                )
                unknown_nodes.add(span_range.span_id)
                if span_range.span_id not in decision_keys:
                    node_scope_decisions.append(
                        _node_decision(
                            node_id=span_range.span_id,
                            span=span,
                            scope_status=UNKNOWN_SCOPE_STATUS,
                            scope_category=scope_category,
                            reason="scope_gate_marked_unknown",
                            confidence=0.5,
                            decided_by="rule",
                        )
                    )
                    decision_keys.add(span_range.span_id)

        if hard_reject:
            rejected.append(
                {
                    "field_plan_id": field.field_plan_id,
                    "semantic_key": field.semantic_key,
                    "reason": "label_text_scope_out_of_scope",
                    "issues": field_issues,
                }
            )
            checks.append(
                {
                    "validation_id": stable_id("val_label_text_scope", len(checks) + 1),
                    "target_id": field.field_plan_id,
                    "check_type": "label_text_scope_gate",
                    "result": "failed",
                    "severity": "high",
                    "message": "Label text scope gate rejected an out-of-scope field candidate.",
                    "semantic_key": field.semantic_key,
                    "issues": field_issues,
                }
            )
            continue

        if field_unknown:
            review_items.append(
                {
                    "field_plan_id": field.field_plan_id,
                    "semantic_key": field.semantic_key,
                    "reason": "label_text_scope_unknown",
                    "issues": field_issues,
                }
            )
            checks.append(
                {
                    "validation_id": stable_id("val_label_text_scope", len(checks) + 1),
                    "target_id": field.field_plan_id,
                    "check_type": "label_text_scope_unknown",
                    "result": "failed",
                    "severity": "high",
                    "message": "Label text scope gate could not confirm whether a field is final printed label text.",
                    "semantic_key": field.semantic_key,
                    "issues": field_issues,
                }
            )
        else:
            checks.append(
                {
                    "validation_id": stable_id("val_label_text_scope", len(checks) + 1),
                    "target_id": field.field_plan_id,
                    "check_type": "label_text_scope_gate",
                    "result": "passed",
                    "severity": "info",
                    "message": "Label text scope gate accepted the field candidate.",
                    "semantic_key": field.semantic_key,
                    "issues": [],
                }
            )
        kept_fields.append(field)

    if any(item.get("scope_status") == UNKNOWN_SCOPE_STATUS for item in node_scope_decisions) and not any(
        check.get("check_type") == "label_text_scope_unknown" for check in checks
    ):
        checks.append(
            {
                "validation_id": stable_id("val_label_text_scope", len(checks) + 1),
                "target_id": "document",
                "check_type": "label_text_scope_unknown",
                "result": "failed",
                "severity": "high",
                "message": "Label text scope gate found nodes with unknown printed-label scope.",
                "unknown_scope_node_count": sum(1 for item in node_scope_decisions if item.get("scope_status") == UNKNOWN_SCOPE_STATUS),
            }
        )

    if not checks:
        checks.append(
            {
                "validation_id": stable_id("val_label_text_scope", 1),
                "target_id": "document",
                "check_type": "label_text_scope_reference",
                "result": "passed",
                "severity": "info",
                "message": "Label text scope reference loaded and no field candidates required scope rejection.",
                "reference_version": reference.get("reference_version"),
            }
        )

    scoped_plan = replace(
        plan,
        fields=kept_fields,
        ignored_nodes=sorted(ignored_nodes),
        unknown_nodes=sorted(unknown_nodes),
        ignored_node_reasons=ignored_node_reasons,
    )
    report = build_label_text_scope_report(
        reference=reference,
        node_scope_decisions=node_scope_decisions,
        rejected_items=rejected,
        validation_checks=checks,
    )
    return scoped_plan, rejected, review_items, checks, report


def _category_summaries(value: Any) -> list[dict[str, Any]]:
    return _catalog_summaries(value, ["category_id", "display_name", "description"])


def _catalog_summaries(value: Any, keys: list[str]) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    return [
        {key: item.get(key) for key in keys if key in item}
        for item in items
        if isinstance(item, dict)
    ]


def _agent_scope_decisions(agent_bodies: list[dict[str, Any] | None], span_by_id: dict[str, TextSpan]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for body in agent_bodies:
        if not isinstance(body, dict):
            continue
        for item in _scope_decision_items(body):
            node_id = str(item.get("node_id") or item.get("span_id") or item.get("source_span_id") or "")
            if not node_id or node_id in seen:
                continue
            span = span_by_id.get(node_id)
            scope_status = str(item.get("scope_status") or UNKNOWN_SCOPE_STATUS)
            if scope_status not in {IN_SCOPE_STATUS, OUT_OF_SCOPE_STATUS, UNKNOWN_SCOPE_STATUS, "ignored"}:
                scope_status = UNKNOWN_SCOPE_STATUS
            decisions.append(
                _node_decision(
                    node_id=node_id,
                    span=span,
                    scope_status=scope_status,
                    scope_category=str(item.get("scope_category") or ""),
                    reason=str(item.get("reason") or "agent_scope_decision"),
                    confidence=_float_or_default(item.get("confidence"), 0.7),
                    decided_by="agent",
                )
            )
            seen.add(node_id)
    return decisions


def _scope_decision_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    items = value.get("node_scope_decisions")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    found: list[dict[str, Any]] = []
    for nested in value.values():
        if isinstance(nested, dict):
            found.extend(_scope_decision_items(nested))
    return found


def _node_decision(
    *,
    node_id: str,
    span: TextSpan | None,
    scope_status: str,
    scope_category: str,
    reason: str,
    confidence: float,
    decided_by: str,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "scope_status": scope_status,
        "scope_category": scope_category or "unknown",
        "reason": reason,
        "confidence": confidence,
        "decided_by": decided_by,
        "page": span.page if span else None,
        "text": span.text if span else "",
        "bbox_normalized": to_jsonable(span.bbox_normalized) if span else None,
    }


def _classify_text(text: str, reference: dict[str, Any]) -> str:
    stripped = text.strip()
    if not stripped:
        return OUT_OF_SCOPE_STATUS
    if _text_out_of_scope(stripped, reference):
        return OUT_OF_SCOPE_STATUS
    if _text_unknown_scope(stripped, reference):
        return UNKNOWN_SCOPE_STATUS
    return IN_SCOPE_STATUS


def _text_out_of_scope(text: str, reference: dict[str, Any]) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in set(str(token) for token in reference.get("placeholder_tokens", [])):
        return True
    if stripped.startswith("示例"):
        return True
    return any(str(keyword) and str(keyword) in stripped for keyword in reference.get("hard_noise_keywords", []))


def _text_unknown_scope(text: str, reference: dict[str, Any]) -> bool:
    stripped = text.strip()
    return any(str(keyword) and str(keyword) in stripped for keyword in reference.get("unknown_scope_keywords", []))


def _scope_category_for_text(text: str, reference: dict[str, Any]) -> str:
    if _text_out_of_scope(text, reference):
        if text.strip() in set(str(token) for token in reference.get("placeholder_tokens", [])) or text.strip().startswith("示例"):
            return "template_placeholder"
        if any(keyword in text for keyword in ("审稿", "校对", "版本记录", "修改记录", "修订记录", "内部备注")):
            return "review_or_revision_note"
        return "artwork_technical_note"
    if "营养" in text or "NRV" in text:
        return "nutrition_label_text"
    if any(keyword in text for keyword in ("委托方", "受委托方", "生产者", "经销商", "进口商", "地址", "许可证")):
        return "business_operator_label_text"
    if any(keyword in text for keyword in ("内容物", "C1", "C2", "C3", "组合装", "礼盒")):
        return "content_item_label_text"
    if any(keyword in text for keyword in ("条码", "回收", "储运", "认证")):
        return "barcode_and_mark_label_text"
    for item in reference.get("field_catalog", []):
        if not isinstance(item, dict):
            continue
        aliases = [str(alias) for alias in item.get("aliases", [])]
        if any(alias and alias in text for alias in aliases):
            return str(item.get("category_id") or "main_label_text")
    return "other_printed_label_text"


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
