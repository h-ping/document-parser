from __future__ import annotations

from typing import Any

from .agents import CANONICAL_PATTERNS, _first_matching_label
from .models import GeneratedSchema, TextSpan
from .structures import CONTENT_ITEM_RE
from .utils import stable_id


def build_schema_audit(schema: GeneratedSchema, spans: list[TextSpan], regions: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    field_semantic_keys = [definition.semantic_key for definition in schema.field_definitions]
    field_semantic_key_set = set(field_semantic_keys)
    table_types = {str(definition.get("table_type")) for definition in schema.table_definitions if isinstance(definition, dict)}
    entity_types = {str(definition.get("entity_type")) for definition in schema.entity_types if isinstance(definition, dict)}
    section_types = {str(section.get("section_type")) for section in schema.sections if isinstance(section, dict)}

    for semantic_key in sorted({key for key in field_semantic_keys if field_semantic_keys.count(key) > 1}):
        _add_issue(
            issues,
            issue_type="schema_duplicate_field_definition",
            severity="high",
            message="generated_schema contains duplicate field definitions.",
            source={"semantic_key": semantic_key},
            expected={"unique_semantic_key": semantic_key},
            actual={"duplicate_count": field_semantic_keys.count(semantic_key)},
            repair_hint="Regenerate schema with one field_definition per semantic_key.",
        )

    for span in spans:
        text = span.text.strip()
        if not text:
            continue
        for pattern in CANONICAL_PATTERNS:
            if pattern.semantic_key in field_semantic_key_set:
                continue
            label = _matching_label(text, pattern.labels)
            if not label:
                continue
            _add_issue(
                issues,
                issue_type="schema_obvious_field_missing",
                severity="high" if pattern.criticality == "critical" else "medium",
                message="Source text contains an obvious field anchor missing from generated_schema.",
                source={"span_id": span.span_id, "page": span.page, "text": span.text, "label": label},
                expected={"semantic_key": pattern.semantic_key, "display_name": pattern.display_name},
                actual={"defined_semantic_keys": sorted(field_semantic_key_set)},
                repair_hint="Regenerate schema or add the missing field definition before extraction.",
            )

    if any("营养成分表" in span.text for span in spans) and "nutrition_facts" not in table_types:
        first_span = next(span for span in spans if "营养成分表" in span.text)
        _add_issue(
            issues,
            issue_type="schema_table_definition_missing",
            severity="high",
            message="Source text contains a nutrition table anchor missing from table_definitions.",
            source={"span_id": first_span.span_id, "page": first_span.page, "text": first_span.text},
            expected={"table_type": "nutrition_facts"},
            actual={"defined_table_types": sorted(table_types)},
            repair_hint="Add a nutrition_facts table definition before table/list extraction.",
        )

    if any(CONTENT_ITEM_RE.match(span.text.strip()) for span in spans) and "content_item" not in entity_types:
        first_span = next(span for span in spans if CONTENT_ITEM_RE.match(span.text.strip()))
        _add_issue(
            issues,
            issue_type="schema_repeatable_entity_missing",
            severity="high",
            message="Source text contains content item anchors but generated_schema lacks content_item entity type.",
            source={"span_id": first_span.span_id, "page": first_span.page, "text": first_span.text},
            expected={"entity_type": "content_item", "repeatable": True},
            actual={"defined_entity_types": sorted(entity_types)},
            repair_hint="Add repeatable content_item entity type and child field definitions.",
        )

    for required_region_type in sorted({region.get("region_type") for region in regions if str(region.get("region_type", "")).startswith("revision_")}):
        if required_region_type not in section_types:
            _add_issue(
                issues,
                issue_type="schema_revision_section_missing",
                severity="medium",
                message="Detected revision region is missing from generated_schema.sections.",
                source={"region_type": required_region_type},
                expected={"section_type": required_region_type},
                actual={"defined_section_types": sorted(section_types)},
                repair_hint="Add generated schema section for the detected revision region.",
            )

    blocking_count = sum(1 for issue in issues if issue["severity"] in {"high", "medium"})
    return {
        "schema_audit_version": "schema_audit_v0.1",
        "status": "review_required" if blocking_count else "pass",
        "blocking_issue_count": blocking_count,
        "issue_count": len(issues),
        "issues": issues,
        "checks": [
            _check("unique_field_definitions", not any(issue["issue_type"] == "schema_duplicate_field_definition" for issue in issues)),
            _check("obvious_field_anchor_coverage", not any(issue["issue_type"] == "schema_obvious_field_missing" for issue in issues)),
            _check("table_definition_coverage", not any(issue["issue_type"] == "schema_table_definition_missing" for issue in issues)),
            _check("repeatable_entity_coverage", not any(issue["issue_type"] == "schema_repeatable_entity_missing" for issue in issues)),
            _check("revision_section_coverage", not any(issue["issue_type"] == "schema_revision_section_missing" for issue in issues)),
        ],
        "summary": {
            "field_definition_count": len(schema.field_definitions),
            "table_definition_count": len(schema.table_definitions),
            "entity_type_count": len(schema.entity_types),
            "section_count": len(schema.sections),
        },
    }


def _matching_label(text: str, labels: tuple[str, ...]) -> str | None:
    return _first_matching_label(text, labels)


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    issue_type: str,
    severity: str,
    message: str,
    source: dict[str, Any],
    expected: dict[str, Any],
    actual: dict[str, Any],
    repair_hint: str,
) -> None:
    issues.append(
        {
            "issue_id": stable_id("schema_audit_issue", len(issues) + 1),
            "issue_type": issue_type,
            "severity": severity,
            "message": message,
            "source": source,
            "expected": expected,
            "actual": actual,
            "repair_hint": repair_hint,
        }
    )


def _check(check_type: str, passed: bool) -> dict[str, Any]:
    return {
        "check_type": check_type,
        "result": "passed" if passed else "failed",
    }
