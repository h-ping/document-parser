from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import ExtractionPlan, FieldPlan, GeneratedSchema, SpanRange, TextSpan, ValueSource
from .utils import stable_id


class AgentCandidateError(RuntimeError):
    pass


def build_agent_extraction_plan(
    schema: GeneratedSchema,
    spans: list[TextSpan],
    body: dict[str, Any],
    *,
    plan_id: str = "plan_agent_001",
) -> tuple[ExtractionPlan, list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(body, dict):
        raise AgentCandidateError("Agent extraction plan must be a JSON object.")
    body = _agent_plan_body(body)
    items = _agent_items(body)
    span_by_id = {span.span_id: span for span in spans}
    field_def_by_id = {definition.field_def_id: definition for definition in schema.field_definitions}
    fields: list[FieldPlan] = []
    rejected: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    known_signatures: set[tuple[str, str | None, tuple[tuple[str, int, int], ...]]] = set()

    for index, item in enumerate(items, start=1):
        field_plan = _field_plan_from_agent_item(
            index=index,
            item=item,
            schema=schema,
            span_by_id=span_by_id,
            field_def_by_id=field_def_by_id,
            field_number=len(fields) + 1,
            candidate_source="llm_extraction_plan",
            duplicate_signatures=known_signatures,
        )
        if field_plan.rejection is not None:
            rejected.append(field_plan.rejection)
            continue
        fields.append(field_plan.field)
        known_signatures.add(_field_signature(field_plan.field))
        if field_plan.review_item is not None:
            review_items.append(field_plan.review_item)

    return (
        ExtractionPlan(
            plan_id=plan_id,
            schema_id=schema.schema_id,
            fields=fields,
            entities=_list_value(body.get("entities")),
            tables=_agent_tables(body),
            requirements=_list_value(body.get("requirements")),
            ignored_nodes=_string_list(body.get("ignored_nodes")),
            unknown_nodes=_string_list(body.get("unknown_nodes")),
            ignored_node_reasons={},
        ),
        rejected,
        review_items,
    )


def apply_rule_fallback_fields(agent_plan: ExtractionPlan, rule_plan: ExtractionPlan) -> tuple[ExtractionPlan, list[dict[str, Any]]]:
    existing_signatures = {_field_signature(field) for field in agent_plan.fields}
    fields = list(agent_plan.fields)
    fallback_items: list[dict[str, Any]] = []

    for rule_field in rule_plan.fields:
        signature = _field_signature(rule_field)
        if signature in existing_signatures:
            continue
        fallback_field = replace(
            rule_field,
            field_plan_id=stable_id("fp", len(fields) + 1),
            confidence={
                key: max(float(value), 0.95) if isinstance(value, float) else value
                for key, value in rule_field.confidence.items()
            },
            boundary={
                **rule_field.boundary,
                "candidate_source": "rule_validation_fallback",
                "fallback_reason": "agent_plan_missing_semantic_key",
            },
        )
        fields.append(fallback_field)
        existing_signatures.add(signature)
        fallback_items.append(
            {
                "field_plan_id": fallback_field.field_plan_id,
                "semantic_key": fallback_field.semantic_key,
                "source": "rule_validation_fallback",
                "reason": "agent_plan_missing_semantic_key",
            }
        )

    return (
        ExtractionPlan(
            plan_id=agent_plan.plan_id,
            schema_id=agent_plan.schema_id,
            fields=fields,
            entities=agent_plan.entities,
            tables=agent_plan.tables,
            requirements=agent_plan.requirements,
            ignored_nodes=agent_plan.ignored_nodes,
            unknown_nodes=agent_plan.unknown_nodes,
            ignored_node_reasons=agent_plan.ignored_node_reasons,
        ),
        fallback_items,
    )


def build_rule_candidate_review_items(agent_plan: ExtractionPlan, rule_plan: ExtractionPlan) -> list[dict[str, Any]]:
    agent_ranges_by_key: dict[str, set[tuple[tuple[str, int, int], ...]]] = {}
    for field in agent_plan.fields:
        agent_ranges_by_key.setdefault(field.semantic_key, set()).add(_field_range_signature(field))
    rule_fields_by_key: dict[str, list[FieldPlan]] = {}
    for field in rule_plan.fields:
        rule_fields_by_key.setdefault(field.semantic_key, []).append(field)

    return [
        {
            "candidate_id": stable_id("rule_review_candidate", index),
            "issue_type": "agent_semantic_slot_missing",
            "semantic_key": field.semantic_key,
            "entity_id": field.entity_id,
            "source_span_ids": [span_range.span_id for span_range in field.value_source.ranges],
            "severity": "medium" if field.criticality == "critical" else "low",
            "message": "Rule anchor suggests a semantic slot that the Agent did not extract; candidate was not compiled.",
            "mandatory_review": field.criticality == "critical",
        }
        for index, field in enumerate(rule_plan.fields, start=1)
        if _rule_candidate_needs_review(field, rule_fields_by_key[field.semantic_key], agent_ranges_by_key)
    ]


def agent_field_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _normalize_agent_item(item, {})
        for item in _agent_items(_agent_plan_body(body))
        if isinstance(item, dict)
    ]


def _rule_candidate_needs_review(
    field: FieldPlan,
    same_key_rule_fields: list[FieldPlan],
    agent_ranges_by_key: dict[str, set[tuple[tuple[str, int, int], ...]]],
) -> bool:
    agent_ranges = agent_ranges_by_key.get(field.semantic_key)
    if not agent_ranges:
        return True
    repeatable = len(same_key_rule_fields) > 1 or len({item.entity_id for item in same_key_rule_fields if item.entity_id}) > 1
    return repeatable and _field_range_signature(field) not in agent_ranges


def merge_agent_candidates(
    plan: ExtractionPlan,
    schema: GeneratedSchema,
    spans: list[TextSpan],
    agent_items_path: Path | None,
) -> tuple[ExtractionPlan, list[dict[str, Any]], list[dict[str, Any]]]:
    if agent_items_path is None:
        return plan, [], []

    body = json.loads(agent_items_path.read_text(encoding="utf-8"))
    return merge_agent_candidate_body(plan, schema, spans, body)


def merge_agent_candidate_body(
    plan: ExtractionPlan,
    schema: GeneratedSchema,
    spans: list[TextSpan],
    body: dict[str, Any],
) -> tuple[ExtractionPlan, list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(body, dict):
        raise AgentCandidateError("Agent candidate file must be a JSON object.")
    items = body.get("items", [])
    if not isinstance(items, list):
        raise AgentCandidateError("Agent candidate file field `items` must be a list.")

    span_by_id = {span.span_id: span for span in spans}
    field_def_by_id = {definition.field_def_id: definition for definition in schema.field_definitions}
    known_signatures = {_field_signature(field) for field in plan.fields}
    fields = list(plan.fields)
    rejected: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        field_plan = _field_plan_from_agent_item(
            index=index,
            item=item,
            schema=schema,
            span_by_id=span_by_id,
            field_def_by_id=field_def_by_id,
            field_number=len(fields) + 1,
            candidate_source="agent_items_path",
            duplicate_signatures=known_signatures,
        )
        if field_plan.rejection is not None:
            rejected.append(field_plan.rejection)
            continue
        fields.append(field_plan.field)
        known_signatures.add(_field_signature(field_plan.field))
        if field_plan.review_item is not None:
            review_items.append(field_plan.review_item)

    return (
        ExtractionPlan(
            plan_id=plan.plan_id,
            schema_id=plan.schema_id,
            fields=fields,
            entities=plan.entities,
            tables=plan.tables,
            requirements=plan.requirements,
            ignored_nodes=plan.ignored_nodes,
            unknown_nodes=plan.unknown_nodes,
            ignored_node_reasons=plan.ignored_node_reasons,
        ),
        rejected,
        review_items,
    )


def _confidence(item: dict[str, Any]) -> float:
    confidence = item.get("confidence", item.get("overall_confidence", 0.95))
    if isinstance(confidence, dict):
        confidence = confidence.get("overall", confidence.get("boundary_confidence", 0.80))
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return 0.80


def _criticality(schema: GeneratedSchema, semantic_key: str) -> str:
    for definition in schema.field_definitions:
        if definition.semantic_key == semantic_key:
            return definition.criticality
    return "non_critical" if semantic_key.startswith(("custom.", "proposed.")) else "critical"


def _normalize_agent_item(item: dict[str, Any], field_def_by_id: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    for nested_key in ("source", "evidence", "location", "span_range", "range"):
        nested = normalized.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                normalized.setdefault(key, value)
    field_def_id = normalized.get("field_def_id")
    if not field_def_id and str(normalized.get("semantic_key") or "").startswith("fdef_"):
        field_def_id = normalized.get("semantic_key")
    definition = field_def_by_id.get(str(field_def_id)) if field_def_id else None
    if definition:
        if not normalized.get("semantic_key") or normalized.get("semantic_key") == field_def_id:
            normalized["semantic_key"] = definition.semantic_key
        if not str(normalized.get("display_name") or "").strip():
            normalized["display_name"] = definition.display_name
        if not str(normalized.get("field_type") or "").strip() or normalized.get("field_type") == "unknown":
            normalized["field_type"] = definition.field_type
    if "semantic_key" not in normalized:
        for alias in ("semanticKey", "field_key", "field_name", "name", "key"):
            if alias in normalized:
                normalized["semantic_key"] = normalized[alias]
                break
    if "display_name" not in normalized and "label" in normalized:
        normalized["display_name"] = normalized["label"]
    if "span_id" not in normalized and "source_span_id" in normalized:
        normalized["span_id"] = normalized["source_span_id"]
    if "text" not in normalized:
        for alias in ("value", "raw_value", "exact_text", "source_text"):
            if alias in normalized:
                normalized["text"] = normalized[alias]
                break
    if "start_offset" not in normalized and "start" in normalized:
        normalized["start_offset"] = normalized["start"]
    if "end_offset" not in normalized and "end" in normalized:
        normalized["end_offset"] = normalized["end"]
    if "start_offset" not in normalized and "offset_start" in normalized:
        normalized["start_offset"] = normalized["offset_start"]
    if "end_offset" not in normalized and "offset_end" in normalized:
        normalized["end_offset"] = normalized["offset_end"]
    offsets = normalized.get("offsets")
    if isinstance(offsets, list) and len(offsets) == 2:
        normalized.setdefault("start_offset", offsets[0])
        normalized.setdefault("end_offset", offsets[1])
    semantic_key = str(normalized.get("semantic_key") or "")
    normalized["semantic_key"] = {
        "business_operator.name": "manufacturer.name",
        "business_operator.address": "manufacturer.address",
        "business_operator.enterprise_address": "manufacturer.address",
        "business_operator.principal_name": "principal.name",
        "business_operator.principal_address": "principal.address",
        "business_operator.entrusted_processor_name": "manufacturer.name",
        "business_operator.entrusted_processor_factory_code": "manufacturer.factory_code",
        "business_operator.entrusted_processor_address": "manufacturer.address",
        "business_operator.license_number": "manufacturer.license_number",
        "business_operator.food_production_license_code": "manufacturer.license_number",
        "business_operator.food_production_license_number": "manufacturer.license_number",
        "business_principal.name": "principal.name",
        "business_principal.address": "principal.address",
        "business_principal.contact": "principal.contact",
        "contract_manufacturer.name": "manufacturer.name",
        "contract_manufacturer.address": "manufacturer.address",
        "contract_manufacturer.origin": "manufacturer.origin",
        "contract_manufacturer.food_production_license": "manufacturer.license_number",
        "manufacturer.food_production_license_code": "manufacturer.license_number",
        "manufacturer.food_production_license_number": "manufacturer.license_number",
        "manufacturer.food_production_license": "manufacturer.license_number",
        "manufacturer.customer_service_hotline": "manufacturer.contact",
        "product.food_production_license_code": "manufacturer.license_number",
        "product.food_production_license_number": "manufacturer.license_number",
        "custom.allergen_statement": "custom.allergen_notice",
    }.get(semantic_key, semantic_key)
    return normalized


class _FieldPlanFromAgentItem:
    def __init__(
        self,
        *,
        field: FieldPlan | None = None,
        rejection: dict[str, Any] | None = None,
        review_item: dict[str, Any] | None = None,
    ) -> None:
        self.field = field
        self.rejection = rejection
        self.review_item = review_item


class _AgentRangeResult:
    def __init__(
        self,
        *,
        ranges: list[SpanRange] | None = None,
        reason: str | None = None,
        actual: str | None = None,
        offset_repaired: bool = False,
    ) -> None:
        self.ranges = ranges or []
        self.reason = reason
        self.actual = actual
        self.offset_repaired = offset_repaired


def _field_plan_from_agent_item(
    *,
    index: int,
    item: Any,
    schema: GeneratedSchema,
    span_by_id: dict[str, TextSpan],
    field_def_by_id: dict[str, Any],
    field_number: int,
    candidate_source: str,
    duplicate_signatures: set[tuple[str, str | None, tuple[tuple[str, int, int], ...]]],
) -> _FieldPlanFromAgentItem:
    if not isinstance(item, dict):
        return _FieldPlanFromAgentItem(rejection=_rejection(index, item, "item_not_object"))
    normalized = _normalize_agent_item(item, field_def_by_id)

    semantic_key = str(normalized.get("semantic_key") or normalized.get("field") or "")
    display_name = str(normalized.get("display_name") or normalized.get("label") or semantic_key)
    field_type = str(normalized.get("field_type") or normalized.get("type") or "string")
    if not semantic_key:
        return _FieldPlanFromAgentItem(rejection=_rejection(index, normalized, "missing_semantic_key"))

    range_result = _span_ranges_from_agent_item(normalized, span_by_id)
    if not range_result.ranges:
        return _FieldPlanFromAgentItem(rejection=_rejection(index, normalized, range_result.reason or "missing_or_unknown_span_id", actual=range_result.actual))

    candidate_signature = (
        semantic_key,
        str(normalized.get("entity_id")) if normalized.get("entity_id") else None,
        tuple((item.span_id, item.start_offset, item.end_offset) for item in range_result.ranges),
    )
    if candidate_signature in duplicate_signatures:
        return _FieldPlanFromAgentItem(rejection=_rejection(index, normalized, "duplicate_of_rule_candidate"))

    confidence = _confidence(normalized)
    criticality = str(normalized.get("criticality") or _criticality(schema, semantic_key))
    first_range = range_result.ranges[0]
    review_item = None
    if confidence < 0.85:
        review_item = {
            "item_index": index,
            "semantic_key": semantic_key,
            "span_id": first_range.span_id,
            "reason": "agent_candidate_low_confidence",
            "confidence": confidence,
        }

    return _FieldPlanFromAgentItem(
        field=FieldPlan(
            field_plan_id=stable_id("fp", field_number),
            semantic_key=semantic_key,
            display_name=display_name,
            field_type=field_type,
            section_id=normalized.get("section_id") or "sec_label_text",
            entity_id=normalized.get("entity_id"),
            value_source=ValueSource(
                mode="span_ranges",
                ranges=range_result.ranges,
            ),
            criticality=criticality,
            confidence={
                "schema_confidence": confidence,
                "boundary_confidence": confidence,
                "entity_linking_confidence": confidence,
            },
            boundary={
                "start_anchor": normalized.get("label") or display_name,
                "end_reason": normalized.get("end_reason") or "agent_offsets",
                "candidate_source": candidate_source,
                "agent_block_id": normalized.get("agent_block_id"),
                "offset_repair": "matched_text_in_source_span" if range_result.offset_repaired else None,
            },
        ),
        review_item=review_item,
    )


def _field_signature(field: FieldPlan) -> tuple[str, str | None, tuple[tuple[str, int, int], ...]]:
    return (
        field.semantic_key,
        field.entity_id,
        tuple((item.span_id, item.start_offset, item.end_offset) for item in field.value_source.ranges),
    )


def _field_range_signature(field: FieldPlan) -> tuple[tuple[str, int, int], ...]:
    return tuple((item.span_id, item.start_offset, item.end_offset) for item in field.value_source.ranges)


def _span_ranges_from_agent_item(item: dict[str, Any], span_by_id: dict[str, TextSpan]) -> _AgentRangeResult:
    range_refs = _range_refs_from_agent_item(item)
    ranges: list[SpanRange] = []
    offset_repaired = False
    for range_ref in range_refs:
        span_id = str(range_ref.get("span_id") or range_ref.get("source_span_id") or "")
        source_span = span_by_id.get(span_id)
        if source_span is None:
            return _AgentRangeResult(reason="missing_or_unknown_span_id")
        try:
            start_offset = int(range_ref.get("start_offset", range_ref.get("char_start", range_ref.get("start", range_ref.get("offset_start", 0)))))
            end_offset = int(
                range_ref.get(
                    "end_offset",
                    range_ref.get("char_end", range_ref.get("end", range_ref.get("offset_end", len(source_span.text)))),
                )
            )
        except (TypeError, ValueError):
            return _AgentRangeResult(reason="invalid_offsets")
        expected_text = range_ref.get("text")
        if isinstance(expected_text, str) and not expected_text:
            return _AgentRangeResult(reason="empty_agent_text")
        offsets_valid = 0 <= start_offset < end_offset <= len(source_span.text)
        source_text = source_span.text[start_offset:end_offset] if offsets_valid else ""
        if isinstance(expected_text, str) and (not offsets_valid or expected_text != source_text):
            repaired_start = source_span.text.find(expected_text)
            if repaired_start < 0:
                reason = "offsets_out_of_range" if not offsets_valid else "text_does_not_match_source_span"
                return _AgentRangeResult(reason=reason, actual=source_text)
            start_offset = repaired_start
            end_offset = repaired_start + len(expected_text)
            offset_repaired = True
        elif not offsets_valid:
            return _AgentRangeResult(reason="offsets_out_of_range")
        ranges.append(SpanRange(span_id=span_id, start_offset=start_offset, end_offset=end_offset))
    return _AgentRangeResult(ranges=ranges, offset_repaired=offset_repaired)


def _range_refs_from_agent_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    if item.get("span_id") or item.get("source_span_id"):
        return [item]
    for key in ("value", "text", "ranges", "source_ranges", "span_ranges"):
        refs = item.get(key)
        if isinstance(refs, list):
            return [ref for ref in refs if isinstance(ref, dict)]
    return []


def _agent_items(body: dict[str, Any]) -> list[Any]:
    direct: list[Any] = []
    for key in ("fields", "field_plans", "extraction_fields", "items"):
        values = body.get(key)
        if isinstance(values, list) and values:
            direct = list(values)
            break
    if not direct:
        direct = _flatten_path_items(body)

    entity_fields = []
    for entity in _list_value(body.get("entities")):
        entity_id = entity.get("entity_id")
        for field in _list_value(entity.get("fields")):
            item = dict(field)
            if entity_id:
                item["entity_id"] = str(entity_id)
            entity_fields.append(item)
    if direct or entity_fields:
        return [*direct, *entity_fields]

    found = _find_agent_items(body)
    return found if found is not None else []


def _find_agent_items(value: Any) -> list[Any] | None:
    if not isinstance(value, dict):
        return None
    saw_explicit_empty = False
    for key in ("fields", "field_plans", "extraction_fields", "items"):
        items = value.get(key)
        if isinstance(items, list):
            if items:
                return items
            saw_explicit_empty = True
    flattened = _flatten_path_items(value)
    if flattened:
        return flattened
    for nested in value.values():
        if isinstance(nested, dict):
            found = _find_agent_items(nested)
            if found is not None:
                return found
    return [] if saw_explicit_empty else None


def _agent_plan_body(body: dict[str, Any]) -> dict[str, Any]:
    for key in ("extraction_plan", "plan", "proposal"):
        nested = body.get(key)
        if isinstance(nested, dict):
            return nested
    return body


PATH_FIELD_MAP = {
    ("product", "name"): ("product.name", "品名", "string"),
    ("product", "product_name"): ("product.name", "品名", "string"),
    ("product", "ingredients"): ("product.ingredients", "配料", "long_text"),
    ("product", "ingredients_text"): ("product.ingredients", "配料", "long_text"),
    ("product", "product_type"): ("product.product_type", "产品类型", "string"),
    ("product", "shelf_life"): ("product.shelf_life", "保质期", "string"),
    ("product", "storage_condition"): ("product.storage_condition", "贮存条件", "string"),
    ("product", "storage_conditions"): ("product.storage_condition", "贮存条件", "string"),
    ("product", "net_weight"): ("product.net_content", "净含量", "string"),
    ("product", "net_content"): ("product.net_content", "净含量", "string"),
    ("product", "product_standard"): ("product.standard_code", "产品标准代号", "string"),
    ("product", "standard_code"): ("product.standard_code", "产品标准代号", "string"),
    ("manufacturer", "name"): ("manufacturer.name", "生产商", "string"),
    ("manufacturer", "origin"): ("manufacturer.origin", "产地", "string"),
    ("manufacturer", "address"): ("manufacturer.address", "地址", "long_text"),
    ("manufacturer", "license"): ("manufacturer.license_number", "许可证编号", "string"),
    ("manufacturer", "license_number"): ("manufacturer.license_number", "许可证编号", "string"),
    ("allergen_statement",): ("custom.allergen_statement", "致敏物质提示", "long_text"),
    ("juice_content_statement",): ("custom.juice_content_statement", "果汁含量说明", "string"),
    ("food_production_license",): ("manufacturer.license_number", "许可证编号", "string"),
    ("product_name",): ("product.name", "品名", "string"),
    ("product_standard",): ("product.standard_code", "产品标准代号", "string"),
    ("customer_service_hotline",): ("custom.customer_service_hotline", "客服热线", "string"),
    ("marketing_claims",): ("custom.marketing_claims", "宣传语", "string"),
    ("net_weight",): ("product.net_content", "净含量", "string"),
    ("nutrition_warning",): ("custom.nutrition_warning", "营养提示", "string"),
}


def _flatten_path_items(value: dict[str, Any], path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, nested in value.items():
        if key in {"field_definitions", "table_definitions", "entity_types", "sections", "requirements", "unknown_nodes", "ignored_nodes"}:
            continue
        child_path = path + (str(key),)
        if child_path and child_path[0] in {"nutrition_facts_table", "nutrition_table"}:
            continue
        if isinstance(nested, dict) and _has_span_ref(nested):
            mapped = _field_mapping(child_path)
            if mapped is None:
                continue
            semantic_key, display_name, field_type = mapped
            item = dict(nested)
            item.setdefault("semantic_key", semantic_key)
            item.setdefault("display_name", display_name)
            item.setdefault("field_type", field_type)
            item.setdefault("confidence", 0.96)
            items.append(item)
            continue
        if isinstance(nested, dict):
            items.extend(_flatten_path_items(nested, child_path))
    return items


def _field_mapping(path: tuple[str, ...]) -> tuple[str, str, str] | None:
    if path in PATH_FIELD_MAP:
        return PATH_FIELD_MAP[path]
    if len(path) == 1:
        key = path[0]
        return (f"custom.{key}", key, "string")
    return None


def _has_span_ref(item: dict[str, Any]) -> bool:
    normalized = dict(item)
    for nested_key in ("source", "evidence", "location", "span_range", "range"):
        nested = normalized.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                normalized.setdefault(key, value)
    return bool(normalized.get("span_id") or normalized.get("source_span_id")) and any(
        key in normalized for key in ("start_offset", "end_offset", "offset_start", "offset_end", "start", "end", "offsets")
    )


def _agent_tables(body: dict[str, Any]) -> list[dict[str, Any]]:
    tables = _list_value(body.get("tables"))
    nutrition_table = body.get("nutrition_facts_table") or body.get("nutrition_table")
    if isinstance(nutrition_table, dict):
        rows = []
        for row in nutrition_table.get("rows", []):
            if not isinstance(row, dict):
                continue
            cells = []
            for key, column_id in (
                ("nutrient", "col_001"),
                ("amount", "col_002"),
                ("unit", "col_003"),
                ("nrv_percent", "col_004"),
            ):
                cell_ref = row.get(key)
                if not isinstance(cell_ref, dict):
                    continue
                cell = _span_ref_table_cell(cell_ref, column_id)
                if cell:
                    cells.append(cell)
            if cells:
                rows.append({"row_key": str(row.get("row_key") or ""), "cells": cells})
        if rows:
            tables.append(
                {
                    "table_type": "nutrition_facts",
                    "title": "营养成分表",
                    "source_span_ids": _table_plan_span_ids(rows),
                    "confidence": 0.96,
                    "rows": rows,
                }
            )
    return tables


def _span_ref_table_cell(cell_ref: dict[str, Any], column_id: str) -> dict[str, Any] | None:
    span_id = cell_ref.get("span_id") or cell_ref.get("source_span_id")
    if not span_id:
        return None
    cell = {"column_id": column_id, "span_id": str(span_id)}
    if "offset_start" in cell_ref:
        cell["start_offset"] = cell_ref["offset_start"]
    if "offset_end" in cell_ref:
        cell["end_offset"] = cell_ref["offset_end"]
    if "start_offset" in cell_ref:
        cell["start_offset"] = cell_ref["start_offset"]
    if "end_offset" in cell_ref:
        cell["end_offset"] = cell_ref["end_offset"]
    if "text" in cell_ref:
        cell["text"] = cell_ref["text"]
    return cell


def _table_plan_span_ids(rows: list[dict[str, Any]]) -> list[str]:
    span_ids: list[str] = []
    for row in rows:
        for cell in row.get("cells", []):
            span_id = cell.get("span_id")
            if span_id and span_id not in span_ids:
                span_ids.append(str(span_id))
    return span_ids


def _list_value(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _rejection(index: int, item: Any, reason: str, actual: str | None = None) -> dict[str, Any]:
    rejected = {
        "item_index": index,
        "reason": reason,
        "item": item,
    }
    if actual is not None:
        rejected["actual_source_text"] = actual
    return rejected
