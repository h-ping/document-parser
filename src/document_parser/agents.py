from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, replace
from typing import Any

from .models import (
    ExtractionPlan,
    FieldDefinition,
    FieldPlan,
    GeneratedSchema,
    SpanRange,
    TextSpan,
    ValueSource,
)
from .structures import CONTENT_ITEM_RE, detect_regions
from .utils import stable_id


@dataclass(frozen=True)
class FieldPattern:
    semantic_key: str
    display_name: str
    field_type: str
    criticality: str
    labels: tuple[str, ...]
    repeatable: bool = False


CANONICAL_PATTERNS: tuple[FieldPattern, ...] = (
    FieldPattern("product.name", "品名", "string", "critical", ("品名", "产品名称", "名称")),
    FieldPattern("product.net_content", "净含量", "string", "critical", ("净含量", "规格")),
    FieldPattern("product.ingredients", "配料", "long_text", "critical", ("配料", "配料表", "成分"), True),
    FieldPattern("product.product_type", "产品类型", "string", "critical", ("产品类型", "产品类别")),
    FieldPattern("product.standard_code", "产品标准代号", "string", "critical", ("产品标准代号", "产品执行标准", "执行标准")),
    FieldPattern("product.shelf_life", "保质期", "string", "critical", ("保质期",)),
    FieldPattern("product.storage_condition", "贮存条件", "string", "critical", ("贮存条件", "储存条件")),
    FieldPattern("content_item.product_category", "产品分类", "string", "critical", ("产品分类",), True),
    FieldPattern("manufacturer.name", "生产商", "string", "critical", ("生产商", "生产者", "生产企业", "受托方", "委托方"), True),
    FieldPattern("manufacturer.address", "地址", "long_text", "critical", ("地址", "生产地址"), True),
    FieldPattern("manufacturer.origin", "产地", "string", "critical", ("产地",), True),
    FieldPattern("manufacturer.license_number", "许可证编号", "string", "critical", ("食品生产许可证编号", "生产许可证编号", "许可证编号"), True),
    FieldPattern("barcode.commodity", "商品条码", "barcode", "critical", ("商品条码", "条码"), True),
    FieldPattern("barcode.outer_case", "外箱条码", "barcode", "critical", ("外箱条码",), True),
    FieldPattern(
        "requirement.text",
        "文字要求",
        "requirement",
        "non_critical",
        ("文字要求", "其它要求", "其他要求", "设计注意", "推广注意", "日期喷印注意", "变化说明"),
        True,
    ),
)

LABEL_SEPARATOR = r"[:：]"


class SchemaInductionAgent:
    def generate(self, spans: list[TextSpan]) -> GeneratedSchema:
        discovered: list[FieldDefinition] = []
        seen: set[str] = set()

        for span in spans:
            for pattern in CANONICAL_PATTERNS:
                if _matches_any_label(span.text, pattern.labels) and pattern.semantic_key not in seen:
                    seen.add(pattern.semantic_key)
                    discovered.append(
                        FieldDefinition(
                            field_def_id=stable_id("fdef", len(discovered) + 1),
                            semantic_key=pattern.semantic_key,
                            display_name=pattern.display_name,
                            field_type=pattern.field_type,
                            criticality=pattern.criticality,
                            repeatable=pattern.repeatable,
                            source_span_ids=[span.span_id],
                        )
                    )

        sections = [
            {"section_id": "sec_document", "section_type": "document", "display_name": "文档"},
            {"section_id": "sec_label_text", "section_type": "label_text", "display_name": "标签文字内容"},
        ]
        for region in detect_regions(spans):
            section_id = f"sec_{region['region_id']}"
            sections.append(
                {
                    "section_id": section_id,
                    "section_type": region["region_type"],
                    "display_name": region["display_name"],
                    "source_span_ids": region["source_span_ids"],
                }
            )
        entity_types = [
            {"entity_type": "product", "repeatable": False},
            {"entity_type": "content_item", "repeatable": True},
            {"entity_type": "manufacturer", "repeatable": True},
            {"entity_type": "barcode", "repeatable": True},
            {"entity_type": "requirement", "repeatable": True},
        ]
        table_definitions: list[dict[str, Any]] = []
        content_item_span_ids = [span.span_id for span in spans if CONTENT_ITEM_RE.match(span.text.strip())]
        if content_item_span_ids:
            if "content_item.name" not in seen:
                seen.add("content_item.name")
                discovered.append(
                    FieldDefinition(
                        field_def_id=stable_id("fdef", len(discovered) + 1),
                        semantic_key="content_item.name",
                        display_name="内容物名称",
                        field_type="string",
                        criticality="critical",
                        repeatable=True,
                        source_span_ids=content_item_span_ids,
                    )
                )
        nutrition_span_ids = [span.span_id for span in spans if "营养成分表" in span.text]
        if nutrition_span_ids:
            table_definitions.append(
                {
                    "table_type": "nutrition_facts",
                    "display_name": "营养成分表",
                    "criticality": "critical",
                    "repeatable": True,
                    "source_span_ids": nutrition_span_ids,
                }
            )

        return GeneratedSchema(
            schema_id="schema_dynamic_001",
            auto_generated=True,
            schema_version="dynamic_v1",
            sections=sections,
            entity_types=entity_types,
            field_definitions=discovered,
            table_definitions=table_definitions,
            requirement_definitions=[],
        )


class ExtractionAgent:
    def create_plan(self, schema: GeneratedSchema, spans: list[TextSpan]) -> ExtractionPlan:
        fields: list[FieldPlan] = []
        current_content_entity_id: str | None = None
        current_manufacturer_entity_id: str | None = None
        manufacturer_index = 0
        for span_index, span in enumerate(spans):
            content_match = CONTENT_ITEM_RE.match(span.text.strip())
            if content_match:
                current_content_entity_id = f"content_item_{int(content_match.group(1)):03d}"
                fields.append(
                    FieldPlan(
                        field_plan_id=stable_id("fp", len(fields) + 1),
                        semantic_key="content_item.name",
                        display_name="内容物名称",
                        field_type="string",
                        section_id="sec_label_text",
                        entity_id=current_content_entity_id,
                        value_source=ValueSource(
                            mode="span_ranges",
                            ranges=[SpanRange(span_id=span.span_id, start_offset=0, end_offset=len(span.text))],
                        ),
                        criticality="critical",
                        confidence={
                            "schema_confidence": 0.95,
                            "boundary_confidence": 0.96,
                            "entity_linking_confidence": 0.98,
                        },
                        boundary={"start_anchor": "内容物", "end_reason": "line_end"},
                    )
                )

            span_starts_manufacturer_group = any(label in span.text for label in ("生产商", "生产者", "生产企业", "受托方", "委托方", "地址："))
            if span_starts_manufacturer_group and any(label in span.text for label in ("许可证编号", "地址", "生产商", "生产者", "受托方", "委托方")):
                manufacturer_index += 1
                current_manufacturer_entity_id = f"manufacturer_{manufacturer_index:03d}"

            matches = _field_matches_for_span(span.text)
            for match_index, match in enumerate(matches):
                pattern = match["pattern"]
                next_start = matches[match_index + 1]["label_start"] if match_index + 1 < len(matches) else len(span.text)
                field_plan_id = stable_id("fp", len(fields) + 1)
                entity_id = _entity_for(pattern.semantic_key, len(fields) + 1, current_content_entity_id, current_manufacturer_entity_id)
                ranges = [
                    SpanRange(
                        span_id=span.span_id,
                        start_offset=match["label_start"],
                        end_offset=next_start,
                    )
                ]
                ranges.extend(_continuation_ranges(pattern, spans, span_index))
                fields.append(
                    FieldPlan(
                        field_plan_id=field_plan_id,
                        semantic_key=pattern.semantic_key,
                        display_name=pattern.display_name,
                        field_type=pattern.field_type,
                        section_id="sec_label_text",
                        entity_id=entity_id,
                        value_source=ValueSource(
                            mode="span_ranges",
                            ranges=ranges,
                        ),
                        criticality=pattern.criticality,
                        confidence={
                            "schema_confidence": 0.90,
                            "boundary_confidence": 0.86 if pattern.field_type == "long_text" else 0.92,
                            "entity_linking_confidence": 0.80 if pattern.semantic_key.startswith("manufacturer.") else 0.90,
                        },
                        boundary={
                            "start_anchor": match["label"],
                            "end_reason": "next_sibling_field_detected" if next_start < len(span.text) else "line_or_continuation_end",
                        },
                    )
                )
        return ExtractionPlan(
            plan_id="plan_001",
            schema_id=schema.schema_id,
            fields=fields,
            unknown_nodes=[],
        )


class AuditAgent:
    def audit(self, compiled_fields: dict[str, Any], schema: GeneratedSchema) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        for field_id, field in compiled_fields.items():
            raw_value = field.get("raw_value", "")
            evidence_refs = field.get("evidence_refs", [])
            if not evidence_refs:
                findings.append(_finding("ungrounded_field", field_id, "high", "字段没有 evidence。"))
            if field.get("criticality") == "critical" and not field.get("has_bbox"):
                findings.append(_finding("critical_field_without_bbox", field_id, "high", "关键字段缺少 bbox。", evidence_refs))
            if field.get("field_type") == "long_text":
                adhesion = _first_adhesion_label(raw_value, schema, field)
                if adhesion:
                    findings.append(
                        _finding(
                            "possible_field_adhesion",
                            field_id,
                            "high",
                            "长文本字段疑似粘入同级字段。",
                            evidence_refs,
                            {
                                "adhesion_label": adhesion["label"],
                                "adhesion_offset": adhesion["offset"],
                            },
                        )
                    )

        return findings


class RepairAgent:
    def repair(
        self,
        plan: ExtractionPlan,
        audit_findings: list[dict[str, Any]],
        spans: list[TextSpan] | None = None,
    ) -> tuple[ExtractionPlan, list[dict[str, Any]]]:
        attempts = []
        repaired_plan = plan
        for finding in audit_findings:
            if finding.get("finding_type") == "possible_field_adhesion" and spans is not None:
                repaired_plan, attempt = _repair_field_adhesion(repaired_plan, finding, spans)
                attempts.append(attempt)
                continue
            attempts.append(
                {
                    "finding_id": finding["finding_id"],
                    "status": "skipped",
                    "reason": "rule_based_repair_not_available_for_issue",
                }
            )
        return repaired_plan, attempts


def clean_field_value(raw_value: str, display_name: str) -> tuple[str, list[str]]:
    if display_name == "内容物名称":
        return raw_value.strip(), []

    labels = {display_name}
    for pattern in CANONICAL_PATTERNS:
        if pattern.display_name == display_name:
            labels.update(pattern.labels)

    for label in sorted(labels, key=len, reverse=True):
        regex = rf"^\s*{_label_pattern(label)}\s*{LABEL_SEPARATOR}?\s*"
        cleaned = re.sub(regex, "", raw_value, count=1)
        if cleaned != raw_value:
            return cleaned.strip(), ["remove_field_label"]
    return raw_value.strip(), []


def normalize_value(value: str, field_type: str) -> tuple[str, list[str]]:
    normalization: list[str] = []
    normalized = value.strip()
    if field_type == "barcode":
        without_spaces = re.sub(r"\s+", "", normalized)
        if without_spaces != normalized:
            normalized = without_spaces
            normalization.append("remove_spaces")
    return normalized, normalization


def _matches_any_label(text: str, labels: tuple[str, ...]) -> bool:
    return bool(_first_matching_label(text, labels))


def _first_matching_label(text: str, labels: tuple[str, ...]) -> str | None:
    for label in sorted(labels, key=len, reverse=True):
        if _label_regex(label).search(text):
            return label
    return None


def _field_matches_for_span(text: str) -> list[dict[str, Any]]:
    raw_matches: list[dict[str, Any]] = []
    for pattern in CANONICAL_PATTERNS:
        for label in sorted(pattern.labels, key=len, reverse=True):
            match = _label_regex(label).search(text)
            if not match:
                continue
            if _is_metadata_name_match(pattern, text, match):
                continue
            raw_matches.append(
                {
                    "pattern": pattern,
                    "label": label,
                    "label_start": match.start("label"),
                    "label_end": match.end("label"),
                    "value_start": match.end(),
                    "label_len": len(label),
                }
            )
            break

    raw_matches.sort(key=lambda item: (item["label_start"], -item["label_len"]))
    selected: list[dict[str, Any]] = []
    occupied: list[range] = []
    for item in raw_matches:
        item_range = range(item["label_start"], item["label_end"])
        if any(_ranges_overlap(item_range, existing) for existing in occupied):
            continue
        selected.append(item)
        occupied.append(item_range)
    return sorted(selected, key=lambda item: item["label_start"])


def _is_metadata_name_match(pattern: FieldPattern, text: str, match: re.Match[str]) -> bool:
    if pattern.semantic_key != "product.name":
        return False
    value_tail = text[match.end() :].strip()
    return value_tail.startswith(("机型", "文件编号", "版本号"))


def _continuation_ranges(pattern: FieldPattern, spans: list[TextSpan], span_index: int) -> list[SpanRange]:
    if pattern.semantic_key not in {"product.ingredients", "requirement.text"}:
        return []

    ranges: list[SpanRange] = []
    current_page = spans[span_index].page
    for next_span in spans[span_index + 1 :]:
        text = next_span.text.strip()
        if not text:
            continue
        if next_span.page != current_page or _is_stop_span(text):
            break
        ranges.append(SpanRange(span_id=next_span.span_id, start_offset=0, end_offset=len(next_span.text)))
    return ranges


def _is_stop_span(text: str) -> bool:
    if CONTENT_ITEM_RE.match(text):
        return True
    if any(marker in text for marker in ("营养成分表", "第一唛", "第二唛", "第三唛", "第四唛", "第五唛", "更改前", "更改后")):
        return True
    return bool(_field_matches_for_span(text))


def _label_regex(label: str) -> re.Pattern[str]:
    return re.compile(rf"(^|[\s，,；;])(?P<label>{_label_pattern(label)})\s*({LABEL_SEPARATOR})?")


def _label_pattern(label: str) -> str:
    variants = {
        "食": "[食⻝⾷]",
        "生": "[生⽣]",
    }
    return "".join(variants.get(char, re.escape(char)) for char in label)


def _ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


def _entity_for(semantic_key: str, index: int, current_content_entity_id: str | None = None, current_manufacturer_entity_id: str | None = None) -> str | None:
    if semantic_key.startswith("manufacturer."):
        return current_manufacturer_entity_id or f"manufacturer_{index:03d}"
    if semantic_key.startswith("barcode."):
        return f"barcode_{index:03d}"
    if semantic_key.startswith("requirement."):
        return f"requirement_{index:03d}"
    if semantic_key.startswith("content_item."):
        return current_content_entity_id
    if semantic_key in {"product.ingredients", "product.product_type"} and current_content_entity_id:
        return current_content_entity_id
    if semantic_key.startswith("product."):
        return "product_001"
    return None


def _first_adhesion_label(raw_value: str, schema: GeneratedSchema, field: dict[str, Any]) -> dict[str, Any] | None:
    own_semantic_key = field.get("semantic_key")
    labels = []
    for definition in schema.field_definitions:
        if definition.semantic_key == own_semantic_key:
            continue
        labels.append(definition.display_name)
        pattern = _pattern_for_semantic_key(definition.semantic_key)
        if pattern:
            labels.extend(pattern.labels)

    for label in sorted({item for item in labels if item}, key=len, reverse=True):
        match = _label_regex(label).search(raw_value)
        if match:
            return {"label": label, "offset": match.start("label")}
    return None


def _pattern_for_semantic_key(semantic_key: str) -> FieldPattern | None:
    for pattern in CANONICAL_PATTERNS:
        if pattern.semantic_key == semantic_key:
            return pattern
    return None


def _repair_field_adhesion(
    plan: ExtractionPlan,
    finding: dict[str, Any],
    spans: list[TextSpan],
) -> tuple[ExtractionPlan, dict[str, Any]]:
    target_index = _field_index_from_target_id(str(finding.get("target_id", "")))
    label = finding.get("details", {}).get("adhesion_label") or finding.get("adhesion_label")
    if target_index is None or target_index >= len(plan.fields) or not isinstance(label, str) or not label:
        return plan, _repair_attempt(finding, "skipped", "repair_target_or_label_unavailable")

    span_by_id = {span.span_id: span for span in spans}
    target_field = plan.fields[target_index]
    repaired_ranges: list[SpanRange] = []
    changed = False

    for span_range in target_field.value_source.ranges:
        source_span = span_by_id.get(span_range.span_id)
        if source_span is None:
            repaired_ranges.append(span_range)
            continue
        range_text = source_span.text[span_range.start_offset : span_range.end_offset]
        match = _label_regex(label).search(range_text)
        if not match:
            repaired_ranges.append(span_range)
            continue

        new_end = span_range.start_offset + match.start("label")
        if new_end > span_range.start_offset:
            repaired_ranges.append(replace(span_range, end_offset=new_end))
        changed = True
        break

    if not changed or not repaired_ranges:
        return plan, _repair_attempt(finding, "skipped", "adhesion_label_not_found_in_target_ranges")

    repaired_field = replace(
        target_field,
        value_source=replace(target_field.value_source, ranges=repaired_ranges),
        confidence={
            **target_field.confidence,
            "boundary_confidence": min(float(target_field.confidence.get("boundary_confidence") or 0.0) + 0.05, 0.95),
        },
        boundary={
            **target_field.boundary,
            "repair": "trimmed_at_sibling_label",
            "repair_label": label,
            "end_reason": "repair_trimmed_before_sibling_field",
        },
    )
    repaired_fields = list(plan.fields)
    repaired_fields[target_index] = repaired_field
    repaired_plan = replace(plan, fields=repaired_fields)
    return repaired_plan, _repair_attempt(
        finding,
        "applied",
        "trimmed_target_field_before_sibling_label",
        {"adhesion_label": label, "target_field_plan_id": target_field.field_plan_id},
    )


def _field_index_from_target_id(target_id: str) -> int | None:
    match = re.fullmatch(r"fld_(\d+)", target_id)
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if index >= 0 else None


def _repair_attempt(
    finding: dict[str, Any],
    status: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "finding_id": finding.get("finding_id"),
        "status": status,
        "reason": reason,
        "details": details or {},
    }


def _finding(
    finding_type: str,
    target_id: str,
    severity: str,
    message: str,
    evidence_refs: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256(f"{finding_type}:{target_id}:{message}".encode("utf-8")).hexdigest()[:8]
    return {
        "finding_id": f"af_{digest}",
        "finding_type": finding_type,
        "target_type": "field",
        "target_id": target_id,
        "severity": severity,
        "message": message,
        "evidence_refs": evidence_refs or [],
        "details": details or {},
    }
