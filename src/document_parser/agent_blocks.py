from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .models import TextSpan


MAX_BLOCK_SPANS = 160
MAX_BLOCK_CHARS = 12_000
BLOCK_OVERLAP = 2
IMPORTANT_FIELD_ANCHORS = (
    "产品名称",
    "品名",
    "内容物",
    "配料",
    "产品分类",
    "净含量",
    "保质期",
    "贮存条件",
    "产品标准",
    "委托方",
    "受托方",
    "被委托方",
    "生产者",
    "地址",
    "许可证",
    "产地",
    "条码",
)


def build_agent_blocks(spans: list[TextSpan], layout_candidates: dict[str, Any]) -> dict[str, Any]:
    span_by_id = {span.span_id: span for span in spans}
    order = {span.span_id: index for index, span in enumerate(spans)}
    assigned: set[str] = set()
    seeds: list[tuple[str, list[str], list[str]]] = []

    for candidate in layout_candidates.get("table_candidates", []):
        candidate_type = str(candidate.get("table_type", ""))
        block_type = "nutrition_table" if candidate_type == "nutrition_facts" else "producer_group" if candidate_type == "producer_info_repeated_rows" else ""
        if not block_type:
            continue
        source_ids = _known_unassigned(candidate.get("source_span_ids", []), span_by_id, assigned)
        if source_ids:
            assigned.update(source_ids)
            seeds.append((block_type, source_ids, [str(candidate.get("table_candidate_id") or candidate.get("table_layer_id") or "")]))

    content_anchors = [index for index, span in enumerate(spans) if re.match(r"^\s*内容物(?:\s*\d+)?\s*[:：]", span.text)]
    major_ids = {
        source_id
        for candidate in layout_candidates.get("table_candidates", [])
        for source_id in candidate.get("source_span_ids", [])
        if source_id in span_by_id
    }
    for anchor_position, start in enumerate(content_anchors):
        page = spans[start].page
        next_anchor = content_anchors[anchor_position + 1] if anchor_position + 1 < len(content_anchors) else len(spans)
        end = next_anchor
        for index in range(start + 1, next_anchor):
            if spans[index].page != page or spans[index].span_id in major_ids:
                end = index
                break
        source_ids = _known_unassigned((span.span_id for span in spans[start:end]), span_by_id, assigned)
        if source_ids:
            assigned.update(source_ids)
            seeds.append(("content_item", source_ids, []))

    side_ids = {
        str(source_id)
        for candidate in layout_candidates.get("side_marker_candidates", [])
        for source_id in candidate.get("source_span_ids", [])
    }
    other_ids = _known_unassigned(side_ids, span_by_id, assigned)
    if other_ids:
        assigned.update(other_ids)
        seeds.append(("other_printed_label", other_ids, []))

    remaining_by_page: dict[int, list[str]] = {}
    for span in spans:
        if span.span_id not in assigned:
            remaining_by_page.setdefault(span.page, []).append(span.span_id)
            assigned.add(span.span_id)
    seeds.extend(("main_label", source_ids, []) for _, source_ids in sorted(remaining_by_page.items()))

    blocks = []
    for block_type, source_ids, candidate_ids in sorted(seeds, key=lambda seed: min(order[source_id] for source_id in seed[1])):
        primary_chunks = _chunk_ids(source_ids, span_by_id)
        for chunk_index, primary_ids in enumerate(primary_chunks):
            previous = primary_chunks[chunk_index - 1][-BLOCK_OVERLAP:] if chunk_index else []
            context_ids = [*previous, *primary_ids]
            block_number = len(blocks) + 1
            block_spans = [span_by_id[source_id] for source_id in primary_ids]
            blocks.append(
                {
                    "block_id": f"agent_block_{block_number:04d}",
                    "block_type": block_type,
                    "pages": sorted({span.page for span in block_spans}),
                    "source_span_ids": primary_ids,
                    "context_span_ids": context_ids,
                    "layout_candidate_ids": [candidate_id for candidate_id in candidate_ids if candidate_id],
                    "span_count": len(primary_ids),
                    "context_span_count": len(context_ids),
                    "character_count": sum(len(span.text) for span in block_spans),
                }
            )
    covered = {source_id for block in blocks for source_id in block["source_span_ids"]}
    all_ids = set(span_by_id)
    return {
        "artifact_version": "agent_blocks_v0.1",
        "status": "pass" if covered == all_ids else "fail",
        "source_span_count": len(all_ids),
        "covered_source_span_count": len(covered),
        "source_span_coverage_rate": round(len(covered) / len(all_ids), 4) if all_ids else 1.0,
        "duplicate_primary_source_span_ids": _duplicates(source_id for block in blocks for source_id in block["source_span_ids"]),
        "blocks": blocks,
    }


def block_spans(block: dict[str, Any], spans: list[TextSpan]) -> list[TextSpan]:
    span_by_id = {span.span_id: span for span in spans}
    return [span_by_id[source_id] for source_id in block.get("context_span_ids", []) if source_id in span_by_id]


def schema_induction_spans(spans: list[TextSpan], max_spans: int = 240) -> list[TextSpan]:
    if len(spans) <= max_spans:
        return list(spans)
    important = [
        span
        for span in spans
        if any(anchor in span.text for anchor in (*IMPORTANT_FIELD_ANCHORS, "营养成分表", "致敏", "警示语"))
    ]
    selected = list(dict.fromkeys(span.span_id for span in important))[:max_spans]
    selected_ids = set(selected)
    by_page: dict[int, list[TextSpan]] = {}
    for span in spans:
        if span.span_id not in selected_ids:
            by_page.setdefault(span.page, []).append(span)
    page_numbers = sorted(by_page)
    while len(selected) < max_spans and any(by_page.values()):
        for page in page_numbers:
            if by_page[page] and len(selected) < max_spans:
                selected.append(by_page[page].pop(0).span_id)
    span_by_id = {span.span_id: span for span in spans}
    return [span_by_id[span_id] for span_id in selected]


def build_block_context(block: dict[str, Any], vdg_context: dict[str, Any]) -> dict[str, Any]:
    pages = set(block.get("pages", []))
    context_ids = set(block.get("context_span_ids", []))
    candidate_ids = set(block.get("layout_candidate_ids", []))
    return {
        "context_version": "vdg_agent_block_context_v0.1",
        "agent_block": block,
        "vdg_quality_status": vdg_context.get("vdg_quality_status"),
        "quality_issues": vdg_context.get("quality_issues", []),
        "label_text_scope": vdg_context.get("label_text_scope", {}),
        "regions": [region for region in vdg_context.get("regions", []) if region.get("page") in pages],
        "candidate_field_groups": [group for group in vdg_context.get("candidate_field_groups", []) if group.get("node_id") in context_ids],
        "table_candidates": [
            candidate
            for candidate in vdg_context.get("table_candidates", [])
            if candidate.get("table_layer_id") in candidate_ids or bool(context_ids & set(candidate.get("source_span_ids", [])))
        ],
        "unknown_nodes": [node for node in vdg_context.get("unknown_nodes", []) if node.get("node_id") in context_ids],
        "conflict_nodes": [node for node in vdg_context.get("conflict_nodes", []) if node.get("node_id") in context_ids],
        "reading_order_candidates": [
            candidate
            for candidate in vdg_context.get("reading_order_candidates", [])
            if _candidate_references_context(candidate, context_ids)
        ],
        "side_marker_candidates": [
            candidate
            for candidate in vdg_context.get("side_marker_candidates", [])
            if _candidate_references_context(candidate, context_ids)
        ],
        "source_fusion": {
            "alternate_readings": [
                alignment
                for alignment in vdg_context.get("source_fusion", {}).get("alternate_readings", [])
                if context_ids & set(alignment.get("pdf_span_ids", []))
            ]
        },
    }


def _candidate_references_context(candidate: dict[str, Any], context_ids: set[str]) -> bool:
    candidate_ids = {
        str(value)
        for key in ("source_span_ids", "span_ids")
        for value in candidate.get(key, [])
    }
    candidate_ids.update(
        str(candidate[key])
        for key in ("source_span_id", "from_span_id", "to_span_id")
        if candidate.get(key)
    )
    return bool(candidate_ids & context_ids)


def merge_schema_bodies(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for body in bodies:
        for item in body.get("field_definitions", body.get("fields", [])):
            if not isinstance(item, dict) or not item.get("semantic_key"):
                continue
            key = str(item["semantic_key"])
            if key not in fields:
                fields[key] = dict(item)
                continue
            existing_ids = list(fields[key].get("source_span_ids", []))
            fields[key]["source_span_ids"] = _unique([*existing_ids, *item.get("source_span_ids", [])])
            fields[key]["repeatable"] = bool(fields[key].get("repeatable")) or bool(item.get("repeatable"))
    return {
        "schema_id": "schema_agent_blocks_001",
        "schema_version": "agent_blocks_v0.1",
        "sections": _merge_objects(bodies, "sections", "section_id"),
        "entity_types": _merge_objects(bodies, "entity_types", "entity_type"),
        "field_definitions": list(fields.values()),
        "table_definitions": _merge_objects(bodies, "table_definitions", "table_type"),
        "requirement_definitions": _merge_objects(bodies, "requirement_definitions", "requirement_key"),
    }


def merge_agent_plan_bodies(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {key: [] for key in ("fields", "entities", "tables", "requirements", "ignored_nodes", "unknown_nodes", "layout_candidate_decisions", "node_scope_decisions")}
    for body_index, body in enumerate(bodies, start=1):
        block_id = str(body.get("_agent_block_id") or f"agent_block_{body_index:04d}")
        entity_map: dict[str, str] = {}
        for entity in body.get("entities", []):
            if not isinstance(entity, dict):
                continue
            item = dict(entity)
            original_id = str(item.get("entity_id") or f"entity_{len(entity_map) + 1:03d}")
            entity_map[original_id] = f"{block_id}:{original_id}"
            item["entity_id"] = entity_map[original_id]
            merged["entities"].append(item)
        for field in body.get("fields", []):
            if not isinstance(field, dict):
                continue
            item = dict(field)
            if item.get("entity_id"):
                original_id = str(item["entity_id"])
                item["entity_id"] = entity_map.get(original_id, f"{block_id}:{original_id}")
            item["agent_block_id"] = block_id
            merged["fields"].append(item)
        for table in body.get("tables", []):
            if isinstance(table, dict):
                item = dict(table)
                item["agent_block_id"] = block_id
                merged["tables"].append(item)
        for key in ("requirements", "ignored_nodes", "unknown_nodes", "layout_candidate_decisions", "node_scope_decisions"):
            values = body.get(key, [])
            if isinstance(values, list):
                merged[key].extend(values)
    merged["ignored_nodes"] = _unique(merged["ignored_nodes"])
    merged["unknown_nodes"] = _unique(merged["unknown_nodes"])
    merged["layout_candidate_decisions"] = _dedupe_decisions(merged["layout_candidate_decisions"])
    return merged


def important_unconsumed_span_ids(body: dict[str, Any], spans: list[TextSpan]) -> list[str]:
    consumed = {
        span_id
        for field in body.get("fields", [])
        if isinstance(field, dict)
        for span_id in _field_span_ids(field)
    }
    return [
        span.span_id
        for span in spans
        if any(anchor in span.text for anchor in IMPORTANT_FIELD_ANCHORS) and span.span_id not in consumed
    ]


def merge_block_retry_body(primary: dict[str, Any], retry: dict[str, Any]) -> dict[str, Any]:
    result = dict(primary)
    fields = []
    seen_ranges: set[tuple[str, str | None, tuple[tuple[str, int | None, int | None], ...]]] = set()
    for field in primary.get("fields", []):
        if not isinstance(field, dict):
            continue
        marker = _field_range_signature(field)
        if marker not in seen_ranges:
            seen_ranges.add(marker)
            fields.append(field)
    for field in retry.get("fields", []):
        if not isinstance(field, dict):
            continue
        marker = _field_range_signature(field)
        if marker not in seen_ranges:
            seen_ranges.add(marker)
            fields.append(field)
    result["fields"] = fields
    for key in ("entities", "tables", "requirements", "ignored_nodes", "unknown_nodes", "layout_candidate_decisions", "node_scope_decisions"):
        result[key] = _unique([*primary.get(key, []), *retry.get(key, [])])
    return result


def _chunk_ids(source_ids: list[str], span_by_id: dict[str, TextSpan]) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    character_count = 0
    for source_id in source_ids:
        span_size = len(span_by_id[source_id].text)
        if current and (len(current) >= MAX_BLOCK_SPANS or character_count + span_size > MAX_BLOCK_CHARS):
            chunks.append(current)
            current = []
            character_count = 0
        current.append(source_id)
        character_count += span_size
    if current:
        chunks.append(current)
    return chunks


def _field_span_ids(field: dict[str, Any]) -> list[str]:
    if field.get("span_id") or field.get("source_span_id"):
        return [str(field.get("span_id") or field.get("source_span_id"))]
    for key in ("ranges", "value", "source_ranges", "span_ranges"):
        ranges = field.get(key)
        if isinstance(ranges, list):
            return [str(item.get("span_id") or item.get("source_span_id")) for item in ranges if isinstance(item, dict) and (item.get("span_id") or item.get("source_span_id"))]
    return []


def _field_range_signature(field: dict[str, Any]) -> tuple[str, str | None, tuple[tuple[str, int | None, int | None], ...]]:
    entity_id = str(field.get("entity_id")) if field.get("entity_id") else None
    ranges = field.get("ranges")
    if not isinstance(ranges, list):
        ranges = field.get("source_ranges") or field.get("span_ranges")
    if isinstance(ranges, list):
        source_ranges = tuple(
            (
                str(item.get("span_id") or item.get("source_span_id")),
                item.get("start_offset", item.get("start")),
                item.get("end_offset", item.get("end")),
            )
            for item in ranges
            if isinstance(item, dict) and (item.get("span_id") or item.get("source_span_id"))
        )
    else:
        span_id = field.get("span_id") or field.get("source_span_id")
        source_ranges = (
            (
                str(span_id),
                field.get("start_offset", field.get("start")),
                field.get("end_offset", field.get("end")),
            ),
        ) if span_id else ()
    return str(field.get("semantic_key", "")), entity_id, source_ranges


def _known_unassigned(source_ids: Iterable[str], span_by_id: dict[str, TextSpan], assigned: set[str]) -> list[str]:
    return [str(source_id) for source_id in source_ids if str(source_id) in span_by_id and str(source_id) not in assigned]


def _merge_objects(bodies: list[dict[str, Any]], key: str, identity_key: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for body in bodies:
        for item in body.get(key, []):
            if not isinstance(item, dict):
                continue
            identity = str(item.get(identity_key) or item.get("name") or len(merged))
            merged.setdefault(identity, item)
    return list(merged.values())


def _dedupe_decisions(decisions: list[Any]) -> list[Any]:
    merged: dict[str, Any] = {}
    for decision in decisions:
        if isinstance(decision, dict):
            key = str(decision.get("layout_candidate_id") or len(merged))
            merged.setdefault(key, decision)
    return list(merged.values())


def _unique(values: Iterable[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        marker = repr(value)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
