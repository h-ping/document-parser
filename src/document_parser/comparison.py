from __future__ import annotations

from typing import Any

from .utils import stable_id


COMPARISON_DIMENSIONS = [
    "semantic_key",
    "normalized_value",
    "value_hash",
    "section_id",
    "entity_id",
    "table_id",
    "row_key",
    "bbox_normalized",
    "evidence_refs",
]


def build_comparison_index(standard_items: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    skipped = []
    for item in standard_items:
        if not item.get("comparison_required", True):
            skipped.append(
                {
                    "standard_item_id": item.get("id"),
                    "field_id": item.get("field_id"),
                    "semantic_key": item.get("semantic_key"),
                    "reason": "comparison_not_required",
                }
            )
            continue
        profile = item.get("comparison_profile") if isinstance(item.get("comparison_profile"), dict) else _profile_from_item(item)
        entries.append(
            {
                "comparison_id": stable_id("cmp", len(entries) + 1),
                "standard_item_id": item.get("id"),
                "field_id": item.get("field_id"),
                "field": item.get("field"),
                "semantic_key": item.get("semantic_key"),
                "label": item.get("label"),
                "text": item.get("text"),
                "normalized_text": item.get("normalized_text"),
                "comparison_key": _comparison_key(profile),
                "matching_dimensions": profile,
                "source": item.get("source"),
                "evidence_refs": item.get("evidence_refs", []),
                "revision_role": item.get("revision_role"),
            }
        )

    return {
        "artifact_version": "comparison_index_v0.1",
        "status": "ready" if entries else "empty",
        "dimension_contract": COMPARISON_DIMENSIONS,
        "entry_count": len(entries),
        "skipped_count": len(skipped),
        "entries": entries,
        "skipped_items": skipped,
    }


def _profile_from_item(item: dict[str, Any]) -> dict[str, Any]:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    return {
        "semantic_key": item.get("semantic_key"),
        "normalized_value": item.get("normalized_text"),
        "value_hash": item.get("value_hash"),
        "section_id": source.get("section"),
        "entity_id": item.get("group_id"),
        "table_id": item.get("table_id"),
        "row_key": item.get("row_key"),
        "bbox_normalized": source.get("bbox_normalized"),
        "evidence_refs": item.get("evidence_refs", []),
    }


def _comparison_key(profile: dict[str, Any]) -> str:
    parts = [
        profile.get("semantic_key"),
        profile.get("section_id"),
        profile.get("entity_id"),
        profile.get("table_id"),
        profile.get("row_key"),
        profile.get("value_hash"),
    ]
    return "|".join("" if part is None else str(part) for part in parts)
