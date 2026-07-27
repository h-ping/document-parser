from __future__ import annotations

from typing import Any

from .models import BBoxPdf, PageInfo, TextSpan, to_jsonable
from .utils import stable_id


_PRODUCER_ANCHORS = ("委托", "受托", "地址", "产地", "许可证编号", "生产者", "生产商")
_NUTRITION_LABELS = ("能量", "蛋白质", "蛋⽩质", "脂肪", "碳水化合物", "碳⽔化合物", "钠")


def build_layout_candidates(spans: list[TextSpan], pages: list[PageInfo]) -> dict[str, Any]:
    page_by_number = {page.page: page for page in pages}
    nutrition = _nutrition_candidates(spans, page_by_number)
    producers = _producer_candidates(spans)
    artifact = {
        "artifact_version": "layout_candidates_v0.1",
        "source_nodes": [_source_node(span) for span in spans],
        "table_candidates": [*nutrition, *producers],
        "reading_order_candidates": _reading_order_candidates(spans),
        "side_marker_candidates": _side_marker_candidates(spans),
        "quality_issues": [],
    }
    artifact["cross_page_candidate_count"] = sum(
        1 for candidate in nutrition if candidate.get("cross_page_table_continuation_suspected")
    )
    return artifact


def build_candidate_table_layers(layout_candidates: dict[str, Any]) -> dict[str, Any]:
    return {
        "parsers": ["layout_geometry_candidate"],
        "tables": list(layout_candidates.get("table_candidates", [])),
        "parser_issues": list(layout_candidates.get("quality_issues", [])),
        "candidate_only": True,
    }


def validate_layout_candidates(artifact: dict[str, Any], spans: list[TextSpan]) -> dict[str, Any]:
    span_ids = [span.span_id for span in spans]
    known_ids = set(span_ids)
    duplicate_ids = sorted({span_id for span_id in span_ids if span_ids.count(span_id) > 1})
    invalid_bbox_ids = sorted(span.span_id for span in spans if not _valid_bbox(span.bbox_pdf))
    refs = _all_source_refs(artifact)
    unresolved_refs = sorted(set(refs) - known_ids)
    missing_nodes = sorted(known_ids - {str(node.get("span_id")) for node in artifact.get("source_nodes", [])})
    review_issues = list(artifact.get("quality_issues", []))
    for table in artifact.get("table_candidates", []):
        if table.get("table_type") == "nutrition_facts" and not any(
            row.get("row_type") == "data" for row in table.get("rows", [])
        ):
            review_issues.append(
                {
                    "issue_type": "table_structure_unresolved",
                    "severity": "high",
                    "table_candidate_id": table.get("table_candidate_id"),
                }
            )
        if table.get("cross_page_table_continuation_suspected"):
            review_issues.append(
                {
                    "issue_type": "cross_page_table_continuation_suspected",
                    "severity": "medium",
                    "table_candidate_id": table.get("table_candidate_id"),
                    "bbox_normalized": table.get("bbox_normalized"),
                }
            )
    blocking = bool(duplicate_ids or invalid_bbox_ids or unresolved_refs or missing_nodes)
    status = "fail" if blocking else "review_required" if review_issues else "pass"
    return {
        "report_version": "layout_quality_v0.1",
        "status": status,
        "source_span_count": len(spans),
        "source_span_coverage_rate": round((len(known_ids) - len(missing_nodes)) / len(known_ids), 4) if known_ids else 0.0,
        "duplicate_atom_id_count": len(duplicate_ids),
        "invalid_bbox_count": len(invalid_bbox_ids),
        "unresolved_source_ref_count": len(unresolved_refs),
        "layout_boundary_issue_count": len(review_issues),
        "issues": [
            *([{"issue_type": "duplicate_atom_id", "severity": "high", "span_ids": duplicate_ids}] if duplicate_ids else []),
            *([{"issue_type": "invalid_atom_bbox", "severity": "high", "span_ids": invalid_bbox_ids}] if invalid_bbox_ids else []),
            *([{"issue_type": "unresolved_candidate_source_ref", "severity": "high", "source_span_ids": unresolved_refs}] if unresolved_refs else []),
            *([{"issue_type": "source_span_missing_from_layout", "severity": "high", "source_span_ids": missing_nodes}] if missing_nodes else []),
            *review_issues,
        ],
    }


def _nutrition_candidates(spans: list[TextSpan], pages: dict[int, PageInfo]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    titles = [span for span in spans if "营养成分表" in span.text and span.bbox_pdf]
    for index, title in enumerate(sorted(titles, key=_sort_key), start=1):
        assert title.bbox_pdf is not None
        page = pages.get(title.page)
        if not page:
            continue
        page_titles = [item for item in titles if item.page == title.page and item.bbox_pdf]
        x1, x2 = _horizontal_partition(title, page_titles, page.width)
        y1 = max(0.0, title.bbox_pdf.y - 6.0)
        y2 = _vertical_limit(title, page_titles, x1, x2, page.height)
        related = [span for span in spans if _inside_window(span, title.page, x1, x2, y1, y2)]
        related.sort(key=_sort_key)
        rows = _rows_from_spans(related, f"layout_tbl_{index:04d}")
        labels = {label for label in _NUTRITION_LABELS if any(label in span.text for span in related)}
        content_bottom = max(
            (span.bbox_pdf.y + span.bbox_pdf.height for span in related if span.bbox_pdf),
            default=title.bbox_pdf.y + title.bbox_pdf.height,
        )
        near_bottom = content_bottom >= page.height - 60 and len(labels) < 5
        candidate_id = stable_id("layout_nutrition", index)
        candidates.append(
            {
                "table_candidate_id": candidate_id,
                "table_layer_id": candidate_id,
                "parser": "layout_geometry_candidate",
                "table_type": "nutrition_facts",
                "page": title.page,
                "title": title.text,
                "title_span_id": title.span_id,
                "columns": [],
                "rows": rows,
                "source_span_ids": [span.span_id for span in related],
                "bbox_pdf": _union_bbox(related),
                "bbox_normalized": _union_normalized(related),
                "bbox_status": "available",
                "confidence": 0.72,
                "candidate_basis": "character_atom_geometry_high_recall",
                "cross_page_table_continuation_suspected": near_bottom,
                "candidate_window": {"x1": x1, "x2": x2, "y1": y1, "y2": y2},
            }
        )
    _attach_cross_page_continuations(candidates, spans, pages)
    return candidates


def _attach_cross_page_continuations(candidates: list[dict[str, Any]], spans: list[TextSpan], pages: dict[int, PageInfo]) -> None:
    for candidate in candidates:
        if not candidate.get("cross_page_table_continuation_suspected"):
            continue
        continuation_page = int(candidate.get("page", 0)) + 1
        page = pages.get(continuation_page)
        if page is None:
            candidate["cross_page_pair_status"] = "unresolved"
            continue
        next_titles = [span for span in spans if span.page == continuation_page and "营养成分表" in span.text and span.bbox_pdf]
        y_limit = min((span.bbox_pdf.y for span in next_titles if span.bbox_pdf), default=page.height * 0.25)
        window = candidate.get("candidate_window", {})
        x1 = float(window.get("x1", 0.0))
        x2 = float(window.get("x2", page.width))
        leading = [
            span
            for span in spans
            if span.page == continuation_page
            and span.bbox_pdf
            and span.bbox_pdf.y < y_limit - 2.0
            and x1 <= _center_x(span.bbox_pdf) <= x2
        ]
        continuation_groups = [
            group
            for group in _group_rows(leading)
            if any(label in " ".join(span.text for span in group) for label in _NUTRITION_LABELS)
        ]
        continuation = [span for group in continuation_groups for span in group]
        candidate["continuation_page"] = continuation_page
        candidate["continuation_source_span_ids"] = [span.span_id for span in continuation]
        candidate["cross_page_pair_status"] = "candidate" if continuation else "unresolved"
        if continuation:
            candidate["source_span_ids"] = [*candidate.get("source_span_ids", []), *[span.span_id for span in continuation]]
            candidate["rows"] = [*candidate.get("rows", []), *_rows_from_spans(continuation, f"{candidate['table_candidate_id']}_continuation")]
            candidate["continuation_bbox_pdf"] = _union_bbox(continuation)
            candidate["continuation_bbox_normalized"] = _union_normalized(continuation)


def _producer_candidates(spans: list[TextSpan]) -> list[dict[str, Any]]:
    candidates = []
    pages = sorted({span.page for span in spans})
    for page in pages:
        related = [span for span in spans if span.page == page and any(anchor in span.text for anchor in _PRODUCER_ANCHORS)]
        if not related:
            continue
        related = _producer_region_spans(spans, page, related)
        candidate_id = stable_id("layout_producer", len(candidates) + 1)
        candidates.append(
            {
                "table_candidate_id": candidate_id,
                "table_layer_id": candidate_id,
                "parser": "layout_geometry_candidate",
                "table_type": "producer_info_repeated_rows",
                "page": page,
                "title": "producer_info_repeated_rows",
                "columns": [],
                "rows": _rows_from_spans(related, candidate_id),
                "source_span_ids": [span.span_id for span in related],
                "bbox_pdf": _union_bbox(related),
                "bbox_normalized": _union_normalized(related),
                "bbox_status": "available",
                "confidence": 0.68,
                "candidate_basis": "producer_anchor_keywords_and_visual_rows",
            }
        )
    return candidates


def _producer_region_spans(spans: list[TextSpan], page: int, anchors: list[TextSpan]) -> list[TextSpan]:
    geometric_anchors = [span for span in anchors if span.bbox_pdf]
    if not geometric_anchors:
        return sorted(anchors, key=_sort_key)
    x1 = min(span.bbox_pdf.x for span in geometric_anchors if span.bbox_pdf) - 2.0
    x2 = max(span.bbox_pdf.x + span.bbox_pdf.width for span in geometric_anchors if span.bbox_pdf) + 2.0
    y1 = min(span.bbox_pdf.y for span in geometric_anchors if span.bbox_pdf) - 2.0
    last_anchor_y = max(span.bbox_pdf.y for span in geometric_anchors if span.bbox_pdf)
    in_column = sorted(
        (
            span
            for span in spans
            if span.page == page
            and span.bbox_pdf
            and span.bbox_pdf.y >= y1
            and x1 <= _center_x(span.bbox_pdf) <= x2
        ),
        key=_sort_key,
    )
    selected = [span for span in in_column if span.bbox_pdf and span.bbox_pdf.y <= last_anchor_y + 2.0]
    previous = selected[-1] if selected else geometric_anchors[-1]
    for span in in_column:
        if span in selected or span.bbox_pdf is None or span.bbox_pdf.y <= last_anchor_y + 2.0:
            continue
        if previous.bbox_pdf is None:
            break
        gap = span.bbox_pdf.y - (previous.bbox_pdf.y + previous.bbox_pdf.height)
        if gap > max(24.0, previous.bbox_pdf.height * 2.5):
            break
        selected.append(span)
        previous = span
    return selected


def _rows_from_spans(spans: list[TextSpan], table_id: str) -> list[dict[str, Any]]:
    rows = []
    for row_index, group in enumerate(_group_rows(spans), start=1):
        cells = [
            {
                "cell_id": f"{table_id}_cell_{row_index:03d}_{column_index:03d}",
                "row_index": row_index,
                "col_index": column_index,
                "text": span.text,
                "source_span_ids": [span.span_id],
                "bbox_pdf": to_jsonable(span.bbox_pdf),
                "bbox_normalized": to_jsonable(span.bbox_normalized),
                "bbox_status": "available",
            }
            for column_index, span in enumerate(group, start=1)
        ]
        text = " | ".join(span.text for span in group)
        rows.append(
            {
                "row_index": row_index,
                "row_type": _row_type(text),
                "text": text,
                "source_span_ids": [span.span_id for span in group],
                "cells": cells,
            }
        )
    return rows


def _group_rows(spans: list[TextSpan]) -> list[list[TextSpan]]:
    groups: list[list[TextSpan]] = []
    for span in sorted((item for item in spans if item.bbox_pdf), key=_sort_key):
        assert span.bbox_pdf is not None
        group = next((items for items in groups if _same_row(items[0], span)), None)
        if group is None:
            groups.append([span])
        else:
            group.append(span)
    return [sorted(group, key=lambda span: span.bbox_pdf.x if span.bbox_pdf else 0) for group in groups]


def _same_row(left: TextSpan, right: TextSpan) -> bool:
    assert left.bbox_pdf is not None and right.bbox_pdf is not None
    tolerance = max(left.bbox_pdf.height, right.bbox_pdf.height, 2.0) * 0.75
    return abs(_center_y(left.bbox_pdf) - _center_y(right.bbox_pdf)) <= tolerance


def _row_type(text: str) -> str:
    if any(anchor in text for anchor in ("营养成分表", "项目", "每100", "NRV", "营养素参考值")):
        return "header"
    if any(label in text for label in _NUTRITION_LABELS):
        return "data"
    return "data" if "：" in text or ":" in text else "unknown"


def _horizontal_partition(title: TextSpan, titles: list[TextSpan], page_width: float) -> tuple[float, float]:
    assert title.bbox_pdf is not None
    center = _center_x(title.bbox_pdf)
    same_band = sorted(
        [item for item in titles if item.bbox_pdf and abs(_center_y(item.bbox_pdf) - _center_y(title.bbox_pdf)) <= 30],
        key=lambda item: _center_x(item.bbox_pdf),
    )
    position = same_band.index(title)
    left = 0.0 if position == 0 else (center + _center_x(same_band[position - 1].bbox_pdf)) / 2
    right = page_width if position == len(same_band) - 1 else (center + _center_x(same_band[position + 1].bbox_pdf)) / 2
    return left, right


def _vertical_limit(title: TextSpan, titles: list[TextSpan], x1: float, x2: float, page_height: float) -> float:
    assert title.bbox_pdf is not None
    later = [
        item for item in titles
        if item is not title and item.bbox_pdf and item.bbox_pdf.y > title.bbox_pdf.y
        and x1 <= _center_x(item.bbox_pdf) <= x2
    ]
    next_title_y = min((item.bbox_pdf.y for item in later if item.bbox_pdf), default=page_height + 6)
    return min(page_height, title.bbox_pdf.y + 135, next_title_y - 6)


def _inside_window(span: TextSpan, page: int, x1: float, x2: float, y1: float, y2: float) -> bool:
    return bool(
        span.page == page and span.bbox_pdf
        and x1 <= _center_x(span.bbox_pdf) <= x2
        and y1 <= _center_y(span.bbox_pdf) <= y2
    )


def _reading_order_candidates(spans: list[TextSpan]) -> list[dict[str, Any]]:
    ordered = sorted((span for span in spans if span.bbox_pdf), key=_sort_key)
    return [
        {"edge_id": stable_id("layout_reading_order", index), "source_span_id": left.span_id, "target_span_id": right.span_id}
        for index, (left, right) in enumerate(zip(ordered, ordered[1:]), start=1)
    ]


def _side_marker_candidates(spans: list[TextSpan]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": f"layout_side_marker_{span.span_id}",
            "candidate_type": "side_marker",
            "text": span.text,
            "source_span_ids": [span.span_id],
            "bbox_normalized": to_jsonable(span.bbox_normalized),
        }
        for span in spans if span.bbox_pdf and len(span.text.strip()) <= 2 and span.bbox_pdf.x < 40
    ]


def _source_node(span: TextSpan) -> dict[str, Any]:
    return {"span_id": span.span_id, "page": span.page, "text": span.text, "source": span.source, "bbox_pdf": to_jsonable(span.bbox_pdf), "bbox_normalized": to_jsonable(span.bbox_normalized)}


def _all_source_refs(artifact: dict[str, Any]) -> list[str]:
    refs = []
    for table in artifact.get("table_candidates", []):
        refs.extend(str(value) for value in table.get("source_span_ids", []))
        for row in table.get("rows", []):
            refs.extend(str(value) for value in row.get("source_span_ids", []))
            for cell in row.get("cells", []):
                refs.extend(str(value) for value in cell.get("source_span_ids", []))
    for edge in artifact.get("reading_order_candidates", []):
        refs.extend((str(edge.get("source_span_id")), str(edge.get("target_span_id"))))
    for marker in artifact.get("side_marker_candidates", []):
        refs.extend(str(value) for value in marker.get("source_span_ids", []))
    return refs


def _valid_bbox(bbox: BBoxPdf | None) -> bool:
    return bool(bbox and bbox.width > 0 and bbox.height > 0 and bbox.x >= 0 and bbox.y >= 0 and bbox.x + bbox.width <= bbox.page_width + 0.01 and bbox.y + bbox.height <= bbox.page_height + 0.01)


def _union_bbox(spans: list[TextSpan]) -> dict[str, Any] | None:
    boxes = [span.bbox_pdf for span in spans if span.bbox_pdf]
    if not boxes:
        return None
    first = boxes[0]
    x1, y1 = min(box.x for box in boxes), min(box.y for box in boxes)
    x2, y2 = max(box.x + box.width for box in boxes), max(box.y + box.height for box in boxes)
    return to_jsonable(BBoxPdf(x1, y1, x2 - x1, y2 - y1, first.page_width, first.page_height))


def _union_normalized(spans: list[TextSpan]) -> dict[str, float] | None:
    boxes = [span.bbox_normalized for span in spans if span.bbox_normalized]
    if not boxes:
        return None
    return {"x1": min(box.x1 for box in boxes), "y1": min(box.y1 for box in boxes), "x2": max(box.x2 for box in boxes), "y2": max(box.y2 for box in boxes)}


def _sort_key(span: TextSpan) -> tuple[int, float, float]:
    return (span.page, span.bbox_pdf.y if span.bbox_pdf else 0, span.bbox_pdf.x if span.bbox_pdf else 0)


def _center_x(bbox: BBoxPdf) -> float:
    return bbox.x + bbox.width / 2


def _center_y(bbox: BBoxPdf) -> float:
    return bbox.y + bbox.height / 2
