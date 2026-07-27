from __future__ import annotations

from typing import Any

from .models import Evidence, ExtractionPlan, PageInfo, TextSpan, to_jsonable
from .utils import stable_id


def build_candidate_visual_document_graph(
    pages: list[PageInfo],
    spans: list[TextSpan],
    regions: list[dict[str, Any]],
    table_layers: dict[str, Any],
) -> dict[str, Any]:
    graph = build_visual_document_graph(
        pages=pages,
        spans=spans,
        regions=regions,
        table_layers=table_layers,
        plan=ExtractionPlan(plan_id="plan_candidate_empty", schema_id="schema_pending", fields=[]),
        tables=[],
        requirements=[],
        evidence=[],
    )
    graph["graph_id"] = "vdg_candidate_0001"
    graph["graph_role"] = "candidate_pre_agent"
    return graph


def build_visual_document_graph(
    pages: list[PageInfo],
    spans: list[TextSpan],
    regions: list[dict[str, Any]],
    table_layers: dict[str, Any],
    plan: ExtractionPlan,
    tables: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    evidence: list[Evidence],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    span_status = _span_statuses(spans, plan, tables, requirements, evidence)

    for page in pages:
        page_node_id = _page_node_id(page.page)
        nodes.append(
            {
                "node_id": page_node_id,
                "node_type": "page",
                "page": page.page,
                "width": page.width,
                "height": page.height,
                "status": "structural",
            }
        )

    for span in spans:
        page_node_id = _page_node_id(span.page)
        nodes.append(
            {
                "node_id": span.span_id,
                "node_type": "text_span",
                "page": span.page,
                "text": span.text,
                "source": span.source,
                "source_span_ids": [span.span_id],
                "bbox_pdf": to_jsonable(span.bbox_pdf),
                "bbox_normalized": to_jsonable(span.bbox_normalized),
                "status": span_status.get(span.span_id, "unknown"),
            }
        )
        _add_edge(edges, edge_keys, page_node_id, span.span_id, "contains")
        _add_edge(edges, edge_keys, span.span_id, page_node_id, "contained_by")

    for page, page_spans in _spans_by_page(spans).items():
        del page
        ordered = _ordered_spans(page_spans)
        for left, right in zip(ordered, ordered[1:]):
            _add_edge(edges, edge_keys, left.span_id, right.span_id, "reading_order_next")
            if _same_row(left, right):
                _add_edge(edges, edge_keys, left.span_id, right.span_id, "same_row")
                if _left_of(left, right):
                    _add_edge(edges, edge_keys, left.span_id, right.span_id, "visual_left_of")
                    _add_edge(edges, edge_keys, right.span_id, left.span_id, "visual_right_of")
            elif _above(left, right):
                _add_edge(edges, edge_keys, left.span_id, right.span_id, "visual_above")
                _add_edge(edges, edge_keys, right.span_id, left.span_id, "visual_below")

    for region in regions:
        region_id = region.get("region_id")
        if not region_id:
            continue
        page_node_id = _page_node_id(int(region.get("page", 1)))
        nodes.append(
            {
                "node_id": region_id,
                "node_type": "region",
                "page": region.get("page"),
                "region_type": region.get("region_type"),
                "text": region.get("display_name", ""),
                "source_span_ids": region.get("source_span_ids", []),
                "bbox_pdf": to_jsonable(region.get("bbox_pdf")),
                "bbox_normalized": to_jsonable(region.get("bbox_normalized")),
                "status": "structural",
            }
        )
        _add_edge(edges, edge_keys, page_node_id, region_id, "contains")
        _add_edge(edges, edge_keys, region_id, page_node_id, "contained_by")
        for span_id in region.get("source_span_ids", []):
            _add_edge(edges, edge_keys, region_id, span_id, "contains")
            _add_edge(edges, edge_keys, span_id, region_id, "belongs_to_region")

    for table in table_layers.get("tables", []):
        _add_table_nodes_and_edges(nodes, edges, edge_keys, table)

    return {
        "graph_id": "vdg_0001",
        "graph_role": "final",
        "schema_version": "vdg_mvp_v0.1",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": _counts(node["node_type"] for node in nodes),
        "edge_types": _counts(edge["edge_type"] for edge in edges),
        "status_counts": _counts(node.get("status", "unknown") for node in nodes),
        "nodes": nodes,
        "edges": edges,
    }


def _span_statuses(
    spans: list[TextSpan],
    plan: ExtractionPlan,
    tables: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    evidence: list[Evidence],
) -> dict[str, str]:
    status = {span.span_id: "unknown" for span in spans}
    for field in plan.fields:
        for span_range in field.value_source.ranges:
            status[span_range.span_id] = "assigned_to_field"

    for table in tables:
        for span_id in table.get("source_span_ids", []):
            if status.get(span_id) == "unknown":
                status[span_id] = "assigned_to_table"

    evidence_by_id = {item.evidence_id: item for item in evidence}
    for requirement in requirements:
        for evidence_ref in requirement.get("evidence_refs", []):
            source_evidence = evidence_by_id.get(evidence_ref)
            if not source_evidence:
                continue
            for span_id in source_evidence.source_node_ids:
                if status.get(span_id) == "unknown":
                    status[span_id] = "assigned_to_requirement"

    return status


def _add_table_nodes_and_edges(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    edge_keys: set[tuple[str, str, str]],
    table: dict[str, Any],
) -> None:
    table_id = table.get("table_layer_id")
    if not table_id:
        return
    page_node_id = _page_node_id(int(table.get("page") or 1))
    row_ids = []
    cell_ids = []
    cells_by_col: dict[Any, list[str]] = {}
    for row in table.get("rows", []):
        row_id = f"{table_id}_row_{int(row.get('row_index', len(row_ids) + 1)):03d}"
        row_ids.append(row_id)
        nodes.append(
            {
                "node_id": row_id,
                "node_type": "table_row",
                "page": table.get("page"),
                "table_layer_id": table_id,
                "row_index": row.get("row_index"),
                "row_type": row.get("row_type"),
                "row_key": row.get("row_key"),
                "source_span_ids": row.get("source_span_ids", []),
                "status": "assigned_to_table",
            }
        )
        _add_edge(edges, edge_keys, table_id, row_id, "contains")
        _add_edge(edges, edge_keys, row_id, table_id, "contained_by")
        _add_edge(edges, edge_keys, row_id, table_id, "belongs_to_table")

        previous_cell_id = None
        for cell in row.get("cells", []):
            cell_id = f"{table_id}_{cell.get('cell_id', stable_id('cell', len(cell_ids) + 1))}"
            cell_ids.append(cell_id)
            cells_by_col.setdefault(cell.get("col_index"), []).append(cell_id)
            nodes.append(
                {
                    "node_id": cell_id,
                    "node_type": "table_cell",
                    "page": cell.get("page") or table.get("page"),
                    "table_layer_id": table_id,
                    "row_index": cell.get("row_index"),
                    "col_index": cell.get("col_index"),
                    "text": cell.get("text", ""),
                    "source_span_ids": cell.get("source_span_ids", []),
                    "bbox_pdf": to_jsonable(cell.get("bbox_pdf")),
                    "bbox_normalized": to_jsonable(cell.get("bbox_normalized")),
                    "status": "assigned_to_table",
                }
            )
            _add_edge(edges, edge_keys, row_id, cell_id, "contains")
            _add_edge(edges, edge_keys, cell_id, row_id, "contained_by")
            _add_edge(edges, edge_keys, cell_id, table_id, "belongs_to_table")
            if previous_cell_id:
                _add_edge(edges, edge_keys, previous_cell_id, cell_id, "same_row")
                _add_edge(edges, edge_keys, previous_cell_id, cell_id, "reading_order_next")
            previous_cell_id = cell_id
            for span_id in cell.get("source_span_ids", []):
                _add_edge(edges, edge_keys, span_id, cell_id, "belongs_to_table")

    nodes.append(
        {
            "node_id": table_id,
            "node_type": "table",
            "page": table.get("page"),
            "detected_type": table.get("table_type"),
            "parser": table.get("parser"),
            "text": table.get("title", ""),
            "source_span_ids": table.get("source_span_ids", []),
            "row_ids": row_ids,
            "cell_ids": cell_ids,
            "bbox_status": table.get("bbox_status"),
            "status": "assigned_to_table",
        }
    )
    _add_edge(edges, edge_keys, page_node_id, table_id, "contains")
    _add_edge(edges, edge_keys, table_id, page_node_id, "contained_by")
    for span_id in table.get("source_span_ids", []):
        _add_edge(edges, edge_keys, span_id, table_id, "belongs_to_table")

    for column_cells in cells_by_col.values():
        for upper, lower in zip(column_cells, column_cells[1:]):
            _add_edge(edges, edge_keys, upper, lower, "same_column")


def _add_edge(
    edges: list[dict[str, Any]],
    edge_keys: set[tuple[str, str, str]],
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
) -> None:
    key = (source_node_id, target_node_id, edge_type)
    if key in edge_keys:
        return
    edge_keys.add(key)
    edges.append(
        {
            "edge_id": stable_id("edge", len(edges) + 1),
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "edge_type": edge_type,
        }
    )


def _spans_by_page(spans: list[TextSpan]) -> dict[int, list[TextSpan]]:
    grouped: dict[int, list[TextSpan]] = {}
    for span in spans:
        grouped.setdefault(span.page, []).append(span)
    return grouped


def _ordered_spans(spans: list[TextSpan]) -> list[TextSpan]:
    indexed = {span.span_id: index for index, span in enumerate(spans)}
    return sorted(
        spans,
        key=lambda span: (
            span.page,
            span.bbox_pdf.y if span.bbox_pdf else indexed[span.span_id],
            span.bbox_pdf.x if span.bbox_pdf else 0,
            indexed[span.span_id],
        ),
    )


def _same_row(left: TextSpan, right: TextSpan) -> bool:
    if not left.bbox_pdf or not right.bbox_pdf:
        return False
    left_center = left.bbox_pdf.y + left.bbox_pdf.height / 2
    right_center = right.bbox_pdf.y + right.bbox_pdf.height / 2
    tolerance = max(left.bbox_pdf.height, right.bbox_pdf.height, 2.0) * 0.6
    return abs(left_center - right_center) <= tolerance


def _left_of(left: TextSpan, right: TextSpan) -> bool:
    if not left.bbox_pdf or not right.bbox_pdf:
        return False
    return left.bbox_pdf.x <= right.bbox_pdf.x


def _above(left: TextSpan, right: TextSpan) -> bool:
    if not left.bbox_pdf or not right.bbox_pdf:
        return False
    return left.bbox_pdf.y + left.bbox_pdf.height <= right.bbox_pdf.y + max(left.bbox_pdf.height, right.bbox_pdf.height)


def _page_node_id(page: int) -> str:
    return stable_id("page", page)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
