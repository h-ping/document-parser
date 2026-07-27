from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .agents import CANONICAL_PATTERNS, _field_matches_for_span
from .models import ExtractionPlan, FieldPlan, PageInfo, SpanRange, TextSpan, ValueSource, to_jsonable
from .utils import stable_id
from .vdg import build_candidate_visual_document_graph


CONSUMABLE_NODE_TYPES = {"text_span", "table_cell"}
STRUCTURAL_NODE_TYPES = {"page", "region", "table", "table_row"}
IMPORTANT_ANCHORS = (
    "营养成分表",
    "配料",
    "配料表",
    "产品标准",
    "执行标准",
    "净含量",
    "保质期",
    "贮存条件",
    "储存条件",
    "许可证编号",
)


def build_pre_agent_vdg_artifacts(
    pages: list[PageInfo],
    spans: list[TextSpan],
    regions: list[dict[str, Any]],
    table_layers: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph = build_candidate_visual_document_graph(pages, spans, regions, table_layers)
    quality_report = build_vdg_quality_report(graph, spans, regions, table_layers)
    agent_context = build_vdg_agent_context(graph, quality_report, spans, regions, table_layers)
    return graph, quality_report, agent_context


def build_vdg_quality_report(
    graph: dict[str, Any],
    spans: list[TextSpan],
    regions: list[dict[str, Any]],
    table_layers: dict[str, Any],
) -> dict[str, Any]:
    del regions
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    node_ids = {str(node.get("node_id")) for node in nodes if node.get("node_id")}
    text_span_node_ids = {
        str(node.get("node_id"))
        for node in nodes
        if node.get("node_type") == "text_span" and node.get("node_id")
    }
    source_span_ids = {span.span_id for span in spans}
    covered_span_ids = source_span_ids & text_span_node_ids
    page_contains_edges = {
        (str(edge.get("source_node_id")), str(edge.get("target_node_id")))
        for edge in edges
        if edge.get("edge_type") == "contains"
    }
    issues: list[dict[str, Any]] = []

    missing_span_ids = sorted(source_span_ids - covered_span_ids)
    if missing_span_ids:
        _add_issue(
            issues,
            "vdg_source_span_missing",
            "high",
            "Some source spans are not represented as VDG text_span nodes.",
            {"missing_span_ids": missing_span_ids},
        )

    missing_contains = [
        span.span_id
        for span in spans
        if (_page_node_id(span.page), span.span_id) not in page_contains_edges
    ]
    if missing_contains:
        _add_issue(
            issues,
            "vdg_page_contains_missing",
            "high",
            "Some source spans are missing page contains edges.",
            {"missing_span_ids": missing_contains},
        )

    unresolved_edges = [
        {
            "edge_id": edge.get("edge_id"),
            "source_node_id": edge.get("source_node_id"),
            "target_node_id": edge.get("target_node_id"),
        }
        for edge in edges
        if edge.get("source_node_id") not in node_ids or edge.get("target_node_id") not in node_ids
    ]
    if unresolved_edges:
        _add_issue(
            issues,
            "vdg_edge_ref_unresolved",
            "high",
            "Some VDG edges reference missing nodes.",
            {"edges": unresolved_edges[:20], "unresolved_edge_count": len(unresolved_edges)},
        )

    invalid_bbox_nodes = [
        str(node.get("node_id"))
        for node in nodes
        if node.get("bbox_normalized") is not None and not _bbox_normalized_valid(node.get("bbox_normalized"))
    ]
    if invalid_bbox_nodes:
        _add_issue(
            issues,
            "vdg_bbox_out_of_range",
            "high",
            "Some normalized bboxes are outside page bounds.",
            {"node_ids": invalid_bbox_nodes[:50], "node_count": len(invalid_bbox_nodes)},
        )

    for span in spans:
        matches = _field_matches_for_span(span.text)
        if len(matches) >= 2:
            _add_issue(
                issues,
                "vdg_multi_anchor_line",
                "medium",
                "A source span contains multiple field anchors and requires VDG boundary validation.",
                {
                    "node_id": span.span_id,
                    "page": span.page,
                    "anchor_count": len(matches),
                    "anchors": [match["label"] for match in matches],
                    "text": span.text,
                },
            )

    nutrition_anchor_count = sum(1 for span in spans if "营养成分表" in span.text)
    nutrition_tables = [
        table
        for table in table_layers.get("tables", [])
        if isinstance(table, dict) and table.get("table_type") == "nutrition_facts"
    ]
    nutrition_cells = [
        cell
        for table in nutrition_tables
        for row in table.get("rows", [])
        if isinstance(row, dict)
        for cell in row.get("cells", [])
        if isinstance(cell, dict)
    ]
    if nutrition_anchor_count and not nutrition_tables:
        _add_issue(
            issues,
            "table_structure_unresolved",
            "high",
            "Nutrition table anchor exists but no table candidate was built.",
            {"anchor_count": nutrition_anchor_count},
        )
    elif nutrition_anchor_count and not nutrition_cells:
        _add_issue(
            issues,
            "table_structure_unresolved",
            "high",
            "Nutrition table candidate exists but no row/cell nodes were recovered.",
            {"nutrition_table_count": len(nutrition_tables)},
        )

    edge_ref_passed = not unresolved_edges
    source_span_coverage_rate = round(len(covered_span_ids) / len(source_span_ids), 4) if source_span_ids else 0.0
    blocking_issue_count = sum(1 for issue in issues if issue["severity"] == "high" and issue["issue_type"] in {
        "vdg_source_span_missing",
        "vdg_page_contains_missing",
        "vdg_edge_ref_unresolved",
        "vdg_bbox_out_of_range",
    })
    review_issue_count = sum(1 for issue in issues if issue["severity"] in {"high", "medium"} and issue["issue_type"] not in {
        "vdg_source_span_missing",
        "vdg_page_contains_missing",
        "vdg_edge_ref_unresolved",
        "vdg_bbox_out_of_range",
    })
    status = "fail" if blocking_issue_count else "review_required" if review_issue_count else "pass"
    return {
        "report_version": "vdg_quality_v0.1",
        "status": status,
        "agent_readiness": "blocked" if status == "fail" else "usable_with_review" if status == "review_required" else "usable",
        "source_span_count": len(source_span_ids),
        "source_span_coverage_rate": source_span_coverage_rate,
        "page_contains_coverage_rate": round((len(source_span_ids) - len(missing_contains)) / len(source_span_ids), 4) if source_span_ids else 0.0,
        "edge_ref_status": "pass" if edge_ref_passed else "fail",
        "edge_ref_unresolved_count": len(unresolved_edges),
        "bbox_issue_count": len(invalid_bbox_nodes),
        "multi_anchor_line_count": sum(1 for issue in issues if issue["issue_type"] == "vdg_multi_anchor_line"),
        "boundary_issue_count": sum(1 for issue in issues if issue["issue_type"].startswith("vdg_") and "boundary" in issue["issue_type"] or issue["issue_type"] == "vdg_multi_anchor_line"),
        "nutrition_table_candidate_status": _nutrition_table_candidate_status(nutrition_anchor_count, nutrition_tables, nutrition_cells),
        "nutrition_table_count": len(nutrition_tables),
        "nutrition_table_row_count": sum(len(table.get("rows", [])) for table in nutrition_tables),
        "nutrition_table_cell_count": len(nutrition_cells),
        "blocking_issue_count": blocking_issue_count,
        "review_issue_count": review_issue_count,
        "issue_count": len(issues),
        "issues": issues,
        "checks": [
            _check("source_span_coverage_full", source_span_coverage_rate == 1.0, source_span_coverage_rate),
            _check("page_contains_text_span_full", not missing_contains, len(missing_contains)),
            _check("edge_refs_resolve", edge_ref_passed, len(unresolved_edges)),
            _check("bbox_normalized_in_bounds", not invalid_bbox_nodes, len(invalid_bbox_nodes)),
            _check("nutrition_table_candidate_visible", not nutrition_anchor_count or bool(nutrition_tables), _nutrition_table_candidate_status(nutrition_anchor_count, nutrition_tables, nutrition_cells)),
        ],
    }


def build_vdg_agent_context(
    graph: dict[str, Any],
    quality_report: dict[str, Any],
    spans: list[TextSpan],
    regions: list[dict[str, Any]],
    table_layers: dict[str, Any],
) -> dict[str, Any]:
    nodes = {str(node.get("node_id")): node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("node_id")}
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        outgoing.setdefault(str(edge.get("source_node_id")), []).append(edge)

    candidate_groups = []
    for span in spans:
        matches = _field_matches_for_span(span.text)
        if not matches:
            continue
        candidate_groups.append(
            {
                "node_id": span.span_id,
                "page": span.page,
                "text": span.text,
                "anchors": [match["label"] for match in matches],
                "bbox_normalized": to_jsonable(span.bbox_normalized),
                "neighbor_edges": [
                    {
                        "edge_type": edge.get("edge_type"),
                        "target_node_id": edge.get("target_node_id"),
                        "target_text": nodes.get(str(edge.get("target_node_id")), {}).get("text"),
                    }
                    for edge in outgoing.get(span.span_id, [])
                    if edge.get("edge_type") in {"reading_order_next", "same_row", "visual_left_of", "visual_right_of"}
                ][:8],
            }
        )

    table_candidates = []
    for table in table_layers.get("tables", []):
        if not isinstance(table, dict):
            continue
        table_candidates.append(
            {
                "table_layer_id": table.get("table_layer_id"),
                "table_type": table.get("table_type"),
                "page": table.get("page"),
                "title": table.get("title"),
                "source_span_ids": table.get("source_span_ids", []),
                "row_count": len(table.get("rows", [])),
                "cell_count": sum(len(row.get("cells", [])) for row in table.get("rows", []) if isinstance(row, dict)),
                "bbox_status": table.get("bbox_status"),
                "candidate_basis": table.get("candidate_basis"),
                "cross_page_table_continuation_suspected": bool(table.get("cross_page_table_continuation_suspected")),
                "rows": [
                    {
                        "row_index": row.get("row_index"),
                        "row_type": row.get("row_type"),
                        "source_span_ids": row.get("source_span_ids", []),
                        "cells": [
                            {
                                "cell_id": cell.get("cell_id"),
                                "text": cell.get("text"),
                                "source_span_ids": cell.get("source_span_ids", []),
                            }
                            for cell in row.get("cells", [])[:8]
                            if isinstance(cell, dict)
                        ],
                    }
                    for row in table.get("rows", [])[:12]
                    if isinstance(row, dict)
                ],
            }
        )

    return {
        "context_version": "vdg_agent_context_v0.1",
        "vdg_quality_status": quality_report.get("status"),
        "agent_readiness": quality_report.get("agent_readiness"),
        "source_span_count": len(spans),
        "regions": [
            {
                "region_id": region.get("region_id"),
                "region_type": region.get("region_type"),
                "page": region.get("page"),
                "display_name": region.get("display_name"),
                "source_span_ids": region.get("source_span_ids", []),
            }
            for region in regions
        ],
        "candidate_field_groups": candidate_groups[:120],
        "table_candidates": table_candidates,
        "unknown_nodes": _nodes_by_status(graph, "unknown")[:120],
        "conflict_nodes": _nodes_by_status(graph, "conflict")[:120],
        "quality_issues": quality_report.get("issues", [])[:80],
    }


def apply_vdg_boundary_gate(
    plan: ExtractionPlan,
    candidate_graph: dict[str, Any],
    spans: list[TextSpan],
) -> tuple[ExtractionPlan, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    span_by_id = {span.span_id: span for span in spans}
    span_regions = _span_regions(candidate_graph)
    span_table_cells = _span_table_cells(candidate_graph)
    rejected: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    kept_fields: list[FieldPlan] = []

    for field in plan.fields:
        field_issues: list[dict[str, Any]] = []
        hard_reject = False
        range_region_ids: set[str] = set()
        range_table_cell_ids: set[str] = set()
        for span_range in field.value_source.ranges:
            source_span = span_by_id.get(span_range.span_id)
            if source_span is None:
                hard_reject = True
                field_issues.append({"issue_type": "vdg_boundary_unknown_span", "span_id": span_range.span_id})
                continue
            text = source_span.text[span_range.start_offset : span_range.end_offset]
            foreign_anchors = _foreign_anchor_labels(text, field)
            if foreign_anchors:
                field_issues.append(
                    {
                        "issue_type": "vdg_boundary_contains_sibling_anchor",
                        "span_id": span_range.span_id,
                        "foreign_anchors": foreign_anchors,
                        "source_text": text,
                    }
                )
            range_region_ids.update(span_regions.get(span_range.span_id, set()))
            range_table_cell_ids.update(span_table_cells.get(span_range.span_id, set()))

        if len(range_region_ids) > 1 and not field.boundary.get("boundary_exception_reason"):
            field_issues.append(
                {
                    "issue_type": "vdg_region_boundary",
                    "region_ids": sorted(range_region_ids),
                }
            )
        if len(range_table_cell_ids) > 1 and field.semantic_key != "product.nutrition_table":
            field_issues.append(
                {
                    "issue_type": "vdg_table_cell_boundary",
                    "table_cell_ids": sorted(range_table_cell_ids),
                }
            )

        if field_issues:
            review_item = {
                "field_plan_id": field.field_plan_id,
                "semantic_key": field.semantic_key,
                "reason": "vdg_boundary_review_required",
                "issues": field_issues,
            }
            review_items.append(review_item)
            checks.append(
                {
                    "validation_id": stable_id("val_vdg_boundary", len(checks) + 1),
                    "target_id": field.field_plan_id,
                    "check_type": "vdg_boundary_validation",
                    "result": "failed",
                    "severity": "high" if hard_reject else "medium",
                    "message": "VDG boundary validation found a field boundary issue.",
                    "semantic_key": field.semantic_key,
                    "issues": field_issues,
                }
            )
        else:
            checks.append(
                {
                    "validation_id": stable_id("val_vdg_boundary", len(checks) + 1),
                    "target_id": field.field_plan_id,
                    "check_type": "vdg_boundary_validation",
                    "result": "passed",
                    "severity": "info",
                    "semantic_key": field.semantic_key,
                    "issues": [],
                }
            )

        if hard_reject:
            rejected.append(
                {
                    "field_plan_id": field.field_plan_id,
                    "semantic_key": field.semantic_key,
                    "reason": "vdg_boundary_hard_reject",
                    "issues": field_issues,
                }
            )
            continue
        kept_fields.append(field)

    return (
        replace(plan, fields=kept_fields),
        rejected,
        review_items,
        checks,
    )


def build_vdg_consumption_report(
    graph: dict[str, Any],
    plan: ExtractionPlan,
    tables: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    evidence: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    consumed_graph = copy.deepcopy(graph)
    extracted_span_ids = {
        span_range.span_id
        for field in plan.fields
        for span_range in field.value_source.ranges
    }
    for table in tables:
        extracted_span_ids.update(str(span_id) for span_id in table.get("source_span_ids", []) if span_id)
        for row in table.get("rows", []):
            if isinstance(row, dict):
                extracted_span_ids.update(str(span_id) for span_id in row.get("source_span_ids", []) if span_id)
    evidence_span_ids = {
        str(span_id)
        for item in evidence
        for span_id in getattr(item, "source_node_ids", []) if span_id
    }
    for requirement in requirements:
        extracted_span_ids.update(str(span_id) for span_id in requirement.get("source_span_ids", []) if span_id)
    extracted_span_ids.update(evidence_span_ids)

    ignored_node_ids = set(plan.ignored_nodes)
    unknown_node_ids = set(plan.unknown_nodes)
    target_counts: dict[str, int] = {}
    for field in plan.fields:
        for span_range in field.value_source.ranges:
            target_counts[span_range.span_id] = target_counts.get(span_range.span_id, 0) + 1

    node_status_counts: dict[str, int] = {}
    important_unknown_nodes: list[dict[str, Any]] = []
    conflict_nodes: list[dict[str, Any]] = []
    for node in consumed_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_type = node.get("node_type")
        node_id = str(node.get("node_id") or "")
        source_span_ids = {str(span_id) for span_id in node.get("source_span_ids", []) if span_id}
        if node_type in STRUCTURAL_NODE_TYPES:
            status = "structural"
        elif node_id in ignored_node_ids:
            status = "ignored"
            node.setdefault("ignore_reason", getattr(plan, "ignored_node_reasons", {}).get(node_id, "agent_marked_ignored"))
        elif target_counts.get(node_id, 0) > 1:
            status = "conflict"
        elif node_id in extracted_span_ids or bool(source_span_ids & extracted_span_ids):
            status = "extracted"
        elif node_id in unknown_node_ids or _important_node(node):
            status = "unknown"
        else:
            status = "ignored"
            node.setdefault("ignore_reason", "not_matched_to_required_or_agent_field")
        node["status"] = status
        node_status_counts[status] = node_status_counts.get(status, 0) + 1
        if status == "unknown" and _important_node(node):
            important_unknown_nodes.append(_node_summary(node))
        if status == "conflict":
            conflict_nodes.append(_node_summary(node))

    consumable_nodes = [
        node for node in consumed_graph.get("nodes", [])
        if isinstance(node, dict) and node.get("node_type") in CONSUMABLE_NODE_TYPES
    ]
    extracted_nodes = [node for node in consumable_nodes if node.get("status") == "extracted"]
    report = {
        "report_version": "vdg_consumption_v0.1",
        "status": "review_required" if important_unknown_nodes or conflict_nodes else "pass",
        "consumable_node_count": len(consumable_nodes),
        "extracted_node_count": len(extracted_nodes),
        "extracted_node_ids": [node.get("node_id") for node in extracted_nodes if node.get("node_id")],
        "extracted_coverage_rate": round(len(extracted_nodes) / len(consumable_nodes), 4) if consumable_nodes else 0.0,
        "unknown_important_node_count": len(important_unknown_nodes),
        "conflict_node_count": len(conflict_nodes),
        "status_counts": node_status_counts,
        "unknown_important_nodes": important_unknown_nodes[:100],
        "conflict_nodes": conflict_nodes[:100],
    }
    consumed_graph["status_counts"] = node_status_counts
    return consumed_graph, report


def vdg_quality_validation_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "validation_id": stable_id("val_vdg_quality", 1),
            "target_id": "candidate_visual_document_graph",
            "check_type": "vdg_quality",
            "result": "failed" if report.get("status") == "fail" else "passed",
            "severity": "high" if report.get("status") == "fail" else "medium" if report.get("status") == "review_required" else "info",
            "message": "VDG quality gate failed." if report.get("status") == "fail" else "VDG quality gate is usable.",
            "vdg_quality_status": report.get("status"),
            "source_span_coverage_rate": report.get("source_span_coverage_rate"),
            "edge_ref_status": report.get("edge_ref_status"),
            "issue_count": report.get("issue_count", 0),
            "issues": report.get("issues", []),
        }
    ]


def vdg_node_coverage_validation_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    failed = bool(report.get("conflict_node_count", 0))
    return [
        {
            "validation_id": stable_id("val_vdg_coverage", 1),
            "target_id": "visual_document_graph",
            "check_type": "vdg_node_coverage",
            "result": "failed" if failed else "passed",
            "severity": "high" if failed else "medium" if report.get("unknown_important_node_count", 0) else "info",
            "message": "VDG node consumption found conflicts." if failed else "VDG node consumption coverage is traceable.",
            "unknown_important_node_count": report.get("unknown_important_node_count", 0),
            "conflict_node_count": report.get("conflict_node_count", 0),
            "extracted_coverage_rate": report.get("extracted_coverage_rate", 0.0),
        }
    ]


def _add_issue(issues: list[dict[str, Any]], issue_type: str, severity: str, message: str, details: dict[str, Any]) -> None:
    issues.append(
        {
            "issue_id": stable_id("vdg_issue", len(issues) + 1),
            "issue_type": issue_type,
            "severity": severity,
            "message": message,
            "details": details,
            "repair_hint": "Review VDG candidates before schema/extraction if this issue affects important label content.",
        }
    )


def _check(check_type: str, passed: bool, actual: Any) -> dict[str, Any]:
    return {"check_type": check_type, "result": "passed" if passed else "failed", "actual": actual}


def _page_node_id(page: int) -> str:
    return f"page_{page:04d}"


def _bbox_normalized_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        x1 = float(value["x1"])
        y1 = float(value["y1"])
        x2 = float(value["x2"])
        y2 = float(value["y2"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1


def _nutrition_table_candidate_status(anchor_count: int, tables: list[dict[str, Any]], cells: list[dict[str, Any]]) -> str:
    if not anchor_count:
        return "not_applicable"
    if not tables:
        return "table_structure_unresolved"
    if not cells:
        return "rows_or_cells_unresolved"
    return "candidate_available"


def _nodes_by_status(graph: dict[str, Any], status: str) -> list[dict[str, Any]]:
    return [
        _node_summary(node)
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("status") == status
    ]


def _node_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "page": node.get("page"),
        "text": node.get("text"),
        "source_span_ids": node.get("source_span_ids", []),
        "bbox_normalized": node.get("bbox_normalized"),
        "status": node.get("status"),
    }


def _span_regions(graph: dict[str, Any]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("edge_type") != "belongs_to_region":
            continue
        source = str(edge.get("source_node_id"))
        target = str(edge.get("target_node_id"))
        mapping.setdefault(source, set()).add(target)
    return mapping


def _span_table_cells(graph: dict[str, Any]) -> dict[str, set[str]]:
    cell_ids = {
        str(node.get("node_id"))
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("node_type") == "table_cell" and node.get("node_id")
    }
    mapping: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("edge_type") != "belongs_to_table":
            continue
        source = str(edge.get("source_node_id"))
        target = str(edge.get("target_node_id"))
        if target in cell_ids:
            mapping.setdefault(source, set()).add(target)
    return mapping


def _foreign_anchor_labels(text: str, field: FieldPlan) -> list[str]:
    own_labels = {field.display_name}
    for pattern in CANONICAL_PATTERNS:
        if pattern.semantic_key == field.semantic_key:
            own_labels.update(pattern.labels)
    foreign = []
    for match in _field_matches_for_span(text):
        label = match["label"]
        if label not in own_labels and label not in foreign:
            foreign.append(label)
    return foreign


def _important_node(node: dict[str, Any]) -> bool:
    text = str(node.get("text") or "")
    return any(anchor in text for anchor in IMPORTANT_ANCHORS)
