import unittest

from document_parser.models import (
    BBoxNormalized,
    BBoxPdf,
    Evidence,
    ExtractionPlan,
    FieldPlan,
    PageInfo,
    SpanRange,
    TextSpan,
    ValueSource,
)
from document_parser.vdg import build_visual_document_graph


class VisualDocumentGraphTests(unittest.TestCase):
    def test_builds_nodes_edges_and_span_statuses(self) -> None:
        page = PageInfo(page=1, width=200, height=100)
        spans = [
            TextSpan(
                "span_0001",
                1,
                "品名：牛奶",
                "pdf_text",
                bbox_pdf=BBoxPdf(10, 10, 40, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.05, 0.1, 0.25, 0.2),
            ),
            TextSpan(
                "span_0002",
                1,
                "营养成分表",
                "pdf_text",
                bbox_pdf=BBoxPdf(10, 30, 50, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.05, 0.3, 0.3, 0.4),
            ),
        ]
        plan = ExtractionPlan(
            plan_id="plan_0001",
            schema_id="schema_0001",
            fields=[
                FieldPlan(
                    field_plan_id="fp_0001",
                    semantic_key="product.name",
                    display_name="品名",
                    field_type="text",
                    section_id=None,
                    entity_id=None,
                    value_source=ValueSource("span_ranges", [SpanRange("span_0001", 0, 5)]),
                    criticality="critical",
                    confidence={"overall": 0.95},
                )
            ],
        )
        table_layers = {
            "tables": [
                {
                    "table_layer_id": "tl_tbl_0001",
                    "parser": "text_span_nutrition",
                    "table_type": "nutrition_facts",
                    "page": 1,
                    "title": "营养成分表",
                    "source_span_ids": ["span_0002"],
                    "rows": [
                        {
                            "row_index": 1,
                            "row_type": "data",
                            "row_key": "energy",
                            "source_span_ids": ["span_0002"],
                            "cells": [
                                {
                                    "cell_id": "cell_001_001",
                                    "row_index": 1,
                                    "col_index": 0,
                                    "text": "能量",
                                    "source_span_ids": ["span_0002"],
                                    "page": 1,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        tables = [{"table_id": "tbl_0001", "source_span_ids": ["span_0002"]}]
        evidence = [
            Evidence(
                evidence_id="ev_0001",
                source_text="品名：牛奶",
                page=1,
                extraction_methods=["pdf_text"],
                bbox_status="available",
                source_node_ids=["span_0001"],
            )
        ]

        graph = build_visual_document_graph(
            [page],
            spans,
            [{"region_id": "reg_0001", "region_type": "nutrition_table_area", "page": 1, "source_span_ids": ["span_0002"]}],
            table_layers,
            plan,
            tables,
            [],
            evidence,
        )

        node_types = graph["node_types"]
        edge_types = graph["edge_types"]
        nodes_by_id = {node["node_id"]: node for node in graph["nodes"]}

        self.assertEqual(node_types["page"], 1)
        self.assertEqual(node_types["text_span"], 2)
        self.assertEqual(node_types["region"], 1)
        self.assertEqual(node_types["table"], 1)
        self.assertEqual(node_types["table_row"], 1)
        self.assertEqual(node_types["table_cell"], 1)
        self.assertGreaterEqual(edge_types["contains"], 1)
        self.assertGreaterEqual(edge_types["reading_order_next"], 1)
        self.assertGreaterEqual(edge_types["belongs_to_table"], 1)
        self.assertEqual(nodes_by_id["span_0001"]["status"], "assigned_to_field")
        self.assertEqual(nodes_by_id["span_0002"]["status"], "assigned_to_table")


if __name__ == "__main__":
    unittest.main()
