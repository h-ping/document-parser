import unittest

from document_parser.models import ExtractionPlan, FieldPlan, PageInfo, SpanRange, TextSpan, ValueSource
from document_parser.table_parser import build_table_layers
from document_parser.vdg_quality import (
    apply_vdg_boundary_gate,
    build_pre_agent_vdg_artifacts,
    build_vdg_consumption_report,
    build_vdg_quality_report,
)


class VdgQualityTests(unittest.TestCase):
    def test_candidate_vdg_covers_source_spans(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：牛奶", "pdf_text"),
            TextSpan("span_0002", 1, "净含量：250mL", "pdf_text"),
        ]
        graph, report, context = build_pre_agent_vdg_artifacts([PageInfo(1, 200, 100)], spans, [], {"tables": []})

        self.assertEqual(report["source_span_coverage_rate"], 1.0)
        self.assertEqual(report["edge_ref_status"], "pass")
        self.assertIn("candidate_field_groups", context)
        self.assertEqual({node["node_id"] for node in graph["nodes"] if node["node_type"] == "text_span"}, {"span_0001", "span_0002"})

    def test_vdg_quality_fails_unresolved_edge_ref(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        graph, _, _ = build_pre_agent_vdg_artifacts([PageInfo(1, 200, 100)], spans, [], {"tables": []})
        graph["edges"][0]["target_node_id"] = "span_missing"

        report = build_vdg_quality_report(graph, spans, [], {"tables": []})

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["edge_ref_status"], "fail")

    def test_nutrition_anchor_without_cells_requires_review(self) -> None:
        spans = [TextSpan("span_0001", 1, "营养成分表", "pdf_text")]
        graph, report, context = build_pre_agent_vdg_artifacts([PageInfo(1, 200, 100)], spans, [], {"tables": []})

        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["nutrition_table_candidate_status"], "table_structure_unresolved")
        self.assertTrue(any(issue["issue_type"] == "table_structure_unresolved" for issue in report["issues"]))
        self.assertEqual(context["vdg_quality_status"], "review_required")
        self.assertTrue(graph["nodes"])

    def test_multi_anchor_field_goes_to_boundary_review(self) -> None:
        spans = [TextSpan("span_0001", 1, "配料：水 产品类型：饮料", "pdf_text")]
        graph, _, _ = build_pre_agent_vdg_artifacts([PageInfo(1, 200, 100)], spans, [], {"tables": []})
        plan = ExtractionPlan(
            plan_id="plan_001",
            schema_id="schema_001",
            fields=[
                FieldPlan(
                    field_plan_id="fp_0001",
                    semantic_key="product.ingredients",
                    display_name="配料",
                    field_type="long_text",
                    section_id="sec_label_text",
                    entity_id="product_001",
                    value_source=ValueSource("span_ranges", [SpanRange("span_0001", 0, len(spans[0].text))]),
                    criticality="critical",
                    confidence={"overall": 0.95},
                )
            ],
        )

        filtered_plan, rejected, review, checks = apply_vdg_boundary_gate(plan, graph, spans)

        self.assertEqual(len(filtered_plan.fields), 1)
        self.assertEqual(rejected, [])
        self.assertTrue(review)
        self.assertTrue(any(check["check_type"] == "vdg_boundary_validation" and check["result"] == "failed" for check in checks))

    def test_consumption_statuses_include_extracted_unknown_and_conflict(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：牛奶", "pdf_text"),
            TextSpan("span_0002", 1, "净含量：250mL", "pdf_text"),
            TextSpan("span_0003", 1, "配料：水", "pdf_text"),
        ]
        graph, _, _ = build_pre_agent_vdg_artifacts([PageInfo(1, 200, 100)], spans, [], {"tables": []})
        plan = ExtractionPlan(
            plan_id="plan_001",
            schema_id="schema_001",
            fields=[
                _field("fp_0001", "product.name", "品名", "span_0001"),
                _field("fp_0002", "custom.duplicate", "重复", "span_0001"),
            ],
            ignored_nodes=["span_0002"],
        )

        consumed_graph, report = build_vdg_consumption_report(graph, plan, [], [], [])

        statuses = {node["node_id"]: node["status"] for node in consumed_graph["nodes"] if node["node_type"] == "text_span"}
        self.assertEqual(statuses["span_0001"], "conflict")
        self.assertEqual(statuses["span_0002"], "ignored")
        self.assertEqual(statuses["span_0003"], "unknown")
        self.assertEqual(report["conflict_node_count"], 1)
        self.assertEqual(report["unknown_important_node_count"], 1)


def _field(field_plan_id: str, semantic_key: str, display_name: str, span_id: str) -> FieldPlan:
    return FieldPlan(
        field_plan_id=field_plan_id,
        semantic_key=semantic_key,
        display_name=display_name,
        field_type="string",
        section_id="sec_label_text",
        entity_id="product_001",
        value_source=ValueSource("span_ranges", [SpanRange(span_id, 0, 5)]),
        criticality="critical",
        confidence={"overall": 0.95},
    )


if __name__ == "__main__":
    unittest.main()
