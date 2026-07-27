import unittest

from document_parser.label_text_scope import (
    OUT_OF_SCOPE_STATUS,
    UNKNOWN_SCOPE_STATUS,
    apply_label_text_scope_gate,
    build_label_text_scope_agent_context,
    load_label_text_scope_reference,
)
from document_parser.models import BBoxNormalized, ExtractionPlan, FieldPlan, SpanRange, TextSpan, ValueSource


def _field(span_id: str, text: str, semantic_key: str = "custom.note") -> FieldPlan:
    return FieldPlan(
        field_plan_id="fp_0001",
        semantic_key=semantic_key,
        display_name="测试字段",
        field_type="string",
        section_id="sec_label_text",
        entity_id=None,
        value_source=ValueSource(mode="span_ranges", ranges=[SpanRange(span_id, 0, len(text))]),
        criticality="non_critical",
        confidence={"schema_confidence": 0.95, "boundary_confidence": 0.95, "entity_linking_confidence": 0.95},
    )


class LabelTextScopeTests(unittest.TestCase):
    def test_reference_and_agent_context_include_template_scope(self) -> None:
        reference = load_label_text_scope_reference()
        context = build_label_text_scope_agent_context(reference)

        fields = {item["semantic_key"] for item in reference["field_catalog"]}
        tables = {item["table_type"] for item in reference["table_catalog"]}

        self.assertIn("product.name", fields)
        self.assertIn("product.ingredients", fields)
        self.assertIn("nutrition_facts", tables)
        self.assertTrue(context["reference_is_not_evidence"])
        self.assertIn("Only final printed packaging label text", context["primary_rule"])
        self.assertTrue(context["in_scope_categories"])
        self.assertTrue(context["out_of_scope_categories"])

    def test_scope_gate_rejects_obvious_artwork_noise(self) -> None:
        reference = load_label_text_scope_reference()
        text = "制稿说明：刀模线不印刷"
        spans = [TextSpan("span_0001", 1, text, "pdf_text", bbox_normalized=BBoxNormalized(0.1, 0.1, 0.2, 0.2))]
        plan = ExtractionPlan("plan_001", "schema_001", [_field("span_0001", text)])

        gated, rejected, review, checks, report = apply_label_text_scope_gate(plan, spans, reference)

        self.assertEqual(gated.fields, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(review, [])
        self.assertIn("span_0001", gated.ignored_nodes)
        self.assertEqual(gated.ignored_node_reasons["span_0001"], "scope_gate_rejected_out_of_scope_candidate")
        self.assertEqual(report["scope_gate_rejected_count"], 1)
        self.assertEqual(report["extracted_out_of_scope_count"], 0)
        self.assertTrue(any(check["check_type"] == "label_text_scope_gate" and check["result"] == "failed" for check in checks))
        self.assertEqual(report["ignored_noise_nodes"][0]["scope_status"], OUT_OF_SCOPE_STATUS)

    def test_scope_gate_keeps_unknown_scope_but_routes_review(self) -> None:
        reference = load_label_text_scope_reference()
        text = "宣传语：待确认"
        spans = [TextSpan("span_0001", 1, text, "pdf_text")]
        plan = ExtractionPlan("plan_001", "schema_001", [_field("span_0001", text, "custom.marketing_claims")])

        gated, rejected, review, checks, report = apply_label_text_scope_gate(plan, spans, reference)

        self.assertEqual(len(gated.fields), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(len(review), 1)
        self.assertIn("span_0001", gated.unknown_nodes)
        self.assertEqual(report["unknown_scope_node_count"], 1)
        self.assertTrue(any(check["check_type"] == "label_text_scope_unknown" and check["result"] == "failed" for check in checks))
        self.assertEqual(report["unknown_scope_nodes"][0]["scope_status"], UNKNOWN_SCOPE_STATUS)

    def test_scope_gate_does_not_hard_delete_non_template_printed_text(self) -> None:
        reference = load_label_text_scope_reference()
        text = "清甜好喝"
        spans = [TextSpan("span_0001", 1, text, "pdf_text")]
        plan = ExtractionPlan("plan_001", "schema_001", [_field("span_0001", text, "custom.marketing_claims")])

        gated, rejected, review, checks, report = apply_label_text_scope_gate(plan, spans, reference)

        self.assertEqual(len(gated.fields), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(review, [])
        self.assertEqual(report["scope_gate_rejected_count"], 0)
        self.assertTrue(any(check["check_type"] == "label_text_scope_gate" and check["result"] == "passed" for check in checks))

    def test_scope_gate_rejects_template_placeholder_value(self) -> None:
        reference = load_label_text_scope_reference()
        text = "填写"
        spans = [TextSpan("span_0001", 1, text, "pdf_text")]
        plan = ExtractionPlan("plan_001", "schema_001", [_field("span_0001", text)])

        gated, rejected, _, _, report = apply_label_text_scope_gate(plan, spans, reference)

        self.assertEqual(gated.fields, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(report["ignored_noise_nodes"][0]["scope_category"], "template_placeholder")


if __name__ == "__main__":
    unittest.main()
