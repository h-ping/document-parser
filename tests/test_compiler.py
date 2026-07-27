import unittest

from document_parser.agents import ExtractionAgent, SchemaInductionAgent
from document_parser.compiler import CompilerError, DeterministicCompiler
from document_parser.models import BBoxNormalized, BBoxPdf, ExtractionPlan, FieldPlan, SpanRange, TextSpan, ValueSource


class CompilerTests(unittest.TestCase):
    def test_compiler_rejects_invalid_span_ranges(self) -> None:
        span = TextSpan("s", 1, "abcdef", "pdf_char_atom")
        for start, end in [(-1, 2), (0, 7), (3, 3), (4, 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(CompilerError):
                DeterministicCompiler().compile(
                    ExtractionPlan(
                        "plan",
                        "schema",
                        [
                            FieldPlan(
                                "field",
                                "custom.text",
                                "文字",
                                "string",
                                None,
                                None,
                                ValueSource("span_ranges", [SpanRange("s", start, end)]),
                                "non_critical",
                                {"schema_confidence": 0.99, "boundary_confidence": 0.99, "entity_linking_confidence": 0.99},
                            )
                        ],
                    ),
                    [span],
                )

    def test_multirange_compiler_joins_same_line_and_preserves_new_line(self) -> None:
        spans = [
            TextSpan("a", 1, "配料：白砂糖，", "pdf_char_atom", BBoxPdf(10, 10, 60, 10, 200, 100)),
            TextSpan("b", 1, "葡萄糖浆", "pdf_char_atom", BBoxPdf(71, 10.2, 35, 10, 200, 100)),
            TextSpan("c", 1, "致敏提示：含乳制品", "pdf_char_atom", BBoxPdf(10, 24, 90, 10, 200, 100)),
        ]
        plan = ExtractionPlan(
            "plan",
            "schema",
            [
                FieldPlan(
                    "field",
                    "product.ingredients",
                    "配料",
                    "long_text",
                    None,
                    None,
                    ValueSource("span_ranges", [SpanRange("a", 0, len(spans[0].text)), SpanRange("b", 0, len(spans[1].text)), SpanRange("c", 0, len(spans[2].text))]),
                    "critical",
                    {"schema_confidence": 0.99, "boundary_confidence": 0.99, "entity_linking_confidence": 0.99},
                )
            ],
        )

        fields, _ = DeterministicCompiler().compile(plan, spans)

        self.assertEqual(next(iter(fields.values())).raw_value, "配料：白砂糖，葡萄糖浆\n致敏提示：含乳制品")

    def test_compiler_outputs_value_only_from_span_evidence(self) -> None:
        spans = [
            TextSpan(
                span_id="span_0001",
                page=1,
                text="品名：红豆奶茶",
                source="pdf_text",
            )
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, evidence = DeterministicCompiler().compile(plan, spans)
        self.assertEqual(len(fields), 1)
        field = next(iter(fields.values()))
        self.assertEqual(field.raw_value, "品名：红豆奶茶")
        self.assertEqual(field.normalized_value, "红豆奶茶")
        self.assertEqual(field.normalization, ["remove_field_label"])
        self.assertEqual(field.evidence_refs, ["ev_0001"])
        self.assertEqual(evidence[0].source_text, "品名：红豆奶茶")
        self.assertEqual(field.status, "manual_review_required")

    def test_uncertain_normalized_non_critical_field_requires_review(self) -> None:
        spans = [
            TextSpan(
                span_id="span_0001",
                page=1,
                text="文字要求：净含量字高≥2mm",
                source="pdf_text",
                bbox_pdf=BBoxPdf(1, 2, 120, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.605, 0.12),
            )
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)
        field = next(iter(fields.values()))

        self.assertEqual(field.normalization, ["remove_field_label"])
        self.assertEqual(field.risk_level, "medium")
        self.assertTrue(field.review_required)
        self.assertEqual(field.reason, "归一化字段置信度低于0.95")


if __name__ == "__main__":
    unittest.main()
