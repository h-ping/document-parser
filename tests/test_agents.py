import unittest

from document_parser.agents import AuditAgent, RepairAgent, SchemaInductionAgent
from document_parser.compiler import DeterministicCompiler
from document_parser.models import BBoxPdf, ExtractionPlan, FieldPlan, SpanRange, TextSpan, ValueSource, to_jsonable


class AgentRepairTests(unittest.TestCase):
    def test_schema_induction_records_source_span_ids_for_definitions(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：牛奶", "pdf_text"),
            TextSpan("span_0002", 1, "营养成分表", "pdf_text"),
        ]

        schema = SchemaInductionAgent().generate(spans)
        field_by_key = {definition.semantic_key: definition for definition in schema.field_definitions}

        self.assertEqual(field_by_key["product.name"].source_span_ids, ["span_0001"])
        self.assertEqual(schema.table_definitions[0]["source_span_ids"], ["span_0002"])

    def test_repair_agent_trims_long_text_field_at_sibling_label(self) -> None:
        spans = [
            TextSpan(
                "span_0001",
                1,
                "配料：水 产品类型：饮料",
                "pdf_text",
                bbox_pdf=BBoxPdf(0, 0, 100, 10, 100, 100),
            )
        ]
        schema = SchemaInductionAgent().generate(spans)
        bad_plan = ExtractionPlan(
            plan_id="plan_bad",
            schema_id=schema.schema_id,
            fields=[
                FieldPlan(
                    field_plan_id="fp_bad_ingredients",
                    semantic_key="product.ingredients",
                    display_name="配料",
                    field_type="long_text",
                    section_id="sec_label_text",
                    entity_id="product_001",
                    value_source=ValueSource(
                        mode="span_ranges",
                        ranges=[SpanRange("span_0001", 0, len(spans[0].text))],
                    ),
                    criticality="critical",
                    confidence={
                        "schema_confidence": 0.95,
                        "boundary_confidence": 0.70,
                        "entity_linking_confidence": 0.95,
                    },
                )
            ],
        )
        fields, evidence = DeterministicCompiler().compile(bad_plan, spans)
        audit_input = {field_id: {**to_jsonable(field), "has_bbox": True} for field_id, field in fields.items()}
        findings = AuditAgent().audit(audit_input, schema)

        repaired_plan, attempts = RepairAgent().repair(bad_plan, findings, spans)
        repaired_fields, _ = DeterministicCompiler().compile(repaired_plan, spans)
        repaired_field = next(iter(repaired_fields.values()))

        self.assertTrue(any(item["finding_type"] == "possible_field_adhesion" for item in findings))
        self.assertEqual(attempts[0]["status"], "applied")
        self.assertEqual(repaired_field.raw_value, "配料：水")
        self.assertEqual(repaired_field.normalized_value, "水")
        self.assertNotIn("产品类型", repaired_field.raw_value)
        self.assertTrue(evidence)


if __name__ == "__main__":
    unittest.main()
