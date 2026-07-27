import unittest

from document_parser.audit_artifacts import build_audit_input_artifact
from document_parser.models import CompiledField, Evidence, ExtractionPlan, FieldDefinition, FieldPlan, GeneratedSchema, SpanRange, ValueSource


class AuditArtifactTests(unittest.TestCase):
    def test_audit_input_artifact_freezes_compiled_review_inputs(self) -> None:
        schema = GeneratedSchema(
            schema_id="schema_dynamic_001",
            auto_generated=True,
            schema_version="dynamic_v1",
            sections=[],
            entity_types=[],
            field_definitions=[
                FieldDefinition(
                    field_def_id="fdef_0001",
                    semantic_key="product.name",
                    display_name="品名",
                    field_type="string",
                    criticality="critical",
                )
            ],
        )
        plan = ExtractionPlan(
            plan_id="plan_001",
            schema_id=schema.schema_id,
            fields=[
                FieldPlan(
                    field_plan_id="fp_0001",
                    semantic_key="product.name",
                    display_name="品名",
                    field_type="string",
                    section_id="sec_label_text",
                    entity_id="product_001",
                    value_source=ValueSource("span_ranges", [SpanRange("span_0001", 0, 5)]),
                    criticality="critical",
                    confidence={"boundary_confidence": 0.98},
                )
            ],
        )
        field = CompiledField(
            field_id="fld_0001",
            semantic_key="product.name",
            display_name="品名",
            field_type="string",
            raw_value="品名：牛奶",
            clean_value="牛奶",
            normalized_value="牛奶",
            value_hash="hash",
            status="verified",
            criticality="critical",
            confidence={"overall": 0.99},
            risk_level="info",
            review_required=False,
            section_id="sec_label_text",
            entity_id="product_001",
            table_id=None,
            row_key=None,
            evidence_refs=["ev_0001"],
        )
        evidence = [
            Evidence(
                evidence_id="ev_0001",
                source_text="品名：牛奶",
                page=1,
                extraction_methods=["pdf_text"],
                bbox_status="missing",
                source_node_ids=["span_0001"],
            )
        ]

        artifact = build_audit_input_artifact(
            page_images={"status": "rendered", "page_count": 1, "rendered_count": 1, "failed_count": 0},
            visual_document_graph={"node_count": 1, "edge_count": 0},
            schema=schema,
            plan=plan,
            compiled_fields={"fld_0001": field},
            evidence=evidence,
            coverage_map={"span_coverage_rate": 1.0},
            audit_findings=[{"finding_id": "af_0001"}],
        )

        self.assertEqual(artifact["stage"], "independent_audit")
        self.assertIn("coverage_map.json", artifact["input_artifacts"])
        self.assertTrue(artifact["separation"]["review_runs_after_compiler"])
        self.assertEqual(artifact["compiled_fields"]["fld_0001"]["raw_value"], "品名：牛奶")
        self.assertEqual(artifact["audit_finding_count"], 1)


if __name__ == "__main__":
    unittest.main()
