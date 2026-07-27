import unittest

from document_parser.models import (
    BBoxNormalized,
    BBoxPdf,
    CompiledField,
    Evidence,
    ExtractionPlan,
    FieldDefinition,
    FieldPlan,
    GeneratedSchema,
    SpanRange,
    ValueSource,
)
from document_parser.mvp_metrics import build_mvp_acceptance_metrics


class MvpAcceptanceMetricsTests(unittest.TestCase):
    def test_metrics_summarize_mvp_acceptance_gates(self) -> None:
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
            confidence={"overall": 0.99, "boundary_confidence": 0.98},
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
                bbox_status="available",
                source_node_ids=["span_0001"],
                bbox_pdf=BBoxPdf(0, 0, 100, 10, 100, 100),
                bbox_normalized=BBoxNormalized(0, 0, 1, 0.1),
            )
        ]

        metrics = build_mvp_acceptance_metrics(
            schema=schema,
            plan=plan,
            compiled_fields={"fld_0001": field},
            evidence=evidence,
            validation=[],
            risks=[],
            review_tasks=[],
            coverage={"text_block_coverage_rate": 1.0, "important_region_coverage_rate": 1.0},
            coverage_map={"span_coverage_rate": 1.0, "anchor_count": 1, "duplicate_coverage_issue_count": 0},
            schema_audit={"status": "pass", "issue_count": 0, "blocking_issue_count": 0, "issues": []},
            structure_audit={
                "anchor_coverage": 1.0,
                "missing_anchor_count": 0,
                "sequence_gap_count": 0,
                "group_issue_count": 0,
                "table_issue_count": 0,
                "required_prefix_issue_count": 0,
                "container_duplicate_issue_count": 0,
                "agent_override_issue_count": 0,
                "duplicate_coverage_issue_count": 0,
            },
            source_layers={
                "status": "pass",
                "source_mode": "pdf_text_only",
                "source_issue_count": 0,
                "text_quality": {"total_text_span_count": 1, "bbox_coverage_rate": 1.0},
            },
            table_quality_report={"status": "pass", "table_count": 0, "issue_count": 0},
            repair_trace={"status": "pass", "round_count": 1, "attempt_count": 0, "applied_attempt_count": 0, "final_audit_finding_count": 0, "rounds": []},
            output_contract_validation_report={"status": "pass", "failed_count": 0},
            missing_item_report={"missing_count": 0},
        )

        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["schema"]["field_definition_count"], 1)
        self.assertEqual(metrics["evidence"]["evidence_coverage_rate"], 1.0)
        self.assertEqual(metrics["evidence"]["critical_field_without_bbox_count"], 0)
        self.assertEqual(metrics["coverage"]["anchor_coverage"], 1.0)
        self.assertEqual(metrics["coverage"]["sequence_gap_count"], 0)
        self.assertEqual(metrics["coverage"]["required_prefix_issue_count"], 0)
        self.assertEqual(metrics["coverage"]["container_duplicate_issue_count"], 0)
        self.assertEqual(metrics["coverage"]["agent_override_issue_count"], 0)
        self.assertEqual(metrics["coverage"]["structure_blocking_issue_count"], 0)
        self.assertTrue(metrics["repair"]["recompiled_after_applied_repair"])


if __name__ == "__main__":
    unittest.main()
