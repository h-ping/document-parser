import unittest

from document_parser.agents import ExtractionAgent, SchemaInductionAgent
from document_parser.compiler import DeterministicCompiler
from document_parser.field_validation import validate_field_format_value
from document_parser.models import BBoxPdf, BBoxNormalized, CompiledField, Evidence, FieldDefinition, GeneratedSchema, TextSpan
from document_parser.pipeline import _internal_consistency_validation_checks, _risks_from_validation, _validation_checks


class FieldValidationTests(unittest.TestCase):
    def test_barcode_gtin_check_digit_validation(self) -> None:
        self.assertEqual(validate_field_format_value("barcode.commodity", "6901234567892")["result"], "passed")
        failed = validate_field_format_value("barcode.commodity", "6901234567893")
        self.assertEqual(failed["result"], "failed")
        self.assertEqual(failed["severity"], "high")

    def test_license_net_content_and_shelf_life_rules(self) -> None:
        self.assertEqual(validate_field_format_value("manufacturer.license_number", "SC11344172300010")["result"], "passed")
        self.assertEqual(validate_field_format_value("manufacturer.license_number", "许可证待定")["result"], "failed")
        self.assertEqual(validate_field_format_value("product.net_content", "250 mL")["result"], "passed")
        self.assertEqual(validate_field_format_value("product.net_content", "一瓶")["result"], "failed")
        self.assertEqual(validate_field_format_value("product.shelf_life", "12个月")["result"], "passed")

    def test_compiler_and_validation_report_format_failure(self) -> None:
        spans = [
            TextSpan(
                "span_0001",
                1,
                "商品条码：6901234567893",
                "pdf_text",
                bbox_pdf=BBoxPdf(1, 2, 60, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.305, 0.12),
            )
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, evidence = DeterministicCompiler().compile(plan, spans)
        field = next(iter(fields.values()))
        validation = _validation_checks(fields, evidence, schema)
        risks = _risks_from_validation(validation)

        self.assertEqual(field.confidence["format_validation_confidence"], 0.85)
        self.assertTrue(any(check["check_type"] == "format_check" and check["result"] == "failed" for check in validation))
        self.assertEqual(risks[0].risk_type, "format_check_failed")
        self.assertEqual(risks[0].risk_level, "high")

    def test_critical_field_without_bbox_fails_bbox_validation(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, evidence = DeterministicCompiler().compile(plan, spans)
        field = next(iter(fields.values()))
        validation = _validation_checks(fields, evidence, schema)
        risks = _risks_from_validation(validation)

        bbox_check = next(check for check in validation if check["check_type"] == "bbox_integrity")
        self.assertEqual(field.status, "manual_review_required")
        self.assertEqual(bbox_check["result"], "failed")
        self.assertEqual(bbox_check["severity"], "high")
        self.assertEqual(bbox_check["evidence_refs"], field.evidence_refs)
        self.assertTrue(any(risk.risk_type == "critical_field_without_bbox" for risk in risks))

    def test_non_critical_field_without_bbox_fails_bbox_validation_as_medium_risk(self) -> None:
        field = CompiledField(
            field_id="fld_0001",
            semantic_key="requirement.text",
            display_name="文字要求",
            field_type="requirement",
            raw_value="设计注意：文字需清晰",
            clean_value="文字需清晰",
            normalized_value="文字需清晰",
            normalization=["remove_field_label"],
            value_hash="sha256:" + "a" * 64,
            status="normalized",
            criticality="non_critical",
            confidence={
                "schema_confidence": 1.0,
                "boundary_confidence": 1.0,
                "entity_linking_confidence": 1.0,
                "evidence_confidence": 1.0,
                "format_validation_confidence": 1.0,
                "overall": 1.0,
            },
            risk_level="low",
            review_required=False,
            section_id="sec_label_text",
            entity_id="requirement_001",
            table_id=None,
            row_key=None,
            evidence_refs=["ev_0001"],
        )
        evidence = [Evidence("ev_0001", "设计注意：文字需清晰", 1, ["pdf_text"], "missing", ["span_0001"])]
        schema = GeneratedSchema(
            schema_id="schema_dynamic_001",
            auto_generated=True,
            schema_version="dynamic_v1",
            sections=[],
            entity_types=[],
            field_definitions=[
                FieldDefinition(
                    field_def_id="fdef_0001",
                    semantic_key="requirement.text",
                    display_name="文字要求",
                    field_type="requirement",
                    criticality="non_critical",
                )
            ],
        )

        validation = _validation_checks({"fld_0001": field}, evidence, schema)
        risks = _risks_from_validation(validation)

        bbox_check = next(check for check in validation if check["check_type"] == "bbox_integrity")
        self.assertEqual(bbox_check["result"], "failed")
        self.assertEqual(bbox_check["severity"], "medium")
        self.assertEqual(bbox_check["risk_type"], "field_without_bbox")
        self.assertTrue(any(risk.risk_type == "field_without_bbox" and risk.risk_level == "medium" for risk in risks))

    def test_field_without_format_rule_still_records_format_check(self) -> None:
        spans = [
            TextSpan(
                "span_0001",
                1,
                "品名：牛奶",
                "pdf_text",
                bbox_pdf=BBoxPdf(1, 2, 60, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.305, 0.12),
            )
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, evidence = DeterministicCompiler().compile(plan, spans)

        validation = _validation_checks(fields, evidence, schema)

        format_check = next(check for check in validation if check["check_type"] == "format_check")
        self.assertEqual(format_check["result"], "passed")
        self.assertEqual(format_check["format_rule_status"], "skipped_no_rule")

    def test_field_missing_from_generated_schema_fails_schema_validation(self) -> None:
        spans = [
            TextSpan(
                "span_0001",
                1,
                "品名：牛奶",
                "pdf_text",
                bbox_pdf=BBoxPdf(1, 2, 60, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.305, 0.12),
            )
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, evidence = DeterministicCompiler().compile(plan, spans)
        empty_schema = GeneratedSchema(
            schema_id="schema_empty",
            auto_generated=True,
            schema_version="dynamic_v1",
            sections=[],
            entity_types=[],
            field_definitions=[],
        )

        validation = _validation_checks(fields, evidence, empty_schema)
        risks = _risks_from_validation(validation)

        schema_check = next(check for check in validation if check["check_type"] == "schema_validation")
        self.assertEqual(schema_check["result"], "failed")
        self.assertEqual(schema_check["issues"][0]["reason"], "semantic_key_not_in_generated_schema")
        self.assertTrue(any(risk.risk_type == "schema_validation_failed" for risk in risks))

    def test_duplicate_field_values_fail_internal_consistency(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：牛奶", "pdf_text", bbox_pdf=BBoxPdf(1, 2, 60, 10, 200, 100)),
            TextSpan("span_0002", 1, "品名：酸奶", "pdf_text", bbox_pdf=BBoxPdf(1, 20, 60, 10, 200, 100)),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)

        validation = _internal_consistency_validation_checks(fields, [])
        risks = _risks_from_validation(validation)

        self.assertEqual(validation[0]["check_type"], "internal_consistency")
        self.assertEqual(validation[0]["result"], "failed")
        self.assertEqual(validation[0]["conflict_count"], 1)
        self.assertEqual(risks[0].risk_type, "field_internal_conflict")

    def test_before_revision_fields_are_ignored_for_internal_consistency(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：旧名称", "pdf_text", bbox_pdf=BBoxPdf(1, 2, 60, 10, 200, 100)),
            TextSpan("span_0002", 1, "品名：新名称", "pdf_text", bbox_pdf=BBoxPdf(1, 20, 60, 10, 200, 100)),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)
        field_ids = list(fields)
        revision_blocks = [{"revision_role": "before", "fields": [{"field_id": field_ids[0]}]}]

        validation = _internal_consistency_validation_checks(fields, revision_blocks)

        self.assertEqual(validation[0]["result"], "passed")
        self.assertEqual(validation[0]["conflict_count"], 0)

    def test_prefix_duplicate_values_pass_internal_consistency(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "净含量/规格：1.4千克（粽子100克×1、汤粽100克×1）", "pdf_text"),
            TextSpan("span_0002", 1, "净含量/规格：1.4千克（粽子100克×1、汤粽", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)

        validation = _internal_consistency_validation_checks(fields, [])

        self.assertEqual(validation[0]["result"], "passed")
        self.assertEqual(validation[0]["conflict_count"], 0)


if __name__ == "__main__":
    unittest.main()
