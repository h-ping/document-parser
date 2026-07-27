import unittest

from document_parser.missing_fields import (
    build_missing_item_report,
    missing_item_validation_checks,
    risks_from_missing_item_report,
)
from document_parser.models import CompiledField


class MissingFieldsTests(unittest.TestCase):
    def test_missing_critical_fields_become_validation_checks_and_risks(self) -> None:
        fields = {
            "fld_0001": _field("fld_0001", "product.name", "品名：牛奶"),
            "fld_0002": _field("fld_0002", "product.ingredients", "配料：牛奶"),
        }
        report = build_missing_item_report(fields, [])
        checks = missing_item_validation_checks(report)
        risks = risks_from_missing_item_report(report)

        self.assertEqual(report["status"], "review_required")
        self.assertTrue(any(item["semantic_key"] == "product.net_content" for item in report["missing_fields"]))
        self.assertTrue(any(item["table_type"] == "nutrition_facts" for item in report["missing_tables"]))
        self.assertTrue(any(check["check_type"] == "missing_required_field" for check in checks))
        self.assertTrue(any(risk.risk_type == "critical_field_missing" for risk in risks))
        self.assertTrue(any(risk.risk_type == "critical_table_missing" for risk in risks))
        self.assertTrue(all(risk.risk_level == "high" for risk in risks))

    def test_report_passes_when_required_items_are_present(self) -> None:
        fields = {
            f"fld_{index:04d}": _field(f"fld_{index:04d}", semantic_key, "value")
            for index, semantic_key in enumerate(
                [
                    "product.name",
                    "product.ingredients",
                    "product.net_content",
                    "product.standard_code",
                    "product.shelf_life",
                    "product.storage_condition",
                    "manufacturer.name",
                    "manufacturer.address",
                    "manufacturer.license_number",
                    "barcode.commodity",
                ],
                start=1,
            )
        }
        report = build_missing_item_report(
            fields,
            [{"table_type": "nutrition_facts", "status": "verified", "evidence_refs": ["ev_0002"], "rows": [{"row_id": "row_0001"}]}],
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(missing_item_validation_checks(report), [])
        self.assertEqual(risks_from_missing_item_report(report), [])

    def test_review_required_agent_table_with_evidence_is_not_missing(self) -> None:
        fields = {
            f"fld_{index:04d}": _field(f"fld_{index:04d}", semantic_key, "value")
            for index, semantic_key in enumerate(
                [
                    "product.name",
                    "product.ingredients",
                    "product.net_content",
                    "product.standard_code",
                    "product.shelf_life",
                    "product.storage_condition",
                    "manufacturer.name",
                    "manufacturer.address",
                    "manufacturer.license_number",
                    "barcode.commodity",
                ],
                start=1,
            )
        }
        report = build_missing_item_report(
            fields,
            [
                {
                    "table_type": "nutrition_facts",
                    "status": "manual_review_required",
                    "evidence_refs": ["ev_0002"],
                    "rows": [{"row_id": "row_0001", "evidence_refs": ["ev_0002"]}],
                }
            ],
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["missing_table_count"], 0)


def _field(field_id: str, semantic_key: str, raw_value: str) -> CompiledField:
    return CompiledField(
        field_id=field_id,
        semantic_key=semantic_key,
        display_name=semantic_key,
        field_type="string",
        raw_value=raw_value,
        clean_value=raw_value,
        normalized_value=raw_value,
        value_hash="hash",
        status="compiled",
        criticality="critical",
        confidence={"overall": 0.99},
        risk_level="info",
        review_required=False,
        section_id=None,
        entity_id=None,
        table_id=None,
        row_key=None,
        evidence_refs=["ev_0001"],
    )


if __name__ == "__main__":
    unittest.main()
