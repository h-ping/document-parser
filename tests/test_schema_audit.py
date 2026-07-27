import unittest

from document_parser.agents import SchemaInductionAgent
from document_parser.models import FieldDefinition, GeneratedSchema, TextSpan
from document_parser.schema_audit import build_schema_audit


class SchemaAuditTests(unittest.TestCase):
    def test_generated_schema_passes_obvious_anchor_audit(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：牛奶", "pdf_text"),
            TextSpan("span_0002", 1, "营养成分表", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)

        audit = build_schema_audit(schema, spans, [])

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["issue_count"], 0)

    def test_schema_audit_flags_obvious_missing_field_definition(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        schema = _empty_schema()

        audit = build_schema_audit(schema, spans, [])

        self.assertEqual(audit["status"], "review_required")
        self.assertEqual(audit["issues"][0]["issue_type"], "schema_obvious_field_missing")
        self.assertEqual(audit["issues"][0]["severity"], "high")
        self.assertEqual(audit["issues"][0]["expected"]["semantic_key"], "product.name")

    def test_schema_audit_flags_missing_table_definition(self) -> None:
        spans = [TextSpan("span_0001", 1, "营养成分表", "pdf_text")]
        schema = _empty_schema()

        audit = build_schema_audit(schema, spans, [])

        self.assertTrue(any(issue["issue_type"] == "schema_table_definition_missing" for issue in audit["issues"]))

    def test_schema_audit_flags_duplicate_field_definition(self) -> None:
        schema = GeneratedSchema(
            schema_id="schema_duplicate",
            auto_generated=True,
            schema_version="dynamic_v1",
            sections=[],
            entity_types=[],
            field_definitions=[
                FieldDefinition("fdef_0001", "product.name", "品名", "string", "critical"),
                FieldDefinition("fdef_0002", "product.name", "品名", "string", "critical"),
            ],
        )

        audit = build_schema_audit(schema, [], [])

        self.assertEqual(audit["status"], "review_required")
        self.assertEqual(audit["issues"][0]["issue_type"], "schema_duplicate_field_definition")


def _empty_schema() -> GeneratedSchema:
    return GeneratedSchema(
        schema_id="schema_empty",
        auto_generated=True,
        schema_version="dynamic_v1",
        sections=[],
        entity_types=[],
        field_definitions=[],
    )


if __name__ == "__main__":
    unittest.main()
