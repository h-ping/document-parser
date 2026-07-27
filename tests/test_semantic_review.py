import unittest

from document_parser.models import CompiledField, Evidence, ExtractionPlan, FieldPlan, SpanRange, TextSpan, ValueSource
from document_parser.semantic_review import (
    build_semantic_review_input,
    call_semantic_review_agent,
    deterministic_semantic_findings,
    replace_block_fields,
    replace_block_tables,
    semantic_state_hash,
    validate_agent_review_findings,
)


def _compiled(field_id: str, semantic_key: str, value: str, evidence_ref: str) -> CompiledField:
    return CompiledField(
        field_id,
        semantic_key,
        semantic_key,
        "string",
        value,
        value,
        value,
        value,
        "verified",
        "critical",
        {"overall": 0.99},
        "low",
        False,
        None,
        None,
        None,
        None,
        [evidence_ref],
    )


def _plan(field_id: str, semantic_key: str, span_id: str, block_id: str) -> FieldPlan:
    return FieldPlan(
        field_id,
        semantic_key,
        semantic_key,
        "string",
        None,
        None,
        ValueSource("span_ranges", [SpanRange(span_id, 0, 3)]),
        "critical",
        {"overall": 0.99},
        {"agent_block_id": block_id},
    )


class SemanticReviewTests(unittest.TestCase):
    def test_review_detects_truncated_identifier_and_missing_anchor(self) -> None:
        spans = [TextSpan("a", 1, "产品标准号：GB/", "pdf_char_atom"), TextSpan("b", 1, "保质期：12个月", "pdf_char_atom")]
        plan = ExtractionPlan("plan", "schema", [_plan("fp_1", "product.standard_code", "a", "block_1")])
        evidence = [Evidence("ev_0001", spans[0].text, 1, ["llm"], "missing", ["a"])]
        review_input = build_semantic_review_input(
            {"blocks": [{"block_id": "block_1", "block_type": "main_label", "context_span_ids": ["a", "b"]}]},
            plan,
            {"fld_0001": _compiled("fld_0001", "product.standard_code", "GB/", "ev_0001")},
            evidence,
            spans,
        )

        findings = deterministic_semantic_findings(review_input)

        self.assertEqual({item["issue_type"] for item in findings}, {"field_value_truncated", "important_anchor_unconsumed"})

    def test_agent_findings_require_resolvable_block_and_span_refs(self) -> None:
        review_input = {"blocks": [{"block_id": "block_1", "source_spans": [{"span_id": "a", "text": "配料：糖"}]}]}
        body = {
            "findings": [
                {"issue_type": "adhesion", "block_id": "block_1", "target_id": "document", "source_span_ids": ["a"], "message": "bad", "severity": "high", "repair_required": True},
                {"issue_type": "adhesion", "block_id": "missing", "target_id": "document", "source_span_ids": ["x"], "message": "bad", "severity": "high", "repair_required": True},
            ]
        }

        accepted, rejected = validate_agent_review_findings(body, review_input)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["reason"], "unknown_block_id")

    def test_semantic_review_batches_at_most_eight_blocks(self) -> None:
        class Agent:
            def __init__(self) -> None:
                self.calls = 0

            def review_compiled_blocks(self, review_input):
                self.calls += 1
                return {"findings": []}

        agent = Agent()
        review_input = {
            "blocks": [
                {"block_id": f"block_{index}", "source_spans": [{"span_id": f"span_{index}", "text": "标签"}]}
                for index in range(9)
            ]
        }

        _, _, call_count = call_semantic_review_agent(agent, review_input)

        self.assertEqual(call_count, 2)
        self.assertEqual(agent.calls, 2)

    def test_block_repair_replaces_only_that_blocks_fields(self) -> None:
        original = ExtractionPlan(
            "plan",
            "schema",
            [_plan("fp_1", "product.ingredients", "a", "block_1"), _plan("fp_2", "product.name", "b", "block_2")],
        )
        repair = ExtractionPlan("repair", "schema", [_plan("fp_3", "product.ingredients", "c", "block_1")])

        replaced = replace_block_fields(original, repair, "block_1")

        self.assertEqual([field.value_source.ranges[0].span_id for field in replaced.fields], ["c", "b"])
        self.assertTrue(replaced.fields[0].boundary["semantic_repair"])

    def test_state_hash_changes_with_values_or_findings(self) -> None:
        first = semantic_state_hash([], {"fld_0001": _compiled("fld_0001", "product.name", "A", "ev_1")})
        second = semantic_state_hash([], {"fld_0001": _compiled("fld_0001", "product.name", "B", "ev_1")})
        self.assertNotEqual(first, second)

    def test_state_hash_accepts_mixed_entity_ids(self) -> None:
        without_entity = _compiled("fld_0001", "product.name", "A", "ev_1")
        with_entity = CompiledField(**{**without_entity.__dict__, "field_id": "fld_0002", "entity_id": "product_001"})

        value = semantic_state_hash([], {"fld_0001": without_entity, "fld_0002": with_entity})

        self.assertEqual(len(value), 64)

    def test_nutrition_block_requires_rows_and_table_repair_replaces_only_local_table(self) -> None:
        review_input = {
            "blocks": [
                {
                    "block_id": "block_1",
                    "block_type": "nutrition_table",
                    "source_spans": [{"span_id": "a", "text": "营养成分表"}],
                    "compiled_fields": [],
                    "planned_tables": [{"agent_block_id": "block_1", "rows": []}],
                }
            ]
        }
        findings = deterministic_semantic_findings(review_input)
        original = ExtractionPlan("plan", "schema", [], tables=[{"agent_block_id": "block_1", "rows": []}, {"agent_block_id": "block_2", "rows": [1]}])
        repair = ExtractionPlan("repair", "schema", [], tables=[{"rows": [{"cells": []}]}])

        replaced = replace_block_tables(original, repair, "block_1")

        self.assertEqual(findings[0]["issue_type"], "nutrition_rows_missing")
        self.assertEqual([table["agent_block_id"] for table in replaced.tables], ["block_2", "block_1"])
        self.assertTrue(replaced.tables[-1]["semantic_repair"])


if __name__ == "__main__":
    unittest.main()
