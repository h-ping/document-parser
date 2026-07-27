import json
import tempfile
import unittest
from pathlib import Path

from document_parser.agent_candidates import (
    apply_rule_fallback_fields,
    build_agent_extraction_plan,
    build_rule_candidate_review_items,
    merge_agent_candidates,
)
from document_parser.agents import ExtractionAgent, SchemaInductionAgent
from document_parser.models import ExtractionPlan, FieldPlan, SpanRange, TextSpan, ValueSource


class AgentCandidateTests(unittest.TestCase):
    def test_exact_agent_text_repairs_cjk_offsets_that_exceed_span_length(self) -> None:
        spans = [TextSpan("span_0001", 1, "配料：葡萄糖浆，白砂", "pdf_char_atom")]
        schema = SchemaInductionAgent().generate(spans)
        body = {
            "fields": [
                {
                    "semantic_key": "product.ingredients",
                    "display_name": "配料",
                    "field_type": "long_text",
                    "span_id": "span_0001",
                    "start_offset": 0,
                    "end_offset": 14,
                    "text": "配料：葡萄糖浆，白砂",
                    "confidence": 0.98,
                    "entity_id": None,
                    "section_id": None,
                }
            ],
            "entities": [],
            "tables": [],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
            "layout_candidate_decisions": [],
        }

        plan, rejected, _ = build_agent_extraction_plan(schema, spans, body)

        self.assertFalse(rejected)
        self.assertEqual(plan.fields[0].value_source.ranges[0].end_offset, len(spans[0].text))
        self.assertEqual(plan.fields[0].boundary["offset_repair"], "matched_text_in_source_span")

    def test_business_operator_semantic_keys_are_canonicalized(self) -> None:
        spans = [TextSpan("span_0001", 1, "生产者：甲公司", "pdf_char_atom")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "semantic_key": "business_operator.name",
                        "display_name": "生产者",
                        "field_type": "string",
                        "span_id": "span_0001",
                        "start_offset": 0,
                        "end_offset": len(spans[0].text),
                        "text": spans[0].text,
                        "confidence": 0.98,
                        "entity_id": "producer_1",
                        "section_id": None,
                    }
                ]
            },
        )

        self.assertFalse(rejected)
        self.assertEqual(plan.fields[0].semantic_key, "manufacturer.name")

    def test_business_operator_role_specific_keys_are_canonicalized(self) -> None:
        spans = [TextSpan("span_0001", 1, "委托方：甲公司", "pdf_char_atom")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "semantic_key": "business_operator.principal_name",
                        "display_name": "委托方",
                        "field_type": "string",
                        "span_id": "span_0001",
                        "start_offset": 0,
                        "end_offset": len(spans[0].text),
                        "text": spans[0].text,
                        "confidence": 0.98,
                        "entity_id": "business_operator_1",
                    }
                ]
            },
        )

        self.assertFalse(rejected)
        self.assertEqual(plan.fields[0].semantic_key, "principal.name")

    def test_food_production_license_alias_is_canonicalized(self) -> None:
        spans = [TextSpan("span_0001", 1, "食品生产许可证编号：SC123", "pdf_char_atom")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "semantic_key": "manufacturer.food_production_license",
                        "span_id": "span_0001",
                        "start_offset": 0,
                        "end_offset": len(spans[0].text),
                    }
                ]
            },
        )

        self.assertFalse(rejected)
        self.assertEqual(plan.fields[0].semantic_key, "manufacturer.license_number")

    def test_nested_entity_fields_are_compiled_with_entity_identity(self) -> None:
        spans = [
            TextSpan("name", 1, "受托方：甲公司", "pdf_char_atom"),
            TextSpan("address", 1, "地址：深圳", "pdf_char_atom"),
        ]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "semantic_key": "contract_manufacturer.name",
                        "span_id": "name",
                        "start_offset": 0,
                        "end_offset": len(spans[0].text),
                        "entity_id": "manufacturer_001",
                    }
                ],
                "entities": [
                    {
                        "entity_id": "manufacturer_001",
                        "entity_type": "manufacturer",
                        "fields": [
                            {
                                "semantic_key": "contract_manufacturer.address",
                                "span_id": "address",
                                "start_offset": 0,
                                "end_offset": len(spans[1].text),
                                "entity_id": "wrong_child_entity",
                            }
                        ],
                    }
                ],
            },
        )

        self.assertFalse(rejected)
        self.assertEqual(
            [(field.semantic_key, field.entity_id) for field in plan.fields],
            [("manufacturer.name", "manufacturer_001"), ("manufacturer.address", "manufacturer_001")],
        )

    def test_rule_fallback_deduplicates_within_entity_not_globally(self) -> None:
        agent_plan = ExtractionPlan("agent", "schema", [_field_plan("agent", "manufacturer.name", "manufacturer_001", "a")])
        rule_plan = ExtractionPlan(
            "rule",
            "schema",
            [
                _field_plan("rule1", "manufacturer.name", "manufacturer_001", "a"),
                _field_plan("rule2", "manufacturer.name", "manufacturer_002", "b"),
            ],
        )

        merged, fallback = apply_rule_fallback_fields(agent_plan, rule_plan)

        self.assertEqual([(field.semantic_key, field.entity_id) for field in merged.fields], [("manufacturer.name", "manufacturer_001"), ("manufacturer.name", "manufacturer_002")])
        self.assertEqual(len(fallback), 1)

    def test_rule_fallback_preserves_distinct_ranges_for_same_entity(self) -> None:
        agent_plan = ExtractionPlan("agent", "schema", [_field_plan("agent", "custom.other_label_text", "product_001", "a")])
        rule_plan = ExtractionPlan("rule", "schema", [_field_plan("rule", "custom.other_label_text", "product_001", "b")])

        merged, fallback = apply_rule_fallback_fields(agent_plan, rule_plan)

        self.assertEqual([field.value_source.ranges[0].span_id for field in merged.fields], ["a", "b"])
        self.assertEqual(len(fallback), 1)

    def test_rule_review_preserves_missing_repeated_entity_occurrence(self) -> None:
        agent_plan = ExtractionPlan(
            "agent",
            "schema",
            [_field_plan("agent", "manufacturer.name", "manufacturer_001", "a")],
        )
        rule_plan = ExtractionPlan(
            "rule",
            "schema",
            [
                _field_plan("rule1", "manufacturer.name", "manufacturer_001", "a"),
                _field_plan("rule2", "manufacturer.name", "manufacturer_002", "b"),
            ],
        )

        review_items = build_rule_candidate_review_items(agent_plan, rule_plan)

        self.assertEqual(len(review_items), 1)
        self.assertEqual(review_items[0]["entity_id"], "manufacturer_002")
        self.assertEqual(review_items[0]["source_span_ids"], ["b"])

    def test_agent_plan_deduplicates_complete_signature_only(self) -> None:
        spans = [TextSpan("a", 1, "甲", "pdf_char_atom"), TextSpan("b", 1, "乙", "pdf_char_atom")]
        schema = SchemaInductionAgent().generate(spans)
        base = {
            "semantic_key": "custom.other_label_text",
            "display_name": "其他标签文字",
            "field_type": "string",
            "start_offset": 0,
            "end_offset": 1,
            "entity_id": "product_001",
        }

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {"fields": [{**base, "span_id": "a"}, {**base, "span_id": "a"}, {**base, "span_id": "b"}]},
        )

        self.assertEqual([field.value_source.ranges[0].span_id for field in plan.fields], ["a", "b"])
        self.assertEqual([item["reason"] for item in rejected], ["duplicate_of_rule_candidate"])

    def test_merges_only_span_grounded_agent_candidates(self) -> None:
        spans = [TextSpan("span_0001", 1, "推广注意：请勿夸大宣传", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agent_items.json"
            path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "semantic_key": "custom.promotion_note",
                                "display_name": "推广注意",
                                "field_type": "requirement",
                                "span_id": "span_0001",
                                "start_offset": 0,
                                "end_offset": len(spans[0].text),
                                "text": spans[0].text,
                                "confidence": 0.92,
                            },
                            {
                                "semantic_key": "custom.bad",
                                "display_name": "坏候选",
                                "span_id": "missing",
                                "start_offset": 0,
                                "end_offset": 2,
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            merged, rejected, review_items = merge_agent_candidates(plan, schema, spans, path)

        self.assertEqual(len(merged.fields), len(plan.fields) + 1)
        self.assertEqual(rejected[0]["reason"], "missing_or_unknown_span_id")
        self.assertEqual(review_items, [])

    def test_merges_llm_alias_shape_from_field_def_and_source_span(self) -> None:
        spans = [TextSpan("span_0001", 1, "贮存条件:勿置于阳光处", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        field_def = next(definition for definition in schema.field_definitions if definition.semantic_key == "product.storage_condition")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agent_items.json"
            path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "field_def_id": field_def.field_def_id,
                                "source_span_id": "span_0001",
                                "start_offset": 0,
                                "end_offset": 10,
                                "value": spans[0].text[:10],
                                "confidence": 0.91,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            merged, rejected, review_items = merge_agent_candidates(plan, schema, spans, path)

        self.assertEqual(rejected, [])
        self.assertEqual(review_items, [])
        self.assertTrue(any(field.semantic_key == "product.storage_condition" for field in merged.fields))

    def test_builds_agent_plan_from_start_end_aliases_without_rule_plan(self) -> None:
        spans = [TextSpan("span_0001", 1, "产品类型：混合胶型", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, review_items = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "semantic_key": "product.product_type",
                        "display_name": "产品类型",
                        "field_type": "string",
                        "span_id": "span_0001",
                        "start": 5,
                        "end": 9,
                        "text": "混合胶型",
                        "confidence": 0.96,
                    }
                ],
                "entities": [],
                "tables": [],
                "requirements": [],
            },
        )

        self.assertEqual(rejected, [])
        self.assertEqual(review_items, [])
        self.assertEqual(len(plan.fields), 1)
        self.assertEqual(plan.fields[0].value_source.ranges[0].start_offset, 5)

    def test_builds_agent_plan_from_nested_extraction_plan_shape(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "extraction_plan": {
                    "field_plans": [
                        {
                            "semantic_key": "product.name",
                            "display_name": "品名",
                            "field_type": "string",
                            "span_id": "span_0001",
                            "start_offset": 0,
                            "end_offset": 5,
                            "text": "品名：牛奶",
                            "confidence": 0.96,
                        }
                    ],
                    "entities": [],
                    "tables": [],
                    "requirements": [],
                }
            },
        )

        self.assertEqual(rejected, [])
        self.assertEqual(plan.fields[0].semantic_key, "product.name")

    def test_builds_empty_agent_plan_when_llm_returns_no_field_container(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, review_items = build_agent_extraction_plan(schema, spans, {"summary": "no fields"})

        self.assertEqual(plan.fields, [])
        self.assertEqual(rejected, [])
        self.assertEqual(review_items, [])

    def test_builds_agent_plan_from_path_shaped_llm_output(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "保质期12个月", "pdf_text"),
            TextSpan("span_0002", 1, "能量", "pdf_text"),
            TextSpan("span_0003", 1, "100kJ", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "product": {
                    "shelf_life": {
                        "span_id": "span_0001",
                        "offset_start": 0,
                        "offset_end": 6,
                    }
                },
                "nutrition_facts_table": {
                    "rows": [
                        {
                            "nutrient": {"span_id": "span_0002", "offset_start": 0, "offset_end": 2},
                            "amount": {"span_id": "span_0003", "offset_start": 0, "offset_end": 5},
                        }
                    ]
                },
            },
        )

        self.assertEqual(rejected, [])
        self.assertEqual(plan.fields[0].semantic_key, "product.shelf_life")
        self.assertEqual(plan.tables[0]["table_type"], "nutrition_facts")
        self.assertEqual(plan.tables[0]["rows"][0]["cells"][0]["span_id"], "span_0002")

    def test_builds_path_fields_even_when_explicit_fields_array_is_empty(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [],
                "product": {
                    "name": {
                        "source": {
                            "span_id": "span_0001",
                            "start_offset": 0,
                            "end_offset": 5,
                        },
                        "text": "品名：牛奶",
                    }
                },
            },
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(plan.fields), 1)
        self.assertEqual(plan.fields[0].semantic_key, "product.name")

    def test_builds_multirange_field_from_field_def_value_refs(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "配料：水", "pdf_text"),
            TextSpan("span_0002", 1, "白砂糖", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)
        field_def = next(definition for definition in schema.field_definitions if definition.semantic_key == "product.ingredients")

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "field_def_id": field_def.field_def_id,
                        "value": [
                            {"span_id": "span_0001", "start": 0, "end": 4},
                            {"span_id": "span_0002", "start": 0, "end": 3},
                        ],
                        "confidence": 0.96,
                    }
                ]
            },
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(plan.fields), 1)
        self.assertEqual(plan.fields[0].semantic_key, "product.ingredients")
        self.assertEqual(len(plan.fields[0].value_source.ranges), 2)

    def test_maps_field_def_id_misplaced_in_semantic_key(self) -> None:
        spans = [TextSpan("span_0001", 1, "保质期12个月", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)
        field_def = next(definition for definition in schema.field_definitions if definition.semantic_key == "product.shelf_life")

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "semantic_key": field_def.field_def_id,
                        "span_id": "span_0001",
                        "start_offset": 0,
                        "end_offset": 6,
                    }
                ]
            },
        )

        self.assertEqual(rejected, [])
        self.assertEqual(plan.fields[0].semantic_key, "product.shelf_life")

    def test_rejects_empty_agent_text_instead_of_repairing_to_zero_width(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        schema = SchemaInductionAgent().generate(spans)

        plan, rejected, _ = build_agent_extraction_plan(
            schema,
            spans,
            {
                "fields": [
                    {
                        "semantic_key": "product.name",
                        "span_id": "span_0001",
                        "start_offset": 0,
                        "end_offset": 5,
                        "text": "",
                    }
                ]
            },
        )

        self.assertEqual(plan.fields, [])
        self.assertEqual(rejected[0]["reason"], "empty_agent_text")


if __name__ == "__main__":
    unittest.main()


def _field_plan(field_id: str, semantic_key: str, entity_id: str, span_id: str) -> FieldPlan:
    return FieldPlan(
        field_id,
        semantic_key,
        semantic_key,
        "string",
        "sec_label_text",
        entity_id,
        ValueSource("span_ranges", [SpanRange(span_id, 0, 1)]),
        "critical",
        {"schema_confidence": 0.95, "boundary_confidence": 0.95, "entity_linking_confidence": 0.95},
    )
