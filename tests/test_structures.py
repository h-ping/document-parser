import unittest
from dataclasses import replace

from document_parser.agents import ExtractionAgent, SchemaInductionAgent
from document_parser.compiler import DeterministicCompiler
from document_parser.models import CompiledField, TextSpan
from document_parser.pipeline import _risks_from_revision_blocks
from document_parser.structures import (
    build_entities,
    build_requirements,
    build_revision_blocks,
    content_item_names,
    detect_regions,
    extract_nutrition_tables,
    extract_nutrition_tables_from_layers,
)


class StructureTests(unittest.TestCase):
    def test_content_item_name_keeps_original_prefix(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "内容物 1：人参乌鸡靓汤粽", "pdf_text"),
            TextSpan("span_0002", 1, "配料表：糯米、水", "pdf_text"),
            TextSpan("span_0003", 1, "产品分类：真空包装类 含肉类", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)

        content_name = next(field for field in fields.values() if field.semantic_key == "content_item.name")
        self.assertEqual(content_name.raw_value, "内容物 1：人参乌鸡靓汤粽")
        self.assertEqual(content_name.normalized_value, "内容物 1：人参乌鸡靓汤粽")
        self.assertEqual(content_name.entity_id, "content_item_001")

        entities = build_entities(fields, [])
        self.assertIn("content_item_001", entities)
        self.assertEqual(entities["content_item_001"]["fields"]["name"]["field_id"], content_name.field_id)
        self.assertEqual(entities["content_item_001"]["fields"]["name"]["value"], content_name.raw_value)

    def test_field_semantics_override_generic_business_operator_entity_type(self) -> None:
        span = TextSpan("span_0001", 1, "委托方：甲公司", "pdf_text")
        schema = SchemaInductionAgent().generate([span])
        plan = ExtractionAgent().create_plan(schema, [span])
        field = replace(
            next(iter(DeterministicCompiler().compile(plan, [span])[0].values())),
            semantic_key="principal.name",
            entity_id="business_operator_1",
        )

        entities = build_entities(
            {field.field_id: field},
            [],
            [{"entity_id": "business_operator_1", "entity_type": "business_operator"}],
        )

        self.assertEqual(entities["business_operator_1"]["entity_type"], "principal")

    def test_multikey_line_is_split_by_neighbor_labels(self) -> None:
        spans = [
            TextSpan(
                "span_0001",
                1,
                "地址：广东省阳江市 产地：广东阳江 ⻝品⽣产许可证编号：SC11344172300010",
                "pdf_text",
            )
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)
        values = {field.semantic_key: field.normalized_value for field in fields.values()}

        self.assertEqual(values["manufacturer.address"], "广东省阳江市")
        self.assertEqual(values["manufacturer.origin"], "广东阳江")
        self.assertEqual(values["manufacturer.license_number"], "SC11344172300010")

    def test_product_name_metadata_line_is_not_extracted_as_label_text(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "产品名称 机型：无", "pdf_text"),
            TextSpan("span_0002", 1, "品名：红豆奶茶", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)
        product_names = [field.raw_value for field in fields.values() if field.semantic_key == "product.name"]

        self.assertEqual(product_names, ["品名：红豆奶茶"])

    def test_ingredient_long_text_keeps_following_source_spans_until_next_anchor(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "内容物 1：人参乌鸡靓汤粽", "pdf_text"),
            TextSpan("span_0002", 1, "配料表：糯米、水、汤汁≥25%", "pdf_text"),
            TextSpan("span_0003", 1, "鸡肉、乌鸡肉≥10%、莲子", "pdf_text"),
            TextSpan("span_0004", 1, "产品分类：真空包装类 含肉类", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)
        ingredients = next(field for field in fields.values() if field.semantic_key == "product.ingredients")

        self.assertIn("配料表：糯米、水、汤汁≥25%", ingredients.raw_value)
        self.assertIn("鸡肉、乌鸡肉≥10%、莲子", ingredients.raw_value)
        self.assertNotIn("产品分类", ingredients.raw_value)

    def test_nutrition_table_links_to_matching_content_item(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "内容物 1：人参乌鸡靓汤粽", "pdf_text"),
            TextSpan("span_0002", 2, "人参乌鸡靓汤粽营养成分表", "pdf_text"),
            TextSpan("span_0003", 2, "项目 每 100 克 营养素参考值%", "pdf_text"),
            TextSpan("span_0004", 2, "能量 810 千焦 10%", "pdf_text"),
            TextSpan("span_0005", 2, "蛋白质 6.7 克 11%", "pdf_text"),
        ]
        tables, evidence = extract_nutrition_tables(spans, content_item_names(spans), 0)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["linked_entity_id"], "content_item_001")
        self.assertEqual(len(tables[0]["rows"]), 2)
        self.assertEqual(tables[0]["rows"][0]["row_key"], "energy")
        self.assertEqual(len(evidence), 4)

    def test_table_layer_without_data_rows_keeps_title_evidence(self) -> None:
        table_layers = {
            "tables": [
                {
                    "table_layer_id": "tl_tbl_0001",
                    "parser": "text_span_nutrition",
                    "table_type": "nutrition_facts",
                    "page": 1,
                    "title": "营养成分表",
                    "columns": [{"column_id": "col_001", "name": "项目"}],
                    "rows": [{"row_type": "header", "cells": [{"text": "项目"}]}],
                    "source_span_ids": ["span_0001", "span_0002"],
                    "bbox_status": "available",
                    "confidence": 0.50,
                }
            ]
        }

        tables, evidence = extract_nutrition_tables_from_layers(table_layers, {}, 0)

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["status"], "manual_review_required")
        self.assertEqual(tables[0]["evidence_refs"], ["ev_0001"])
        self.assertEqual(evidence[0].source_text, "营养成分表")
        self.assertEqual(evidence[0].source_node_ids, ["span_0001"])

    def test_revision_blocks_do_not_auto_assign_ambiguous_fields(self) -> None:
        spans = [TextSpan("span_0001", 1, "更改前 更改后", "pdf_text")]
        regions = detect_regions(spans)
        blocks = build_revision_blocks(regions, {})
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(block["fields"] == [] for block in blocks))
        risks = _risks_from_revision_blocks(blocks)
        self.assertEqual(len(risks), 2)
        self.assertTrue(all(risk.risk_type == "revision_assignment_uncertain" for risk in risks))

    def test_revision_blocks_assign_fields_when_marker_order_is_clear(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "更改前", "pdf_text"),
            TextSpan("span_0002", 1, "品名：旧名称", "pdf_text"),
            TextSpan("span_0003", 1, "更改后", "pdf_text"),
            TextSpan("span_0004", 1, "品名：新名称", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, evidence = DeterministicCompiler().compile(plan, spans)
        regions = detect_regions(spans)

        blocks = build_revision_blocks(regions, fields, evidence, spans)
        by_role = {block["revision_role"]: block for block in blocks}

        self.assertEqual(by_role["before"]["assignment_status"], "assigned_by_span_order")
        self.assertEqual(by_role["after"]["assignment_status"], "assigned_by_span_order")
        self.assertEqual(by_role["before"]["fields"][0]["semantic_key"], "product.name")
        self.assertEqual(by_role["after"]["fields"][0]["semantic_key"], "product.name")
        self.assertTrue(by_role["after"]["is_current_standard"])
        self.assertEqual(_risks_from_revision_blocks(blocks), [])

    def test_requirement_types_are_classified_from_original_text(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "文字要求：净含量字高≥2mm", "pdf_text"),
            TextSpan("span_0002", 1, "日期喷印注意：生产日期需喷印于瓶底", "pdf_text"),
            TextSpan("span_0003", 1, "推广注意：不得夸大宣传", "pdf_text"),
            TextSpan("span_0004", 1, "设计注意：保持正面排版居中", "pdf_text"),
            TextSpan("span_0005", 1, "设计注意：保留红色图标", "pdf_text"),
        ]
        schema = SchemaInductionAgent().generate(spans)
        plan = ExtractionAgent().create_plan(schema, spans)
        fields, _ = DeterministicCompiler().compile(plan, spans)

        requirements = build_requirements(fields)
        by_text = {requirement["requirement_text"]: requirement for requirement in requirements}

        self.assertEqual(by_text["文字要求：净含量字高≥2mm"]["requirement_type"], "text_size")
        self.assertEqual(by_text["文字要求：净含量字高≥2mm"]["target"], "净含量")
        self.assertEqual(by_text["日期喷印注意：生产日期需喷印于瓶底"]["requirement_type"], "date_printing_requirement")
        self.assertEqual(by_text["日期喷印注意：生产日期需喷印于瓶底"]["target"], "生产日期")
        self.assertEqual(by_text["推广注意：不得夸大宣传"]["requirement_type"], "advertising_claim_restriction")
        self.assertEqual(by_text["推广注意：不得夸大宣传"]["target"], "推广")
        self.assertEqual(by_text["设计注意：保持正面排版居中"]["requirement_type"], "layout_requirement")
        self.assertEqual(by_text["设计注意：保留红色图标"]["requirement_type"], "design_note")
        self.assertTrue(all(requirement["verification_status"] == "not_verified_in_mvp" for requirement in requirements))

    def test_requirement_target_prefers_specific_barcode_target(self) -> None:
        requirements = build_requirements(
            {
                "fld_0001": CompiledField(
                    field_id="fld_0001",
                    semantic_key="requirement.text",
                    display_name="其它要求",
                    field_type="requirement",
                    raw_value="其它要求：商品条码放置于侧唛",
                    clean_value="其它要求：商品条码放置于侧唛",
                    normalized_value="其它要求：商品条码放置于侧唛",
                    value_hash="hash",
                    status="extracted",
                    criticality="non_critical",
                    confidence={"overall": 0.98},
                    risk_level="info",
                    review_required=False,
                    section_id=None,
                    entity_id=None,
                    table_id=None,
                    row_key=None,
                    evidence_refs=["ev_0001"],
                )
            }
        )

        self.assertEqual(requirements[0]["requirement_type"], "barcode_requirement")
        self.assertEqual(requirements[0]["target"], "商品条码")


if __name__ == "__main__":
    unittest.main()
