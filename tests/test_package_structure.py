import base64
import json
import tempfile
import unittest
from pathlib import Path

from document_parser.image_compare import compare_standard_to_ocr
from document_parser.image_compare_cli import run_compare_package_image
from document_parser.llm import LlmClient
from document_parser.models import BBoxNormalized, OcrLine
from document_parser.package_structure import (
    PACKAGE_STRUCTURE_LLM_MAX_TOKENS,
    build_package_glm_blocks,
    normalize_package_structure_output,
    run_package_structure_stage,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_IMAGE = ROOT / "test_png" / "粽子包装图.jpg"


class PackageStructureTests(unittest.TestCase):
    def test_glm_blocks_clean_markdown_and_parse_html_table(self) -> None:
        lines = [
            _glm_line("ocr_1", "block_1", 1, "text", "## 产品名称：红豆奶茶", 0.1, 0.1, 0.3, 0.12),
            _glm_line(
                "ocr_2",
                "block_2",
                2,
                "table",
                "<table><tr><td>项目</td><td>每100克</td><td>NRV%</td></tr><tr><td>能量</td><td>100千焦</td><td>1%</td></tr></table>",
                0.1,
                0.2,
                0.4,
                0.3,
            ),
        ]

        blocks = build_package_glm_blocks(lines)

        self.assertEqual(blocks["provider_features"]["block_count"], 2)
        self.assertEqual(blocks["blocks"][0]["cleaned_text"], "产品名称：红豆奶茶")
        self.assertEqual(blocks["blocks"][1]["table"]["rows"][1]["cells"][0]["text"], "能量")
        self.assertEqual(blocks["blocks"][1]["table"]["rows"][1]["cells"][1]["cell_id"], "block_2:r2c2")

    def test_structure_output_rejects_unknown_source_and_marks_unsupported_text_for_review(self) -> None:
        blocks = build_package_glm_blocks([_glm_line("ocr_1", "block_1", 1, "text", "产品名称：红豆奶茶", 0.1, 0.1, 0.3, 0.12)])

        structured, quality = normalize_package_structure_output(
            {
                "fields": [
                    {
                        "semantic_key": "product.name",
                        "label": "产品名称",
                        "text": "产品名称：红豆奶茶",
                        "source_ids": ["block_1"],
                        "confidence": 0.98,
                        "review_required": False,
                    },
                    {
                        "semantic_key": "product.shelf_life",
                        "label": "保质期",
                        "text": "保质期：12个月",
                        "source_ids": ["missing_block"],
                        "confidence": 0.9,
                        "review_required": False,
                    },
                    {
                        "semantic_key": "product.storage",
                        "label": "贮存条件",
                        "text": "贮存条件：阴凉干燥处保存",
                        "source_ids": ["block_1"],
                        "confidence": 0.9,
                        "review_required": False,
                    },
                ],
                "field_groups": [],
                "content_items": [],
                "nutrition_tables": [],
                "other_text": [],
                "warnings": [],
            },
            blocks,
        )

        self.assertEqual(quality["status"], "review_required")
        self.assertEqual(len(structured["fields"]), 2)
        self.assertEqual(len(structured["rejected_items"]), 1)
        storage = next(item for item in structured["fields"] if item["semantic_key"] == "product.storage")
        self.assertTrue(storage["review_required"])

    def test_compare_uses_structured_package_items_and_keeps_prefix_strict(self) -> None:
        artifacts = {
            "standard_items": [
                {
                    "id": "std_1",
                    "semantic_key": "product.name",
                    "label": "产品名称",
                    "text": "产品名称：红豆奶茶",
                    "comparison_required": True,
                    "review_required": False,
                    "group_id": "product_001",
                }
            ],
            "tables": [],
            "field_groups": [],
            "lists": [],
        }
        lines = [_glm_line("ocr_1", "block_1", 1, "text", "错误文字", 0.1, 0.1, 0.3, 0.12)]
        package_structure = {
            "enabled": True,
            "fields": [
                {
                    "semantic_key": "product.name",
                    "label": "产品名称",
                    "text": "红豆奶茶",
                    "source_ocr_line_ids": ["ocr_1"],
                    "bbox_normalized": {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.12},
                    "confidence": 0.99,
                    "review_required": False,
                }
            ],
            "nutrition_tables": [],
        }

        result = compare_standard_to_ocr(artifacts, lines, PACKAGE_IMAGE, package_structure=package_structure)["comparison_result"]["results"][0]

        self.assertEqual(result["status"], "critical_mismatch")
        self.assertEqual(result["selected_candidate"]["candidate_type"], "package_structured_field")
        self.assertIn("llm_structured_package_item", result["match_quality_flags"])

    def test_hybrid_compare_uses_ppocr_fields_and_glm_nutrition_tables(self) -> None:
        artifacts = {
            "standard_items": [
                {
                    "id": "std_1",
                    "semantic_key": "product.name",
                    "label": "产品名称",
                    "text": "产品名称：红豆奶茶",
                    "comparison_required": True,
                    "review_required": False,
                    "group_id": "product_001",
                }
            ],
            "field_groups": [],
            "lists": [],
            "tables": [
                {
                    "table_id": "nutrition_n1",
                    "table_type": "nutrition_facts",
                    "title": "营养成分表",
                    "columns": [
                        {"column_id": "item", "name": "项目"},
                        {"column_id": "amount", "name": "每100克"},
                        {"column_id": "nrv", "name": "NRV%"},
                    ],
                    "rows": [
                        {
                            "row_key": "能量",
                            "cells": [
                                {"column_id": "item", "raw_value": "能量"},
                                {"column_id": "amount", "raw_value": "100千焦"},
                                {"column_id": "nrv", "raw_value": "1%"},
                            ],
                        },
                        {
                            "row_key": "--糖",
                            "cells": [
                                {"column_id": "item", "raw_value": "--糖"},
                                {"column_id": "amount", "raw_value": "2.1 克"},
                                {"column_id": "nrv", "raw_value": "-"},
                            ],
                        }
                    ],
                    "footnotes": [],
                }
            ],
        }
        ppocr_lines = [
            _ppocr_line("pp_1", "产品名称：红豆奶茶", 0.1, 0.1, 0.3, 0.12),
            _ppocr_line("pp_2", "营养成分表", 0.1, 0.2, 0.25, 0.22),
            _ppocr_line("pp_3", "项目 每100克 NRV%", 0.1, 0.23, 0.4, 0.25),
            _ppocr_line("pp_4", "能量 999千焦 99%", 0.1, 0.26, 0.4, 0.28),
        ]
        package_structure = {
            "enabled": True,
            "fields": [
                {
                    "semantic_key": "product.name",
                    "label": "产品名称",
                    "text": "产品名称：错误奶茶",
                    "source_ocr_line_ids": ["glm_1"],
                    "bbox_normalized": {"x1": 0.5, "y1": 0.1, "x2": 0.7, "y2": 0.12},
                    "confidence": 0.99,
                    "review_required": False,
                }
            ],
            "nutrition_tables": [
                {
                    "table_id": "nutrition_n1",
                    "title": "营养成分表",
                    "columns": [
                        {"column_id": "col_001", "name": "项目"},
                        "{'column_id': 'col_002', 'name': '每100克'}",
                        {"column_id": "col_003", "name": "NRV%"},
                    ],
                    "rows": [
                        {
                            "row_key": "能量",
                            "cells": ["能量", "100千焦", "1%"],
                            "source_ocr_line_ids": ["glm_table_1"],
                            "bbox_normalized": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.3},
                            "confidence": 0.99,
                            "review_required": False,
                        },
                        {
                            "row_key": "糖",
                            "cells": ["糖", "2.1克", "-"],
                            "source_ocr_line_ids": ["glm_table_1"],
                            "bbox_normalized": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.3},
                            "confidence": 0.99,
                            "review_required": False,
                        }
                    ],
                    "footnotes": [],
                }
            ],
        }

        comparison = compare_standard_to_ocr(
            artifacts,
            ppocr_lines,
            PACKAGE_IMAGE,
            package_structure=package_structure,
            package_structure_scope="nutrition",
        )["comparison_result"]
        results = comparison["results"]

        name = _result_for(results, semantic_key="product.name")
        header = _result_for(results, target_type="nutrition_header", table_id="nutrition_n1")
        row = _result_for(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="能量")
        sugar = _result_for(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="--糖")
        self.assertEqual(name["status"], "pass")
        self.assertEqual(name["selected_candidate"]["source_ocr_line_ids"], ["pp_1"])
        self.assertEqual(header["status"], "pass")
        self.assertEqual(header["selected_candidate"]["text"], "项目 每100克 NRV%")
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["selected_candidate"]["candidate_type"], "package_structured_nutrition_row")
        self.assertIn("glm_structured_table_match", row["match_quality_flags"])
        self.assertIn("ppocr_glm_table_conflict", row["match_quality_flags"])
        self.assertEqual(sugar["status"], "pass")
        self.assertEqual(sugar["selected_candidate"]["text"], "糖 2.1克 -")

    def test_all_scope_structured_fields_win_over_contaminated_ppocr_candidates(self) -> None:
        artifacts = {
            "standard_items": [
                _standard_item("std_1", "product.ingredients", "配料", "配料：葡萄糖浆，白砂糖，青提汁≥1.5%"),
                _standard_item("std_2", "custom.allergen_notice", "致敏原提示", "致敏物质提示：可能含有乳制品、蛋制品。"),
                _standard_item("std_3", "product.storage_condition", "贮存条件", "贮存条件：勿置于阳光直射及潮湿处，室温下保存。"),
                {**_standard_item("std_4", "manufacturer.name", "企业名称", "生产者：东莞徐记食品有限公司"), "group_id": "manufacturer_e1"},
                {**_standard_item("std_5", "manufacturer.address", "地址", "地址：广东省东莞市东城街道狮长路29号"), "group_id": "manufacturer_e1"},
            ],
            "field_groups": [],
            "lists": [],
            "tables": [],
        }
        ppocr_lines = [
            _ppocr_line("pp_1", "配料：葡萄糖浆，白砂糖，青提汁≥1.5%致敏物质提示：可能含有乳制品、蛋制品。", 0.1, 0.1, 0.8, 0.12),
            _ppocr_line("pp_2", "请注意：温度超过36℃将会造", 0.1, 0.14, 0.4, 0.16),
            _ppocr_line("pp_3", "生产者：东莞徐记食品有限", 0.1, 0.18, 0.4, 0.2),
            _ppocr_line("pp_4", "公司 地址：广东省东莞市东城街道狮长路29号", 0.1, 0.22, 0.7, 0.24),
        ]
        package_structure = {
            "enabled": True,
            "fields": [
                _structured_field("product.ingredients", "配料", "配料：葡萄糖浆，白砂糖，青提汁≥1.5%", "glm_1"),
                _structured_field("custom.allergen_notice", "致敏原提示", "致敏物质提示：可能含有乳制品、蛋制品。", "glm_2"),
                _structured_field("product.storage_condition", "贮存条件", "贮存条件：勿置于阳光直射及潮湿处，室温下保存。", "glm_3"),
                _structured_field("manufacturer.name", "企业名称", "生产者：东莞徐记食品有限公司", "glm_4", group_id="manufacturer_e1"),
                _structured_field("manufacturer.address", "地址", "地址：广东省东莞市东城街道狮长路29号", "glm_5", group_id="manufacturer_e1"),
            ],
            "nutrition_tables": [],
        }

        results = compare_standard_to_ocr(
            artifacts,
            ppocr_lines,
            PACKAGE_IMAGE,
            package_structure=package_structure,
            package_structure_scope="all",
        )["comparison_result"]["results"]

        for semantic_key in ("product.ingredients", "custom.allergen_notice", "product.storage_condition", "manufacturer.name", "manufacturer.address"):
            result = _result_for(results, semantic_key=semantic_key)
            self.assertEqual(result["status"], "pass", semantic_key)
            self.assertEqual(result["selected_candidate"]["candidate_type"], "package_structured_field", semantic_key)
            self.assertIn("glm_structured_field_match", result["match_quality_flags"], semantic_key)
            if semantic_key != "custom.allergen_notice":
                self.assertIn("ppocr_glm_field_conflict", result["match_quality_flags"], semantic_key)

    def test_review_required_structured_field_does_not_auto_pass_when_ppocr_can_match(self) -> None:
        artifacts = {
            "standard_items": [_standard_item("std_1", "product.storage_condition", "贮存条件", "贮存条件：阴凉干燥处保存")],
            "field_groups": [],
            "lists": [],
            "tables": [],
        }
        package_structure = {
            "enabled": True,
            "fields": [
                {
                    **_structured_field("product.storage_condition", "贮存条件", "贮存条件：阴凉干燥处保存", "glm_1"),
                    "review_required": True,
                }
            ],
            "nutrition_tables": [],
        }

        result = compare_standard_to_ocr(
            artifacts,
            [_ppocr_line("pp_1", "贮存条件：阴凉干燥处保存", 0.1, 0.1, 0.4, 0.12)],
            PACKAGE_IMAGE,
            package_structure=package_structure,
            package_structure_scope="all",
        )["comparison_result"]["results"][0]

        self.assertEqual(result["status"], "pass")
        self.assertNotEqual(result["selected_candidate"]["candidate_type"], "package_structured_field")
        self.assertIn("ppocr_glm_field_conflict", result["match_quality_flags"])

    def test_barcode_target_does_not_use_llm_structured_field(self) -> None:
        artifacts = {
            "standard_items": [_standard_item("std_1", "barcode.commodity", "条码", "6900000000000")],
            "field_groups": [],
            "lists": [],
            "tables": [],
        }
        package_structure = {
            "enabled": True,
            "fields": [_structured_field("barcode.commodity", "条码", "6900000000000", "glm_1")],
            "nutrition_tables": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            blank_image = Path(temp_dir) / "blank.png"
            blank_image.write_bytes(base64.b64decode(_BLANK_PNG_BASE64))

            result = compare_standard_to_ocr(
                artifacts,
                [_ppocr_line("pp_noise", "量", 0.1, 0.1, 0.11, 0.12)],
                blank_image,
                package_structure=package_structure,
                package_structure_scope="all",
            )["comparison_result"]["results"][0]

        self.assertEqual(result["status"], "critical_missing")
        self.assertIsNone(result["selected_candidate"])

    def test_generic_other_text_structured_candidates_do_not_cross_repeat_items(self) -> None:
        artifacts = {
            "standard_items": [_standard_item("std_1", "custom.other_label_text", "标识", "1/3包")],
            "field_groups": [],
            "lists": [],
            "tables": [],
        }
        package_structure = {
            "enabled": True,
            "fields": [
                _structured_field("custom.other_label_text", "标识", "合格", "glm_1"),
                _structured_field("custom.other_label_text", "条形码", "6914782221734", "glm_2"),
            ],
            "nutrition_tables": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            blank_image = Path(temp_dir) / "blank.png"
            blank_image.write_bytes(base64.b64decode(_BLANK_PNG_BASE64))

            result = compare_standard_to_ocr(
                artifacts,
                [],
                blank_image,
                package_structure=package_structure,
                package_structure_scope="all",
            )["comparison_result"]["results"][0]

        self.assertEqual(result["status"], "critical_missing")
        self.assertIsNone(result["selected_candidate"])

    def test_generic_other_long_text_missing_does_not_select_ppocr_noise(self) -> None:
        artifacts = {
            "standard_items": [
                _standard_item(
                    "std_1",
                    "custom.other_label_text",
                    "宣传语/卖点",
                    "** 72亿颗是指徐福记生产的所有果汁含量≥2.5%的软糖产品年销售数量,源自徐福记2024年1月-12月内部销售数据统计",
                )
            ],
            "field_groups": [],
            "lists": [],
            "tables": [],
        }
        package_structure = {
            "enabled": True,
            "fields": [
                _structured_field(
                    "custom.other_label_text",
                    "宣传语/卖点",
                    "*100%果汁指总果汁含量以原果汁计≥100%",
                    "glm_1",
                )
            ],
            "nutrition_tables": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            blank_image = Path(temp_dir) / "blank.png"
            blank_image.write_bytes(base64.b64decode(_BLANK_PNG_BASE64))

            result = compare_standard_to_ocr(
                artifacts,
                [_ppocr_line("pp_noise", "量", 0.1, 0.1, 0.11, 0.12)],
                blank_image,
                package_structure=package_structure,
                package_structure_scope="all",
            )["comparison_result"]["results"][0]

        self.assertEqual(result["status"], "critical_missing")
        self.assertIsNone(result["selected_candidate"])

    def test_run_compare_package_image_with_glm_fixture_and_fake_llm_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            standard_dir = temp_path / "standard"
            output_dir = temp_path / "out"
            fixture = temp_path / "glm_ocr.json"
            standard_dir.mkdir()
            _write_standard_artifacts(standard_dir)
            fixture.write_text(json.dumps(_glm_fixture(), ensure_ascii=False), encoding="utf-8")

            result = run_compare_package_image(
                standard_dir=standard_dir,
                image_path=PACKAGE_IMAGE,
                output_dir=output_dir,
                glm_ocr_fixture_path=fixture,
                ocr_mode="glm",
                llm_mode="required",
                llm_client=FakePackageStructureLlm(),
            )

            self.assertEqual(result["pass_count"], result["target_count"])
            self.assertTrue((output_dir / "package_glm_blocks.json").exists())
            self.assertTrue((output_dir / "package_llm_structure_input.json").exists())
            self.assertTrue((output_dir / "package_llm_structure_output.json").exists())
            self.assertTrue((output_dir / "package_structured_items.json").exists())
            self.assertTrue((output_dir / "package_structure_quality_report.json").exists())
            runtime = json.loads((output_dir / "runtime_policy.json").read_text(encoding="utf-8"))
            self.assertTrue(runtime["llm_structure"]["enabled"])
            self.assertEqual(runtime["llm_structure"]["final_decision_owner"], "rules")

    def test_llm_mode_auto_without_client_disables_structure_and_preserves_artifacts(self) -> None:
        blocks = build_package_glm_blocks([_glm_line("ocr_1", "block_1", 1, "text", "产品名称：红豆奶茶", 0.1, 0.1, 0.3, 0.12)])

        run = run_package_structure_stage(artifacts={"standard_items": []}, ocr_lines=[_glm_line("ocr_1", "block_1", 1, "text", "产品名称：红豆奶茶", 0.1, 0.1, 0.3, 0.12)], llm_mode="auto", llm_client=None)

        self.assertEqual(blocks["provider_features"]["provider"], "glm_ocr")
        self.assertFalse(run.package_structured_items["enabled"])
        self.assertEqual(run.package_structure_quality_report["status"], "disabled")

    def test_package_structure_llm_input_includes_contract_and_large_output_budget(self) -> None:
        fake = BudgetCapturingPackageStructureLlm()

        run_package_structure_stage(
            artifacts={"standard_items": []},
            ocr_lines=[_glm_line("ocr_1", "block_1", 1, "text", "产品名称：红豆奶茶", 0.1, 0.1, 0.3, 0.12)],
            llm_mode="required",
            llm_client=fake,
        )

        body = json.loads(fake.user)
        self.assertEqual(fake.max_tokens, PACKAGE_STRUCTURE_LLM_MAX_TOKENS)
        self.assertIn("output_contract", body)
        self.assertEqual(
            body["output_contract"]["root_required_keys"],
            ["fields", "field_groups", "content_items", "nutrition_tables", "other_text", "warnings"],
        )


class FakePackageStructureLlm(LlmClient):
    def structured_json(self, system: str, user: str, schema: dict) -> dict:
        del system, schema
        body = json.loads(user)
        blocks = body["glm_blocks"]
        title_block = blocks[0]
        table_block = blocks[1]
        cells = table_block["table"]["rows"]
        return {
            "fields": [
                {
                    "semantic_key": "product.name",
                    "label": "产品名称",
                    "text": "产品名称：红豆奶茶",
                    "source_ids": [title_block["block_id"]],
                    "group_id": None,
                    "table_id": None,
                    "row_key": None,
                    "confidence": 0.99,
                    "review_required": False,
                }
            ],
            "field_groups": [],
            "content_items": [],
            "nutrition_tables": [
                {
                    "table_id": "nutrition_n1",
                    "title": "营养成分表",
                    "source_ids": [table_block["block_id"]],
                    "columns": ["项目", "每100克", "NRV%"],
                    "rows": [
                        {
                            "row_key": "能量",
                            "cells": ["能量", "100千焦", "1%"],
                            "source_ids": [cell["cell_id"] for cell in cells[1]["cells"]],
                            "confidence": 0.99,
                            "review_required": False,
                        }
                    ],
                    "footnotes": [],
                    "confidence": 0.99,
                    "review_required": False,
                }
            ],
            "other_text": [],
            "warnings": [],
        }


class BudgetCapturingPackageStructureLlm(FakePackageStructureLlm):
    def __init__(self) -> None:
        self.user = ""
        self.max_tokens = None

    def structured_json_with_max_tokens(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        del system, schema
        self.user = user
        self.max_tokens = max_tokens
        return {
            "fields": [],
            "field_groups": [],
            "content_items": [],
            "nutrition_tables": [],
            "other_text": [],
            "warnings": [],
        }


def _write_standard_artifacts(standard_dir: Path) -> None:
    (standard_dir / "standard_items.json").write_text(
        json.dumps(
            [
                {
                    "id": "std_1",
                    "semantic_key": "product.name",
                    "label": "产品名称",
                    "text": "产品名称：红豆奶茶",
                    "comparison_required": True,
                    "review_required": False,
                    "group_id": "product_001",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (standard_dir / "tables.json").write_text(
        json.dumps(
            [
                {
                    "table_id": "nutrition_n1",
                    "table_type": "nutrition_facts",
                    "title": "营养成分表",
                    "columns": [
                        {"column_id": "item", "name": "项目"},
                        {"column_id": "amount", "name": "每100克"},
                        {"column_id": "nrv", "name": "NRV%"},
                    ],
                    "rows": [
                        {
                            "row_key": "能量",
                            "cells": [
                                {"column_id": "item", "raw_value": "能量"},
                                {"column_id": "amount", "raw_value": "100千焦"},
                                {"column_id": "nrv", "raw_value": "1%"},
                            ],
                        }
                    ],
                    "footnotes": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (standard_dir / "field_groups.json").write_text("[]", encoding="utf-8")
    (standard_dir / "lists.json").write_text("[]", encoding="utf-8")


def _glm_fixture() -> dict:
    return {
        "layout_details": [
            [
                {
                    "index": 1,
                    "label": "text",
                    "bbox_2d": [10, 10, 300, 50],
                    "content": "## 产品名称：红豆奶茶",
                    "width": 1000,
                    "height": 1000,
                },
                {
                    "index": 2,
                    "label": "table",
                    "bbox_2d": [10, 80, 400, 180],
                    "content": "<table><tr><td>项目</td><td>每100克</td><td>NRV%</td></tr><tr><td>能量</td><td>100千焦</td><td>1%</td></tr></table>",
                    "width": 1000,
                    "height": 1000,
                },
            ]
        ],
        "data_info": {"pages": [{"width": 1000, "height": 1000}]},
    }


def _glm_line(line_id: str, block_id: str, detail_index: int, label: str, text: str, x1: float, y1: float, x2: float, y2: float) -> OcrLine:
    return OcrLine(
        ocr_line_id=line_id,
        page=1,
        text=text,
        confidence=1.0,
        bbox_normalized=BBoxNormalized(x1=x1, y1=y1, x2=x2, y2=y2),
        block_id=block_id,
        metadata={"provider": "glm_ocr", "block_id": block_id, "detail_index": detail_index, "detail_label": label},
    )


def _ppocr_line(line_id: str, text: str, x1: float, y1: float, x2: float, y2: float) -> OcrLine:
    return OcrLine(
        ocr_line_id=line_id,
        page=1,
        text=text,
        confidence=0.99,
        bbox_normalized=BBoxNormalized(x1=x1, y1=y1, x2=x2, y2=y2),
        metadata={"provider": "ppocrv6"},
    )


def _standard_item(item_id: str, semantic_key: str, label: str, text: str) -> dict:
    return {
        "id": item_id,
        "semantic_key": semantic_key,
        "label": label,
        "text": text,
        "comparison_required": True,
        "review_required": False,
        "group_id": "product_001",
    }


def _structured_field(semantic_key: str, label: str, text: str, line_id: str, group_id: str | None = None) -> dict:
    return {
        "semantic_key": semantic_key,
        "label": label,
        "text": text,
        "source_ocr_line_ids": [line_id],
        "bbox_normalized": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.12},
        "group_id": group_id,
        "confidence": 0.99,
        "review_required": False,
    }


def _result_for(
    results: list[dict],
    *,
    semantic_key: str | None = None,
    target_type: str | None = None,
    table_id: str | None = None,
    row_key: str | None = None,
) -> dict:
    for result in results:
        if semantic_key is not None and result.get("semantic_key") != semantic_key:
            continue
        if target_type is not None and result.get("target_type") != target_type:
            continue
        if table_id is not None and result.get("table_id") != table_id:
            continue
        if row_key is not None and result.get("row_key") != row_key:
            continue
        return result
    raise AssertionError("No matching result found")


_BLANK_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


if __name__ == "__main__":
    unittest.main()
