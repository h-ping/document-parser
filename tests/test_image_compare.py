import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from document_parser.image_compare import (
    build_ocr_quality_report,
    compare_standard_to_ocr,
    normalize_compare_text,
    normalize_ppocr_fixture_page,
)
from document_parser.image_compare_cli import image_size
from document_parser.image_compare_html import build_image_compare_html
from document_parser.models import BBoxNormalized, OcrLine, PageInfo
from document_parser.ocr import RecordedOcrClient


ROOT = Path(__file__).resolve().parents[1]
STANDARD_DIR = ROOT / "out" / "standard_xlsx_html_preview"
PACKAGE_IMAGE = ROOT / "test_png" / "粽子包装图.jpg"
OCR_FIXTURE = ROOT / "test_ocr" / "zongzi_ocr_result.json"
XUFUJI_IMAGE = ROOT / "test_png" / "果汁软糖包装图.jpg"
XUFUJI_OCR_FIXTURE = ROOT / "test_ocr" / "xufuji_ocr_result.json"


class ImageCompareTests(unittest.TestCase):
    def test_ocr_fixture_normalizes_to_lines_with_bbox(self) -> None:
        width, height = image_size(PACKAGE_IMAGE)
        lines = RecordedOcrClient(OCR_FIXTURE).recognize_image(PACKAGE_IMAGE, PageInfo(page=1, width=width, height=height))

        self.assertEqual(len(lines), 294)
        self.assertEqual(sum(1 for line in lines if line.bbox_normalized is not None), 294)
        self.assertEqual(lines[0].text, "产品名称:粽粽有礼粽子礼盒")

    def test_pdf_type_fixture_uses_page_size_for_bbox_space(self) -> None:
        if not XUFUJI_OCR_FIXTURE.exists():
            self.skipTest("xufuji OCR fixture is not available")
        image_width, image_height = image_size(XUFUJI_IMAGE)
        ocr_width, ocr_height = normalize_ppocr_fixture_page(XUFUJI_OCR_FIXTURE, image_width, image_height)
        lines = RecordedOcrClient(XUFUJI_OCR_FIXTURE).recognize_image(
            XUFUJI_IMAGE,
            PageInfo(page=1, width=ocr_width, height=ocr_height),
        )
        quality = build_ocr_quality_report(
            fixture_path=XUFUJI_OCR_FIXTURE,
            input_image_width=image_width,
            input_image_height=image_height,
            ocr_page_width=ocr_width,
            ocr_page_height=ocr_height,
            ocr_lines=lines,
        )

        self.assertEqual((ocr_width, ocr_height), (1250, 1584))
        self.assertGreater(quality["bbox_range"]["x2"], 0.8)
        self.assertEqual(quality["bbox_overlay_status"], "source_image_mismatch")

    def test_normalization_allows_spacing_and_punctuation_but_not_missing_prefix(self) -> None:
        self.assertEqual(
            normalize_compare_text("产品 名称 ：粽子"),
            normalize_compare_text("产品名称:粽子"),
        )
        self.assertEqual(
            normalize_compare_text("净含量/规格：1.4 千克（100 克×1）"),
            normalize_compare_text("净含量/规格:1.4千克(100克x1)"),
        )
        self.assertNotEqual(
            normalize_compare_text("产品名称：粽粽有礼粽子礼盒"),
            normalize_compare_text("粽粽有礼粽子礼盒"),
        )

    def test_html_renders_bbox_when_line_contains_dataclass_bbox(self) -> None:
        html = build_image_compare_html(
            output_path=Path("/tmp/result_preview.html"),
            image_path=PACKAGE_IMAGE,
            image_width=100,
            image_height=100,
            package_ocr_lines=[
                _ocr_line("ocr_0001", "产品名称：红豆奶茶", 0.1, 0.1, 0.3, 0.12).__dict__,
                _ocr_line("ocr_0002", "净含量：100克", 0.2, 0.2, 0.4, 0.22),
            ],
            comparison_result={"status": "pass", "target_count": 0, "pass_count": 0, "critical_count": 0, "manual_review_count": 0, "results": []},
        )

        self.assertEqual(html.count('class="ocr-box"'), 2)
        self.assertIn('data-line-id="ocr_0001"', html)
        self.assertIn('data-line-id="ocr_0002"', html)

    def test_html_supports_image_zoom_and_selected_bbox_focus(self) -> None:
        html = build_image_compare_html(
            output_path=Path("/tmp/result_preview.html"),
            image_path=PACKAGE_IMAGE,
            image_width=100,
            image_height=100,
            package_ocr_lines=[],
            comparison_result={
                "status": "pass",
                "target_count": 0,
                "pass_count": 0,
                "critical_count": 0,
                "manual_review_count": 0,
                "results": [],
            },
        )

        self.assertIn('id="zoom-out"', html)
        self.assertIn('id="zoom-in"', html)
        self.assertIn('id="zoom-reset"', html)
        self.assertIn('id="zoom-label" aria-live="polite">100%</span>', html)
        self.assertIn("const AUTO_FOCUS_ZOOM = 2;", html)
        self.assertIn("setImageZoom(Math.max(imageZoom, AUTO_FOCUS_ZOOM));", html)
        self.assertIn("requestAnimationFrame(() => first.scrollIntoView", html)
        self.assertIn('class="image-viewport"', html)
        self.assertIn("document.querySelector('.image-viewport')", html)
        self.assertNotIn("document.querySelector('.image-pane');", html)

    def test_short_field_does_not_pass_by_substring_inside_longer_line(self) -> None:
        artifacts = {
            "standard_items": [
                {
                    "id": "std_0001",
                    "semantic_key": "custom.brand_text",
                    "label": "品牌文字",
                    "text": "广州酒家",
                    "comparison_required": True,
                    "review_required": False,
                    "group_id": "product_001",
                }
            ],
            "tables": [],
            "field_groups": [],
            "lists": [],
        }
        lines = [
            OcrLine(
                ocr_line_id="ocr_0001",
                page=1,
                text="委托方：广州酒家集团食品营销管理有限公司",
                confidence=0.99,
                bbox_normalized=BBoxNormalized(x1=0.1, y1=0.1, x2=0.4, y2=0.12),
            )
        ]

        result = compare_standard_to_ocr(artifacts, lines, PACKAGE_IMAGE)["comparison_result"]

        self.assertEqual(result["results"][0]["status"], "critical_missing")

    def test_multi_field_ocr_line_can_be_split_by_known_prefix(self) -> None:
        artifacts = {
            "standard_items": [
                _standard_item("std_0001", "manufacturer.address", "地址", "地址：南京市江宁区湖熟街道6号"),
                _standard_item("std_0002", "manufacturer.origin", "产地", "产地：江苏南京"),
            ],
            "tables": [],
            "field_groups": [],
            "lists": [],
        }
        lines = [
            _ocr_line("ocr_0001", "地址：南京市江宁区湖熟街道6号产地：江苏南京", 0.1, 0.1, 0.5, 0.12)
        ]

        results = compare_standard_to_ocr(artifacts, lines, PACKAGE_IMAGE)["comparison_result"]["results"]

        self.assertEqual(_status(results, semantic_key="manufacturer.address"), "pass")
        self.assertEqual(_status(results, semantic_key="manufacturer.origin"), "pass")
        origin = _result_for(results, semantic_key="manufacturer.origin")
        self.assertEqual(origin["selected_candidate"]["candidate_type"], "split_line")
        self.assertIn("multi_field_ocr_line", origin["match_quality_flags"])

    def test_right_side_enterprise_anchor_assigns_group_scope(self) -> None:
        artifacts = {
            "standard_items": [
                {
                    **_standard_item("std_0001", "manufacturer.name", "生产者", "生产者：东莞徐记食品有限公司"),
                    "group_id": "manufacturer_e2",
                },
                {
                    **_standard_item("std_0002", "manufacturer.address", "地址", "地址：广东省东莞市"),
                    "group_id": "manufacturer_e2",
                },
            ],
            "tables": [],
            "field_groups": [],
            "lists": [],
        }
        lines = [
            _ocr_line("ocr_0001", "地址：错误地址", 0.1, 0.14, 0.22, 0.16),
            _ocr_line("ocr_0002", "生产者：东莞徐记食品", 0.55, 0.1, 0.72, 0.12),
            _ocr_line("ocr_0003", "有限公司", 0.55, 0.12, 0.62, 0.14),
            _ocr_line("ocr_0004", "地址：广东省东莞市", 0.55, 0.14, 0.7, 0.16),
        ]

        results = compare_standard_to_ocr(artifacts, lines, PACKAGE_IMAGE)["comparison_result"]["results"]

        address = _result_for(results, semantic_key="manufacturer.address")
        self.assertEqual(address["status"], "pass")
        self.assertTrue(str(address.get("scope_id") or "").startswith("enterprise_"))
        self.assertEqual(address["selected_candidate"]["text"], "地址：广东省东莞市")

    def test_long_text_sequence_ignores_noise_lines(self) -> None:
        artifacts = {
            "standard_items": [
                _standard_item(
                    "std_0001",
                    "product.directions",
                    "冲调方法",
                    "冲调方法：将奶茶粉倒入杯中加入热水搅拌均匀即可饮用",
                )
            ],
            "tables": [],
            "field_groups": [],
            "lists": [],
        }
        lines = [
            _ocr_line("ocr_0001", "冲调方法：将奶茶粉倒入杯中", 0.1, 0.1, 0.38, 0.12),
            _ocr_line("ocr_0002", "0", 0.7, 0.118, 0.72, 0.13),
            _ocr_line("ocr_0003", "优乐美", 0.7, 0.135, 0.76, 0.15),
            _ocr_line("ocr_0004", "加入热水搅拌均匀即可饮用", 0.1, 0.14, 0.38, 0.16),
        ]

        result = compare_standard_to_ocr(artifacts, lines, PACKAGE_IMAGE)["comparison_result"]["results"][0]

        self.assertEqual(result["status"], "pass")
        self.assertNotIn("0", result["selected_candidate"]["text"])
        self.assertNotIn("优乐美", result["selected_candidate"]["text"])

    def test_structured_nutrition_row_binds_label_value_and_nrv(self) -> None:
        artifacts = {
            "standard_items": [],
            "field_groups": [],
            "lists": [],
            "tables": [
                {
                    "table_id": "nutrition_n1",
                    "table_type": "nutrition_facts",
                    "title": "营养成分表",
                    "columns": [
                        {"column_id": "item", "name": "项目"},
                        {"column_id": "amount", "name": "每100克(g)"},
                        {"column_id": "nrv", "name": "营养素参考值%"},
                    ],
                    "rows": [
                        {
                            "row_key": "能量",
                            "cells": [
                                {"column_id": "item", "raw_value": "能量"},
                                {"column_id": "amount", "raw_value": "1048kJ"},
                                {"column_id": "nrv", "raw_value": "12%"},
                            ],
                        }
                    ],
                    "footnotes": [],
                }
            ],
        }
        lines = [
            _ocr_line("ocr_0001", "营养成分表", 0.1, 0.1, 0.2, 0.12),
            _ocr_line("ocr_0002", "项目", 0.1, 0.13, 0.15, 0.145),
            _ocr_line("ocr_0003", "每100克(g)", 0.22, 0.13, 0.32, 0.145),
            _ocr_line("ocr_0004", "营养素参考值%", 0.42, 0.13, 0.55, 0.145),
            _ocr_line("ocr_0005", "能量", 0.1, 0.16, 0.15, 0.175),
            _ocr_line("ocr_0006", "1048kJ", 0.22, 0.16, 0.3, 0.175),
            _ocr_line("ocr_0007", "12%", 0.42, 0.16, 0.46, 0.175),
        ]

        results = compare_standard_to_ocr(artifacts, lines, PACKAGE_IMAGE)["comparison_result"]["results"]

        self.assertEqual(_status(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="能量"), "pass")
        row = _result_for(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="能量")
        self.assertEqual(row["reason"], "nutrition_row_structured_match")

    def test_barcode_missing_reports_decoder_status(self) -> None:
        artifacts = {
            "standard_items": [_standard_item("std_0001", "barcode.commodity", "条码", "6900000000000")],
            "tables": [],
            "field_groups": [],
            "lists": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            blank_image = Path(temp_dir) / "blank.png"
            blank_image.write_bytes(base64.b64decode(_BLANK_PNG_BASE64))

            result = compare_standard_to_ocr(artifacts, [], blank_image)["comparison_result"]["results"][0]

        self.assertEqual(result["status"], "critical_missing")
        self.assertIn(result["reason"], {"barcode_decoder_unavailable", "barcode_not_decoded"})
        self.assertIn(result["reason"], result["match_quality_flags"])

    def test_compare_package_image_cli_with_zongzi_fixture(self) -> None:
        if not STANDARD_DIR.exists():
            self.skipTest("standard fixture output directory is not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "out"
            env = dict(os.environ)
            env.pop("GLM_OCR_API_KEY", None)
            env.pop("ZAI_API_KEY", None)
            env.pop("ZHIPUAI_API_KEY", None)
            env["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_package_image.py"),
                    "--standard-dir",
                    str(STANDARD_DIR),
                    "--image",
                    str(PACKAGE_IMAGE),
                    "--ocr-fixture",
                    str(OCR_FIXTURE),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            for artifact_name in (
                "comparison_result.json",
                "standard_targets.json",
                "package_ocr_lines.json",
                "package_overlay_lines.json",
                "package_ppocr_lines.json",
                "package_glm_lines.json",
                "package_fusion_evidence.json",
                "package_fusion_quality_report.json",
                "package_candidates.json",
                "package_extracted_items.json",
                "unmatched_print_text.json",
                "package_ocr_quality_report.json",
                "package_glm_blocks.json",
                "package_llm_structure_input.json",
                "package_llm_structure_output.json",
                "package_structured_items.json",
                "package_structure_quality_report.json",
                "result_preview.html",
                "artifacts/index.json",
            ):
                self.assertTrue((output_dir / artifact_name).exists(), artifact_name)

            ocr_lines = json.loads((output_dir / "package_ocr_lines.json").read_text(encoding="utf-8"))
            overlay_lines = json.loads((output_dir / "package_overlay_lines.json").read_text(encoding="utf-8"))
            ppocr_lines = json.loads((output_dir / "package_ppocr_lines.json").read_text(encoding="utf-8"))
            fusion = json.loads((output_dir / "package_fusion_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(len(ocr_lines), 294)
            self.assertEqual(len(overlay_lines), 294)
            self.assertEqual(len(ppocr_lines), 294)
            self.assertEqual(fusion["mode"], "ppocr")
            comparison = json.loads((output_dir / "comparison_result.json").read_text(encoding="utf-8"))
            results = comparison["results"]
            for result in results:
                self.assertIn(result.get("issue_origin"), {"package_image", "ocr", "matcher"})
                self.assertIn("issue_category", result)
                self.assertIn("resolution_hint", result)
                self.assertIsInstance(result.get("match_quality_flags"), list)

            self.assertEqual(_status(results, semantic_key="product.name"), "pass")
            self.assertEqual(_status(results, semantic_key="custom.brand_text"), "critical_missing")
            self.assertEqual(_status(results, semantic_key="product.date_marking"), "critical_mismatch")
            self.assertEqual(_status(results, semantic_key="barcode.outer_case"), "critical_missing")
            self.assertEqual(_status(results, semantic_key="content_item.name", group_id="content_c2"), "pass")
            self.assertEqual(_status(results, semantic_key="content_item.ingredients", group_id="content_c2"), "pass")
            self.assertEqual(_status(results, semantic_key="manufacturer.address", group_id="manufacturer_e3"), "pass")
            self.assertEqual(_status(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="能量"), "pass")
            self.assertEqual(_status(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="钠"), "pass")
            self.assertEqual(_status(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="--糖"), "manual_review")

            sugar_row = _result_for(results, target_type="nutrition_row", table_id="nutrition_n1", row_key="--糖")
            self.assertEqual(sugar_row["issue_category"], "table_reconstruction")
            self.assertEqual(sugar_row["issue_origin"], "ocr")

            html = (output_dir / "result_preview.html").read_text(encoding="utf-8")
            self.assertIn("standard-pane", html)
            self.assertIn("standard-doc", html)
            self.assertIn("standard-item", html)
            self.assertIn("image-wrap", html)
            self.assertIn("ocr-box", html)
            self.assertIn("result-card", html)
            self.assertIn("standardItem.scrollIntoView", html)
            self.assertIn("if (card && card.hidden) applyFilter('all', '全部');", html)
            self.assertIn("card.scrollIntoView", html)
            self.assertIn("检查结论", html)
            self.assertIn("不一致/缺失", html)
            self.assertIn("需人工复核", html)
            self.assertIn("包装图多出文字", html)
            self.assertIn("标准文档", html)
            self.assertIn("包装图文字", html)
            self.assertIn("文字识别位置已与包装图对齐", html)
            self.assertIn('data-filter-status="critical"', html)
            self.assertIn('data-filter-status="manual_review"', html)
            self.assertIn('data-filter-status="info_extra_text"', html)
            self.assertIn('data-status="critical_mismatch"', html)
            self.assertIn('data-status="info_extra_text"', html)
            self.assertIn("applyFilter", html)
            self.assertNotIn("semantic_key", html)
            self.assertNotIn("issue_category", html)
            self.assertNotIn("issue_origin", html)
            self.assertNotIn("OCR lines", html)
            self.assertNotIn("OCR bbox", html)
            self.assertNotIn("Matcher", html)
            self.assertNotIn("<details>", html)


def _status(
    results: list[dict],
    *,
    semantic_key: str | None = None,
    target_type: str | None = None,
    group_id: str | None = None,
    table_id: str | None = None,
    row_key: str | None = None,
) -> str:
    for result in results:
        if semantic_key is not None and result.get("semantic_key") != semantic_key:
            continue
        if target_type is not None and result.get("target_type") != target_type:
            continue
        if group_id is not None and result.get("group_id") != group_id:
            continue
        if table_id is not None and result.get("table_id") != table_id:
            continue
        if row_key is not None and result.get("row_key") != row_key:
            continue
        return str(result.get("status"))
    raise AssertionError("No matching result found")


def _result_for(
    results: list[dict],
    *,
    semantic_key: str | None = None,
    target_type: str | None = None,
    group_id: str | None = None,
    table_id: str | None = None,
    row_key: str | None = None,
) -> dict:
    for result in results:
        if semantic_key is not None and result.get("semantic_key") != semantic_key:
            continue
        if target_type is not None and result.get("target_type") != target_type:
            continue
        if group_id is not None and result.get("group_id") != group_id:
            continue
        if table_id is not None and result.get("table_id") != table_id:
            continue
        if row_key is not None and result.get("row_key") != row_key:
            continue
        return result
    raise AssertionError("No matching result found")


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


def _ocr_line(line_id: str, text: str, x1: float, y1: float, x2: float, y2: float) -> OcrLine:
    return OcrLine(
        ocr_line_id=line_id,
        page=1,
        text=text,
        confidence=0.99,
        bbox_normalized=BBoxNormalized(x1=x1, y1=y1, x2=x2, y2=y2),
    )


_BLANK_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


if __name__ == "__main__":
    unittest.main()
