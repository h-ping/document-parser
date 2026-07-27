import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from document_parser.models import PageInfo
from document_parser.pdf import (
    PdfPerceptionReader,
    PdfReadError,
    _bbox_from_pdfplumber_words,
    _group_words_into_lines,
    bbox_from_points,
    normalized_bbox_from_pdf,
)


class PdfGeometryTests(unittest.TestCase):
    def test_groups_words_by_visual_line(self) -> None:
        words = [
            {"text": "品名：", "x0": 10, "x1": 30, "top": 10, "bottom": 20},
            {"text": "红豆奶茶", "x0": 34, "x1": 80, "top": 10.5, "bottom": 20.5},
            {"text": "配料：", "x0": 10, "x1": 30, "top": 40, "bottom": 50},
        ]
        lines = _group_words_into_lines(words)
        self.assertEqual(len(lines), 2)
        self.assertEqual([word["text"] for word in lines[0]], ["品名：", "红豆奶茶"])

    def test_does_not_merge_side_marker_into_table_row(self) -> None:
        words = [
            {"text": "标", "x0": 24.1, "x1": 34.6, "top": 237.2, "bottom": 247.7},
            {"text": "地址：四川省遂宁市经济技术开发区南区内", "x0": 45.388, "x1": 244.888, "top": 245.2, "bottom": 255.7},
            {"text": "产地：四川遂宁", "x0": 276.978, "x1": 350.478, "top": 245.2, "bottom": 255.7},
            {"text": "食品生产许可证编号：SC11351090100010", "x0": 362.147, "x1": 562.718, "top": 244.045, "bottom": 255.7},
        ]

        lines = _group_words_into_lines(words)
        texts = [" ".join(word["text"] for word in line) for line in lines]

        self.assertIn("标", texts)
        self.assertNotIn("标 地址：四川省遂宁市经济技术开发区南区内", texts)
        self.assertIn("地址：四川省遂宁市经济技术开发区南区内", texts)
        self.assertIn("产地：四川遂宁", texts)
        self.assertIn("食品生产许可证编号：SC11351090100010", texts)

    def test_orders_same_visual_line_left_to_right_before_top(self) -> None:
        words = [
            {"text": "地址：深圳市南山区粤海街道蔚蓝海岸社区中心路3033号喜之郎大厦701", "x0": 220.391, "x1": 555.268, "top": 97.045, "bottom": 108.7},
            {"text": "委托方：广东喜之郎集团有限公司", "x0": 45.388, "x1": 202.888, "top": 98.2, "bottom": 108.7},
        ]

        lines = _group_words_into_lines(words)
        texts = [" ".join(word["text"] for word in line) for line in lines]

        self.assertEqual(
            texts,
            [
                "委托方：广东喜之郎集团有限公司",
                "地址：深圳市南山区粤海街道蔚蓝海岸社区中心路3033号喜之郎大厦701",
            ],
        )

    def test_builds_pdf_and_normalized_bbox_from_words(self) -> None:
        page = PageInfo(page=1, width=200, height=100)
        words = [
            {"text": "品名：", "x0": 10, "x1": 30, "top": 10, "bottom": 20},
            {"text": "红豆奶茶", "x0": 34, "x1": 80, "top": 12, "bottom": 22},
        ]
        bbox = _bbox_from_pdfplumber_words(words, page)
        normalized = normalized_bbox_from_pdf(bbox)
        self.assertEqual(bbox.x, 10)
        self.assertEqual(bbox.y, 10)
        self.assertEqual(bbox.width, 70)
        self.assertEqual(bbox.height, 12)
        self.assertEqual(normalized.x1, 0.05)
        self.assertEqual(normalized.x2, 0.4)

    def test_bbox_from_ocr_points_is_clamped_to_page_bounds(self) -> None:
        page = PageInfo(page=1, width=200, height=100)
        bbox, normalized = bbox_from_points(
            [[-10, 20], [250, 20], [250, 120], [-10, 120]],
            page,
            source_width=200,
            source_height=100,
        )

        self.assertEqual(bbox.x, 0)
        self.assertEqual(bbox.y, 20)
        self.assertEqual(bbox.width, 200)
        self.assertEqual(bbox.height, 80)
        self.assertEqual(normalized.x1, 0)
        self.assertEqual(normalized.x2, 1)
        self.assertEqual(normalized.y2, 1)

    def test_invalid_pdf_returns_stable_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_pdf = Path(temp_dir) / "broken.pdf"
            invalid_pdf.write_bytes(b"this is not a pdf")

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(PdfReadError) as context:
                    PdfPerceptionReader().read(invalid_pdf)

        message = str(context.exception)
        self.assertIn("Failed to read PDF", message)
        self.assertIn("pdfplumber failed", message)
        self.assertIn("pypdf failed", message)


if __name__ == "__main__":
    unittest.main()
