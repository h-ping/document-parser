import unittest

from document_parser.layout_candidates import build_layout_candidates, validate_layout_candidates
from document_parser.models import BBoxNormalized, BBoxPdf, PageInfo, TextSpan


def _span(span_id: str, text: str, x: float, y: float, width: float = 40, page: int = 1) -> TextSpan:
    page_width = 600
    page_height = 800
    bbox = BBoxPdf(x=x, y=y, width=width, height=10, page_width=page_width, page_height=page_height)
    return TextSpan(
        span_id=span_id,
        page=page,
        text=text,
        source="pdf_char_atom",
        bbox_pdf=bbox,
        bbox_normalized=BBoxNormalized(x / page_width, y / page_height, (x + width) / page_width, (y + 10) / page_height),
    )


class LayoutCandidateTests(unittest.TestCase):
    def test_single_nutrition_candidate_includes_left_label_column(self) -> None:
        spans = [
            _span("title", "营养成分表", 250, 450, 60),
            _span("header", "项目", 75, 470),
            _span("label", "能量", 75, 490),
            _span("value", "1390千焦", 310, 490, 70),
            _span("nrv", "17%", 430, 490),
        ]

        artifact = build_layout_candidates(spans, [PageInfo(page=1, width=600, height=800)])

        table = artifact["table_candidates"][0]
        self.assertIn("label", table["source_span_ids"])
        self.assertIn("能量", " ".join(row["text"] for row in table["rows"]))

    def test_parallel_nutrition_tables_are_partitioned_at_title_midpoint(self) -> None:
        spans = [
            _span("title_left", "营养成分表", 90, 100, 60),
            _span("left_energy", "能量", 60, 140),
            _span("title_right", "营养成分表", 390, 100, 60),
            _span("right_energy", "能量", 360, 140),
        ]

        artifact = build_layout_candidates(spans, [PageInfo(page=1, width=600, height=800)])

        left, right = artifact["table_candidates"]
        self.assertIn("left_energy", left["source_span_ids"])
        self.assertNotIn("right_energy", left["source_span_ids"])
        self.assertIn("right_energy", right["source_span_ids"])

    def test_vertical_nutrition_tables_stop_at_next_title(self) -> None:
        spans = [
            _span("title_top", "营养成分表", 250, 100, 60),
            _span("top_energy", "能量", 60, 140),
            _span("title_bottom", "营养成分表", 250, 260, 60),
            _span("bottom_energy", "能量", 60, 300),
        ]

        artifact = build_layout_candidates(spans, [PageInfo(page=1, width=600, height=800)])

        top, bottom = artifact["table_candidates"]
        self.assertNotIn("bottom_energy", top["source_span_ids"])
        self.assertIn("bottom_energy", bottom["source_span_ids"])

    def test_producer_rows_side_marker_and_reading_order_are_traceable(self) -> None:
        spans = [
            _span("marker", "标", 12, 100, 10),
            _span("principal", "委托方：甲公司", 80, 100, 100),
            _span("address", "地址：某地", 220, 100, 100),
        ]

        artifact = build_layout_candidates(spans, [PageInfo(page=1, width=600, height=800)])

        producer = next(item for item in artifact["table_candidates"] if item["table_type"] == "producer_info_repeated_rows")
        self.assertEqual(producer["source_span_ids"], ["principal", "address"])
        self.assertEqual(artifact["side_marker_candidates"][0]["source_span_ids"], ["marker"])
        self.assertTrue(artifact["reading_order_candidates"])

    def test_producer_candidate_includes_same_column_continuation_lines(self) -> None:
        spans = [
            _span("producer", "生产者：甲食品", 300, 100, 100),
            _span("producer_tail", "有限公司", 300, 111, 50),
            _span("address", "地址：广东省东", 300, 122, 100),
            _span("address_tail", "莞市某路1号", 300, 133, 80),
            _span("other_column", "宣传文字", 470, 111, 60),
            _span("later_section", "营养成分表", 250, 220, 60),
        ]

        artifact = build_layout_candidates(spans, [PageInfo(page=1, width=600, height=800)])

        producer = next(item for item in artifact["table_candidates"] if item["table_type"] == "producer_info_repeated_rows")
        self.assertEqual(
            producer["source_span_ids"],
            ["producer", "producer_tail", "address", "address_tail"],
        )

    def test_page_bottom_incomplete_nutrition_table_is_marked_cross_page(self) -> None:
        spans = [
            _span("title", "营养成分表", 250, 720, 60),
            _span("energy", "能量", 60, 755),
        ]

        artifact = build_layout_candidates(spans, [PageInfo(page=1, width=600, height=800)])

        self.assertTrue(artifact["table_candidates"][0]["cross_page_table_continuation_suspected"])
        self.assertEqual(artifact["cross_page_candidate_count"], 1)

    def test_cross_page_candidate_includes_leading_continuation_rows_from_next_page(self) -> None:
        spans = [
            _span("title", "营养成分表", 250, 720, 60),
            _span("energy", "能量", 60, 755),
            _span("carbohydrate", "碳水化合物", 60, 20, page=2),
            _span("carbohydrate_value", "28.8克", 280, 20, page=2),
            _span("next_title", "另一营养成分表", 250, 120, 90, page=2),
        ]

        artifact = build_layout_candidates(spans, [PageInfo(page=1, width=600, height=800), PageInfo(page=2, width=600, height=800)])

        first = artifact["table_candidates"][0]
        self.assertEqual(first["continuation_page"], 2)
        self.assertIn("carbohydrate", first["source_span_ids"])
        self.assertEqual(first["cross_page_pair_status"], "candidate")

    def test_page_bottom_detection_uses_candidate_content_not_only_window_limit(self) -> None:
        spans = [
            _span("title", "营养成分表", 250, 704, 60),
            _span("energy", "能量", 60, 786),
        ]
        page = PageInfo(page=1, width=600, height=842)

        artifact = build_layout_candidates(spans, [page])

        self.assertTrue(artifact["table_candidates"][0]["cross_page_table_continuation_suspected"])

    def test_validation_fails_for_unresolved_candidate_ref(self) -> None:
        artifact = build_layout_candidates(
            [_span("title", "营养成分表", 250, 100, 60)],
            [PageInfo(page=1, width=600, height=800)],
        )
        artifact["table_candidates"][0]["source_span_ids"].append("missing")

        report = validate_layout_candidates(artifact, [_span("title", "营养成分表", 250, 100, 60)])

        self.assertEqual(report["status"], "fail")
        self.assertGreater(report["unresolved_source_ref_count"], 0)


if __name__ == "__main__":
    unittest.main()
