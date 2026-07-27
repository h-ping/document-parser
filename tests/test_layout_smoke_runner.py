import runpy
import unittest
from pathlib import Path

from document_parser.llm import _chat_completions_url
from document_parser.models import BBoxNormalized, BBoxPdf, TextSpan


RUNNER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "run_layout_structure_smoke.py"))


def _span(span_id: str, text: str, x: float, y: float, width: float = 80) -> TextSpan:
    bbox = BBoxPdf(x=x, y=y, width=width, height=10, page_width=625, page_height=792)
    return TextSpan(
        span_id=span_id,
        page=1,
        text=text,
        source="pdf_char_atom",
        bbox_pdf=bbox,
        bbox_normalized=BBoxNormalized(x / 625, y / 792, (x + width) / 625, (y + 10) / 792),
    )


class LayoutSmokeRunnerTests(unittest.TestCase):
    def test_summary_preserves_review_required_case_status(self) -> None:
        self.assertEqual(RUNNER["_aggregate_status"]("pass", "review_required"), "review_required")
        self.assertEqual(RUNNER["_aggregate_status"]("review_required", "blocked"), "blocked")

    def test_acceptance_text_normalizes_compatible_cjk_glyphs(self) -> None:
        self.assertEqual(RUNNER["_normalized_text"]("蛋⽩质"), "蛋白质")

    def test_ark_v3_base_url_does_not_gain_a_v1_segment(self) -> None:
        self.assertEqual(
            _chat_completions_url("https://ark.cn-beijing.volces.com/api/v3"),
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )
        self.assertEqual(
            RUNNER["_chat_completions_url"]("https://ark.cn-beijing.volces.com/api/v3"),
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )

    def test_single_nutrition_candidate_keeps_left_label_column(self) -> None:
        spans = [
            _span("title", "营养成分表", 248, 450, 50),
            _span("header", "项目", 75, 470, 30),
            _span("label", "能量", 75, 490, 30),
            _span("value", "1390千焦", 310, 490, 60),
            _span("nrv", "17%", 420, 490, 30),
        ]

        candidates = RUNNER["_nutrition_table_candidates"](spans)

        self.assertEqual(len(candidates), 1)
        self.assertIn("label", candidates[0]["source_span_ids"])
        self.assertIn("能量", " ".join(row["text"] for row in candidates[0]["rows"]))


if __name__ == "__main__":
    unittest.main()
