import unittest

from document_parser.models import BBoxPdf, BBoxNormalized, OcrLine, TextSpan
from document_parser.pipeline import _risks_from_source_consistency, _risks_from_validation, _source_consistency_validation_checks
from document_parser.source_consistency import build_source_consistency_report


class SourceConsistencyTests(unittest.TestCase):
    def test_skips_when_no_ocr_lines_are_available(self) -> None:
        report = build_source_consistency_report([TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")], [])
        validation = _source_consistency_validation_checks(report)

        self.assertEqual(report["status"], "skipped_no_ocr")
        self.assertEqual(report["issue_count"], 0)
        self.assertEqual(validation[0]["check_type"], "multi_method_agreement")
        self.assertEqual(validation[0]["result"], "passed")

    def test_passes_when_ocr_text_matches_pdf_text(self) -> None:
        spans = [TextSpan("span_0001", 1, "品名：牛奶", "pdf_text")]
        lines = [OcrLine("ocr_0001", 1, "品名：牛奶", 0.99)]

        report = build_source_consistency_report(spans, lines)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["matched_ocr_line_count"], 1)
        self.assertEqual(report["issue_count"], 0)

    def test_bbox_overlap_text_mismatch_becomes_conflict_risk(self) -> None:
        bbox = BBoxPdf(10, 20, 80, 10, 200, 100)
        spans = [
            TextSpan(
                "span_0001",
                1,
                "品名：牛奶",
                "pdf_text",
                bbox_pdf=bbox,
                bbox_normalized=BBoxNormalized(0.05, 0.2, 0.45, 0.3),
            )
        ]
        lines = [
            OcrLine(
                "ocr_0001",
                1,
                "品名：酸奶",
                0.95,
                bbox_pdf=bbox,
                bbox_normalized=BBoxNormalized(0.05, 0.2, 0.45, 0.3),
            )
        ]

        report = build_source_consistency_report(spans, lines)
        risks = _risks_from_source_consistency(report)
        validation = _source_consistency_validation_checks(report)
        validation_risks = _risks_from_validation(validation)

        self.assertEqual(report["status"], "review_required")
        self.assertTrue(any(issue["issue_type"] == "pdf_ocr_text_conflict" for issue in report["issues"]))
        self.assertEqual(validation[0]["result"], "failed")
        self.assertEqual(validation[0]["severity"], "medium")
        self.assertTrue(any(risk.risk_type == "pdf_ocr_text_conflict" for risk in risks))
        self.assertTrue(any(risk.risk_type == "multi_method_agreement_failed" for risk in validation_risks))

    def test_low_confidence_important_ocr_becomes_medium_issue(self) -> None:
        report = build_source_consistency_report(
            [],
            [OcrLine("ocr_0001", 1, "许可证编号：SC123", 0.50)],
        )

        self.assertEqual(report["status"], "review_required")
        issue = report["issues"][0]
        self.assertEqual(issue["issue_type"], "ocr_low_confidence")
        self.assertEqual(issue["severity"], "medium")

    def test_low_confidence_non_important_ocr_still_enters_risks(self) -> None:
        report = build_source_consistency_report(
            [],
            [OcrLine("ocr_0001", 1, "侧边说明", 0.50)],
        )

        risks = _risks_from_source_consistency(report)

        self.assertEqual(report["status"], "pass")
        issue = report["issues"][0]
        self.assertEqual(issue["issue_type"], "ocr_low_confidence")
        self.assertEqual(issue["severity"], "low")
        self.assertTrue(any(risk.risk_type == "ocr_low_confidence" and risk.risk_level == "low" for risk in risks))


if __name__ == "__main__":
    unittest.main()
