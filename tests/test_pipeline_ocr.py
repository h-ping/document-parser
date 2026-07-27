import tempfile
import unittest
from pathlib import Path

import document_parser.pipeline as pipeline_module
from document_parser.models import BBoxPdf, BBoxNormalized, OcrLine, PageInfo, TextSpan
from document_parser.ocr import OcrClient, OcrError
from document_parser.pdf import PdfPerception
from document_parser.pipeline import DocumentParser


class OcrOnlyClient(OcrClient):
    def recognize_pdf(self, pdf_path, pages):
        return [
            OcrLine(
                "ocr_0001",
                1,
                "品名：牛奶",
                0.99,
                bbox_pdf=BBoxPdf(10, 20, 60, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.05, 0.2, 0.35, 0.3),
            )
        ]


class LowConfidenceSideTextOcrClient(OcrClient):
    def recognize_pdf(self, pdf_path, pages):
        bbox = BBoxPdf(10, 20, 60, 10, 200, 100)
        return [
            OcrLine(
                "ocr_0001",
                1,
                "品名：牛奶",
                0.99,
                bbox_pdf=bbox,
                bbox_normalized=BBoxNormalized(0.05, 0.2, 0.35, 0.3),
            ),
            OcrLine(
                "ocr_0002",
                1,
                "侧边说明",
                0.50,
                bbox_pdf=BBoxPdf(100, 20, 50, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.5, 0.2, 0.75, 0.3),
            ),
        ]


class EmptyTextPdfReader:
    def read(self, path):
        return PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100)],
            text_spans=[],
            text_layer_available=False,
            warnings=[],
        )


class FailingOcrClient(OcrClient):
    def recognize_pdf(self, pdf_path, pages):
        raise OcrError("GLM-OCR request failed with HTTP 503")


class EmptyOcrClient(OcrClient):
    def recognize_pdf(self, pdf_path, pages):
        return []


class PdfTextReader:
    def read(self, path):
        bbox = BBoxPdf(10, 20, 60, 10, 200, 100)
        return PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100)],
            text_spans=[
                TextSpan(
                    "span_0001",
                    1,
                    "品名：牛奶",
                    "pdf_text",
                    bbox_pdf=bbox,
                    bbox_normalized=BBoxNormalized(0.05, 0.2, 0.35, 0.3),
                )
            ],
            text_layer_available=True,
            warnings=[],
        )


class NutritionTablePdfReader:
    def read(self, path):
        bbox = BBoxPdf(10, 20, 160, 10, 200, 100)
        return PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100), PageInfo(page=2, width=200, height=100)],
            text_spans=[
                TextSpan(
                    "span_0001",
                    2,
                    "营养成分表",
                    "pdf_text",
                    bbox_pdf=bbox,
                    bbox_normalized=BBoxNormalized(0.05, 0.2, 0.85, 0.3),
                ),
                TextSpan(
                    "span_0002",
                    2,
                    "项目 每100克 NRV%",
                    "pdf_text",
                    bbox_pdf=bbox,
                    bbox_normalized=BBoxNormalized(0.05, 0.3, 0.85, 0.4),
                ),
                TextSpan(
                    "span_0003",
                    2,
                    "能量 810千焦 10%",
                    "pdf_text",
                    bbox_pdf=bbox,
                    bbox_normalized=BBoxNormalized(0.05, 0.4, 0.85, 0.5),
                ),
            ],
            text_layer_available=True,
            warnings=[],
        )


class PipelineOcrTests(unittest.TestCase):
    def test_pipeline_extracts_fields_from_ocr_only_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=OcrOnlyClient(),
                pdf_reader=EmptyTextPdfReader(),
            ).parse(pdf)

        fields = result.extracted_data["fields"]
        product_name = next(field for field in fields.values() if field["semantic_key"] == "product.name")
        evidence = result.evidence[0]

        self.assertEqual(product_name["raw_value"], "品名：牛奶")
        self.assertEqual(evidence.extraction_methods, ["ocr"])
        self.assertEqual(evidence.bbox_status, "available")
        self.assertFalse(result.document["pdf_text_layer_available"])
        self.assertEqual(result.metadata["source_layers"]["source_mode"], "ocr_only")
        self.assertEqual(result.metadata["source_layers"]["status"], "pass")

    def test_pipeline_routes_low_confidence_ocr_regions_to_risks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=LowConfidenceSideTextOcrClient(),
                pdf_reader=PdfTextReader(),
            ).parse(pdf)

        self.assertTrue(
            any(risk.risk_type == "ocr_low_confidence" and risk.risk_level == "low" for risk in result.risks)
        )
        self.assertTrue(
            any(issue["issue_type"] == "ocr_low_confidence" for issue in result.metadata["source_consistency"]["issues"])
        )

    def test_pipeline_falls_back_to_pdf_text_when_ocr_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=FailingOcrClient(),
                pdf_reader=PdfTextReader(),
            ).parse(pdf)

        fields = result.extracted_data["fields"]
        product_name = next(field for field in fields.values() if field["semantic_key"] == "product.name")

        self.assertEqual(product_name["raw_value"], "品名：牛奶")
        self.assertEqual(result.metadata["source_layers"]["layers"]["ocr"]["status"], "failed")
        self.assertEqual(result.metadata["source_layers"]["layers"]["ocr"]["error"], "GLM-OCR request failed with HTTP 503")
        self.assertTrue(result.metadata["source_layers"]["layers"]["ocr"]["fallback_used"])
        self.assertTrue(any(risk.risk_type == "ocr_failed" for risk in result.risks))
        self.assertTrue(any(task.target_type == "document" and task.target_id == "document" for task in result.review_tasks))

    def test_pipeline_marks_document_partial_failed_when_page_render_is_partial(self) -> None:
        original_render = pipeline_module.render_page_images
        pipeline_module.render_page_images = lambda pdf_path, output_dir: {
            "status": "partial_failed",
            "reason": "page_render_partial_failed",
            "dpi": 144,
            "page_count": 2,
            "rendered_page_count": 1,
            "failed_page_count": 1,
            "pages": [
                {"page": 1, "path": str(Path("page_001.png")), "render_status": "rendered"},
                {"page": 2, "render_status": "failed", "reason": "render_failed", "error": "RuntimeError"},
            ],
            "failed_pages": [{"page": 2, "render_status": "failed", "reason": "render_failed", "error": "RuntimeError"}],
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf = Path(temp_dir) / "input.pdf"
                pdf.write_bytes(b"%PDF-1.4\n")
                result = DocumentParser(
                    ocr_client=OcrOnlyClient(),
                    pdf_reader=PdfTextReader(),
                ).parse(pdf)
        finally:
            pipeline_module.render_page_images = original_render

        self.assertEqual(result.document["parse_status"], "partial_failed")
        self.assertEqual(result.document["page_image_status"], "partial_failed")
        self.assertTrue(any(risk.risk_type == "page_image_render_failed" for risk in result.risks))

    def test_feed_structure_adds_table_backed_standard_item(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, runtime_policy={"table_parser": {"mode": "feed_structure"}})

        items = result.metadata["standard_artifacts"]["standard_items"]
        nutrition_item = next(item for item in items if item["field"] == "nutrition_table")

        self.assertEqual(nutrition_item["semantic_key"], "product.nutrition_table")
        self.assertEqual(nutrition_item["extraction_method"], "table_parser_feed")
        self.assertEqual(nutrition_item["table_id"], "tbl_0001")
        self.assertEqual(nutrition_item["source"]["page"], 2)
        self.assertIn("能量 810千焦 10%", nutrition_item["text"])
        feed_evidence = next(item for item in result.evidence if item.evidence_id in nutrition_item["evidence_refs"])
        self.assertEqual(feed_evidence.page, 2)
        self.assertEqual(result.metadata["table_parser"]["table_feed_items"]["status"], "applied")
        self.assertEqual(result.metadata["output_contract_validation_report"]["status"], "pass")

    def test_validate_only_keeps_table_parser_out_of_standard_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, runtime_policy={"table_parser": {"mode": "validate_only"}})

        fields = {item["field"] for item in result.metadata["standard_artifacts"]["standard_items"]}

        self.assertNotIn("nutrition_table", fields)
        self.assertEqual(result.metadata["table_parser"]["table_feed_items"]["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
