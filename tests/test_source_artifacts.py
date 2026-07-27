import unittest

from document_parser.models import BBoxPdf, BBoxNormalized, CompiledField, Evidence, OcrLine, PageInfo, TextSpan
from document_parser.pdf import PdfPerception
from document_parser.pipeline import _risks_from_source_layers
from document_parser.source_artifacts import build_coverage_map, build_source_layers


class SourceArtifactTests(unittest.TestCase):
    def test_source_layers_summarizes_bbox_and_quality_issues(self) -> None:
        perception = PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100)],
            text_spans=[],
            text_layer_available=True,
            warnings=[],
        )
        spans = [
            TextSpan(
                "span_0001",
                1,
                "品 名：牛奶",
                "pdf_text",
                bbox_pdf=BBoxPdf(1, 2, 30, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.155, 0.12),
            ),
            TextSpan("span_0002", 1, "配料：生牛乳 净含量：250mL 许可证编号：SC123", "pdf_text"),
        ]
        source_layers = build_source_layers(perception, [], spans, {"parsers": ["text_span_nutrition"], "tables": []})

        self.assertEqual(source_layers["layers"]["pdf_text"]["span_count"], 2)
        self.assertEqual(source_layers["text_quality"]["bbox_coverage_rate"], 0.5)
        self.assertEqual(source_layers["status"], "review_required")
        self.assertTrue(any(issue["issue_type"] == "source_bbox_missing" for issue in source_layers["source_issues"]))
        self.assertTrue(any(issue["issue_type"] == "source_cjk_spacing_noise" for issue in source_layers["source_issues"]))

        risks = _risks_from_source_layers(source_layers)
        self.assertTrue(any(risk.risk_type == "source_bbox_missing" for risk in risks))

    def test_source_layers_marks_ocr_only_mode_without_blocking(self) -> None:
        perception = PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100)],
            text_spans=[],
            text_layer_available=False,
            warnings=[],
        )
        bbox = BBoxPdf(1, 2, 30, 10, 200, 100)
        ocr_lines = [
            OcrLine(
                "ocr_0001",
                1,
                "品名：牛奶",
                0.99,
                bbox_pdf=bbox,
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.155, 0.12),
                block_id="ocr_block_001",
                tokens=[
                    {
                        "token_id": "ocr_tok_0001",
                        "page": 1,
                        "text": "品名",
                        "bbox_status": "missing",
                    }
                ],
            )
        ]
        spans = [
            TextSpan(
                "ocr_span_0001",
                1,
                "品名：牛奶",
                "ocr",
                bbox_pdf=bbox,
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.155, 0.12),
                confidence=0.99,
            )
        ]

        source_layers = build_source_layers(perception, ocr_lines, spans, {"parsers": [], "tables": []})

        self.assertEqual(source_layers["source_mode"], "ocr_only")
        self.assertEqual(source_layers["status"], "pass")
        self.assertEqual(source_layers["layers"]["ocr"]["status"], "pass")
        self.assertIsNone(source_layers["layers"]["ocr"]["error"])
        self.assertFalse(source_layers["layers"]["ocr"]["fallback_used"])
        self.assertEqual(source_layers["layers"]["ocr"]["block_count"], 1)
        self.assertEqual(source_layers["layers"]["ocr"]["token_count"], 1)
        self.assertEqual(source_layers["layers"]["ocr"]["blocks"][0]["line_ids"], ["ocr_0001"])
        self.assertEqual(source_layers["layers"]["ocr"]["lines"][0]["tokens"][0]["text"], "品名")
        self.assertTrue(any(issue["issue_type"] == "pdf_text_missing_ocr_used" for issue in source_layers["source_issues"]))

    def test_source_layers_records_ocr_failure_with_pdf_text_fallback(self) -> None:
        perception = PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100)],
            text_spans=[],
            text_layer_available=True,
            warnings=[],
        )
        spans = [
            TextSpan(
                "span_0001",
                1,
                "品名：牛奶",
                "pdf_text",
                bbox_pdf=BBoxPdf(1, 2, 30, 10, 200, 100),
                bbox_normalized=BBoxNormalized(0.005, 0.02, 0.155, 0.12),
            )
        ]

        source_layers = build_source_layers(
            perception,
            [],
            spans,
            {"parsers": [], "tables": []},
            ocr_error="GLM-OCR request failed with HTTP 503",
        )

        self.assertEqual(source_layers["layers"]["ocr"]["status"], "failed")
        self.assertEqual(source_layers["layers"]["ocr"]["error"], "GLM-OCR request failed with HTTP 503")
        self.assertTrue(source_layers["layers"]["ocr"]["fallback_used"])

    def test_coverage_map_links_fields_tables_requirements_and_regions(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：牛奶", "pdf_text"),
            TextSpan("span_0002", 1, "营养成分表", "pdf_text"),
        ]
        evidence = [
            Evidence(
                evidence_id="ev_0001",
                source_text="品名：牛奶",
                page=1,
                extraction_methods=["pdf_text"],
                bbox_status="missing",
                source_node_ids=["span_0001"],
            )
        ]
        field = CompiledField(
            field_id="fld_0001",
            semantic_key="product.name",
            display_name="品名",
            field_type="string",
            raw_value="品名：牛奶",
            clean_value="牛奶",
            normalized_value="牛奶",
            value_hash="hash",
            status="verified",
            criticality="critical",
            confidence={"overall": 1.0},
            risk_level="info",
            review_required=False,
            section_id=None,
            entity_id="product_001",
            table_id=None,
            row_key=None,
            evidence_refs=["ev_0001"],
        )
        coverage_map = build_coverage_map(
            spans,
            {"fld_0001": field},
            evidence,
            [{"table_id": "tbl_0001", "table_type": "nutrition_facts", "source_span_ids": ["span_0002"]}],
            [{"requirement_id": "req_0001", "evidence_refs": ["ev_0001"]}],
            [{"region_id": "reg_0001", "region_type": "nutrition_table_area", "source_span_ids": ["span_0002"]}],
            [
                {"span_id": "span_0001", "anchor_type": "field_anchor", "page": 1, "text": "品名：牛奶"},
                {"span_id": "span_0002", "anchor_type": "nutrition_table_area", "page": 1, "text": "营养成分表"},
            ],
        )

        self.assertEqual(coverage_map["covered_span_count"], 2)
        self.assertEqual(coverage_map["missing_anchor_count"], 0)
        field_anchor = next(anchor for anchor in coverage_map["anchors"] if anchor["span_id"] == "span_0001")
        self.assertEqual(field_anchor["coverage_status"], "covered")
        self.assertTrue(any(mapping["target_type"] == "field" for mapping in field_anchor["mappings"]))
        table_anchor = next(anchor for anchor in coverage_map["anchors"] if anchor["span_id"] == "span_0002")
        self.assertTrue(any(mapping["target_type"] == "table" for mapping in table_anchor["mappings"]))


if __name__ == "__main__":
    unittest.main()
