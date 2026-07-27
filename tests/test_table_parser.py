import unittest

from document_parser.structures import content_item_names, extract_nutrition_tables_from_layers
from document_parser.table_parser import build_table_parser_outputs, build_table_quality_report
from document_parser.pipeline import _risks_from_table_quality_report, _risks_from_validation, _table_structure_validation_checks
from document_parser.models import BBoxNormalized, BBoxPdf, TextSpan


class TableParserTests(unittest.TestCase):
    def test_builds_table_layers_and_final_tables_from_spans(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "内容物 1：人参乌鸡靓汤粽", "pdf_text"),
            TextSpan("span_0002", 2, "人参乌鸡靓汤粽营养成分表", "pdf_text"),
            TextSpan("span_0003", 2, "项目 每 100 克 营养素参考值%", "pdf_text"),
            TextSpan("span_0004", 2, "能量 810 千焦 10%", "pdf_text"),
            TextSpan("span_0005", 2, "蛋白质 6.7 克 11%", "pdf_text"),
        ]

        table_layers, quality = build_table_parser_outputs(spans)
        self.assertEqual(quality["status"], "pass")
        self.assertEqual(len(table_layers["tables"]), 1)
        self.assertEqual(table_layers["tables"][0]["parser"], "text_span_nutrition")
        self.assertEqual(len(table_layers["tables"][0]["rows"]), 3)

        tables, evidence = extract_nutrition_tables_from_layers(table_layers, content_item_names(spans), 0)
        self.assertEqual(tables[0]["linked_entity_id"], "content_item_001")
        self.assertEqual(tables[0]["table_layer_id"], "tl_tbl_0001")
        self.assertEqual(tables[0]["page"], 2)
        self.assertEqual(tables[0]["rows"][0]["row_key"], "energy")
        self.assertEqual(len(evidence), 3)
        self.assertEqual(tables[0]["evidence_refs"], ["ev_0001", "ev_0002", "ev_0003"])
        self.assertEqual(evidence[0].source_text, "人参乌鸡靓汤粽营养成分表")
        self.assertEqual(evidence[0].page, 2)

    def test_table_layers_preserve_top_level_bbox_for_table_evidence(self) -> None:
        bbox = BBoxPdf(10, 20, 100, 10, 200, 100)
        spans = [
            TextSpan("span_0001", 2, "营养成分表", "pdf_text", bbox_pdf=bbox, bbox_normalized=BBoxNormalized(0.05, 0.2, 0.55, 0.3)),
            TextSpan("span_0002", 2, "项目 每100克 NRV%", "pdf_text", bbox_pdf=bbox, bbox_normalized=BBoxNormalized(0.05, 0.3, 0.55, 0.4)),
            TextSpan("span_0003", 2, "能量 810千焦 10%", "pdf_text", bbox_pdf=bbox, bbox_normalized=BBoxNormalized(0.05, 0.4, 0.55, 0.5)),
        ]

        table_layers, _quality = build_table_parser_outputs(spans)
        tables, evidence = extract_nutrition_tables_from_layers(table_layers, {}, 0)

        self.assertEqual(table_layers["tables"][0]["bbox_status"], "available")
        self.assertEqual(tables[0]["bbox_status"], "available")
        self.assertEqual(evidence[0].bbox_status, "available")
        self.assertEqual(evidence[0].bbox_pdf["x"], 10)

    def test_quality_report_flags_empty_table_rows(self) -> None:
        spans = [TextSpan("span_0001", 1, "营养成分表", "pdf_text")]
        table_layers, quality = build_table_parser_outputs(spans)
        self.assertEqual(len(table_layers["tables"]), 1)
        self.assertEqual(quality["status"], "review_required")
        self.assertEqual(quality["issues"][0]["issue_type"], "nutrition_table_rows_incomplete")

    def test_parser_agreement_reports_table_count_match(self) -> None:
        table_layers = {
            "parsers": ["text_span_nutrition", "pdfplumber"],
            "parser_issues": [],
            "tables": [
                {
                    "table_layer_id": "tl_tbl_0001",
                    "parser": "text_span_nutrition",
                    "table_type": "nutrition_facts",
                    "columns": [{"name": "项目"}, {"name": "含量"}, {"name": "NRV%"}],
                    "rows": [{"row_type": "data"}],
                },
                {
                    "table_layer_id": "pdfplumber_tbl_0001",
                    "parser": "pdfplumber",
                    "table_type": "nutrition_facts",
                    "columns": [{"name": "项目"}, {"name": "含量"}, {"name": "NRV%"}],
                    "rows": [{"row_type": "data"}],
                },
            ],
        }
        quality = build_table_quality_report(table_layers)
        self.assertEqual(quality["parser_agreement"]["status"], "table_count_match")
        self.assertEqual(quality["table_count"], 1)
        self.assertEqual(quality["candidate_table_count"], 2)

    def test_parser_agreement_conflict_becomes_medium_risk(self) -> None:
        quality = {
            "parser_agreement": {"status": "table_count_conflict"},
            "issues": [],
        }
        risks = _risks_from_table_quality_report(quality)
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].risk_level, "medium")
        self.assertEqual(risks[0].risk_type, "parser_agreement_conflict")

    def test_table_structure_validation_passes_for_verified_table(self) -> None:
        table = {
            "table_id": "tbl_0001",
            "table_type": "nutrition_facts",
            "columns": [{"column_id": "col_001"}],
            "rows": [{"row_id": "row_0001"}],
            "criticality": "critical",
            "review_required": False,
            "evidence_refs": ["ev_0001"],
        }
        quality = {
            "status": "pass",
            "parser_agreement": {"status": "table_count_match"},
            "issues": [],
            "issue_count": 0,
        }

        validation = _table_structure_validation_checks([table], quality)

        self.assertEqual(validation[0]["check_type"], "table_structure")
        self.assertEqual(validation[0]["result"], "passed")
        self.assertEqual(validation[1]["target_id"], "table_quality_report")
        self.assertEqual(validation[1]["result"], "passed")

    def test_table_structure_validation_failure_routes_to_risk(self) -> None:
        table = {
            "table_id": "tbl_0001",
            "table_type": "nutrition_facts",
            "columns": [],
            "rows": [],
            "criticality": "critical",
            "review_required": True,
            "evidence_refs": ["ev_0001"],
        }
        quality = {
            "status": "pass",
            "parser_agreement": {"status": "single_parser_only"},
            "issues": [],
            "issue_count": 0,
        }

        validation = _table_structure_validation_checks([table], quality)
        risks = _risks_from_validation(validation)

        self.assertEqual(validation[0]["result"], "failed")
        self.assertEqual(validation[0]["severity"], "high")
        self.assertEqual(risks[0].target_type, "table")
        self.assertEqual(risks[0].target_id, "tbl_0001")
        self.assertEqual(risks[0].risk_type, "table_structure_validation_failed")
        self.assertEqual(risks[0].evidence_refs, ["ev_0001"])

    def test_table_quality_validation_failure_routes_to_parser_risk(self) -> None:
        quality = {
            "status": "review_required",
            "parser_agreement": {"status": "table_count_conflict"},
            "issues": [{"issue_type": "parser_agreement_conflict", "severity": "medium"}],
            "issue_count": 1,
        }

        validation = _table_structure_validation_checks([], quality)
        risks = _risks_from_validation(validation)

        self.assertEqual(validation[0]["target_id"], "table_quality_report")
        self.assertEqual(validation[0]["result"], "failed")
        self.assertEqual(risks[0].target_type, "table_parser")
        self.assertEqual(risks[0].risk_type, "table_structure_validation_failed")


if __name__ == "__main__":
    unittest.main()
