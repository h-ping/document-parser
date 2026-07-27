import tempfile
import unittest
from pathlib import Path

import document_parser.pipeline as pipeline_module
from document_parser.layout_evidence import LayoutEvidence
from document_parser.models import BBoxNormalized, BBoxPdf, ExtractionPlan, FieldDefinition, GeneratedSchema, OcrLine, PageInfo, TextSpan
from document_parser.ocr import OcrClient
from document_parser.pdf import PdfPerception
from document_parser.pipeline import (
    DocumentParser,
    ParseError,
    _agent_table_columns,
    _agent_cell_text,
    _extract_tables_from_accepted_layout_candidates,
    _field_retry_spans,
    _merge_spans,
    _merge_nutrition_marker_cells,
    _schema_from_agent_body,
    _schema_consumption_findings,
    _semantic_repair_spans,
    _source_fusion_validation_checks,
    _table_retry_decision,
)
from document_parser.source_fusion import build_source_fusion


class EmptyOcrClient(OcrClient):
    def recognize_pdf(self, pdf_path, pages):
        return []


class FakePdfReader:
    def read(self, path):
        return PdfPerception(
            pages=[PageInfo(page=1, width=100, height=100)],
            text_spans=[TextSpan("span_0001", 1, "设计注意：保留", "pdf_text")],
            text_layer_available=True,
            warnings=[],
        )


class FakeLlmAgent:
    def generate_schema(self, spans):
        return {
            "sections": [],
            "entity_types": [{"entity_type": "requirement", "repeatable": True}],
            "field_definitions": [
                {
                    "semantic_key": "custom.design_note",
                    "display_name": "设计注意",
                    "field_type": "requirement",
                    "criticality": "non_critical",
                    "repeatable": True,
                    "source_span_ids": ["span_0001"],
                }
            ],
            "table_definitions": [],
            "requirement_definitions": [],
        }

    def generate_extraction_plan(self, schema, spans):
        return {
            "fields": [
                {
                    "semantic_key": "custom.design_note",
                    "display_name": "设计注意",
                    "field_type": "requirement",
                    "span_id": "span_0001",
                    "start_offset": 0,
                    "end_offset": len("设计注意：保留"),
                    "text": "设计注意：保留",
                    "confidence": 0.92,
                    "entity_id": None,
                    "section_id": "sec_label_text",
                }
            ],
            "entities": [],
            "tables": [],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
        }


class BlockRecordingLlmAgent:
    def __init__(self) -> None:
        self.schema_calls: list[list[str]] = []
        self.extraction_calls: list[list[str]] = []
        self.review_calls = 0
        self.reconciliation_calls = 0

    def generate_schema(self, spans, vdg_context=None):
        self.schema_calls.append([span.span_id for span in spans])
        first = spans[0]
        return {
            "sections": [],
            "entity_types": [],
            "field_definitions": [{"semantic_key": "custom.other_label_text", "display_name": "标签文字", "field_type": "string", "criticality": "non_critical", "repeatable": True, "source_span_ids": [first.span_id]}],
            "table_definitions": [],
            "requirement_definitions": [],
        }

    def generate_extraction_plan(self, schema, spans, vdg_context=None):
        self.extraction_calls.append([span.span_id for span in spans])
        first = spans[0]
        return {
            "fields": [{"semantic_key": "custom.other_label_text", "display_name": "标签文字", "field_type": "string", "span_id": first.span_id, "start_offset": 0, "end_offset": len(first.text), "text": first.text, "confidence": 0.96, "entity_id": None, "section_id": "sec_label_text"}],
            "entities": [], "tables": [], "requirements": [], "ignored_nodes": [], "unknown_nodes": [], "layout_candidate_decisions": [],
        }

    def generate_field_extraction_plan(self, schema, spans, vdg_context=None):
        return self.generate_extraction_plan(schema, spans, vdg_context)

    def review_compiled_blocks(self, review_input):
        self.review_calls += 1
        return {"findings": []}

    def reconcile_extraction_plan(self, schema, spans, proposals, vdg_context=None):
        self.reconciliation_calls += 1
        return proposals

class SchemaSourceRefGateTests(unittest.TestCase):
    def test_schema_consumption_findings_drive_late_spans_into_global_repair(self) -> None:
        spans = [TextSpan(f"atom_{index:04d}", 1, f"文字{index}", "pdf_char_atom") for index in range(260)]
        schema = GeneratedSchema(
            "schema",
            True,
            "v1",
            [],
            [],
            [FieldDefinition("fdef_1", "manufacturer.name", "生产者", "string", "critical", source_span_ids=["atom_0259"])],
        )
        plan = ExtractionPlan("plan", "schema", [])
        blocks = {
            "blocks": [
                {
                    "block_id": "block_late",
                    "source_span_ids": ["atom_0259"],
                    "context_span_ids": ["atom_0258", "atom_0259"],
                }
            ]
        }

        findings = _schema_consumption_findings(schema, plan, blocks)
        repair_spans = _semantic_repair_spans(findings, blocks, spans)

        self.assertEqual(findings[0]["source_span_ids"], ["atom_0259"])
        self.assertIn("atom_0259", {span.span_id for span in repair_spans})
        self.assertEqual(
            {span.span_id for span in repair_spans},
            {"atom_0258", "atom_0259"},
        )

    def test_source_fusion_uses_geometry_for_repeated_identical_text(self) -> None:
        pdf_spans = [
            TextSpan("first", 1, "地址", "pdf_char_atom", bbox_pdf=BBoxPdf(10, 10, 20, 10, 100, 100)),
            TextSpan("second", 1, "地址", "pdf_char_atom", bbox_pdf=BBoxPdf(10, 60, 20, 10, 100, 100)),
        ]
        ocr_lines = [OcrLine("ocr_1", 1, "地址", 0.99, bbox_pdf=BBoxPdf(10, 60, 20, 10, 100, 100))]

        result = build_source_fusion(pdf_spans, ocr_lines, enabled=True)

        self.assertEqual(result.alignments[0].pdf_span_ids, ["second"])

    def test_source_fusion_replaces_pdf_atom_confirmed_as_two_ocr_columns(self) -> None:
        pdf_spans = [
            TextSpan(
                "adhered",
                1,
                "配料内容贮存内容",
                "pdf_char_atom",
                bbox_pdf=BBoxPdf(10, 20, 80, 10, 100, 100),
            )
        ]
        ocr_lines = [
            OcrLine("ocr_left", 1, "配料内容", 0.99, bbox_pdf=BBoxPdf(10, 20, 35, 10, 100, 100)),
            OcrLine("ocr_right", 1, "贮存内容", 0.99, bbox_pdf=BBoxPdf(55, 20, 35, 10, 100, 100)),
        ]

        result = build_source_fusion(pdf_spans, ocr_lines, enabled=True)

        self.assertEqual([span.text for span in result.canonical_spans], ["配料内容", "贮存内容"])
        self.assertEqual(result.report["superseded_adhesion_span_count"], 1)
        self.assertEqual(
            _source_fusion_validation_checks(result.report, "char_atoms_high_recall")[0]["result"],
            "passed",
        )

    def test_merge_spans_keeps_overlapping_ocr_as_an_alternate_reading(self) -> None:
        bbox = BBoxPdf(10, 10, 60, 10, 100, 100)
        pdf_spans = [TextSpan("atom_0001", 1, "产品名称：测试", "pdf_char_atom", bbox_pdf=bbox)]
        ocr_lines = [OcrLine("ocr_0001", 1, "产品名称:测试", 0.98, bbox_pdf=bbox)]

        spans = _merge_spans(pdf_spans, ocr_lines)

        self.assertEqual([span.span_id for span in spans], ["atom_0001"])

    def test_merge_spans_keeps_unmatched_ocr_as_canonical_evidence(self) -> None:
        pdf_spans = [TextSpan("atom_0001", 1, "产品名称：测试", "pdf_char_atom", bbox_pdf=BBoxPdf(10, 10, 60, 10, 100, 100))]
        ocr_lines = [OcrLine("ocr_0001", 1, "净含量：100克", 0.98, bbox_pdf=BBoxPdf(10, 40, 60, 10, 100, 100))]

        spans = _merge_spans(pdf_spans, ocr_lines)

        self.assertEqual([span.span_id for span in spans], ["atom_0001", "ocr_span_0001"])

    def test_table_cell_text_supports_ordered_ranges(self) -> None:
        spans = {
            "amount": TextSpan("amount", 1, "2.8", "pdf_char_atom"),
            "unit": TextSpan("unit", 1, "克", "pdf_char_atom"),
        }
        cell = {
            "ranges": [
                {"span_id": "amount", "start_offset": 0, "end_offset": 3, "text": "2.8"},
                {"span_id": "unit", "start_offset": 0, "end_offset": 1, "text": "克"},
            ]
        }

        self.assertEqual(_agent_cell_text(cell, spans), "2.8克")

    def test_nutrition_marker_merges_only_when_geometry_confirms_same_row(self) -> None:
        spans = {
            "marker": TextSpan("marker", 1, "--", "pdf_char_atom", bbox_pdf=BBoxPdf(10, 10, 8, 10, 100, 100)),
            "label": TextSpan("label", 1, "饱和脂肪", "pdf_char_atom", bbox_pdf=BBoxPdf(20, 10.2, 30, 10, 100, 100)),
            "other_row": TextSpan("other_row", 1, "糖", "pdf_char_atom", bbox_pdf=BBoxPdf(20, 30, 15, 10, 100, 100)),
        }
        same_row = [
            {"text": "--", "source_span_ids": ["marker"]},
            {"text": "饱和脂肪", "source_span_ids": ["label"]},
            {"text": "2.2克"},
            {"text": "11%"},
        ]
        different_row = [
            {"text": "--", "source_span_ids": ["marker"]},
            {"text": "糖", "source_span_ids": ["other_row"]},
        ]

        merged = _merge_nutrition_marker_cells(same_row, spans)

        self.assertEqual(merged[0]["text"], "--饱和脂肪")
        self.assertEqual(len(merged), 3)
        self.assertEqual(_merge_nutrition_marker_cells(different_row, spans), different_row)

    def test_non_span_vdg_refs_are_removed_from_schema_containers(self) -> None:
        spans = [TextSpan("span_0001", 1, "营养成分表", "pdf_char_atom")]
        fallback = GeneratedSchema("fallback", True, "v1", [], [], [], [], [])

        schema = _schema_from_agent_body(
            {
                "sections": [{"section_id": "nutrition", "source_span_ids": ["reg_0001"]}],
                "entity_types": [],
                "field_definitions": [],
                "table_definitions": [{"table_type": "nutrition_facts", "source_span_ids": ["layout_nutrition_0001", "span_0001"]}],
                "requirement_definitions": [],
            },
            spans,
            fallback,
        )

        self.assertEqual(schema.sections[0]["source_span_ids"], [])
        self.assertEqual(schema.table_definitions[0]["source_span_ids"], ["span_0001"])

    def test_string_table_columns_are_normalized(self) -> None:
        columns = _agent_table_columns(
            {"columns": ["nutrient_item", "per_100g_value", "nrv_percent"]},
            [],
        )

        self.assertEqual(
            columns,
            [
                {"column_id": "nutrient_item", "name": "nutrient_item"},
                {"column_id": "per_100g_value", "name": "per_100g_value"},
                {"column_id": "nrv_percent", "name": "nrv_percent"},
            ],
        )

    def test_only_agent_accepted_layout_candidate_is_promoted_to_final_table(self) -> None:
        spans = [
            TextSpan(
                "atom_p1_0001",
                1,
                "能量",
                "pdf_char_atom",
                bbox_pdf=BBoxPdf(10, 10, 20, 10, 100, 100),
                bbox_normalized=BBoxNormalized(0.1, 0.1, 0.3, 0.2),
            ),
            TextSpan(
                "atom_p1_0002",
                1,
                "100千焦",
                "pdf_char_atom",
                bbox_pdf=BBoxPdf(40, 10, 30, 10, 100, 100),
                bbox_normalized=BBoxNormalized(0.4, 0.1, 0.7, 0.2),
            ),
        ]
        candidates = {
            "table_candidates": [
                {
                    "table_candidate_id": "layout_nutrition_0001",
                    "table_type": "nutrition_facts",
                    "page": 1,
                    "title": "营养成分表",
                    "source_span_ids": ["atom_p1_0001", "atom_p1_0002"],
                    "rows": [
                        {
                            "row_key": "能量",
                            "source_span_ids": ["atom_p1_0001", "atom_p1_0002"],
                            "cells": [
                                {"text": "能量", "source_span_ids": ["atom_p1_0001"]},
                                {"text": "100千焦", "source_span_ids": ["atom_p1_0002"]},
                            ],
                        }
                    ],
                }
            ]
        }

        rejected_tables, rejected_evidence = _extract_tables_from_accepted_layout_candidates(
            candidates,
            {"decisions": [{"layout_candidate_id": "layout_nutrition_0001", "decision": "unresolved"}]},
            spans,
            {},
            [],
            0,
        )
        accepted_tables, accepted_evidence = _extract_tables_from_accepted_layout_candidates(
            candidates,
            {"decisions": [{"layout_candidate_id": "layout_nutrition_0001", "decision": "accept"}]},
            spans,
            {},
            [],
            0,
        )

        self.assertEqual((rejected_tables, rejected_evidence), ([], []))
        self.assertEqual(len(accepted_tables), 1)
        self.assertEqual(len(accepted_evidence), 1)
        self.assertEqual(accepted_tables[0]["source"], "llm_accepted_layout_candidate")
        self.assertTrue(accepted_tables[0]["review_required"])
        self.assertEqual(len(accepted_tables[0]["rows"][0]["cells"]), len(accepted_tables[0]["columns"]))

class ScopeAwareLlmAgent(FakeLlmAgent):
    def __init__(self) -> None:
        self.schema_vdg_context = None
        self.extraction_vdg_context = None

    def generate_schema(self, spans, vdg_context=None):
        self.schema_vdg_context = vdg_context
        return super().generate_schema(spans)

    def generate_extraction_plan(self, schema, spans, vdg_context=None):
        self.extraction_vdg_context = vdg_context
        return super().generate_extraction_plan(schema, spans)


class SparseSectionLlmAgent(FakeLlmAgent):
    def generate_schema(self, spans):
        schema = super().generate_schema(spans)
        schema["sections"] = [{"section_id": "doc_header"}, {"section_id": "label_info"}]
        return schema


class LowConfidenceLlmAgent:
    def generate_schema(self, spans):
        return FakeLlmAgent().generate_schema(spans)

    def generate_extraction_plan(self, schema, spans):
        return {
            "fields": [
                {
                    "semantic_key": "custom.design_note",
                    "display_name": "设计注意",
                    "field_type": "requirement",
                    "span_id": "span_0001",
                    "start_offset": 0,
                    "end_offset": len("设计注意：保留"),
                    "text": "设计注意：保留",
                    "confidence": 0.70,
                    "entity_id": None,
                    "section_id": "sec_label_text",
                }
            ],
            "entities": [],
            "tables": [],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
        }


class FieldRetryLlmAgent:
    def generate_schema(self, spans):
        return FakeLlmAgent().generate_schema(spans)

    def generate_extraction_plan(self, schema, spans):
        return {
            "fields": [],
            "entities": [],
            "tables": [],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
        }

    def generate_field_extraction_plan(self, schema, spans):
        return FakeLlmAgent().generate_extraction_plan(schema, spans)


class RepairPdfReader:
    def read(self, path):
        return PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100)],
            text_spans=[
                TextSpan(
                    "span_0001",
                    1,
                    "配料：水 产品类型：饮料",
                    "pdf_text",
                    bbox_pdf=BBoxPdf(0, 0, 160, 10, 200, 100),
                )
            ],
            text_layer_available=True,
            warnings=[],
        )


class NutritionTablePdfReader:
    def read(self, path):
        return PdfPerception(
            pages=[PageInfo(page=1, width=200, height=100)],
            text_spans=[
                TextSpan(
                    "span_0001",
                    1,
                    "营养成分表",
                    "pdf_text",
                    bbox_pdf=BBoxPdf(0, 0, 80, 10, 200, 100),
                    bbox_normalized=BBoxNormalized(0, 0, 0.4, 0.1),
                ),
                TextSpan(
                    "span_0002",
                    1,
                    "能量 100kJ 1%",
                    "pdf_text",
                    bbox_pdf=BBoxPdf(0, 12, 120, 10, 200, 100),
                    bbox_normalized=BBoxNormalized(0, 0.12, 0.6, 0.22),
                ),
            ],
            text_layer_available=True,
            warnings=[],
        )


class NutritionTableLlmAgent:
    def generate_schema(self, spans):
        return {
            "sections": [],
            "entity_types": [{"entity_type": "product", "repeatable": False}],
            "field_definitions": [],
            "table_definitions": [
                {
                    "table_type": "nutrition_facts",
                    "display_name": "营养成分表",
                    "criticality": "critical",
                    "repeatable": True,
                    "source_span_ids": ["span_0001"],
                }
            ],
            "requirement_definitions": [],
        }

    def generate_extraction_plan(self, schema, spans):
        return {
            "fields": [],
            "entities": [],
            "tables": [
                {
                    "table_type": "nutrition_facts",
                    "title": "营养成分表",
                    "source_span_ids": ["span_0001", "span_0002"],
                    "confidence": 0.96,
                    "rows": [
                        {
                            "row_key": "能量",
                            "source_span_ids": ["span_0002"],
                            "cells": [
                                {"column_id": "col_001", "text": "能量", "span_id": "span_0002"},
                                {"column_id": "col_002", "text": "100kJ", "span_id": "span_0002"},
                                {"column_id": "col_003", "text": "1%", "span_id": "span_0002"},
                            ],
                        }
                    ],
                }
            ],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
        }


class SemanticNutritionColumnLlmAgent(NutritionTableLlmAgent):
    def generate_extraction_plan(self, schema, spans):
        plan = super().generate_extraction_plan(schema, spans)
        plan["tables"][0]["rows"][0]["cells"] = [
            {"column_id": "nutrition_item", "text": "能量", "span_id": "span_0002"},
            {"column_id": "per_serving", "text": "100kJ", "span_id": "span_0002"},
            {"column_id": "nrv_percent", "text": "1%", "span_id": "span_0002"},
        ]
        return plan


class SourceSpanByTableLlmAgent(NutritionTableLlmAgent):
    def generate_schema(self, spans):
        return {
            "sections": [],
            "entity_types": [{"entity_type": "product", "repeatable": False}],
            "field_definitions": [],
            "table_definitions": [
                {
                    "table_type": "nutrition_facts",
                    "display_name": "营养成分表",
                    "criticality": "critical",
                    "repeatable": True,
                    "source_span_ids_by_table": {"营养成分表": ["span_0001", "span_0002"]},
                }
            ],
            "requirement_definitions": [],
        }


class SparseTableDefinitionLlmAgent(NutritionTableLlmAgent):
    def generate_schema(self, spans):
        return {
            "sections": [],
            "entity_types": [{"entity_type": "product", "repeatable": False}],
            "field_definitions": [],
            "table_definitions": [
                {
                    "table_type": "nutrition_facts",
                    "display_name": "营养成分表",
                    "criticality": "critical",
                    "repeatable": True,
                }
            ],
            "requirement_definitions": [],
        }


class TableRetryLlmAgent(NutritionTableLlmAgent):
    def generate_extraction_plan(self, schema, spans):
        return {
            "fields": [],
            "entities": [],
            "tables": [],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
        }

    def generate_table_extraction_plan(self, schema, spans):
        return super().generate_extraction_plan(schema, spans)


class RowlessTableRetryLlmAgent(NutritionTableLlmAgent):
    def generate_extraction_plan(self, schema, spans):
        return {
            "fields": [],
            "entities": [],
            "tables": [{"table_type": "nutrition_facts", "source_span_ids": ["span_0001", "span_0002"]}],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
        }

    def generate_table_extraction_plan(self, schema, spans):
        return super().generate_extraction_plan(schema, spans)


class AdhesionLlmAgent:
    def generate_schema(self, spans):
        return {
            "sections": [],
            "entity_types": [{"entity_type": "product", "repeatable": False}],
            "field_definitions": [
                {
                    "semantic_key": "product.ingredients",
                    "display_name": "配料",
                    "field_type": "long_text",
                    "criticality": "critical",
                    "repeatable": False,
                    "source_span_ids": ["span_0001"],
                }
            ],
            "table_definitions": [],
            "requirement_definitions": [],
        }

    def generate_extraction_plan(self, schema, spans):
        return {
            "fields": [
                {
                    "semantic_key": "product.ingredients",
                    "display_name": "配料",
                    "field_type": "long_text",
                    "span_id": "span_0001",
                    "start_offset": 0,
                    "end_offset": len(spans[0].text),
                    "text": spans[0].text,
                    "confidence": 0.95,
                    "entity_id": "product_001",
                    "section_id": "sec_label_text",
                }
            ],
            "entities": [{"entity_id": "product_001", "entity_type": "product"}],
            "tables": [],
            "requirements": [],
            "ignored_nodes": [],
            "unknown_nodes": [],
        }


class PipelineLlmTests(unittest.TestCase):
    def test_enhanced_pipeline_routes_rule_only_fields_to_review(self) -> None:
        atoms = [TextSpan("atom_0001", 1, "产品名称：测试软糖", "pdf_char_atom", bbox_pdf=BBoxPdf(10, 10, 70, 10, 100, 100))]
        evidence = LayoutEvidence(
            canonical_pdf_spans=atoms,
            character_atoms=atoms,
            candidate_table_layers={"parsers": [], "tables": [], "parser_issues": [], "candidate_only": True},
            layout_candidates={"source_nodes": [], "table_candidates": [], "reading_order_candidates": [], "side_marker_candidates": [], "quality_issues": [], "cross_page_candidate_count": 0},
            layout_quality_report={"status": "pass", "mode": "char_atoms_high_recall", "pdf_character_atom_count": 1},
            mode="char_atoms_high_recall",
            fallback_used=False,
            failure_reason=None,
        )
        original = pipeline_module.build_layout_evidence
        pipeline_module.build_layout_evidence = lambda *_args, **_kwargs: evidence
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf = Path(temp_dir) / "input.pdf"
                pdf.write_bytes(b"%PDF-1.4\n")
                result = DocumentParser(
                    ocr_client=EmptyOcrClient(),
                    llm_agent=BlockRecordingLlmAgent(),
                    pdf_reader=FakePdfReader(),
                ).parse(
                    pdf,
                    use_llm_agent=True,
                    runtime_policy={"effective_options": {"layout_mode": "char_atoms_high_recall"}},
                )
        finally:
            pipeline_module.build_layout_evidence = original

        self.assertEqual(result.metadata["agent_harness"]["rule_fallback_field_count"], 0)
        self.assertTrue(any(item.get("source") == "rule_validation_candidate" for item in result.metadata["agent_harness"]["review_items"]))
        self.assertFalse(any(field["semantic_key"] == "product.name" for field in result.extracted_data["fields"].values()))

    def test_enhanced_pipeline_calls_agent_per_block_and_covers_late_atoms(self) -> None:
        atoms = [TextSpan(f"atom_{index:04d}", 1, f"标签文字{index}", "pdf_char_atom") for index in range(170)]
        evidence = LayoutEvidence(
            canonical_pdf_spans=atoms,
            character_atoms=atoms,
            candidate_table_layers={"parsers": [], "tables": [], "parser_issues": [], "candidate_only": True},
            layout_candidates={"source_nodes": [], "table_candidates": [], "reading_order_candidates": [], "side_marker_candidates": [], "quality_issues": [], "cross_page_candidate_count": 0},
            layout_quality_report={"status": "pass", "mode": "char_atoms_high_recall", "pdf_character_atom_count": 170},
            mode="char_atoms_high_recall",
            fallback_used=False,
            failure_reason=None,
        )
        original = pipeline_module.build_layout_evidence
        pipeline_module.build_layout_evidence = lambda *_args, **_kwargs: evidence
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf = Path(temp_dir) / "input.pdf"
                pdf.write_bytes(b"%PDF-1.4\n")
                agent = BlockRecordingLlmAgent()
                result = DocumentParser(ocr_client=EmptyOcrClient(), llm_agent=agent, pdf_reader=FakePdfReader()).parse(
                    pdf,
                    use_llm_agent=True,
                    runtime_policy={"effective_options": {"layout_mode": "char_atoms_high_recall"}},
                )
        finally:
            pipeline_module.build_layout_evidence = original

        self.assertEqual(result.metadata["agent_blocks"]["source_span_coverage_rate"], 1.0)
        self.assertEqual(len(agent.schema_calls), 1)
        self.assertIn("atom_0169", {span_id for call in agent.extraction_calls for span_id in call})
        self.assertEqual(agent.review_calls, 1)
        self.assertEqual(agent.reconciliation_calls, 1)
        self.assertEqual(result.metadata["global_reconciliation_report"]["status"], "applied")
        self.assertEqual(result.metadata["semantic_review_report"]["status"], "pass")
        self.assertTrue(result.metadata["semantic_review_report"]["review_agent_independent"])

    def test_pipeline_compiles_llm_agent_plan_after_span_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=FakeLlmAgent(),
                pdf_reader=FakePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        fields = result.extracted_data["fields"]
        self.assertTrue(any(field["semantic_key"] == "custom.design_note" for field in fields.values()))
        self.assertEqual(result.metadata["agent_harness"]["llm_agent_candidate_count"], 1)
        self.assertEqual(result.metadata["agent_harness"]["accepted_agent_item_count"], 1)
        report = result.metadata["agent_execution_report"]
        extraction_agent = next(agent for agent in report["agents"] if agent["agent_id"] == "extraction_agent")
        self.assertEqual(extraction_agent["mode"], "agent_led_extraction_with_rule_validation_fallback")
        self.assertIn("group_agent", {agent["agent_id"] for agent in report["agents"]})
        self.assertIn("table_list_agent", {agent["agent_id"] for agent in report["agents"]})

    def test_pipeline_injects_label_text_scope_context_into_vdg_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            llm_agent = ScopeAwareLlmAgent()
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=llm_agent,
                pdf_reader=FakePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        self.assertIn("label_text_scope", llm_agent.schema_vdg_context)
        self.assertIn("label_text_scope", llm_agent.extraction_vdg_context)
        self.assertTrue(llm_agent.schema_vdg_context["label_text_scope"]["reference_is_not_evidence"])
        self.assertEqual(result.metadata["label_text_scope_reference"]["reference_version"], "label_text_scope_reference_v0.1")
        self.assertIn("label_text_scope_report", result.metadata)

    def test_pipeline_normalizes_sparse_llm_schema_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=SparseSectionLlmAgent(),
                pdf_reader=FakePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        sections = result.generated_schema.sections
        sparse_sections = [section for section in sections if section.get("section_id") in {"doc_header", "label_info"}]
        self.assertEqual(len(sparse_sections), 2)
        self.assertTrue(all(section.get("section_type") and section.get("display_name") for section in sparse_sections))
        self.assertEqual(result.metadata["output_contract_validation_report"]["status"], "pass")

    def test_pipeline_retries_field_plan_when_first_agent_plan_has_no_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=FieldRetryLlmAgent(),
                pdf_reader=FakePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        fields = result.extracted_data["fields"]
        harness = result.metadata["agent_harness"]

        self.assertTrue(any(field["semantic_key"] == "custom.design_note" for field in fields.values()))
        self.assertTrue(harness["field_retry_used"])
        self.assertEqual(harness["agent_field_retry_count"], 1)
        self.assertEqual(harness["agent_plan_field_count"], 1)

    def test_pipeline_routes_low_confidence_agent_candidate_to_repair_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=LowConfidenceLlmAgent(),
                pdf_reader=FakePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        repair_plan = result.metadata["repair_loop"]["repair_plan"]

        self.assertTrue(any(action["issue_type"] == "agent_candidate_low_confidence" for action in repair_plan["actions"]))
        self.assertTrue(any(action.get("input_artifact") == "review_items.json" for action in repair_plan["actions"]))
        self.assertEqual(result.metadata["agent_harness"]["review_items"][0]["reason"], "agent_candidate_low_confidence")

    def test_pipeline_uses_agent_table_when_parser_is_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=NutritionTableLlmAgent(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        tables = result.extracted_data["tables"]
        table_quality = result.metadata["table_parser"]["table_quality_report"]

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["source"], "llm_table_agent")
        self.assertEqual(tables[0]["rows"][0]["row_key"], "能量")
        self.assertEqual(table_quality["agent_table_acceptance"]["status"], "accepted")

    def test_pipeline_normalizes_semantic_agent_table_column_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=SemanticNutritionColumnLlmAgent(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        table = result.extracted_data["tables"][0]
        cell_ids = [cell["column_id"] for cell in table["rows"][0]["cells"]]

        self.assertEqual([column["column_id"] for column in table["columns"]], ["col_001", "col_002", "col_003"])
        self.assertEqual(cell_ids, ["col_001", "col_002", "col_003"])
        self.assertEqual(len(table["rows"][0]["cells"]), len(table["columns"]))
        self.assertEqual(result.metadata["output_contract_validation_report"]["status"], "pass")

    def test_pipeline_normalizes_agent_table_source_span_map_and_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=SourceSpanByTableLlmAgent(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        table_definition = result.generated_schema.table_definitions[0]
        table = result.extracted_data["tables"][0]

        self.assertEqual(table_definition["source_span_ids"], ["span_0001", "span_0002"])
        self.assertEqual(table["bbox_status"], "available")
        self.assertIsNotNone(table["bbox_pdf"])
        self.assertIsNotNone(table["bbox_normalized"])
        self.assertEqual(result.metadata["output_contract_validation_report"]["status"], "pass")

    def test_pipeline_fills_sparse_agent_table_definition_from_fallback_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=SparseTableDefinitionLlmAgent(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        table_definition = result.generated_schema.table_definitions[0]

        self.assertEqual(table_definition["table_type"], "nutrition_facts")
        self.assertTrue(table_definition["source_span_ids"])
        self.assertEqual(result.metadata["output_contract_validation_report"]["status"], "pass")

    def test_pipeline_retries_table_plan_when_first_agent_plan_has_no_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=TableRetryLlmAgent(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        harness = result.metadata["agent_harness"]
        report = result.metadata["agent_execution_report"]
        table_agent = next(agent for agent in report["agents"] if agent["agent_id"] == "table_list_agent")

        self.assertEqual(len(result.extracted_data["tables"]), 1)
        self.assertTrue(harness["table_retry_used"])
        self.assertEqual(harness["agent_table_retry_count"], 1)
        self.assertEqual(table_agent["output_counts"]["table_retry_plan_count"], 1)

    def test_pipeline_retries_rowless_table_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=RowlessTableRetryLlmAgent(),
                pdf_reader=NutritionTablePdfReader(),
            ).parse(pdf, use_llm_agent=True)

        self.assertTrue(result.metadata["agent_harness"]["table_retry_used"])
        self.assertEqual(len(result.extracted_data["tables"][0]["rows"]), 1)

    def test_table_retry_decision_skips_large_candidate_set(self) -> None:
        table_layers = {
            "tables": [
                {
                    "table_type": "nutrition_facts",
                    "rows": [{"cells": [{"text": "能量"}, {"text": "100kJ"}, {"text": "1%"}]} for _ in range(10)],
                }
                for _ in range(4)
            ]
        }

        decision = _table_retry_decision(table_layers)

        self.assertFalse(decision["should_retry"])
        self.assertEqual(decision["reason"], "llm_table_retry_skipped_large_candidate_set")
        self.assertEqual(decision["candidate_table_count"], 4)
        self.assertEqual(decision["candidate_cell_count"], 120)

        enhanced_decision = _table_retry_decision(table_layers, allow_large_candidate_set=True)
        self.assertTrue(enhanced_decision["should_retry"])

    def test_field_retry_spans_focuses_field_anchors(self) -> None:
        schema = GeneratedSchema(
            schema_id="schema_test",
            auto_generated=False,
            schema_version="test",
            sections=[],
            entity_types=[],
            field_definitions=[
                FieldDefinition(
                    field_def_id="fd_001",
                    semantic_key="product.ingredients",
                    display_name="配料",
                    field_type="long_text",
                    criticality="critical",
                    source_span_ids=["span_0002"],
                )
            ],
            table_definitions=[],
        )
        spans = [
            TextSpan("span_0001", 1, "能量 100kJ 1%", "pdf_text"),
            TextSpan("span_0002", 1, "配料表：糯米、水", "pdf_text"),
            TextSpan("span_0003", 1, "蛋白质 2g 3%", "pdf_text"),
            TextSpan("span_0004", 1, "保质期：120天", "pdf_text"),
        ]

        selected = _field_retry_spans(spans, schema)

        self.assertEqual([span.span_id for span in selected], ["span_0002", "span_0004"])

    def test_pipeline_repair_trace_records_recompile_after_boundary_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            result = DocumentParser(
                ocr_client=EmptyOcrClient(),
                llm_agent=AdhesionLlmAgent(),
                pdf_reader=RepairPdfReader(),
            ).parse(pdf, use_llm_agent=True)

        trace = result.metadata["repair_loop"]["trace"]
        attempts = result.metadata["repair_loop"]["attempts"]
        patches = result.metadata["repair_loop"]["repair_plan_patches"]
        audit_input = result.metadata["audit_input"]

        self.assertEqual(trace["status"], "pass")
        self.assertGreaterEqual(trace["round_count"], 2)
        self.assertEqual(trace["rounds"][0]["status"], "repaired_and_recompiled")
        self.assertTrue(trace["rounds"][0]["compiled_after_repair"])
        self.assertEqual(trace["rounds"][1]["status"], "passed_after_repair")
        self.assertEqual(attempts["status"], "attempted")
        self.assertTrue(any(attempt["status"] == "applied" for attempt in attempts["attempts"]))
        self.assertEqual(patches["status"], "applied")
        self.assertEqual(patches["patches"][0]["operation"], "adjust_field_boundary")
        self.assertTrue(audit_input["separation"]["review_runs_after_compiler"])
        self.assertIn("coverage_map.json", audit_input["input_artifacts"])

    def test_pipeline_rejects_llm_agent_mode_without_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            parser = DocumentParser(ocr_client=EmptyOcrClient(), pdf_reader=FakePdfReader())
            with self.assertRaises(ParseError):
                parser.parse(pdf, use_llm_agent=True)


if __name__ == "__main__":
    unittest.main()
