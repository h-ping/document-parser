import json
import tempfile
import unittest
from pathlib import Path

from document_parser.benchmark import (
    BenchmarkThresholds,
    evaluate_benchmark,
    load_benchmark_manifest,
    normalize_benchmark_value,
)


class BenchmarkTests(unittest.TestCase):
    def test_normalization_preserves_material_symbols_and_strips_known_label(self) -> None:
        self.assertEqual(normalize_benchmark_value("配料：白砂糖 ≥ 2.5%", "product.ingredients"), "白砂糖≥2.5%")
        self.assertEqual(
            normalize_benchmark_value("食品生产许可证编\n号：SC123", "manufacturer.license_number"),
            "SC123",
        )
        self.assertEqual(normalize_benchmark_value("食品生产许可证编号SC123", "manufacturer.license_number"), "SC123")
        self.assertEqual(normalize_benchmark_value("商品条码6926475208328", "barcode.commodity"), "6926475208328")
        self.assertNotEqual(
            normalize_benchmark_value("白砂糖≥2.5%", "product.ingredients"),
            normalize_benchmark_value("白砂糖≥25%", "product.ingredients"),
        )

    def test_manifest_maps_pdf_patterns_to_external_xlsx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.xlsx"
            reference.touch()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "benchmark_manifest_v0.1",
                        "cases": [
                            {
                                "case_id": "sample",
                                "pdf_pattern": "sample*.pdf",
                                "reference_xlsx": str(reference),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_benchmark_manifest(manifest_path)

        self.assertEqual(manifest.cases[0].case_id, "sample")
        self.assertEqual(manifest.cases[0].reference_xlsx, reference)

    def test_evaluation_counts_aliases_repeated_entities_and_table_boundaries(self) -> None:
        reference = _payload(
            fields=[
                _field("e1", "product.date_marking", "生产日期：见顶部", "product_001"),
                _field("e2", "manufacturer.name", "受托方：甲公司", "manufacturer_001"),
                _field("e3", "manufacturer.name", "受托方：乙公司", "manufacturer_002"),
            ],
            entities=[
                _entity("product_001", "product"),
                _entity("manufacturer_001", "manufacturer"),
                _entity("manufacturer_002", "manufacturer"),
            ],
            tables=[_nutrition_table("营养成分表", marker_in_own_cell=False)],
        )
        current = _payload(
            fields=[
                _field("c1", "product.production_date_mark", "生产日期：见顶部", "product_001"),
                _field("c2", "manufacturer.name", "受托方：甲公司", "manufacturer_001"),
            ],
            entities=[_entity("product_001", "product"), _entity("manufacturer_001", "manufacturer")],
            tables=[_nutrition_table("营养成分表", marker_in_own_cell=True)],
        )

        evaluation, diff = evaluate_benchmark(
            case_id="sample",
            current=current,
            reference=reference,
            thresholds=BenchmarkThresholds.disabled(),
        )

        self.assertEqual(evaluation["field_exact_count"], 2)
        self.assertEqual(evaluation["field_expected_count"], 3)
        self.assertEqual(evaluation["repeated_entity_group_recall"], 0.5)
        self.assertEqual(evaluation["nutrition_table_recall"], 1.0)
        self.assertEqual(evaluation["nutrition_row_value_accuracy"], 1.0)
        self.assertEqual(evaluation["nutrition_cell_boundary_conformance"], 0.0)
        self.assertEqual(len(diff["missing_fields"]), 1)

    def test_repeated_entity_group_requires_detail_fields_to_match(self) -> None:
        reference = _payload(
            fields=[
                _field("e1", "manufacturer.name", "受托方：甲公司", "m1"),
                _field("e2", "manufacturer.address", "地址：深圳", "m1"),
            ],
            entities=[_entity("m1", "manufacturer")],
            tables=[],
        )
        current = _payload(
            fields=[
                _field("c1", "manufacturer.name", "受托方：甲公司", "m2"),
                _field("c2", "manufacturer.address", "地址：广州", "m2"),
            ],
            entities=[_entity("m2", "manufacturer")],
            tables=[],
        )

        evaluation, diff = evaluate_benchmark(case_id="sample", current=current, reference=reference, thresholds=BenchmarkThresholds.disabled())

        self.assertEqual(evaluation["repeated_entity_group_recall"], 0.0)
        self.assertEqual(diff["entity_groups"][0]["status"], "detail_mismatch")

    def test_manufacturer_factory_code_can_be_embedded_in_reference_name(self) -> None:
        reference = _payload(
            fields=[_field("e1", "manufacturer.name", "1：甲公司（工厂代码：阳）", "m1")],
            entities=[_entity("m1", "manufacturer")],
            tables=[],
        )
        current = _payload(
            fields=[
                _field("c1", "manufacturer.name", "甲公司", "m2"),
                _field("c2", "manufacturer.factory_code", "阳", "m2"),
            ],
            entities=[_entity("m2", "manufacturer")],
            tables=[],
        )

        evaluation, diff = evaluate_benchmark(case_id="sample", current=current, reference=reference, thresholds=BenchmarkThresholds.disabled())

        self.assertEqual(evaluation["repeated_entity_group_recall"], 1.0)
        self.assertEqual(diff["entity_groups"][0]["status"], "exact")

    def test_global_custom_field_without_entity_matches_product_reference(self) -> None:
        reference = _payload(
            fields=[_field("e1", "custom.allergen_notice", "致敏物质提示：含乳制品", "product_001")],
            entities=[_entity("product_001", "product")],
            tables=[],
        )
        current_field = _field("c1", "custom.allergen_notice", "致敏物质提示：含乳制品", "")
        current_field["entity_id"] = None
        current = _payload(fields=[current_field], entities=[], tables=[])

        evaluation, _ = evaluate_benchmark(case_id="sample", current=current, reference=reference, thresholds=BenchmarkThresholds.disabled())

        self.assertEqual(evaluation["field_exact_recall"], 1.0)

    def test_title_only_nutrition_table_does_not_count_as_recalled(self) -> None:
        reference = _payload(fields=[], entities=[], tables=[_nutrition_table("营养成分表", marker_in_own_cell=False)])
        current = _payload(
            fields=[],
            entities=[],
            tables=[{"table_id": "current", "table_type": "nutrition_facts", "title": "营养成分表", "columns": [], "rows": []}],
        )

        evaluation, _ = evaluate_benchmark(case_id="sample", current=current, reference=reference, thresholds=BenchmarkThresholds.disabled())

        self.assertEqual(evaluation["nutrition_table_recall"], 0.0)


def _payload(*, fields: list[dict], entities: list[dict], tables: list[dict]) -> dict:
    return {
        "extracted_data": {
            "fields": {field["field_id"]: field for field in fields},
            "entities": {entity["entity_id"]: entity for entity in entities},
            "tables": tables,
        },
        "evidence": {},
    }


def _field(field_id: str, semantic_key: str, value: str, entity_id: str) -> dict:
    return {
        "field_id": field_id,
        "semantic_key": semantic_key,
        "display_name": semantic_key,
        "clean_value": value,
        "normalized_value": value,
        "criticality": "critical",
        "entity_id": entity_id,
        "evidence_refs": [],
    }


def _entity(entity_id: str, entity_type: str) -> dict:
    return {"entity_id": entity_id, "entity_type": entity_type, "fields": {}}


def _nutrition_table(title: str, *, marker_in_own_cell: bool) -> dict:
    label_cells = ["--", "饱和脂肪"] if marker_in_own_cell else ["--饱和脂肪"]
    return {
        "table_id": title,
        "table_type": "nutrition_facts",
        "title": title,
        "columns": [{"name": "项目"}, {"name": "每100克"}, {"name": "NRV%"}],
        "rows": [
            {
                "row_key": "saturated_fat",
                "cells": [
                    *[{"raw_value": value} for value in label_cells],
                    {"raw_value": "2.2克"},
                    {"raw_value": "11%"},
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
