import unittest

from document_parser.comparison import COMPARISON_DIMENSIONS, build_comparison_index


class ComparisonIndexTests(unittest.TestCase):
    def test_builds_entries_for_comparison_required_standard_items(self) -> None:
        item = {
            "id": "std_0001",
            "field_id": "fld_0001",
            "field": "product_name",
            "semantic_key": "product.name",
            "label": "品名",
            "text": "品名：牛奶",
            "normalized_text": "牛奶",
            "value_hash": "sha256:" + "0" * 64,
            "source": {"section": "sec_label_text", "bbox_normalized": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}},
            "group_id": "product_001",
            "table_id": None,
            "row_key": None,
            "evidence_refs": ["ev_0001"],
            "comparison_required": True,
            "comparison_profile": {
                "semantic_key": "product.name",
                "normalized_value": "牛奶",
                "value_hash": "sha256:" + "0" * 64,
                "section_id": "sec_label_text",
                "entity_id": "product_001",
                "table_id": None,
                "row_key": None,
                "bbox_normalized": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
                "evidence_refs": ["ev_0001"],
            },
        }

        index = build_comparison_index([item, {**item, "id": "std_0002", "comparison_required": False}])

        self.assertEqual(index["dimension_contract"], COMPARISON_DIMENSIONS)
        self.assertEqual(index["entry_count"], 1)
        self.assertEqual(index["skipped_count"], 1)
        self.assertEqual(index["entries"][0]["standard_item_id"], "std_0001")
        self.assertEqual(index["entries"][0]["matching_dimensions"]["value_hash"], "sha256:" + "0" * 64)
        self.assertIn("product.name", index["entries"][0]["comparison_key"])


if __name__ == "__main__":
    unittest.main()
