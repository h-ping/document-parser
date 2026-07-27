import json
import unittest
from pathlib import Path


class LayoutGoldFixtureTests(unittest.TestCase):
    def test_fixture_contains_only_compact_boundary_and_structure_expectations(self) -> None:
        path = Path(__file__).parent / "fixtures" / "layout_char_atoms_high_recall_gold_v0.1.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(fixture["privacy_policy"], "field_boundaries_and_structure_counts_only")
        self.assertEqual(len(fixture["cases"]["xufuji"]["required_independent_atoms"]), 4)
        self.assertEqual(fixture["cases"]["youleme"]["expected_nutrition_item_count"], 8)
        self.assertEqual(fixture["cases"]["zongzi"]["expected_nutrition_candidate_count"], 9)
        self.assertNotIn("地址：", json.dumps(fixture, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
