import runpy
import unittest
from pathlib import Path


REPORT = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "build_layout_gold_comparison.py"))


class LayoutGoldComparisonTests(unittest.TestCase):
    def test_nutrition_recall_accepts_project_specific_item_column(self) -> None:
        gold_case = {
            "nutrition_rows": [
                {"营养表编号": "N1", "项目": "能量"},
                {"营养表编号": "N1", "项目": "蛋白质"},
            ]
        }
        tables = [{"rows": [{"text": "能量 1048kJ"}, {"text": "蛋⽩质 2.8g"}]}]

        recall = REPORT["_nutrition_row_recall"](gold_case, tables, "youleme_nutrition")

        self.assertEqual(recall["matched"], 2)
        self.assertEqual(recall["expected"], 2)


if __name__ == "__main__":
    unittest.main()
