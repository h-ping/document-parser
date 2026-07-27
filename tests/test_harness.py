import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from document_parser.harness import _field_diff, _list_diff, _table_diff


ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def test_field_diff_reports_boundary_f1_and_label_accuracy(self) -> None:
        diff = _field_diff(
            [
                {
                    "semantic_key": "product.name",
                    "text": "品名：牛奶",
                    "label": "品名",
                    "char_start": 0,
                    "char_end": 5,
                }
            ],
            [
                {
                    "id": "std_0001",
                    "semantic_key": "product.name",
                    "field": "product_name",
                    "label": "品名",
                    "text": "品名：牛奶",
                    "source": {"char_start": 0, "char_end": 5},
                    "evidence_refs": ["ev_0001"],
                }
            ],
        )

        self.assertEqual(diff["boundary_char_f1"], 1.0)
        self.assertEqual(diff["label_accuracy"], 1.0)

    def test_table_and_list_diff_report_count_checks(self) -> None:
        table_diff = _table_diff(
            [{"table_type": "nutrition_facts", "min_rows": 2}],
            [{"table_type": "nutrition_facts", "rows": [{"row_id": "row_0001"}]}],
        )
        list_diff = _list_diff(
            [{"list_type": "content_items", "item_count": 2}],
            [{"list_type": "content_items", "items": [{"index": 1}]}],
        )

        self.assertEqual(table_diff["row_count_checks"][0]["result"], "failed")
        self.assertEqual(list_diff["item_count_checks"][0]["result"], "failed")

    def test_run_agent_harness_outputs_score_and_diffs(self) -> None:
        sample_pdf = ROOT / "test-documents" / "识别文字.pdf"
        if not sample_pdf.exists():
            self.skipTest("sample PDF not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            case_dir = temp_path / "case"
            output_dir = temp_path / "out"
            case_dir.mkdir()
            (case_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_pdf": str(sample_pdf),
                        "ocr_fixture_path": str(ROOT / "tests" / "fixtures" / "ppocrv6_empty.json"),
                        "quality_gate": "strict",
                        "repair_mode": "plan_only",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (case_dir / "expected.json").write_text(
                json.dumps(
                    {
                        "fields": [
                            {
                                "semantic_key": "product.ingredients",
                                "text": "配料：牛奶",
                                "label": "配料",
                            }
                        ],
                        "tables": [
                            {
                                "table_type": "nutrition_facts",
                                "min_rows": 1,
                            }
                        ],
                        "lists": [
                            {
                                "list_type": "content_items",
                                "min_items": 1,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_agent_harness.py"),
                    "--case-dir",
                    str(case_dir),
                    "--rule-only",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "harness_summary.json").exists())
            self.assertTrue((output_dir / "field_diff.json").exists())
            self.assertTrue((output_dir / "list_diff.json").exists())
            self.assertTrue((output_dir / "score_report.md").exists())
            self.assertTrue((output_dir / "accepted_items.json").exists())
            self.assertTrue((output_dir / "comparison_index.json").exists())
            self.assertTrue((output_dir / "auto_ingest_candidates.json").exists())
            self.assertTrue((output_dir / "rejected_agent_items.json").exists())
            self.assertTrue((output_dir / "review_items.json").exists())
            self.assertTrue((output_dir / "extracted_data.json").exists())
            self.assertTrue((output_dir / "revision_blocks.json").exists())
            self.assertTrue((output_dir / "evidence.json").exists())
            self.assertTrue((output_dir / "validation.json").exists())
            self.assertTrue((output_dir / "risks.json").exists())
            self.assertTrue((output_dir / "review_tasks.json").exists())
            self.assertTrue((output_dir / "runtime_policy.json").exists())
            self.assertTrue((output_dir / "audit_input.json").exists())
            self.assertTrue((output_dir / "agent_execution_report.json").exists())
            self.assertTrue((output_dir / "missing_item_report.json").exists())
            self.assertTrue((output_dir / "missing_fields.json").exists())
            self.assertTrue((output_dir / "missing_tables.json").exists())
            self.assertTrue((output_dir / "output_contract_validation_report.json").exists())
            self.assertTrue((output_dir / "mvp_acceptance_metrics.json").exists())
            self.assertTrue((output_dir / "visual_document_graph.json").exists())
            self.assertTrue((output_dir / "schema_audit.json").exists())
            self.assertTrue((output_dir / "structure_audit.json").exists())
            self.assertTrue((output_dir / "repair_plan_patches.json").exists())
            self.assertTrue((output_dir / "artifacts" / "index.json").exists())

            summary = json.loads((output_dir / "harness_summary.json").read_text(encoding="utf-8"))
            field_diff = json.loads((output_dir / "field_diff.json").read_text(encoding="utf-8"))
            table_diff = json.loads((output_dir / "table_diff.json").read_text(encoding="utf-8"))
            list_diff = json.loads((output_dir / "list_diff.json").read_text(encoding="utf-8"))
            indexed = json.loads((output_dir / "artifacts" / "index.json").read_text(encoding="utf-8"))
            indexed_paths = {item["path"] for item in indexed["artifacts"]}

            self.assertTrue(summary["rule_only"])
            self.assertIn("quality_status", summary)
            self.assertIn("output_contract_status", summary)
            self.assertEqual(summary["output_contract_status"], "pass")
            self.assertEqual(summary["runtime_policy_status"], "pass")
            self.assertEqual(summary["runtime_policy_source"], "manifest")
            self.assertIn("label_accuracy", summary)
            self.assertIn("boundary_char_f1", summary)
            self.assertIn("sequence_gap_count", summary)
            self.assertIn("required_prefix_issue_count", summary)
            self.assertIn("container_duplicate_issue_count", summary)
            self.assertIn("agent_override_issue_count", summary)
            self.assertIn("duplicate_coverage_issue_count", summary)
            self.assertEqual(field_diff["expected_count"], 1)
            self.assertIn("label_accuracy", field_diff)
            self.assertIn("boundary_char_f1", field_diff)
            self.assertTrue(table_diff["row_count_checks"])
            self.assertTrue(list_diff["item_count_checks"])
            score_report = (output_dir / "score_report.md").read_text(encoding="utf-8")
            self.assertIn("sequence_gap_count", score_report)
            self.assertIn("required_prefix_issue_count", score_report)
            self.assertIn("harness_summary.json", indexed_paths)
            self.assertIn("list_diff.json", indexed_paths)
            self.assertIn("visual_document_graph.json", indexed_paths)
            self.assertIn("schema_audit.json", indexed_paths)
            self.assertIn("revision_blocks.json", indexed_paths)
            self.assertIn("runtime_policy.json", indexed_paths)
            self.assertIn("audit_input.json", indexed_paths)
            self.assertIn("agent_execution_report.json", indexed_paths)
            self.assertIn("missing_item_report.json", indexed_paths)
            self.assertIn("output_contract_validation_report.json", indexed_paths)
            self.assertIn("mvp_acceptance_metrics.json", indexed_paths)
            self.assertIn("auto_ingest_candidates.json", indexed_paths)
            self.assertIn("comparison_index.json", indexed_paths)
            self.assertIn("review_tasks.json", indexed_paths)
            self.assertIn("repair_plan_patches.json", indexed_paths)
            self.assertIn("score_report.md", indexed_paths)


if __name__ == "__main__":
    unittest.main()
