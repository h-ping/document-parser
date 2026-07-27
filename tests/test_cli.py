import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from document_parser.output_contract import FINAL_JSON_ROOT_KEYS


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_cli_runs_with_recorded_ocr_fixture(self) -> None:
        sample_pdf = ROOT / "test-documents" / "识别文字.pdf"
        if not sample_pdf.exists():
            self.skipTest("sample PDF not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.json"
            debug_dir = Path(temp_dir) / "debug"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "document_parser.cli",
                    "--input",
                    str(sample_pdf),
                    "--output",
                    str(output),
                    "--debug-dir",
                    str(debug_dir),
                    "--ocr-fixture",
                    str(ROOT / "tests" / "fixtures" / "ppocrv6_empty.json"),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["job"]["job_type"], "standard_pdf_to_structured_json")
            self.assertIn("generated_schema", data)
            self.assertIn("quality", data)
            self.assertIn("cross_validation", data)
            self.assertIn("regions", data["extracted_data"])
            self.assertIn("tables", data["extracted_data"])
            self.assertIn("structure_audit", data["metadata"])
            self.assertIn("source_layers", data["metadata"])
            self.assertIn("source_consistency", data["metadata"])
            self.assertIn("source_anchor_inventory", data["metadata"])
            self.assertIn("coverage_map", data["metadata"])
            self.assertIn("standard_artifacts", data["metadata"])
            self.assertIn("agent_execution_report", data["metadata"])
            self.assertIn("agent_harness", data["metadata"])
            self.assertIn("table_parser", data["metadata"])
            self.assertIn("repair_loop", data["metadata"])
            self.assertIn("schema_audit", data["metadata"])
            self.assertIn("visual_document_graph", data["metadata"])
            self.assertIn("missing_item_report", data["metadata"])
            self.assertIn("runtime_policy", data["metadata"])
            self.assertIn("json_export", data["metadata"])
            self.assertIn("output_contract_validation_report", data["metadata"])
            self.assertIn("mvp_acceptance_metrics", data["metadata"])
            self.assertIn("label_text_scope_reference", data["metadata"])
            self.assertIn("label_text_scope_agent_context", data["metadata"])
            self.assertIn("label_text_scope_report", data["metadata"])
            self.assertIn("audit_input", data["metadata"])
            self.assertIn("output_contract", data["cross_validation"])
            self.assertEqual(data["metadata"]["runtime_policy"]["ocr"]["mode"], "recorded_fixture")
            self.assertEqual(data["metadata"]["runtime_policy"]["secrets"]["required_env_vars"], [])
            self.assertEqual(data["metadata"]["runtime_policy"]["secrets"]["optional_env_vars"], ["GLM_OCR_MODEL"])
            self.assertEqual(data["metadata"]["runtime_policy"]["llm_agent"]["required_env_vars"], [])
            self.assertFalse(data["metadata"]["runtime_policy"]["llm_agent"]["runtime_managed_online_llm"])
            self.assertTrue((output.parent / "schemas" / "final_result.schema.json").exists())
            self.assertEqual(data["metadata"]["json_export"]["media_type"], "application/json")
            self.assertEqual(data["metadata"]["json_export"]["encoding"], "utf-8")
            self.assertEqual(data["metadata"]["json_export"]["schema_artifact"], "schemas/final_result.schema.json")
            self.assertIn("evidence_refs_resolve", data["metadata"]["json_export"]["contract_checks"])
            self.assertEqual(data["metadata"]["output_contract_validation_report"]["status"], "pass")
            agent_report = data["metadata"]["agent_execution_report"]
            self.assertTrue(any(check["check_type"] == "extraction_and_review_are_independent_agents" and check["result"] == "passed" for check in agent_report["separation_checks"]))
            self.assertIn("repair_agent_candidates", data["metadata"]["repair_loop"])
            self.assertIn("repaired_source_layers", data["metadata"]["repair_loop"])
            self.assertEqual(data["document"]["page_image_status"], "rendered")
            self.assertEqual(data["metadata"]["page_images"]["status"], "rendered")
            self.assertGreater(data["metadata"]["visual_document_graph"]["edge_count"], 0)
            self.assertGreaterEqual(data["metadata"]["missing_item_report"]["missing_count"], 1)
            self.assertIn("missing_fields", data["extracted_data"])
            self.assertTrue(any(risk["risk_type"] == "critical_field_missing" for risk in data["risks"]))
            self.assertTrue(any(task["target_type"] == "missing_field" for task in data["review_tasks"]))
            self.assertTrue(any(item.get("bbox_status") == "available" for item in data["evidence"]))
            self.assertTrue(data["metadata"]["standard_artifacts"]["standard_items"])
            self.assertIn("field_id", data["metadata"]["standard_artifacts"]["standard_items"][0])
            self.assertIn("comparison_profile", data["metadata"]["standard_artifacts"]["standard_items"][0])
            self.assertIn("comparison_index", data["metadata"]["standard_artifacts"])
            self.assertIn("auto_ingest_candidates", data["metadata"]["standard_artifacts"])
            self.assertIn("status", data["metadata"]["standard_artifacts"]["quality_report"])
            self.assertFalse(data["metadata"]["standard_artifacts"]["quality_report"]["downstream_allowed"])
            self.assertTrue((debug_dir / "page_images.json").exists())
            self.assertTrue((debug_dir / "runtime_policy.json").exists())
            self.assertTrue((debug_dir / "json_export.json").exists())
            self.assertTrue((debug_dir / "page_images" / "page_001.png").exists())
            self.assertTrue((debug_dir / "vdg.json").exists())
            self.assertTrue((debug_dir / "visual_document_graph.json").exists())
            self.assertTrue((debug_dir / "vdg_nodes.json").exists())
            self.assertTrue((debug_dir / "label_text_scope_reference.json").exists())
            self.assertTrue((debug_dir / "label_text_scope_agent_context.json").exists())
            self.assertTrue((debug_dir / "label_text_scope_report.json").exists())
            self.assertTrue((debug_dir / "standard_items.json").exists())
            self.assertTrue((debug_dir / "comparison_index.json").exists())
            self.assertTrue((debug_dir / "quality_report.json").exists())
            self.assertTrue((debug_dir / "structured_document.json").exists())
            self.assertTrue((debug_dir / "taxonomy_proposals.json").exists())
            self.assertTrue((debug_dir / "field_groups.json").exists())
            self.assertTrue((debug_dir / "tables.json").exists())
            self.assertTrue((debug_dir / "lists.json").exists())
            self.assertTrue((debug_dir / "auto_ingest_candidates.json").exists())
            self.assertTrue((debug_dir / "table_layers.json").exists())
            self.assertTrue((debug_dir / "table_quality_report.json").exists())
            self.assertTrue((debug_dir / "source_layers.json").exists())
            self.assertTrue((debug_dir / "source_consistency_report.json").exists())
            self.assertTrue((debug_dir / "01_text_extraction" / "page_images.json").exists())
            self.assertTrue((debug_dir / "00_inputs" / "runtime_policy.json").exists())
            self.assertTrue((debug_dir / "00_inputs" / "json_export.json").exists())
            self.assertTrue((debug_dir / "01_text_extraction" / "visual_document_graph.json").exists())
            self.assertTrue((debug_dir / "01_text_extraction" / "source_consistency_report.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "source_consistency_report.json").exists())
            self.assertTrue((debug_dir / "source_anchor_inventory.json").exists())
            self.assertTrue((debug_dir / "coverage_map.json").exists())
            self.assertTrue((debug_dir / "schema_audit.json").exists())
            self.assertTrue((debug_dir / "structure_audit.json").exists())
            self.assertTrue((debug_dir / "missing_item_report.json").exists())
            self.assertTrue((debug_dir / "missing_fields.json").exists())
            self.assertTrue((debug_dir / "missing_tables.json").exists())
            self.assertTrue((debug_dir / "revision_blocks.json").exists())
            self.assertTrue((debug_dir / "output_contract_validation_report.json").exists())
            self.assertTrue((debug_dir / "mvp_acceptance_metrics.json").exists())
            self.assertTrue((debug_dir / "extracted_data.json").exists())
            self.assertTrue((debug_dir / "evidence.json").exists())
            self.assertTrue((debug_dir / "validation.json").exists())
            self.assertTrue((debug_dir / "risks.json").exists())
            self.assertTrue((debug_dir / "review_tasks.json").exists())
            self.assertTrue((debug_dir / "03_field_structure" / "extracted_data.json").exists())
            self.assertTrue((debug_dir / "03_field_structure" / "comparison_index.json").exists())
            self.assertTrue((debug_dir / "03_field_structure" / "schema_audit.json").exists())
            self.assertTrue((debug_dir / "03_field_structure" / "label_text_scope_report.json").exists())
            self.assertTrue((debug_dir / "02_section_detection" / "revision_blocks.json").exists())
            self.assertTrue((debug_dir / "03_field_structure" / "auto_ingest_candidates.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "evidence.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "auto_ingest_candidates.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "missing_item_report.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "schema_audit.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "label_text_scope_report.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "output_contract_validation_report.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "mvp_acceptance_metrics.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "json_export.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "risks.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "review_tasks.json").exists())
            self.assertTrue((debug_dir / "repair_plan.json").exists())
            self.assertTrue((debug_dir / "repair_trace.json").exists())
            self.assertTrue((debug_dir / "repair_attempts.json").exists())
            self.assertTrue((debug_dir / "repair_agent_candidates.json").exists())
            self.assertTrue((debug_dir / "repaired_source_layers.json").exists())
            self.assertTrue((debug_dir / "llm_agent_items.json").exists())
            self.assertTrue((debug_dir / "rejected_agent_items.json").exists())
            self.assertTrue((debug_dir / "review_items.json").exists())
            self.assertTrue((debug_dir / "agent_execution_report.json").exists())
            self.assertTrue((debug_dir / "agent_harness_report.json").exists())
            self.assertTrue((debug_dir / "03_field_structure" / "agent_execution_report.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "agent_execution_report.json").exists())
            self.assertTrue((debug_dir / "05_repair" / "repair_trace.json").exists())
            self.assertTrue((debug_dir / "extraction_plan.json").exists())
            self.assertTrue((debug_dir / "audit_input.json").exists())
            self.assertTrue((debug_dir / "schemas" / "final_result.schema.json").exists())
            self.assertTrue((debug_dir / "repair_plan_patches.json").exists())
            self.assertTrue((debug_dir / "04_validation" / "audit_input.json").exists())
            self.assertTrue((debug_dir / "05_repair" / "repair_plan_patches.json").exists())
            self.assertTrue((debug_dir / "artifacts" / "index.json").exists())
            artifact_index = json.loads((debug_dir / "artifacts" / "index.json").read_text(encoding="utf-8"))
            indexed_paths = {item["path"] for item in artifact_index["artifacts"]}
            self.assertIn("standard_items.json", indexed_paths)
            self.assertIn("comparison_index.json", indexed_paths)
            self.assertIn("runtime_policy.json", indexed_paths)
            self.assertIn("json_export.json", indexed_paths)
            self.assertIn("00_inputs/runtime_policy.json", indexed_paths)
            self.assertIn("00_inputs/json_export.json", indexed_paths)
            self.assertIn("quality_report.json", indexed_paths)
            self.assertIn("auto_ingest_candidates.json", indexed_paths)
            self.assertIn("page_images/page_001.png", indexed_paths)
            self.assertIn("visual_document_graph.json", indexed_paths)
            self.assertIn("label_text_scope_reference.json", indexed_paths)
            self.assertIn("label_text_scope_agent_context.json", indexed_paths)
            self.assertIn("label_text_scope_report.json", indexed_paths)
            self.assertIn("missing_item_report.json", indexed_paths)
            self.assertIn("revision_blocks.json", indexed_paths)
            self.assertIn("agent_execution_report.json", indexed_paths)
            self.assertIn("schema_audit.json", indexed_paths)
            self.assertIn("output_contract_validation_report.json", indexed_paths)
            self.assertIn("mvp_acceptance_metrics.json", indexed_paths)
            self.assertIn("audit_input.json", indexed_paths)
            self.assertIn("repair_plan_patches.json", indexed_paths)
            self.assertIn("04_validation/review_tasks.json", indexed_paths)
            self.assertIn("04_validation/label_text_scope_report.json", indexed_paths)
            self.assertIn("04_validation/mvp_acceptance_metrics.json", indexed_paths)
            self.assertIn("04_validation/audit_input.json", indexed_paths)
            self.assertIn("03_field_structure/comparison_index.json", indexed_paths)
            self.assertIn("03_field_structure/label_text_scope_report.json", indexed_paths)
            self.assertIn("05_repair/repair_plan_patches.json", indexed_paths)
            self.assertIn("schemas/final_result.schema.json", indexed_paths)

    def test_cli_writes_structured_failure_json_for_invalid_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            broken_pdf = temp_path / "broken.pdf"
            output = temp_path / "result.json"
            broken_pdf.write_bytes(b"this is not a pdf")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "document_parser.cli",
                    "--input",
                    str(broken_pdf),
                    "--output",
                    str(output),
                    "--ocr-fixture",
                    str(ROOT / "tests" / "fixtures" / "ppocrv6_empty.json"),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("parse-pdf failed:", completed.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue((output.parent / "schemas" / "final_result.schema.json").exists())

        self.assertEqual(list(data.keys()), FINAL_JSON_ROOT_KEYS)
        self.assertEqual(data["job"]["status"], "failed")
        self.assertEqual(data["document"]["parse_status"], "failed")
        self.assertEqual(data["metadata"]["failure"]["stage"], "pdf_read")
        self.assertEqual(data["metadata"]["failure"]["error_type"], "PdfReadError")
        self.assertEqual(data["metadata"]["json_export"]["schema_artifact"], "schemas/final_result.schema.json")
        self.assertIn("Failed to read PDF", data["metadata"]["failure"]["reason"])
        self.assertEqual(data["risks"][0]["risk_type"], "parse_failed")
        self.assertTrue(data["review_tasks"][0]["required"])


if __name__ == "__main__":
    unittest.main()
