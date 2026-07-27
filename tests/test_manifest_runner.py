import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from document_parser.manifest import load_manifest, manifest_input_pdf, redacted_manifest


ROOT = Path(__file__).resolve().parents[1]


class ManifestRunnerTests(unittest.TestCase):
    def test_manifest_resolves_nested_relative_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            pdf = base / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps({"input": {"path": "sample.pdf"}, "api_key": "secret"}), encoding="utf-8")

            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest_input_pdf(manifest, manifest_path), pdf.resolve())
            self.assertEqual(redacted_manifest(manifest)["api_key"], "***REDACTED***")

    def test_extract_structure_rejects_cloud_ocr_without_consent_before_upload(self) -> None:
        sample_pdf = ROOT / "test-documents" / "识别文字.pdf"
        if not sample_pdf.exists():
            self.skipTest("sample PDF not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.json"
            output_dir = temp_path / "out"
            manifest_path.write_text(
                json.dumps(
                    {
                        "input_pdf": str(sample_pdf),
                        "ocr_mode": "glm_ocr",
                        "cloud_ocr_consent": False,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "extract_structure.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src"), "GLM_OCR_API_KEY": "dummy-key"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Runtime policy failed", completed.stderr)
            self.assertIn("cloud_ocr_consent", completed.stderr)
            self.assertFalse((output_dir / "00_inputs" / "file_inventory.json").exists())
            failure = json.loads((output_dir / "failure_result.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["job"]["status"], "failed")
            self.assertEqual(failure["metadata"]["failure"]["stage"], "runtime_policy")
            self.assertIn("cloud_ocr_consent", failure["metadata"]["failure"]["reason"])
            self.assertTrue((output_dir / "schemas" / "final_result.schema.json").exists())
            artifact_index = json.loads((output_dir / "artifacts" / "index.json").read_text(encoding="utf-8"))
            indexed_paths = {item["path"] for item in artifact_index["artifacts"]}
            self.assertIn("failure_result.json", indexed_paths)
            self.assertIn("schemas/final_result.schema.json", indexed_paths)

    def test_extract_structure_requires_llm_env_when_online_agent_enabled(self) -> None:
        sample_pdf = ROOT / "test-documents" / "识别文字.pdf"
        if not sample_pdf.exists():
            self.skipTest("sample PDF not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.json"
            output_dir = temp_path / "out"
            manifest_path.write_text(
                json.dumps(
                    {
                        "input_pdf": str(sample_pdf),
                        "ocr_fixture_path": str(ROOT / "tests" / "fixtures" / "ppocrv6_empty.json"),
                        "use_llm_agent": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "extract_structure.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src"), "LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("LLM_API_KEY", completed.stderr)
            self.assertFalse((output_dir / "00_inputs" / "file_inventory.json").exists())
            failure = json.loads((output_dir / "failure_result.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["job"]["status"], "failed")
            self.assertEqual(failure["metadata"]["failure"]["stage"], "runtime_config")
            self.assertIn("LLM_API_KEY", failure["metadata"]["failure"]["reason"])
            self.assertTrue((output_dir / "schemas" / "final_result.schema.json").exists())

    def test_extract_structure_script_writes_layered_artifacts(self) -> None:
        sample_pdf = ROOT / "test-documents" / "识别文字.pdf"
        if not sample_pdf.exists():
            self.skipTest("sample PDF not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.json"
            output_dir = temp_path / "out"
            manifest_path.write_text(
                json.dumps(
                    {
                        "input_pdf": str(sample_pdf),
                        "ocr_fixture_path": str(ROOT / "tests" / "fixtures" / "ppocrv6_empty.json"),
                        "quality_gate": "strict",
                        "extraction_mode": "agent_plus_rule",
                        "repair_mode": "plan_only",
                        "max_repair_rounds": 1,
                        "service_api_key": "do-not-write-this",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "extract_structure.py"),
                    "--manifest",
                    str(manifest_path),
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
            self.assertTrue((output_dir / "page_images.json").exists())
            self.assertTrue((output_dir / "page_images" / "page_001.png").exists())
            self.assertTrue((output_dir / "visual_document_graph.json").exists())
            self.assertTrue((output_dir / "vdg_nodes.json").exists())
            self.assertTrue((output_dir / "missing_item_report.json").exists())
            self.assertTrue((output_dir / "missing_fields.json").exists())
            self.assertTrue((output_dir / "missing_tables.json").exists())
            self.assertTrue((output_dir / "revision_blocks.json").exists())
            self.assertTrue((output_dir / "schema_audit.json").exists())
            self.assertTrue((output_dir / "output_contract_validation_report.json").exists())
            self.assertTrue((output_dir / "mvp_acceptance_metrics.json").exists())
            self.assertTrue((output_dir / "audit_input.json").exists())
            self.assertTrue((output_dir / "runtime_policy.json").exists())
            self.assertTrue((output_dir / "json_export.json").exists())
            self.assertTrue((output_dir / "result_preview.html").exists())
            self.assertTrue((output_dir / "schemas" / "final_result.schema.json").exists())
            self.assertTrue((output_dir / "agent_execution_report.json").exists())
            self.assertTrue((output_dir / "standard_items.json").exists())
            self.assertTrue((output_dir / "comparison_index.json").exists())
            self.assertTrue((output_dir / "quality_report.json").exists())
            self.assertTrue((output_dir / "auto_ingest_candidates.json").exists())
            self.assertTrue((output_dir / "00_inputs" / "input_manifest.redacted.json").exists())
            self.assertTrue((output_dir / "00_inputs" / "runtime_policy.json").exists())
            self.assertTrue((output_dir / "00_inputs" / "json_export.json").exists())
            self.assertTrue((output_dir / "01_text_extraction" / "page_images.json").exists())
            self.assertTrue((output_dir / "01_text_extraction" / "visual_document_graph.json").exists())
            self.assertTrue((output_dir / "01_text_extraction" / "source_layers.json").exists())
            self.assertTrue((output_dir / "01_table_extraction" / "table_layers.json").exists())
            self.assertTrue((output_dir / "02_section_detection" / "regions.json").exists())
            self.assertTrue((output_dir / "02_section_detection" / "revision_blocks.json").exists())
            self.assertTrue((output_dir / "03_field_structure" / "standard_items.json").exists())
            self.assertTrue((output_dir / "03_field_structure" / "result_preview.html").exists())
            self.assertTrue((output_dir / "03_field_structure" / "comparison_index.json").exists())
            self.assertTrue((output_dir / "03_field_structure" / "schema_audit.json").exists())
            self.assertTrue((output_dir / "03_field_structure" / "agent_execution_report.json").exists())
            self.assertTrue((output_dir / "03_field_structure" / "auto_ingest_candidates.json").exists())
            self.assertTrue((output_dir / "03_field_structure" / "extracted_data.json").exists())
            self.assertTrue((output_dir / "04_validation" / "quality_report.json").exists())
            self.assertTrue((output_dir / "04_validation" / "auto_ingest_candidates.json").exists())
            self.assertTrue((output_dir / "04_validation" / "missing_item_report.json").exists())
            self.assertTrue((output_dir / "04_validation" / "schema_audit.json").exists())
            self.assertTrue((output_dir / "04_validation" / "output_contract_validation_report.json").exists())
            self.assertTrue((output_dir / "04_validation" / "mvp_acceptance_metrics.json").exists())
            self.assertTrue((output_dir / "04_validation" / "json_export.json").exists())
            self.assertTrue((output_dir / "04_validation" / "audit_input.json").exists())
            self.assertTrue((output_dir / "04_validation" / "evidence.json").exists())
            self.assertTrue((output_dir / "04_validation" / "risks.json").exists())
            self.assertTrue((output_dir / "04_validation" / "review_tasks.json").exists())
            self.assertTrue((output_dir / "04_validation" / "agent_execution_report.json").exists())
            self.assertTrue((output_dir / "05_repair" / "repair_plan.json").exists())
            self.assertTrue((output_dir / "05_repair" / "repair_plan_patches.json").exists())

            redacted = json.loads((output_dir / "00_inputs" / "input_manifest.redacted.json").read_text(encoding="utf-8"))
            self.assertEqual(redacted["service_api_key"], "***REDACTED***")
            runtime_policy = json.loads((output_dir / "runtime_policy.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime_policy["source"], "manifest")
            self.assertEqual(runtime_policy["ocr"]["mode"], "recorded_fixture")
            self.assertEqual(runtime_policy["repair"]["max_repair_rounds"], 1)
            json_export = json.loads((output_dir / "json_export.json").read_text(encoding="utf-8"))
            self.assertEqual(json_export["schema_version"], "mvp_final_json_v0.1")
            self.assertIn("review_task_targets_resolve", json_export["contract_checks"])

            artifact_index = json.loads((output_dir / "artifacts" / "index.json").read_text(encoding="utf-8"))
            indexed_paths = {item["path"] for item in artifact_index["artifacts"]}
            self.assertIn("00_inputs/input_manifest.redacted.json", indexed_paths)
            self.assertIn("00_inputs/runtime_policy.json", indexed_paths)
            self.assertIn("page_images/page_001.png", indexed_paths)
            self.assertIn("visual_document_graph.json", indexed_paths)
            self.assertIn("missing_item_report.json", indexed_paths)
            self.assertIn("revision_blocks.json", indexed_paths)
            self.assertIn("agent_execution_report.json", indexed_paths)
            self.assertIn("schema_audit.json", indexed_paths)
            self.assertIn("output_contract_validation_report.json", indexed_paths)
            self.assertIn("mvp_acceptance_metrics.json", indexed_paths)
            self.assertIn("audit_input.json", indexed_paths)
            self.assertIn("runtime_policy.json", indexed_paths)
            self.assertIn("json_export.json", indexed_paths)
            self.assertIn("result_preview.html", indexed_paths)
            self.assertIn("schemas/final_result.schema.json", indexed_paths)
            self.assertIn("auto_ingest_candidates.json", indexed_paths)
            self.assertIn("comparison_index.json", indexed_paths)
            self.assertIn("03_field_structure/standard_items.json", indexed_paths)
            self.assertIn("03_field_structure/result_preview.html", indexed_paths)
            self.assertIn("03_field_structure/comparison_index.json", indexed_paths)
            self.assertIn("04_validation/risks.json", indexed_paths)
            self.assertIn("04_validation/mvp_acceptance_metrics.json", indexed_paths)
            self.assertIn("04_validation/audit_input.json", indexed_paths)
            self.assertIn("05_repair/repair_plan.json", indexed_paths)
            self.assertIn("05_repair/repair_plan_patches.json", indexed_paths)


if __name__ == "__main__":
    unittest.main()
