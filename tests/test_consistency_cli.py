import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_parser.config import RuntimeConfig
from document_parser.consistency_cli import ConsistencyPipelineError, _ensure_ocr_token_for_real_ocr
from tests.test_standard_xlsx import _write_standard_xlsx_fixture


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_IMAGE = ROOT / "test_png" / "粽子包装图.jpg"
OCR_FIXTURE = ROOT / "test_ocr" / "zongzi_ocr_result.json"


class ConsistencyCliTests(unittest.TestCase):
    def test_check_package_consistency_help_is_available(self) -> None:
        completed = _run_cli("--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("生成包装设计图文字一致性报告", completed.stdout)
        self.assertIn("--standard", completed.stdout)
        self.assertIn("--image", completed.stdout)
        self.assertIn("--output-dir", completed.stdout)
        self.assertIn("--ocr-mode", completed.stdout)
        self.assertIn("--ppocr-fixture", completed.stdout)
        self.assertIn("--glm-ocr-fixture", completed.stdout)
        self.assertIn("--llm-mode", completed.stdout)
        self.assertIn("--publish-cos", completed.stdout)
        self.assertIn("--cos-dry-run", completed.stdout)
        self.assertNotIn("--standard-manifest", completed.stdout)
        self.assertNotIn(".pdf", completed.stdout)

    def test_interactive_ocr_token_sets_process_env_for_real_ocr(self) -> None:
        original_key = os.environ.pop("GLM_OCR_API_KEY", None)
        original_zai = os.environ.pop("ZAI_API_KEY", None)
        original_zhipu = os.environ.pop("ZHIPUAI_API_KEY", None)
        try:
            with patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value="token-123"):
                _ensure_ocr_token_for_real_ocr(None, runtime_config=RuntimeConfig())

            self.assertEqual(os.environ["GLM_OCR_API_KEY"], "token-123")
        finally:
            os.environ.pop("GLM_OCR_API_KEY", None)
            if original_key is not None:
                os.environ["GLM_OCR_API_KEY"] = original_key
            if original_zai is not None:
                os.environ["ZAI_API_KEY"] = original_zai
            if original_zhipu is not None:
                os.environ["ZHIPUAI_API_KEY"] = original_zhipu

    def test_missing_ocr_token_in_non_interactive_mode_fails_before_real_ocr(self) -> None:
        original_key = os.environ.pop("GLM_OCR_API_KEY", None)
        original_zai = os.environ.pop("ZAI_API_KEY", None)
        original_zhipu = os.environ.pop("ZHIPUAI_API_KEY", None)
        try:
            with patch("sys.stdin.isatty", return_value=False):
                with self.assertRaises(ConsistencyPipelineError) as context:
                    _ensure_ocr_token_for_real_ocr(None, runtime_config=RuntimeConfig())

            self.assertEqual(context.exception.stage, "package_image_comparison")
            self.assertEqual(context.exception.error_type, "MissingOcrTokenError")
            self.assertIn("缺少 GLM-OCR token", str(context.exception))
        finally:
            if original_key is not None:
                os.environ["GLM_OCR_API_KEY"] = original_key
            if original_zai is not None:
                os.environ["ZAI_API_KEY"] = original_zai
            if original_zhipu is not None:
                os.environ["ZHIPUAI_API_KEY"] = original_zhipu

    def test_ocr_token_check_uses_runtime_config_loaded_from_dotenv(self) -> None:
        with patch("sys.stdin.isatty", return_value=False):
            _ensure_ocr_token_for_real_ocr(
                None,
                ocr_mode="hybrid",
                runtime_config=RuntimeConfig(glm_ocr_api_key="glm-token", ppocrv6_api_key="pp-token"),
            )

    def test_check_package_consistency_cli_runs_xlsx_then_image_compare_with_fixture(self) -> None:
        if not PACKAGE_IMAGE.exists() or not OCR_FIXTURE.exists():
            self.skipTest("zongzi package image or OCR fixture is not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            standard_path = temp_path / "standard.xlsx"
            output_dir = temp_path / "report"
            _write_standard_xlsx_fixture(standard_path)

            completed = _run_cli(
                "--standard",
                str(standard_path),
                "--image",
                str(PACKAGE_IMAGE),
                "--ocr-fixture",
                str(OCR_FIXTURE),
                "--output-dir",
                str(output_dir),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            for artifact_name in (
                "standard_structure/standard_items.json",
                "standard_structure/tables.json",
                "standard_structure/field_groups.json",
                "standard_structure/quality_report.json",
                "comparison_result.json",
                "package_ocr_lines.json",
                "package_ppocr_lines.json",
                "package_glm_lines.json",
                "package_fusion_evidence.json",
                "package_fusion_quality_report.json",
                "package_ocr_quality_report.json",
                "package_glm_blocks.json",
                "package_llm_structure_input.json",
                "package_llm_structure_output.json",
                "package_structured_items.json",
                "package_structure_quality_report.json",
                "result_preview.html",
                "pipeline_summary.json",
                "artifacts/index.json",
            ):
                self.assertTrue((output_dir / artifact_name).exists(), artifact_name)

            manifest = json.loads((output_dir / "00_inputs" / "standard_manifest.generated.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest, {"input_xlsx": str(standard_path.resolve())})
            summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertRegex(summary["run_id"], r"^pkg_consistency_")
            self.assertEqual(summary["status"], "completed")
            self.assertIsNotNone(summary["created_at"])
            self.assertIsNotNone(summary["completed_at"])
            self.assertIsInstance(summary["duration_seconds"], float)
            self.assertEqual(summary["stages"]["standard_structure"]["status"], "pass")
            self.assertEqual(summary["stages"]["package_image_comparison"]["status"], "pass")
            self.assertIsInstance(summary["stages"]["standard_structure"]["duration_seconds"], float)
            self.assertIsInstance(summary["stages"]["package_image_comparison"]["duration_seconds"], float)
            self.assertTrue(summary["inputs"]["standard_manifest_generated"])
            self.assertIn(summary["comparison_status"], {"pass", "fail", "manual_review"})

            html = (output_dir / "result_preview.html").read_text(encoding="utf-8")
            self.assertIn("检查结论", html)
            self.assertIn("标准文档", html)
            self.assertIn("包装图文字", html)
            self.assertIn("standardItem.scrollIntoView", html)
            self.assertNotIn("OCR lines", html)
            self.assertNotIn("semantic_key", html)

    def test_check_package_consistency_cli_reports_missing_hybrid_tokens_without_downgrading(self) -> None:
        if not PACKAGE_IMAGE.exists():
            self.skipTest("zongzi package image is not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            standard_path = temp_path / "standard.xlsx"
            output_dir = temp_path / "report"
            _write_standard_xlsx_fixture(standard_path)

            completed = _run_cli(
                "--standard",
                str(standard_path),
                "--image",
                str(PACKAGE_IMAGE),
                "--output-dir",
                str(output_dir),
                cwd=temp_path,
            )

            self.assertEqual(completed.returncode, 1)
            failure = json.loads((output_dir / "failure_result.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "package_image_comparison")
            self.assertEqual(failure["error_type"], "MissingOcrTokenError")
            self.assertIn("缺少 GLM-OCR token", failure["reason"])
            self.assertIn("缺少 PP-OCR token", failure["reason"])
            self.assertEqual(summary["failed_stage"], "package_image_comparison")
            self.assertEqual(summary["stages"]["package_image_comparison"]["error_type"], "MissingOcrTokenError")
            self.assertFalse((output_dir / "comparison_result.json").exists())

    def test_check_package_consistency_cli_can_build_cos_dry_run_bundle(self) -> None:
        if not PACKAGE_IMAGE.exists() or not OCR_FIXTURE.exists():
            self.skipTest("zongzi package image or OCR fixture is not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            standard_path = temp_path / "standard.xlsx"
            output_dir = temp_path / "report"
            _write_standard_xlsx_fixture(standard_path)

            completed = _run_cli(
                "--standard",
                str(standard_path),
                "--image",
                str(PACKAGE_IMAGE),
                "--ocr-fixture",
                str(OCR_FIXTURE),
                "--output-dir",
                str(output_dir),
                "--publish-cos",
                "--cos-dry-run",
                "--cos-key-prefix",
                "test/{run_id}",
                env_overrides=_fake_cos_env(),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            for artifact_name in (
                "artifacts/06_publish/cos_upload_result.json",
                "artifacts/06_publish/public_bundle/result_preview.html",
                "artifacts/06_publish/public_bundle/package_image.jpg",
                "artifacts/06_publish/public_bundle/pipeline_summary.public.json",
                "artifacts/06_publish/public_bundle/artifacts/index.public.json",
            ):
                self.assertTrue((output_dir / artifact_name).exists(), artifact_name)

            summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            publish_result = json.loads((output_dir / "artifacts/06_publish/cos_upload_result.json").read_text(encoding="utf-8"))
            public_summary_text = (output_dir / "artifacts/06_publish/public_bundle/pipeline_summary.public.json").read_text(encoding="utf-8")
            self.assertEqual(summary["stages"]["publish"]["status"], "pass")
            self.assertEqual(summary["publish"]["status"], "success")
            self.assertTrue(summary["publish"]["dry_run"])
            self.assertEqual(publish_result["status"], "success")
            self.assertIn("https://cdn.example.test/test/pkg_consistency_", summary["key_artifacts"]["published_report_html"])
            self.assertNotIn(str(temp_path), public_summary_text)
            self.assertNotIn("fake-secret", public_summary_text)
            self.assertNotIn("PACKAGING_COS_SECRET", public_summary_text)

    def test_check_package_consistency_cli_stops_before_compare_when_standard_fails(self) -> None:
        if not PACKAGE_IMAGE.exists() or not OCR_FIXTURE.exists():
            self.skipTest("zongzi package image or OCR fixture is not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            standard_path = temp_path / "standard.txt"
            standard_path.write_text("unsupported", encoding="utf-8")
            output_dir = temp_path / "report"

            completed = _run_cli(
                "--standard",
                str(standard_path),
                "--image",
                str(PACKAGE_IMAGE),
                "--ocr-fixture",
                str(OCR_FIXTURE),
                "--output-dir",
                str(output_dir),
            )

            self.assertEqual(completed.returncode, 1)
            self.assertTrue((output_dir / "failure_result.json").exists())
            self.assertTrue((output_dir / "pipeline_summary.json").exists())
            self.assertFalse((output_dir / "comparison_result.json").exists())
            failure = json.loads((output_dir / "failure_result.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertRegex(summary["run_id"], r"^pkg_consistency_failed_")
            self.assertEqual(failure["stage"], "standard_structure")
            self.assertEqual(failure["error_type"], "UnsupportedStandardInputError")
            self.assertIn("仅支持标准模板 Excel", failure["reason"])
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failed_stage"], "standard_structure")
            self.assertEqual(summary["stages"]["package_image_comparison"]["status"], "not_started")


def _fake_cos_env() -> dict[str, str]:
    return {
        "PACKAGING_COS_SECRET_ID": "fake-id",
        "PACKAGING_COS_SECRET_KEY": "fake-secret",
        "PACKAGING_COS_BUCKET_URL": "https://cos.example.test?bucket=test-bucket&region=ap-guangzhou",
        "PACKAGING_COS_CDN_DOMAIN": "https://cdn.example.test",
    }


def _run_cli(*args: str, env_overrides: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("GLM_OCR_API_KEY", None)
    env.pop("ZAI_API_KEY", None)
    env.pop("ZHIPUAI_API_KEY", None)
    env.pop("PPOCRV6_API_KEY", None)
    env.pop("PPOCRV6_TOKEN", None)
    if env_overrides:
        env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if part)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_package_consistency.py"), *args],
        cwd=cwd or ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


if __name__ == "__main__":
    unittest.main()
