import json
import tempfile
import unittest
from pathlib import Path

from document_parser.pdf_regression import REQUIRED_DEBUG_ARTIFACTS, validate_regression_payload


class PdfRegressionTests(unittest.TestCase):
    def test_validate_regression_payload_accepts_current_mvp_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "case.json"
            debug_dir = root / "case-debug"
            payload = _valid_payload()
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            _write_required_debug_artifacts(debug_dir)

            issues = validate_regression_payload(payload, output_path, debug_dir)

        self.assertEqual(issues, [])

    def test_validate_regression_payload_fails_missing_contract_and_comparison_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "case.json"
            debug_dir = root / "case-debug"
            payload = _valid_payload()
            payload["metadata"]["output_contract_validation_report"] = {"status": "review_required", "failed_count": 1}
            payload["metadata"]["standard_artifacts"]["comparison_index"]["entries"] = []
            payload["metadata"]["standard_artifacts"]["comparison_index"]["entry_count"] = 1
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            _write_required_debug_artifacts(debug_dir, skip={"comparison_index.json"})

            issues = validate_regression_payload(payload, output_path, debug_dir)

        checks = {issue["check"] for issue in issues}
        self.assertIn("output_contract_passed", checks)
        self.assertIn("comparison_index_entry_count", checks)
        self.assertIn("comparison_index_covers_standard_items", checks)
        self.assertIn("debug_artifact_exists", checks)


def _valid_payload() -> dict:
    return {
        "job": {"status": "completed_with_warnings"},
        "document": {"parse_status": "completed_with_warnings"},
        "metadata": {
            "output_contract_validation_report": {"status": "pass", "failed_count": 0},
            "mvp_acceptance_metrics": {
                "metrics_version": "mvp_acceptance_metrics_v0.1",
                "risk": {"output_contract_status": "pass"},
            },
            "vdg_quality_report": {
                "status": "pass",
                "source_span_coverage_rate": 1.0,
                "edge_ref_status": "pass",
            },
            "vdg_consumption_report": {
                "unknown_important_node_count": 0,
                "conflict_node_count": 0,
            },
            "label_text_scope_report": {
                "status": "pass",
                "extracted_out_of_scope_count": 0,
                "ignored_noise_node_count": 0,
                "unknown_scope_node_count": 0,
                "scope_gate_rejected_count": 0,
            },
            "layout_quality_report": {
                "report_version": "layout_quality_v0.1",
                "status": "disabled",
                "mode": "legacy",
            },
            "standard_artifacts": {
                "standard_items": [{"id": "std_0001", "comparison_required": True}],
                "comparison_index": {
                    "entry_count": 1,
                    "skipped_count": 0,
                    "entries": [{"comparison_id": "cmp_0001", "standard_item_id": "std_0001"}],
                    "skipped_items": [],
                },
                "quality_report": {"status": "review_required", "downstream_allowed": False},
            },
        },
    }


def _write_required_debug_artifacts(debug_dir: Path, skip: set[str] | None = None) -> None:
    skipped = skip or set()
    for artifact in REQUIRED_DEBUG_ARTIFACTS:
        if artifact in skipped:
            continue
        path = debug_dir / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
