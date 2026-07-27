import unittest

from document_parser.auto_ingest import build_auto_ingest_candidates
from document_parser.models import Risk


class AutoIngestTests(unittest.TestCase):
    def test_builds_candidates_for_verified_evidence_bound_items(self) -> None:
        report = build_auto_ingest_candidates(
            [_item("std_0001", "fld_0001", "product.name")],
            [{"target_id": "fld_0001", "result": "passed", "check_type": "no_guessing"}],
            [Risk("risk_0001", "field", "fld_0001", "low", "normalization_applied", "normalized")],
            {"overall_status": "pass_with_warnings", "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 1, "auto_ingest_allowed": True},
        )

        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["document_auto_ingest_allowed"])
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["blocked_count"], 0)
        self.assertEqual(report["candidates"][0]["field_id"], "fld_0001")

    def test_blocks_items_with_high_risk_or_failed_validation(self) -> None:
        report = build_auto_ingest_candidates(
            [_item("std_0001", "fld_0001", "product.name")],
            [{"target_id": "fld_0001", "result": "failed", "check_type": "format_check", "validation_id": "val_0001"}],
            [Risk("risk_0001", "field", "fld_0001", "high", "format_check_failed", "bad format")],
            {"overall_status": "manual_review_required", "high_risk_count": 1, "medium_risk_count": 0, "low_risk_count": 0, "auto_ingest_allowed": False},
        )

        self.assertEqual(report["status"], "blocked_by_document_quality")
        self.assertFalse(report["document_auto_ingest_allowed"])
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["blocked_count"], 1)
        reasons = {item["reason"] for item in report["blocked_items"][0]["block_reasons"]}
        self.assertIn("validation_failed", reasons)
        self.assertIn("risk_present", reasons)


def _item(item_id: str, field_id: str, semantic_key: str) -> dict:
    return {
        "id": item_id,
        "field_id": field_id,
        "semantic_key": semantic_key,
        "field": semantic_key,
        "label": semantic_key,
        "text": "value",
        "normalized_text": "value",
        "confidence": 0.97,
        "source": {"page": 1},
        "evidence_refs": ["ev_0001"],
        "comparison_required": True,
        "review_required": False,
        "status": "verified",
    }


if __name__ == "__main__":
    unittest.main()
