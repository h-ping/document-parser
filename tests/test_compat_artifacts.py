import unittest
from pathlib import Path

from document_parser.compat_artifacts import (
    build_field_groups,
    build_quality_report,
    build_standard_items,
    build_taxonomy_proposals,
)
from document_parser.models import BBoxPdf, BBoxNormalized, CompiledField, Evidence, Risk, TextSpan


class CompatArtifactTests(unittest.TestCase):
    def test_standard_items_are_evidence_bound(self) -> None:
        spans = [
            TextSpan(
                "span_0001",
                1,
                "品名：牛奶",
                "pdf_text",
                bbox_pdf=BBoxPdf(10, 20, 30, 12, 200, 100),
                bbox_normalized=BBoxNormalized(0.05, 0.2, 0.2, 0.32),
            )
        ]
        evidence = [
            Evidence(
                "ev_0001",
                "品名：牛奶",
                1,
                ["pdf_text"],
                "available",
                ["span_0001"],
                bbox_pdf=spans[0].bbox_pdf,
                bbox_normalized=spans[0].bbox_normalized,
            )
        ]
        field = _field("fld_0001", "product.name", "品名", "品名：牛奶", ["ev_0001"])

        items = build_standard_items({"fld_0001": field}, evidence, spans, Path("sample.pdf"))

        self.assertEqual(items[0]["field"], "product_name")
        self.assertEqual(items[0]["text"], "品名：牛奶")
        self.assertEqual(items[0]["source"]["char_start"], 0)
        self.assertEqual(items[0]["source"]["char_end"], 5)
        self.assertEqual(items[0]["source"]["bbox"], [10, 20, 30, 12])
        self.assertEqual(items[0]["source"]["bbox_normalized"], {"x1": 0.05, "y1": 0.2, "x2": 0.2, "y2": 0.32})
        self.assertTrue(items[0]["value_hash"].startswith("sha256:"))
        self.assertEqual(items[0]["comparison_profile"]["semantic_key"], "product.name")
        self.assertEqual(items[0]["comparison_profile"]["bbox_normalized"], {"x1": 0.05, "y1": 0.2, "x2": 0.2, "y2": 0.32})
        self.assertEqual(items[0]["comparison_profile"]["evidence_refs"], ["ev_0001"])
        self.assertTrue(items[0]["comparison_required"])

    def test_standard_items_mark_before_revision_as_historical(self) -> None:
        spans = [
            TextSpan("span_0001", 1, "品名：旧名称", "pdf_text"),
            TextSpan("span_0002", 1, "品名：新名称", "pdf_text"),
        ]
        evidence = [
            Evidence("ev_0001", "品名：旧名称", 1, ["pdf_text"], "missing", ["span_0001"]),
            Evidence("ev_0002", "品名：新名称", 1, ["pdf_text"], "missing", ["span_0002"]),
        ]
        fields = {
            "fld_0001": _field("fld_0001", "product.name", "品名", "品名：旧名称", ["ev_0001"]),
            "fld_0002": _field("fld_0002", "product.name", "品名", "品名：新名称", ["ev_0002"]),
        }
        revision_blocks = [
            {"revision_role": "before", "fields": [{"field_id": "fld_0001"}]},
            {"revision_role": "after", "fields": [{"field_id": "fld_0002"}]},
        ]

        items = build_standard_items(fields, evidence, spans, Path("sample.pdf"), revision_blocks)
        by_field = {item["field_id"]: item for item in items}

        self.assertEqual(by_field["fld_0001"]["revision_role"], "historical_reference")
        self.assertFalse(by_field["fld_0001"]["is_current_standard"])
        self.assertFalse(by_field["fld_0001"]["comparison_required"])
        self.assertEqual(by_field["fld_0002"]["revision_role"], "current_standard")
        self.assertTrue(by_field["fld_0002"]["is_current_standard"])
        self.assertTrue(by_field["fld_0002"]["comparison_required"])

    def test_quality_report_blocks_downstream_when_strict_gate_fails(self) -> None:
        risk = Risk("risk_0001", "field", "fld_0001", "high", "manual_review_required", "需要复核")
        report = build_quality_report(
            {"overall_status": "manual_review_required", "high_risk_count": 1, "medium_risk_count": 0, "low_risk_count": 0},
            [risk],
            [{"result": "passed", "target_id": "fld_0001"}],
            {"status": "pass", "issues": []},
            {"anchor_coverage": 0.5, "missing_anchor_count": 1, "group_issue_count": 0, "table_issue_count": 0, "duplicate_coverage_issue_count": 0},
            {"status": "review_required", "source_issues": []},
            {"status": "pass", "issues": []},
            {"repair_mode": "plan_only", "status": "review_required", "actions": [{"action_id": "repair_action_0001"}]},
        )

        self.assertEqual(report["status"], "review_required")
        self.assertFalse(report["downstream_allowed"])
        self.assertTrue(any(issue["issue_type"] == "manual_review_required" for issue in report["issues"]))

    def test_quality_report_blocks_downstream_when_schema_audit_fails(self) -> None:
        report = build_quality_report(
            {"overall_status": "pass", "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0, "auto_ingest_allowed": True},
            [],
            [{"result": "passed", "target_id": "fld_0001"}],
            {
                "status": "review_required",
                "issues": [{"issue_type": "schema_obvious_field_missing", "severity": "high", "message": "缺少明显字段"}],
            },
            {"anchor_coverage": 1.0, "missing_anchor_count": 0, "group_issue_count": 0, "table_issue_count": 0, "duplicate_coverage_issue_count": 0},
            {"status": "pass", "source_issues": []},
            {"status": "pass", "issues": []},
            {"repair_mode": "plan_only", "status": "review_required", "actions": []},
        )

        self.assertEqual(report["status"], "review_required")
        self.assertFalse(report["downstream_allowed"])
        self.assertFalse(report["auto_ingest_allowed"])
        self.assertTrue(any(check["check"] == "schema_audit_pass" and check["result"] == "failed" for check in report["gate_checks"]))
        self.assertTrue(any(issue["issue_type"] == "schema_obvious_field_missing" for issue in report["issues"]))

    def test_quality_report_summarizes_repair_loop_state(self) -> None:
        report = build_quality_report(
            {"overall_status": "manual_review_required", "high_risk_count": 1, "medium_risk_count": 0, "low_risk_count": 0},
            [Risk("risk_0001", "field", "fld_0001", "high", "manual_review_required", "需要修复")],
            [{"result": "passed", "target_id": "fld_0001"}],
            {"status": "pass", "issues": []},
            {"anchor_coverage": 1.0, "missing_anchor_count": 0, "group_issue_count": 0, "table_issue_count": 0, "duplicate_coverage_issue_count": 0},
            {"status": "pass", "source_issues": []},
            {"status": "pass", "issues": []},
            {
                "repair_mode": "plan_only",
                "max_repair_rounds": 2,
                "status": "review_required",
                "actions": [{"action_id": "repair_action_0001"}],
            },
            {"status": "skipped_plan_only", "attempt_count": 1, "attempts": []},
            {
                "status": "review_required",
                "round_count": 2,
                "applied_attempt_count": 1,
                "final_audit_finding_count": 1,
            },
            {"status": "pending_agent_fill", "candidate_count": 1, "candidates": []},
        )

        self.assertEqual(
            report["repair_summary"],
            {
                "repair_mode": "plan_only",
                "status": "review_required",
                "plan_status": "review_required",
                "action_count": 1,
                "max_repair_rounds": 2,
                "attempt_status": "skipped_plan_only",
                "attempt_count": 1,
                "trace_status": "review_required",
                "round_count": 2,
                "applied_attempt_count": 1,
                "final_audit_finding_count": 1,
                "agent_candidate_status": "pending_agent_fill",
                "agent_candidate_count": 1,
                "downstream_blocking_reason": "repair_plan_pending",
            },
        )

    def test_quality_report_blocks_on_full_structure_audit_gate(self) -> None:
        issue = {
            "expected": "expected",
            "actual": "actual",
            "source": {"field_id": "fld_0001"},
            "repair_hint": "repair",
        }
        report = build_quality_report(
            {"overall_status": "pass", "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0, "auto_ingest_allowed": True},
            [],
            [{"result": "passed", "target_id": "fld_0001"}],
            {"status": "pass", "issues": []},
            {
                "anchor_coverage": 1.0,
                "missing_anchor_count": 0,
                "sequence_gap_count": 1,
                "group_issue_count": 0,
                "table_issue_count": 0,
                "required_prefix_issue_count": 1,
                "required_prefix_issues": [issue],
                "container_duplicate_issue_count": 1,
                "container_duplicate_issues": [issue],
                "agent_override_issue_count": 1,
                "agent_override_issues": [issue],
                "duplicate_coverage_issue_count": 1,
                "duplicate_coverage_issues": [issue],
                "anchor_inventory": [{"span_id": "span_0001"}],
            },
            {"status": "pass", "source_issues": []},
            {"status": "pass", "issues": []},
            {"repair_mode": "plan_only", "status": "pass", "actions": []},
        )

        self.assertEqual(report["status"], "review_required")
        self.assertFalse(report["downstream_allowed"])
        failed_checks = {check["check"] for check in report["gate_checks"] if check["result"] == "failed"}
        self.assertEqual(
            failed_checks,
            {
                "sequence_gap_count_zero",
                "required_prefix_issue_count_zero",
                "container_duplicate_issue_count_zero",
                "agent_override_issue_count_zero",
                "duplicate_coverage_issue_count_zero",
            },
        )
        issue_types = {issue["issue_type"] for issue in report["issues"]}
        self.assertTrue(
            {
                "content_sequence_gap",
                "required_prefix_issue",
                "container_duplicate_issue",
                "agent_override_issue",
                "duplicate_coverage_issue",
            }.issubset(issue_types)
        )

    def test_field_groups_and_taxonomy_proposals_are_derived_from_fields(self) -> None:
        custom = _field("fld_0001", "custom.design_note", "设计注意", "设计注意：保留", ["ev_0001"], entity_id="requirement_001")
        standard_items = [
            {
                "id": "std_0001",
                "field_id": "fld_0001",
                "semantic_key": "custom.design_note",
                "group_id": "requirement_001",
                "evidence_refs": ["ev_0001"],
            }
        ]
        groups = build_field_groups(
            {"requirement_001": {"entity_id": "requirement_001", "entity_type": "requirement", "index": 1, "linked_table_ids": []}},
            {"fld_0001": custom},
            standard_items,
        )
        proposals = build_taxonomy_proposals(
            {"fld_0001": custom},
            [Evidence("ev_0001", "设计注意：保留", 1, ["pdf_text"], "missing", ["span_0001"])],
        )

        self.assertEqual(groups[0]["fields"][0]["standard_item_id"], "std_0001")
        self.assertEqual(proposals[0]["field"], "custom.design_note")
        self.assertTrue(proposals[0]["required_review"])

    def test_field_groups_match_standard_items_by_field_id_when_evidence_is_shared(self) -> None:
        first = _field("fld_0001", "manufacturer.name", "生产者", "甲公司", ["ev_shared"], entity_id="manufacturer_001")
        second = _field("fld_0002", "manufacturer.address", "地址", "深圳", ["ev_shared"], entity_id="manufacturer_001")
        groups = build_field_groups(
            {"manufacturer_001": {"entity_id": "manufacturer_001", "entity_type": "manufacturer", "index": 1, "linked_table_ids": []}},
            {first.field_id: first, second.field_id: second},
            [
                {"id": "std_0001", "field_id": "fld_0001", "semantic_key": first.semantic_key, "group_id": first.entity_id, "evidence_refs": ["ev_shared"]},
                {"id": "std_0002", "field_id": "fld_0002", "semantic_key": second.semantic_key, "group_id": second.entity_id, "evidence_refs": ["ev_shared"]},
            ],
        )

        self.assertEqual([item["standard_item_id"] for item in groups[0]["fields"]], ["std_0001", "std_0002"])


def _field(
    field_id: str,
    semantic_key: str,
    display_name: str,
    raw_value: str,
    evidence_refs: list[str],
    entity_id: str | None = "product_001",
) -> CompiledField:
    return CompiledField(
        field_id=field_id,
        semantic_key=semantic_key,
        display_name=display_name,
        field_type="string",
        raw_value=raw_value,
        clean_value=raw_value,
        normalized_value=raw_value,
        value_hash="sha256:" + "0" * 64,
        status="verified",
        criticality="critical",
        confidence={"overall": 0.99},
        risk_level="info",
        review_required=False,
        section_id="sec_label_text",
        entity_id=entity_id,
        table_id=None,
        row_key=None,
        evidence_refs=evidence_refs,
    )


if __name__ == "__main__":
    unittest.main()
