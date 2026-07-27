import unittest

from document_parser.structures import build_repair_plan
from document_parser.repair_artifacts import (
    build_repair_agent_candidates,
    build_repair_attempts_artifact,
    build_repair_plan_patches,
    build_repair_trace_artifact,
    build_repaired_source_layers,
)
from document_parser.models import ExtractionPlan, FieldPlan, SpanRange, ValueSource


class RepairArtifactTests(unittest.TestCase):
    def test_repair_trace_summarizes_rounds_and_final_audit_state(self) -> None:
        trace = build_repair_trace_artifact(
            [
                {
                    "round": 1,
                    "status": "repaired_and_recompiled",
                    "audit_finding_count": 1,
                    "audit_findings": [{"finding_id": "af_0001"}],
                    "attempts": [{"status": "applied"}],
                    "compiled_after_repair": True,
                },
                {
                    "round": 2,
                    "status": "passed_after_repair",
                    "audit_finding_count": 0,
                    "audit_findings": [],
                    "attempts": [],
                    "compiled_after_repair": False,
                },
            ],
            max_repair_rounds=2,
        )

        self.assertEqual(trace["status"], "pass")
        self.assertEqual(trace["round_count"], 2)
        self.assertEqual(trace["attempt_count"], 1)
        self.assertEqual(trace["applied_attempt_count"], 1)
        self.assertEqual(trace["final_audit_finding_count"], 0)
        self.assertEqual(trace["validation_after_repair"], "final_pipeline_validation")

    def test_execute_plan_repair_outputs_pending_attempt_and_agent_template(self) -> None:
        repair_plan = {
            "repair_mode": "execute_plan",
            "actions": [
                {
                    "action_id": "repair_action_0001",
                    "target_type": "anchor",
                    "target_id": "span_0001",
                    "issue_type": "missing_anchor",
                    "recommended_agent": "anchor_agent",
                    "expected_output": "field/table assignment using existing source span",
                    "acceptance_gate": "structure_audit_anchor_coverage_pass",
                }
            ],
        }

        attempts = build_repair_attempts_artifact([], repair_plan)
        candidates = build_repair_agent_candidates(repair_plan)
        repaired_source_layers = build_repaired_source_layers(
            {"status": "review_required", "source_issue_count": 1, "source_issues": [{"issue_type": "source_bbox_missing"}]},
            attempts,
        )

        self.assertEqual(attempts["status"], "pending_agent_repair")
        self.assertEqual(attempts["attempt_count"], 1)
        self.assertEqual(attempts["attempts"][0]["action_id"], "repair_action_0001")
        self.assertEqual(attempts["attempts"][0]["reason"], "requires_agent_patch_not_available_in_current_round")
        self.assertEqual(candidates["status"], "pending_agent_fill")
        self.assertEqual(candidates["candidates"][0]["candidate_template"], {"items": [], "groups": [], "tables": [], "lists": []})
        self.assertEqual(repaired_source_layers["status"], "not_modified")
        self.assertFalse(repaired_source_layers["repair_applied"])

    def test_applied_plan_repair_does_not_modify_source_layers(self) -> None:
        repaired_source_layers = build_repaired_source_layers(
            {"status": "pass", "source_issue_count": 0, "source_issues": []},
            {
                "status": "attempted",
                "attempts": [
                    {
                        "attempt_id": "repair_attempt_0001",
                        "status": "applied",
                        "reason": "trimmed_target_field_before_sibling_label",
                    }
                ],
            },
        )

        self.assertEqual(repaired_source_layers["status"], "source_layers_not_modified_plan_repair_applied")
        self.assertTrue(repaired_source_layers["repair_applied"])
        self.assertTrue(repaired_source_layers["repair_attempted"])
        self.assertEqual(repaired_source_layers["reason"], "extraction_plan_repaired_and_recompiled")

    def test_repair_plan_routes_review_and_rejected_agent_items(self) -> None:
        plan = build_repair_plan(
            [],
            [],
            {"missing_anchor_issues": []},
            max_rounds=2,
            review_items=[
                {
                    "item_index": 2,
                    "semantic_key": "custom.warning",
                    "span_id": "span_0002",
                    "reason": "agent_candidate_low_confidence",
                    "confidence": 0.70,
                }
            ],
            rejected_agent_items=[
                {
                    "item_index": 3,
                    "reason": "text_does_not_match_source_span",
                    "item": {"span_id": "span_0003"},
                    "actual_source_text": "原文",
                }
            ],
        )

        self.assertEqual(plan["status"], "review_required")
        self.assertEqual(len(plan["actions"]), 2)
        self.assertEqual(plan["actions"][0]["target_id"], "review_item_0002")
        self.assertEqual(plan["actions"][0]["input_artifact"], "review_items.json")
        self.assertEqual(plan["actions"][0]["acceptance_gate"], "agent_candidate_span_validation_and_compiler_pass")
        self.assertEqual(plan["actions"][1]["target_id"], "rejected_item_0003")
        self.assertEqual(plan["actions"][1]["input_artifact"], "rejected_agent_items.json")

    def test_repair_plan_routes_structure_audit_diagnostics(self) -> None:
        plan = build_repair_plan(
            [],
            [],
            {
                "sequence_gap_count": 1,
                "required_prefix_issues": [{"source": {"field_id": "fld_0001"}}],
                "container_duplicate_issues": [{"source": {"field_id": "fld_0002"}}],
                "agent_override_issues": [{"source": {"field_id": "fld_0003"}}],
                "duplicate_coverage_issues": [{"source": {"span_id": "span_0004"}}],
            },
            max_rounds=2,
        )

        by_issue_type = {action["issue_type"]: action for action in plan["actions"]}
        self.assertEqual(plan["status"], "review_required")
        self.assertEqual(by_issue_type["content_sequence_gap"]["recommended_agent"], "anchor_agent")
        self.assertEqual(by_issue_type["required_prefix_issue"]["target_id"], "fld_0001")
        self.assertEqual(by_issue_type["container_duplicate_issue"]["recommended_agent"], "dedupe_agent")
        self.assertEqual(by_issue_type["agent_override_issue"]["target_type"], "agent_candidate")
        self.assertEqual(by_issue_type["duplicate_coverage_issue"]["target_id"], "span_0004")

    def test_repair_plan_patches_record_plan_boundary_changes_only(self) -> None:
        before_plan = ExtractionPlan(
            plan_id="plan_001",
            schema_id="schema_dynamic_001",
            fields=[
                FieldPlan(
                    field_plan_id="fp_0001",
                    semantic_key="product.ingredients",
                    display_name="配料",
                    field_type="long_text",
                    section_id="sec_label_text",
                    entity_id="product_001",
                    value_source=ValueSource("span_ranges", [SpanRange("span_0001", 0, 20)]),
                    criticality="critical",
                    confidence={"boundary_confidence": 0.70},
                    boundary={"end_reason": "line_end"},
                )
            ],
        )
        after_plan = ExtractionPlan(
            plan_id="plan_001",
            schema_id="schema_dynamic_001",
            fields=[
                FieldPlan(
                    field_plan_id="fp_0001",
                    semantic_key="product.ingredients",
                    display_name="配料",
                    field_type="long_text",
                    section_id="sec_label_text",
                    entity_id="product_001",
                    value_source=ValueSource("span_ranges", [SpanRange("span_0001", 0, 5)]),
                    criticality="critical",
                    confidence={"boundary_confidence": 0.90},
                    boundary={"repair": "trimmed_at_sibling_label"},
                )
            ],
        )

        patches = build_repair_plan_patches(
            before_plan,
            after_plan,
            {
                "attempts": [
                    {
                        "status": "applied",
                        "details": {"target_field_plan_id": "fp_0001"},
                    }
                ]
            },
        )

        self.assertEqual(patches["status"], "applied")
        self.assertEqual(patches["patch_count"], 1)
        self.assertEqual(patches["patches"][0]["operation"], "adjust_field_boundary")
        self.assertEqual(patches["patches"][0]["previous_value_source"]["ranges"][0]["end_offset"], 20)
        self.assertEqual(patches["patches"][0]["new_value_source"]["ranges"][0]["end_offset"], 5)
        self.assertEqual(patches["patches"][0]["attempts"][0]["status"], "applied")


if __name__ == "__main__":
    unittest.main()
