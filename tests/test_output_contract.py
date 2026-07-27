import copy
import unittest

from document_parser.output_contract import _joined_evidence_text, build_output_contract_validation_report


class OutputContractTests(unittest.TestCase):
    def test_character_atom_evidence_uses_compiler_geometry_joining(self) -> None:
        evidence = {
            "ev_1": {
                "source_text": "SC113",
                "page": 1,
                "extraction_methods": ["pdf_char_atom"],
                "bbox_pdf": {"x": 10, "y": 20, "width": 20, "height": 10},
            },
            "ev_2": {
                "source_text": "5109",
                "page": 1,
                "extraction_methods": ["pdf_char_atom"],
                "bbox_pdf": {"x": 35, "y": 20, "width": 16, "height": 10},
            },
            "ev_3": {
                "source_text": "地址：深圳",
                "page": 1,
                "extraction_methods": ["pdf_char_atom"],
                "bbox_pdf": {"x": 10, "y": 40, "width": 40, "height": 10},
            },
        }

        self.assertEqual(_joined_evidence_text(["ev_1", "ev_2", "ev_3"], evidence), "SC113 5109\n地址：深圳")

    def test_passes_for_complete_result_contract(self) -> None:
        report = build_output_contract_validation_report(_valid_result())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failed_count"], 0)

    def test_fails_when_standard_item_references_missing_evidence(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["standard_items"][0]["evidence_refs"] = ["ev_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("standard_item_refs", failed_types)

    def test_fails_when_comparison_profile_is_missing(self) -> None:
        result = _valid_result()
        del result["metadata"]["standard_artifacts"]["standard_items"][0]["comparison_profile"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("standard_item_contract", failed_types)
        self.assertIn("standard_item_comparison_profile", failed_types)

    def test_fails_when_comparison_index_misses_required_item(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["comparison_index"]["entries"] = []

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("comparison_index_contract", failed_types)

    def test_fails_when_job_status_is_missing(self) -> None:
        result = _valid_result()
        del result["job"]["status"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("job_contract", failed_types)

    def test_fails_when_document_hash_or_page_count_is_invalid(self) -> None:
        result = _valid_result()
        result["document"]["file_hash"] = "not-a-sha256"
        result["document"]["page_count"] = 2

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("document_contract", failed_types)

    def test_fails_when_created_at_is_missing(self) -> None:
        result = _valid_result()
        del result["metadata"]["created_at"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("metadata_created_at", failed_types)

    def test_fails_when_no_guessing_policy_is_false(self) -> None:
        result = _valid_result()
        result["metadata"]["no_guessing"] = False
        result["metadata"]["json_export"]["no_guessing"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("no_guessing_policy", failed_types)
        self.assertIn("json_export_contract", failed_types)

    def test_fails_when_json_export_manifest_is_missing_or_incomplete(self) -> None:
        result = _valid_result()
        del result["metadata"]["json_export"]["contract_checks"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("json_export_contract", failed_types)

    def test_fails_when_json_export_root_keys_do_not_match_final_json(self) -> None:
        result = _valid_result()
        result["metadata"]["json_export"]["root_keys"] = ["job", "document"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("json_export_contract", failed_types)

    def test_fails_when_generated_schema_field_type_is_invalid(self) -> None:
        result = _valid_result()
        result["generated_schema"]["field_definitions"][0]["field_type"] = "freeform"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("generated_schema_contract", failed_types)

    def test_fails_when_generated_schema_source_span_is_unknown(self) -> None:
        result = _valid_result()
        result["generated_schema"]["field_definitions"][0]["source_span_ids"] = ["span_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("generated_schema_source_refs", failed_types)

    def test_fails_when_evidence_id_is_duplicated(self) -> None:
        result = _valid_result()
        result["evidence"].append(copy.deepcopy(result["evidence"][0]))

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("evidence_unique_ids", failed_types)

    def test_fails_when_evidence_is_missing_source_text(self) -> None:
        result = _valid_result()
        del result["evidence"][0]["source_text"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("evidence_contract", failed_types)

    def test_fails_when_available_evidence_has_no_bbox_coordinates(self) -> None:
        result = _valid_result()
        del result["evidence"][0]["bbox_normalized"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("evidence_bbox_status_consistency", failed_types)

    def test_fails_when_available_evidence_bbox_normalized_is_out_of_range(self) -> None:
        result = _valid_result()
        result["evidence"][0]["bbox_normalized"]["x2"] = 1.2

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("evidence_bbox_status_consistency", failed_types)

    def test_fails_when_ocr_line_available_bbox_has_no_coordinates(self) -> None:
        result = _valid_result()
        ocr_line = result["metadata"]["source_layers"]["layers"]["ocr"]["lines"][0]
        del ocr_line["bbox_pdf"]
        del ocr_line["bbox_normalized"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("ocr_line_contract", failed_types)

    def test_fails_when_ocr_block_references_missing_line(self) -> None:
        result = _valid_result()
        result["metadata"]["source_layers"]["layers"]["ocr"]["blocks"][0]["line_ids"] = ["ocr_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("ocr_block_refs", failed_types)

    def test_fails_when_region_source_span_is_unknown(self) -> None:
        result = _valid_result()
        result["extracted_data"]["regions"][0]["source_span_ids"] = ["span_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("region_source_refs", failed_types)
        self.assertIn("region_vdg_refs", failed_types)

    def test_fails_when_region_field_ref_is_unknown(self) -> None:
        result = _valid_result()
        result["extracted_data"]["regions"][0]["fields"] = ["fld_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("region_artifact_refs", failed_types)

    def test_fails_when_uncertain_panel_assignment_is_not_review_required(self) -> None:
        result = _valid_result()
        region = result["extracted_data"]["regions"][0]
        region["assignment_status"] = "uncertain"
        region["review_required"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("region_contract", failed_types)

    def test_fails_when_vdg_edge_references_unknown_node(self) -> None:
        result = _valid_result()
        result["metadata"]["visual_document_graph"]["edges"][0]["target_node_id"] = "node_missing"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("visual_document_graph_contract", failed_types)

    def test_fails_when_high_risk_has_no_review_task(self) -> None:
        result = _valid_result()
        result["review_tasks"] = []

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("high_risk_review_tasks", failed_types)

    def test_fails_when_field_raw_value_does_not_match_evidence_text(self) -> None:
        result = _valid_result()
        result["extracted_data"]["fields"]["fld_0001"]["raw_value"] = "品名：酸奶"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_text_from_evidence", failed_types)

    def test_fails_when_field_confidence_is_missing(self) -> None:
        result = _valid_result()
        del result["extracted_data"]["fields"]["fld_0001"]["confidence"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_contract", failed_types)

    def test_fails_when_critical_low_confidence_field_is_not_review_required(self) -> None:
        result = _valid_result()
        field = result["extracted_data"]["fields"]["fld_0001"]
        field["confidence"]["overall"] = 0.90
        field["review_required"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_contract", failed_types)

    def test_fails_when_critical_low_confidence_field_has_no_high_risk_review_route(self) -> None:
        result = _valid_result()
        field = result["extracted_data"]["fields"]["fld_0001"]
        field["confidence"]["overall"] = 0.90
        field["status"] = "manual_review_required"
        field["risk_level"] = "high"
        field["review_required"] = True

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("critical_low_confidence_review_tasks", failed_types)

    def test_fails_when_field_normalization_key_is_missing(self) -> None:
        result = _valid_result()
        del result["extracted_data"]["fields"]["fld_0001"]["normalization"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_contract", failed_types)
        self.assertIn("field_normalization_contract", failed_types)

    def test_fails_when_changed_field_value_has_no_normalization_rule(self) -> None:
        result = _valid_result()
        result["extracted_data"]["fields"]["fld_0001"]["normalization"] = []

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_normalization_contract", failed_types)

    def test_fails_when_normalized_field_is_marked_verified_info(self) -> None:
        result = _valid_result()
        field = result["extracted_data"]["fields"]["fld_0001"]
        field["status"] = "verified"
        field["risk_level"] = "info"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_normalization_contract", failed_types)

    def test_fails_when_uncertain_normalized_field_is_not_review_required(self) -> None:
        result = _valid_result()
        field = result["extracted_data"]["fields"]["fld_0001"]
        field["criticality"] = "non_critical"
        field["confidence"]["overall"] = 0.90
        field["review_required"] = False
        entity_field = result["extracted_data"]["entities"]["product_001"]["fields"]["name"]
        entity_field["criticality"] = "non_critical"
        entity_field["confidence"]["overall"] = 0.90
        entity_field["review_required"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_normalization_contract", failed_types)

    def test_fails_when_critical_field_missing_bbox_has_no_high_risk_review(self) -> None:
        result = _valid_result()
        result["evidence"][0]["bbox_status"] = "missing"
        del result["evidence"][0]["bbox_pdf"]
        del result["evidence"][0]["bbox_normalized"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("critical_field_bbox_risk", failed_types)

    def test_fails_when_non_critical_field_missing_bbox_has_no_risk(self) -> None:
        result = _valid_result()
        field = result["extracted_data"]["fields"]["fld_0002"]
        field["criticality"] = "non_critical"
        result["evidence"][3]["bbox_status"] = "missing"
        del result["evidence"][3]["bbox_pdf"]
        del result["evidence"][3]["bbox_normalized"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_missing_bbox_risks", failed_types)

    def test_fails_when_missing_item_has_no_high_risk_review_route(self) -> None:
        result = _valid_result()
        result["risks"] = [
            risk
            for risk in result["risks"]
            if not (risk.get("target_type") == "missing_field" and risk.get("target_id") == "missing_field_0001")
        ]
        result["review_tasks"] = []

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("missing_item_risk_routes", failed_types)

    def test_fails_when_review_required_table_has_no_risk_route(self) -> None:
        result = _valid_result()
        table = result["metadata"]["standard_artifacts"]["tables"][0]
        table["rows"] = []
        table["status"] = "manual_review_required"
        table["risk_level"] = "high"
        table["review_required"] = True

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_structure_risk_routes", failed_types)

    def test_fails_when_required_cross_validation_check_is_missing(self) -> None:
        result = _valid_result()
        result["validation"] = [
            check for check in result["validation"] if check["check_type"] != "format_check"
        ]
        result["cross_validation"]["checks"] = copy.deepcopy(result["validation"])

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("cross_validation_required_checks", failed_types)

    def test_fails_when_cross_validation_checks_do_not_mirror_validation(self) -> None:
        result = _valid_result()
        result["cross_validation"]["checks"] = result["cross_validation"]["checks"][:-1]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("cross_validation_required_checks", failed_types)

    def test_fails_when_failed_validation_has_no_risk_route(self) -> None:
        result = _valid_result()
        failed_check = {
            "validation_id": "val_failed_format_0001",
            "target_id": "fld_0001",
            "check_type": "format_check",
            "result": "failed",
            "severity": "high",
            "message": "格式错误。",
            "evidence_refs": ["ev_0001"],
        }
        result["validation"].append(failed_check)
        result["cross_validation"]["checks"] = copy.deepcopy(result["validation"])

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("failed_validation_risk_routes", failed_types)

    def test_fails_when_low_confidence_ocr_issue_has_no_risk_route(self) -> None:
        result = _valid_result()
        result["cross_validation"]["source_consistency"] = {
            "status": "pass",
            "pdf_text_span_count": 4,
            "ocr_line_count": 2,
            "matched_ocr_line_count": 1,
            "issue_count": 1,
            "issues": [
                {
                    "issue_id": "source_consistency_issue_0001",
                    "issue_type": "ocr_low_confidence",
                    "severity": "low",
                    "page": 1,
                    "message": "OCR line confidence is below threshold.",
                    "detail": {"ocr_line_id": "ocr_0002", "confidence": 0.5, "threshold": 0.8},
                }
            ],
        }

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("source_consistency_issue_risk_routes", failed_types)

    def test_allows_review_required_field_with_empty_normalized_value(self) -> None:
        result = _valid_result()
        field = result["extracted_data"]["fields"]["fld_0001"]
        field["raw_value"] = "品名"
        field["clean_value"] = ""
        field["normalized_value"] = ""
        field["value_hash"] = "sha256:" + "0" * 64
        field["status"] = "manual_review_required"
        field["confidence"]["overall"] = 0.85
        field["risk_level"] = "high"
        field["review_required"] = True
        result["evidence"][0]["source_text"] = "品名"
        result["metadata"]["standard_artifacts"]["standard_items"][0]["text"] = "品名"
        result["metadata"]["standard_artifacts"]["auto_ingest_candidates"]["candidates"][0]["text"] = "品名"
        entity_field = result["extracted_data"]["entities"]["product_001"]["fields"]["name"]
        entity_field["value"] = "品名"
        entity_field["normalized_value"] = ""
        entity_field["status"] = "manual_review_required"
        entity_field["confidence"]["overall"] = 0.85
        entity_field["risk_level"] = "high"
        entity_field["review_required"] = True
        result["risks"].append(
            {
                "risk_id": "risk_0003",
                "target_type": "field",
                "target_id": "fld_0001",
                "risk_level": "high",
                "risk_type": "manual_review_required",
                "message": "关键字段置信度低于0.95",
                "evidence_refs": ["ev_0001"],
            }
        )
        result["review_tasks"].append(
            {
                "task_id": "review_0002",
                "target_type": "field",
                "target_id": "fld_0001",
                "risk_level": "high",
                "reason": "关键字段置信度低于0.95",
                "required": True,
                "evidence_refs": ["ev_0001"],
            }
        )

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "pass")

    def test_fails_when_entity_field_points_to_wrong_entity(self) -> None:
        result = _valid_result()
        entity_field = result["extracted_data"]["entities"]["content_item_001"]["fields"]["name"]
        entity_field["field_id"] = "fld_0001"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("entity_refs", failed_types)

    def test_fails_when_low_confidence_entity_is_not_uncertain(self) -> None:
        result = _valid_result()
        entity = result["extracted_data"]["entities"]["content_item_001"]
        entity["confidence"]["entity_linking_confidence"] = 0.88
        entity["confidence"]["overall"] = 0.88
        entity["status"] = "verified"
        entity["review_required"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("entity_contract", failed_types)

    def test_fails_when_entity_table_link_mismatches_table_owner(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["tables"][0]["linked_entity_id"] = "product_001"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("entity_refs", failed_types)

    def test_fails_when_revision_after_is_not_current_standard(self) -> None:
        result = _valid_result()
        result["extracted_data"]["revision_blocks"] = _valid_revision_blocks()
        result["extracted_data"]["revision_blocks"][1]["is_current_standard"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("revision_block_contract", failed_types)

    def test_fails_when_revision_field_is_reused_across_before_and_after(self) -> None:
        result = _valid_result()
        result["extracted_data"]["revision_blocks"] = _valid_revision_blocks()
        result["extracted_data"]["revision_blocks"][1]["fields"][0]["field_id"] = "fld_0002"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("revision_block_refs", failed_types)

    def test_fails_when_pending_revision_assignment_is_not_high_risk(self) -> None:
        result = _valid_result()
        result["extracted_data"]["revision_blocks"] = _valid_revision_blocks()
        pending = result["extracted_data"]["revision_blocks"][0]
        pending["fields"] = []
        pending["assignment_status"] = "region_detected_field_assignment_pending"
        pending["status"] = "verified"
        pending["risk_level"] = "info"
        pending["review_required"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("revision_block_contract", failed_types)

    def test_fails_when_standard_item_text_does_not_match_evidence_text(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["standard_items"][0]["text"] = "品名：酸奶"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("standard_item_text_from_evidence", failed_types)

    def test_fails_when_auto_ingest_candidate_rewrites_standard_item_text(self) -> None:
        result = _valid_result()
        candidate = result["metadata"]["standard_artifacts"]["auto_ingest_candidates"]["candidates"][0]
        candidate["text"] = "品名：酸奶"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("auto_ingest_text_matches_standard_item", failed_types)

    def test_fails_when_table_references_missing_evidence(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["tables"][0]["evidence_refs"] = ["ev_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_evidence_refs", failed_types)

    def test_fails_when_table_confidence_is_missing(self) -> None:
        result = _valid_result()
        del result["metadata"]["standard_artifacts"]["tables"][0]["confidence"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_contract", failed_types)

    def test_fails_when_critical_low_confidence_table_is_not_review_required(self) -> None:
        result = _valid_result()
        table = result["metadata"]["standard_artifacts"]["tables"][0]
        table["confidence"]["table_structure_confidence"] = 0.90
        table["review_required"] = False

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_contract", failed_types)

    def test_fails_when_available_table_bbox_has_no_coordinates(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["tables"][0]["bbox_status"] = "available"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_contract", failed_types)

    def test_fails_when_table_column_has_no_id(self) -> None:
        result = _valid_result()
        del result["metadata"]["standard_artifacts"]["tables"][0]["columns"][0]["column_id"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_contract", failed_types)

    def test_fails_when_table_row_has_no_evidence_refs(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["tables"][0]["rows"][0]["evidence_refs"] = []

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_contract", failed_types)

    def test_fails_when_table_cell_text_does_not_match_evidence_text(self) -> None:
        result = _valid_result()
        cell = result["metadata"]["standard_artifacts"]["tables"][0]["rows"][0]["cells"][0]
        cell["raw_value"] = "蛋白质"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_cell_text_from_evidence", failed_types)

    def test_fails_when_table_cell_has_no_evidence_refs(self) -> None:
        result = _valid_result()
        cell = result["metadata"]["standard_artifacts"]["tables"][0]["rows"][0]["cells"][0]
        cell["evidence_refs"] = []

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("table_contract", failed_types)

    def test_allows_empty_table_cell_raw_value_when_evidence_is_present(self) -> None:
        result = _valid_result()
        cell = result["metadata"]["standard_artifacts"]["tables"][0]["rows"][0]["cells"][1]
        cell["raw_value"] = ""
        cell["normalized_value"] = ""

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "pass")

    def test_fails_when_requirement_references_missing_evidence(self) -> None:
        result = _valid_result()
        result["extracted_data"]["requirements"][0]["evidence_refs"] = ["ev_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("requirement_evidence_refs", failed_types)

    def test_fails_when_requirement_text_does_not_match_evidence_text(self) -> None:
        result = _valid_result()
        result["extracted_data"]["requirements"][0]["requirement_text"] = "不得上市销售"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("requirement_text_from_evidence", failed_types)

    def test_fails_when_requirement_has_no_evidence_refs(self) -> None:
        result = _valid_result()
        result["extracted_data"]["requirements"][0]["evidence_refs"] = []

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("requirement_contract", failed_types)

    def test_fails_when_requirement_type_is_invalid(self) -> None:
        result = _valid_result()
        result["extracted_data"]["requirements"][0]["requirement_type"] = "design_requirement"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("requirement_contract", failed_types)

    def test_fails_when_requirement_verification_is_not_mvp_status(self) -> None:
        result = _valid_result()
        result["extracted_data"]["requirements"][0]["verification_status"] = "verified"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("requirement_contract", failed_types)

    def test_fails_when_field_group_references_unknown_standard_item(self) -> None:
        result = _valid_result()
        group_field = result["metadata"]["standard_artifacts"]["field_groups"][0]["fields"][0]
        group_field["standard_item_id"] = "std_missing"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_group_refs", failed_types)

    def test_fails_when_field_group_text_rewrites_standard_item_text(self) -> None:
        result = _valid_result()
        group_field = result["metadata"]["standard_artifacts"]["field_groups"][0]["fields"][0]
        group_field["text"] = "内容物 1：错误名称"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("field_group_text_matches_standard_item", failed_types)

    def test_fails_when_list_item_references_unknown_group(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["lists"][0]["items"][0]["group_id"] = "content_item_missing"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("list_group_refs", failed_types)

    def test_fails_when_list_item_text_rewrites_group_container_text(self) -> None:
        result = _valid_result()
        result["metadata"]["standard_artifacts"]["lists"][0]["items"][0]["text"] = "内容物 1：错误名称"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("list_group_refs", failed_types)

    def test_fails_when_risk_references_missing_evidence(self) -> None:
        result = _valid_result()
        result["risks"][0]["evidence_refs"] = ["ev_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("risk_evidence_refs", failed_types)

    def test_fails_when_risk_points_to_unknown_stable_target(self) -> None:
        result = _valid_result()
        result["risks"][0]["target_type"] = "field"
        result["risks"][0]["target_id"] = "fld_missing"
        result["review_tasks"][0]["target_type"] = "field"
        result["review_tasks"][0]["target_id"] = "fld_missing"

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("risk_target_refs", failed_types)

    def test_fails_when_risk_uses_unknown_target_type(self) -> None:
        result = _valid_result()
        result["risks"].append(
            {
                "risk_id": "risk_unknown_target_type",
                "target_type": "unknown_artifact",
                "target_id": "whatever",
                "risk_level": "low",
                "risk_type": "unknown_route",
                "message": "Unknown risk target type.",
                "evidence_refs": [],
            }
        )

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("risk_target_refs", failed_types)

    def test_passes_when_risk_points_to_table_parser_candidate_table_layer(self) -> None:
        result = _valid_result()
        result["metadata"]["table_parser"] = {
            "table_layers": {
                "tables": [
                    {
                        "table_layer_id": "tl_tbl_0001",
                        "parser": "text_span_nutrition",
                    }
                ]
            }
        }
        result["risks"].append(
            {
                "risk_id": "risk_table_layer_0001",
                "target_type": "table",
                "target_id": "tl_tbl_0001",
                "risk_level": "low",
                "risk_type": "nutrition_table_rows_incomplete",
                "message": "营养成分表没有恢复出数据行。",
                "evidence_refs": [],
            }
        )

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "pass")

    def test_fails_when_source_consistency_risk_points_to_unknown_issue(self) -> None:
        result = _valid_result()
        result["cross_validation"]["source_consistency"]["issues"] = [
            {
                "issue_id": "source_consistency_issue_0001",
                "issue_type": "ocr_low_confidence",
                "severity": "low",
                "page": 1,
                "message": "OCR line confidence is below threshold.",
                "detail": {"ocr_line_id": "ocr_0001", "confidence": 0.5, "threshold": 0.8},
            }
        ]
        result["risks"].append(
            {
                "risk_id": "risk_source_consistency_0001",
                "target_type": "source_consistency",
                "target_id": "source_consistency_issue_0001",
                "risk_level": "low",
                "risk_type": "ocr_low_confidence",
                "message": "OCR line confidence is below threshold.",
                "evidence_refs": [],
            }
        )
        result["risks"].append(
            {
                "risk_id": "risk_source_consistency_missing",
                "target_type": "source_consistency",
                "target_id": "source_consistency_issue_missing",
                "risk_level": "low",
                "risk_type": "ocr_low_confidence",
                "message": "OCR line confidence is below threshold.",
                "evidence_refs": [],
            }
        )

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("risk_target_refs", failed_types)

    def test_fails_when_source_layer_risk_points_to_unknown_issue(self) -> None:
        result = _valid_result()
        result["metadata"]["source_layers"]["source_issues"] = [
            {
                "issue_id": "source_issue_0001",
                "issue_type": "source_bbox_missing",
                "severity": "medium",
                "message": "Some source spans are missing bbox.",
            }
        ]
        result["risks"].append(
            {
                "risk_id": "risk_source_layer_missing",
                "target_type": "source_layer",
                "target_id": "source_issue_missing",
                "risk_level": "medium",
                "risk_type": "source_bbox_missing",
                "message": "Some source spans are missing bbox.",
                "evidence_refs": [],
            }
        )

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("risk_target_refs", failed_types)

    def test_fails_when_document_risk_points_to_unknown_document_artifact(self) -> None:
        result = _valid_result()
        result["risks"].append(
            {
                "risk_id": "risk_document_missing",
                "target_type": "document",
                "target_id": "missing_artifact",
                "risk_level": "medium",
                "risk_type": "page_image_render_failed",
                "message": "页面渲染未完成。",
                "evidence_refs": [],
            }
        )

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("risk_target_refs", failed_types)

    def test_fails_when_review_task_references_missing_evidence(self) -> None:
        result = _valid_result()
        result["review_tasks"][0]["evidence_refs"] = ["ev_missing"]

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("review_task_evidence_refs", failed_types)

    def test_fails_when_review_task_points_to_unknown_stable_target(self) -> None:
        result = _valid_result()
        result["review_tasks"].append(
            {
                "task_id": "review_0002",
                "target_type": "field",
                "target_id": "fld_missing",
                "risk_level": "low",
                "required": False,
            }
        )

        report = build_output_contract_validation_report(result)

        self.assertEqual(report["status"], "review_required")
        failed_types = {check["check_type"] for check in report["checks"] if check["result"] == "failed"}
        self.assertIn("review_task_target_refs", failed_types)


def _valid_result() -> dict:
    result = {
        "job": {"job_id": "job_0001", "job_type": "standard_pdf_to_structured_json", "status": "completed_with_warnings"},
        "document": {
            "file_name": "sample.pdf",
            "file_hash": "sha256:" + "a" * 64,
            "page_count": 1,
            "page_sizes": [{"page": 1, "width": 595.28, "height": 841.89}],
            "parse_status": "completed_with_warnings",
        },
        "generated_schema": {
            "schema_id": "schema_dynamic_001",
            "auto_generated": True,
            "schema_version": "dynamic_v1",
            "sections": [
                {"section_id": "sec_document", "section_type": "document", "display_name": "文档"},
                {"section_id": "sec_label_text", "section_type": "label_text", "display_name": "标签文字内容"},
                {
                    "section_id": "sec_reg_0001",
                    "section_type": "package_panel",
                    "display_name": "第一唛",
                    "source_span_ids": ["span_0001"],
                },
            ],
            "entity_types": [
                {"entity_type": "product", "repeatable": False},
                {"entity_type": "content_item", "repeatable": True},
            ],
            "field_definitions": [
                {
                    "field_def_id": "fdef_0001",
                    "semantic_key": "product.name",
                    "semantic_key_type": "canonical",
                    "display_name": "品名",
                    "field_type": "string",
                    "criticality": "critical",
                    "repeatable": False,
                    "source_span_ids": ["span_0001"],
                },
                {
                    "field_def_id": "fdef_0002",
                    "semantic_key": "content_item.name",
                    "semantic_key_type": "canonical",
                    "display_name": "内容物名称",
                    "field_type": "string",
                    "criticality": "critical",
                    "repeatable": True,
                    "source_span_ids": ["span_0004"],
                },
            ],
            "table_definitions": [
                {
                    "table_type": "nutrition_facts",
                    "display_name": "营养成分表",
                    "criticality": "critical",
                    "repeatable": True,
                    "source_span_ids": ["span_0002"],
                }
            ],
            "requirement_definitions": [],
        },
        "extracted_data": {
            "fields": {
                "fld_0001": {
                    "field_id": "fld_0001",
                    "semantic_key": "product.name",
                    "display_name": "品名",
                    "field_type": "string",
                    "raw_value": "品名：牛奶",
                    "clean_value": "牛奶",
                    "normalized_value": "牛奶",
                    "normalization": ["remove_field_label"],
                    "value_hash": "sha256:" + "b" * 64,
                    "status": "normalized",
                    "criticality": "critical",
                    "confidence": {
                        "schema_confidence": 0.98,
                        "boundary_confidence": 0.98,
                        "entity_linking_confidence": 0.98,
                        "evidence_confidence": 1.0,
                        "format_validation_confidence": 1.0,
                        "overall": 0.98,
                    },
                    "risk_level": "low",
                    "review_required": False,
                    "entity_id": "product_001",
                    "evidence_refs": ["ev_0001"],
                },
                "fld_0002": {
                    "field_id": "fld_0002",
                    "semantic_key": "content_item.name",
                    "display_name": "内容物名称",
                    "field_type": "string",
                    "raw_value": "内容物 1：原味牛奶",
                    "clean_value": "内容物 1：原味牛奶",
                    "normalized_value": "内容物 1：原味牛奶",
                    "normalization": [],
                    "value_hash": "sha256:" + "c" * 64,
                    "status": "verified",
                    "criticality": "critical",
                    "confidence": {
                        "schema_confidence": 0.98,
                        "boundary_confidence": 0.98,
                        "entity_linking_confidence": 0.98,
                        "evidence_confidence": 1.0,
                        "format_validation_confidence": 1.0,
                        "overall": 0.98,
                    },
                    "risk_level": "info",
                    "review_required": False,
                    "entity_id": "content_item_001",
                    "evidence_refs": ["ev_0004"],
                }
            },
            "entities": {
                "product_001": {
                    "entity_id": "product_001",
                    "entity_type": "product",
                    "index": 1,
                    "fields": {
                        "name": {
                            "field_id": "fld_0001",
                            "semantic_key": "product.name",
                            "value": "品名：牛奶",
                            "normalized_value": "牛奶",
                            "status": "normalized",
                            "criticality": "critical",
                            "confidence": {
                                "schema_confidence": 0.98,
                                "boundary_confidence": 0.98,
                                "entity_linking_confidence": 0.98,
                                "evidence_confidence": 1.0,
                                "format_validation_confidence": 1.0,
                                "overall": 0.98,
                            },
                            "risk_level": "low",
                            "review_required": False,
                            "evidence_refs": ["ev_0001"],
                        }
                    },
                    "linked_table_ids": [],
                    "evidence_refs": ["ev_0001"],
                    "status": "verified",
                    "confidence": {"entity_linking_confidence": 0.98, "overall": 0.98},
                    "risk_level": "low",
                    "review_required": False,
                },
                "content_item_001": {
                    "entity_id": "content_item_001",
                    "entity_type": "content_item",
                    "index": 1,
                    "fields": {
                        "name": {
                            "field_id": "fld_0002",
                            "semantic_key": "content_item.name",
                            "value": "内容物 1：原味牛奶",
                            "normalized_value": "内容物 1：原味牛奶",
                            "status": "verified",
                            "criticality": "critical",
                            "confidence": {
                                "schema_confidence": 0.98,
                                "boundary_confidence": 0.98,
                                "entity_linking_confidence": 0.98,
                                "evidence_confidence": 1.0,
                                "format_validation_confidence": 1.0,
                                "overall": 0.98,
                            },
                            "risk_level": "info",
                            "review_required": False,
                            "evidence_refs": ["ev_0004"],
                        }
                    },
                    "linked_table_ids": ["tbl_0001"],
                    "evidence_refs": ["ev_0004", "ev_0002"],
                    "status": "verified",
                    "confidence": {"entity_linking_confidence": 0.98, "overall": 0.98},
                    "risk_level": "info",
                    "review_required": False,
                },
            },
            "missing_fields": [],
            "missing_tables": [],
            "regions": [
                {
                    "region_id": "reg_0001",
                    "panel_id": "reg_0001",
                    "panel_name": "第一唛",
                    "panel_type": "package_panel",
                    "region_type": "package_panel",
                    "display_name": "第一唛",
                    "page": 1,
                    "source_span_ids": ["span_0001"],
                    "bbox_status": "missing",
                    "confidence": 0.90,
                    "status": "verified",
                    "risk_level": "info",
                    "review_required": False,
                    "evidence_refs": ["ev_0001"],
                    "fields": ["fld_0001"],
                    "tables": ["tbl_0001"],
                    "entities": ["product_001", "content_item_001"],
                    "assignment_status": "assigned",
                }
            ],
            "requirements": [
                {
                    "requirement_id": "req_0001",
                    "requirement_type": "other",
                    "target": None,
                    "requirement_text": "设计注意：文字需清晰",
                    "status": "extracted",
                    "confidence": {
                        "schema_confidence": 0.98,
                        "boundary_confidence": 0.98,
                        "entity_linking_confidence": 0.98,
                        "evidence_confidence": 1.0,
                        "format_validation_confidence": 1.0,
                        "overall": 0.98,
                    },
                    "verification_status": "not_verified_in_mvp",
                    "risk_level": "info",
                    "review_required": False,
                    "evidence_refs": ["ev_0003"],
                }
            ],
            "revision_blocks": [],
        },
        "evidence": [
            {
                "evidence_id": "ev_0001",
                "source_text": "品名：牛奶",
                "page": 1,
                "extraction_methods": ["pdf_text"],
                "bbox_status": "available",
                "bbox_pdf": {
                    "x": 10,
                    "y": 20,
                    "width": 60,
                    "height": 10,
                    "unit": "pt",
                    "origin": "top_left",
                    "page_width": 200,
                    "page_height": 100,
                },
                "bbox_normalized": {"x1": 0.05, "y1": 0.2, "x2": 0.35, "y2": 0.3},
                "source_node_ids": ["span_0001"],
            },
            {
                "evidence_id": "ev_0002",
                "source_text": "能量 100kJ 1%",
                "page": 1,
                "extraction_methods": ["table_parser"],
                "bbox_status": "missing",
                "source_node_ids": ["span_0002"],
            },
            {
                "evidence_id": "ev_0003",
                "source_text": "设计注意：文字需清晰",
                "page": 1,
                "extraction_methods": ["pdf_text"],
                "bbox_status": "missing",
                "source_node_ids": ["span_0003"],
            },
            {
                "evidence_id": "ev_0004",
                "source_text": "内容物 1：原味牛奶",
                "page": 1,
                "extraction_methods": ["pdf_text"],
                "bbox_status": "available",
                "bbox_pdf": {
                    "x": 10,
                    "y": 60,
                    "width": 80,
                    "height": 10,
                    "unit": "pt",
                    "origin": "top_left",
                    "page_width": 200,
                    "page_height": 100,
                },
                "bbox_normalized": {"x1": 0.05, "y1": 0.6, "x2": 0.45, "y2": 0.7},
                "source_node_ids": ["span_0004"],
            },
        ],
        "cross_validation": {"checks": []},
        "coverage": {},
        "validation": [],
        "quality": {"auto_ingest_allowed": False},
        "risks": [
            {
                "risk_id": "risk_0001",
                "target_type": "missing_field",
                "target_id": "missing_field_0001",
                "risk_level": "high",
                "risk_type": "critical_field_missing",
                "message": "未从原文证据中抽取到 MVP 关键字段：配料。",
                "evidence_refs": [],
            },
            {
                "risk_id": "risk_0002",
                "target_type": "field",
                "target_id": "fld_0001",
                "risk_level": "low",
                "risk_type": "normalization_applied",
                "message": "字段生成了 normalized_value，原文已保留。",
                "evidence_refs": ["ev_0001"],
            },
        ],
        "review_tasks": [
            {
                "task_id": "review_0001",
                "target_type": "missing_field",
                "target_id": "missing_field_0001",
                "risk_level": "high",
                "reason": "未从原文证据中抽取到 MVP 关键字段：配料。",
                "required": True,
                "evidence_refs": [],
            }
        ],
        "metadata": {
            "no_guessing": True,
            "created_at": "2026-07-03T00:00:00+00:00",
            "json_export": {
                "schema_version": "mvp_final_json_v0.1",
                "media_type": "application/json",
                "encoding": "utf-8",
                "root_keys": [
                    "job",
                    "document",
                    "generated_schema",
                    "extracted_data",
                    "evidence",
                    "cross_validation",
                    "coverage",
                    "validation",
                    "quality",
                    "risks",
                    "review_tasks",
                    "metadata",
                ],
                "contract_checks": [
                    "machine_parseable_json",
                    "root_keys_present",
                    "evidence_refs_resolve",
                    "risk_targets_resolve",
                    "review_task_targets_resolve",
                    "no_guessing",
                ],
                "no_guessing": True,
                "primary_artifact": "result.json",
                "contract_artifact": "output_contract_validation_report.json",
                "schema_artifact": "schemas/final_result.schema.json",
            },
            "ocr_provider": "glm_ocr",
            "runtime_policy": {"status": "pass"},
            "pdf_character_atoms": [],
            "layout_candidates": {
                "artifact_version": "layout_candidates_v0.1",
                "status": "disabled",
                "table_candidates": [],
            },
            "layout_quality_report": {
                "report_version": "layout_quality_v0.1",
                "status": "disabled",
                "mode": "legacy",
            },
            "layout_candidate_acceptance_report": {
                "status": "disabled",
                "decisions": [],
                "unaccepted_candidate_count": 0,
            },
            "page_images": {},
            "visual_document_graph": {
                "graph_id": "vdg_0001",
                "graph_role": "final",
                "schema_version": "vdg_mvp_v0.1",
                "node_count": 3,
                "edge_count": 4,
                "nodes": [
                    {"node_id": "page_0001", "node_type": "page", "page": 1, "status": "structural"},
                    {"node_id": "span_0001", "node_type": "text_span", "page": 1, "status": "assigned_to_field", "source_span_ids": ["span_0001"]},
                    {
                        "node_id": "reg_0001",
                        "node_type": "region",
                        "page": 1,
                        "region_type": "package_panel",
                        "status": "structural",
                        "source_span_ids": ["span_0001"],
                    },
                ],
                "edges": [
                    {"edge_id": "edge_0001", "source_node_id": "page_0001", "target_node_id": "span_0001", "edge_type": "contains"},
                    {"edge_id": "edge_0002", "source_node_id": "page_0001", "target_node_id": "reg_0001", "edge_type": "contains"},
                    {"edge_id": "edge_0003", "source_node_id": "reg_0001", "target_node_id": "span_0001", "edge_type": "contains"},
                    {"edge_id": "edge_0004", "source_node_id": "span_0001", "target_node_id": "reg_0001", "edge_type": "belongs_to_region"},
                ],
            },
            "candidate_visual_document_graph": {
                "graph_id": "vdg_candidate_0001",
                "graph_role": "candidate_pre_agent",
                "schema_version": "vdg_mvp_v0.1",
                "node_count": 3,
                "edge_count": 4,
                "nodes": [
                    {"node_id": "page_0001", "node_type": "page", "page": 1, "status": "structural"},
                    {"node_id": "span_0001", "node_type": "text_span", "page": 1, "status": "unknown", "source_span_ids": ["span_0001"]},
                    {
                        "node_id": "reg_0001",
                        "node_type": "region",
                        "page": 1,
                        "region_type": "package_panel",
                        "status": "structural",
                        "source_span_ids": ["span_0001"],
                    },
                ],
                "edges": [
                    {"edge_id": "edge_0001", "source_node_id": "page_0001", "target_node_id": "span_0001", "edge_type": "contains"},
                    {"edge_id": "edge_0002", "source_node_id": "page_0001", "target_node_id": "reg_0001", "edge_type": "contains"},
                    {"edge_id": "edge_0003", "source_node_id": "reg_0001", "target_node_id": "span_0001", "edge_type": "contains"},
                    {"edge_id": "edge_0004", "source_node_id": "span_0001", "target_node_id": "reg_0001", "edge_type": "belongs_to_region"},
                ],
            },
            "vdg_quality_report": {
                "report_version": "vdg_quality_v0.1",
                "status": "pass",
                "source_span_coverage_rate": 1.0,
                "edge_ref_status": "pass",
                "issues": [],
                "checks": [],
            },
            "vdg_agent_context": {
                "context_version": "vdg_agent_context_v0.1",
                "vdg_quality_status": "pass",
                "agent_readiness": "usable",
                "candidate_field_groups": [],
                "table_candidates": [],
                "quality_issues": [],
            },
            "vdg_consumption_report": {
                "report_version": "vdg_consumption_v0.1",
                "status": "pass",
                "consumable_node_count": 1,
                "extracted_node_count": 1,
                "extracted_coverage_rate": 1.0,
                "status_counts": {"structural": 2, "extracted": 1},
            },
            "label_text_scope_reference": {
                "reference_version": "label_text_scope_reference_v0.1",
                "scope_policy": {
                    "reference_is_not_evidence": True,
                    "template_placeholders_are_not_values": True,
                },
                "in_scope_categories": [],
                "out_of_scope_categories": [],
                "field_catalog": [],
                "entity_catalog": [],
                "table_catalog": [],
            },
            "label_text_scope_agent_context": {
                "context_version": "label_text_scope_agent_context_v0.1",
                "reference_version": "label_text_scope_reference_v0.1",
                "primary_rule": "Only final printed packaging label text can become extracted_data.",
                "reference_is_not_evidence": True,
                "in_scope_categories": [],
                "out_of_scope_categories": [],
            },
            "label_text_scope_report": {
                "report_version": "label_text_scope_report_v0.1",
                "reference_version": "label_text_scope_reference_v0.1",
                "status": "pass",
                "extracted_out_of_scope_count": 0,
                "ignored_noise_node_count": 0,
                "unknown_scope_node_count": 0,
                "scope_gate_rejected_count": 0,
                "node_scope_decisions": [],
                "checks": [],
            },
            "missing_item_report": {
                "missing_fields": [
                    {
                        "missing_id": "missing_field_0001",
                        "semantic_key": "product.ingredients",
                        "field": "ingredients",
                        "status": "missing",
                        "evidence_refs": [],
                    }
                ],
                "missing_tables": [],
            },
            "repair_loop": {},
            "schema_audit": {},
            "structure_audit": {},
            "source_layers": {
                "spans": [
                    {
                        "span_id": "span_0001",
                        "page": 1,
                        "source": "pdf_text",
                        "text": "品名：牛奶",
                        "bbox_status": "available",
                        "bbox_pdf": {
                            "x": 10,
                            "y": 20,
                            "width": 60,
                            "height": 10,
                            "unit": "pt",
                            "origin": "top_left",
                            "page_width": 200,
                            "page_height": 100,
                        },
                        "bbox_normalized": {"x1": 0.05, "y1": 0.2, "x2": 0.35, "y2": 0.3},
                    },
                    {"span_id": "span_0002", "page": 1, "source": "pdf_text", "text": "能量 100kJ 1%", "bbox_status": "missing"},
                    {"span_id": "span_0003", "page": 1, "source": "pdf_text", "text": "设计注意：文字需清晰", "bbox_status": "missing"},
                    {
                        "span_id": "span_0004",
                        "page": 1,
                        "source": "pdf_text",
                        "text": "内容物 1：原味牛奶",
                        "bbox_status": "available",
                        "bbox_pdf": {
                            "x": 10,
                            "y": 60,
                            "width": 80,
                            "height": 10,
                            "unit": "pt",
                            "origin": "top_left",
                            "page_width": 200,
                            "page_height": 100,
                        },
                        "bbox_normalized": {"x1": 0.05, "y1": 0.6, "x2": 0.45, "y2": 0.7},
                    },
                ],
                "layers": {
                    "ocr": {
                        "line_count": 1,
                        "block_count": 1,
                        "token_count": 1,
                        "bbox_available_count": 1,
                        "blocks": [
                            {
                                "block_id": "ocr_block_001",
                                "page": 1,
                                "line_ids": ["ocr_0001"],
                                "line_count": 1,
                                "token_count": 1,
                            }
                        ],
                        "lines": [
                            {
                                "ocr_line_id": "ocr_0001",
                                "block_id": "ocr_block_001",
                                "page": 1,
                                "text": "品名：牛奶",
                                "confidence": 0.99,
                                "bbox_status": "available",
                                "bbox_pdf": {
                                    "x": 1,
                                    "y": 2,
                                    "width": 30,
                                    "height": 10,
                                    "unit": "pt",
                                    "origin": "top_left",
                                    "page_width": 200,
                                    "page_height": 100,
                                },
                                "bbox_normalized": {"x1": 0.01, "y1": 0.02, "x2": 0.3, "y2": 0.12},
                                "tokens": [
                                    {
                                        "token_id": "ocr_tok_0001",
                                        "page": 1,
                                        "text": "品名",
                                        "confidence": 0.98,
                                        "bbox_status": "missing",
                                    }
                                ],
                            }
                        ],
                    }
                }
            },
            "source_anchor_inventory": [],
            "coverage_map": {},
            "table_parser": {},
            "agent_execution_report": {},
            "agent_harness": {},
            "standard_artifacts": {
                "standard_items": [
                    {
                        "id": "std_0001",
                        "field_id": "fld_0001",
                        "field": "product_name",
                        "semantic_key": "product.name",
                        "text": "品名：牛奶",
                        "normalized_text": "牛奶",
                        "value_hash": "sha256:" + "b" * 64,
                        "source": {"page": 1, "section": None, "bbox_normalized": {"x1": 0.05, "y1": 0.2, "x2": 0.35, "y2": 0.3}},
                        "evidence_refs": ["ev_0001"],
                        "status": "normalized",
                        "comparison_required": True,
                        "comparison_profile": {
                            "semantic_key": "product.name",
                            "normalized_value": "牛奶",
                            "value_hash": "sha256:" + "b" * 64,
                            "section_id": None,
                            "entity_id": "product_001",
                            "table_id": None,
                            "row_key": None,
                            "bbox_normalized": {"x1": 0.05, "y1": 0.2, "x2": 0.35, "y2": 0.3},
                            "evidence_refs": ["ev_0001"],
                        },
                    },
                    {
                        "id": "std_0002",
                        "field_id": "fld_0002",
                        "field": "content_name",
                        "semantic_key": "content_item.name",
                        "text": "内容物 1：原味牛奶",
                        "normalized_text": "内容物 1：原味牛奶",
                        "value_hash": "sha256:" + "c" * 64,
                        "source": {"page": 1, "section": None, "bbox_normalized": {"x1": 0.05, "y1": 0.6, "x2": 0.45, "y2": 0.7}},
                        "evidence_refs": ["ev_0004"],
                        "group_id": "content_item_001",
                        "status": "verified",
                        "comparison_required": True,
                        "comparison_profile": {
                            "semantic_key": "content_item.name",
                            "normalized_value": "内容物 1：原味牛奶",
                            "value_hash": "sha256:" + "c" * 64,
                            "section_id": None,
                            "entity_id": "content_item_001",
                            "table_id": None,
                            "row_key": None,
                            "bbox_normalized": {"x1": 0.05, "y1": 0.6, "x2": 0.45, "y2": 0.7},
                            "evidence_refs": ["ev_0004"],
                        },
                    }
                ],
                "comparison_index": {
                    "artifact_version": "comparison_index_v0.1",
                    "status": "ready",
                    "dimension_contract": [
                        "semantic_key",
                        "normalized_value",
                        "value_hash",
                        "section_id",
                        "entity_id",
                        "table_id",
                        "row_key",
                        "bbox_normalized",
                        "evidence_refs",
                    ],
                    "entry_count": 2,
                    "skipped_count": 0,
                    "entries": [
                        {
                            "comparison_id": "cmp_0001",
                            "standard_item_id": "std_0001",
                            "field_id": "fld_0001",
                            "semantic_key": "product.name",
                            "comparison_key": "product.name|||||sha256:" + "b" * 64,
                            "matching_dimensions": {
                                "semantic_key": "product.name",
                                "normalized_value": "牛奶",
                                "value_hash": "sha256:" + "b" * 64,
                                "section_id": None,
                                "entity_id": "product_001",
                                "table_id": None,
                                "row_key": None,
                                "bbox_normalized": {"x1": 0.05, "y1": 0.2, "x2": 0.35, "y2": 0.3},
                                "evidence_refs": ["ev_0001"],
                            },
                        },
                        {
                            "comparison_id": "cmp_0002",
                            "standard_item_id": "std_0002",
                            "field_id": "fld_0002",
                            "semantic_key": "content_item.name",
                            "comparison_key": "content_item.name||content_item_001|||sha256:" + "c" * 64,
                            "matching_dimensions": {
                                "semantic_key": "content_item.name",
                                "normalized_value": "内容物 1：原味牛奶",
                                "value_hash": "sha256:" + "c" * 64,
                                "section_id": None,
                                "entity_id": "content_item_001",
                                "table_id": None,
                                "row_key": None,
                                "bbox_normalized": {"x1": 0.05, "y1": 0.6, "x2": 0.45, "y2": 0.7},
                                "evidence_refs": ["ev_0004"],
                            },
                        },
                    ],
                    "skipped_items": [],
                },
                "quality_report": {"status": "review_required", "downstream_allowed": False},
                "structured_document": {},
                "taxonomy_proposals": [],
                "field_groups": [
                    {
                        "group_id": "content_item_001",
                        "group_type": "content_item",
                        "instance_index": 1,
                        "fields": [
                            {
                                "field_id": "fld_0002",
                                "semantic_key": "content_item.name",
                                "standard_item_id": "std_0002",
                                "text": "内容物 1：原味牛奶",
                            }
                        ],
                        "linked_table_ids": ["tbl_0001"],
                        "container_text": "内容物 1：原味牛奶",
                    }
                ],
                "tables": [
                    {
                        "table_id": "tbl_0001",
                        "table_type": "nutrition_facts",
                        "title": "营养成分表",
                        "linked_entity_id": "content_item_001",
                        "columns": [
                            {"column_id": "col_001", "name": "项目"},
                            {"column_id": "col_002", "name": "每100克"},
                            {"column_id": "col_003", "name": "NRV%"},
                        ],
                        "rows": [
                            {
                                "row_id": "row_0001",
                                "row_key": "energy",
                                "evidence_refs": ["ev_0002"],
                                "cells": [
                                    {
                                        "column_id": "col_001",
                                        "raw_value": "能量",
                                        "normalized_value": "能量",
                                        "evidence_refs": ["ev_0002"],
                                    },
                                    {
                                        "column_id": "col_002",
                                        "raw_value": "100kJ",
                                        "normalized_value": "100kJ",
                                        "evidence_refs": ["ev_0002"],
                                    },
                                    {
                                        "column_id": "col_003",
                                        "raw_value": "1%",
                                        "normalized_value": "1%",
                                        "evidence_refs": ["ev_0002"],
                                    },
                                ],
                            }
                        ],
                        "status": "verified",
                        "bbox_status": "missing",
                        "confidence": {
                            "table_structure_confidence": 0.99,
                            "evidence_confidence": 0.99,
                            "overall": 0.99,
                        },
                        "criticality": "critical",
                        "risk_level": "low",
                        "review_required": False,
                        "evidence_refs": ["ev_0002"],
                    }
                ],
                "lists": [
                    {
                        "list_id": "content_items_001",
                        "list_type": "content_items",
                        "item_count": 1,
                        "items": [
                            {
                                "index": 1,
                                "group_id": "content_item_001",
                                "text": "内容物 1：原味牛奶",
                            }
                        ],
                    }
                ],
                "auto_ingest_candidates": {
                    "document_auto_ingest_allowed": False,
                    "quality_snapshot": {"high_risk_count": 1},
                    "candidates": [
                        {
                            "candidate_id": "auto_ingest_0001",
                            "standard_item_id": "std_0001",
                            "field_id": "fld_0001",
                            "text": "品名：牛奶",
                            "evidence_refs": ["ev_0001"],
                        }
                    ],
                    "blocked_items": [],
                },
            },
        },
    }
    validation = _valid_validation_checks()
    result["validation"] = copy.deepcopy(validation)
    result["cross_validation"] = {
        "checks": copy.deepcopy(validation),
        "source_consistency": {
            "status": "pass",
            "pdf_text_span_count": 4,
            "ocr_line_count": 1,
            "matched_ocr_line_count": 1,
            "issue_count": 0,
            "issues": [],
        },
    }
    return copy.deepcopy(result)


def _valid_validation_checks() -> list[dict]:
    return [
        {
            "validation_id": "val_0001",
            "target_id": "fld_0001",
            "check_type": "schema_validation",
            "result": "passed",
            "semantic_key": "product.name",
            "severity": "info",
            "evidence_refs": ["ev_0001"],
        },
        {
            "validation_id": "val_0002",
            "target_id": "fld_0001",
            "check_type": "bbox_integrity",
            "result": "passed",
            "severity": "high",
            "evidence_refs": [],
            "bbox_status": "available",
        },
        {
            "validation_id": "val_0003",
            "target_id": "fld_0001",
            "check_type": "format_check",
            "result": "passed",
            "format_rule_status": "skipped_no_rule",
            "semantic_key": "product.name",
            "severity": "info",
            "evidence_refs": ["ev_0001"],
        },
        {
            "validation_id": "val_0004",
            "target_id": "fld_0002",
            "check_type": "schema_validation",
            "result": "passed",
            "semantic_key": "content_item.name",
            "severity": "info",
            "evidence_refs": ["ev_0004"],
        },
        {
            "validation_id": "val_0005",
            "target_id": "fld_0002",
            "check_type": "bbox_integrity",
            "result": "passed",
            "severity": "high",
            "evidence_refs": [],
            "bbox_status": "available",
        },
        {
            "validation_id": "val_0006",
            "target_id": "fld_0002",
            "check_type": "format_check",
            "result": "passed",
            "format_rule_status": "skipped_no_rule",
            "semantic_key": "content_item.name",
            "severity": "info",
            "evidence_refs": ["ev_0004"],
        },
        {
            "validation_id": "val_0007",
            "target_id": "source_consistency",
            "check_type": "multi_method_agreement",
            "result": "passed",
            "agreement": "pass",
            "severity": "info",
            "issue_count": 0,
            "issues": [],
        },
        {
            "validation_id": "val_0008",
            "target_id": "internal_consistency",
            "check_type": "internal_consistency",
            "result": "passed",
            "severity": "info",
            "conflict_count": 0,
            "conflicts": [],
            "evidence_refs": [],
        },
        {
            "validation_id": "val_0009",
            "target_id": "tbl_0001",
            "check_type": "table_structure",
            "result": "passed",
            "severity": "info",
            "table_type": "nutrition_facts",
            "row_count": 1,
            "column_count": 3,
            "evidence_refs": ["ev_0002"],
        },
        {
            "validation_id": "val_0010",
            "target_id": "table_quality_report",
            "check_type": "table_structure",
            "result": "passed",
            "severity": "info",
            "table_quality_status": "pass",
            "parser_agreement": "single_parser_only",
            "issue_count": 0,
            "issues": [],
        },
    ]


def _valid_revision_blocks() -> list[dict]:
    return [
        {
            "revision_block_id": "revision_before",
            "region_id": "reg_0001",
            "revision_role": "before",
            "revision_status": "historical_reference",
            "display_name": "更改前",
            "fields": [
                {
                    "field_id": "fld_0002",
                    "semantic_key": "content_item.name",
                    "display_name": "内容物名称",
                    "evidence_refs": ["ev_0004"],
                    "source_span_ids": ["span_0004"],
                    "assignment_confidence": 0.90,
                    "assignment_reason": "field_source_span_after_revision_before_marker",
                }
            ],
            "is_current_standard": False,
            "status": "verified",
            "risk_level": "info",
            "review_required": False,
            "evidence_refs": ["ev_0004"],
            "source_span_ids": ["span_0004"],
            "assignment_status": "assigned_by_span_order",
            "assignment_method": "source_span_order_between_revision_markers",
        },
        {
            "revision_block_id": "revision_after",
            "region_id": "reg_0001",
            "revision_role": "after",
            "revision_status": "current_standard",
            "display_name": "更改后",
            "fields": [
                {
                    "field_id": "fld_0001",
                    "semantic_key": "product.name",
                    "display_name": "品名",
                    "evidence_refs": ["ev_0001"],
                    "source_span_ids": ["span_0001"],
                    "assignment_confidence": 0.90,
                    "assignment_reason": "field_source_span_after_revision_after_marker",
                }
            ],
            "is_current_standard": True,
            "status": "verified",
            "risk_level": "info",
            "review_required": False,
            "evidence_refs": ["ev_0001"],
            "source_span_ids": ["span_0001"],
            "assignment_status": "assigned_by_span_order",
            "assignment_method": "source_span_order_between_revision_markers",
        },
    ]


if __name__ == "__main__":
    unittest.main()
