from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .output_contract import FINAL_JSON_ROOT_KEYS


JSON_EXPORT_CONTRACT_CHECKS = [
    "machine_parseable_json",
    "root_keys_present",
    "failure_stage_present",
    "failure_reason_present",
    "no_guessing",
]


def build_failure_result(
    *,
    input_path: Path | None,
    stage: str,
    reason: str,
    error_type: str,
    runtime_policy: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = dt.datetime.now(dt.UTC).isoformat()
    file_name = input_path.name if input_path else None
    failure = {
        "stage": stage,
        "reason": reason,
        "error_type": error_type,
        "recoverable": False,
    }
    metadata = {
        "parser_version": "mvp_v0.1",
        "pipeline_version": "mvp_pipeline_v0.2",
        "schema_mode": "not_started",
        "no_guessing": True,
        "json_export": {
            "schema_version": "mvp_final_json_v0.1",
            "media_type": "application/json",
            "encoding": "utf-8",
            "root_keys": FINAL_JSON_ROOT_KEYS,
            "contract_checks": JSON_EXPORT_CONTRACT_CHECKS,
            "no_guessing": True,
            "primary_artifact": "result.json",
            "contract_artifact": None,
            "schema_artifact": "schemas/final_result.schema.json",
        },
        "runtime_policy": runtime_policy or {},
        "failure": failure,
        "created_at": created_at,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "job": {
            "job_id": f"job_failed_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}",
            "job_type": "standard_pdf_to_structured_json",
            "status": "failed",
        },
        "document": {
            "file_name": file_name,
            "file_hash": None,
            "page_count": 0,
            "page_sizes": [],
            "detected_document_types": [],
            "language": [],
            "parse_status": "failed",
            "pdf_text_layer_available": False,
            "page_image_status": "not_rendered",
            "warnings": [reason],
            "failure_stage": stage,
            "failure_reason": reason,
        },
        "generated_schema": {
            "schema_id": "schema_not_generated",
            "auto_generated": False,
            "schema_version": "dynamic_v1",
            "sections": [],
            "entity_types": [],
            "field_definitions": [],
            "table_definitions": [],
            "requirement_definitions": [],
        },
        "extracted_data": {
            "sections": [],
            "regions": [],
            "entities": [],
            "fields": {},
            "missing_fields": [],
            "missing_tables": [],
            "tables": [],
            "requirements": [],
            "revision_blocks": [],
        },
        "evidence": [],
        "cross_validation": {
            "checks": [],
            "source_consistency": {"status": "not_run", "issue_count": 0, "issues": []},
        },
        "coverage": {
            "status": "not_run",
            "assigned_node_count": 0,
            "total_node_count": 0,
            "coverage_rate": 0.0,
        },
        "validation": [],
        "quality": {
            "overall_status": "failed",
            "critical_confidence_threshold": 0.95,
            "field_completion_rate": 0.0,
            "critical_field_pass_rate": 0.0,
            "high_risk_count": 1,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "auto_ingest_allowed": False,
            "reason": reason,
        },
        "risks": [
            {
                "risk_id": "risk_parse_failed",
                "target_type": "document",
                "target_id": "document",
                "risk_level": "high",
                "risk_type": "parse_failed",
                "message": reason,
                "evidence_refs": [],
            }
        ],
        "review_tasks": [
            {
                "task_id": "review_parse_failed",
                "target_type": "document",
                "target_id": "document",
                "risk_level": "high",
                "reason": reason,
                "required": True,
                "evidence_refs": [],
            }
        ],
        "metadata": metadata,
    }


def failure_stage_for_exception(exc: BaseException) -> str:
    name = exc.__class__.__name__
    message = str(exc)
    if name == "ConfigError":
        return "runtime_config"
    if name == "RuntimePolicyError":
        return "runtime_policy"
    if name == "ManifestError":
        return "manifest"
    if name == "ParseError":
        return "input_validation"
    if name == "LayoutEvidenceError":
        return "layout_evidence"
    if name == "PdfReadError" or message.startswith("Failed to read PDF"):
        return "pdf_read"
    if name == "JSONDecodeError":
        return "json"
    if isinstance(exc, OSError):
        return "io"
    return "runtime"
