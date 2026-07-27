from __future__ import annotations

from typing import Any

from .models import CompiledField, Evidence, ExtractionPlan, GeneratedSchema, to_jsonable


def build_audit_input_artifact(
    *,
    page_images: dict[str, Any],
    visual_document_graph: dict[str, Any],
    schema: GeneratedSchema,
    plan: ExtractionPlan,
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    coverage_map: dict[str, Any],
    audit_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_version": "independent_audit_input_v0.1",
        "stage": "independent_audit",
        "purpose": "Freeze the inputs reviewed after deterministic compilation.",
        "input_artifacts": [
            "page_images.json",
            "visual_document_graph.json",
            "generated_schema",
            "extraction_plan.json",
            "extracted_data.fields",
            "evidence.json",
            "coverage_map.json",
        ],
        "separation": {
            "review_runs_after_compiler": True,
            "review_input_type": "compiled_fields_with_evidence",
            "extraction_output_type": "extraction_plan_ranges",
            "audit_may_not_modify_final_json": True,
        },
        "page_images": {
            "status": page_images.get("status"),
            "page_count": page_images.get("page_count"),
            "rendered_count": page_images.get("rendered_count"),
            "failed_count": page_images.get("failed_count"),
        },
        "visual_document_graph": {
            "node_count": visual_document_graph.get("node_count"),
            "edge_count": visual_document_graph.get("edge_count"),
        },
        "generated_schema": to_jsonable(schema),
        "extraction_plan": to_jsonable(plan),
        "compiled_fields": {field_id: to_jsonable(field) for field_id, field in compiled_fields.items()},
        "evidence": to_jsonable(evidence),
        "coverage_map": coverage_map,
        "audit_findings": audit_findings,
        "audit_finding_count": len(audit_findings),
    }
