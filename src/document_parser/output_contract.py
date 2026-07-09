from __future__ import annotations

import datetime as dt
from typing import Any

from .models import to_jsonable


FIELD_STATUSES = {
    "extracted",
    "verified",
    "normalized",
    "missing",
    "uncertain",
    "conflict",
    "not_applicable",
    "cannot_verify",
    "manual_review_required",
    "compiled",
    "rejected",
}

BBOX_STATUSES = {"available", "missing"}
JOB_STATUSES = {"queued", "running", "completed", "completed_with_warnings", "failed"}
PARSE_STATUSES = {"completed", "completed_with_warnings", "failed", "partial_failed"}
RISK_LEVELS = {"high", "medium", "low", "info"}
RISK_TARGET_TYPES = {
    "document",
    "generated_schema",
    "field",
    "entity",
    "table",
    "region",
    "requirement",
    "missing_field",
    "missing_table",
    "revision_block",
    "table_parser",
    "source_layer",
    "source_consistency",
}
SCHEMA_FIELD_TYPES = {"string", "long_text", "number", "date", "barcode", "table", "entity", "requirement", "enum", "unknown"}
SCHEMA_CRITICALITIES = {"critical", "non_critical", "medium", "low", "info", "unknown"}
REQUIREMENT_TYPES = {
    "text_size",
    "layout_requirement",
    "barcode_requirement",
    "recycling_mark_requirement",
    "printing_requirement",
    "date_printing_requirement",
    "advertising_claim_restriction",
    "design_note",
    "change_note",
    "other",
}
REGION_TYPES = {
    "document_header",
    "label_text",
    "nutrition_table",
    "nutrition_table_area",
    "package_panel",
    "revision_before",
    "revision_after",
    "requirements",
    "barcode_area",
    "manufacturer_info",
    "content_item_block",
    "free_text_block",
    "unknown",
}
REQUIRED_VALIDATION_CHECK_TYPES = {
    "multi_method_agreement",
    "internal_consistency",
    "table_structure",
}
REQUIRED_FIELD_VALIDATION_CHECK_TYPES = {
    "schema_validation",
    "format_check",
    "bbox_integrity",
}
FINAL_JSON_ROOT_KEYS = [
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
]
JSON_EXPORT_REQUIRED_CHECKS = {
    "machine_parseable_json",
    "root_keys_present",
    "evidence_refs_resolve",
    "risk_targets_resolve",
    "review_task_targets_resolve",
    "no_guessing",
}


def build_output_contract_validation_report(result: Any) -> dict[str, Any]:
    data = to_jsonable(result)
    checks: list[dict[str, Any]] = []

    _check_required_keys(
        checks,
        "result",
        data,
        FINAL_JSON_ROOT_KEYS,
    )

    metadata = _as_dict(data.get("metadata"))
    json_export = _as_dict(metadata.get("json_export"))
    page_images = _as_dict(metadata.get("page_images"))
    source_layers = _as_dict(metadata.get("source_layers"))
    ocr_layer = _as_dict(_as_dict(source_layers.get("layers")).get("ocr"))
    table_parser = _as_dict(metadata.get("table_parser"))
    candidate_visual_document_graph = _as_dict(metadata.get("candidate_visual_document_graph"))
    visual_document_graph = _as_dict(metadata.get("visual_document_graph"))
    vdg_quality_report = _as_dict(metadata.get("vdg_quality_report"))
    vdg_agent_context = _as_dict(metadata.get("vdg_agent_context"))
    vdg_consumption_report = _as_dict(metadata.get("vdg_consumption_report"))
    label_text_scope_reference = _as_dict(metadata.get("label_text_scope_reference"))
    label_text_scope_agent_context = _as_dict(metadata.get("label_text_scope_agent_context"))
    label_text_scope_report = _as_dict(metadata.get("label_text_scope_report"))
    _check_required_keys(
        checks,
        "metadata",
        metadata,
        [
            "no_guessing",
            "ocr_provider",
            "runtime_policy",
            "page_images",
            "candidate_visual_document_graph",
            "visual_document_graph",
            "vdg_quality_report",
            "vdg_agent_context",
            "vdg_consumption_report",
            "label_text_scope_reference",
            "label_text_scope_agent_context",
            "label_text_scope_report",
            "missing_item_report",
            "repair_loop",
            "schema_audit",
            "structure_audit",
            "source_layers",
            "source_anchor_inventory",
            "coverage_map",
            "json_export",
            "standard_artifacts",
            "table_parser",
            "agent_execution_report",
            "agent_harness",
        ],
    )

    standard = _as_dict(metadata.get("standard_artifacts"))
    _check_required_keys(
        checks,
        "metadata.standard_artifacts",
        standard,
        [
            "standard_items",
            "quality_report",
            "structured_document",
            "taxonomy_proposals",
            "field_groups",
            "tables",
            "lists",
            "comparison_index",
            "auto_ingest_candidates",
        ],
    )

    extracted_data = _as_dict(data.get("extracted_data"))
    cross_validation = _as_dict(data.get("cross_validation"))
    source_consistency_report = _as_dict(cross_validation.get("source_consistency") or metadata.get("source_consistency"))
    generated_schema = _as_dict(data.get("generated_schema"))
    job = _as_dict(data.get("job"))
    document = _as_dict(data.get("document"))
    fields = _as_dict(extracted_data.get("fields"))
    entities = _as_dict(extracted_data.get("entities"))
    evidence = _as_list(data.get("evidence"))
    standard_items = _as_list(standard.get("standard_items"))
    comparison_index = _as_dict(standard.get("comparison_index"))
    quality_report = _as_dict(standard.get("quality_report"))
    auto_ingest = _as_dict(standard.get("auto_ingest_candidates"))
    missing_item_report = _as_dict(metadata.get("missing_item_report"))
    risks = _as_list(data.get("risks"))
    review_tasks = _as_list(data.get("review_tasks"))
    validation = _as_list(data.get("validation"))
    cross_validation_checks = _as_list(cross_validation.get("checks"))
    tables = _as_list(standard.get("tables"))
    requirements = _as_list(extracted_data.get("requirements"))
    regions = _as_list(extracted_data.get("regions"))
    field_groups = _as_list(standard.get("field_groups"))
    lists = _as_list(standard.get("lists"))
    revision_blocks = _as_list(extracted_data.get("revision_blocks"))

    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in evidence
        if isinstance(item, dict) and item.get("evidence_id")
    }
    evidence_ids = set(evidence_by_id)
    field_ids = set(fields)
    standard_item_by_id = {str(item.get("id")): item for item in standard_items if isinstance(item, dict) and item.get("id")}
    table_ids = {str(table.get("table_id")) for table in tables if isinstance(table, dict) and table.get("table_id")}
    region_ids = {str(region.get("region_id")) for region in regions if isinstance(region, dict) and region.get("region_id")}
    stable_targets = _stable_target_ids(
        fields,
        entities,
        tables,
        regions,
        requirements,
        missing_item_report,
        revision_blocks,
        document,
        page_images,
        source_layers,
        source_consistency_report,
        table_parser,
    )
    source_span_ids = _known_source_span_ids(source_layers, evidence, visual_document_graph)

    _add_check(
        checks,
        "job_contract",
        "job",
        not _job_contract_failures(job),
        "Job metadata includes stable task identifiers and status.",
        {"failures": _job_contract_failures(job)},
    )
    _add_check(
        checks,
        "document_contract",
        "document",
        not _document_contract_failures(document),
        "Document metadata includes file identity, hash, page count and page sizes.",
        {"failures": _document_contract_failures(document)},
    )
    _add_check(
        checks,
        "no_guessing_policy",
        "metadata.no_guessing",
        metadata.get("no_guessing") is True,
        "Final JSON declares no-guessing mode.",
        {"actual": metadata.get("no_guessing")},
    )
    _add_check(
        checks,
        "json_export_contract",
        "metadata.json_export",
        not _json_export_contract_failures(json_export, data),
        "Final JSON export is self-described for machine consumers.",
        {"failures": _json_export_contract_failures(json_export, data)},
    )
    _add_check(
        checks,
        "metadata_created_at",
        "metadata.created_at",
        _is_iso_datetime(metadata.get("created_at")),
        "metadata.created_at records the parse creation time as an ISO timestamp.",
        {"created_at": metadata.get("created_at")},
    )
    _add_check(
        checks,
        "generated_schema_contract",
        "generated_schema",
        not _generated_schema_contract_failures(generated_schema),
        "Generated schema includes sections, entities, field definitions and table definitions with stable schema keys.",
        {"failures": _generated_schema_contract_failures(generated_schema)},
    )
    _add_check(
        checks,
        "generated_schema_source_refs",
        "generated_schema",
        not _generated_schema_source_ref_failures(generated_schema, source_span_ids),
        "Generated schema definitions with source_span_ids resolve to emitted source spans.",
        {"failures": _generated_schema_source_ref_failures(generated_schema, source_span_ids)},
    )

    _add_check(
        checks,
        "evidence_contract",
        "evidence",
        not _evidence_contract_failures(evidence),
        "Every evidence item includes stable ids, source text, page, method and bbox status.",
        {"failures": _evidence_contract_failures(evidence)},
    )
    _add_check(
        checks,
        "evidence_unique_ids",
        "evidence",
        not _duplicate_evidence_ids(evidence),
        "Every evidence_id is unique.",
        {"failures": _duplicate_evidence_ids(evidence)},
    )
    _add_check(
        checks,
        "evidence_bbox_status_consistency",
        "evidence",
        not _evidence_bbox_status_failures(evidence),
        "Evidence with bbox_status=available includes bbox coordinates.",
        {"failures": _evidence_bbox_status_failures(evidence)},
    )

    _add_check(
        checks,
        "field_contract",
        "extracted_data.fields",
        not _field_contract_failures(fields),
        "Every extracted field includes a stable field_id, status and valid status value.",
        {"failures": _field_contract_failures(fields)},
    )
    _add_check(
        checks,
        "field_evidence_refs",
        "extracted_data.fields",
        not _field_evidence_ref_failures(fields, evidence_ids),
        "Every extracted field evidence_ref points to evidence[].evidence_id.",
        {"failures": _field_evidence_ref_failures(fields, evidence_ids)},
    )
    _add_check(
        checks,
        "field_text_from_evidence",
        "extracted_data.fields",
        not _field_text_integrity_failures(fields, evidence_by_id),
        "Every extracted field raw_value is compiled from its evidence source_text.",
        {"failures": _field_text_integrity_failures(fields, evidence_by_id)},
    )
    _add_check(
        checks,
        "field_normalization_contract",
        "extracted_data.fields",
        not _field_normalization_failures(fields, risks),
        "Field normalization preserves raw values, records rules and routes uncertain normalized values to review.",
        {"failures": _field_normalization_failures(fields, risks)},
    )
    _add_check(
        checks,
        "critical_field_bbox_risk",
        "extracted_data.fields",
        not _critical_field_bbox_failures(fields, evidence_by_id, risks, review_tasks),
        "Every critical field with missing bbox evidence is high risk and requires review.",
        {"failures": _critical_field_bbox_failures(fields, evidence_by_id, risks, review_tasks)},
    )
    _add_check(
        checks,
        "critical_low_confidence_review_tasks",
        "extracted_data.fields",
        not _critical_low_confidence_route_failures(fields, risks, review_tasks),
        "Every critical field below the confidence threshold has a high-risk review route.",
        {"failures": _critical_low_confidence_route_failures(fields, risks, review_tasks)},
    )
    _add_check(
        checks,
        "field_missing_bbox_risks",
        "extracted_data.fields",
        not _field_missing_bbox_risk_failures(fields, evidence_by_id, risks, review_tasks),
        "Every field with missing bbox evidence is routed to risk; critical fields also require review.",
        {"failures": _field_missing_bbox_risk_failures(fields, evidence_by_id, risks, review_tasks)},
    )
    _add_check(
        checks,
        "cross_validation_required_checks",
        "cross_validation.checks",
        not _cross_validation_required_check_failures(validation, cross_validation_checks, fields),
        "Cross validation includes required deterministic checks and mirrors validation[].",
        {"failures": _cross_validation_required_check_failures(validation, cross_validation_checks, fields)},
    )
    _add_check(
        checks,
        "failed_validation_risk_routes",
        "validation",
        not _failed_validation_route_failures(validation, risks, review_tasks),
        "Every failed validation check is routed to risks or required review tasks.",
        {"failures": _failed_validation_route_failures(validation, risks, review_tasks)},
    )
    _add_check(
        checks,
        "source_consistency_issue_risk_routes",
        "cross_validation.source_consistency",
        not _source_consistency_issue_route_failures(source_consistency_report, risks),
        "Every OCR/PDF consistency issue that represents source uncertainty is routed to risks.",
        {"failures": _source_consistency_issue_route_failures(source_consistency_report, risks)},
    )
    _add_check(
        checks,
        "entity_contract",
        "extracted_data.entities",
        not _entity_contract_failures(entities),
        "Every extracted entity includes stable identity, status, confidence and evidence-bearing fields.",
        {"failures": _entity_contract_failures(entities)},
    )
    _add_check(
        checks,
        "entity_refs",
        "extracted_data.entities",
        not _entity_ref_failures(entities, fields, tables, field_groups, evidence_ids),
        "Every entity field, table, group and evidence reference resolves without mixing ownership.",
        {"failures": _entity_ref_failures(entities, fields, tables, field_groups, evidence_ids)},
    )

    _add_check(
        checks,
        "standard_item_contract",
        "metadata.standard_artifacts.standard_items",
        not _standard_item_contract_failures(standard_items),
        "Every standard item includes the stable downstream contract keys.",
        {"failures": _standard_item_contract_failures(standard_items)},
    )
    _add_check(
        checks,
        "standard_item_refs",
        "metadata.standard_artifacts.standard_items",
        not _standard_item_ref_failures(standard_items, field_ids, evidence_ids),
        "Every standard item field_id and evidence_refs resolve to extracted fields and evidence.",
        {"failures": _standard_item_ref_failures(standard_items, field_ids, evidence_ids)},
    )
    _add_check(
        checks,
        "standard_item_text_from_evidence",
        "metadata.standard_artifacts.standard_items",
        not _standard_item_text_integrity_failures(standard_items, evidence_by_id),
        "Every standard item text is compiled from its evidence source_text.",
        {"failures": _standard_item_text_integrity_failures(standard_items, evidence_by_id)},
    )
    _add_check(
        checks,
        "standard_item_comparison_profile",
        "metadata.standard_artifacts.standard_items",
        not _standard_item_comparison_profile_failures(standard_items),
        "Every standard item carries the next-phase comparison profile dimensions.",
        {"failures": _standard_item_comparison_profile_failures(standard_items)},
    )
    _add_check(
        checks,
        "comparison_index_contract",
        "metadata.standard_artifacts.comparison_index",
        not _comparison_index_contract_failures(comparison_index, standard_item_by_id),
        "Every comparison-required standard item is represented in comparison_index with stable matching dimensions.",
        {"failures": _comparison_index_contract_failures(comparison_index, standard_item_by_id)},
    )

    _add_check(
        checks,
        "missing_items_excluded",
        "metadata.missing_item_report",
        not _missing_item_leaks(missing_item_report, standard_items, tables),
        "Missing required fields and tables are not emitted as accepted standard items.",
        {"failures": _missing_item_leaks(missing_item_report, standard_items, tables)},
    )
    _add_check(
        checks,
        "missing_item_risk_routes",
        "metadata.missing_item_report",
        not _missing_item_risk_route_failures(missing_item_report, risks, review_tasks),
        "Every missing required field/table is high risk and requires review.",
        {"failures": _missing_item_risk_route_failures(missing_item_report, risks, review_tasks)},
    )
    _add_check(
        checks,
        "table_contract",
        "metadata.standard_artifacts.tables",
        not _table_contract_failures(tables),
        "Every table includes stable ids, rows, cells and evidence-bearing contract keys.",
        {"failures": _table_contract_failures(tables)},
    )
    _add_check(
        checks,
        "table_evidence_refs",
        "metadata.standard_artifacts.tables",
        not _table_evidence_ref_failures(tables, evidence_ids),
        "Every table and table cell evidence_ref points to evidence[].evidence_id.",
        {"failures": _table_evidence_ref_failures(tables, evidence_ids)},
    )
    _add_check(
        checks,
        "table_cell_text_from_evidence",
        "metadata.standard_artifacts.tables",
        not _table_cell_text_failures(tables, evidence_by_id),
        "Every table cell raw_value is present in its evidence source_text.",
        {"failures": _table_cell_text_failures(tables, evidence_by_id)},
    )
    _add_check(
        checks,
        "table_structure_risk_routes",
        "metadata.standard_artifacts.tables",
        not _table_structure_risk_route_failures(tables, risks, review_tasks),
        "Every unrecovered or review-required table has a risk route; high-risk tables require review.",
        {"failures": _table_structure_risk_route_failures(tables, risks, review_tasks)},
    )
    _add_check(
        checks,
        "requirement_contract",
        "extracted_data.requirements",
        not _requirement_contract_failures(requirements),
        "Every extracted requirement includes stable ids, status and evidence-bearing contract keys.",
        {"failures": _requirement_contract_failures(requirements)},
    )
    _add_check(
        checks,
        "requirement_evidence_refs",
        "extracted_data.requirements",
        not _requirement_evidence_ref_failures(requirements, evidence_ids),
        "Every extracted requirement evidence_ref points to evidence[].evidence_id.",
        {"failures": _requirement_evidence_ref_failures(requirements, evidence_ids)},
    )
    _add_check(
        checks,
        "requirement_text_from_evidence",
        "extracted_data.requirements",
        not _requirement_text_failures(requirements, evidence_by_id),
        "Every requirement_text is compiled from its evidence source_text.",
        {"failures": _requirement_text_failures(requirements, evidence_by_id)},
    )
    _add_check(
        checks,
        "region_contract",
        "extracted_data.regions",
        not _region_contract_failures(regions),
        "Every detected region includes stable id, type, display name, page and source spans.",
        {"failures": _region_contract_failures(regions)},
    )
    _add_check(
        checks,
        "region_source_refs",
        "extracted_data.regions",
        not _region_source_ref_failures(regions, source_span_ids),
        "Every region source_span_id resolves to an emitted source span or VDG text span.",
        {"failures": _region_source_ref_failures(regions, source_span_ids)},
    )
    _add_check(
        checks,
        "region_artifact_refs",
        "extracted_data.regions",
        not _region_artifact_ref_failures(regions, field_ids, table_ids, set(entities), evidence_ids),
        "Every region evidence, field, table and entity reference resolves to final artifacts.",
        {"failures": _region_artifact_ref_failures(regions, field_ids, table_ids, set(entities), evidence_ids)},
    )
    _add_check(
        checks,
        "candidate_visual_document_graph_contract",
        "metadata.candidate_visual_document_graph",
        not _visual_document_graph_contract_failures(candidate_visual_document_graph),
        "The candidate visual document graph includes stable nodes, edges and resolvable node references.",
        {"failures": _visual_document_graph_contract_failures(candidate_visual_document_graph)},
    )
    _add_check(
        checks,
        "visual_document_graph_contract",
        "metadata.visual_document_graph",
        not _visual_document_graph_contract_failures(visual_document_graph),
        "The visual document graph includes stable nodes, edges and resolvable node references.",
        {"failures": _visual_document_graph_contract_failures(visual_document_graph)},
    )
    _add_check(
        checks,
        "vdg_quality_report_contract",
        "metadata.vdg_quality_report",
        not _vdg_quality_report_failures(vdg_quality_report),
        "VDG quality report is machine-readable and not failed.",
        {"failures": _vdg_quality_report_failures(vdg_quality_report)},
    )
    _add_check(
        checks,
        "vdg_agent_context_contract",
        "metadata.vdg_agent_context",
        not _vdg_agent_context_failures(vdg_agent_context),
        "VDG agent context is machine-readable.",
        {"failures": _vdg_agent_context_failures(vdg_agent_context)},
    )
    _add_check(
        checks,
        "vdg_consumption_report_contract",
        "metadata.vdg_consumption_report",
        not _vdg_consumption_report_failures(vdg_consumption_report),
        "VDG consumption report is machine-readable.",
        {"failures": _vdg_consumption_report_failures(vdg_consumption_report)},
    )
    _add_check(
        checks,
        "label_text_scope_reference_contract",
        "metadata.label_text_scope_reference",
        not _label_text_scope_reference_failures(label_text_scope_reference),
        "Label text scope reference is present and declares the packaging-label scope policy.",
        {"failures": _label_text_scope_reference_failures(label_text_scope_reference)},
    )
    _add_check(
        checks,
        "label_text_scope_agent_context_contract",
        "metadata.label_text_scope_agent_context",
        not _label_text_scope_agent_context_failures(label_text_scope_agent_context),
        "Label text scope agent context is machine-readable.",
        {"failures": _label_text_scope_agent_context_failures(label_text_scope_agent_context)},
    )
    _add_check(
        checks,
        "label_text_scope_report_contract",
        "metadata.label_text_scope_report",
        not _label_text_scope_report_failures(label_text_scope_report),
        "Label text scope report is machine-readable and blocks out-of-scope extraction.",
        {"failures": _label_text_scope_report_failures(label_text_scope_report)},
    )
    _add_check(
        checks,
        "region_vdg_refs",
        "metadata.visual_document_graph",
        not _region_vdg_ref_failures(regions, visual_document_graph),
        "Every detected region is represented by a VDG region node.",
        {"failures": _region_vdg_ref_failures(regions, visual_document_graph)},
    )
    _add_check(
        checks,
        "field_group_contract",
        "metadata.standard_artifacts.field_groups",
        not _field_group_contract_failures(field_groups),
        "Every field group includes stable ids, type and field/link containers.",
        {"failures": _field_group_contract_failures(field_groups)},
    )
    _add_check(
        checks,
        "field_group_refs",
        "metadata.standard_artifacts.field_groups",
        not _field_group_ref_failures(field_groups, field_ids, standard_item_by_id, table_ids),
        "Every field group field and table reference resolves to final artifacts.",
        {"failures": _field_group_ref_failures(field_groups, field_ids, standard_item_by_id, table_ids)},
    )
    _add_check(
        checks,
        "field_group_text_matches_standard_item",
        "metadata.standard_artifacts.field_groups",
        not _field_group_text_failures(field_groups, fields, standard_item_by_id),
        "Every field group field text preserves the linked field and standard item text.",
        {"failures": _field_group_text_failures(field_groups, fields, standard_item_by_id)},
    )
    _add_check(
        checks,
        "list_contract",
        "metadata.standard_artifacts.lists",
        not _list_contract_failures(lists),
        "Every list includes stable ids, type, item count and items.",
        {"failures": _list_contract_failures(lists)},
    )
    _add_check(
        checks,
        "list_group_refs",
        "metadata.standard_artifacts.lists",
        not _list_group_ref_failures(lists, field_groups),
        "Every list item resolves to a field group and preserves its container text when present.",
        {"failures": _list_group_ref_failures(lists, field_groups)},
    )
    _add_check(
        checks,
        "revision_block_contract",
        "extracted_data.revision_blocks",
        not _revision_block_contract_failures(revision_blocks),
        "Every revision block includes stable role, current/historical status, assignment status and evidence.",
        {"failures": _revision_block_contract_failures(revision_blocks)},
    )
    _add_check(
        checks,
        "revision_block_refs",
        "extracted_data.revision_blocks",
        not _revision_block_ref_failures(revision_blocks, field_ids, region_ids, evidence_ids, source_span_ids),
        "Revision block fields, evidence and source spans resolve without before/after field mixing.",
        {"failures": _revision_block_ref_failures(revision_blocks, field_ids, region_ids, evidence_ids, source_span_ids)},
    )

    _add_check(
        checks,
        "auto_ingest_refs",
        "metadata.standard_artifacts.auto_ingest_candidates",
        not _auto_ingest_ref_failures(auto_ingest, standard_items),
        "Auto-ingest candidates and blocked items refer to existing standard items and field ids.",
        {"failures": _auto_ingest_ref_failures(auto_ingest, standard_items)},
    )
    _add_check(
        checks,
        "auto_ingest_text_matches_standard_item",
        "metadata.standard_artifacts.auto_ingest_candidates",
        not _auto_ingest_text_failures(auto_ingest, standard_items),
        "Auto-ingest candidates preserve the linked standard item text and evidence refs.",
        {"failures": _auto_ingest_text_failures(auto_ingest, standard_items)},
    )
    _add_check(
        checks,
        "auto_ingest_quality_gate",
        "metadata.standard_artifacts.auto_ingest_candidates",
        _auto_ingest_quality_gate_passed(auto_ingest, risks),
        "Document auto-ingest is not allowed when high risks are present.",
        {
            "document_auto_ingest_allowed": auto_ingest.get("document_auto_ingest_allowed"),
            "actual_high_risk_count": _high_risk_count(risks),
            "quality_snapshot": auto_ingest.get("quality_snapshot"),
        },
    )
    _add_check(
        checks,
        "risk_contract",
        "risks",
        not _risk_contract_failures(risks),
        "Every risk includes stable target, level, type, message and evidence refs.",
        {"failures": _risk_contract_failures(risks)},
    )
    _add_check(
        checks,
        "review_task_contract",
        "review_tasks",
        not _review_task_contract_failures(review_tasks),
        "Every review task includes stable target, risk level, reason, required flag and evidence refs.",
        {"failures": _review_task_contract_failures(review_tasks)},
    )
    _add_check(
        checks,
        "risk_evidence_refs",
        "risks",
        not _risk_evidence_ref_failures(risks, evidence_ids),
        "Every risk evidence_ref points to evidence[].evidence_id when refs are present.",
        {"failures": _risk_evidence_ref_failures(risks, evidence_ids)},
    )
    _add_check(
        checks,
        "risk_target_refs",
        "risks",
        not _target_ref_failures(risks, stable_targets, "risk_id"),
        "Every risk with a stable target_type points to an emitted artifact.",
        {"failures": _target_ref_failures(risks, stable_targets, "risk_id")},
    )
    _add_check(
        checks,
        "review_task_evidence_refs",
        "review_tasks",
        not _review_task_evidence_ref_failures(review_tasks, evidence_ids),
        "Every review task evidence_ref points to evidence[].evidence_id when refs are present.",
        {"failures": _review_task_evidence_ref_failures(review_tasks, evidence_ids)},
    )
    _add_check(
        checks,
        "review_task_target_refs",
        "review_tasks",
        not _target_ref_failures(review_tasks, stable_targets, "task_id"),
        "Every review task with a stable target_type points to an emitted artifact.",
        {"failures": _target_ref_failures(review_tasks, stable_targets, "task_id")},
    )

    _add_check(
        checks,
        "high_risk_review_tasks",
        "review_tasks",
        not _uncovered_high_risks(risks, review_tasks),
        "Every high-risk item has a required review task for the same target.",
        {"failures": _uncovered_high_risks(risks, review_tasks)},
    )
    _add_check(
        checks,
        "quality_report_downstream_gate",
        "metadata.standard_artifacts.quality_report",
        quality_report.get("downstream_allowed") == (quality_report.get("status") == "pass"),
        "quality_report.downstream_allowed matches quality_report.status.",
        {
            "status": quality_report.get("status"),
            "downstream_allowed": quality_report.get("downstream_allowed"),
        },
    )
    _add_check(
        checks,
        "ocr_line_contract",
        "metadata.source_layers.layers.ocr.lines",
        not _ocr_line_contract_failures(ocr_layer),
        "Every OCR line includes stable id, page, text, confidence and bbox status.",
        {"failures": _ocr_line_contract_failures(ocr_layer)},
    )
    _add_check(
        checks,
        "ocr_token_contract",
        "metadata.source_layers.layers.ocr.lines.tokens",
        not _ocr_token_contract_failures(ocr_layer),
        "Every OCR token includes stable id, page, text and bbox status.",
        {"failures": _ocr_token_contract_failures(ocr_layer)},
    )
    _add_check(
        checks,
        "ocr_block_refs",
        "metadata.source_layers.layers.ocr.blocks",
        not _ocr_block_ref_failures(ocr_layer),
        "Every OCR block references emitted OCR lines and reports matching counts.",
        {"failures": _ocr_block_ref_failures(ocr_layer)},
    )

    failed_count = sum(1 for check in checks if check["result"] == "failed")
    return {
        "contract_version": "mvp_output_contract_v0.1",
        "artifact": "output_contract_validation_report.json",
        "status": "pass" if failed_count == 0 else "review_required",
        "check_count": len(checks),
        "failed_count": failed_count,
        "checks": checks,
    }


def _check_required_keys(checks: list[dict[str, Any]], target: str, value: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in value]
    _add_check(
        checks,
        "required_keys_present",
        target,
        not missing,
        f"{target} contains required keys.",
        {"missing_keys": missing},
    )


def _add_check(
    checks: list[dict[str, Any]],
    check_type: str,
    target: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "check_id": f"contract_{len(checks) + 1:04d}",
            "check_type": check_type,
            "target": target,
            "result": "passed" if passed else "failed",
            "message": message,
            "details": details or {},
        }
    )


def _evidence_contract_failures(evidence: list[Any]) -> list[dict[str, Any]]:
    required = ["evidence_id", "source_text", "page", "extraction_methods", "bbox_status", "source_node_ids"]
    failures = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            failures.append({"index": index, "reason": "evidence_not_object"})
            continue
        missing = [key for key in required if key not in item or item.get(key) in (None, "")]
        if not isinstance(item.get("page"), int) or int(item.get("page", 0)) <= 0:
            missing.append("valid_page")
        if not _as_list(item.get("extraction_methods")):
            missing.append("non_empty_extraction_methods")
        if not isinstance(item.get("source_node_ids"), list):
            missing.append("source_node_ids_list")
        if item.get("bbox_status") and item.get("bbox_status") not in BBOX_STATUSES:
            missing.append("valid_bbox_status")
        if missing:
            failures.append({"evidence_id": item.get("evidence_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _job_contract_failures(job: dict[str, Any]) -> list[dict[str, Any]]:
    missing = [key for key in ("job_id", "job_type", "status") if key not in job or job.get(key) in (None, "")]
    if job.get("job_type") and job.get("job_type") not in {"standard_pdf_to_structured_json", "standard_xlsx_to_structured_json"}:
        missing.append("supported_job_type")
    if job.get("status") and job.get("status") not in JOB_STATUSES:
        missing.append("valid_status")
    return [{"missing_or_invalid": sorted(set(missing))}] if missing else []


def _document_contract_failures(document: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("file_name", "file_hash", "page_count", "page_sizes", "parse_status") if key not in document or document.get(key) in (None, "")]
    if document.get("file_hash") and not _is_sha256(document.get("file_hash")):
        missing.append("sha256_file_hash")
    if not isinstance(document.get("page_count"), int) or int(document.get("page_count", 0)) <= 0:
        missing.append("positive_page_count")
    if document.get("parse_status") and document.get("parse_status") not in PARSE_STATUSES:
        missing.append("valid_parse_status")

    page_sizes = _as_list(document.get("page_sizes"))
    if document.get("page_count") != len(page_sizes):
        missing.append("page_sizes_match_page_count")
    for index, page in enumerate(page_sizes):
        if not isinstance(page, dict):
            failures.append({"index": index, "reason": "page_size_not_object"})
            continue
        page_missing = [key for key in ("page", "width", "height") if key not in page or page.get(key) in (None, "")]
        if not isinstance(page.get("page"), int) or int(page.get("page", 0)) <= 0:
            page_missing.append("valid_page")
        if not _is_positive_number(page.get("width")):
            page_missing.append("positive_width")
        if not _is_positive_number(page.get("height")):
            page_missing.append("positive_height")
        if page_missing:
            failures.append({"page": page.get("page"), "missing_or_invalid": sorted(set(page_missing))})

    if missing:
        failures.insert(0, {"missing_or_invalid": sorted(set(missing))})
    return failures


def _json_export_contract_failures(json_export: dict[str, Any], data: dict[str, Any]) -> list[dict[str, Any]]:
    missing = [
        key
        for key in ("schema_version", "media_type", "encoding", "root_keys", "contract_checks", "no_guessing", "schema_artifact")
        if key not in json_export or json_export.get(key) in (None, "")
    ]
    if json_export.get("schema_version") and json_export.get("schema_version") != "mvp_final_json_v0.1":
        missing.append("supported_schema_version")
    if json_export.get("media_type") and json_export.get("media_type") != "application/json":
        missing.append("application_json_media_type")
    if json_export.get("encoding") and str(json_export.get("encoding")).lower() != "utf-8":
        missing.append("utf_8_encoding")
    if json_export.get("no_guessing") is not True:
        missing.append("no_guessing_true")
    if json_export.get("schema_artifact") and json_export.get("schema_artifact") != "schemas/final_result.schema.json":
        missing.append("final_result_schema_artifact")

    root_keys = json_export.get("root_keys")
    if not isinstance(root_keys, list):
        missing.append("root_keys_list")
    elif root_keys != FINAL_JSON_ROOT_KEYS:
        missing.append("root_keys_match_contract")
    actual_root_keys = list(data.keys())
    if actual_root_keys != FINAL_JSON_ROOT_KEYS:
        missing.append("actual_root_keys_match_contract")

    contract_checks = json_export.get("contract_checks")
    if not isinstance(contract_checks, list):
        missing.append("contract_checks_list")
    else:
        check_set = {str(check) for check in contract_checks}
        missing_checks = sorted(JSON_EXPORT_REQUIRED_CHECKS - check_set)
        if missing_checks:
            missing.append("required_contract_checks")

    failures = []
    if missing:
        failure: dict[str, Any] = {"missing_or_invalid": sorted(set(missing))}
        if "required_contract_checks" in missing and isinstance(contract_checks, list):
            failure["missing_contract_checks"] = sorted(JSON_EXPORT_REQUIRED_CHECKS - {str(check) for check in contract_checks})
        failure["actual_root_keys"] = actual_root_keys
        failures.append(failure)
    return failures


def _generated_schema_contract_failures(schema: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("schema_id", "auto_generated", "schema_version", "sections", "entity_types", "field_definitions") if key not in schema or schema.get(key) in (None, "")]
    if not isinstance(schema.get("auto_generated"), bool):
        missing.append("auto_generated_bool")
    if not isinstance(schema.get("sections"), list):
        missing.append("sections_list")
    if not isinstance(schema.get("entity_types"), list):
        missing.append("entity_types_list")
    if not isinstance(schema.get("field_definitions"), list):
        missing.append("field_definitions_list")
    if schema.get("table_definitions") is not None and not isinstance(schema.get("table_definitions"), list):
        missing.append("table_definitions_list")
    if schema.get("requirement_definitions") is not None and not isinstance(schema.get("requirement_definitions"), list):
        missing.append("requirement_definitions_list")
    if missing:
        failures.append({"missing_or_invalid": sorted(set(missing))})

    for index, section in enumerate(_as_list(schema.get("sections"))):
        if not isinstance(section, dict):
            failures.append({"index": index, "reason": "schema_section_not_object"})
            continue
        section_missing = [key for key in ("section_id", "section_type", "display_name") if key not in section or section.get(key) in (None, "")]
        if section.get("source_span_ids") is not None and not isinstance(section.get("source_span_ids"), list):
            section_missing.append("source_span_ids_list")
        if section_missing:
            failures.append({"section_id": section.get("section_id"), "missing_or_invalid": sorted(set(section_missing))})

    for index, entity in enumerate(_as_list(schema.get("entity_types"))):
        if not isinstance(entity, dict):
            failures.append({"index": index, "reason": "schema_entity_type_not_object"})
            continue
        entity_missing = [key for key in ("entity_type", "repeatable") if key not in entity or entity.get(key) is None]
        if not isinstance(entity.get("repeatable"), bool):
            entity_missing.append("repeatable_bool")
        if entity_missing:
            failures.append({"entity_type": entity.get("entity_type"), "missing_or_invalid": sorted(set(entity_missing))})

    for index, definition in enumerate(_as_list(schema.get("field_definitions"))):
        if not isinstance(definition, dict):
            failures.append({"index": index, "reason": "schema_field_definition_not_object"})
            continue
        field_missing = [
            key
            for key in ("field_def_id", "semantic_key", "semantic_key_type", "display_name", "field_type", "criticality", "repeatable", "source_span_ids")
            if key not in definition or definition.get(key) in (None, "")
        ]
        if definition.get("field_type") and definition.get("field_type") not in SCHEMA_FIELD_TYPES:
            field_missing.append("valid_field_type")
        if definition.get("criticality") and definition.get("criticality") not in SCHEMA_CRITICALITIES:
            field_missing.append("valid_criticality")
        if not isinstance(definition.get("repeatable"), bool):
            field_missing.append("repeatable_bool")
        if not _as_list(definition.get("source_span_ids")):
            field_missing.append("non_empty_source_span_ids")
        if field_missing:
            failures.append({"field_def_id": definition.get("field_def_id"), "missing_or_invalid": sorted(set(field_missing))})

    for index, table_definition in enumerate(_as_list(schema.get("table_definitions"))):
        if not isinstance(table_definition, dict):
            failures.append({"index": index, "reason": "schema_table_definition_not_object"})
            continue
        table_missing = [key for key in ("table_type", "display_name", "criticality", "repeatable", "source_span_ids") if key not in table_definition or table_definition.get(key) in (None, "")]
        if table_definition.get("criticality") and table_definition.get("criticality") not in SCHEMA_CRITICALITIES:
            table_missing.append("valid_criticality")
        if not isinstance(table_definition.get("repeatable"), bool):
            table_missing.append("repeatable_bool")
        if not _as_list(table_definition.get("source_span_ids")):
            table_missing.append("non_empty_source_span_ids")
        if table_missing:
            failures.append({"table_type": table_definition.get("table_type"), "missing_or_invalid": sorted(set(table_missing))})

    return failures


def _generated_schema_source_ref_failures(schema: dict[str, Any], source_span_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for container_name, id_key in (
        ("sections", "section_id"),
        ("field_definitions", "field_def_id"),
        ("table_definitions", "table_type"),
        ("requirement_definitions", "requirement_type"),
    ):
        for item in _as_list(schema.get(container_name)):
            if not isinstance(item, dict):
                continue
            missing_refs = [span_id for span_id in _as_list(item.get("source_span_ids")) if str(span_id) not in source_span_ids]
            if missing_refs:
                failures.append({"schema_container": container_name, id_key: item.get(id_key), "missing_source_span_ids": missing_refs})
    return failures


def _duplicate_evidence_ids(evidence: list[Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or not item.get("evidence_id"):
            continue
        evidence_id = str(item.get("evidence_id"))
        if evidence_id in seen:
            duplicates.add(evidence_id)
        seen.add(evidence_id)
    return [{"evidence_id": evidence_id} for evidence_id in sorted(duplicates)]


def _evidence_bbox_status_failures(evidence: list[Any]) -> list[dict[str, Any]]:
    failures = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if item.get("bbox_status") != "available":
            continue
        missing = _bbox_coordinate_failures(item)
        if missing:
            failures.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "bbox_status": item.get("bbox_status"),
                    "missing_or_invalid": missing,
                }
            )
    return failures


def _ocr_line_contract_failures(ocr_layer: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for index, line in enumerate(_as_list(ocr_layer.get("lines"))):
        if not isinstance(line, dict):
            failures.append({"index": index, "reason": "ocr_line_not_object"})
            continue
        missing = [key for key in ("ocr_line_id", "page", "text", "confidence", "bbox_status", "tokens") if key not in line or line.get(key) in (None, "")]
        if not isinstance(line.get("page"), int) or int(line.get("page", 0)) <= 0:
            missing.append("valid_page")
        if not _is_number(line.get("confidence")):
            missing.append("numeric_confidence")
        if line.get("bbox_status") and line.get("bbox_status") not in BBOX_STATUSES:
            missing.append("valid_bbox_status")
        if line.get("bbox_status") == "available" and not _has_bbox_coordinates(line):
            missing.append("bbox_coordinates")
        if not isinstance(line.get("tokens"), list):
            missing.append("tokens_list")
        if missing:
            failures.append({"ocr_line_id": line.get("ocr_line_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _ocr_token_contract_failures(ocr_layer: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for line in _as_list(ocr_layer.get("lines")):
        if not isinstance(line, dict):
            continue
        for token_index, token in enumerate(_as_list(line.get("tokens"))):
            if not isinstance(token, dict):
                failures.append({"ocr_line_id": line.get("ocr_line_id"), "token_index": token_index, "reason": "ocr_token_not_object"})
                continue
            missing = [key for key in ("token_id", "page", "text", "bbox_status") if key not in token or token.get(key) in (None, "")]
            if not isinstance(token.get("page"), int) or int(token.get("page", 0)) <= 0:
                missing.append("valid_page")
            if token.get("confidence") is not None and not _is_number(token.get("confidence")):
                missing.append("numeric_confidence")
            if token.get("bbox_status") and token.get("bbox_status") not in BBOX_STATUSES:
                missing.append("valid_bbox_status")
            if token.get("bbox_status") == "available" and not _has_bbox_coordinates(token):
                missing.append("bbox_coordinates")
            if missing:
                failures.append(
                    {
                        "ocr_line_id": line.get("ocr_line_id"),
                        "token_id": token.get("token_id"),
                        "token_index": token_index,
                        "missing_or_invalid": sorted(set(missing)),
                    }
                )
    return failures


def _ocr_block_ref_failures(ocr_layer: dict[str, Any]) -> list[dict[str, Any]]:
    lines = [line for line in _as_list(ocr_layer.get("lines")) if isinstance(line, dict)]
    blocks = _as_list(ocr_layer.get("blocks"))
    if lines and not blocks:
        return [{"reason": "ocr_lines_without_blocks", "line_count": len(lines)}]

    line_by_id = {str(line.get("ocr_line_id")): line for line in lines if line.get("ocr_line_id")}
    failures = []
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            failures.append({"index": block_index, "reason": "ocr_block_not_object"})
            continue
        missing = [key for key in ("block_id", "page", "line_ids", "line_count", "token_count") if key not in block or block.get(key) in (None, "")]
        line_ids = _as_list(block.get("line_ids"))
        if not line_ids:
            missing.append("non_empty_line_ids")
        if not isinstance(block.get("page"), int) or int(block.get("page", 0)) <= 0:
            missing.append("valid_page")
        if block.get("line_count") != len(line_ids):
            missing.append("line_count_matches_line_ids")
        missing_line_ids = [line_id for line_id in line_ids if str(line_id) not in line_by_id]
        token_count = sum(len(_as_list(line_by_id[str(line_id)].get("tokens"))) for line_id in line_ids if str(line_id) in line_by_id)
        if block.get("token_count") != token_count:
            missing.append("token_count_matches_lines")
        if missing or missing_line_ids:
            failures.append(
                {
                    "block_id": block.get("block_id"),
                    "missing_or_invalid": sorted(set(missing)),
                    "missing_line_ids": missing_line_ids,
                }
            )
    return failures


def _field_contract_failures(fields: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for key, field in fields.items():
        if not isinstance(field, dict):
            failures.append({"field_key": key, "reason": "field_not_object"})
            continue
        required_non_empty = (
            "field_id",
            "semantic_key",
            "display_name",
            "field_type",
            "raw_value",
            "value_hash",
            "status",
            "criticality",
            "confidence",
            "risk_level",
            "evidence_refs",
        )
        required_present = ("clean_value", "normalized_value", "normalization", "review_required")
        missing = [required for required in required_non_empty if required not in field or field.get(required) in (None, "")]
        missing.extend(required for required in required_present if required not in field or field.get(required) is None)
        if field.get("field_id") and field.get("field_id") != key:
            missing.append("field_id_matches_key")
        if field.get("status") and field.get("status") not in FIELD_STATUSES:
            missing.append("valid_status")
        if field.get("field_type") and field.get("field_type") not in SCHEMA_FIELD_TYPES:
            missing.append("valid_field_type")
        if field.get("criticality") and field.get("criticality") not in SCHEMA_CRITICALITIES:
            missing.append("valid_criticality")
        if field.get("risk_level") and field.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if not isinstance(field.get("review_required"), bool):
            missing.append("review_required_bool")
        if field.get("value_hash") and not _is_sha256(field.get("value_hash")):
            missing.append("sha256_value_hash")
        confidence = field.get("confidence")
        if not isinstance(confidence, dict):
            missing.append("confidence_object")
        else:
            overall = confidence.get("overall")
            if not _is_number(overall) or not 0 <= float(overall) <= 1:
                missing.append("confidence_overall_0_to_1")
            for confidence_key, confidence_value in confidence.items():
                if confidence_value is not None and (not _is_number(confidence_value) or not 0 <= float(confidence_value) <= 1):
                    missing.append(f"confidence_{confidence_key}_0_to_1")
        if field.get("criticality") == "critical" and isinstance(confidence, dict):
            overall = confidence.get("overall")
            if _is_number(overall) and float(overall) < 0.95 and field.get("review_required") is not True:
                missing.append("critical_low_confidence_requires_review")
        if not _as_list(field.get("evidence_refs")):
            missing.append("non_empty_evidence_refs")
        if missing:
            failures.append({"field_key": key, "missing_or_invalid": sorted(set(missing))})
    return failures


def _field_evidence_ref_failures(fields: dict[str, Any], evidence_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        missing_refs = [ref for ref in _as_list(field.get("evidence_refs")) if str(ref) not in evidence_ids]
        if missing_refs:
            failures.append({"field_id": field.get("field_id") or key, "missing_refs": missing_refs})
    return failures


def _field_text_integrity_failures(fields: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        evidence_text = _joined_evidence_text(_as_list(field.get("evidence_refs")), evidence_by_id)
        if evidence_text is None:
            continue
        if field.get("raw_value") != evidence_text:
            failures.append(
                {
                    "field_id": field.get("field_id") or key,
                    "expected_from_evidence": evidence_text,
                    "actual_raw_value": field.get("raw_value"),
                }
            )
    return failures


def _field_normalization_failures(
    fields: dict[str, Any],
    risks: list[Any],
) -> list[dict[str, Any]]:
    normalization_risk_targets = {
        str(risk.get("target_id"))
        for risk in risks
        if isinstance(risk, dict)
        and risk.get("target_type") == "field"
        and risk.get("risk_type") == "normalization_applied"
    }
    failures = []
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("field_id") or key)
        missing = []
        normalization = field.get("normalization")
        if not isinstance(normalization, list):
            missing.append("normalization_list")
            normalization_rules: list[Any] = []
        else:
            normalization_rules = normalization
            invalid_rules = [rule for rule in normalization_rules if not isinstance(rule, str) or not rule]
            if invalid_rules:
                missing.append("normalization_rules_non_empty_strings")

        raw_value = field.get("raw_value")
        clean_value = field.get("clean_value")
        normalized_value = field.get("normalized_value")
        changed_by_cleaning = isinstance(raw_value, str) and isinstance(clean_value, str) and clean_value != raw_value.strip()
        changed_by_normalization = isinstance(clean_value, str) and isinstance(normalized_value, str) and normalized_value != clean_value.strip()
        has_normalization = bool(normalization_rules)
        if (changed_by_cleaning or changed_by_normalization) and not has_normalization:
            missing.append("changed_value_requires_normalization_rules")
        if has_normalization:
            if field.get("raw_value") in (None, ""):
                missing.append("raw_value_preserved")
            if field.get("status") == "verified":
                missing.append("normalized_field_status_not_verified")
            if field.get("risk_level") == "info":
                missing.append("normalized_field_records_non_info_risk")
            if field_id not in normalization_risk_targets and field.get("review_required") is not True:
                missing.append("normalization_applied_risk")

        if _normalization_uncertain(field) and field.get("review_required") is not True:
            missing.append("uncertain_normalization_requires_review")
        if missing:
            failures.append({"field_id": field_id, "missing_or_invalid": sorted(set(missing))})
    return failures


def _normalization_uncertain(field: dict[str, Any]) -> bool:
    if not _as_list(field.get("normalization")):
        return False
    status = field.get("status")
    if status in {"uncertain", "conflict", "cannot_verify", "manual_review_required"}:
        return True
    confidence_floor, confidence_failures = _confidence_floor(field.get("confidence"))
    return not confidence_failures and confidence_floor is not None and confidence_floor < 0.95


def _critical_field_bbox_failures(
    fields: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    risks: list[Any],
    review_tasks: list[Any],
) -> list[dict[str, Any]]:
    high_bbox_risk_targets = {
        str(risk.get("target_id"))
        for risk in risks
        if isinstance(risk, dict)
        and risk.get("target_type") == "field"
        and risk.get("risk_level") == "high"
        and risk.get("risk_type") == "critical_field_without_bbox"
    }
    high_review_targets = {
        str(task.get("target_id"))
        for task in review_tasks
        if isinstance(task, dict)
        and task.get("target_type") == "field"
        and task.get("risk_level") == "high"
        and task.get("required") is True
    }
    failures = []
    for key, field in fields.items():
        if not isinstance(field, dict) or field.get("criticality") != "critical":
            continue
        field_id = str(field.get("field_id") or key)
        missing_bbox_refs = [
            str(ref)
            for ref in _as_list(field.get("evidence_refs"))
            if isinstance(evidence_by_id.get(str(ref)), dict)
            and evidence_by_id[str(ref)].get("bbox_status") == "missing"
            and not _is_xlsx_structured_evidence(evidence_by_id[str(ref)])
        ]
        if not missing_bbox_refs:
            continue
        missing = []
        if field.get("review_required") is not True:
            missing.append("review_required_true")
        if field.get("risk_level") != "high":
            missing.append("risk_level_high")
        if field.get("status") not in {"manual_review_required", "uncertain"}:
            missing.append("non_passing_status")
        if field_id not in high_bbox_risk_targets:
            missing.append("critical_field_without_bbox_high_risk")
        if field_id not in high_review_targets:
            missing.append("high_risk_review_task")
        if missing:
            failures.append(
                {
                    "field_id": field_id,
                    "missing_bbox_refs": missing_bbox_refs,
                    "missing_or_invalid": sorted(set(missing)),
                }
            )
    return failures


def _critical_low_confidence_route_failures(
    fields: dict[str, Any],
    risks: list[Any],
    review_tasks: list[Any],
) -> list[dict[str, Any]]:
    high_risk_targets = _risk_targets(risks, risk_levels={"high"})
    high_review_targets = _review_targets(review_tasks, risk_levels={"high"}, required_only=True)
    failures = []
    for key, field in fields.items():
        if not isinstance(field, dict) or field.get("criticality") != "critical":
            continue
        overall = _confidence_value(field.get("confidence"), "overall")
        if overall is None or overall >= 0.95:
            continue
        field_id = str(field.get("field_id") or key)
        missing = []
        if field.get("risk_level") != "high":
            missing.append("risk_level_high")
        if field.get("review_required") is not True:
            missing.append("review_required")
        if ("field", field_id) not in high_risk_targets:
            missing.append("high_risk")
        if ("field", field_id) not in high_review_targets:
            missing.append("high_risk_review_task")
        if missing:
            failures.append({"field_id": field_id, "missing_or_invalid": sorted(set(missing))})
    return failures


def _field_missing_bbox_risk_failures(
    fields: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    risks: list[Any],
    review_tasks: list[Any],
) -> list[dict[str, Any]]:
    bbox_risk_targets = _risk_targets(
        risks,
        risk_types={"critical_field_without_bbox", "field_without_bbox"},
    )
    high_bbox_risk_targets = _risk_targets(
        risks,
        risk_levels={"high"},
        risk_types={"critical_field_without_bbox", "field_without_bbox"},
    )
    high_review_targets = _review_targets(review_tasks, risk_levels={"high"}, required_only=True)
    failures = []
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("field_id") or key)
        missing_bbox_refs = [
            str(ref)
            for ref in _as_list(field.get("evidence_refs"))
            if isinstance(evidence_by_id.get(str(ref)), dict)
            and evidence_by_id[str(ref)].get("bbox_status") == "missing"
            and not _is_xlsx_structured_evidence(evidence_by_id[str(ref)])
        ]
        if not missing_bbox_refs:
            continue
        missing = []
        if ("field", field_id) not in bbox_risk_targets:
            missing.append("missing_bbox_risk")
        if field.get("criticality") == "critical":
            if field.get("risk_level") != "high":
                missing.append("risk_level_high")
            if field.get("review_required") is not True:
                missing.append("review_required")
            if ("field", field_id) not in high_bbox_risk_targets:
                missing.append("critical_missing_bbox_high_risk")
            if ("field", field_id) not in high_review_targets:
                missing.append("high_risk_review_task")
        if missing:
            failures.append(
                {
                    "field_id": field_id,
                    "missing_bbox_refs": missing_bbox_refs,
                    "missing_or_invalid": sorted(set(missing)),
                }
            )
    return failures


def _cross_validation_required_check_failures(
    validation: list[Any],
    cross_validation_checks: list[Any],
    fields: dict[str, Any],
) -> list[dict[str, Any]]:
    validation_check_ids = [
        str(check.get("validation_id"))
        for check in validation
        if isinstance(check, dict) and check.get("validation_id")
    ]
    cross_validation_check_ids = [
        str(check.get("validation_id"))
        for check in cross_validation_checks
        if isinstance(check, dict) and check.get("validation_id")
    ]
    validation_types = {
        str(check.get("check_type"))
        for check in validation
        if isinstance(check, dict) and check.get("check_type")
    }
    required_types = set(REQUIRED_VALIDATION_CHECK_TYPES)
    if fields:
        required_types.update(REQUIRED_FIELD_VALIDATION_CHECK_TYPES)
    failures = []
    missing_types = sorted(required_types - validation_types)
    if missing_types:
        failures.append({"reason": "missing_required_validation_check_types", "missing": missing_types})
    if not validation:
        failures.append({"reason": "validation_checks_empty"})
    if not cross_validation_checks:
        failures.append({"reason": "cross_validation_checks_empty"})
    if validation_check_ids != cross_validation_check_ids:
        failures.append(
            {
                "reason": "cross_validation_checks_do_not_mirror_validation",
                "validation_check_ids": validation_check_ids,
                "cross_validation_check_ids": cross_validation_check_ids,
            }
        )
    return failures


def _is_xlsx_structured_evidence(evidence: dict[str, Any]) -> bool:
    return any(str(method).startswith("xlsx_") for method in _as_list(evidence.get("extraction_methods")))


def _failed_validation_route_failures(
    validation: list[Any],
    risks: list[Any],
    review_tasks: list[Any],
) -> list[dict[str, Any]]:
    risk_routes = {
        (
            str(risk.get("target_type")),
            str(risk.get("target_id")),
            str(risk.get("risk_type")),
        )
        for risk in risks
        if isinstance(risk, dict)
    }
    review_routes = {
        (str(task.get("target_type")), str(task.get("target_id")))
        for task in review_tasks
        if isinstance(task, dict) and task.get("required") is True
    }
    failures = []
    for check in validation:
        if not isinstance(check, dict) or check.get("result") != "failed":
            continue
        route_options = _validation_route_options(check)
        if not route_options:
            failures.append(
                {
                    "validation_id": check.get("validation_id"),
                    "check_type": check.get("check_type"),
                    "reason": "no_route_mapping",
                }
            )
            continue
        has_risk = any(route in risk_routes for route in route_options)
        has_review = any((target_type, target_id) in review_routes for target_type, target_id, _ in route_options)
        if not has_risk and not has_review:
            failures.append(
                {
                    "validation_id": check.get("validation_id"),
                    "check_type": check.get("check_type"),
                    "target_id": check.get("target_id"),
                    "expected_routes": [
                        {"target_type": target_type, "target_id": target_id, "risk_type": risk_type}
                        for target_type, target_id, risk_type in route_options
                    ],
                }
            )
    return failures


def _source_consistency_issue_route_failures(source_consistency_report: dict[str, Any], risks: list[Any]) -> list[dict[str, Any]]:
    routed_issue_types = {
        "ocr_low_confidence",
        "pdf_ocr_text_conflict",
        "ocr_important_text_unmatched",
        "pdf_important_text_unconfirmed_by_ocr",
    }
    risk_routes = {
        (
            str(risk.get("target_type")),
            str(risk.get("target_id")),
            str(risk.get("risk_type")),
        )
        for risk in risks
        if isinstance(risk, dict)
    }
    failures = []
    for issue in _as_list(source_consistency_report.get("issues")):
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("issue_type") or "")
        if issue_type not in routed_issue_types:
            continue
        issue_id = str(issue.get("issue_id") or "")
        expected_route = ("source_consistency", issue_id, issue_type)
        if expected_route not in risk_routes:
            failures.append(
                {
                    "issue_id": issue.get("issue_id"),
                    "issue_type": issue.get("issue_type"),
                    "expected_route": {
                        "target_type": "source_consistency",
                        "target_id": issue_id,
                        "risk_type": issue_type,
                    },
                }
            )
    return failures


def _validation_route_options(check: dict[str, Any]) -> list[tuple[str, str, str]]:
    check_type = str(check.get("check_type", ""))
    target_id = str(check.get("target_id", ""))
    if check_type == "bbox_integrity":
        risk_type = str(check.get("risk_type") or "")
        if risk_type in {"critical_field_without_bbox", "field_without_bbox"}:
            return [("field", target_id, risk_type)]
        return [
            ("field", target_id, "critical_field_without_bbox"),
            ("field", target_id, "field_without_bbox"),
        ]
    if check_type == "schema_validation":
        return [("field", target_id, "schema_validation_failed")]
    if check_type == "multi_method_agreement":
        return [("source_consistency", target_id, "multi_method_agreement_failed")]
    if check_type == "internal_consistency":
        return [("field", target_id, "field_internal_conflict")]
    if check_type == "table_structure":
        target_type = "table_parser" if target_id == "table_quality_report" else "table"
        return [(target_type, target_id, "table_structure_validation_failed")]
    if check_type == "format_check":
        return [("field", target_id, "format_check_failed")]
    if check_type == "missing_required_field":
        return [("missing_field", target_id, "critical_field_missing")]
    if check_type == "missing_required_table":
        return [("missing_table", target_id, "critical_table_missing")]
    if check_type == "schema_audit":
        issue_types = [
            str(issue.get("issue_type"))
            for issue in _as_list(check.get("issues"))
            if isinstance(issue, dict) and issue.get("issue_type")
        ]
        if issue_types:
            return [("generated_schema", "generated_schema", issue_type) for issue_type in issue_types]
        return [("generated_schema", "generated_schema", "schema_audit_issue")]
    if check_type in {
        "vdg_quality",
        "vdg_boundary_validation",
        "vdg_node_coverage",
        "vdg_table_cell_boundary",
        "vdg_region_boundary",
        "label_text_scope",
        "label_text_scope_reference",
        "label_text_scope_gate",
        "label_text_scope_unknown",
    }:
        return [("document", "document", f"{check_type}_failed")]
    return []


def _entity_contract_failures(entities: dict[str, Any]) -> list[dict[str, Any]]:
    if not entities:
        return [{"reason": "entities_missing_or_empty"}]
    failures = []
    for entity_key, entity in entities.items():
        if not isinstance(entity, dict):
            failures.append({"entity_id": entity_key, "reason": "entity_not_object"})
            continue
        missing = [
            key
            for key in ("entity_id", "entity_type", "index", "fields", "linked_table_ids", "status", "confidence", "risk_level", "review_required", "evidence_refs")
            if key not in entity or entity.get(key) is None
        ]
        if not isinstance(entity.get("index"), int) or int(entity.get("index", 0)) <= 0:
            missing.append("positive_index")
        if entity.get("status") and entity.get("status") not in FIELD_STATUSES:
            missing.append("valid_status")
        if entity.get("risk_level") and entity.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if not isinstance(entity.get("review_required"), bool):
            missing.append("review_required_bool")
        if not isinstance(entity.get("fields"), dict):
            missing.append("fields_object")
        if not isinstance(entity.get("linked_table_ids"), list):
            missing.append("linked_table_ids_list")
        if not isinstance(entity.get("evidence_refs"), list):
            missing.append("evidence_refs_list")
        elif (entity.get("fields") or entity.get("linked_table_ids")) and not _as_list(entity.get("evidence_refs")):
            missing.append("non_empty_evidence_refs")
        confidence_floor, confidence_failures = _confidence_floor(entity.get("confidence"))
        missing.extend(confidence_failures)
        entity_linking_confidence = _confidence_value(entity.get("confidence"), "entity_linking_confidence")
        if entity_linking_confidence is not None and entity_linking_confidence < 0.90:
            if entity.get("status") != "uncertain":
                missing.append("low_entity_linking_confidence_requires_uncertain_status")
            if entity.get("review_required") is not True:
                missing.append("low_entity_linking_confidence_requires_review")
        if confidence_floor is not None and confidence_floor < 0.90 and entity.get("review_required") is not True:
            missing.append("low_entity_confidence_requires_review")

        for slot, entity_field in _as_dict(entity.get("fields")).items():
            if not isinstance(entity_field, dict):
                failures.append({"entity_id": entity.get("entity_id"), "field_slot": slot, "reason": "entity_field_not_object"})
                continue
            field_missing = [
                key
                for key in ("field_id", "semantic_key", "value", "status", "criticality", "confidence", "risk_level", "review_required", "evidence_refs")
                if key not in entity_field or entity_field.get(key) is None
            ]
            if entity_field.get("status") and entity_field.get("status") not in FIELD_STATUSES:
                field_missing.append("valid_status")
            if entity_field.get("criticality") and entity_field.get("criticality") not in SCHEMA_CRITICALITIES:
                field_missing.append("valid_criticality")
            if entity_field.get("risk_level") and entity_field.get("risk_level") not in RISK_LEVELS:
                field_missing.append("valid_risk_level")
            if not isinstance(entity_field.get("review_required"), bool):
                field_missing.append("review_required_bool")
            field_confidence_floor, field_confidence_failures = _confidence_floor(entity_field.get("confidence"))
            field_missing.extend(field_confidence_failures)
            if entity_field.get("criticality") == "critical" and field_confidence_floor is not None and field_confidence_floor < 0.95 and entity_field.get("review_required") is not True:
                field_missing.append("low_field_confidence_requires_review")
            if not _as_list(entity_field.get("evidence_refs")):
                field_missing.append("non_empty_evidence_refs")
            if field_missing:
                failures.append(
                    {
                        "entity_id": entity.get("entity_id"),
                        "field_slot": slot,
                        "missing_or_invalid": sorted(set(field_missing)),
                    }
                )
        if missing:
            failures.append({"entity_id": entity.get("entity_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _entity_ref_failures(
    entities: dict[str, Any],
    fields: dict[str, Any],
    tables: list[Any],
    field_groups: list[Any],
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    table_by_id = {str(table.get("table_id")): table for table in tables if isinstance(table, dict) and table.get("table_id")}
    group_by_id = {str(group.get("group_id")): group for group in field_groups if isinstance(group, dict) and group.get("group_id")}
    failures = []
    for entity_key, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("entity_id", ""))
        if entity_id != str(entity_key):
            failures.append({"entity_id": entity.get("entity_id"), "reason": "entity_id_key_mismatch", "key": entity_key})
        missing_entity_refs = [ref for ref in _as_list(entity.get("evidence_refs")) if str(ref) not in evidence_ids]
        if missing_entity_refs:
            failures.append({"entity_id": entity_id, "reason": "missing_evidence_refs", "refs": missing_entity_refs})

        for slot, entity_field in _as_dict(entity.get("fields")).items():
            if not isinstance(entity_field, dict):
                continue
            field_id = str(entity_field.get("field_id", ""))
            field = _as_dict(fields.get(field_id))
            field_failures = []
            if not field:
                field_failures.append({"reason": "unknown_field_id", "field_id": entity_field.get("field_id")})
            else:
                if field.get("entity_id") != entity_id:
                    field_failures.append(
                        {
                            "reason": "field_entity_mismatch",
                            "expected": entity_id,
                            "actual": field.get("entity_id"),
                        }
                    )
                if entity_field.get("semantic_key") != field.get("semantic_key"):
                    field_failures.append(
                        {
                            "reason": "semantic_key_mismatch",
                            "expected": field.get("semantic_key"),
                            "actual": entity_field.get("semantic_key"),
                        }
                    )
                if entity_field.get("value") != field.get("raw_value"):
                    field_failures.append(
                        {
                            "reason": "field_value_rewritten",
                            "expected": field.get("raw_value"),
                            "actual": entity_field.get("value"),
                        }
                    )
                if _as_list(entity_field.get("evidence_refs")) != _as_list(field.get("evidence_refs")):
                    field_failures.append(
                        {
                            "reason": "evidence_refs_mismatch",
                            "expected": field.get("evidence_refs"),
                            "actual": entity_field.get("evidence_refs"),
                        }
                    )
            missing_field_refs = [ref for ref in _as_list(entity_field.get("evidence_refs")) if str(ref) not in evidence_ids]
            if missing_field_refs:
                field_failures.append({"reason": "missing_evidence_refs", "refs": missing_field_refs})
            if field_failures:
                failures.append({"entity_id": entity_id, "field_slot": slot, "failures": field_failures})

        for table_id in _as_list(entity.get("linked_table_ids")):
            table = table_by_id.get(str(table_id))
            if not table:
                failures.append({"entity_id": entity_id, "reason": "unknown_table_id", "table_id": table_id})
                continue
            linked_entity_id = str(table.get("linked_entity_id") or "product_001")
            if linked_entity_id != entity_id:
                failures.append(
                    {
                        "entity_id": entity_id,
                        "reason": "table_entity_mismatch",
                        "table_id": table_id,
                        "actual_linked_entity_id": linked_entity_id,
                    }
                )

        if entity_id != "product_001" and (entity.get("fields") or entity.get("linked_table_ids")):
            group = group_by_id.get(entity_id)
            if not group:
                failures.append({"entity_id": entity_id, "reason": "missing_matching_field_group"})
            elif group.get("group_type") != entity.get("entity_type"):
                failures.append(
                    {
                        "entity_id": entity_id,
                        "reason": "field_group_type_mismatch",
                        "expected": entity.get("entity_type"),
                        "actual": group.get("group_type"),
                    }
                )
    return failures


def _standard_item_contract_failures(items: list[Any]) -> list[dict[str, Any]]:
    required_non_empty = [
        "id",
        "field_id",
        "field",
        "semantic_key",
        "text",
        "value_hash",
        "source",
        "evidence_refs",
        "status",
        "comparison_required",
        "comparison_profile",
    ]
    required_present = ["normalized_text"]
    failures = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            failures.append({"index": index, "reason": "standard_item_not_object"})
            continue
        missing = [key for key in required_non_empty if key not in item or item.get(key) in (None, "")]
        missing.extend(key for key in required_present if key not in item or item.get(key) is None)
        if item.get("status") and item.get("status") not in FIELD_STATUSES:
            missing.append("valid_status")
        if item.get("value_hash") and not _is_sha256(item.get("value_hash")):
            missing.append("value_hash_sha256")
        if not _as_list(item.get("evidence_refs")):
            missing.append("non_empty_evidence_refs")
        if missing:
            failures.append({"standard_item_id": item.get("id"), "missing_or_invalid": sorted(set(missing))})
    return failures


COMPARISON_PROFILE_KEYS = [
    "semantic_key",
    "normalized_value",
    "value_hash",
    "section_id",
    "entity_id",
    "table_id",
    "row_key",
    "bbox_normalized",
    "evidence_refs",
]


def _standard_item_comparison_profile_failures(items: list[Any]) -> list[dict[str, Any]]:
    failures = []
    for item in items:
        if not isinstance(item, dict):
            continue
        profile = item.get("comparison_profile")
        item_failures = _comparison_profile_failures(profile)
        if isinstance(profile, dict):
            if profile.get("semantic_key") != item.get("semantic_key"):
                item_failures.append("semantic_key_mismatch")
            if profile.get("normalized_value") != item.get("normalized_text"):
                item_failures.append("normalized_value_mismatch")
            if profile.get("value_hash") != item.get("value_hash"):
                item_failures.append("value_hash_mismatch")
            if _as_list(profile.get("evidence_refs")) != _as_list(item.get("evidence_refs")):
                item_failures.append("evidence_refs_mismatch")
        if item_failures:
            failures.append({"standard_item_id": item.get("id"), "failures": sorted(set(item_failures))})
    return failures


def _comparison_index_contract_failures(index: dict[str, Any], standard_item_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    if not isinstance(index, dict) or not index:
        return [{"reason": "comparison_index_missing"}]
    entries = _as_list(index.get("entries"))
    dimension_contract = _as_list(index.get("dimension_contract"))
    missing_dimensions = [key for key in COMPARISON_PROFILE_KEYS if key not in dimension_contract]
    if missing_dimensions:
        failures.append({"reason": "dimension_contract_incomplete", "missing": missing_dimensions})

    required_item_ids = {
        str(item.get("id"))
        for item in standard_item_by_id.values()
        if isinstance(item, dict) and item.get("id") and item.get("comparison_required") is True
    }
    entry_item_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append({"reason": "comparison_entry_not_object"})
            continue
        standard_item_id = str(entry.get("standard_item_id") or "")
        entry_item_ids.add(standard_item_id)
        linked_item = standard_item_by_id.get(standard_item_id)
        entry_failures = []
        if not linked_item:
            entry_failures.append("unknown_standard_item_id")
        else:
            if entry.get("field_id") != linked_item.get("field_id"):
                entry_failures.append("field_id_mismatch")
            if entry.get("semantic_key") != linked_item.get("semantic_key"):
                entry_failures.append("semantic_key_mismatch")
        profile = entry.get("matching_dimensions")
        entry_failures.extend(_comparison_profile_failures(profile))
        if isinstance(profile, dict) and linked_item:
            linked_profile = linked_item.get("comparison_profile")
            if isinstance(linked_profile, dict):
                for key in COMPARISON_PROFILE_KEYS:
                    if profile.get(key) != linked_profile.get(key):
                        entry_failures.append(f"{key}_differs_from_standard_item")
        if not entry.get("comparison_key"):
            entry_failures.append("comparison_key_required")
        if entry_failures:
            failures.append({"comparison_id": entry.get("comparison_id"), "standard_item_id": standard_item_id, "failures": sorted(set(entry_failures))})
    missing_entries = sorted(required_item_ids - entry_item_ids)
    if missing_entries:
        failures.append({"reason": "missing_comparison_entries", "standard_item_ids": missing_entries})
    return failures


def _comparison_profile_failures(profile: Any) -> list[str]:
    if not isinstance(profile, dict):
        return ["comparison_profile_object"]
    failures = []
    for key in COMPARISON_PROFILE_KEYS:
        if key not in profile:
            failures.append(f"{key}_missing")
    if not profile.get("semantic_key"):
        failures.append("semantic_key_required")
    if "normalized_value" not in profile:
        failures.append("normalized_value_missing")
    if not _is_sha256(profile.get("value_hash")):
        failures.append("value_hash_sha256")
    if not _as_list(profile.get("evidence_refs")):
        failures.append("evidence_refs_required")
    bbox_normalized = profile.get("bbox_normalized")
    if bbox_normalized is not None:
        failures.extend(_bbox_normalized_failures(bbox_normalized))
    return failures


def _bbox_normalized_failures(bbox_normalized: Any) -> list[str]:
    if not isinstance(bbox_normalized, dict):
        return ["bbox_normalized_object"]
    failures = []
    for key in ("x1", "y1", "x2", "y2"):
        if not _is_number(bbox_normalized.get(key)):
            failures.append(f"bbox_normalized_{key}_number")
        elif not 0 <= float(bbox_normalized.get(key)) <= 1:
            failures.append(f"bbox_normalized_{key}_0_to_1")
    if _is_number(bbox_normalized.get("x1")) and _is_number(bbox_normalized.get("x2")) and float(bbox_normalized.get("x1")) > float(bbox_normalized.get("x2")):
        failures.append("bbox_normalized_x_order")
    if _is_number(bbox_normalized.get("y1")) and _is_number(bbox_normalized.get("y2")) and float(bbox_normalized.get("y1")) > float(bbox_normalized.get("y2")):
        failures.append("bbox_normalized_y_order")
    return failures


def _standard_item_text_integrity_failures(items: list[Any], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_text = _joined_evidence_text(_as_list(item.get("evidence_refs")), evidence_by_id)
        if evidence_text is None:
            continue
        if item.get("text") != evidence_text:
            failures.append(
                {
                    "standard_item_id": item.get("id"),
                    "expected_from_evidence": evidence_text,
                    "actual_text": item.get("text"),
                }
            )
    return failures


def _standard_item_ref_failures(items: list[Any], field_ids: set[str], evidence_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_failures = []
        if str(item.get("field_id")) not in field_ids:
            item_failures.append({"reason": "missing_field_id", "field_id": item.get("field_id")})
        missing_refs = [ref for ref in _as_list(item.get("evidence_refs")) if str(ref) not in evidence_ids]
        if missing_refs:
            item_failures.append({"reason": "missing_evidence_refs", "refs": missing_refs})
        if item_failures:
            failures.append({"standard_item_id": item.get("id"), "failures": item_failures})
    return failures


def _joined_evidence_text(refs: list[Any], evidence_by_id: dict[str, dict[str, Any]]) -> str | None:
    if not refs:
        return None
    texts = []
    for ref in refs:
        evidence = evidence_by_id.get(str(ref))
        if not evidence:
            return None
        texts.append(str(evidence.get("source_text", "")))
    return "\n".join(texts).strip()


def _missing_item_leaks(
    missing_item_report: dict[str, Any],
    standard_items: list[Any],
    tables: list[Any],
) -> list[dict[str, Any]]:
    failures = []
    standard_semantic_keys = {
        item.get("semantic_key") for item in standard_items if isinstance(item, dict) and item.get("semantic_key")
    }
    accepted_table_statuses = {"verified", "normalized", "compiled", "extracted"}
    table_types = {
        item.get("table_type")
        for item in tables
        if isinstance(item, dict)
        and item.get("table_type")
        and item.get("status") in accepted_table_statuses
        and not item.get("review_required")
    }
    for item in _as_list(missing_item_report.get("missing_fields")):
        if isinstance(item, dict) and item.get("semantic_key") in standard_semantic_keys:
            failures.append({"type": "missing_field_in_standard_items", "semantic_key": item.get("semantic_key")})
    for item in _as_list(missing_item_report.get("missing_tables")):
        if isinstance(item, dict) and item.get("table_type") in table_types:
            failures.append({"type": "missing_table_in_tables", "table_type": item.get("table_type")})
    return failures


def _missing_item_risk_route_failures(
    missing_item_report: dict[str, Any],
    risks: list[Any],
    review_tasks: list[Any],
) -> list[dict[str, Any]]:
    high_risk_targets = _risk_targets(risks, risk_levels={"high"})
    high_review_targets = _review_targets(review_tasks, risk_levels={"high"}, required_only=True)
    failures = []
    for item in _as_list(missing_item_report.get("missing_fields")):
        if not isinstance(item, dict):
            continue
        missing_id = str(item.get("missing_id") or "")
        missing = []
        if ("missing_field", missing_id) not in high_risk_targets:
            missing.append("high_risk")
        if ("missing_field", missing_id) not in high_review_targets:
            missing.append("high_risk_review_task")
        if missing:
            failures.append({"target_type": "missing_field", "target_id": missing_id, "missing_or_invalid": sorted(set(missing))})
    for item in _as_list(missing_item_report.get("missing_tables")):
        if not isinstance(item, dict):
            continue
        missing_id = str(item.get("missing_id") or "")
        missing = []
        if ("missing_table", missing_id) not in high_risk_targets:
            missing.append("high_risk")
        if ("missing_table", missing_id) not in high_review_targets:
            missing.append("high_risk_review_task")
        if missing:
            failures.append({"target_type": "missing_table", "target_id": missing_id, "missing_or_invalid": sorted(set(missing))})
    return failures


def _table_contract_failures(tables: list[Any]) -> list[dict[str, Any]]:
    required = [
        "table_id",
        "table_type",
        "title",
        "columns",
        "rows",
        "status",
        "bbox_status",
        "confidence",
        "criticality",
        "risk_level",
        "review_required",
        "evidence_refs",
    ]
    failures = []
    for table_index, table in enumerate(tables):
        if not isinstance(table, dict):
            failures.append({"index": table_index, "reason": "table_not_object"})
            continue
        missing = [key for key in required if key not in table or table.get(key) in (None, "")]
        if table.get("status") and table.get("status") not in FIELD_STATUSES:
            missing.append("valid_status")
        if table.get("criticality") and table.get("criticality") not in SCHEMA_CRITICALITIES:
            missing.append("valid_criticality")
        if table.get("risk_level") and table.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if table.get("bbox_status") and table.get("bbox_status") not in BBOX_STATUSES:
            missing.append("valid_bbox_status")
        if table.get("bbox_status") == "available" and not _has_bbox_coordinates(table):
            missing.append("bbox_coordinates")
        if not isinstance(table.get("review_required"), bool):
            missing.append("review_required_bool")
        if not _as_list(table.get("evidence_refs")):
            missing.append("non_empty_evidence_refs")
        confidence_floor, confidence_failures = _confidence_floor(table.get("confidence"))
        missing.extend(confidence_failures)
        if table.get("criticality") == "critical" and confidence_floor is not None and confidence_floor < 0.95 and table.get("review_required") is not True:
            missing.append("critical_low_confidence_requires_review")

        columns = _as_list(table.get("columns"))
        column_ids: set[str] = set()
        if not columns:
            missing.append("non_empty_columns")
        for column_index, column in enumerate(columns):
            if not isinstance(column, dict):
                failures.append({"table_id": table.get("table_id"), "column_index": column_index, "reason": "column_not_object"})
                continue
            column_missing = [key for key in ("column_id", "name") if key not in column or column.get(key) in (None, "")]
            column_id = str(column.get("column_id", ""))
            if column_id:
                if column_id in column_ids:
                    column_missing.append("unique_column_id")
                column_ids.add(column_id)
            if column_missing:
                failures.append(
                    {
                        "table_id": table.get("table_id"),
                        "column_index": column_index,
                        "missing_or_invalid": sorted(set(column_missing)),
                    }
                )

        rows = _as_list(table.get("rows"))
        if not rows and not table.get("review_required"):
            missing.append("rows_required_when_not_review_required")
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                failures.append({"table_id": table.get("table_id"), "row_index": row_index, "reason": "row_not_object"})
                continue
            row_missing = [key for key in ("row_id", "row_key", "evidence_refs", "cells") if key not in row or row.get(key) in (None, "")]
            if not _as_list(row.get("evidence_refs")):
                row_missing.append("non_empty_evidence_refs")
            cells = _as_list(row.get("cells"))
            if not cells:
                failures.append({"table_id": table.get("table_id"), "row_id": row.get("row_id"), "reason": "row_without_cells"})
            if cells and column_ids and len(cells) != len(column_ids):
                row_missing.append("cell_count_matches_columns")
            if row_missing:
                failures.append(
                    {
                        "table_id": table.get("table_id"),
                        "row_id": row.get("row_id"),
                        "row_index": row_index,
                        "missing_or_invalid": sorted(set(row_missing)),
                    }
                )
            if not cells:
                continue
            for cell_index, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    failures.append(
                        {
                            "table_id": table.get("table_id"),
                            "row_id": row.get("row_id"),
                            "cell_index": cell_index,
                            "reason": "cell_not_object",
                        }
                    )
                    continue
                cell_missing = [key for key in ("column_id", "evidence_refs") if key not in cell or cell.get(key) in (None, "")]
                if "raw_value" not in cell or cell.get("raw_value") is None:
                    cell_missing.append("raw_value")
                if column_ids and str(cell.get("column_id")) not in column_ids:
                    cell_missing.append("known_column_id")
                if not _as_list(cell.get("evidence_refs")):
                    cell_missing.append("non_empty_evidence_refs")
                if cell_missing:
                    failures.append(
                        {
                            "table_id": table.get("table_id"),
                            "row_id": row.get("row_id"),
                            "cell_index": cell_index,
                            "missing_or_invalid": sorted(set(cell_missing)),
                        }
                    )
        if missing:
            failures.append({"table_id": table.get("table_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _table_structure_risk_route_failures(
    tables: list[Any],
    risks: list[Any],
    review_tasks: list[Any],
) -> list[dict[str, Any]]:
    table_risk_targets = _risk_targets(
        risks,
        risk_types={"table_structure_validation_failed", "table_structure_unrecovered", "nutrition_table_rows_incomplete"},
    )
    high_risk_targets = _risk_targets(risks, risk_levels={"high"})
    high_review_targets = _review_targets(review_tasks, risk_levels={"high"}, required_only=True)
    failures = []
    for index, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("table_id") or f"table_{index + 1}")
        rows = _as_list(table.get("rows"))
        requires_route = (
            table.get("review_required") is True
            or table.get("status") in {"manual_review_required", "uncertain", "conflict", "cannot_verify"}
            or not rows
        )
        if not requires_route:
            continue
        missing = []
        if ("table", table_id) not in table_risk_targets:
            missing.append("table_structure_risk")
        if table.get("risk_level") == "high":
            if ("table", table_id) not in high_risk_targets:
                missing.append("high_risk")
            if ("table", table_id) not in high_review_targets:
                missing.append("high_risk_review_task")
        if missing:
            failures.append({"table_id": table_id, "missing_or_invalid": sorted(set(missing))})
    return failures


def _table_evidence_ref_failures(tables: list[Any], evidence_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        missing_table_refs = [ref for ref in _as_list(table.get("evidence_refs")) if str(ref) not in evidence_ids]
        if missing_table_refs:
            failures.append({"table_id": table.get("table_id"), "missing_refs": missing_table_refs})
        for row in _as_list(table.get("rows")):
            if not isinstance(row, dict):
                continue
            missing_row_refs = [ref for ref in _as_list(row.get("evidence_refs")) if str(ref) not in evidence_ids]
            if missing_row_refs:
                failures.append(
                    {
                        "table_id": table.get("table_id"),
                        "row_id": row.get("row_id"),
                        "missing_refs": missing_row_refs,
                    }
                )
            for cell_index, cell in enumerate(_as_list(row.get("cells"))):
                if not isinstance(cell, dict):
                    continue
                missing_cell_refs = [ref for ref in _as_list(cell.get("evidence_refs")) if str(ref) not in evidence_ids]
                if missing_cell_refs:
                    failures.append(
                        {
                            "table_id": table.get("table_id"),
                            "row_id": row.get("row_id"),
                            "cell_index": cell_index,
                            "missing_refs": missing_cell_refs,
                        }
                    )
    return failures


def _table_cell_text_failures(tables: list[Any], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        for row in _as_list(table.get("rows")):
            if not isinstance(row, dict):
                continue
            for cell_index, cell in enumerate(_as_list(row.get("cells"))):
                if not isinstance(cell, dict):
                    continue
                evidence_text = _joined_evidence_text(_as_list(cell.get("evidence_refs")), evidence_by_id)
                if evidence_text is None:
                    continue
                raw_value = str(cell.get("raw_value", ""))
                if raw_value and _compact_text(raw_value) not in _compact_text(evidence_text):
                    failures.append(
                        {
                            "table_id": table.get("table_id"),
                            "row_id": row.get("row_id"),
                            "cell_index": cell_index,
                            "expected_source_text": evidence_text,
                            "actual_raw_value": cell.get("raw_value"),
                        }
                    )
    return failures


def _requirement_contract_failures(requirements: list[Any]) -> list[dict[str, Any]]:
    required_non_empty = [
        "requirement_id",
        "requirement_type",
        "requirement_text",
        "status",
        "confidence",
        "verification_status",
        "risk_level",
        "evidence_refs",
    ]
    required_present = ["target", "review_required"]
    failures = []
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            failures.append({"index": index, "reason": "requirement_not_object"})
            continue
        missing = [key for key in required_non_empty if key not in requirement or requirement.get(key) in (None, "")]
        missing.extend(key for key in required_present if key not in requirement)
        if requirement.get("requirement_type") and requirement.get("requirement_type") not in REQUIREMENT_TYPES:
            missing.append("valid_requirement_type")
        if requirement.get("status") and requirement.get("status") not in FIELD_STATUSES:
            missing.append("valid_status")
        if requirement.get("verification_status") != "not_verified_in_mvp":
            missing.append("verification_status_not_verified_in_mvp")
        if requirement.get("risk_level") and requirement.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if not isinstance(requirement.get("review_required"), bool):
            missing.append("review_required_bool")
        confidence_floor, confidence_failures = _confidence_floor(requirement.get("confidence"))
        missing.extend(confidence_failures)
        if confidence_floor is not None and confidence_floor < 0.90 and requirement.get("review_required") is not True:
            missing.append("low_confidence_requires_review")
        if requirement.get("target") is not None and not isinstance(requirement.get("target"), str):
            missing.append("target_string_or_null")
        if not _as_list(requirement.get("evidence_refs")):
            missing.append("non_empty_evidence_refs")
        if missing:
            failures.append(
                {
                    "requirement_id": requirement.get("requirement_id"),
                    "missing_or_invalid": sorted(set(missing)),
                }
            )
    return failures


def _requirement_evidence_ref_failures(requirements: list[Any], evidence_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        missing_refs = [ref for ref in _as_list(requirement.get("evidence_refs")) if str(ref) not in evidence_ids]
        if missing_refs:
            failures.append({"requirement_id": requirement.get("requirement_id"), "missing_refs": missing_refs})
    return failures


def _requirement_text_failures(requirements: list[Any], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        evidence_text = _joined_evidence_text(_as_list(requirement.get("evidence_refs")), evidence_by_id)
        if evidence_text is None:
            continue
        if requirement.get("requirement_text") != evidence_text:
            failures.append(
                {
                    "requirement_id": requirement.get("requirement_id"),
                    "expected_from_evidence": evidence_text,
                    "actual_requirement_text": requirement.get("requirement_text"),
                }
            )
    return failures


def _region_contract_failures(regions: list[Any]) -> list[dict[str, Any]]:
    failures = []
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            failures.append({"index": index, "reason": "region_not_object"})
            continue
        missing = [
            key
            for key in (
                "region_id",
                "region_type",
                "display_name",
                "page",
                "source_span_ids",
                "bbox_status",
                "confidence",
                "status",
                "risk_level",
                "review_required",
                "evidence_refs",
                "fields",
                "tables",
                "entities",
                "assignment_status",
            )
            if key not in region or region.get(key) is None
        ]
        if region.get("region_type") and region.get("region_type") not in REGION_TYPES:
            missing.append("valid_region_type")
        if not isinstance(region.get("page"), int) or int(region.get("page", 0)) <= 0:
            missing.append("valid_page")
        if not _as_list(region.get("source_span_ids")):
            missing.append("non_empty_source_span_ids")
        if not _as_list(region.get("evidence_refs")):
            missing.append("non_empty_evidence_refs")
        if region.get("bbox_status") and region.get("bbox_status") not in BBOX_STATUSES:
            missing.append("valid_bbox_status")
        if region.get("bbox_status") == "available" and not _has_bbox_coordinates(region):
            missing.append("bbox_coordinates")
        if not _is_number(region.get("confidence")):
            missing.append("numeric_confidence")
        elif not 0 <= float(region.get("confidence")) <= 1:
            missing.append("confidence_0_to_1")
        if region.get("status") and region.get("status") not in FIELD_STATUSES:
            missing.append("valid_status")
        if region.get("risk_level") and region.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if not isinstance(region.get("review_required"), bool):
            missing.append("review_required_bool")
        if not isinstance(region.get("fields"), list):
            missing.append("fields_list")
        if not isinstance(region.get("tables"), list):
            missing.append("tables_list")
        if not isinstance(region.get("entities"), list):
            missing.append("entities_list")
        if region.get("region_type") == "package_panel":
            panel_missing = [key for key in ("panel_id", "panel_name", "panel_type") if key not in region or region.get(key) in (None, "")]
            missing.extend(panel_missing)
            if region.get("panel_id") != region.get("region_id"):
                missing.append("panel_id_matches_region_id")
            if region.get("panel_name") != region.get("display_name"):
                missing.append("panel_name_matches_display_name")
            if region.get("panel_type") != "package_panel":
                missing.append("panel_type_package_panel")
            if region.get("assignment_status") == "uncertain" and region.get("review_required") is not True:
                missing.append("uncertain_panel_assignment_requires_review")
        if region.get("bbox_pdf") and not isinstance(region.get("bbox_pdf"), dict):
            missing.append("bbox_pdf_object")
        if region.get("bbox_normalized") and not isinstance(region.get("bbox_normalized"), dict):
            missing.append("bbox_normalized_object")
        if missing:
            failures.append({"region_id": region.get("region_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _region_source_ref_failures(regions: list[Any], source_span_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        missing_refs = [span_id for span_id in _as_list(region.get("source_span_ids")) if str(span_id) not in source_span_ids]
        if missing_refs:
            failures.append({"region_id": region.get("region_id"), "missing_source_span_ids": missing_refs})
    return failures


def _region_artifact_ref_failures(
    regions: list[Any],
    field_ids: set[str],
    table_ids: set[str],
    entity_ids: set[str],
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    failures = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        missing_evidence_refs = [ref for ref in _as_list(region.get("evidence_refs")) if str(ref) not in evidence_ids]
        missing_field_ids = [field_id for field_id in _as_list(region.get("fields")) if str(field_id) not in field_ids]
        missing_table_ids = [table_id for table_id in _as_list(region.get("tables")) if str(table_id) not in table_ids]
        missing_entity_ids = [entity_id for entity_id in _as_list(region.get("entities")) if str(entity_id) not in entity_ids]
        item_failures = []
        if missing_evidence_refs:
            item_failures.append({"reason": "missing_evidence_refs", "refs": missing_evidence_refs})
        if missing_field_ids:
            item_failures.append({"reason": "missing_field_ids", "field_ids": missing_field_ids})
        if missing_table_ids:
            item_failures.append({"reason": "missing_table_ids", "table_ids": missing_table_ids})
        if missing_entity_ids:
            item_failures.append({"reason": "missing_entity_ids", "entity_ids": missing_entity_ids})
        if item_failures:
            failures.append({"region_id": region.get("region_id"), "failures": item_failures})
    return failures


def _visual_document_graph_contract_failures(graph: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("graph_id", "schema_version", "node_count", "edge_count", "nodes", "edges") if key not in graph or graph.get(key) in (None, "")]
    nodes = _as_list(graph.get("nodes"))
    edges = _as_list(graph.get("edges"))
    if graph.get("node_count") != len(nodes):
        missing.append("node_count_matches_nodes")
    if graph.get("edge_count") != len(edges):
        missing.append("edge_count_matches_edges")
    if missing:
        failures.append({"missing_or_invalid": sorted(set(missing))})

    node_ids: set[str] = set()
    duplicate_node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            failures.append({"index": index, "reason": "vdg_node_not_object"})
            continue
        node_missing = [key for key in ("node_id", "node_type", "page", "status") if key not in node or node.get(key) in (None, "")]
        if node.get("node_id") in node_ids:
            duplicate_node_ids.add(str(node.get("node_id")))
        if node.get("node_id"):
            node_ids.add(str(node.get("node_id")))
        if not isinstance(node.get("page"), int) or int(node.get("page", 0)) <= 0:
            node_missing.append("valid_page")
        if node_missing:
            failures.append({"node_id": node.get("node_id"), "missing_or_invalid": sorted(set(node_missing))})
    for node_id in sorted(duplicate_node_ids):
        failures.append({"node_id": node_id, "reason": "duplicate_node_id"})

    edge_ids: set[str] = set()
    duplicate_edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            failures.append({"index": index, "reason": "vdg_edge_not_object"})
            continue
        edge_missing = [key for key in ("edge_id", "source_node_id", "target_node_id", "edge_type") if key not in edge or edge.get(key) in (None, "")]
        if edge.get("edge_id") in edge_ids:
            duplicate_edge_ids.add(str(edge.get("edge_id")))
        if edge.get("edge_id"):
            edge_ids.add(str(edge.get("edge_id")))
        if edge.get("source_node_id") and str(edge.get("source_node_id")) not in node_ids:
            edge_missing.append("source_node_resolves")
        if edge.get("target_node_id") and str(edge.get("target_node_id")) not in node_ids:
            edge_missing.append("target_node_resolves")
        if edge_missing:
            failures.append({"edge_id": edge.get("edge_id"), "missing_or_invalid": sorted(set(edge_missing))})
    for edge_id in sorted(duplicate_edge_ids):
        failures.append({"edge_id": edge_id, "reason": "duplicate_edge_id"})

    return failures


def _vdg_quality_report_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("report_version", "status", "source_span_coverage_rate", "edge_ref_status", "issues", "checks") if key not in report]
    if report.get("status") not in {"pass", "review_required", "fail"}:
        missing.append("valid_status")
    if report.get("status") == "fail":
        missing.append("status_not_fail")
    coverage = report.get("source_span_coverage_rate")
    if not _is_number(coverage) or not 0 <= float(coverage) <= 1:
        missing.append("source_span_coverage_rate_0_to_1")
    if report.get("edge_ref_status") not in {"pass", "fail"}:
        missing.append("valid_edge_ref_status")
    if not isinstance(report.get("issues"), list):
        missing.append("issues_list")
    if not isinstance(report.get("checks"), list):
        missing.append("checks_list")
    if missing:
        failures.append({"missing_or_invalid": sorted(set(missing))})
    return failures


def _vdg_agent_context_failures(context: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("context_version", "vdg_quality_status", "agent_readiness", "candidate_field_groups", "table_candidates", "quality_issues") if key not in context]
    if context.get("vdg_quality_status") not in {"pass", "review_required", "fail"}:
        missing.append("valid_vdg_quality_status")
    for key in ("candidate_field_groups", "table_candidates", "quality_issues"):
        if key in context and not isinstance(context.get(key), list):
            missing.append(f"{key}_list")
    if missing:
        failures.append({"missing_or_invalid": sorted(set(missing))})
    return failures


def _vdg_consumption_report_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("report_version", "status", "consumable_node_count", "extracted_node_count", "extracted_coverage_rate", "status_counts") if key not in report]
    if report.get("status") not in {"pass", "review_required"}:
        missing.append("valid_status")
    coverage = report.get("extracted_coverage_rate")
    if not _is_number(coverage) or not 0 <= float(coverage) <= 1:
        missing.append("extracted_coverage_rate_0_to_1")
    if not isinstance(report.get("status_counts"), dict):
        missing.append("status_counts_object")
    if missing:
        failures.append({"missing_or_invalid": sorted(set(missing))})
    return failures


def _label_text_scope_reference_failures(reference: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("reference_version", "scope_policy", "in_scope_categories", "out_of_scope_categories", "field_catalog", "entity_catalog", "table_catalog") if key not in reference]
    policy = _as_dict(reference.get("scope_policy"))
    if not str(reference.get("reference_version") or "").startswith("label_text_scope_reference_"):
        missing.append("valid_reference_version")
    if policy.get("reference_is_not_evidence") is not True:
        missing.append("reference_is_not_evidence_true")
    if policy.get("template_placeholders_are_not_values") is not True:
        missing.append("template_placeholders_are_not_values_true")
    for key in ("in_scope_categories", "out_of_scope_categories", "field_catalog", "entity_catalog", "table_catalog"):
        if key in reference and not isinstance(reference.get(key), list):
            missing.append(f"{key}_list")
    if missing:
        failures.append({"missing_or_invalid": sorted(set(missing))})
    return failures


def _label_text_scope_agent_context_failures(context: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("context_version", "reference_version", "primary_rule", "in_scope_categories", "out_of_scope_categories") if key not in context]
    if context.get("context_version") != "label_text_scope_agent_context_v0.1":
        missing.append("valid_context_version")
    if context.get("reference_is_not_evidence") is not True:
        missing.append("reference_is_not_evidence_true")
    for key in ("in_scope_categories", "out_of_scope_categories"):
        if key in context and not isinstance(context.get(key), list):
            missing.append(f"{key}_list")
    if missing:
        failures.append({"missing_or_invalid": sorted(set(missing))})
    return failures


def _label_text_scope_report_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    missing = [key for key in ("report_version", "reference_version", "status", "extracted_out_of_scope_count", "ignored_noise_node_count", "unknown_scope_node_count", "scope_gate_rejected_count", "node_scope_decisions", "checks") if key not in report]
    if report.get("report_version") != "label_text_scope_report_v0.1":
        missing.append("valid_report_version")
    if report.get("status") not in {"pass", "review_required", "fail"}:
        missing.append("valid_status")
    if report.get("status") == "fail":
        missing.append("status_not_fail")
    if report.get("extracted_out_of_scope_count") != 0:
        missing.append("extracted_out_of_scope_count_zero")
    for key in ("ignored_noise_node_count", "unknown_scope_node_count", "scope_gate_rejected_count"):
        if not _is_non_negative_int(report.get(key)):
            missing.append(f"{key}_non_negative_int")
    decisions = _as_list(report.get("node_scope_decisions"))
    if "node_scope_decisions" in report and not isinstance(report.get("node_scope_decisions"), list):
        missing.append("node_scope_decisions_list")
    for index, item in enumerate(decisions):
        if not isinstance(item, dict):
            failures.append({"index": index, "reason": "node_scope_decision_not_object"})
            continue
        item_missing = [key for key in ("node_id", "scope_status", "scope_category", "reason", "decided_by") if not item.get(key)]
        if item.get("scope_status") not in {"in_scope_label_text", "out_of_scope_noise", "unknown_scope", "ignored"}:
            item_missing.append("valid_scope_status")
        if item_missing:
            failures.append({"node_id": item.get("node_id"), "missing_or_invalid": sorted(set(item_missing))})
    if "checks" in report and not isinstance(report.get("checks"), list):
        missing.append("checks_list")
    if missing:
        failures.insert(0, {"missing_or_invalid": sorted(set(missing))})
    return failures


def _region_vdg_ref_failures(regions: list[Any], graph: dict[str, Any]) -> list[dict[str, Any]]:
    region_nodes = {
        str(node.get("node_id")): node
        for node in _as_list(graph.get("nodes"))
        if isinstance(node, dict) and node.get("node_type") == "region" and node.get("node_id")
    }
    failures = []
    for region in regions:
        if not isinstance(region, dict) or not region.get("region_id"):
            continue
        node = region_nodes.get(str(region.get("region_id")))
        if not node:
            failures.append({"region_id": region.get("region_id"), "reason": "missing_region_node"})
            continue
        if node.get("region_type") != region.get("region_type"):
            failures.append(
                {
                    "region_id": region.get("region_id"),
                    "reason": "region_type_mismatch",
                    "expected": region.get("region_type"),
                    "actual": node.get("region_type"),
                }
            )
        if set(_as_list(node.get("source_span_ids"))) != set(_as_list(region.get("source_span_ids"))):
            failures.append(
                {
                    "region_id": region.get("region_id"),
                    "reason": "source_span_ids_mismatch",
                    "expected": region.get("source_span_ids"),
                    "actual": node.get("source_span_ids"),
                }
            )
    return failures


def _field_group_contract_failures(field_groups: list[Any]) -> list[dict[str, Any]]:
    failures = []
    for index, group in enumerate(field_groups):
        if not isinstance(group, dict):
            failures.append({"index": index, "reason": "field_group_not_object"})
            continue
        missing = [key for key in ("group_id", "group_type", "fields", "linked_table_ids") if key not in group or group.get(key) is None]
        if not isinstance(group.get("fields"), list):
            missing.append("fields_list")
        if not isinstance(group.get("linked_table_ids"), list):
            missing.append("linked_table_ids_list")
        if missing:
            failures.append({"group_id": group.get("group_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _field_group_ref_failures(
    field_groups: list[Any],
    field_ids: set[str],
    standard_item_by_id: dict[str, dict[str, Any]],
    table_ids: set[str],
) -> list[dict[str, Any]]:
    failures = []
    for group in field_groups:
        if not isinstance(group, dict):
            continue
        group_id = group.get("group_id")
        for field_index, group_field in enumerate(_as_list(group.get("fields"))):
            if not isinstance(group_field, dict):
                failures.append({"group_id": group_id, "field_index": field_index, "reason": "group_field_not_object"})
                continue
            field_id = str(group_field.get("field_id", ""))
            standard_item_id = str(group_field.get("standard_item_id", ""))
            field_failures = []
            if field_id not in field_ids:
                field_failures.append({"reason": "unknown_field_id", "field_id": group_field.get("field_id")})
            linked_item = standard_item_by_id.get(standard_item_id)
            if not linked_item:
                field_failures.append({"reason": "unknown_standard_item_id", "standard_item_id": group_field.get("standard_item_id")})
            elif linked_item.get("field_id") != group_field.get("field_id"):
                field_failures.append(
                    {
                        "reason": "field_id_mismatch",
                        "expected": linked_item.get("field_id"),
                        "actual": group_field.get("field_id"),
                    }
                )
            if field_failures:
                failures.append({"group_id": group_id, "field_index": field_index, "failures": field_failures})
        missing_tables = [table_id for table_id in _as_list(group.get("linked_table_ids")) if str(table_id) not in table_ids]
        if missing_tables:
            failures.append({"group_id": group_id, "missing_table_ids": missing_tables})
    return failures


def _field_group_text_failures(
    field_groups: list[Any],
    fields: dict[str, Any],
    standard_item_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for group in field_groups:
        if not isinstance(group, dict):
            continue
        for field_index, group_field in enumerate(_as_list(group.get("fields"))):
            if not isinstance(group_field, dict):
                continue
            field = _as_dict(fields.get(str(group_field.get("field_id"))))
            linked_item = standard_item_by_id.get(str(group_field.get("standard_item_id", "")))
            if not field or not linked_item:
                continue
            text = group_field.get("text")
            expected = linked_item.get("text")
            if text != expected or text != field.get("raw_value"):
                failures.append(
                    {
                        "group_id": group.get("group_id"),
                        "field_index": field_index,
                        "field_id": group_field.get("field_id"),
                        "expected_text": expected,
                        "actual_text": text,
                    }
                )
    return failures


def _list_contract_failures(lists: list[Any]) -> list[dict[str, Any]]:
    failures = []
    for index, item_list in enumerate(lists):
        if not isinstance(item_list, dict):
            failures.append({"index": index, "reason": "list_not_object"})
            continue
        missing = [key for key in ("list_id", "list_type", "item_count", "items") if key not in item_list or item_list.get(key) is None]
        items = _as_list(item_list.get("items"))
        if not isinstance(item_list.get("items"), list):
            missing.append("items_list")
        if item_list.get("item_count") != len(items):
            missing.append("item_count_matches_items")
        for item_index, list_item in enumerate(items):
            if not isinstance(list_item, dict):
                failures.append({"list_id": item_list.get("list_id"), "item_index": item_index, "reason": "list_item_not_object"})
                continue
            item_missing = [key for key in ("index", "group_id") if key not in list_item or list_item.get(key) is None]
            if item_missing:
                failures.append(
                    {
                        "list_id": item_list.get("list_id"),
                        "item_index": item_index,
                        "missing_or_invalid": item_missing,
                    }
                )
        if missing:
            failures.append({"list_id": item_list.get("list_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _list_group_ref_failures(lists: list[Any], field_groups: list[Any]) -> list[dict[str, Any]]:
    failures = []
    group_by_id = {
        str(group.get("group_id")): group
        for group in field_groups
        if isinstance(group, dict) and group.get("group_id")
    }
    for item_list in lists:
        if not isinstance(item_list, dict):
            continue
        for item_index, list_item in enumerate(_as_list(item_list.get("items"))):
            if not isinstance(list_item, dict):
                continue
            group = group_by_id.get(str(list_item.get("group_id", "")))
            if not group:
                failures.append(
                    {
                        "list_id": item_list.get("list_id"),
                        "item_index": item_index,
                        "reason": "unknown_group_id",
                        "group_id": list_item.get("group_id"),
                    }
                )
                continue
            container_text = group.get("container_text")
            if container_text is not None and list_item.get("text") != container_text:
                failures.append(
                    {
                        "list_id": item_list.get("list_id"),
                        "item_index": item_index,
                        "group_id": list_item.get("group_id"),
                        "expected_text": container_text,
                        "actual_text": list_item.get("text"),
                    }
                )
    return failures


def _revision_block_contract_failures(revision_blocks: list[Any]) -> list[dict[str, Any]]:
    failures = []
    valid_status_by_role = {
        "before": "historical_reference",
        "after": "current_standard",
    }
    for index, block in enumerate(revision_blocks):
        if not isinstance(block, dict):
            failures.append({"index": index, "reason": "revision_block_not_object"})
            continue
        missing = [
            key
            for key in (
                "revision_block_id",
                "region_id",
                "revision_role",
                "revision_status",
                "display_name",
                "fields",
                "is_current_standard",
                "status",
                "risk_level",
                "review_required",
                "evidence_refs",
                "source_span_ids",
                "assignment_status",
            )
            if key not in block or block.get(key) is None
        ]
        role = block.get("revision_role")
        if role not in {"before", "after"}:
            missing.append("valid_revision_role")
        if block.get("revision_status") != valid_status_by_role.get(role):
            missing.append("revision_status_matches_role")
        if not isinstance(block.get("is_current_standard"), bool):
            missing.append("is_current_standard_bool")
        elif block.get("is_current_standard") is not (role == "after"):
            missing.append("is_current_standard_matches_role")
        if block.get("status") and block.get("status") not in FIELD_STATUSES:
            missing.append("valid_status")
        if block.get("risk_level") and block.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if not isinstance(block.get("review_required"), bool):
            missing.append("review_required_bool")
        if not isinstance(block.get("fields"), list):
            missing.append("fields_list")
        if not _as_list(block.get("evidence_refs")):
            missing.append("non_empty_evidence_refs")
        if not _as_list(block.get("source_span_ids")):
            missing.append("non_empty_source_span_ids")
        if block.get("assignment_status") == "region_detected_field_assignment_pending":
            if block.get("status") != "uncertain":
                missing.append("pending_assignment_requires_uncertain_status")
            if block.get("risk_level") != "high":
                missing.append("pending_assignment_requires_high_risk")
            if block.get("review_required") is not True:
                missing.append("pending_assignment_requires_review")
        if missing:
            failures.append({"revision_block_id": block.get("revision_block_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _revision_block_ref_failures(
    revision_blocks: list[Any],
    field_ids: set[str],
    region_ids: set[str],
    evidence_ids: set[str],
    source_span_ids: set[str],
) -> list[dict[str, Any]]:
    failures = []
    role_by_field_id: dict[str, str] = {}
    for block in revision_blocks:
        if not isinstance(block, dict):
            continue
        block_id = block.get("revision_block_id")
        missing_evidence_refs = [ref for ref in _as_list(block.get("evidence_refs")) if str(ref) not in evidence_ids]
        missing_source_span_ids = [span_id for span_id in _as_list(block.get("source_span_ids")) if str(span_id) not in source_span_ids]
        item_failures = []
        if str(block.get("region_id")) not in region_ids:
            item_failures.append({"reason": "unknown_region_id", "region_id": block.get("region_id")})
        if missing_evidence_refs:
            item_failures.append({"reason": "missing_evidence_refs", "refs": missing_evidence_refs})
        if missing_source_span_ids:
            item_failures.append({"reason": "missing_source_span_ids", "source_span_ids": missing_source_span_ids})

        for field_index, field_ref in enumerate(_as_list(block.get("fields"))):
            if not isinstance(field_ref, dict):
                item_failures.append({"reason": "revision_field_not_object", "field_index": field_index})
                continue
            field_id = str(field_ref.get("field_id", ""))
            if field_id not in field_ids:
                item_failures.append({"reason": "unknown_field_id", "field_id": field_ref.get("field_id")})
                continue
            previous_role = role_by_field_id.get(field_id)
            current_role = str(block.get("revision_role"))
            if previous_role and previous_role != current_role:
                item_failures.append(
                    {
                        "reason": "field_assigned_to_multiple_revision_roles",
                        "field_id": field_id,
                        "previous_role": previous_role,
                        "current_role": current_role,
                    }
                )
            role_by_field_id[field_id] = current_role
        if item_failures:
            failures.append({"revision_block_id": block_id, "failures": item_failures})
    return failures


def _auto_ingest_ref_failures(auto_ingest: dict[str, Any], standard_items: list[Any]) -> list[dict[str, Any]]:
    failures = []
    item_by_id = {item.get("id"): item for item in standard_items if isinstance(item, dict) and item.get("id")}
    field_ids = {item.get("field_id") for item in standard_items if isinstance(item, dict) and item.get("field_id")}
    ingest_items = [
        *[(item, "candidate") for item in _as_list(auto_ingest.get("candidates"))],
        *[(item, "blocked_item") for item in _as_list(auto_ingest.get("blocked_items"))],
    ]
    for item, item_type in ingest_items:
        if not isinstance(item, dict):
            failures.append({"type": item_type, "reason": "auto_ingest_item_not_object"})
            continue
        item_failures = []
        standard_item_id = item.get("standard_item_id")
        field_id = item.get("field_id")
        if standard_item_id not in item_by_id:
            item_failures.append({"reason": "unknown_standard_item_id", "standard_item_id": standard_item_id})
        if field_id not in field_ids:
            item_failures.append({"reason": "unknown_field_id", "field_id": field_id})
        linked_item = item_by_id.get(standard_item_id)
        if linked_item and field_id != linked_item.get("field_id"):
            item_failures.append({"reason": "field_id_mismatch", "expected": linked_item.get("field_id"), "actual": field_id})
        if item_failures:
            failures.append({"type": item_type, "item": item.get("candidate_id") or standard_item_id, "failures": item_failures})
    return failures


def _auto_ingest_text_failures(auto_ingest: dict[str, Any], standard_items: list[Any]) -> list[dict[str, Any]]:
    failures = []
    item_by_id = {item.get("id"): item for item in standard_items if isinstance(item, dict) and item.get("id")}
    for candidate in _as_list(auto_ingest.get("candidates")):
        if not isinstance(candidate, dict):
            continue
        linked_item = item_by_id.get(candidate.get("standard_item_id"))
        if not linked_item:
            continue
        candidate_failures = []
        if candidate.get("text") != linked_item.get("text"):
            candidate_failures.append(
                {
                    "reason": "text_mismatch",
                    "expected": linked_item.get("text"),
                    "actual": candidate.get("text"),
                }
            )
        if _as_list(candidate.get("evidence_refs")) != _as_list(linked_item.get("evidence_refs")):
            candidate_failures.append(
                {
                    "reason": "evidence_refs_mismatch",
                    "expected": linked_item.get("evidence_refs"),
                    "actual": candidate.get("evidence_refs"),
                }
            )
        if candidate_failures:
            failures.append({"candidate_id": candidate.get("candidate_id"), "failures": candidate_failures})
    return failures


def _auto_ingest_quality_gate_passed(auto_ingest: dict[str, Any], risks: list[Any]) -> bool:
    if not auto_ingest.get("document_auto_ingest_allowed"):
        return True
    snapshot = _as_dict(auto_ingest.get("quality_snapshot"))
    return _high_risk_count(risks) == 0 and int(snapshot.get("high_risk_count", 0) or 0) == 0


def _stable_target_ids(
    fields: dict[str, Any],
    entities: dict[str, Any],
    tables: list[Any],
    regions: list[Any],
    requirements: list[Any],
    missing_item_report: dict[str, Any],
    revision_blocks: list[Any],
    document: dict[str, Any],
    page_images: dict[str, Any],
    source_layers: dict[str, Any],
    source_consistency_report: dict[str, Any],
    table_parser: dict[str, Any],
) -> dict[str, set[str]]:
    table_ids = {
        str(table.get("table_id"))
        for table in tables
        if isinstance(table, dict) and table.get("table_id")
    }
    table_ids.update(
        str(table.get("table_layer_id"))
        for table in tables
        if isinstance(table, dict) and table.get("table_layer_id")
    )
    parser_table_layers = _as_dict(table_parser.get("table_layers"))
    table_ids.update(
        str(table.get("table_layer_id"))
        for table in _as_list(parser_table_layers.get("tables"))
        if isinstance(table, dict) and table.get("table_layer_id")
    )
    table_ids.update(
        str(table.get("table_id"))
        for table in _as_list(parser_table_layers.get("tables"))
        if isinstance(table, dict) and table.get("table_id")
    )
    document_targets = {"document"}
    if page_images:
        document_targets.add("page_images")
    if document.get("file_name"):
        document_targets.add(str(document.get("file_name")))
    targets = {
        "document": document_targets,
        "generated_schema": {"generated_schema"},
        "field": set(fields),
        "entity": {
            str(entity.get("entity_id"))
            for entity in entities.values()
            if isinstance(entity, dict) and entity.get("entity_id")
        },
        "table": table_ids,
        "region": {
            str(region.get("region_id"))
            for region in regions
            if isinstance(region, dict) and region.get("region_id")
        },
        "requirement": {
            str(requirement.get("requirement_id"))
            for requirement in requirements
            if isinstance(requirement, dict) and requirement.get("requirement_id")
        },
        "missing_field": {
            str(item.get("missing_id"))
            for item in _as_list(missing_item_report.get("missing_fields"))
            if isinstance(item, dict) and item.get("missing_id")
        },
        "missing_table": {
            str(item.get("missing_id"))
            for item in _as_list(missing_item_report.get("missing_tables"))
            if isinstance(item, dict) and item.get("missing_id")
        },
        "revision_block": {
            str(block.get("revision_block_id") or f"revision_{block.get('revision_role')}")
            for block in revision_blocks
            if isinstance(block, dict) and block.get("revision_role")
        },
        "table_parser": {"table_quality_report", "table_layers"},
        "source_layer": {
            "source_layers",
            *(
                str(issue.get("issue_id"))
                for issue in _as_list(source_layers.get("source_issues"))
                if isinstance(issue, dict) and issue.get("issue_id")
            ),
        },
        "source_consistency": {
            "source_consistency",
            *(
                str(issue.get("issue_id"))
                for issue in _as_list(source_consistency_report.get("issues"))
                if isinstance(issue, dict) and issue.get("issue_id")
            ),
        },
    }
    missing_target_types = RISK_TARGET_TYPES - set(targets)
    if missing_target_types:
        targets.update({target_type: set() for target_type in missing_target_types})
    return targets


def _known_source_span_ids(source_layers: dict[str, Any], evidence: list[Any], graph: dict[str, Any]) -> set[str]:
    span_ids = {
        str(span.get("span_id"))
        for span in _as_list(source_layers.get("spans"))
        if isinstance(span, dict) and span.get("span_id")
    }
    for item in evidence:
        if not isinstance(item, dict):
            continue
        span_ids.update(str(span_id) for span_id in _as_list(item.get("source_node_ids")) if span_id)
    for node in _as_list(graph.get("nodes")):
        if not isinstance(node, dict):
            continue
        if node.get("node_type") == "text_span" and node.get("node_id"):
            span_ids.add(str(node.get("node_id")))
        span_ids.update(str(span_id) for span_id in _as_list(node.get("source_span_ids")) if span_id)
    return span_ids


def _risk_contract_failures(risks: list[Any]) -> list[dict[str, Any]]:
    failures = []
    for index, risk in enumerate(risks):
        if not isinstance(risk, dict):
            failures.append({"index": index, "reason": "risk_not_object"})
            continue
        missing = [
            key
            for key in ("risk_id", "target_type", "target_id", "risk_level", "risk_type", "message", "evidence_refs")
            if key not in risk or risk.get(key) is None
        ]
        if risk.get("risk_level") and risk.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if not isinstance(risk.get("evidence_refs"), list):
            missing.append("evidence_refs_list")
        if missing:
            failures.append({"risk_id": risk.get("risk_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _review_task_contract_failures(review_tasks: list[Any]) -> list[dict[str, Any]]:
    failures = []
    for index, task in enumerate(review_tasks):
        if not isinstance(task, dict):
            failures.append({"index": index, "reason": "review_task_not_object"})
            continue
        missing = [
            key
            for key in ("task_id", "target_type", "target_id", "risk_level", "reason", "required", "evidence_refs")
            if key not in task or task.get(key) is None
        ]
        if task.get("risk_level") and task.get("risk_level") not in RISK_LEVELS:
            missing.append("valid_risk_level")
        if not isinstance(task.get("required"), bool):
            missing.append("required_bool")
        if not isinstance(task.get("evidence_refs"), list):
            missing.append("evidence_refs_list")
        if missing:
            failures.append({"task_id": task.get("task_id"), "missing_or_invalid": sorted(set(missing))})
    return failures


def _risk_targets(
    risks: list[Any],
    *,
    risk_levels: set[str] | None = None,
    risk_types: set[str] | None = None,
) -> set[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        if risk_levels is not None and str(risk.get("risk_level")) not in risk_levels:
            continue
        if risk_types is not None and str(risk.get("risk_type")) not in risk_types:
            continue
        targets.add((str(risk.get("target_type")), str(risk.get("target_id"))))
    return targets


def _review_targets(
    review_tasks: list[Any],
    *,
    risk_levels: set[str] | None = None,
    required_only: bool = False,
) -> set[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()
    for task in review_tasks:
        if not isinstance(task, dict):
            continue
        if required_only and task.get("required") is not True:
            continue
        if risk_levels is not None and str(task.get("risk_level")) not in risk_levels:
            continue
        targets.add((str(task.get("target_type")), str(task.get("target_id"))))
    return targets


def _risk_evidence_ref_failures(risks: list[Any], evidence_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        missing_refs = [ref for ref in _as_list(risk.get("evidence_refs")) if str(ref) not in evidence_ids]
        if missing_refs:
            failures.append({"risk_id": risk.get("risk_id"), "missing_refs": missing_refs})
    return failures


def _review_task_evidence_ref_failures(review_tasks: list[Any], evidence_ids: set[str]) -> list[dict[str, Any]]:
    failures = []
    for task in review_tasks:
        if not isinstance(task, dict):
            continue
        missing_refs = [ref for ref in _as_list(task.get("evidence_refs")) if str(ref) not in evidence_ids]
        if missing_refs:
            failures.append({"task_id": task.get("task_id"), "missing_refs": missing_refs})
    return failures


def _target_ref_failures(items: list[Any], stable_targets: dict[str, set[str]], id_key: str) -> list[dict[str, Any]]:
    failures = []
    for item in items:
        if not isinstance(item, dict):
            continue
        target_type = str(item.get("target_type", ""))
        if target_type not in stable_targets:
            failures.append(
                {
                    id_key: item.get(id_key),
                    "target_type": item.get("target_type"),
                    "target_id": item.get("target_id"),
                    "reason": "unknown_target_type",
                }
            )
            continue
        target_id = str(item.get("target_id", ""))
        if target_id not in stable_targets[target_type]:
            failures.append(
                {
                    id_key: item.get(id_key),
                    "target_type": item.get("target_type"),
                    "target_id": item.get("target_id"),
                }
            )
    return failures


def _uncovered_high_risks(risks: list[Any], review_tasks: list[Any]) -> list[dict[str, Any]]:
    review_keys = {
        (str(task.get("target_type")), str(task.get("target_id")))
        for task in review_tasks
        if isinstance(task, dict) and task.get("required") and task.get("risk_level") == "high"
    }
    failures = []
    for risk in risks:
        if not isinstance(risk, dict) or risk.get("risk_level") != "high":
            continue
        key = (str(risk.get("target_type")), str(risk.get("target_id")))
        if key not in review_keys:
            failures.append(
                {
                    "risk_id": risk.get("risk_id"),
                    "target_type": risk.get("target_type"),
                    "target_id": risk.get("target_id"),
                    "risk_type": risk.get("risk_type"),
                }
            )
    return failures


def _high_risk_count(risks: list[Any]) -> int:
    return sum(1 for risk in risks if isinstance(risk, dict) and risk.get("risk_level") == "high")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _confidence_floor(confidence: Any) -> tuple[float | None, list[str]]:
    if _is_number(confidence):
        value = float(confidence)
        if 0 <= value <= 1:
            return value, []
        return None, ["confidence_0_to_1"]
    if not isinstance(confidence, dict):
        return None, ["confidence_0_to_1_or_object"]
    numeric_values = []
    failures = []
    for confidence_key, confidence_value in confidence.items():
        if confidence_value is None:
            continue
        if not _is_number(confidence_value) or not 0 <= float(confidence_value) <= 1:
            failures.append(f"confidence_{confidence_key}_0_to_1")
            continue
        numeric_values.append(float(confidence_value))
    if not numeric_values:
        failures.append("confidence_numeric_values")
        return None, failures
    return min(numeric_values), failures


def _confidence_value(confidence: Any, key: str) -> float | None:
    if not isinstance(confidence, dict):
        return None
    value = confidence.get(key)
    if _is_number(value) and 0 <= float(value) <= 1:
        return float(value)
    return None


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and float(value) > 0


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest.lower())


def _is_iso_datetime(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _has_bbox_coordinates(item: dict[str, Any]) -> bool:
    return not _bbox_coordinate_failures(item)


def _bbox_coordinate_failures(item: dict[str, Any]) -> list[str]:
    failures = []
    bbox_pdf = item.get("bbox_pdf")
    bbox_normalized = item.get("bbox_normalized")
    if not isinstance(bbox_pdf, dict):
        failures.append("bbox_pdf_object")
    else:
        for key in ("x", "y", "width", "height", "page_width", "page_height"):
            if not _is_number(bbox_pdf.get(key)):
                failures.append(f"bbox_pdf_{key}_number")
        for key in ("width", "height", "page_width", "page_height"):
            if _is_number(bbox_pdf.get(key)) and float(bbox_pdf.get(key)) <= 0:
                failures.append(f"bbox_pdf_{key}_positive")
        for key in ("x", "y"):
            if _is_number(bbox_pdf.get(key)) and float(bbox_pdf.get(key)) < 0:
                failures.append(f"bbox_pdf_{key}_non_negative")
        if bbox_pdf.get("unit") != "pt":
            failures.append("bbox_pdf_unit_pt")
        if bbox_pdf.get("origin") != "top_left":
            failures.append("bbox_pdf_origin_top_left")

    if not isinstance(bbox_normalized, dict):
        failures.append("bbox_normalized_object")
    else:
        for key in ("x1", "y1", "x2", "y2"):
            if not _is_number(bbox_normalized.get(key)):
                failures.append(f"bbox_normalized_{key}_number")
            elif not 0 <= float(bbox_normalized.get(key)) <= 1:
                failures.append(f"bbox_normalized_{key}_0_to_1")
        if (
            _is_number(bbox_normalized.get("x1"))
            and _is_number(bbox_normalized.get("x2"))
            and float(bbox_normalized.get("x1")) > float(bbox_normalized.get("x2"))
        ):
            failures.append("bbox_normalized_x_order")
        if (
            _is_number(bbox_normalized.get("y1"))
            and _is_number(bbox_normalized.get("y2"))
            and float(bbox_normalized.get("y1")) > float(bbox_normalized.get("y2"))
        ):
            failures.append("bbox_normalized_y_order")
    return sorted(set(failures))


def _compact_text(value: str) -> str:
    return "".join(value.split())
