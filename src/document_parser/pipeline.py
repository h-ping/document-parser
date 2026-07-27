from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

from .agent_blocks import (
    block_spans,
    build_agent_blocks,
    build_block_context,
    merge_agent_plan_bodies,
    merge_block_retry_body,
    schema_induction_spans,
)
from .agent_candidates import (
    apply_rule_fallback_fields,
    build_agent_extraction_plan,
    build_rule_candidate_review_items,
    merge_agent_candidates,
)
from .agents import AuditAgent, ExtractionAgent, RepairAgent, SchemaInductionAgent
from .audit_artifacts import build_audit_input_artifact
from .auto_ingest import build_auto_ingest_candidates
from .comparison import build_comparison_index
from .compat_artifacts import (
    build_field_groups,
    build_lists_artifact,
    build_quality_report,
    build_standard_items,
    build_structured_document,
    build_tables_artifact,
    build_taxonomy_proposals,
)
from .compiler import DeterministicCompiler
from .field_validation import format_checks_for_field
from .global_reconciliation import (
    GlobalReconciliationError,
    disabled_reconciliation_report,
    validate_and_finalize_reconciliation,
)
from .html_report import write_result_preview_html
from .label_text_scope import (
    apply_label_text_scope_gate,
    build_label_text_scope_agent_context,
    load_label_text_scope_reference,
)
from .layout_evidence import ENHANCED_LAYOUT_MODE, LayoutEvidenceError, build_layout_evidence
from .llm_agents import SpanGroundedFieldAgent
from .models import (
    CompiledField,
    Evidence,
    ExtractionPlan,
    FieldDefinition,
    GeneratedSchema,
    OcrLine,
    ParseResult,
    Risk,
    ReviewTask,
    TextSpan,
    VdgNode,
    to_jsonable,
)
from .missing_fields import build_missing_item_report, missing_item_validation_checks, risks_from_missing_item_report
from .mvp_metrics import build_mvp_acceptance_metrics
from .ocr import OcrClient, OcrError
from .output_contract import build_output_contract_validation_report
from .page_render import render_page_images
from .pdf import PdfPerceptionReader
from .repair_artifacts import build_repair_agent_candidates, build_repair_attempts_artifact, build_repair_plan_patches, build_repair_trace_artifact, build_repaired_source_layers
from .schema_audit import build_schema_audit
from .semantic_review import (
    build_semantic_review_input,
    build_semantic_review_report,
    call_semantic_review_agent,
    deterministic_semantic_findings,
    merge_semantic_findings,
    semantic_review_validation_checks,
)
from .source_consistency import build_source_consistency_report
from .source_fusion import build_source_fusion
from .source_artifacts import build_coverage_map, build_source_layers
from .schema_artifacts import write_schema_artifacts
from .structures import (
    assign_region_memberships,
    attach_region_evidence,
    build_entities,
    build_repair_plan,
    build_requirements,
    build_revision_blocks,
    build_structure_audit,
    content_item_names,
    detect_regions,
    extract_nutrition_tables_from_layers,
)
from .table_parser import build_table_parser_outputs
from .utils import sha256_file, sha256_text, stable_id, write_json
from .vdg import build_visual_document_graph
from .vdg_quality import (
    apply_vdg_boundary_gate,
    build_pre_agent_vdg_artifacts,
    build_vdg_consumption_report,
    vdg_node_coverage_validation_checks,
    vdg_quality_validation_checks,
)


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
JSON_EXPORT_CONTRACT_CHECKS = [
    "machine_parseable_json",
    "root_keys_present",
    "evidence_refs_resolve",
    "risk_targets_resolve",
    "review_task_targets_resolve",
    "no_guessing",
]


class ParseError(RuntimeError):
    pass


class DocumentParser:
    def __init__(
        self,
        ocr_client: OcrClient,
        llm_agent: SpanGroundedFieldAgent | None = None,
        pdf_reader: PdfPerceptionReader | None = None,
        max_repair_rounds: int = 2,
    ) -> None:
        self._ocr_client = ocr_client
        self._llm_agent = llm_agent
        self._pdf_reader = pdf_reader or PdfPerceptionReader()
        self._schema_agent = SchemaInductionAgent()
        self._extraction_agent = ExtractionAgent()
        self._compiler = DeterministicCompiler()
        self._audit_agent = AuditAgent()
        self._repair_agent = RepairAgent()
        self._max_repair_rounds = max_repair_rounds

    def parse(
        self,
        input_pdf: Path,
        debug_dir: Path | None = None,
        agent_items_path: Path | None = None,
        use_llm_agent: bool = False,
        runtime_policy: dict[str, Any] | None = None,
    ) -> ParseResult:
        if not input_pdf.exists():
            raise ParseError(f"Input PDF does not exist: {input_pdf}")
        if input_pdf.suffix.lower() != ".pdf":
            raise ParseError("Input file must be a PDF")

        file_hash = sha256_file(input_pdf)
        perception = self._pdf_reader.read(input_pdf)
        layout_mode = _layout_mode(runtime_policy)
        if layout_mode == ENHANCED_LAYOUT_MODE and not use_llm_agent:
            raise LayoutEvidenceError("char_atoms_high_recall layout mode requires use_llm_agent=true.")
        layout_evidence = build_layout_evidence(input_pdf, perception, layout_mode)
        page_images = render_page_images(input_pdf, debug_dir / "page_images" if debug_dir else None)
        ocr_lines: list[OcrLine] = []
        ocr_error: str | None = None
        perception_risks: list[Risk] = []

        try:
            ocr_lines = self._ocr_client.recognize_pdf(input_pdf, perception.pages)
        except OcrError as exc:
            ocr_error = str(exc)
            perception_risks.append(
                Risk(
                    risk_id="risk_ocr_failed",
                    target_type="document",
                    target_id="document",
                    risk_level="high",
                    risk_type="ocr_failed",
                    message=ocr_error,
                )
            )

        source_fusion = build_source_fusion(
            layout_evidence.canonical_pdf_spans,
            ocr_lines,
            enabled=layout_mode == ENHANCED_LAYOUT_MODE,
        )
        spans = source_fusion.canonical_spans
        vdg_nodes = _build_vdg(spans)
        regions = detect_regions(spans)
        table_layers, table_quality_report = build_table_parser_outputs(spans, str(input_pdf))
        agent_table_layers = layout_evidence.candidate_table_layers if layout_mode == ENHANCED_LAYOUT_MODE else table_layers
        candidate_visual_document_graph, vdg_quality_report, vdg_agent_context = build_pre_agent_vdg_artifacts(
            perception.pages,
            spans,
            regions,
            agent_table_layers,
        )
        label_text_scope_reference = load_label_text_scope_reference()
        label_text_scope_agent_context = build_label_text_scope_agent_context(label_text_scope_reference)
        vdg_agent_context = {
            **vdg_agent_context,
            "label_text_scope": label_text_scope_agent_context,
            "layout_quality": layout_evidence.layout_quality_report,
            "reading_order_candidates": layout_evidence.layout_candidates.get("reading_order_candidates", [])[:160],
            "side_marker_candidates": layout_evidence.layout_candidates.get("side_marker_candidates", [])[:80],
            "source_fusion": source_fusion.agent_context(),
        }
        effective_vdg_agent_context = None if vdg_quality_report.get("status") == "fail" else vdg_agent_context
        agent_blocks = (
            build_agent_blocks(spans, layout_evidence.layout_candidates)
            if layout_mode == ENHANCED_LAYOUT_MODE
            else {
                "artifact_version": "agent_blocks_v0.1",
                "status": "disabled",
                "source_span_count": len(spans),
                "covered_source_span_count": 0,
                "source_span_coverage_rate": 0.0,
                "duplicate_primary_source_span_ids": [],
                "blocks": [],
            }
        )
        if layout_mode == ENHANCED_LAYOUT_MODE and agent_blocks.get("status") != "pass":
            raise ParseError("Enhanced Agent block partition did not cover every source span exactly once.")
        agent_block_inputs: list[dict[str, Any]] = []
        agent_block_results: list[dict[str, Any]] = []
        schema_agent_input_span_ids: list[str] = []
        rule_schema = self._schema_agent.generate(spans)
        llm_schema_items: dict[str, Any] | None = None
        if use_llm_agent:
            if self._llm_agent is None:
                raise ParseError("LLM agent requested but no LLM agent is configured.")
            if not hasattr(self._llm_agent, "generate_schema") or not hasattr(self._llm_agent, "generate_extraction_plan"):
                raise ParseError("LLM agent mode requires schema and extraction plan agent methods.")
            if layout_mode == ENHANCED_LAYOUT_MODE and (
                not hasattr(self._llm_agent, "review_compiled_blocks")
                or not hasattr(self._llm_agent, "generate_field_extraction_plan")
            ):
                raise ParseError("char_atoms_high_recall requires independent semantic review and block repair agent methods.")
            if layout_mode == ENHANCED_LAYOUT_MODE:
                schema_spans = schema_induction_spans(spans)
                schema_agent_input_span_ids = [span.span_id for span in schema_spans]
                schema_context = {
                    **(effective_vdg_agent_context or {}),
                    "agent_block_inventory": [
                        {
                            "block_id": block["block_id"],
                            "block_type": block["block_type"],
                            "pages": block["pages"],
                            "source_span_ids": block["source_span_ids"],
                        }
                        for block in agent_blocks["blocks"]
                    ],
                }
                llm_schema_items = _call_llm_agent_method(
                    self._llm_agent,
                    "generate_schema",
                    schema_spans,
                    vdg_context=schema_context,
                )
            else:
                llm_schema_items = _call_llm_agent_method(self._llm_agent, "generate_schema", spans, vdg_context=effective_vdg_agent_context)
            schema = _schema_from_agent_body(llm_schema_items, spans, rule_schema)
        else:
            schema = rule_schema
        schema_audit = build_schema_audit(schema, spans, regions)
        rule_plan = self._extraction_agent.create_plan(schema, spans)
        rule_field_count = len(rule_plan.fields)
        llm_agent_items: dict[str, Any] | None = None
        llm_field_retry_items: dict[str, Any] | None = None
        llm_table_retry_items: dict[str, Any] | None = None
        global_reconciliation_input: dict[str, Any] = {}
        global_reconciliation_output: dict[str, Any] = {}
        global_reconciliation_report = disabled_reconciliation_report("not_requested")
        rule_fallback_items: list[dict[str, Any]] = []
        agent_plan_field_count = 0
        agent_field_retry_count = 0
        agent_table_retry_count = 0
        rejected_agent_items: list[dict[str, Any]] = []
        review_items: list[dict[str, Any]] = []
        label_text_scope_checks: list[dict[str, Any]] = []
        label_text_scope_report: dict[str, Any] = {}
        vdg_boundary_checks: list[dict[str, Any]] = []
        if use_llm_agent:
            if layout_mode == ENHANCED_LAYOUT_MODE:
                extraction_bodies = []
                for block in agent_blocks["blocks"]:
                    local_spans = block_spans(block, spans)
                    local_context = build_block_context(block, effective_vdg_agent_context or {})
                    agent_block_inputs.append(_agent_block_input(block, local_spans, local_context))
                    block_result: dict[str, Any] = {
                        "block_id": block["block_id"],
                        "block_type": block["block_type"],
                    }
                    method_name = "generate_table_extraction_plan" if block["block_type"] == "nutrition_table" else "generate_extraction_plan"
                    if not hasattr(self._llm_agent, method_name):
                        method_name = "generate_extraction_plan"
                    body = _call_llm_agent_method(self._llm_agent, method_name, schema, local_spans, vdg_context=local_context)
                    body = {**body, "_agent_block_id": block["block_id"]}
                    extraction_bodies.append(body)
                    block_result["extraction"] = body
                    block_result["extraction_method"] = method_name
                    agent_block_results.append(block_result)
                llm_agent_items = merge_agent_plan_bodies(extraction_bodies)
                global_reconciliation_input = llm_agent_items
                if hasattr(self._llm_agent, "reconcile_extraction_plan"):
                    global_reconciliation_output = _call_llm_agent_method(
                        self._llm_agent,
                        "reconcile_extraction_plan",
                        schema,
                        spans,
                        global_reconciliation_input,
                        vdg_context=effective_vdg_agent_context,
                    )
                    try:
                        llm_agent_items, global_reconciliation_report = validate_and_finalize_reconciliation(
                            global_reconciliation_input,
                            global_reconciliation_output,
                        )
                    except GlobalReconciliationError as exc:
                        raise ParseError(str(exc)) from exc
                else:
                    global_reconciliation_output = global_reconciliation_input
                    global_reconciliation_report = disabled_reconciliation_report(
                        "agent_method_unavailable",
                        global_reconciliation_input,
                    )
            else:
                llm_agent_items = _call_llm_agent_method(self._llm_agent, "generate_extraction_plan", schema, spans, vdg_context=effective_vdg_agent_context)
            plan, llm_rejected, llm_review = build_agent_extraction_plan(schema, spans, llm_agent_items)
            agent_plan_field_count = len(plan.fields)
            rejected_agent_items.extend({"source": "llm_agent", **item} for item in llm_rejected)
            review_items.extend({"source": "llm_agent", **item} for item in llm_review)
            if layout_mode != ENHANCED_LAYOUT_MODE and not plan.fields and schema.field_definitions:
                field_retry_spans = _field_retry_spans(spans, schema)
                if hasattr(self._llm_agent, "generate_field_extraction_plan"):
                    llm_field_retry_items = _call_llm_agent_method(
                        self._llm_agent,
                        "generate_field_extraction_plan",
                        schema,
                        field_retry_spans,
                        vdg_context=effective_vdg_agent_context,
                    )
                else:
                    llm_field_retry_items = _call_llm_agent_method(
                        self._llm_agent,
                        "generate_extraction_plan",
                        schema,
                        field_retry_spans,
                        vdg_context=effective_vdg_agent_context,
                    )
                retry_plan, retry_rejected, retry_review = build_agent_extraction_plan(
                    schema,
                    spans,
                    llm_field_retry_items,
                    plan_id="plan_agent_field_retry_001",
                )
                rejected_agent_items.extend({"source": "llm_field_retry", **item} for item in retry_rejected)
                review_items.extend({"source": "llm_field_retry", **item} for item in retry_review)
                if retry_plan.fields:
                    plan = _merge_field_retry_plan(plan, retry_plan)
                    agent_field_retry_count = len(retry_plan.fields)
                    agent_plan_field_count = len(plan.fields)
            if not _has_structured_agent_table(plan.tables) and schema.table_definitions:
                table_retry_decision = _table_retry_decision(
                    agent_table_layers,
                    allow_large_candidate_set=layout_mode == ENHANCED_LAYOUT_MODE,
                )
                if table_retry_decision["should_retry"]:
                    if hasattr(self._llm_agent, "generate_table_extraction_plan"):
                        table_retry_spans = _table_retry_spans(spans, agent_table_layers)
                        table_retry_context = _table_retry_context(effective_vdg_agent_context, agent_table_layers)
                        llm_table_retry_items = _call_llm_agent_method(
                            self._llm_agent,
                            "generate_table_extraction_plan",
                            schema,
                            table_retry_spans,
                            vdg_context=table_retry_context,
                        )
                    else:
                        llm_table_retry_items = _call_llm_agent_method(
                            self._llm_agent,
                            "generate_extraction_plan",
                            schema,
                            spans,
                            vdg_context=effective_vdg_agent_context,
                        )
                    table_retry_plan, table_retry_rejected, table_retry_review = build_agent_extraction_plan(
                        schema,
                        spans,
                        llm_table_retry_items,
                        plan_id="plan_agent_table_retry_001",
                    )
                    rejected_agent_items.extend({"source": "llm_table_retry", **item} for item in table_retry_rejected)
                    review_items.extend({"source": "llm_table_retry", **item} for item in table_retry_review)
                    if table_retry_plan.tables:
                        plan = _merge_table_retry_plan(plan, table_retry_plan)
                        agent_table_retry_count = len(table_retry_plan.tables)
                else:
                    review_items.append({"source": "llm_table_retry", **table_retry_decision})
            plan, path_rejected, path_review = merge_agent_candidates(plan, schema, spans, agent_items_path)
            rejected_agent_items.extend({"source": "agent_items_path", **item} for item in path_rejected)
            review_items.extend({"source": "agent_items_path", **item} for item in path_review)
            if layout_mode == ENHANCED_LAYOUT_MODE:
                review_items.extend(
                    {"source": "rule_validation_candidate", **item}
                    for item in build_rule_candidate_review_items(plan, rule_plan)
                )
            else:
                plan, rule_fallback_items = apply_rule_fallback_fields(plan, rule_plan)
        else:
            plan = rule_plan
            plan, rejected_agent_items, review_items = merge_agent_candidates(plan, schema, spans, agent_items_path)
        layout_acceptance_report, layout_review, layout_validation_checks = _layout_candidate_acceptance(
            layout_evidence.layout_candidates,
            spans,
            [llm_agent_items, llm_field_retry_items, llm_table_retry_items],
            layout_mode,
        )
        review_items.extend({"source": "layout_candidate_gate", **item} for item in layout_review)
        plan, scope_rejected, scope_review, label_text_scope_checks, label_text_scope_report = apply_label_text_scope_gate(
            plan,
            spans,
            label_text_scope_reference,
            agent_bodies=[llm_agent_items, llm_field_retry_items, llm_table_retry_items],
        )
        rejected_agent_items.extend({"source": "label_text_scope_gate", **item} for item in scope_rejected)
        review_items.extend({"source": "label_text_scope_gate", **item} for item in scope_review)
        plan, vdg_rejected, vdg_review, vdg_boundary_checks = apply_vdg_boundary_gate(plan, candidate_visual_document_graph, spans)
        rejected_agent_items.extend({"source": "vdg_boundary_gate", **item} for item in vdg_rejected)
        review_items.extend({"source": "vdg_boundary_gate", **item} for item in vdg_review)
        pre_repair_plan = plan
        compiled_fields, evidence = self._compiler.compile(plan, spans)
        semantic_review_input: dict[str, Any] = {"artifact_version": "semantic_review_input_v0.1", "blocks": []}
        semantic_repair_rounds: list[dict[str, Any]] = []
        semantic_review_rejected: list[dict[str, Any]] = []
        semantic_review_findings: list[dict[str, Any]] = []
        semantic_review_enabled = bool(
            layout_mode == ENHANCED_LAYOUT_MODE
            and use_llm_agent
            and self._llm_agent is not None
            and hasattr(self._llm_agent, "review_compiled_blocks")
        )
        if semantic_review_enabled:
            semantic_review_input = build_semantic_review_input(agent_blocks, plan, compiled_fields, evidence, spans)
            agent_findings, rejected_findings, review_call_count = call_semantic_review_agent(self._llm_agent, semantic_review_input)
            semantic_review_rejected.extend(rejected_findings)
            semantic_review_findings = merge_semantic_findings(
                deterministic_semantic_findings(semantic_review_input),
                agent_findings,
                _schema_consumption_findings(schema, plan, agent_blocks),
            )
            round_record: dict[str, Any] = {
                "round": 1,
                "finding_count": len(semantic_review_findings),
                "findings": semantic_review_findings,
                "review_call_count": review_call_count,
                "repair_attempts": [],
            }
            repairable_findings = [
                finding
                for finding in semantic_review_findings
                if finding.get("repair_required")
                and not str(finding.get("issue_type") or "").startswith("nutrition_")
            ]
            can_repair_globally = bool(
                repairable_findings
                and global_reconciliation_input
                and hasattr(self._llm_agent, "reconcile_extraction_plan")
            )
            if can_repair_globally:
                repair_spans = _semantic_repair_spans(repairable_findings, agent_blocks, spans)
                repair_context = {
                    **(effective_vdg_agent_context or {}),
                    "semantic_review_findings": repairable_findings,
                    "previous_reconciled_plan": llm_agent_items,
                }
                extraction_repair_output = _call_llm_agent_method(
                    self._llm_agent,
                    "generate_field_extraction_plan",
                    schema,
                    repair_spans,
                    vdg_context=repair_context,
                )
                repair_proposals = merge_block_retry_body(
                    llm_agent_items or global_reconciliation_input,
                    merge_agent_plan_bodies(
                        [{**extraction_repair_output, "_agent_block_id": "global_semantic_repair"}]
                    ),
                )
                repair_output = repair_proposals
                try:
                    repaired_body, repair_report = validate_and_finalize_reconciliation(
                        repair_proposals,
                        repair_output,
                    )
                except GlobalReconciliationError as exc:
                    raise ParseError(str(exc)) from exc
                repair_plan, repair_rejected, repair_review = build_agent_extraction_plan(
                    schema,
                    spans,
                    repaired_body,
                    plan_id="plan_global_semantic_repair_001",
                )
                rejected_agent_items.extend({"source": "global_semantic_repair", **item} for item in repair_rejected)
                review_items.extend({"source": "global_semantic_repair", **item} for item in repair_review)
                repair_plan, repair_scope_rejected, repair_scope_review, _, _ = apply_label_text_scope_gate(
                    repair_plan,
                    spans,
                    label_text_scope_reference,
                    agent_bodies=[repaired_body],
                )
                rejected_agent_items.extend({"source": "global_semantic_repair_scope", **item} for item in repair_scope_rejected)
                review_items.extend({"source": "global_semantic_repair_scope", **item} for item in repair_scope_review)
                repair_plan, repair_vdg_rejected, repair_vdg_review, _ = apply_vdg_boundary_gate(
                    repair_plan,
                    candidate_visual_document_graph,
                    spans,
                )
                rejected_agent_items.extend({"source": "global_semantic_repair_vdg", **item} for item in repair_vdg_rejected)
                review_items.extend({"source": "global_semantic_repair_vdg", **item} for item in repair_vdg_review)
                if repair_plan != plan:
                    candidate_fields, candidate_evidence = self._compiler.compile(repair_plan, spans)
                    candidate_review_input = build_semantic_review_input(
                        agent_blocks,
                        repair_plan,
                        candidate_fields,
                        candidate_evidence,
                        spans,
                    )
                    candidate_agent_findings, candidate_rejected, candidate_review_calls = call_semantic_review_agent(
                        self._llm_agent,
                        candidate_review_input,
                    )
                    semantic_review_rejected.extend(candidate_rejected)
                    candidate_findings = merge_semantic_findings(
                        deterministic_semantic_findings(candidate_review_input),
                        candidate_agent_findings,
                        _schema_consumption_findings(schema, repair_plan, agent_blocks),
                    )
                    improved = _semantic_finding_score(candidate_findings) < _semantic_finding_score(semantic_review_findings)
                    round_record["repair_attempts"].append(
                        {
                            "scope": "document",
                            "candidate_field_count": len(repair_plan.fields),
                            "candidate_table_count": len(repair_plan.tables),
                            "extraction_repair_field_count": len(extraction_repair_output.get("fields", [])),
                            "candidate_finding_count": len(candidate_findings),
                            "review_call_count": candidate_review_calls,
                            "status": "applied" if improved else "rejected_no_improvement",
                        }
                    )
                    if improved:
                        plan = repair_plan
                        compiled_fields = candidate_fields
                        evidence = candidate_evidence
                        semantic_review_input = candidate_review_input
                        semantic_review_findings = candidate_findings
                        llm_agent_items = repaired_body
                        agent_plan_field_count = len(plan.fields)
                        global_reconciliation_output = repair_output
                        global_reconciliation_report = {
                            **global_reconciliation_report,
                            "repair_applied": True,
                            "repair_candidate_field_count": repair_report.get("input_field_count", 0),
                            "repair_output_field_count": repair_report.get("output_field_count", 0),
                        }
                        round_record["status"] = "repaired_and_recompiled"
                    else:
                        round_record["status"] = "rejected_no_improvement"
                else:
                    round_record["status"] = "no_plan_change"
            else:
                round_record["status"] = "passed" if not semantic_review_findings else "review_required_no_repair"
            semantic_repair_rounds.append(round_record)
        semantic_review_report = build_semantic_review_report(
            semantic_review_input,
            semantic_review_findings,
            semantic_review_rejected,
            semantic_repair_rounds,
            enabled=semantic_review_enabled,
        )
        review_items.extend({"source": "semantic_review", **item} for item in semantic_review_findings)
        repair_attempts: list[dict[str, Any]] = []
        repair_trace_rounds: list[dict[str, Any]] = []
        audit_findings: list[dict[str, Any]] = []
        for round_index in range(self._max_repair_rounds + 1):
            audit_input = _audit_input(compiled_fields, evidence)
            audit_findings = self._audit_agent.audit(audit_input, schema)
            round_record: dict[str, Any] = {
                "round": round_index + 1,
                "audit_finding_count": len(audit_findings),
                "audit_findings": audit_findings,
                "attempts": [],
                "compiled_after_repair": False,
            }
            if not audit_findings:
                round_record["status"] = "passed_no_repair_needed" if round_index == 0 else "passed_after_repair"
                repair_trace_rounds.append(round_record)
                break
            if round_index == self._max_repair_rounds:
                round_record["status"] = "max_rounds_reached"
                repair_trace_rounds.append(round_record)
                break
            repaired_plan, attempts = self._repair_agent.repair(plan, audit_findings, spans)
            round_attempts = [{"round": round_index + 1, **attempt} for attempt in attempts]
            repair_attempts.extend(round_attempts)
            round_record["attempts"] = round_attempts
            if repaired_plan == plan:
                round_record["status"] = "no_plan_change"
                repair_trace_rounds.append(round_record)
                break
            plan = repaired_plan
            compiled_fields, evidence = self._compiler.compile(plan, spans)
            round_record["status"] = "repaired_and_recompiled"
            round_record["compiled_after_repair"] = True
            repair_trace_rounds.append(round_record)

        regions, region_evidence = attach_region_evidence(regions, spans, len(evidence))
        evidence.extend(region_evidence)
        agent_tables, agent_table_evidence = _extract_tables_from_agent_plan(plan.tables, spans, content_item_names(spans), len(evidence))
        agent_tables = _drop_incomplete_agent_tables_replaced_by_layout_candidates(
            agent_tables,
            layout_evidence.layout_candidates,
            layout_acceptance_report,
        )
        accepted_layout_tables, accepted_layout_evidence = _extract_tables_from_accepted_layout_candidates(
            layout_evidence.layout_candidates,
            layout_acceptance_report,
            spans,
            content_item_names(spans),
            agent_tables,
            len(evidence) + len(agent_table_evidence),
        )
        if agent_tables or accepted_layout_tables:
            tables = [*agent_tables, *accepted_layout_tables]
            table_evidence = [*agent_table_evidence, *accepted_layout_evidence]
            table_quality_report = _table_quality_with_agent_acceptance(table_quality_report, tables)
        else:
            tables, table_evidence = extract_nutrition_tables_from_layers(table_layers, content_item_names(spans), len(evidence))
        evidence.extend(table_evidence)
        table_feed_items = _apply_table_feed_structure(compiled_fields, evidence, tables, table_quality_report, runtime_policy)
        entities = build_entities(compiled_fields, tables, plan.entities)
        regions = assign_region_memberships(regions, compiled_fields, evidence, tables, entities, spans)
        requirements = build_requirements(compiled_fields)
        revision_blocks = build_revision_blocks(regions, compiled_fields, evidence, spans)
        missing_item_report = build_missing_item_report(compiled_fields, tables)
        source_layers = build_source_layers(perception, ocr_lines, spans, table_layers, ocr_error=ocr_error)
        source_consistency_report = build_source_consistency_report(perception.text_spans, ocr_lines)
        visual_document_graph = build_visual_document_graph(
            perception.pages,
            spans,
            regions,
            agent_table_layers,
            plan,
            tables,
            requirements,
            evidence,
        )
        visual_document_graph, vdg_consumption_report = build_vdg_consumption_report(
            visual_document_graph,
            plan,
            tables,
            requirements,
            evidence,
        )

        validation = (
            _validation_checks(compiled_fields, evidence, schema)
            + _schema_audit_validation_checks(schema_audit)
            + label_text_scope_checks
            + vdg_quality_validation_checks(vdg_quality_report)
            + _layout_quality_validation_checks(layout_evidence.layout_quality_report)
            + _source_fusion_validation_checks(source_fusion.report, layout_mode)
            + layout_validation_checks
            + vdg_boundary_checks
            + semantic_review_validation_checks(semantic_review_report)
            + vdg_node_coverage_validation_checks(vdg_consumption_report)
            + _source_consistency_validation_checks(source_consistency_report)
            + _internal_consistency_validation_checks(compiled_fields, revision_blocks)
            + _table_structure_validation_checks(tables, table_quality_report)
            + missing_item_validation_checks(missing_item_report)
        )
        risks = (
            perception_risks
            + _risks_from_fields(compiled_fields, evidence)
            + risks_from_missing_item_report(missing_item_report)
            + _risks_from_schema_audit(schema_audit)
            + _risks_from_vdg_quality(vdg_quality_report)
            + _risks_from_vdg_consumption(vdg_consumption_report)
            + _risks_from_regions(regions)
            + _risks_from_tables(tables)
            + _risks_from_table_quality_report(table_quality_report)
            + _risks_from_source_layers(source_layers)
            + _risks_from_source_consistency(source_consistency_report)
            + _risks_from_page_images(page_images)
            + _risks_from_validation(validation)
            + _risks_from_revision_blocks(revision_blocks)
            + _risks_from_audit(audit_findings)
            + _risks_from_semantic_review(semantic_review_report)
        )
        structure_audit = build_structure_audit(spans, compiled_fields, evidence, tables, entities, regions)
        coverage_map = build_coverage_map(
            spans,
            compiled_fields,
            evidence,
            tables,
            requirements,
            regions,
            structure_audit["anchor_inventory"],
        )
        structure_audit["duplicate_coverage_issues"] = coverage_map["duplicate_coverage_issues"]
        structure_audit["duplicate_coverage_issue_count"] = coverage_map["duplicate_coverage_issue_count"]
        repair_plan = build_repair_plan(
            risks,
            audit_findings,
            structure_audit,
            self._max_repair_rounds,
            review_items,
            rejected_agent_items,
        )
        repair_attempts_artifact = build_repair_attempts_artifact(repair_attempts, repair_plan)
        repair_trace = build_repair_trace_artifact(repair_trace_rounds, self._max_repair_rounds)
        repair_plan_patches = build_repair_plan_patches(pre_repair_plan, plan, repair_attempts_artifact)
        repair_agent_candidates = build_repair_agent_candidates(repair_plan)
        repaired_source_layers = build_repaired_source_layers(source_layers, repair_attempts_artifact)
        review_tasks = _review_tasks_from_risks(risks)
        quality = _quality(compiled_fields, risks)
        coverage = _coverage_from_vdg_consumption(vdg_consumption_report)
        standard_items = build_standard_items(compiled_fields, evidence, spans, input_pdf, revision_blocks)
        comparison_index = build_comparison_index(standard_items)
        auto_ingest_candidates = build_auto_ingest_candidates(standard_items, validation, risks, quality)
        field_groups = build_field_groups(entities, compiled_fields, standard_items)
        tables_artifact = build_tables_artifact(tables)
        lists_artifact = build_lists_artifact(field_groups)
        quality_report = build_quality_report(
            quality,
            risks,
            validation,
            schema_audit,
            structure_audit,
            source_layers,
            table_quality_report,
            repair_plan,
            repair_attempts_artifact,
            repair_trace,
            repair_agent_candidates,
            vdg_quality_report,
            label_text_scope_report,
        )
        structured_document = build_structured_document(
            {
                "file_name": input_pdf.name,
                "file_hash": file_hash,
                "page_count": len(perception.pages),
                "language": ["zh-CN"],
            },
            source_layers,
            schema,
            regions,
            standard_items,
            field_groups,
            tables_artifact,
            lists_artifact,
        )
        taxonomy_proposals = build_taxonomy_proposals(compiled_fields, evidence)
        agent_execution_report = _agent_execution_report(
            schema_agent_name=type(self._schema_agent).__name__,
            extraction_agent_name=type(self._extraction_agent).__name__,
            audit_agent_name=type(self._audit_agent).__name__,
            repair_agent_name=type(self._repair_agent).__name__,
            llm_agent_name=type(self._llm_agent).__name__ if self._llm_agent else None,
            schema=schema,
            plan=plan,
            compiled_fields=compiled_fields,
            evidence=evidence,
            audit_findings=audit_findings,
            schema_audit=schema_audit,
            repair_trace=repair_trace,
            agent_items_path=agent_items_path,
            use_llm_agent=use_llm_agent,
            llm_agent_items=llm_agent_items,
            llm_field_retry_items=llm_field_retry_items,
            llm_table_retry_items=llm_table_retry_items,
            llm_schema_items=llm_schema_items,
            rule_field_count=rule_field_count,
            agent_plan_field_count=agent_plan_field_count,
            agent_field_retry_count=agent_field_retry_count,
            agent_table_retry_count=agent_table_retry_count,
            rule_fallback_items=rule_fallback_items,
            rejected_agent_items=rejected_agent_items,
            review_items=review_items,
            layout_mode=layout_mode,
            global_reconciliation_report=global_reconciliation_report,
        )
        audit_input = build_audit_input_artifact(
            page_images=page_images,
            visual_document_graph=visual_document_graph,
            schema=schema,
            plan=plan,
            compiled_fields=compiled_fields,
            evidence=evidence,
            coverage_map=coverage_map,
            audit_findings=audit_findings,
        )

        result = ParseResult(
            job={
                "job_id": f"job_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}",
                "job_type": "standard_pdf_to_structured_json",
                "status": "completed_with_warnings" if risks else "completed",
            },
            document={
                "file_name": input_pdf.name,
                "file_hash": file_hash,
                "page_count": len(perception.pages),
                "page_sizes": [to_jsonable(page) for page in perception.pages],
                "detected_document_types": _detected_document_types(regions, tables),
                "language": ["zh-CN"],
                "parse_status": _document_parse_status(risks, page_images),
                "pdf_text_layer_available": perception.text_layer_available,
                "page_image_status": page_images["status"],
                "warnings": perception.warnings,
            },
            generated_schema=schema,
            extracted_data={
                "sections": schema.sections,
                "regions": regions,
                "entities": entities,
                "fields": {field_id: to_jsonable(field) for field_id, field in compiled_fields.items()},
                "missing_fields": missing_item_report["missing_fields"],
                "missing_tables": missing_item_report["missing_tables"],
                "tables": tables,
                "requirements": requirements,
                "revision_blocks": revision_blocks,
            },
            evidence=evidence,
            cross_validation={
                "checks": validation,
                "source_consistency": source_consistency_report,
            },
            coverage=coverage,
            validation=validation,
            quality=quality,
            risks=risks,
            review_tasks=review_tasks,
            metadata={
                "parser_version": "mvp_v0.1",
                "pipeline_version": "mvp_pipeline_v0.2",
                "schema_mode": "agent_generated" if use_llm_agent else "auto_generated",
                "no_guessing": True,
                "json_export": _json_export_manifest(),
                "ocr_provider": "glm_ocr",
                "runtime_policy": runtime_policy or {},
                "pdf_character_atoms": to_jsonable(layout_evidence.character_atoms),
                "layout_candidates": layout_evidence.layout_candidates,
                "layout_quality_report": layout_evidence.layout_quality_report,
                "source_fusion_report": source_fusion.report,
                "source_alignments": to_jsonable(source_fusion.alignments),
                "layout_candidate_acceptance_report": layout_acceptance_report,
                "page_images": page_images,
                "candidate_visual_document_graph": candidate_visual_document_graph,
                "visual_document_graph": visual_document_graph,
                "vdg_quality_report": vdg_quality_report,
                "vdg_agent_context": vdg_agent_context,
                "agent_blocks": agent_blocks,
                "agent_block_inputs": agent_block_inputs,
                "agent_block_results": agent_block_results,
                "schema_agent_input_span_ids": schema_agent_input_span_ids,
                "global_reconciliation_input": global_reconciliation_input,
                "global_reconciliation_output": global_reconciliation_output,
                "global_reconciliation_report": global_reconciliation_report,
                "semantic_review_input": semantic_review_input,
                "semantic_review_report": semantic_review_report,
                "semantic_repair_rounds": semantic_repair_rounds,
                "vdg_consumption_report": vdg_consumption_report,
                "label_text_scope_reference": label_text_scope_reference,
                "label_text_scope_agent_context": label_text_scope_agent_context,
                "label_text_scope_report": label_text_scope_report,
                "missing_item_report": missing_item_report,
                "repair_loop": {
                    "max_repair_rounds": self._max_repair_rounds,
                    "audit_finding_count": len(audit_findings),
                    "trace": repair_trace,
                    "attempts": repair_attempts_artifact,
                    "repair_plan": repair_plan,
                    "repair_plan_patches": repair_plan_patches,
                    "repair_agent_candidates": repair_agent_candidates,
                    "repaired_source_layers": repaired_source_layers,
                    "policy": "agent_repair_execute_plan_recompile_validate",
                },
                "audit_input": audit_input,
                "schema_audit": schema_audit,
                "structure_audit": structure_audit,
                "source_layers": source_layers,
                "source_consistency": source_consistency_report,
                "source_anchor_inventory": structure_audit["anchor_inventory"],
                "coverage_map": coverage_map,
                "standard_artifacts": {
                    "standard_items": standard_items,
                    "comparison_index": comparison_index,
                    "quality_report": quality_report,
                    "structured_document": structured_document,
                    "taxonomy_proposals": taxonomy_proposals,
                    "field_groups": field_groups,
                    "tables": tables_artifact,
                    "lists": lists_artifact,
                    "auto_ingest_candidates": auto_ingest_candidates,
                },
                "table_parser": {
                    "table_layers": table_layers,
                    "candidate_table_layers": agent_table_layers,
                    "table_quality_report": table_quality_report,
                    "table_feed_items": table_feed_items,
                },
                "agent_execution_report": agent_execution_report,
                "agent_harness": {
                    "agent_items_path": str(agent_items_path) if agent_items_path else None,
                    "accepted_agent_item_count": agent_plan_field_count if use_llm_agent else max(len(plan.fields) - rule_field_count, 0),
                    "agent_plan_field_count": agent_plan_field_count,
                    "agent_field_retry_count": agent_field_retry_count,
                    "agent_table_retry_count": agent_table_retry_count,
                    "field_retry_used": llm_field_retry_items is not None,
                    "table_retry_used": llm_table_retry_items is not None,
                    "rule_fallback_field_count": len(rule_fallback_items),
                    "rule_fallback_items": rule_fallback_items,
                    "llm_agent_enabled": use_llm_agent,
                    "vdg_agent_context_used": effective_vdg_agent_context is not None,
                    "vdg_quality_status": vdg_quality_report.get("status"),
                    "label_text_scope_status": label_text_scope_report.get("status"),
                    "scope_gate_rejected_count": label_text_scope_report.get("scope_gate_rejected_count", 0),
                    "llm_agent_candidate_count": _agent_plan_item_count(llm_agent_items) or agent_plan_field_count,
                    "llm_field_retry_candidate_count": _agent_plan_item_count(llm_field_retry_items),
                    "llm_table_retry_table_count": _agent_table_plan_count(llm_table_retry_items),
                    "llm_field_retry_items": llm_field_retry_items or {"fields": []},
                    "llm_table_retry_items": llm_table_retry_items or {"tables": []},
                    "llm_agent_items": llm_agent_items or {"fields": []},
                    "llm_schema_items": llm_schema_items or {},
                    "global_reconciliation_status": global_reconciliation_report.get("status"),
                    "global_reconciliation_removed_field_count": global_reconciliation_report.get("removed_field_count", 0),
                    "rejected_agent_items": rejected_agent_items,
                    "review_items": review_items,
                },
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
            },
        )
        output_contract_validation_report = build_output_contract_validation_report(result)
        result.cross_validation["output_contract"] = output_contract_validation_report
        result.metadata["output_contract_validation_report"] = output_contract_validation_report
        result.metadata["mvp_acceptance_metrics"] = build_mvp_acceptance_metrics(
            schema=schema,
            plan=plan,
            compiled_fields=compiled_fields,
            evidence=evidence,
            validation=validation,
            risks=risks,
            review_tasks=review_tasks,
            coverage=coverage,
            coverage_map=coverage_map,
            schema_audit=schema_audit,
            structure_audit=structure_audit,
            source_layers=source_layers,
            table_quality_report=table_quality_report,
            repair_trace=repair_trace,
            output_contract_validation_report=output_contract_validation_report,
            missing_item_report=missing_item_report,
        )

        if debug_dir:
            self._write_debug(debug_dir, perception, ocr_lines, spans, vdg_nodes, visual_document_graph, plan, audit_findings, result)
        return result

    def _write_debug(
        self,
        debug_dir: Path,
        perception: Any,
        ocr_lines: list[OcrLine],
        spans: list[TextSpan],
        vdg_nodes: list[VdgNode],
        visual_document_graph: dict[str, Any],
        plan: ExtractionPlan,
        audit_findings: list[dict[str, Any]],
        result: ParseResult,
    ) -> None:
        write_json(debug_dir / "perception.json", to_jsonable(perception))
        write_json(debug_dir / "runtime_policy.json", result.metadata["runtime_policy"])
        write_json(debug_dir / "json_export.json", result.metadata["json_export"])
        write_json(debug_dir / "page_images.json", result.metadata["page_images"])
        write_json(debug_dir / "ocr_lines.json", to_jsonable(ocr_lines))
        write_json(debug_dir / "spans.json", to_jsonable(spans))
        write_json(debug_dir / "pdf_character_atoms.json", result.metadata["pdf_character_atoms"])
        write_json(debug_dir / "layout_candidates.json", result.metadata["layout_candidates"])
        write_json(debug_dir / "layout_quality_report.json", result.metadata["layout_quality_report"])
        write_json(debug_dir / "source_fusion_report.json", result.metadata["source_fusion_report"])
        write_json(debug_dir / "source_alignments.json", result.metadata["source_alignments"])
        write_json(debug_dir / "candidate_visual_document_graph.json", to_jsonable(result.metadata["candidate_visual_document_graph"]))
        write_json(debug_dir / "vdg.json", to_jsonable(visual_document_graph))
        write_json(debug_dir / "visual_document_graph.json", to_jsonable(visual_document_graph))
        write_json(debug_dir / "vdg_nodes.json", to_jsonable(vdg_nodes))
        write_json(debug_dir / "vdg_quality_report.json", to_jsonable(result.metadata["vdg_quality_report"]))
        write_json(debug_dir / "vdg_agent_context.json", to_jsonable(result.metadata["vdg_agent_context"]))
        write_json(debug_dir / "agent_blocks.json", to_jsonable(result.metadata["agent_blocks"]))
        write_json(debug_dir / "agent_block_inputs.json", to_jsonable(result.metadata["agent_block_inputs"]))
        write_json(debug_dir / "agent_block_results.json", to_jsonable(result.metadata["agent_block_results"]))
        write_json(debug_dir / "global_reconciliation_input.json", to_jsonable(result.metadata["global_reconciliation_input"]))
        write_json(debug_dir / "global_reconciliation_output.json", to_jsonable(result.metadata["global_reconciliation_output"]))
        write_json(debug_dir / "global_reconciliation_report.json", to_jsonable(result.metadata["global_reconciliation_report"]))
        write_json(debug_dir / "semantic_review_input.json", to_jsonable(result.metadata["semantic_review_input"]))
        write_json(debug_dir / "semantic_review_report.json", to_jsonable(result.metadata["semantic_review_report"]))
        write_json(debug_dir / "semantic_repair_rounds.json", to_jsonable(result.metadata["semantic_repair_rounds"]))
        write_json(debug_dir / "vdg_consumption_report.json", to_jsonable(result.metadata["vdg_consumption_report"]))
        write_json(debug_dir / "label_text_scope_reference.json", to_jsonable(result.metadata["label_text_scope_reference"]))
        write_json(debug_dir / "label_text_scope_agent_context.json", to_jsonable(result.metadata["label_text_scope_agent_context"]))
        write_json(debug_dir / "label_text_scope_report.json", to_jsonable(result.metadata["label_text_scope_report"]))
        write_json(debug_dir / "extraction_plan.json", to_jsonable(plan))
        write_json(debug_dir / "schema_audit.json", result.metadata["schema_audit"])
        write_json(debug_dir / "audit_input.json", result.metadata["audit_input"])
        write_json(debug_dir / "audit_findings.json", audit_findings)
        write_json(debug_dir / "standard_items.json", to_jsonable(result.metadata["standard_artifacts"]["standard_items"]))
        write_json(debug_dir / "comparison_index.json", to_jsonable(result.metadata["standard_artifacts"]["comparison_index"]))
        write_json(debug_dir / "quality_report.json", to_jsonable(result.metadata["standard_artifacts"]["quality_report"]))
        write_json(debug_dir / "structured_document.json", to_jsonable(result.metadata["standard_artifacts"]["structured_document"]))
        write_json(debug_dir / "taxonomy_proposals.json", to_jsonable(result.metadata["standard_artifacts"]["taxonomy_proposals"]))
        write_json(debug_dir / "field_groups.json", to_jsonable(result.metadata["standard_artifacts"]["field_groups"]))
        write_json(debug_dir / "tables.json", to_jsonable(result.metadata["standard_artifacts"]["tables"]))
        write_json(debug_dir / "lists.json", to_jsonable(result.metadata["standard_artifacts"]["lists"]))
        write_json(debug_dir / "auto_ingest_candidates.json", to_jsonable(result.metadata["standard_artifacts"]["auto_ingest_candidates"]))
        write_json(debug_dir / "table_layers.json", to_jsonable(result.metadata["table_parser"]["table_layers"]))
        write_json(debug_dir / "table_quality_report.json", to_jsonable(result.metadata["table_parser"]["table_quality_report"]))
        write_json(debug_dir / "table_feed_items.candidates.json", to_jsonable(result.metadata["table_parser"]["table_feed_items"]))
        write_json(debug_dir / "source_layers.json", result.metadata["source_layers"])
        write_json(debug_dir / "source_consistency_report.json", result.metadata["source_consistency"])
        write_json(debug_dir / "source_anchor_inventory.json", result.metadata["source_anchor_inventory"])
        write_json(debug_dir / "coverage_map.json", result.metadata["coverage_map"])
        write_json(debug_dir / "structure_audit.json", result.metadata["structure_audit"])
        write_json(debug_dir / "missing_item_report.json", result.metadata["missing_item_report"])
        write_json(debug_dir / "missing_fields.json", result.metadata["missing_item_report"]["missing_fields"])
        write_json(debug_dir / "missing_tables.json", result.metadata["missing_item_report"]["missing_tables"])
        write_json(debug_dir / "revision_blocks.json", to_jsonable(result.extracted_data["revision_blocks"]))
        write_json(debug_dir / "output_contract_validation_report.json", result.metadata["output_contract_validation_report"])
        write_json(debug_dir / "mvp_acceptance_metrics.json", result.metadata["mvp_acceptance_metrics"])
        write_json(debug_dir / "extracted_data.json", to_jsonable(result.extracted_data))
        write_json(debug_dir / "evidence.json", to_jsonable(result.evidence))
        write_json(debug_dir / "validation.json", to_jsonable(result.validation))
        write_json(debug_dir / "risks.json", to_jsonable(result.risks))
        write_json(debug_dir / "review_tasks.json", to_jsonable(result.review_tasks))
        write_json(debug_dir / "repair_plan.json", result.metadata["repair_loop"]["repair_plan"])
        write_json(debug_dir / "repair_trace.json", result.metadata["repair_loop"]["trace"])
        write_json(debug_dir / "repair_attempts.json", result.metadata["repair_loop"]["attempts"])
        write_json(debug_dir / "repair_plan_patches.json", result.metadata["repair_loop"]["repair_plan_patches"])
        write_json(debug_dir / "repair_agent_candidates.json", result.metadata["repair_loop"]["repair_agent_candidates"])
        write_json(debug_dir / "repaired_source_layers.json", result.metadata["repair_loop"]["repaired_source_layers"])
        write_json(debug_dir / "llm_agent_items.json", result.metadata["agent_harness"]["llm_agent_items"])
        write_json(debug_dir / "rejected_agent_items.json", result.metadata["agent_harness"]["rejected_agent_items"])
        write_json(debug_dir / "review_items.json", result.metadata["agent_harness"]["review_items"])
        write_json(debug_dir / "agent_execution_report.json", result.metadata["agent_execution_report"])
        write_json(debug_dir / "agent_harness_report.json", result.metadata["agent_harness"])
        write_json(debug_dir / "result.json", to_jsonable(result))
        write_result_preview_html(result, debug_dir / "result_preview.html", artifact_root=debug_dir)
        write_schema_artifacts(debug_dir)
        _write_layered_debug_artifacts(debug_dir, result, perception, ocr_lines, spans, vdg_nodes, visual_document_graph, plan, audit_findings)
        write_artifact_index(debug_dir)


def write_artifact_index(debug_dir: Path) -> None:
    index_path = debug_dir / "artifacts" / "index.json"
    artifacts = []
    indexed_suffixes = {".json", ".md", ".png", ".html"}
    for path in sorted(item for item in debug_dir.rglob("*") if item.is_file() and item.suffix in indexed_suffixes):
        if path == index_path:
            continue
        artifacts.append(
            {
                "path": path.relative_to(debug_dir).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(
        index_path,
        {
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )


def _schema_from_agent_body(body: dict[str, Any], spans: list[TextSpan], fallback_schema: GeneratedSchema) -> GeneratedSchema:
    if not isinstance(body, dict):
        raise ParseError("LLM schema agent must return a JSON object.")
    valid_span_ids = {span.span_id for span in spans}
    field_items = body.get("field_definitions", body.get("fields", body.get("items", [])))
    if not isinstance(field_items, list):
        raise ParseError("LLM schema agent output must contain field_definitions.")

    fields: list[FieldDefinition] = []
    seen_keys: set[str] = set()
    for item in field_items:
        if not isinstance(item, dict):
            continue
        semantic_key = _schema_semantic_key(item)
        display_name = str(item.get("display_name") or item.get("label") or item.get("name") or item.get("field_name") or semantic_key)
        field_type = _schema_field_type(item)
        if not semantic_key or semantic_key in seen_keys:
            continue
        source_span_ids = [
            span_id
            for span_id in _source_span_ids_from_schema_item(item)
            if span_id in valid_span_ids
        ]
        if not source_span_ids:
            source_span_ids = _schema_anchor_span_ids(item, semantic_key, display_name, spans)
        if not source_span_ids:
            continue
        fields.append(
            FieldDefinition(
                field_def_id=stable_id("fdef", len(fields) + 1),
                semantic_key=semantic_key,
                display_name=display_name,
                field_type=field_type,
                criticality=_schema_criticality(item, semantic_key),
                repeatable=bool(item.get("repeatable", False)),
                semantic_key_type=str(item.get("semantic_key_type") or "agent_generated"),
                source_span_ids=source_span_ids,
            )
        )
        seen_keys.add(semantic_key)

    for fallback_field in fallback_schema.field_definitions:
        if fallback_field.semantic_key in seen_keys:
            continue
        fields.append(
            FieldDefinition(
                field_def_id=stable_id("fdef", len(fields) + 1),
                semantic_key=fallback_field.semantic_key,
                display_name=fallback_field.display_name,
                field_type=fallback_field.field_type,
                criticality=fallback_field.criticality,
                repeatable=fallback_field.repeatable,
                semantic_key_type="rule_validation_fallback",
                source_span_ids=fallback_field.source_span_ids,
            )
        )
        seen_keys.add(fallback_field.semantic_key)

    return GeneratedSchema(
        schema_id=str(body.get("schema_id") or "schema_agent_001"),
        auto_generated=True,
        schema_version=str(body.get("schema_version") or "agent_led_v1"),
        sections=_merge_schema_objects(_schema_objects_with_valid_source_refs(_list_value(body.get("sections")), valid_span_ids), fallback_schema.sections, "section_id"),
        entity_types=_merge_schema_objects(_list_value(body.get("entity_types")), fallback_schema.entity_types, "entity_type"),
        field_definitions=fields,
        table_definitions=_merge_schema_objects(_schema_table_definitions_with_valid_refs(_list_value(body.get("table_definitions")), spans), fallback_schema.table_definitions, "table_type"),
        requirement_definitions=_schema_objects_with_valid_source_refs(_list_value(body.get("requirement_definitions")), valid_span_ids),
    )


def _agent_block_input(block: dict[str, Any], spans: list[TextSpan], context: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block.get("block_id"),
        "block_type": block.get("block_type"),
        "source_span_ids": block.get("source_span_ids", []),
        "context_span_ids": block.get("context_span_ids", []),
        "spans": [
            {"span_id": span.span_id, "page": span.page, "text": span.text, "bbox_pdf": to_jsonable(span.bbox_pdf)}
            for span in spans
        ],
        "vdg_context": context,
    }


def _call_llm_agent_method(agent: Any, method_name: str, *args: Any, vdg_context: dict[str, Any] | None) -> dict[str, Any]:
    method = getattr(agent, method_name)
    block_id = _get_nested(vdg_context or {}, "agent_block", "block_id") or "document"
    try:
        try:
            return method(*args, vdg_context=vdg_context)
        except TypeError as exc:
            if "vdg_context" not in str(exc) and "unexpected keyword" not in str(exc):
                raise
            return method(*args)
    except Exception as exc:
        raise ParseError(f"LLM agent call failed: method={method_name}, block_id={block_id}, reason={exc}") from exc


def _get_nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _source_span_ids_from_schema_item(item: dict[str, Any]) -> list[str]:
    if isinstance(item.get("source_span_ids"), list):
        return [str(span_id) for span_id in item["source_span_ids"]]
    if item.get("source_span_id"):
        return [str(item["source_span_id"])]
    if item.get("span_id"):
        return [str(item["span_id"])]
    return []


def _schema_objects_with_valid_source_refs(items: list[dict[str, Any]], valid_span_ids: set[str]) -> list[dict[str, Any]]:
    sanitized = []
    for item in items:
        normalized = dict(item)
        if isinstance(normalized.get("source_span_ids"), list):
            normalized["source_span_ids"] = [
                str(span_id) for span_id in normalized["source_span_ids"] if str(span_id) in valid_span_ids
            ]
        by_table = normalized.get("source_span_ids_by_table")
        if isinstance(by_table, dict):
            normalized["source_span_ids_by_table"] = {
                str(key): [str(span_id) for span_id in value if str(span_id) in valid_span_ids]
                for key, value in by_table.items()
                if isinstance(value, list)
            }
        sanitized.append(normalized)
    return sanitized


def _schema_table_definitions_with_valid_refs(items: list[dict[str, Any]], spans: list[TextSpan]) -> list[dict[str, Any]]:
    valid_span_ids = {span.span_id for span in spans}
    sanitized = _schema_objects_with_valid_source_refs(items, valid_span_ids)
    supported = []
    for item in sanitized:
        if not item.get("source_span_ids"):
            nested_refs = [span_id for span_id in _source_span_ids_from_schema_table(item) if span_id in valid_span_ids]
            item["source_span_ids"] = nested_refs or _table_definition_anchor_span_ids(item, spans)
        if item["source_span_ids"]:
            supported.append(item)
    return supported


def _table_definition_anchor_span_ids(item: dict[str, Any], spans: list[TextSpan]) -> list[str]:
    table_type = str(item.get("table_type") or item.get("type") or "").lower()
    if "nutrition" in table_type:
        anchors = ("营养成分表",)
    elif any(value in table_type for value in ("producer", "manufacturer", "operator")):
        anchors = ("委托方", "受委托方", "生产者", "生产商", "许可证编号")
    elif any(value in table_type for value in ("content", "inventory", "item")):
        anchors = ("内容物", "内装", "规格")
    else:
        return []
    return [span.span_id for span in spans if any(anchor in span.text for anchor in anchors)][:20]


def _schema_anchor_span_ids(item: dict[str, Any], semantic_key: str, display_name: str, spans: list[TextSpan]) -> list[str]:
    anchors = [
        str(value).strip()
        for value in (
            display_name,
            item.get("label"),
            item.get("name"),
            item.get("field"),
            semantic_key.rsplit(".", 1)[-1].replace("_", " "),
        )
        if value
    ]
    span_ids: list[str] = []
    for span in spans:
        text = span.text.strip()
        if not text:
            continue
        if any(anchor and anchor in text for anchor in anchors):
            span_ids.append(span.span_id)
    return span_ids[:3]


def _schema_semantic_key(item: dict[str, Any]) -> str:
    explicit = str(item.get("semantic_key") or "")
    if explicit:
        return explicit
    name = str(item.get("name") or item.get("field") or item.get("field_name") or "").strip()
    if not name:
        return ""
    if "." in name:
        return name
    normalized = name.lower().replace(" ", "_").replace("-", "_")
    entity = str(item.get("entity") or item.get("entity_type") or "").lower()
    aliases = {
        "product_name": "product.name",
        "name": "product.name" if entity == "product" else "",
        "product_type": "product.product_type",
        "ingredients": "product.ingredients",
        "ingredients_text": "product.ingredients",
        "ingredient_list": "product.ingredients",
        "allergen_statement": "custom.allergen_statement",
        "shelf_life": "product.shelf_life",
        "storage_condition": "product.storage_condition",
        "storage_conditions": "product.storage_condition",
        "net_content": "product.net_content",
        "net_weight": "product.net_content",
        "standard_code": "product.standard_code",
        "product_standard_code": "product.standard_code",
        "product_standard_number": "product.standard_code",
        "product_standard": "product.standard_code",
        "origin": "manufacturer.origin",
        "origin_place": "manufacturer.origin",
        "manufacturer_origin": "manufacturer.origin",
        "manufacturer": "manufacturer.name",
        "manufacturer_name": "manufacturer.name",
        "producer": "manufacturer.name",
        "producer_name": "manufacturer.name",
        "address": "manufacturer.address",
        "manufacturer_address": "manufacturer.address",
        "producer_address": "manufacturer.address",
        "license": "manufacturer.license_number",
        "license_number": "manufacturer.license_number",
        "production_license_number": "manufacturer.license_number",
        "barcode": "barcode.commodity",
        "commodity_barcode": "barcode.commodity",
        "food_production_license": "manufacturer.license_number",
        "customer_hotline": "custom.customer_service_hotline",
        "total_juice_content": "custom.total_juice_content",
    }
    mapped = aliases.get(normalized)
    if mapped:
        return mapped
    prefix = {
        "product": "product",
        "manufacturer": "manufacturer",
        "producer": "manufacturer",
        "barcode": "barcode",
        "requirement": "requirement",
    }.get(entity, "custom")
    return f"{prefix}.{normalized}"


def _schema_field_type(item: dict[str, Any]) -> str:
    raw_type = str(item.get("field_type") or item.get("type") or item.get("data_type") or "string")
    normalized = raw_type.lower()
    if normalized.startswith("list"):
        return "long_text"
    if normalized in {"integer", "float", "decimal"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "enum"
    if normalized in {"text", "string"}:
        return "string"
    if normalized in {"long_text", "number", "date", "barcode", "table", "entity", "requirement", "enum", "unknown"}:
        return normalized
    return "string"


def _schema_criticality(item: dict[str, Any], semantic_key: str) -> str:
    raw = str(item.get("criticality") or "").lower()
    if raw in {"critical", "required", "mandatory", "high"}:
        return "critical"
    if raw in {"non_critical", "optional"}:
        return "non_critical"
    if raw in {"medium", "low", "info", "unknown"}:
        return raw
    if semantic_key.startswith("custom.") or semantic_key in {"product.allergens", "product.serving_size"}:
        return "non_critical"
    return "critical"


def _merge_schema_objects(agent_items: list[dict[str, Any]], fallback_items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in agent_items + fallback_items:
        normalized = _normalize_schema_object(item, key)
        item_key = str(normalized.get(key) or "")
        if not item_key:
            continue
        if item_key in seen:
            existing = next((candidate for candidate in merged if str(candidate.get(key) or "") == item_key), None)
            if existing is not None:
                _fill_missing_schema_object_fields(existing, normalized)
            continue
        merged.append(normalized)
        seen.add(item_key)
    return merged


def _fill_missing_schema_object_fields(target: dict[str, Any], fallback: dict[str, Any]) -> None:
    for field_key, value in fallback.items():
        if field_key not in target or target.get(field_key) in (None, "", []):
            if value not in (None, "", []):
                target[field_key] = value


def _normalize_schema_object(item: dict[str, Any], key: str) -> dict[str, Any]:
    normalized = dict(item)
    if key == "section_id":
        section_id = str(normalized.get("section_id") or normalized.get("id") or normalized.get("name") or "").strip()
        if section_id:
            section_type = str(
                normalized.get("section_type")
                or normalized.get("type")
                or normalized.get("name")
                or section_id.removeprefix("sec_")
            ).strip()
            normalized["section_id"] = section_id
            normalized["section_type"] = section_type or section_id
            normalized.setdefault(
                "display_name",
                normalized.get("title") or normalized.get("description") or normalized["section_type"],
            )
            if "source_span_ids" in normalized and not isinstance(normalized["source_span_ids"], list):
                normalized["source_span_ids"] = [str(normalized["source_span_ids"])]
    if key == "entity_type":
        entity_type = str(normalized.get("entity_type") or normalized.get("name") or "").strip()
        if entity_type:
            entity_type = {
                "manufacturer": "manufacturer",
                "producer": "manufacturer",
                "product": "product",
                "contentitem": "content_item",
                "content_item": "content_item",
                "barcode": "barcode",
                "requirement": "requirement",
            }.get(entity_type.replace(" ", "").lower(), entity_type)
            normalized["entity_type"] = entity_type
            normalized.setdefault("repeatable", entity_type in {"manufacturer", "content_item", "barcode", "requirement"})
    if key == "table_type":
        table_type = str(normalized.get("table_type") or normalized.get("table_name") or normalized.get("name") or "").strip()
        if table_type:
            normalized["table_type"] = _normalize_table_type(table_type)
            normalized.setdefault("display_name", normalized.get("title") or normalized.get("description") or normalized["table_type"])
            normalized.setdefault("repeatable", True)
            normalized.setdefault("criticality", "critical" if normalized["table_type"] == "nutrition_facts" else "non_critical")
            if not normalized.get("source_span_ids"):
                normalized["source_span_ids"] = _source_span_ids_from_schema_table(normalized)
    return normalized


def _source_span_ids_from_schema_table(table: dict[str, Any]) -> list[str]:
    span_ids: list[str] = []
    by_table = table.get("source_span_ids_by_table")
    if isinstance(by_table, dict):
        for raw_span_ids in by_table.values():
            if isinstance(raw_span_ids, list):
                span_ids.extend(str(span_id) for span_id in raw_span_ids)
    for collection_key in ("columns", "rows", "column_definitions", "row_definitions"):
        collection = table.get(collection_key)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            for span_key in ("source_span_ids", "source_span_ids_example"):
                raw_span_ids = item.get(span_key)
                if isinstance(raw_span_ids, list):
                    span_ids.extend(str(span_id) for span_id in raw_span_ids)
    return _unique_refs(span_ids)


def _normalize_table_type(table_type: str) -> str:
    normalized = table_type.replace(" ", "_").replace("-", "_").lower()
    if normalized in {"nutrition", "nutritiontable", "nutrition_table", "nutritionfacts", "nutrition_facts"}:
        return "nutrition_facts"
    return table_type


def _list_value(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _merge_field_retry_plan(base_plan: ExtractionPlan, retry_plan: ExtractionPlan) -> ExtractionPlan:
    return ExtractionPlan(
        plan_id=base_plan.plan_id,
        schema_id=base_plan.schema_id,
        fields=retry_plan.fields,
        entities=base_plan.entities or retry_plan.entities,
        tables=base_plan.tables or retry_plan.tables,
        requirements=base_plan.requirements or retry_plan.requirements,
        ignored_nodes=base_plan.ignored_nodes or retry_plan.ignored_nodes,
        unknown_nodes=base_plan.unknown_nodes or retry_plan.unknown_nodes,
        ignored_node_reasons=base_plan.ignored_node_reasons or retry_plan.ignored_node_reasons,
    )


def _merge_table_retry_plan(base_plan: ExtractionPlan, retry_plan: ExtractionPlan) -> ExtractionPlan:
    return ExtractionPlan(
        plan_id=base_plan.plan_id,
        schema_id=base_plan.schema_id,
        fields=base_plan.fields,
        entities=base_plan.entities or retry_plan.entities,
        tables=retry_plan.tables,
        requirements=base_plan.requirements or retry_plan.requirements,
        ignored_nodes=base_plan.ignored_nodes or retry_plan.ignored_nodes,
        unknown_nodes=base_plan.unknown_nodes or retry_plan.unknown_nodes,
        ignored_node_reasons=base_plan.ignored_node_reasons or retry_plan.ignored_node_reasons,
    )


def _agent_schema_field_count(body: dict[str, Any] | None) -> int:
    if not isinstance(body, dict):
        return 0
    for key in ("field_definitions", "fields", "items"):
        items = body.get(key)
        if isinstance(items, list):
            return len(items)
    return 0


def _agent_table_plan_count(body: dict[str, Any] | None) -> int:
    if not isinstance(body, dict):
        return 0
    for key in ("extraction_plan", "plan", "proposal"):
        nested = body.get(key)
        if isinstance(nested, dict):
            return _agent_table_plan_count(nested)
    if isinstance(body.get("tables"), list):
        return len(body["tables"])
    if isinstance(body.get("nutrition_facts_table"), dict) or isinstance(body.get("nutrition_table"), dict):
        return 1
    return 0


def _agent_plan_item_count(body: dict[str, Any] | None) -> int:
    if not body:
        return 0
    for key in ("extraction_plan", "plan", "proposal"):
        nested = body.get(key)
        if isinstance(nested, dict):
            return _agent_plan_item_count(nested)
    if isinstance(body.get("fields"), list):
        return len(body["fields"])
    if isinstance(body.get("field_plans"), list):
        return len(body["field_plans"])
    if isinstance(body.get("extraction_fields"), list):
        return len(body["extraction_fields"])
    if isinstance(body.get("items"), list):
        return len(body["items"])
    return 0


def _apply_table_feed_structure(
    compiled_fields: dict[str, CompiledField],
    evidence: list[Evidence],
    tables: list[dict[str, Any]],
    table_quality_report: dict[str, Any],
    runtime_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    mode = _table_parser_mode(runtime_policy)
    feed = {
        "mode": mode,
        "status": "disabled",
        "items": [],
        "tables": [],
        "warnings": [],
    }
    if mode != "feed_structure":
        feed["warnings"].append("feed_structure_mode_not_enabled")
        return feed
    if table_quality_report.get("status") != "pass":
        feed["status"] = "skipped_quality_not_pass"
        feed["warnings"].append("table_quality_report_not_pass")
        return feed

    for table in tables:
        if table.get("table_type") != "nutrition_facts":
            continue
        raw_value = _table_feed_raw_value(table)
        if not raw_value:
            feed["warnings"].append(f"table_feed_empty_text:{table.get('table_id', 'unknown_table')}")
            continue
        evidence_id = stable_id("ev", len(evidence) + 1)
        bbox_status = "available" if table.get("bbox_status") == "available" else "missing"
        evidence.append(
            Evidence(
                evidence_id=evidence_id,
                source_text=raw_value,
                page=_table_page(table),
                extraction_methods=["table_parser_feed"],
                bbox_status=bbox_status,
                source_node_ids=[str(span_id) for span_id in table.get("source_span_ids", []) if span_id],
                bbox_pdf=table.get("bbox_pdf"),
                bbox_normalized=table.get("bbox_normalized"),
            )
        )
        confidence = _table_feed_confidence(table, bbox_status)
        review_required = float(confidence["overall"]) < 0.95
        field_id = stable_id("fld", len(compiled_fields) + 1)
        compiled_fields[field_id] = CompiledField(
            field_id=field_id,
            semantic_key="product.nutrition_table",
            display_name="营养成分表",
            field_type="table",
            raw_value=raw_value,
            clean_value=raw_value,
            normalized_value=raw_value,
            value_hash=sha256_text(raw_value),
            status="manual_review_required" if review_required else "verified",
            criticality="critical",
            confidence=confidence,
            risk_level="high" if review_required else "info",
            review_required=review_required,
            section_id="sec_label_text",
            entity_id=table.get("linked_entity_id") or "product_001",
            table_id=table.get("table_id"),
            row_key=None,
            evidence_refs=[evidence_id],
            normalization=[],
            reason="表格字段置信度低于0.95" if review_required else None,
        )
        feed["items"].append(
            {
                "field_id": field_id,
                "semantic_key": "product.nutrition_table",
                "table_id": table.get("table_id"),
                "table_layer_id": table.get("table_layer_id"),
                "evidence_refs": [evidence_id],
                "status": "manual_review_required" if review_required else "verified",
                "review_required": review_required,
            }
        )
        feed["tables"].append(
            {
                "table_id": table.get("table_id"),
                "table_layer_id": table.get("table_layer_id"),
                "table_type": table.get("table_type"),
                "row_count": len(table.get("rows", [])),
            }
        )

    feed["status"] = "applied" if feed["items"] else "skipped_no_supported_tables"
    return feed


def _table_parser_mode(runtime_policy: dict[str, Any] | None) -> str:
    policy = runtime_policy or {}
    table_parser = policy.get("table_parser") if isinstance(policy.get("table_parser"), dict) else {}
    effective_options = policy.get("effective_options") if isinstance(policy.get("effective_options"), dict) else {}
    return str(table_parser.get("mode") or effective_options.get("table_parser_mode") or "validate_only")


def _layout_mode(runtime_policy: dict[str, Any] | None) -> str:
    policy = runtime_policy or {}
    effective_options = policy.get("effective_options") if isinstance(policy.get("effective_options"), dict) else {}
    return str(effective_options.get("layout_mode") or policy.get("layout_mode") or "legacy")


def _layout_candidate_acceptance(
    layout_candidates: dict[str, Any],
    spans: list[TextSpan],
    agent_bodies: list[dict[str, Any] | None],
    layout_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if layout_mode != ENHANCED_LAYOUT_MODE:
        return {"status": "disabled", "decisions": [], "unaccepted_candidate_count": 0}, [], []
    candidates = {
        str(candidate.get("table_candidate_id")): candidate
        for candidate in layout_candidates.get("table_candidates", [])
        if isinstance(candidate, dict) and candidate.get("table_candidate_id")
    }
    valid_span_ids = {span.span_id for span in spans}
    decisions: dict[str, dict[str, Any]] = {}
    invalid_decisions: list[dict[str, Any]] = []
    for body in agent_bodies:
        if not isinstance(body, dict):
            continue
        for item in body.get("layout_candidate_decisions", []):
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("layout_candidate_id") or "")
            decision = str(item.get("decision") or "")
            refs = [str(value) for value in item.get("source_span_ids", []) if value]
            candidate = candidates.get(candidate_id)
            candidate_refs = set(candidate.get("source_span_ids", [])) if candidate else set()
            if candidate is None or decision not in {"accept", "reject", "unresolved"} or not refs or not set(refs) <= valid_span_ids or not set(refs) <= candidate_refs:
                invalid_decisions.append({"layout_candidate_id": candidate_id, "reason": "invalid_layout_candidate_decision_refs"})
                continue
            decisions[candidate_id] = {
                "layout_candidate_id": candidate_id,
                "decision": decision,
                "source_span_ids": refs,
                "reason": str(item.get("reason") or "agent_layout_decision"),
                "confidence": item.get("confidence"),
            }

    reviews: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        decision = decisions.get(candidate_id)
        accepted = bool(decision and decision["decision"] == "accept")
        checks.append(
            {
                "validation_id": stable_id("layout_val", len(checks) + 1),
                "target_id": candidate_id,
                "check_type": "layout_candidate_acceptance",
                "result": "passed" if accepted else "failed",
                "severity": "high" if candidate.get("table_type") == "nutrition_facts" else "medium",
                "message": "Layout candidate was accepted by the extraction agent." if accepted else "Layout candidate was not accepted by the extraction agent and requires review.",
                "source_span_ids": candidate.get("source_span_ids", []),
            }
        )
        if not accepted:
            reviews.append(
                {
                    "target_id": candidate_id,
                    "layout_candidate_id": candidate_id,
                    "candidate_type": candidate.get("table_type"),
                    "reason": f"layout_candidate_{decision['decision'] if decision else 'missing_agent_decision'}",
                    "required": True,
                    "source_span_ids": candidate.get("source_span_ids", []),
                    "bbox_normalized": candidate.get("bbox_normalized"),
                }
            )
    quality_status = str(layout_candidates.get("status") or "enabled")
    return (
        {
            "report_version": "layout_candidate_acceptance_v0.1",
            "status": "pass" if not reviews and not invalid_decisions else "review_required",
            "layout_mode": layout_mode,
            "candidate_count": len(candidates),
            "accepted_candidate_count": sum(1 for item in decisions.values() if item["decision"] == "accept"),
            "unaccepted_candidate_count": len(reviews),
            "invalid_decision_count": len(invalid_decisions),
            "decisions": list(decisions.values()),
            "invalid_decisions": invalid_decisions,
            "candidate_artifact_status": quality_status,
        },
        reviews,
        checks,
    )


def _layout_quality_validation_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    status = str(report.get("status") or "fail")
    if status == "disabled":
        return []
    return [
        {
            "validation_id": "layout_quality_0001",
            "target_id": "document",
            "check_type": "layout_quality",
            "result": "passed" if status == "pass" else "failed",
            "severity": "high" if status == "fail" else "medium",
            "message": "Layout evidence quality passed." if status == "pass" else "Layout evidence has unresolved structural issues.",
            "issues": report.get("issues", []),
        }
    ]


def _table_feed_raw_value(table: dict[str, Any]) -> str:
    parts = [str(table.get("title", "")).strip()]
    for row in table.get("rows", []):
        if not isinstance(row, dict):
            continue
        cell_text = " ".join(
            str(cell.get("raw_value") or cell.get("normalized_value") or "").strip()
            for cell in row.get("cells", [])
            if isinstance(cell, dict) and str(cell.get("raw_value") or cell.get("normalized_value") or "").strip()
        ).strip()
        if cell_text:
            parts.append(cell_text)
    return "\n".join(part for part in parts if part).strip()


def _table_page(table: dict[str, Any]) -> int:
    page = table.get("page")
    if isinstance(page, int):
        return page
    try:
        return int(page)
    except (TypeError, ValueError):
        return 1


def _table_feed_confidence(table: dict[str, Any], bbox_status: str) -> dict[str, float]:
    table_confidence = table.get("confidence") if isinstance(table.get("confidence"), dict) else {}
    confidence = {
        "schema_confidence": 0.95,
        "boundary_confidence": 0.96,
        "table_structure_confidence": float(table_confidence.get("table_structure_confidence") or 0.50),
        "evidence_confidence": float(table_confidence.get("evidence_confidence") or (1.0 if bbox_status == "available" else 0.80)),
    }
    confidence["overall"] = min(confidence.values())
    return confidence


def _extract_tables_from_agent_plan(
    table_plans: list[dict[str, Any]],
    spans: list[TextSpan],
    content_names: dict[str, str],
    existing_evidence_count: int,
) -> tuple[list[dict[str, Any]], list[Evidence]]:
    span_by_id = {span.span_id: span for span in spans}
    tables: list[dict[str, Any]] = []
    evidence: list[Evidence] = []
    for table_index, table_plan in enumerate(table_plans, start=1):
        if not isinstance(table_plan, dict):
            continue
        table_type = _normalize_table_type(str(table_plan.get("table_type", "nutrition_facts")))
        rows = [row for row in table_plan.get("rows", []) if isinstance(row, dict)]
        if not rows:
            continue
        table_span_ids = _table_source_span_ids(table_plan, rows)
        source_spans = [span_by_id[span_id] for span_id in table_span_ids if span_id in span_by_id]
        table_evidence_refs: list[str] = []
        final_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows, start=1):
            cells = [cell for cell in row.get("cells", []) if isinstance(cell, dict)]
            if table_type == "nutrition_facts":
                cells = _merge_nutrition_marker_cells(cells, span_by_id)
                cells = _canonical_nutrition_cells(cells, span_by_id)
            row_span_ids = _table_source_span_ids(row, cells) or table_span_ids
            for span_id in row_span_ids:
                if span_id not in table_span_ids:
                    table_span_ids.append(span_id)
            row_text = " ".join(_agent_cell_text(cell, span_by_id) for cell in cells).strip()
            if not row_text:
                row_text = str(row.get("text") or row.get("raw_value") or row.get("value") or "").strip()
            if not cells and not row_text:
                continue
            ev_id = stable_id("ev", existing_evidence_count + len(evidence) + 1)
            bbox_span = next((span_by_id[span_id] for span_id in row_span_ids if span_id in span_by_id and span_by_id[span_id].bbox_pdf), None)
            evidence.append(
                Evidence(
                    evidence_id=ev_id,
                    source_text=row_text,
                    page=int(table_plan.get("page") or (bbox_span.page if bbox_span else 1)),
                    extraction_methods=["llm_table_agent"],
                    bbox_status="available" if bbox_span and bbox_span.bbox_pdf else "missing",
                    source_node_ids=[span_id for span_id in row_span_ids if span_id in span_by_id],
                    bbox_pdf=bbox_span.bbox_pdf if bbox_span else None,
                    bbox_normalized=bbox_span.bbox_normalized if bbox_span else None,
                )
            )
            table_evidence_refs.append(ev_id)
            final_rows.append(
                {
                    "row_id": stable_id("row", row_index),
                    "row_key": str(row.get("row_key") or _first_cell_text(cells) or "unknown"),
                    "evidence_refs": [ev_id],
                    "cells": [
                        {
                            "column_id": str(cell.get("column_id") or f"col_{cell_index + 1:03d}"),
                            "raw_value": _agent_cell_text(cell, span_by_id),
                            "normalized_value": str(cell.get("normalized_value") or _agent_cell_text(cell, span_by_id)),
                            "evidence_refs": [ev_id],
                        }
                        for cell_index, cell in enumerate(cells)
                    ],
                }
            )

        confidence = float(table_plan.get("confidence", 0.96))
        if not final_rows:
            continue
        source_spans = [span_by_id[span_id] for span_id in table_span_ids if span_id in span_by_id]
        table_page = int(table_plan.get("page") or (source_spans[0].page if source_spans else 1))
        bbox_metadata = _table_bbox_metadata_from_spans(source_spans, table_page)
        has_bbox = bbox_metadata["bbox_status"] == "available"
        columns = _nutrition_columns() if table_type == "nutrition_facts" else _agent_table_columns(table_plan, final_rows)
        if not columns:
            continue
        _normalize_agent_table_cell_columns(final_rows, columns, table_type)
        _pad_rows_to_columns(final_rows, columns)
        table_confidence = {
            "table_structure_confidence": confidence,
            "evidence_confidence": 1.0 if has_bbox else 0.80,
        }
        table_review_required = not final_rows or min(table_confidence.values()) < 0.95
        tables.append(
                {
                    "table_id": stable_id("tbl", table_index),
                    "table_type": table_type,
                "page": table_page,
                "title": table_plan.get("title", "营养成分表"),
                "linked_entity_id": table_plan.get("linked_entity_id") or _linked_content_entity_from_names(str(table_plan.get("title", "")), content_names),
                "columns": columns,
                "rows": final_rows,
                "status": "manual_review_required" if table_review_required else "verified",
                "bbox_status": bbox_metadata["bbox_status"],
                "bbox_pdf": bbox_metadata.get("bbox_pdf"),
                "bbox_normalized": bbox_metadata.get("bbox_normalized"),
                "confidence": table_confidence,
                "criticality": table_plan.get("criticality", "critical"),
                "risk_level": "high" if table_review_required else "low",
                "review_required": table_review_required,
                "evidence_refs": table_evidence_refs,
                "source_span_ids": table_span_ids,
                "source": "llm_table_agent",
            }
        )
    return tables, evidence


def _extract_tables_from_accepted_layout_candidates(
    layout_candidates: dict[str, Any],
    acceptance_report: dict[str, Any],
    spans: list[TextSpan],
    content_names: dict[str, str],
    existing_tables: list[dict[str, Any]],
    existing_evidence_count: int,
) -> tuple[list[dict[str, Any]], list[Evidence]]:
    accepted_ids = {
        str(item.get("layout_candidate_id"))
        for item in acceptance_report.get("decisions", [])
        if isinstance(item, dict) and item.get("decision") == "accept"
    }
    span_by_id = {span.span_id: span for span in spans}
    tables: list[dict[str, Any]] = []
    evidence: list[Evidence] = []
    for candidate in layout_candidates.get("table_candidates", []):
        if not isinstance(candidate, dict) or candidate.get("table_type") != "nutrition_facts":
            continue
        candidate_id = str(candidate.get("table_candidate_id") or "")
        candidate_ref_list = [str(value) for value in candidate.get("source_span_ids", [])]
        candidate_refs = set(candidate_ref_list)
        if candidate_id not in accepted_ids or _layout_candidate_is_represented(candidate_refs, existing_tables):
            continue
        final_rows = []
        table_evidence_refs = []
        for raw_row in candidate.get("rows", []):
            if not isinstance(raw_row, dict):
                continue
            raw_cells = _merge_nutrition_marker_cells(
                [cell for cell in raw_row.get("cells", []) if isinstance(cell, dict)],
                span_by_id,
            )
            raw_cells = _canonical_nutrition_cells(raw_cells, span_by_id)
            row_refs = [str(value) for value in raw_row.get("source_span_ids", []) if str(value) in span_by_id]
            if not raw_cells or not row_refs:
                continue
            first_span = next((span_by_id[span_id] for span_id in row_refs if span_by_id[span_id].bbox_pdf), None)
            row_text = " ".join(str(cell.get("text") or "") for cell in raw_cells).strip()
            evidence_id = stable_id("ev", existing_evidence_count + len(evidence) + 1)
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    source_text=row_text,
                    page=first_span.page if first_span else int(candidate.get("page") or 1),
                    extraction_methods=["llm_accepted_layout_candidate"],
                    bbox_status="available" if first_span and first_span.bbox_pdf else "missing",
                    source_node_ids=row_refs,
                    bbox_pdf=first_span.bbox_pdf if first_span else None,
                    bbox_normalized=first_span.bbox_normalized if first_span else None,
                )
            )
            table_evidence_refs.append(evidence_id)
            final_rows.append(
                {
                    "row_id": stable_id("row", len(final_rows) + 1),
                    "row_key": str(raw_row.get("row_key") or _first_cell_text(raw_cells) or "unknown"),
                    "evidence_refs": [evidence_id],
                    "cells": [
                        {
                            "column_id": str(cell.get("column_id") or cell.get("col_id") or f"col_{index + 1:03d}"),
                            "raw_value": str(cell.get("text") or ""),
                            "normalized_value": str(cell.get("text") or ""),
                            "evidence_refs": [evidence_id],
                        }
                        for index, cell in enumerate(raw_cells)
                    ],
                }
            )
        if not final_rows:
            continue
        table_index = len(existing_tables) + len(tables) + 1
        columns = _nutrition_columns()
        _normalize_agent_table_cell_columns(final_rows, columns, "nutrition_facts")
        _pad_rows_to_columns(final_rows, columns)
        tables.append(
            {
                "table_id": stable_id("tbl", table_index),
                "table_layer_id": candidate_id,
                "table_type": "nutrition_facts",
                "page": int(candidate.get("page") or 1),
                "title": candidate.get("title", "营养成分表"),
                "linked_entity_id": _linked_content_entity_from_names(str(candidate.get("title") or ""), content_names),
                "columns": columns,
                "rows": final_rows,
                "status": "manual_review_required",
                "bbox_status": candidate.get("bbox_status", "available"),
                "bbox_pdf": candidate.get("bbox_pdf"),
                "bbox_normalized": candidate.get("bbox_normalized"),
                "confidence": {"table_structure_confidence": float(candidate.get("confidence", 0.72)), "evidence_confidence": 1.0},
                "criticality": "critical",
                "risk_level": "high",
                "review_required": True,
                "evidence_refs": table_evidence_refs,
                "source_span_ids": candidate_ref_list,
                "source": "llm_accepted_layout_candidate",
            }
        )
    return tables, evidence


def _layout_candidate_is_represented(candidate_refs: set[str], tables: list[dict[str, Any]]) -> bool:
    if not candidate_refs:
        return False
    return any(
        len(candidate_refs & {str(value) for value in table.get("source_span_ids", [])}) / len(candidate_refs) >= 0.5
        and len(table.get("rows", [])) >= 3
        for table in tables
        if isinstance(table, dict)
    )


def _merge_nutrition_marker_cells(cells: list[dict[str, Any]], span_by_id: dict[str, TextSpan]) -> list[dict[str, Any]]:
    if len(cells) < 2:
        return cells
    marker = _agent_cell_text(cells[0], span_by_id).strip()
    label = _agent_cell_text(cells[1], span_by_id).strip()
    if marker not in {"-", "--", "—", "——"} or label not in {"饱和脂肪", "反式脂肪酸", "糖"}:
        return cells
    marker_refs = _cell_source_span_ids(cells[0])
    label_refs = _cell_source_span_ids(cells[1])
    if not _source_refs_share_row(marker_refs, label_refs, span_by_id):
        return cells
    merged = dict(cells[0])
    merged_text = marker + label
    merged["text"] = merged_text
    merged["raw_value"] = merged_text
    merged["normalized_value"] = merged_text
    merged["source_span_ids"] = [*marker_refs, *[ref for ref in label_refs if ref not in marker_refs]]
    merged.pop("span_id", None)
    merged.pop("source_span_id", None)
    return [merged, *cells[2:]]


def _canonical_nutrition_cells(cells: list[dict[str, Any]], span_by_id: dict[str, TextSpan]) -> list[dict[str, Any]]:
    if len(cells) <= 1:
        return cells
    texts = [_agent_cell_text(cell, span_by_id).strip() for cell in cells]
    if texts[0] == "项目":
        middle = _merge_table_cells(cells[1:-1], span_by_id)
        return [cells[0], middle, cells[-1]] if middle else [cells[0], cells[-1]]
    if len(cells) >= 4 and re.fullmatch(r"(?:千焦|克|毫克|微克|毫升|升)", texts[-2]):
        return [cells[0], _merge_table_cells(cells[1:-1], span_by_id), cells[-1]]
    return cells[:3]


def _merge_table_cells(cells: list[dict[str, Any]], span_by_id: dict[str, TextSpan]) -> dict[str, Any]:
    if not cells:
        return {}
    merged = dict(cells[0])
    text = "".join(_agent_cell_text(cell, span_by_id) for cell in cells)
    source_ids = []
    for cell in cells:
        source_ids.extend(span_id for span_id in _cell_source_span_ids(cell) if span_id not in source_ids)
    merged["text"] = text
    merged["raw_value"] = text
    merged["normalized_value"] = text
    merged["source_span_ids"] = source_ids
    merged.pop("span_id", None)
    merged.pop("source_span_id", None)
    return merged


def _nutrition_columns() -> list[dict[str, Any]]:
    return [
        {"column_id": "col_001", "name": "项目", "semantic_key": "nutrient_item"},
        {"column_id": "col_002", "name": "含量", "semantic_key": "amount_with_unit"},
        {"column_id": "col_003", "name": "NRV%", "semantic_key": "nrv_percent"},
    ]


def _cell_source_span_ids(cell: dict[str, Any]) -> list[str]:
    values = cell.get("source_span_ids")
    if isinstance(values, list):
        return [str(value) for value in values]
    source_id = cell.get("span_id") or cell.get("source_span_id")
    if source_id:
        return [str(source_id)]
    return [
        str(item.get("span_id") or item.get("source_span_id"))
        for item in cell.get("ranges", []) if isinstance(cell.get("ranges"), list)
        if isinstance(item, dict) and (item.get("span_id") or item.get("source_span_id"))
    ]


def _source_refs_share_row(left_refs: list[str], right_refs: list[str], span_by_id: dict[str, TextSpan]) -> bool:
    for left_ref in left_refs:
        left = span_by_id.get(left_ref)
        if left is None or left.bbox_pdf is None:
            continue
        for right_ref in right_refs:
            right = span_by_id.get(right_ref)
            if right is None or right.bbox_pdf is None or left.page != right.page:
                continue
            tolerance = max(left.bbox_pdf.height, right.bbox_pdf.height, 2.0) * 0.75
            left_center = left.bbox_pdf.y + left.bbox_pdf.height / 2
            right_center = right.bbox_pdf.y + right.bbox_pdf.height / 2
            if abs(left_center - right_center) <= tolerance:
                return True
    return False


def _drop_incomplete_agent_tables_replaced_by_layout_candidates(
    tables: list[dict[str, Any]],
    layout_candidates: dict[str, Any],
    acceptance_report: dict[str, Any],
) -> list[dict[str, Any]]:
    accepted_ids = {
        str(item.get("layout_candidate_id"))
        for item in acceptance_report.get("decisions", [])
        if isinstance(item, dict) and item.get("decision") == "accept"
    }
    accepted_ref_sets = [
        {str(value) for value in candidate.get("source_span_ids", [])}
        for candidate in layout_candidates.get("table_candidates", [])
        if isinstance(candidate, dict)
        and candidate.get("table_type") == "nutrition_facts"
        and str(candidate.get("table_candidate_id")) in accepted_ids
    ]
    return [
        table
        for table in tables
        if len(table.get("rows", [])) >= 3
        or not any({str(value) for value in table.get("source_span_ids", [])} & refs for refs in accepted_ref_sets)
    ]


def _has_structured_agent_table(table_plans: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(table, dict)
        and any(isinstance(row, dict) and row.get("cells") for row in table.get("rows", []))
        for table in table_plans
    )


def _table_retry_decision(table_layers: dict[str, Any], *, allow_large_candidate_set: bool = False) -> dict[str, Any]:
    candidate_tables = [table for table in table_layers.get("tables", []) if isinstance(table, dict)]
    candidate_cell_count = sum(
        len(row.get("cells", []))
        for table in candidate_tables
        for row in table.get("rows", [])
        if isinstance(row, dict)
    )
    candidate_row_count = sum(
        len(table.get("rows", []))
        for table in candidate_tables
    )
    decision = {
        "item_type": "table_retry",
        "reason": "llm_table_retry_allowed",
        "should_retry": True,
        "candidate_table_count": len(candidate_tables),
        "candidate_row_count": candidate_row_count,
        "candidate_cell_count": candidate_cell_count,
    }
    if not allow_large_candidate_set and (len(candidate_tables) > 3 or candidate_cell_count > 80):
        decision.update(
            {
                "reason": "llm_table_retry_skipped_large_candidate_set",
                "should_retry": False,
                "message": "Table parser/VDG candidates are used with review instead of one large online LLM table retry.",
            }
        )
    return decision


def _table_retry_spans(spans: list[TextSpan], table_layers: dict[str, Any]) -> list[TextSpan]:
    nutrition_refs = {
        str(span_id)
        for table in table_layers.get("tables", [])
        if isinstance(table, dict) and table.get("table_type") == "nutrition_facts"
        for span_id in table.get("source_span_ids", [])
    }
    return [span for span in spans if span.span_id in nutrition_refs]


def _table_retry_context(vdg_context: dict[str, Any] | None, table_layers: dict[str, Any]) -> dict[str, Any] | None:
    if not vdg_context:
        return None
    nutrition_tables = [
        table for table in table_layers.get("tables", [])
        if isinstance(table, dict) and table.get("table_type") == "nutrition_facts"
    ]
    nutrition_refs = {
        str(span_id)
        for table in nutrition_tables
        for span_id in table.get("source_span_ids", [])
    }
    context = {
        "context_version": vdg_context.get("context_version"),
        "vdg_quality_status": vdg_context.get("vdg_quality_status"),
        "agent_readiness": vdg_context.get("agent_readiness"),
        "table_candidates": [
            candidate for candidate in vdg_context.get("table_candidates", [])
            if isinstance(candidate, dict) and candidate.get("table_type") == "nutrition_facts"
        ],
        "reading_order_candidates": [
            edge for edge in vdg_context.get("reading_order_candidates", [])
            if isinstance(edge, dict) and edge.get("source_span_id") in nutrition_refs and edge.get("target_span_id") in nutrition_refs
        ],
        "quality_issues": [
            issue for issue in vdg_context.get("quality_issues", [])
            if isinstance(issue, dict) and (issue.get("node_id") in nutrition_refs or "table" in str(issue.get("issue_type", "")))
        ],
        "layout_quality": vdg_context.get("layout_quality", {}),
        "label_text_scope": vdg_context.get("label_text_scope", {}),
    }
    return context


FIELD_RETRY_ANCHORS = (
    "产品名称",
    "品名",
    "品牌",
    "净含量",
    "规格",
    "产品类别",
    "配料",
    "致敏",
    "贮存",
    "储存",
    "保质期",
    "生产日期",
    "批号",
    "食用方法",
    "冲调方法",
    "警示",
    "产品标准",
    "执行标准",
    "质量等级",
    "产地",
    "商品条码",
    "外箱条码",
    "委托方",
    "受委托方",
    "被委托方",
    "生产者",
    "生产商",
    "经销商",
    "进口商",
    "地址",
    "许可证",
    "联系方式",
)


def _field_retry_spans(spans: list[TextSpan], schema: GeneratedSchema, max_spans: int = 120) -> list[TextSpan]:
    schema_span_ids: set[str] = set()
    for definition in schema.field_definitions:
        schema_span_ids.update(str(span_id) for span_id in definition.source_span_ids)
    selected: list[TextSpan] = []
    for span in spans:
        if span.span_id in schema_span_ids or any(anchor in span.text for anchor in FIELD_RETRY_ANCHORS):
            selected.append(span)
        if len(selected) >= max_spans:
            break
    return selected or spans[:max_spans]


def _table_source_span_ids(container: dict[str, Any], nested_items: list[dict[str, Any]]) -> list[str]:
    span_ids: list[str] = []
    raw_span_ids = container.get("source_span_ids", [])
    if isinstance(raw_span_ids, list):
        span_ids.extend(str(span_id) for span_id in raw_span_ids)
    by_table = container.get("source_span_ids_by_table")
    if isinstance(by_table, dict):
        for item_span_ids in by_table.values():
            if isinstance(item_span_ids, list):
                span_ids.extend(str(span_id) for span_id in item_span_ids)
    for key in ("source_span_id", "span_id"):
        if container.get(key):
            span_ids.append(str(container[key]))
    for item in nested_items:
        item_span_ids = item.get("source_span_ids", [])
        if isinstance(item_span_ids, list):
            span_ids.extend(str(span_id) for span_id in item_span_ids)
        for key in ("source_span_id", "span_id"):
            if item.get(key):
                span_ids.append(str(item[key]))
        for range_item in item.get("ranges", []) if isinstance(item.get("ranges"), list) else []:
            if isinstance(range_item, dict) and (range_item.get("span_id") or range_item.get("source_span_id")):
                span_ids.append(str(range_item.get("span_id") or range_item.get("source_span_id")))
    deduped: list[str] = []
    for span_id in span_ids:
        if span_id not in deduped:
            deduped.append(span_id)
    return deduped


def _table_bbox_metadata_from_spans(spans: list[TextSpan], page: int) -> dict[str, Any]:
    bbox_spans = [span for span in spans if span.page == page and span.bbox_pdf]
    if not bbox_spans:
        return {"bbox_status": "missing"}
    x1 = min(float(span.bbox_pdf.x) for span in bbox_spans if span.bbox_pdf)
    y1 = min(float(span.bbox_pdf.y) for span in bbox_spans if span.bbox_pdf)
    x2 = max(float(span.bbox_pdf.x + span.bbox_pdf.width) for span in bbox_spans if span.bbox_pdf)
    y2 = max(float(span.bbox_pdf.y + span.bbox_pdf.height) for span in bbox_spans if span.bbox_pdf)
    first_bbox = next(span.bbox_pdf for span in bbox_spans if span.bbox_pdf)
    page_width = float(first_bbox.page_width)
    page_height = float(first_bbox.page_height)
    return {
        "bbox_status": "available",
        "bbox_pdf": {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "page_width": page_width,
            "page_height": page_height,
            "unit": first_bbox.unit,
            "origin": first_bbox.origin,
        },
        "bbox_normalized": {
            "x1": _clamp_ratio(x1, page_width),
            "y1": _clamp_ratio(y1, page_height),
            "x2": _clamp_ratio(x2, page_width),
            "y2": _clamp_ratio(y2, page_height),
        },
    }


def _clamp_ratio(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, round(value / denominator, 6)))


def _first_cell_text(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return ""
    return str(cells[0].get("text") or cells[0].get("raw_value") or cells[0].get("value") or "")


def _agent_cell_text(cell: dict[str, Any], span_by_id: dict[str, TextSpan]) -> str:
    explicit = str(cell.get("text") or cell.get("raw_value") or cell.get("value") or "")
    if explicit:
        return explicit
    ranges = cell.get("ranges")
    if isinstance(ranges, list) and ranges:
        parts = []
        for item in ranges:
            if not isinstance(item, dict):
                return ""
            span_id = str(item.get("span_id") or item.get("source_span_id") or "")
            source_span = span_by_id.get(span_id)
            if source_span is None:
                return ""
            try:
                start_offset = int(item.get("start_offset", 0))
                end_offset = int(item.get("end_offset", len(source_span.text)))
            except (TypeError, ValueError):
                return ""
            if start_offset < 0 or end_offset > len(source_span.text) or start_offset >= end_offset:
                return ""
            source_text = source_span.text[start_offset:end_offset]
            if isinstance(item.get("text"), str) and item["text"] != source_text:
                return ""
            parts.append(source_text)
        return "".join(parts)
    span_id = str(cell.get("span_id") or cell.get("source_span_id") or "")
    source_span = span_by_id.get(span_id)
    if source_span is None:
        return ""
    try:
        start_offset = int(cell.get("start_offset", cell.get("offset_start", 0)))
        end_offset = int(cell.get("end_offset", cell.get("offset_end", len(source_span.text))))
    except (TypeError, ValueError):
        return ""
    if start_offset < 0 or end_offset > len(source_span.text) or start_offset >= end_offset:
        return ""
    return source_span.text[start_offset:end_offset]


def _agent_table_columns(table_plan: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = table_plan.get("columns", [])
    if isinstance(columns, list) and columns:
        normalized_columns = []
        for index, column in enumerate(columns, start=1):
            if isinstance(column, str) and column.strip():
                normalized_columns.append({"column_id": column.strip(), "name": column.strip()})
                continue
            if not isinstance(column, dict):
                continue
            normalized = dict(column)
            normalized.setdefault("column_id", f"col_{index:03d}")
            normalized.setdefault("name", normalized.get("column_name") or normalized.get("display_name") or normalized["column_id"])
            normalized_columns.append(normalized)
        return normalized_columns
    max_cell_count = max((len(row.get("cells", [])) for row in rows), default=0)
    return [{"column_id": f"col_{index + 1:03d}", "name": f"Column {index + 1}"} for index in range(max_cell_count)]


def _normalize_agent_table_cell_columns(rows: list[dict[str, Any]], columns: list[dict[str, Any]], table_type: str) -> None:
    column_ids = [str(column.get("column_id")) for column in columns if column.get("column_id")]
    if not column_ids:
        return
    for row in rows:
        cells = row.get("cells", [])
        if not isinstance(cells, list):
            continue
        normalized_cells: list[dict[str, Any]] = []
        cell_index_by_column: dict[str, int] = {}
        for index, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                continue
            normalized = dict(cell)
            column_id = _canonical_agent_table_column_id(str(cell.get("column_id") or cell.get("col_id") or ""), index, column_ids, table_type)
            normalized["column_id"] = column_id
            existing_index = cell_index_by_column.get(column_id)
            if existing_index is None:
                cell_index_by_column[column_id] = len(normalized_cells)
                normalized_cells.append(normalized)
            else:
                _merge_agent_table_cell(normalized_cells[existing_index], normalized)
        normalized_cells.sort(key=lambda cell: column_ids.index(str(cell.get("column_id"))) if str(cell.get("column_id")) in column_ids else len(column_ids))
        row["cells"] = normalized_cells


def _canonical_agent_table_column_id(raw_column_id: str, index: int, column_ids: list[str], table_type: str) -> str:
    if raw_column_id in column_ids:
        return raw_column_id
    normalized_key = _normalize_agent_table_column_key(raw_column_id)
    if table_type == "nutrition_facts":
        ordinal = _nutrition_column_ordinal(normalized_key, len(column_ids))
        if ordinal and ordinal <= len(column_ids):
            return column_ids[ordinal - 1]
    fallback = f"col_{index:03d}"
    if fallback in column_ids:
        return fallback
    return column_ids[min(index - 1, len(column_ids) - 1)]


def _normalize_agent_table_column_key(value: str) -> str:
    normalized = value.strip().lower()
    for old, new in (
        ("％", "%"),
        ("（", "("),
        ("）", ")"),
        ("-", "_"),
        (" ", "_"),
        ("/", "_"),
        ("(", "_"),
        (")", "_"),
    ):
        normalized = normalized.replace(old, new)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _nutrition_column_ordinal(key: str, column_count: int) -> int | None:
    if key in {
        "nutrition_item",
        "nutrient",
        "nutrient_name",
        "item",
        "name",
        "component",
        "成分",
        "营养成分",
        "项目",
    }:
        return 1
    if key in {
        "per_serving",
        "per_100g",
        "per_100ml",
        "per100g",
        "per100ml",
        "amount",
        "content",
        "value",
        "含量",
        "每100g含量",
        "每100ml含量",
        "每份含量",
    }:
        return 2
    if key in {"unit", "单位"}:
        return 3 if column_count >= 4 else 2
    if key in {
        "nrv",
        "nrv%",
        "nrv_percent",
        "nrv_percentage",
        "percentage_nrv",
        "nutrient_reference_value",
        "营养素参考值",
        "营养素参考值%",
    }:
        return 4 if column_count >= 4 else 3
    return None


def _merge_agent_table_cell(target: dict[str, Any], source: dict[str, Any]) -> None:
    target_text = str(target.get("raw_value") or "").strip()
    source_text = str(source.get("raw_value") or "").strip()
    if not target_text and source_text:
        target["raw_value"] = source.get("raw_value", "")
        target["normalized_value"] = source.get("normalized_value", source.get("raw_value", ""))
    target_refs = target.get("evidence_refs") if isinstance(target.get("evidence_refs"), list) else []
    for ref in source.get("evidence_refs", []) if isinstance(source.get("evidence_refs"), list) else []:
        if ref not in target_refs:
            target_refs.append(ref)
    target["evidence_refs"] = target_refs


def _pad_rows_to_columns(rows: list[dict[str, Any]], columns: list[dict[str, Any]]) -> None:
    column_ids = [str(column.get("column_id")) for column in columns if column.get("column_id")]
    for row in rows:
        cells = row.get("cells", [])
        if not isinstance(cells, list):
            continue
        present = {str(cell.get("column_id")) for cell in cells if isinstance(cell, dict)}
        row_evidence_refs = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), list) else []
        for column_id in column_ids:
            if column_id in present:
                continue
            cells.append(
                {
                    "column_id": column_id,
                    "raw_value": "",
                    "normalized_value": "",
                    "evidence_refs": row_evidence_refs,
                }
            )


def _linked_content_entity_from_names(title: str, content_names: dict[str, str]) -> str | None:
    for entity_id, name in content_names.items():
        if name and name in title:
            return entity_id
    return None


def _table_quality_with_agent_acceptance(table_quality_report: dict[str, Any], agent_tables: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_tables = [table for table in agent_tables if table.get("rows")]
    if not accepted_tables:
        return table_quality_report
    issues = [
        issue
        for issue in table_quality_report.get("issues", [])
        if issue.get("issue_type") not in {"nutrition_table_rows_incomplete", "table_count_conflict"}
    ]
    parser_agreement = dict(table_quality_report.get("parser_agreement", {}))
    parser_agreement["status"] = "agent_table_accepted"
    parser_agreement["agent_table_count"] = len(accepted_tables)
    updated = dict(table_quality_report)
    updated.update(
        {
            "status": "pass" if not issues else "review_required",
            "parser_agreement": parser_agreement,
            "agent_table_acceptance": {
                "status": "accepted",
                "table_count": len(accepted_tables),
                "source": "llm_table_agent",
            },
            "issues": issues,
            "issue_count": len(issues),
        }
    )
    return updated


def _json_export_manifest() -> dict[str, Any]:
    return {
        "schema_version": "mvp_final_json_v0.1",
        "media_type": "application/json",
        "encoding": "utf-8",
        "root_keys": FINAL_JSON_ROOT_KEYS,
        "contract_checks": JSON_EXPORT_CONTRACT_CHECKS,
        "no_guessing": True,
        "primary_artifact": "result.json",
        "contract_artifact": "output_contract_validation_report.json",
        "schema_artifact": "schemas/final_result.schema.json",
    }


def _write_layered_debug_artifacts(
    debug_dir: Path,
    result: ParseResult,
    perception: Any,
    ocr_lines: list[OcrLine],
    spans: list[TextSpan],
    vdg_nodes: list[VdgNode],
    visual_document_graph: dict[str, Any],
    plan: ExtractionPlan,
    audit_findings: list[dict[str, Any]],
) -> None:
    standard = result.metadata["standard_artifacts"]
    table_parser = result.metadata["table_parser"]
    repair = result.metadata["repair_loop"]

    write_json(debug_dir / "01_text_extraction" / "perception.json", to_jsonable(perception))
    write_json(debug_dir / "00_inputs" / "runtime_policy.json", to_jsonable(result.metadata["runtime_policy"]))
    write_json(debug_dir / "00_inputs" / "json_export.json", to_jsonable(result.metadata["json_export"]))
    write_json(debug_dir / "01_text_extraction" / "page_images.json", to_jsonable(result.metadata["page_images"]))
    write_json(debug_dir / "01_text_extraction" / "ocr_lines.json", to_jsonable(ocr_lines))
    write_json(debug_dir / "01_text_extraction" / "spans.json", to_jsonable(spans))
    write_json(debug_dir / "01_text_extraction" / "pdf_character_atoms.json", to_jsonable(result.metadata["pdf_character_atoms"]))
    write_json(debug_dir / "01_text_extraction" / "layout_candidates.json", to_jsonable(result.metadata["layout_candidates"]))
    write_json(debug_dir / "01_text_extraction" / "source_fusion_report.json", to_jsonable(result.metadata["source_fusion_report"]))
    write_json(debug_dir / "01_text_extraction" / "source_alignments.json", to_jsonable(result.metadata["source_alignments"]))
    write_json(
        debug_dir / "01_text_extraction" / "candidate_visual_document_graph.json",
        to_jsonable(result.metadata["candidate_visual_document_graph"]),
    )
    write_json(debug_dir / "01_text_extraction" / "vdg.json", to_jsonable(visual_document_graph))
    write_json(debug_dir / "01_text_extraction" / "visual_document_graph.json", to_jsonable(visual_document_graph))
    write_json(debug_dir / "01_text_extraction" / "vdg_nodes.json", to_jsonable(vdg_nodes))
    write_json(debug_dir / "01_text_extraction" / "source_layers.json", to_jsonable(result.metadata["source_layers"]))
    write_json(debug_dir / "01_text_extraction" / "source_consistency_report.json", to_jsonable(result.metadata["source_consistency"]))

    write_json(debug_dir / "01_table_extraction" / "table_layers.json", to_jsonable(table_parser["table_layers"]))
    write_json(debug_dir / "01_table_extraction" / "table_quality_report.json", to_jsonable(table_parser["table_quality_report"]))
    write_json(
        debug_dir / "01_table_extraction" / "table_parser_issues.json",
        to_jsonable(table_parser["table_layers"].get("parser_issues", [])),
    )

    write_json(debug_dir / "02_section_detection" / "sections.json", to_jsonable(result.extracted_data["sections"]))
    write_json(debug_dir / "02_section_detection" / "regions.json", to_jsonable(result.extracted_data["regions"]))
    write_json(debug_dir / "02_section_detection" / "revision_blocks.json", to_jsonable(result.extracted_data["revision_blocks"]))

    write_json(debug_dir / "03_field_structure" / "extraction_plan.json", to_jsonable(plan))
    write_json(debug_dir / "03_field_structure" / "schema_audit.json", to_jsonable(result.metadata["schema_audit"]))
    write_json(debug_dir / "03_field_structure" / "vdg_quality_report.json", to_jsonable(result.metadata["vdg_quality_report"]))
    write_json(debug_dir / "03_field_structure" / "vdg_agent_context.json", to_jsonable(result.metadata["vdg_agent_context"]))
    write_json(debug_dir / "03_field_structure" / "label_text_scope_reference.json", to_jsonable(result.metadata["label_text_scope_reference"]))
    write_json(debug_dir / "03_field_structure" / "label_text_scope_agent_context.json", to_jsonable(result.metadata["label_text_scope_agent_context"]))
    write_json(debug_dir / "03_field_structure" / "label_text_scope_report.json", to_jsonable(result.metadata["label_text_scope_report"]))
    write_json(debug_dir / "03_field_structure" / "standard_items.json", to_jsonable(standard["standard_items"]))
    write_result_preview_html(result, debug_dir / "03_field_structure" / "result_preview.html", artifact_root=debug_dir)
    write_json(debug_dir / "03_field_structure" / "comparison_index.json", to_jsonable(standard["comparison_index"]))
    write_json(debug_dir / "03_field_structure" / "field_groups.json", to_jsonable(standard["field_groups"]))
    write_json(debug_dir / "03_field_structure" / "tables.json", to_jsonable(standard["tables"]))
    write_json(debug_dir / "03_field_structure" / "lists.json", to_jsonable(standard["lists"]))
    write_json(debug_dir / "03_field_structure" / "auto_ingest_candidates.json", to_jsonable(standard["auto_ingest_candidates"]))
    write_json(debug_dir / "03_field_structure" / "extracted_data.json", to_jsonable(result.extracted_data))
    write_json(debug_dir / "03_field_structure" / "taxonomy_proposals.json", to_jsonable(standard["taxonomy_proposals"]))
    write_json(debug_dir / "03_field_structure" / "llm_agent_items.json", to_jsonable(result.metadata["agent_harness"]["llm_agent_items"]))
    write_json(debug_dir / "03_field_structure" / "rejected_agent_items.json", to_jsonable(result.metadata["agent_harness"]["rejected_agent_items"]))
    write_json(debug_dir / "03_field_structure" / "review_items.json", to_jsonable(result.metadata["agent_harness"]["review_items"]))
    write_json(debug_dir / "03_field_structure" / "agent_execution_report.json", to_jsonable(result.metadata["agent_execution_report"]))
    write_json(debug_dir / "03_field_structure" / "global_reconciliation_input.json", to_jsonable(result.metadata["global_reconciliation_input"]))
    write_json(debug_dir / "03_field_structure" / "global_reconciliation_output.json", to_jsonable(result.metadata["global_reconciliation_output"]))
    write_json(debug_dir / "03_field_structure" / "global_reconciliation_report.json", to_jsonable(result.metadata["global_reconciliation_report"]))
    write_json(
        debug_dir / "03_field_structure" / "table_feed_items.candidates.json",
        to_jsonable(table_parser["table_feed_items"]),
    )

    write_json(debug_dir / "04_validation" / "quality_report.json", to_jsonable(standard["quality_report"]))
    write_json(debug_dir / "04_validation" / "layout_quality_report.json", to_jsonable(result.metadata["layout_quality_report"]))
    write_json(debug_dir / "04_validation" / "json_export.json", to_jsonable(result.metadata["json_export"]))
    write_json(debug_dir / "04_validation" / "schema_audit.json", to_jsonable(result.metadata["schema_audit"]))
    write_json(debug_dir / "04_validation" / "auto_ingest_candidates.json", to_jsonable(standard["auto_ingest_candidates"]))
    write_json(debug_dir / "04_validation" / "missing_item_report.json", to_jsonable(result.metadata["missing_item_report"]))
    write_json(debug_dir / "04_validation" / "missing_fields.json", to_jsonable(result.metadata["missing_item_report"]["missing_fields"]))
    write_json(debug_dir / "04_validation" / "missing_tables.json", to_jsonable(result.metadata["missing_item_report"]["missing_tables"]))
    write_json(
        debug_dir / "04_validation" / "output_contract_validation_report.json",
        to_jsonable(result.metadata["output_contract_validation_report"]),
    )
    write_json(debug_dir / "04_validation" / "mvp_acceptance_metrics.json", to_jsonable(result.metadata["mvp_acceptance_metrics"]))
    write_json(debug_dir / "04_validation" / "source_anchor_inventory.json", to_jsonable(result.metadata["source_anchor_inventory"]))
    write_json(debug_dir / "04_validation" / "coverage_map.json", to_jsonable(result.metadata["coverage_map"]))
    write_json(debug_dir / "04_validation" / "vdg_consumption_report.json", to_jsonable(result.metadata["vdg_consumption_report"]))
    write_json(debug_dir / "04_validation" / "label_text_scope_reference.json", to_jsonable(result.metadata["label_text_scope_reference"]))
    write_json(debug_dir / "04_validation" / "label_text_scope_agent_context.json", to_jsonable(result.metadata["label_text_scope_agent_context"]))
    write_json(debug_dir / "04_validation" / "label_text_scope_report.json", to_jsonable(result.metadata["label_text_scope_report"]))
    write_json(debug_dir / "04_validation" / "structure_audit.json", to_jsonable(result.metadata["structure_audit"]))
    write_json(debug_dir / "04_validation" / "source_consistency_report.json", to_jsonable(result.metadata["source_consistency"]))
    write_json(debug_dir / "04_validation" / "evidence.json", to_jsonable(result.evidence))
    write_json(debug_dir / "04_validation" / "validation.json", to_jsonable(result.validation))
    write_json(debug_dir / "04_validation" / "risks.json", to_jsonable(result.risks))
    write_json(debug_dir / "04_validation" / "review_tasks.json", to_jsonable(result.review_tasks))
    write_json(debug_dir / "04_validation" / "audit_input.json", to_jsonable(result.metadata["audit_input"]))
    write_json(debug_dir / "04_validation" / "audit_findings.json", audit_findings)
    write_json(debug_dir / "04_validation" / "agent_execution_report.json", to_jsonable(result.metadata["agent_execution_report"]))

    write_json(debug_dir / "05_repair" / "repair_plan.json", to_jsonable(repair["repair_plan"]))
    write_json(debug_dir / "05_repair" / "repair_trace.json", to_jsonable(repair["trace"]))
    write_json(debug_dir / "05_repair" / "repair_attempts.json", to_jsonable(repair["attempts"]))
    write_json(debug_dir / "05_repair" / "repair_plan_patches.json", to_jsonable(repair["repair_plan_patches"]))
    write_json(debug_dir / "05_repair" / "repair_agent_candidates.json", to_jsonable(repair["repair_agent_candidates"]))
    write_json(debug_dir / "05_repair" / "repaired_source_layers.json", to_jsonable(repair["repaired_source_layers"]))


def _merge_spans(pdf_spans: list[TextSpan], ocr_lines: list[OcrLine]) -> list[TextSpan]:
    return build_source_fusion(pdf_spans, ocr_lines, enabled=True).canonical_spans


def _build_vdg(spans: list[TextSpan]) -> list[VdgNode]:
    nodes: list[VdgNode] = []
    for index, span in enumerate(spans, start=1):
        nodes.append(
            VdgNode(
                node_id=stable_id("node", index),
                node_type="text_span",
                page=span.page,
                text=span.text,
                source_span_ids=[span.span_id],
                bbox_pdf=span.bbox_pdf,
                bbox_normalized=span.bbox_normalized,
            )
        )
    return nodes


def _audit_input(compiled_fields: dict[str, Any], evidence: list[Any]) -> dict[str, Any]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    audit_fields: dict[str, Any] = {}
    for field_id, field in compiled_fields.items():
        refs = field.evidence_refs
        audit_fields[field_id] = {
            **to_jsonable(field),
            "has_bbox": all(evidence_by_id[ref].bbox_pdf is not None for ref in refs if ref in evidence_by_id),
        }
    return audit_fields


def _semantic_finding_score(findings: list[dict[str, Any]]) -> int:
    weights = {"high": 100, "medium": 10, "low": 1}
    return sum(weights.get(str(finding.get("severity")), 10) for finding in findings)


def _schema_consumption_findings(
    schema: GeneratedSchema,
    plan: ExtractionPlan,
    agent_blocks: dict[str, Any],
) -> list[dict[str, Any]]:
    consumed_by_key: dict[str, set[str]] = {}
    for field in plan.fields:
        consumed_by_key.setdefault(field.semantic_key, set()).update(
            span_range.span_id for span_range in field.value_source.ranges
        )
    block_by_span = {
        str(span_id): str(block.get("block_id") or "document")
        for block in agent_blocks.get("blocks", [])
        for span_id in block.get("source_span_ids", [])
    }
    findings: list[dict[str, Any]] = []
    for definition in schema.field_definitions:
        source_ids = list(dict.fromkeys(str(span_id) for span_id in definition.source_span_ids))
        missing_ids = [
            span_id
            for span_id in source_ids
            if span_id not in consumed_by_key.get(definition.semantic_key, set())
        ]
        if not missing_ids:
            continue
        block_id = block_by_span.get(missing_ids[0], "document")
        findings.append(
            {
                "issue_type": "schema_source_unconsumed",
                "block_id": block_id,
                "target_type": "schema_field",
                "target_id": definition.semantic_key,
                "source_span_ids": missing_ids[:40],
                "message": "Schema-supported source spans remain unconsumed by the reconciled extraction plan.",
                "severity": "high" if definition.criticality == "critical" else "medium",
                "repair_required": True,
            }
        )
    return findings


def _semantic_repair_spans(
    findings: list[dict[str, Any]],
    agent_blocks: dict[str, Any],
    spans: list[TextSpan],
    max_spans: int = 240,
) -> list[TextSpan]:
    cited_ids = {
        str(span_id)
        for finding in findings
        for span_id in finding.get("source_span_ids", [])
    }
    affected_blocks = {
        str(finding.get("block_id") or "")
        for finding in findings
        if finding.get("block_id")
    }
    context_ids = {
        str(span_id)
        for block in agent_blocks.get("blocks", [])
        if str(block.get("block_id") or "") in affected_blocks
        for span_id in block.get("context_span_ids", [])
    }
    prioritized_ids = [
        span.span_id
        for span in spans
        if span.span_id in cited_ids
    ]
    prioritized_ids.extend(
        span.span_id
        for span in spans
        if span.span_id in context_ids and span.span_id not in cited_ids
    )
    span_by_id = {span.span_id: span for span in spans}
    return [span_by_id[span_id] for span_id in prioritized_ids[:max_spans]]


def _agent_execution_report(
    *,
    schema_agent_name: str,
    extraction_agent_name: str,
    audit_agent_name: str,
    repair_agent_name: str,
    llm_agent_name: str | None,
    schema: GeneratedSchema,
    plan: ExtractionPlan,
    compiled_fields: dict[str, Any],
    evidence: list[Any],
    audit_findings: list[dict[str, Any]],
    schema_audit: dict[str, Any],
    repair_trace: dict[str, Any],
    agent_items_path: Path | None,
    use_llm_agent: bool,
    llm_agent_items: dict[str, Any] | None,
    llm_field_retry_items: dict[str, Any] | None,
    llm_table_retry_items: dict[str, Any] | None,
    llm_schema_items: dict[str, Any] | None,
    rule_field_count: int,
    agent_plan_field_count: int,
    agent_field_retry_count: int,
    agent_table_retry_count: int,
    rule_fallback_items: list[dict[str, Any]],
    rejected_agent_items: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
    layout_mode: str,
    global_reconciliation_report: dict[str, Any],
) -> dict[str, Any]:
    accepted_agent_item_count = agent_plan_field_count if use_llm_agent else max(len(plan.fields) - rule_field_count, 0)
    llm_candidate_count = _agent_plan_item_count(llm_agent_items) or agent_plan_field_count
    llm_field_retry_count = _agent_plan_item_count(llm_field_retry_items)
    llm_table_retry_count = _agent_table_plan_count(llm_table_retry_items)
    return {
        "report_version": "agent_execution_report_v0.1",
        "status": "pass",
        "architecture": "agent_led_semantic_planning_with_rule_validation_and_deterministic_compiler_gate",
        "agents": [
            {
                "agent_id": "schema_induction_agent",
                "implementation": llm_agent_name if use_llm_agent else schema_agent_name,
                "mode": "agent_led_with_rule_schema_audit" if use_llm_agent else "deterministic_rule_fallback",
                "responsibility": "Generate the dynamic schema from source spans; rules audit and supplement obvious anchors only.",
                "input_artifacts": ["source_layers.spans"],
                "output_artifacts": ["generated_schema", "schema_audit.json"],
                "output_counts": {
                    "field_definition_count": len(schema.field_definitions),
                    "table_definition_count": len(schema.table_definitions),
                    "requirement_definition_count": len(schema.requirement_definitions),
                    "schema_audit_issue_count": schema_audit.get("issue_count", 0),
                    "agent_schema_proposal_field_count": _agent_schema_field_count(llm_schema_items),
                    "agent_generated_field_definition_count": sum(
                        1 for definition in schema.field_definitions if definition.semantic_key_type == "agent_generated"
                    ),
                },
            },
            {
                "agent_id": "extraction_agent",
                "implementation": llm_agent_name if use_llm_agent else extraction_agent_name,
                "mode": (
                    "agent_led_extraction_with_rule_review_only"
                    if use_llm_agent and layout_mode == ENHANCED_LAYOUT_MODE
                    else "agent_led_extraction_with_rule_validation_fallback"
                    if use_llm_agent
                    else "deterministic_rule_fallback"
                ),
                "responsibility": (
                    "Create the evidence-bound extraction plan; rules validate evidence and route missing anchors to review without compiling fallback values."
                    if layout_mode == ENHANCED_LAYOUT_MODE
                    else "Create the evidence-bound extraction plan; rules validate spans, offsets, duplicates, and missing low-risk fallbacks."
                ),
                "input_artifacts": ["generated_schema", "source_layers.spans", "agent_items_path"],
                "output_artifacts": ["extraction_plan.json", "rejected_agent_items.json", "review_items.json"],
                "output_counts": {
                    "rule_field_count": rule_field_count,
                    "final_plan_field_count": len(plan.fields),
                    "agent_plan_field_count": agent_plan_field_count,
                    "agent_field_retry_count": agent_field_retry_count,
                    "rule_fallback_field_count": len(rule_fallback_items),
                    "accepted_agent_item_count": accepted_agent_item_count,
                    "rejected_agent_item_count": len(rejected_agent_items),
                    "review_item_count": len(review_items),
                },
            },
            {
                "agent_id": "global_reconciliation_agent",
                "implementation": llm_agent_name if use_llm_agent else "disabled",
                "mode": global_reconciliation_report.get("status", "disabled"),
                "responsibility": "Resolve duplicate observations and entity ownership inside the existing proposal evidence envelope.",
                "input_artifacts": ["global_reconciliation_input.json", "source_fusion_report.json"],
                "output_artifacts": ["global_reconciliation_output.json", "global_reconciliation_report.json"],
                "output_counts": {
                    "input_field_count": global_reconciliation_report.get("input_field_count", 0),
                    "output_field_count": global_reconciliation_report.get("output_field_count", 0),
                    "removed_field_count": global_reconciliation_report.get("removed_field_count", 0),
                },
            },
            {
                "agent_id": "group_agent",
                "implementation": llm_agent_name if use_llm_agent else extraction_agent_name,
                "mode": "agent_led_entity_grouping_with_rule_consistency_check" if use_llm_agent else "deterministic_rule_fallback",
                "responsibility": "Assign fields to product, manufacturer, content item, barcode, and requirement entities; rules validate consistency and duplicates.",
                "input_artifacts": ["generated_schema", "source_layers.spans", "extraction_plan.json"],
                "output_artifacts": ["extracted_data.entities", "field_groups.json"],
                "output_counts": {
                    "entity_plan_count": len(plan.entities),
                    "field_with_entity_count": sum(1 for field in plan.fields if field.entity_id),
                },
            },
            {
                "agent_id": "table_list_agent",
                "implementation": llm_agent_name if use_llm_agent else extraction_agent_name,
                "mode": "agent_led_table_list_semantics_with_parser_validation" if use_llm_agent else "deterministic_table_parser_fallback",
                "responsibility": "Recover table/list semantics from evidence-bound spans; parser output validates and provides fallback.",
                "input_artifacts": ["generated_schema", "source_layers.spans", "table_layers.json"],
                "output_artifacts": ["extracted_data.tables", "lists.json", "table_quality_report.json"],
                "output_counts": {
                    "table_plan_count": len(plan.tables),
                    "table_retry_plan_count": agent_table_retry_count,
                    "requirement_plan_count": len(plan.requirements),
                },
            },
            {
                "agent_id": "llm_field_agent",
                "implementation": llm_agent_name,
                "mode": "agent" if use_llm_agent else "disabled",
                "responsibility": "Generate schema and extraction-plan proposals without writing final JSON field values.",
                "input_artifacts": ["generated_schema", "source_layers.spans"],
                "output_artifacts": ["llm_agent_items.json", "generated_schema"],
                "required_env_vars": [],
                "runtime_managed_online_llm": False,
                "agent_items_path": str(agent_items_path) if agent_items_path else None,
                "output_counts": {
                    "proposal_field_count": llm_candidate_count,
                    "field_retry_proposal_count": llm_field_retry_count,
                    "field_retry_accepted_count": agent_field_retry_count,
                    "table_retry_proposal_count": llm_table_retry_count,
                    "table_retry_accepted_count": agent_table_retry_count,
                },
            },
            {
                "agent_id": "review_audit_agent",
                "implementation": audit_agent_name,
                "mode": "deterministic",
                "responsibility": "Review compiled evidence-bound fields independently from extraction plan creation.",
                "input_artifacts": ["audit_input.json", "compiled_fields", "evidence", "generated_schema", "coverage_map.json"],
                "output_artifacts": ["audit_findings.json"],
                "output_counts": {
                    "compiled_field_count": len(compiled_fields),
                    "evidence_count": len(evidence),
                    "audit_finding_count": len(audit_findings),
                },
            },
            {
                "agent_id": "repair_agent",
                "implementation": repair_agent_name,
                "mode": "execute_plan_repair_then_recompile",
                "responsibility": "Patch extraction plan boundaries when possible, then require compiler and validation to rerun.",
                "input_artifacts": ["audit_findings.json", "extraction_plan.json", "source_layers.spans", "review_items.json", "rejected_agent_items.json"],
                "output_artifacts": ["repair_plan.json", "repair_plan_patches.json", "repair_trace.json", "repair_attempts.json", "repair_agent_candidates.json"],
                "output_counts": {
                    "repair_round_count": repair_trace.get("round_count", 0),
                    "repair_attempt_count": repair_trace.get("attempt_count", 0),
                    "applied_attempt_count": repair_trace.get("applied_attempt_count", 0),
                    "final_audit_finding_count": repair_trace.get("final_audit_finding_count", 0),
                },
            },
        ],
        "separation_checks": [
            {
                "check_type": "extraction_and_review_are_independent_agents",
                "result": "passed" if extraction_agent_name != audit_agent_name else "failed",
                "details": {
                    "extraction_agent": extraction_agent_name,
                    "review_audit_agent": audit_agent_name,
                },
            },
            {
                "check_type": "review_runs_after_compiler",
                "result": "passed",
                "details": {
                    "review_input": "compiled_fields_with_evidence",
                    "extraction_output": "extraction_plan_ranges",
                },
            },
            {
                "check_type": "llm_does_not_write_final_json",
                "result": "passed",
                "details": {
                    "llm_agent_enabled": use_llm_agent,
                    "merge_gate": "span_grounded_plan_validation",
                    "final_value_writer": "DeterministicCompiler",
                },
            },
            {
                "check_type": "repair_requires_recompile",
                "result": "passed" if _repair_trace_recompiled_when_applied(repair_trace) else "failed",
                "details": {
                    "validation_after_repair": repair_trace.get("validation_after_repair"),
                    "applied_attempt_count": repair_trace.get("applied_attempt_count", 0),
                },
            },
        ],
    }


def _repair_trace_recompiled_when_applied(repair_trace: dict[str, Any]) -> bool:
    rounds = repair_trace.get("rounds", [])
    if not isinstance(rounds, list):
        return False
    applied_seen = False
    for round_record in rounds:
        if not isinstance(round_record, dict):
            continue
        attempts = round_record.get("attempts", [])
        if not isinstance(attempts, list):
            continue
        if any(isinstance(attempt, dict) and attempt.get("status") == "applied" for attempt in attempts):
            applied_seen = True
            if not round_record.get("compiled_after_repair"):
                return False
    return True if applied_seen else True


def _validation_checks(compiled_fields: dict[str, Any], evidence: list[Any], schema: GeneratedSchema) -> list[dict[str, Any]]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    evidence_ids = set(evidence_by_id)
    schema_defs = {definition.semantic_key: definition for definition in schema.field_definitions}
    checks: list[dict[str, Any]] = []
    for field_id, field in compiled_fields.items():
        schema_definition = schema_defs.get(field.semantic_key)
        schema_issues = _schema_validation_issues(field, schema_definition)
        checks.append(
            {
                "validation_id": stable_id("val", len(checks) + 1),
                "target_id": field_id,
                "check_type": "schema_validation",
                "result": "failed" if schema_issues else "passed",
                "semantic_key": field.semantic_key,
                "expected": "generated_schema field definition or approved dynamic field",
                "actual": {
                    "semantic_key": field.semantic_key,
                    "field_type": field.field_type,
                    "criticality": field.criticality,
                },
                "severity": "high" if field.criticality == "critical" else "medium",
                "message": "字段未通过 generated_schema 校验。" if schema_issues else "Schema validation passed.",
                "issues": schema_issues,
                "evidence_refs": field.evidence_refs,
            }
        )
        missing_refs = [ref for ref in field.evidence_refs if ref not in evidence_ids]
        checks.append(
            {
                "validation_id": stable_id("val", len(checks) + 1),
                "target_id": field_id,
                "check_type": "evidence_ref_integrity",
                "result": "failed" if missing_refs else "passed",
                "missing_refs": missing_refs,
            }
        )
        checks.append(
            {
                "validation_id": stable_id("val", len(checks) + 1),
                "target_id": field_id,
                "check_type": "no_guessing",
                "result": "passed" if field.raw_value else "failed",
            }
        )
        missing_bbox_refs = [
            ref
            for ref in field.evidence_refs
            if ref in evidence_by_id and evidence_by_id[ref].bbox_status == "missing"
        ]
        checks.append(
            {
                "validation_id": stable_id("val", len(checks) + 1),
                "target_id": field_id,
                "check_type": "bbox_integrity",
                "result": "failed" if missing_bbox_refs else "passed",
                "severity": "high" if field.criticality == "critical" else "medium",
                "message": "关键字段有值但没有 bbox。" if field.criticality == "critical" else "字段证据缺少 bbox。",
                "evidence_refs": missing_bbox_refs,
                "bbox_status": "missing" if missing_bbox_refs else "available",
                "risk_type": "critical_field_without_bbox" if field.criticality == "critical" else "field_without_bbox",
            }
        )
        for format_check in format_checks_for_field(field):
            checks.append(
                {
                    "validation_id": stable_id("val", len(checks) + 1),
                    **format_check,
                }
            )
    return checks


def _schema_validation_issues(field: Any, schema_definition: Any | None) -> list[dict[str, Any]]:
    if schema_definition is None:
        if str(field.semantic_key).startswith(("custom.", "proposed.")):
            return []
        return [{"reason": "semantic_key_not_in_generated_schema", "semantic_key": field.semantic_key}]

    issues = []
    if field.field_type != schema_definition.field_type:
        issues.append(
            {
                "reason": "field_type_mismatch",
                "expected": schema_definition.field_type,
                "actual": field.field_type,
            }
        )
    if field.criticality != schema_definition.criticality:
        issues.append(
            {
                "reason": "criticality_mismatch",
                "expected": schema_definition.criticality,
                "actual": field.criticality,
            }
        )
    return issues


def _schema_audit_validation_checks(schema_audit: dict[str, Any]) -> list[dict[str, Any]]:
    failed = schema_audit.get("status") == "review_required"
    issues = schema_audit.get("issues", [])
    return [
        {
            "validation_id": stable_id("val_schema_audit", 1),
            "target_id": "generated_schema",
            "check_type": "schema_audit",
            "result": "failed" if failed else "passed",
            "severity": _max_issue_severity(issues) if failed else "info",
            "message": "generated_schema 审计发现遗漏或结构问题。" if failed else "Schema audit passed.",
            "issue_count": schema_audit.get("issue_count", len(issues)),
            "blocking_issue_count": schema_audit.get("blocking_issue_count", 0),
            "issues": issues,
        }
    ]


def _source_consistency_validation_checks(source_consistency_report: dict[str, Any]) -> list[dict[str, Any]]:
    status = source_consistency_report.get("status")
    failed = status == "review_required"
    issues = source_consistency_report.get("issues", [])
    return [
        {
            "validation_id": stable_id("val_source_consistency", 1),
            "target_id": "source_consistency",
            "check_type": "multi_method_agreement",
            "result": "failed" if failed else "passed",
            "agreement": status,
            "severity": _max_issue_severity(issues) if failed else "info",
            "message": "PDF text 与 OCR 多通道一致性验证发现冲突。" if failed else "PDF text 与 OCR 多通道一致性验证通过或无 OCR 可比对。",
            "pdf_text_span_count": source_consistency_report.get("pdf_text_span_count", 0),
            "ocr_line_count": source_consistency_report.get("ocr_line_count", 0),
            "matched_ocr_line_count": source_consistency_report.get("matched_ocr_line_count", 0),
            "issue_count": source_consistency_report.get("issue_count", 0),
            "issues": issues,
        }
    ]


def _source_fusion_validation_checks(source_fusion_report: dict[str, Any], layout_mode: str) -> list[dict[str, Any]]:
    enhanced = layout_mode == ENHANCED_LAYOUT_MODE
    passed = (not enhanced) or (
        source_fusion_report.get("status") == "pass"
        and bool(source_fusion_report.get("enabled"))
        and int(source_fusion_report.get("canonical_span_count", 0))
        == int(source_fusion_report.get("pdf_span_count", 0))
        - int(source_fusion_report.get("superseded_adhesion_span_count", 0))
        + int(source_fusion_report.get("unmatched_ocr_line_count", 0))
    )
    return [
        {
            "validation_id": stable_id("val_source_fusion", 1),
            "check_type": "source_fusion",
            "target_id": "source_fusion",
            "result": "passed" if passed else "failed",
            "severity": "info" if passed else "high",
            "message": "PDF/OCR observations are represented by one canonical occurrence in enhanced layout mode.",
            "duplicate_span_count_prevented": source_fusion_report.get("duplicate_span_count_prevented", 0),
            "superseded_adhesion_span_count": source_fusion_report.get("superseded_adhesion_span_count", 0),
        }
    ]


def _internal_consistency_validation_checks(
    compiled_fields: dict[str, Any],
    revision_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    historical_field_ids = _historical_revision_field_ids(revision_blocks)
    grouped: dict[tuple[str, str], list[Any]] = {}
    for field in compiled_fields.values():
        if field.field_id in historical_field_ids or field.field_type == "requirement":
            continue
        key = (str(field.semantic_key), str(field.entity_id or "document"))
        grouped.setdefault(key, []).append(field)

    conflicts = []
    for (semantic_key, entity_id), fields in grouped.items():
        values: list[dict[str, Any]] = []
        for field in fields:
            normalized_value = str(field.normalized_value)
            matching_value = next(
                (
                    item
                    for item in values
                    if _field_values_equivalent(normalized_value, str(item["normalized_value"]))
                ),
                None,
            )
            if matching_value:
                matching_value["fields"].append(field)
                if len(normalized_value) > len(str(matching_value["normalized_value"])):
                    matching_value["normalized_value"] = normalized_value
                continue
            values.append({"normalized_value": normalized_value, "fields": [field]})
        if len(values) <= 1:
            continue
        conflict_fields = [field for value in values for field in value["fields"]]
        conflicts.append(
            {
                "semantic_key": semantic_key,
                "entity_id": entity_id,
                "field_ids": [field.field_id for field in conflict_fields],
                "values": [
                    {
                        "normalized_value": value["normalized_value"],
                        "field_ids": [field.field_id for field in value["fields"]],
                    }
                    for value in values
                ],
                "evidence_refs": _unique_refs(ref for field in conflict_fields for ref in field.evidence_refs),
            }
        )

    first_conflict = conflicts[0] if conflicts else None
    return [
        {
            "validation_id": stable_id("val_internal_consistency", 1),
            "target_id": first_conflict["field_ids"][0] if first_conflict else "internal_consistency",
            "check_type": "internal_consistency",
            "result": "failed" if conflicts else "passed",
            "severity": "medium" if conflicts else "info",
            "message": "同一实体内重复字段存在不一致值。" if conflicts else "Internal consistency check passed.",
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
            "evidence_refs": _unique_refs(ref for conflict in conflicts for ref in conflict.get("evidence_refs", [])),
        }
    ]


def _table_structure_validation_checks(
    tables: list[dict[str, Any]],
    table_quality_report: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for index, table in enumerate(tables, start=1):
        rows = table.get("rows", [])
        failed = bool(table.get("review_required")) or len(rows) == 0
        checks.append(
            {
                "validation_id": stable_id("val_table_structure", len(checks) + 1),
                "target_id": table.get("table_id", stable_id("table", index)),
                "check_type": "table_structure",
                "result": "failed" if failed else "passed",
                "severity": "high" if failed and table.get("criticality") == "critical" else "medium" if failed else "info",
                "message": "表格结构未恢复到可交付状态。" if failed else "Table structure validation passed.",
                "table_type": table.get("table_type"),
                "row_count": len(rows),
                "column_count": len(table.get("columns", [])),
                "evidence_refs": table.get("evidence_refs", []),
            }
        )

    quality_issues = table_quality_report.get("issues", [])
    agreement_status = table_quality_report.get("parser_agreement", {}).get("status")
    failed_quality = table_quality_report.get("status") == "review_required" or agreement_status == "table_count_conflict"
    checks.append(
        {
            "validation_id": stable_id("val_table_structure", len(checks) + 1),
            "target_id": "table_quality_report",
            "check_type": "table_structure",
            "result": "failed" if failed_quality else "passed",
            "severity": _max_issue_severity(quality_issues) if quality_issues else "medium" if failed_quality else "info",
            "message": "表格 parser 质量验证发现问题。" if failed_quality else "Table parser quality validation passed.",
            "table_quality_status": table_quality_report.get("status"),
            "parser_agreement": agreement_status,
            "issue_count": table_quality_report.get("issue_count", len(quality_issues)),
            "issues": quality_issues,
        }
    )
    return checks


def _field_values_equivalent(left: str, right: str) -> bool:
    left_normalized = _normalize_for_internal_consistency(left)
    right_normalized = _normalize_for_internal_consistency(right)
    if left_normalized == right_normalized:
        return True
    if min(len(left_normalized), len(right_normalized)) < 8:
        return False
    return left_normalized in right_normalized or right_normalized in left_normalized


def _normalize_for_internal_consistency(value: str) -> str:
    return "".join(str(value).split())


def _historical_revision_field_ids(revision_blocks: list[dict[str, Any]]) -> set[str]:
    return {
        str(field_ref.get("field_id"))
        for block in revision_blocks
        if block.get("revision_role") == "before"
        for field_ref in block.get("fields", [])
        if isinstance(field_ref, dict) and field_ref.get("field_id")
    }


def _unique_refs(refs: Any) -> list[str]:
    unique: list[str] = []
    for ref in refs:
        value = str(ref)
        if value not in unique:
            unique.append(value)
    return unique


def _max_issue_severity(issues: list[dict[str, Any]]) -> str:
    order = {"info": 0, "low": 1, "medium": 2, "high": 3}
    highest = "info"
    for issue in issues:
        severity = str(issue.get("severity", "info"))
        if order.get(severity, 0) > order[highest]:
            highest = severity
    return highest


def _risks_from_fields(compiled_fields: dict[str, Any], evidence: list[Any]) -> list[Risk]:
    risks: list[Risk] = []
    for field in compiled_fields.values():
        if field.review_required:
            risks.append(
                Risk(
                    risk_id=stable_id("risk", len(risks) + 1),
                    target_type="field",
                    target_id=field.field_id,
                    risk_level=field.risk_level,
                    risk_type="manual_review_required",
                    message=field.reason or "字段需要人工复核。",
                    evidence_refs=field.evidence_refs,
                )
            )
        if field.normalization:
            risks.append(
                Risk(
                    risk_id=stable_id("risk", len(risks) + 1),
                    target_type="field",
                    target_id=field.field_id,
                    risk_level="low",
                    risk_type="normalization_applied",
                    message="字段生成了 normalized_value，原文已保留。",
                    evidence_refs=field.evidence_refs,
                )
            )
    return risks


def _risks_from_audit(findings: list[dict[str, Any]]) -> list[Risk]:
    risks: list[Risk] = []
    for finding in findings:
        risks.append(
            Risk(
                risk_id=stable_id("audit_risk", len(risks) + 1),
                target_type=finding["target_type"],
                target_id=finding["target_id"],
                risk_level=finding["severity"],
                risk_type=finding["finding_type"],
                message=finding["message"],
                evidence_refs=finding.get("evidence_refs", []),
            )
        )
    return risks


def _risks_from_semantic_review(report: dict[str, Any]) -> list[Risk]:
    return [
        Risk(
            risk_id=stable_id("semantic_review_risk", index),
            target_type="document",
            target_id="document",
            risk_level=str(finding.get("severity") or "high"),
            risk_type=str(finding.get("issue_type") or "semantic_review_issue"),
            message=str(finding.get("message") or "Independent semantic review found an unresolved issue."),
        )
        for index, finding in enumerate(report.get("findings", []), start=1)
        if str(finding.get("severity") or "high") in {"high", "medium", "low"}
    ]


def _risks_from_schema_audit(schema_audit: dict[str, Any]) -> list[Risk]:
    risks: list[Risk] = []
    for issue in schema_audit.get("issues", []):
        severity = issue.get("severity", "medium")
        if severity not in {"high", "medium"}:
            continue
        risks.append(
            Risk(
                risk_id=stable_id("schema_audit_risk", len(risks) + 1),
                target_type="generated_schema",
                target_id="generated_schema",
                risk_level=severity,
                risk_type=issue.get("issue_type", "schema_audit_issue"),
                message=issue.get("message", "generated_schema 审计发现问题。"),
            )
        )
    return risks


def _risks_from_vdg_quality(vdg_quality_report: dict[str, Any]) -> list[Risk]:
    risks: list[Risk] = []
    for issue in vdg_quality_report.get("issues", []):
        severity = str(issue.get("severity", "medium"))
        if severity not in {"high", "medium"}:
            continue
        risks.append(
            Risk(
                risk_id=stable_id("vdg_quality_risk", len(risks) + 1),
                target_type="document",
                target_id="document",
                risk_level=severity,
                risk_type=issue.get("issue_type", "vdg_quality_issue"),
                message=issue.get("message", "VDG quality check found an issue."),
            )
        )
    return risks


def _risks_from_vdg_consumption(vdg_consumption_report: dict[str, Any]) -> list[Risk]:
    risks: list[Risk] = []
    if vdg_consumption_report.get("conflict_node_count", 0):
        risks.append(
            Risk(
                risk_id=stable_id("vdg_consumption_risk", len(risks) + 1),
                target_type="document",
                target_id="document",
                risk_level="high",
                risk_type="vdg_conflict_nodes",
                message="VDG consumption found conflict nodes.",
            )
        )
    if vdg_consumption_report.get("unknown_important_node_count", 0):
        risks.append(
            Risk(
                risk_id=stable_id("vdg_consumption_risk", len(risks) + 1),
                target_type="document",
                target_id="document",
                risk_level="high",
                risk_type="vdg_unknown_important_nodes",
                message="VDG consumption found important unknown nodes.",
            )
        )
    return risks


def _risks_from_validation(validation: list[dict[str, Any]]) -> list[Risk]:
    risks: list[Risk] = []
    for check in validation:
        if check.get("result") != "failed":
            continue
        if check.get("check_type") == "bbox_integrity":
            risk_type = check.get("risk_type") or (
                "critical_field_without_bbox" if check.get("severity") == "high" else "field_without_bbox"
            )
            risks.append(
                Risk(
                    risk_id=stable_id("validation_risk", len(risks) + 1),
                    target_type="field",
                    target_id=check.get("target_id", "unknown_field"),
                    risk_level=check.get("severity", "high"),
                    risk_type=risk_type,
                    message=check.get("message", "关键字段有值但没有 bbox。"),
                    evidence_refs=check.get("evidence_refs", []),
                )
            )
            continue
        if check.get("check_type") == "schema_validation":
            risks.append(
                Risk(
                    risk_id=stable_id("validation_risk", len(risks) + 1),
                    target_type="field",
                    target_id=check.get("target_id", "unknown_field"),
                    risk_level=check.get("severity", "medium"),
                    risk_type="schema_validation_failed",
                    message=check.get("message", "字段未通过 generated_schema 校验。"),
                    evidence_refs=check.get("evidence_refs", []),
                )
            )
            continue
        if check.get("check_type") == "multi_method_agreement":
            risks.append(
                Risk(
                    risk_id=stable_id("validation_risk", len(risks) + 1),
                    target_type="source_consistency",
                    target_id=check.get("target_id", "source_consistency"),
                    risk_level=check.get("severity", "medium"),
                    risk_type="multi_method_agreement_failed",
                    message=check.get("message", "多通道一致性验证失败。"),
                )
            )
            continue
        if check.get("check_type") == "internal_consistency":
            conflicts = check.get("conflicts", [])
            if conflicts:
                for conflict in conflicts:
                    field_ids = conflict.get("field_ids", [])
                    risks.append(
                        Risk(
                            risk_id=stable_id("validation_risk", len(risks) + 1),
                            target_type="field",
                            target_id=field_ids[0] if field_ids else check.get("target_id", "unknown_field"),
                            risk_level=check.get("severity", "medium"),
                            risk_type="field_internal_conflict",
                            message=check.get("message", "同一实体内重复字段存在不一致值。"),
                            evidence_refs=conflict.get("evidence_refs", []),
                        )
                    )
                continue
            risks.append(
                Risk(
                    risk_id=stable_id("validation_risk", len(risks) + 1),
                    target_type="field",
                    target_id=check.get("target_id", "unknown_field"),
                    risk_level=check.get("severity", "medium"),
                    risk_type="field_internal_conflict",
                    message=check.get("message", "同一实体内重复字段存在不一致值。"),
                    evidence_refs=check.get("evidence_refs", []),
                )
            )
            continue
        if check.get("check_type") == "table_structure":
            target_id = check.get("target_id", "table_quality_report")
            risks.append(
                Risk(
                    risk_id=stable_id("validation_risk", len(risks) + 1),
                    target_type="table_parser" if target_id == "table_quality_report" else "table",
                    target_id=target_id,
                    risk_level=check.get("severity", "medium"),
                    risk_type="table_structure_validation_failed",
                    message=check.get("message", "表格结构校验失败。"),
                    evidence_refs=check.get("evidence_refs", []),
                )
            )
            continue
        if check.get("check_type") in {
            "vdg_quality",
            "vdg_boundary_validation",
            "vdg_node_coverage",
            "vdg_table_cell_boundary",
            "vdg_region_boundary",
            "label_text_scope",
            "label_text_scope_reference",
            "label_text_scope_gate",
            "label_text_scope_unknown",
            "layout_quality",
            "layout_candidate_acceptance",
            "layout_boundary_validation",
        }:
            risks.append(
                Risk(
                    risk_id=stable_id("validation_risk", len(risks) + 1),
                    target_type="document",
                    target_id="document",
                    risk_level=check.get("severity", "medium"),
                    risk_type=f"{check.get('check_type')}_failed",
                    message=check.get("message", "Document-level validation failed."),
                )
            )
            continue
        if check.get("check_type") != "format_check":
            continue
        risks.append(
            Risk(
                risk_id=stable_id("validation_risk", len(risks) + 1),
                target_type="field",
                target_id=check.get("target_id", "unknown_field"),
                risk_level=check.get("severity", "medium"),
                risk_type="format_check_failed",
                message=check.get("message", "字段格式校验失败。"),
                evidence_refs=check.get("evidence_refs", []),
            )
        )
    return risks


def _risks_from_tables(tables: list[dict[str, Any]]) -> list[Risk]:
    risks: list[Risk] = []
    for table in tables:
        if table.get("review_required"):
            risks.append(
                Risk(
                    risk_id=stable_id("table_risk", len(risks) + 1),
                    target_type="table",
                    target_id=table["table_id"],
                    risk_level="high",
                    risk_type="table_structure_unrecovered",
                    message="营养成分表未恢复出稳定行结构。",
                    evidence_refs=table.get("evidence_refs", []),
                )
            )
    return risks


def _risks_from_table_quality_report(table_quality_report: dict[str, Any]) -> list[Risk]:
    risks: list[Risk] = []
    agreement_status = table_quality_report.get("parser_agreement", {}).get("status")
    if agreement_status == "table_count_conflict":
        risks.append(
            Risk(
                risk_id=stable_id("table_quality_risk", len(risks) + 1),
                target_type="table_parser",
                target_id="table_layers",
                risk_level="medium",
                risk_type="parser_agreement_conflict",
                message="text_span_nutrition 与 pdfplumber 的营养表候选数量不一致。",
            )
        )
    for issue in table_quality_report.get("issues", []):
        risks.append(
            Risk(
                risk_id=stable_id("table_quality_risk", len(risks) + 1),
                target_type="table",
                target_id=issue.get("table_layer_id", "unknown_table"),
                risk_level=issue.get("severity", "medium"),
                risk_type=issue.get("issue_type", "table_quality_issue"),
                message=issue.get("message", "表格质量检查发现问题。"),
            )
        )
    return risks


def _risks_from_source_layers(source_layers: dict[str, Any]) -> list[Risk]:
    risks: list[Risk] = []
    for issue in source_layers.get("source_issues", []):
        severity = issue.get("severity", "medium")
        if severity not in {"high", "medium"}:
            continue
        risks.append(
            Risk(
                risk_id=stable_id("source_layer_risk", len(risks) + 1),
                target_type="source_layer",
                target_id=issue.get("issue_id", "source_layers"),
                risk_level=severity,
                risk_type=issue.get("issue_type", "source_layer_issue"),
                message=issue.get("message", "源文本层质量检查发现问题。"),
            )
        )
    return risks


def _risks_from_source_consistency(source_consistency_report: dict[str, Any]) -> list[Risk]:
    risks: list[Risk] = []
    for issue in source_consistency_report.get("issues", []):
        severity = issue.get("severity", "low")
        if severity not in {"high", "medium", "low"}:
            continue
        risks.append(
            Risk(
                risk_id=stable_id("source_consistency_risk", len(risks) + 1),
                target_type="source_consistency",
                target_id=issue.get("issue_id", "source_consistency"),
                risk_level=severity,
                risk_type=issue.get("issue_type", "source_consistency_issue"),
                message=issue.get("message", "PDF text and OCR consistency check found an issue."),
            )
        )
    return risks


def _risks_from_page_images(page_images: dict[str, Any]) -> list[Risk]:
    status = page_images.get("status")
    if status in {"rendered", "not_rendered"}:
        return []
    return [
        Risk(
            risk_id="risk_page_image_render_failed",
            target_type="document",
            target_id="page_images",
            risk_level="medium",
            risk_type="page_image_render_failed",
            message=f"页面渲染未完成：{page_images.get('reason', status)}",
        )
    ]


def _risks_from_regions(regions: list[dict[str, Any]]) -> list[Risk]:
    risks: list[Risk] = []
    for region in regions:
        if region.get("region_type") != "package_panel":
            continue
        if region.get("assignment_status") != "uncertain":
            continue
        risks.append(
            Risk(
                risk_id=stable_id("region_risk", len(risks) + 1),
                target_type="region",
                target_id=region.get("region_id", "unknown_region"),
                risk_level="high",
                risk_type="panel_assignment_uncertain",
                message="已检测到唛面区域，但字段或表格归属尚未可靠绑定。",
                evidence_refs=region.get("evidence_refs", []),
            )
        )
    return risks


def _risks_from_revision_blocks(revision_blocks: list[dict[str, Any]]) -> list[Risk]:
    risks: list[Risk] = []
    for block in revision_blocks:
        if block.get("assignment_status") == "region_detected_field_assignment_pending":
            risks.append(
                Risk(
                    risk_id=stable_id("revision_risk", len(risks) + 1),
                    target_type="revision_block",
                    target_id=block.get("revision_block_id", f"revision_{block['revision_role']}"),
                    risk_level="high",
                    risk_type="revision_assignment_uncertain",
                    message="已检测到更改前/更改后区域，但字段归属尚未可靠区分。",
                    evidence_refs=block.get("evidence_refs", []),
                )
            )
    return risks


def _review_tasks_from_risks(risks: list[Risk]) -> list[ReviewTask]:
    tasks: list[ReviewTask] = []
    for risk in risks:
        if risk.risk_level != "high":
            continue
        tasks.append(
            ReviewTask(
                task_id=stable_id("review", len(tasks) + 1),
                target_type=risk.target_type,
                target_id=risk.target_id,
                risk_level=risk.risk_level,
                reason=risk.message,
                required=True,
                evidence_refs=risk.evidence_refs,
            )
        )
    return tasks


def _quality(compiled_fields: dict[str, Any], risks: list[Risk]) -> dict[str, Any]:
    high_count = sum(1 for risk in risks if risk.risk_level == "high")
    medium_count = sum(1 for risk in risks if risk.risk_level == "medium")
    low_count = sum(1 for risk in risks if risk.risk_level == "low")
    critical_fields = [field for field in compiled_fields.values() if field.criticality == "critical"]
    critical_pass = [
        field
        for field in critical_fields
        if not field.review_required and float(field.confidence.get("overall") or 0) >= 0.95
    ]
    return {
        "overall_status": "manual_review_required" if high_count else "pass_with_warnings" if medium_count or low_count else "pass",
        "critical_confidence_threshold": 0.95,
        "field_completion_rate": 1.0 if compiled_fields else 0.0,
        "critical_field_pass_rate": round(len(critical_pass) / len(critical_fields), 4) if critical_fields else 0.0,
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "low_risk_count": low_count,
        "auto_ingest_allowed": high_count == 0,
        "reason": "存在 high risk，需要人工复核。" if high_count else None,
    }


def _document_parse_status(risks: list[Risk], page_images: dict[str, Any]) -> str:
    if page_images.get("status") == "partial_failed":
        return "partial_failed"
    return "completed_with_warnings" if risks else "completed"


def _coverage(vdg_nodes: list[VdgNode], plan: ExtractionPlan, tables: list[dict[str, Any]]) -> dict[str, Any]:
    assigned_span_ids = {
        span_range.span_id
        for field in plan.fields
        for span_range in field.value_source.ranges
    }
    assigned_span_ids.update(
        source_span_id
        for table in tables
        for source_span_id in table.get("source_span_ids", [])
    )
    assigned_nodes = [
        node.node_id
        for node in vdg_nodes
        if any(span_id in assigned_span_ids for span_id in node.source_span_ids)
    ]
    total = len(vdg_nodes)
    return {
        "text_block_coverage_rate": round(len(assigned_nodes) / total, 4) if total else 0.0,
        "important_region_coverage_rate": round(len(assigned_nodes) / total, 4) if total else 0.0,
        "table_cell_coverage_rate": 0.0,
        "unknown_important_block_count": max(total - len(assigned_nodes), 0),
        "assigned_node_ids": assigned_nodes,
    }


def _coverage_from_vdg_consumption(report: dict[str, Any]) -> dict[str, Any]:
    coverage_rate = float(report.get("extracted_coverage_rate") or 0.0)
    return {
        "text_block_coverage_rate": coverage_rate,
        "important_region_coverage_rate": coverage_rate,
        "table_cell_coverage_rate": coverage_rate,
        "unknown_important_block_count": report.get("unknown_important_node_count", 0),
        "conflict_node_count": report.get("conflict_node_count", 0),
        "status_counts": report.get("status_counts", {}),
        "assigned_node_ids": report.get("extracted_node_ids", []),
    }


def _detected_document_types(regions: list[dict[str, Any]], tables: list[dict[str, Any]]) -> list[str]:
    detected = {"packaging_label_standard"}
    if any(region["region_type"] == "package_panel" for region in regions):
        detected.add("packaging_specification")
    if any(region["region_type"] in {"revision_before", "revision_after"} for region in regions):
        detected.add("packaging_design_technical_standard")
    if tables:
        detected.add("nutrition_table_document")
    return sorted(detected)
