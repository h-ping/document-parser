from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkCase, BenchmarkError, BenchmarkManifest, BenchmarkThresholds, evaluate_benchmark, load_benchmark_manifest
from .benchmark_report import write_benchmark_report
from .config import ConfigError, RuntimeConfig
from .llm import OpenAICompatibleLlmClient
from .llm_agents import SpanGroundedFieldAgent
from .models import to_jsonable
from .ocr import GLMOcrClient, RecordedOcrClient
from .pipeline import DocumentParser, ParseError
from .runtime_policy import assert_runtime_policy_passed, build_runtime_policy, default_runtime_options
from .schema_artifacts import write_schema_artifacts
from .standard_xlsx import StandardXlsxParser
from .utils import write_json


REQUIRED_DEBUG_ARTIFACTS = [
    "result.json",
    "standard_items.json",
    "quality_report.json",
    "comparison_index.json",
    "output_contract_validation_report.json",
    "mvp_acceptance_metrics.json",
    "candidate_visual_document_graph.json",
    "vdg_quality_report.json",
    "vdg_agent_context.json",
    "vdg_consumption_report.json",
    "label_text_scope_reference.json",
    "label_text_scope_agent_context.json",
    "label_text_scope_report.json",
    "pdf_character_atoms.json",
    "layout_candidates.json",
    "layout_quality_report.json",
    "agent_blocks.json",
    "agent_block_inputs.json",
    "agent_block_results.json",
    "semantic_review_input.json",
    "semantic_review_report.json",
    "semantic_repair_rounds.json",
    "artifacts/index.json",
    "03_field_structure/comparison_index.json",
    "03_field_structure/label_text_scope_report.json",
    "04_validation/output_contract_validation_report.json",
    "04_validation/mvp_acceptance_metrics.json",
    "04_validation/label_text_scope_report.json",
]


class PdfRegressionError(RuntimeError):
    pass


def run_pdf_regression(
    *,
    input_dir: Path,
    output_dir: Path,
    ocr_fixture: Path | None,
    pattern: str = "*.pdf",
    patterns: list[str] | None = None,
    use_llm_agent: bool = False,
    layout_mode: str = "legacy",
    benchmark_manifest: Path | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    selected_patterns = patterns or [pattern]
    pdfs = sorted({pdf for item in selected_patterns for pdf in input_dir.glob(item)})
    if not pdfs:
        raise PdfRegressionError(f"No PDF files matched {selected_patterns!r} under {input_dir}")

    manifest = load_benchmark_manifest(benchmark_manifest) if benchmark_manifest else None
    output_dir.mkdir(parents=True, exist_ok=True)
    case_reports = [
        _run_case(pdf, output_dir, ocr_fixture, clean, use_llm_agent, layout_mode, manifest.match(pdf.name) if manifest else None, manifest is not None)
        for pdf in pdfs
    ]
    failed_cases = [case for case in case_reports if case["status"] != "pass"]
    summary = {
        "regression_version": "pdf_regression_v0.1",
        "status": "pass" if not failed_cases else "failed",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "ocr_mode": "recorded_fixture" if ocr_fixture else "cloud",
        "ocr_fixture": str(ocr_fixture) if ocr_fixture else None,
        "llm_agent_enabled": use_llm_agent,
        "layout_mode": layout_mode,
        "benchmark_manifest": str(benchmark_manifest) if benchmark_manifest else None,
        "benchmark_enabled": manifest is not None,
        "patterns": selected_patterns,
        "case_count": len(case_reports),
        "failed_count": len(failed_cases),
        "cases": case_reports,
    }
    write_json(output_dir / "regression_summary.json", summary)
    if manifest:
        write_json(
            output_dir / "benchmark_summary.json",
            {
                "benchmark_summary_version": "benchmark_summary_v0.1",
                "status": summary["status"],
                "case_count": len(case_reports),
                "failed_count": len(failed_cases),
                "cases": [case.get("benchmark", {"case_id": case["case_id"], "status": "failed"}) for case in case_reports],
            },
        )
    write_schema_artifacts(output_dir)
    return summary


def validate_regression_payload(payload: dict[str, Any], output_path: Path, debug_dir: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not output_path.exists():
        issues.append({"check": "output_json_exists", "path": str(output_path)})
    else:
        try:
            json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append({"check": "output_json_parseable", "path": str(output_path), "reason": str(exc)})

    job_status = _get(payload, "job", "status")
    if job_status not in {"completed", "completed_with_warnings"}:
        issues.append({"check": "job_completed", "actual": job_status})

    contract = _as_dict(_get(payload, "metadata", "output_contract_validation_report"))
    if contract.get("status") != "pass":
        issues.append(
            {
                "check": "output_contract_passed",
                "actual": contract.get("status"),
                "failed_count": contract.get("failed_count"),
            }
        )

    standard = _as_dict(_get(payload, "metadata", "standard_artifacts"))
    standard_items = _as_list(standard.get("standard_items"))
    if not standard_items:
        issues.append({"check": "standard_items_non_empty"})
    comparison_index = _as_dict(standard.get("comparison_index"))
    entries = _as_list(comparison_index.get("entries"))
    skipped = _as_list(comparison_index.get("skipped_items"))
    if comparison_index.get("entry_count") != len(entries):
        issues.append(
            {
                "check": "comparison_index_entry_count",
                "actual": comparison_index.get("entry_count"),
                "expected": len(entries),
            }
        )
    if len(entries) + len(skipped) != len(standard_items):
        issues.append(
            {
                "check": "comparison_index_covers_standard_items",
                "entry_count": len(entries),
                "skipped_count": len(skipped),
                "standard_item_count": len(standard_items),
            }
        )

    metrics = _as_dict(_get(payload, "metadata", "mvp_acceptance_metrics"))
    if metrics.get("metrics_version") != "mvp_acceptance_metrics_v0.1":
        issues.append({"check": "mvp_acceptance_metrics_present", "actual": metrics.get("metrics_version")})
    if _get(metrics, "risk", "output_contract_status") != "pass":
        issues.append({"check": "mvp_metrics_contract_status", "actual": _get(metrics, "risk", "output_contract_status")})

    vdg_quality = _as_dict(_get(payload, "metadata", "vdg_quality_report"))
    if vdg_quality.get("status") not in {"pass", "review_required"}:
        issues.append({"check": "vdg_quality_not_failed", "actual": vdg_quality.get("status")})
    if vdg_quality.get("source_span_coverage_rate") != 1.0:
        issues.append({"check": "vdg_source_span_coverage_full", "actual": vdg_quality.get("source_span_coverage_rate")})
    if vdg_quality.get("edge_ref_status") != "pass":
        issues.append({"check": "vdg_edge_refs_resolve", "actual": vdg_quality.get("edge_ref_status")})

    label_text_scope = _as_dict(_get(payload, "metadata", "label_text_scope_report"))
    if label_text_scope.get("status") not in {"pass", "review_required"}:
        issues.append({"check": "label_text_scope_not_failed", "actual": label_text_scope.get("status")})
    if label_text_scope.get("extracted_out_of_scope_count") != 0:
        issues.append({"check": "label_text_scope_extracted_out_of_scope_zero", "actual": label_text_scope.get("extracted_out_of_scope_count")})

    layout_quality = _as_dict(_get(payload, "metadata", "layout_quality_report"))
    if layout_quality.get("status") not in {"disabled", "pass", "review_required"}:
        issues.append({"check": "layout_quality_not_failed", "actual": layout_quality.get("status")})

    for artifact in REQUIRED_DEBUG_ARTIFACTS:
        path = debug_dir / artifact
        if not path.exists():
            issues.append({"check": "debug_artifact_exists", "path": artifact})
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline regression over packaging-label PDF samples.")
    parser.add_argument("--input-dir", type=Path, default=Path("test-documents"), help="Directory containing PDF samples.")
    parser.add_argument("--output-dir", type=Path, default=Path("out/regression"), help="Regression output directory.")
    parser.add_argument("--ocr-fixture", type=Path, help="Recorded OCR response fixture for offline regression.")
    parser.add_argument("--pattern", default="*.pdf", help="Glob pattern for selecting sample PDFs.")
    parser.add_argument("--patterns", nargs="+", help="Glob patterns for selecting multiple sample PDFs.")
    parser.add_argument("--use-llm-agent", action="store_true", help="Run regression through the configured online LLM agent path.")
    parser.add_argument("--layout-mode", choices=["legacy", "char_atoms_high_recall"], default="legacy", help="PDF layout evidence mode.")
    parser.add_argument("--benchmark-manifest", type=Path, help="External XLSX benchmark manifest used only by regression scoring.")
    parser.add_argument("--clean", action="store_true", help="Remove each case's previous output JSON and debug directory before running.")
    args = parser.parse_args(argv)

    try:
        summary = run_pdf_regression(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            ocr_fixture=args.ocr_fixture,
            pattern=args.pattern,
            patterns=args.patterns,
            use_llm_agent=args.use_llm_agent,
            layout_mode=args.layout_mode,
            benchmark_manifest=args.benchmark_manifest,
            clean=args.clean,
        )
    except (BenchmarkError, ConfigError, ParseError, PdfRegressionError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"run-pdf-regression failed: {exc}", file=sys.stderr)
        return 1
    return 0 if summary["status"] == "pass" else 1


def _run_case(
    pdf: Path,
    output_dir: Path,
    ocr_fixture: Path | None,
    clean: bool,
    use_llm_agent: bool,
    layout_mode: str,
    benchmark_case: BenchmarkCase | None,
    benchmark_required: bool,
) -> dict[str, Any]:
    output_path = output_dir / f"{pdf.stem}.json"
    debug_dir = output_dir / f"{pdf.stem}-debug"
    if clean:
        if output_path.exists():
            output_path.unlink()
        if debug_dir.exists():
            shutil.rmtree(debug_dir)

    try:
        payload = _parse_pdf(pdf, output_path, debug_dir, ocr_fixture, use_llm_agent, layout_mode)
        issues = validate_regression_payload(payload, output_path, debug_dir)
        benchmark_evaluation = None
        if benchmark_required and benchmark_case is None:
            issues.append({"check": "benchmark_case_matched", "actual": pdf.name})
        elif benchmark_case is not None:
            reference = to_jsonable(StandardXlsxParser().parse(benchmark_case.reference_xlsx))
            benchmark_evaluation, benchmark_diff = evaluate_benchmark(
                case_id=benchmark_case.case_id,
                current=payload,
                reference=reference,
                thresholds=BenchmarkThresholds(),
            )
            write_json(debug_dir / "benchmark_evaluation.json", benchmark_evaluation)
            write_json(debug_dir / "benchmark_diff.json", benchmark_diff)
            write_json(
                debug_dir / "benchmark_summary.json",
                {
                    "benchmark_summary_version": "benchmark_summary_v0.1",
                    "case_id": benchmark_case.case_id,
                    "status": benchmark_evaluation["status"],
                    "checks": benchmark_evaluation["checks"],
                },
            )
            write_benchmark_report(debug_dir / "benchmark_diff_report.html", benchmark_evaluation, benchmark_diff)
            if benchmark_evaluation["status"] != "pass":
                issues.append({"check": "benchmark_thresholds_passed", "actual": benchmark_evaluation["status"], "checks": benchmark_evaluation["checks"]})
        standard = _as_dict(_get(payload, "metadata", "standard_artifacts"))
        contract = _as_dict(_get(payload, "metadata", "output_contract_validation_report"))
        quality = _as_dict(standard.get("quality_report"))
        comparison_index = _as_dict(standard.get("comparison_index"))
        vdg_quality = _as_dict(_get(payload, "metadata", "vdg_quality_report"))
        vdg_consumption = _as_dict(_get(payload, "metadata", "vdg_consumption_report"))
        label_text_scope = _as_dict(_get(payload, "metadata", "label_text_scope_report"))
        layout_quality = _as_dict(_get(payload, "metadata", "layout_quality_report"))
        source_fusion = _as_dict(_get(payload, "metadata", "source_fusion_report"))
        reconciliation = _as_dict(_get(payload, "metadata", "global_reconciliation_report"))
        report = {
            "case_id": pdf.stem,
            "input_pdf": str(pdf),
            "output_json": str(output_path),
            "debug_dir": str(debug_dir),
            "status": "pass" if not issues else "failed",
            "issue_count": len(issues),
            "issues": issues,
            "job_status": _get(payload, "job", "status"),
            "parse_status": _get(payload, "document", "parse_status"),
            "output_contract_status": contract.get("status"),
            "output_contract_failed_count": contract.get("failed_count"),
            "quality_status": quality.get("status"),
            "downstream_allowed": quality.get("downstream_allowed"),
            "standard_item_count": len(_as_list(standard.get("standard_items"))),
            "comparison_entry_count": comparison_index.get("entry_count"),
            "vdg_quality_status": vdg_quality.get("status"),
            "vdg_source_span_coverage_rate": vdg_quality.get("source_span_coverage_rate"),
            "vdg_unknown_important_node_count": vdg_consumption.get("unknown_important_node_count"),
            "vdg_conflict_node_count": vdg_consumption.get("conflict_node_count"),
            "vdg_boundary_issue_count": vdg_quality.get("boundary_issue_count"),
            "nutrition_table_candidate_status": vdg_quality.get("nutrition_table_candidate_status"),
            "nutrition_table_row_count": vdg_quality.get("nutrition_table_row_count"),
            "nutrition_table_cell_count": vdg_quality.get("nutrition_table_cell_count"),
            "label_text_scope_status": label_text_scope.get("status"),
            "extracted_out_of_scope_count": label_text_scope.get("extracted_out_of_scope_count"),
            "ignored_noise_node_count": label_text_scope.get("ignored_noise_node_count"),
            "unknown_scope_node_count": label_text_scope.get("unknown_scope_node_count"),
            "scope_gate_rejected_count": label_text_scope.get("scope_gate_rejected_count"),
            "layout_mode": layout_quality.get("mode", layout_mode),
            "layout_quality_status": layout_quality.get("status"),
            "pdf_character_atom_count": layout_quality.get("pdf_character_atom_count", 0),
            "dropped_control_char_count": layout_quality.get("dropped_control_char_count", 0),
            "layout_candidate_count": layout_quality.get("layout_candidate_count", 0),
            "nutrition_layout_candidate_count": layout_quality.get("nutrition_layout_candidate_count", 0),
            "producer_layout_candidate_count": layout_quality.get("producer_layout_candidate_count", 0),
            "layout_boundary_issue_count": layout_quality.get("layout_boundary_issue_count", 0),
            "cross_page_candidate_count": layout_quality.get("cross_page_candidate_count", 0),
            "layout_fallback_used": layout_quality.get("fallback_used", False),
            "source_fusion_status": source_fusion.get("status"),
            "canonical_span_count": source_fusion.get("canonical_span_count", 0),
            "aligned_ocr_line_count": source_fusion.get("aligned_ocr_line_count", 0),
            "duplicate_span_count_prevented": source_fusion.get("duplicate_span_count_prevented", 0),
            "superseded_adhesion_span_count": source_fusion.get("superseded_adhesion_span_count", 0),
            "global_reconciliation_status": reconciliation.get("status"),
            "global_reconciliation_input_field_count": reconciliation.get("input_field_count", 0),
            "global_reconciliation_output_field_count": reconciliation.get("output_field_count", 0),
            "global_reconciliation_removed_field_count": reconciliation.get("removed_field_count", 0),
            "global_reconciliation_repair_applied": reconciliation.get("repair_applied", False),
        }
        if benchmark_evaluation is not None:
            report["benchmark"] = benchmark_evaluation
        return report
    except (ConfigError, ParseError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        return {
            "case_id": pdf.stem,
            "input_pdf": str(pdf),
            "output_json": str(output_path),
            "debug_dir": str(debug_dir),
            "status": "failed",
            "issue_count": 1,
            "issues": [{"check": "case_parse_failed", "reason": str(exc), "error_type": exc.__class__.__name__}],
        }


def _parse_pdf(pdf: Path, output_path: Path, debug_dir: Path, ocr_fixture: Path | None, use_llm_agent: bool, layout_mode: str) -> dict[str, Any]:
    required_env_vars: list[str] = []
    if ocr_fixture is None:
        required_env_vars.append("GLM_OCR_API_KEY")
    if use_llm_agent:
        required_env_vars.extend(["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"])
    config = RuntimeConfig.from_env(require_secrets=bool(required_env_vars), required_env_vars=required_env_vars)
    runtime_options = default_runtime_options()
    runtime_options["layout_mode"] = layout_mode
    runtime_policy = build_runtime_policy(
        source="pdf_regression",
        options=runtime_options,
        config=config,
        required_env_vars=required_env_vars,
        ocr_fixture_path=ocr_fixture,
        agent_items_path=None,
        use_llm_agent=use_llm_agent,
        max_repair_rounds=2,
    )
    assert_runtime_policy_passed(runtime_policy)
    ocr_client = RecordedOcrClient(ocr_fixture) if ocr_fixture else GLMOcrClient(config)
    llm_agent = SpanGroundedFieldAgent(OpenAICompatibleLlmClient(config)) if use_llm_agent else None
    result = DocumentParser(ocr_client=ocr_client, llm_agent=llm_agent).parse(
        pdf,
        debug_dir=debug_dir,
        use_llm_agent=use_llm_agent,
        runtime_policy=runtime_policy,
    )
    payload = to_jsonable(result)
    write_json(output_path, payload)
    write_schema_artifacts(output_path.parent)
    return payload


def _get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
