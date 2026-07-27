from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .manifest import ManifestError
from .models import to_jsonable
from .pipeline import write_artifact_index
from .runner import run_manifest
from .utils import write_json


class HarnessError(RuntimeError):
    pass


def run_agent_harness(
    case_dir: Path,
    output_dir: Path,
    agent_items_path: Path | None = None,
    rule_only: bool = False,
) -> dict[str, Any]:
    manifest_path = case_dir / "manifest.json"
    expected_path = case_dir / "expected.json"
    if not manifest_path.exists():
        raise HarnessError(f"Missing case manifest: {manifest_path}")

    run_manifest_path = output_dir / "manifest.effective.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise HarnessError("Case manifest must be a JSON object.")
    if rule_only:
        manifest.pop("agent_items_path", None)
        manifest["use_llm_agent"] = False
        manifest["llm_mode"] = "disabled"
    elif agent_items_path is not None:
        manifest["agent_items_path"] = str(agent_items_path)
    write_json(run_manifest_path, manifest)

    result = run_manifest(run_manifest_path, output_dir / "run")
    expected = _load_expected(expected_path)
    actual_items = result.metadata["standard_artifacts"]["standard_items"]
    field_diff = _field_diff(expected.get("fields", []), actual_items)
    group_diff = _group_diff(expected.get("groups", []), result.metadata["standard_artifacts"]["field_groups"])
    table_diff = _table_diff(expected.get("tables", []), result.metadata["standard_artifacts"]["tables"])
    list_diff = _list_diff(expected.get("lists", []), result.metadata["standard_artifacts"]["lists"])
    table_layers_diff = _table_layers_diff(expected.get("table_layers", []), result.metadata["table_parser"]["table_layers"])
    taxonomy_diff = _taxonomy_diff(expected.get("taxonomy_proposals", []), result.metadata["standard_artifacts"]["taxonomy_proposals"])
    summary = _summary(
        result,
        field_diff,
        group_diff,
        table_diff,
        list_diff,
        table_layers_diff,
        taxonomy_diff,
        rule_only,
        agent_items_path,
    )

    _write_harness_outputs(output_dir, result, summary, field_diff, group_diff, table_diff, list_diff, table_layers_diff, taxonomy_diff)
    write_artifact_index(output_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline document parser harness for a benchmark case.")
    parser.add_argument("--case-dir", required=True, type=Path, help="Benchmark case directory containing manifest.json.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Harness output directory.")
    parser.add_argument("--agent-items-path", type=Path, help="Optional span-grounded agent candidate JSON.")
    parser.add_argument("--rule-only", action="store_true", help="Ignore any agent candidates and run rule-only mode.")
    args = parser.parse_args(argv)

    try:
        run_agent_harness(args.case_dir, args.output_dir, args.agent_items_path, args.rule_only)
    except (HarnessError, ManifestError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"run-agent-harness failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _load_expected(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HarnessError("expected.json must be a JSON object.")
    return data


def _field_diff(expected_fields: list[dict[str, Any]], actual_items: list[dict[str, Any]]) -> dict[str, Any]:
    actual_by_key = _items_by_key(actual_items)
    matched = []
    missing = []
    mismatched_text = []
    boundary_scores = []
    boundary_mismatches = []
    label_checks = []
    for expected in expected_fields:
        key = expected.get("semantic_key") or expected.get("field")
        candidates = actual_by_key.get(str(key), [])
        expected_text = expected.get("text")
        match = _first_text_match(candidates, expected_text)
        if match:
            match_record = {"expected": expected, "actual": match}
            boundary_score = _boundary_char_f1(expected, match)
            if boundary_score is not None:
                boundary_scores.append(boundary_score)
                match_record["boundary_char_f1"] = boundary_score
                if boundary_score < 1.0:
                    boundary_mismatches.append(match_record)
            label_check = _label_check(expected, match)
            if label_check is not None:
                label_checks.append(label_check)
                match_record["label_match"] = label_check
            matched.append(match_record)
        elif candidates:
            mismatched_text.append({"expected": expected, "actual_candidates": candidates})
        else:
            missing.append(expected)

    expected_keys = {str(item.get("semantic_key") or item.get("field")) for item in expected_fields}
    unexpected = [item for item in actual_items if item.get("semantic_key") not in expected_keys and item.get("field") not in expected_keys] if expected_fields else []
    field_recall = round(len(matched) / len(expected_fields), 4) if expected_fields else None
    field_precision = round(len(matched) / max(len(matched) + len(unexpected) + len(mismatched_text), 1), 4) if expected_fields else None
    boundary_char_f1 = round(sum(boundary_scores) / len(boundary_scores), 4) if boundary_scores else None
    label_accuracy = round(sum(1 for item in label_checks if item["matched"]) / len(label_checks), 4) if label_checks else None
    hallucinated = [item for item in actual_items if not item.get("sources") and not item.get("evidence_refs")]
    return {
        "expected_count": len(expected_fields),
        "actual_count": len(actual_items),
        "matched_count": len(matched),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched_text": mismatched_text,
        "field_recall": field_recall,
        "field_precision": field_precision,
        "boundary_char_f1": boundary_char_f1,
        "boundary_mismatches": boundary_mismatches,
        "label_accuracy": label_accuracy,
        "label_checks": label_checks,
        "hallucinated_items": hallucinated,
        "hallucination_count": len(hallucinated),
    }


def _group_diff(expected_groups: list[dict[str, Any]], actual_groups: list[dict[str, Any]]) -> dict[str, Any]:
    return _type_count_diff(expected_groups, actual_groups, "group_type")


def _table_diff(expected_tables: list[dict[str, Any]], actual_tables: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_type_count_diff(expected_tables, actual_tables, "table_type"),
        "row_count_checks": _count_checks(expected_tables, actual_tables, "table_type", "rows", "row_count", "min_rows"),
    }


def _list_diff(expected_lists: list[dict[str, Any]], actual_lists: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_type_count_diff(expected_lists, actual_lists, "list_type"),
        "item_count_checks": _count_checks(expected_lists, actual_lists, "list_type", "items", "item_count", "min_items"),
    }


def _table_layers_diff(expected_layers: list[dict[str, Any]], actual_layers: dict[str, Any]) -> dict[str, Any]:
    actual_tables = actual_layers.get("tables", [])
    return {
        **_type_count_diff(expected_layers, actual_tables, "table_type"),
        "parser_count": len(actual_layers.get("parsers", [])),
        "parser_issues": actual_layers.get("parser_issues", []),
    }


def _taxonomy_diff(expected_proposals: list[dict[str, Any]], actual_proposals: list[dict[str, Any]]) -> dict[str, Any]:
    return _type_count_diff(expected_proposals, actual_proposals, "field")


def _summary(
    result: Any,
    field_diff: dict[str, Any],
    group_diff: dict[str, Any],
    table_diff: dict[str, Any],
    list_diff: dict[str, Any],
    table_layers_diff: dict[str, Any],
    taxonomy_diff: dict[str, Any],
    rule_only: bool,
    agent_items_path: Path | None,
) -> dict[str, Any]:
    quality_report = result.metadata["standard_artifacts"]["quality_report"]
    output_contract = result.metadata["output_contract_validation_report"]
    runtime_policy = result.metadata["runtime_policy"]
    structure_audit = result.metadata["structure_audit"]
    table_quality = result.metadata["table_parser"]["table_quality_report"]
    harness_status = (
        "pass"
        if quality_report.get("status") == "pass"
        and output_contract.get("status") == "pass"
        and field_diff["hallucination_count"] == 0
        else "review_required"
    )
    return {
        "harness_status": harness_status,
        "rule_only": rule_only,
        "agent_items_path": str(agent_items_path) if agent_items_path else None,
        "quality_status": quality_report.get("status"),
        "output_contract_status": output_contract.get("status"),
        "output_contract_failed_count": output_contract.get("failed_count"),
        "runtime_policy_status": runtime_policy.get("status"),
        "runtime_policy_source": runtime_policy.get("source"),
        "downstream_allowed": quality_report.get("downstream_allowed"),
        "field_recall": field_diff.get("field_recall"),
        "field_precision": field_diff.get("field_precision"),
        "boundary_char_f1": field_diff.get("boundary_char_f1"),
        "label_accuracy": field_diff.get("label_accuracy"),
        "hallucination_count": field_diff["hallucination_count"],
        "anchor_coverage": structure_audit.get("anchor_coverage"),
        "missing_anchor_count": structure_audit.get("missing_anchor_count"),
        "sequence_gap_count": structure_audit.get("sequence_gap_count"),
        "group_issue_count": structure_audit.get("group_issue_count"),
        "table_issue_count": structure_audit.get("table_issue_count"),
        "required_prefix_issue_count": structure_audit.get("required_prefix_issue_count"),
        "container_duplicate_issue_count": structure_audit.get("container_duplicate_issue_count"),
        "agent_override_issue_count": structure_audit.get("agent_override_issue_count"),
        "duplicate_coverage_issue_count": structure_audit.get("duplicate_coverage_issue_count"),
        "table_quality_status": table_quality.get("status"),
        "parser_agreement": table_quality.get("parser_agreement", {}).get("status"),
        "risk_counts": quality_report.get("risk_counts", {}),
        "field_diff": {
            "expected_count": field_diff["expected_count"],
            "actual_count": field_diff["actual_count"],
            "matched_count": field_diff["matched_count"],
            "missing_count": len(field_diff["missing"]),
            "unexpected_count": len(field_diff["unexpected"]),
            "mismatched_text_count": len(field_diff["mismatched_text"]),
        },
        "group_diff": group_diff,
        "table_diff": table_diff,
        "list_diff": list_diff,
        "table_layers_diff": table_layers_diff,
        "taxonomy_diff": taxonomy_diff,
    }


def _write_harness_outputs(
    output_dir: Path,
    result: Any,
    summary: dict[str, Any],
    field_diff: dict[str, Any],
    group_diff: dict[str, Any],
    table_diff: dict[str, Any],
    list_diff: dict[str, Any],
    table_layers_diff: dict[str, Any],
    taxonomy_diff: dict[str, Any],
) -> None:
    run_dir = output_dir / "run"
    write_json(output_dir / "harness_summary.json", summary)
    write_json(output_dir / "field_diff.json", field_diff)
    write_json(output_dir / "group_diff.json", group_diff)
    write_json(output_dir / "table_diff.json", table_diff)
    write_json(output_dir / "list_diff.json", list_diff)
    write_json(output_dir / "table_layers_diff.json", table_layers_diff)
    write_json(output_dir / "taxonomy_diff.json", taxonomy_diff)
    write_json(output_dir / "accepted_items.json", to_jsonable(result.metadata["standard_artifacts"]["standard_items"]))
    write_json(output_dir / "comparison_index.json", to_jsonable(result.metadata["standard_artifacts"]["comparison_index"]))
    write_json(output_dir / "auto_ingest_candidates.json", to_jsonable(result.metadata["standard_artifacts"]["auto_ingest_candidates"]))
    write_json(output_dir / "rejected_agent_items.json", to_jsonable(result.metadata["agent_harness"]["rejected_agent_items"]))
    write_json(output_dir / "review_items.json", to_jsonable(result.metadata["agent_harness"]["review_items"]))
    write_json(output_dir / "extracted_data.json", to_jsonable(result.extracted_data))
    write_json(output_dir / "revision_blocks.json", to_jsonable(result.extracted_data.get("revision_blocks", [])))
    write_json(output_dir / "evidence.json", to_jsonable(result.evidence))
    write_json(output_dir / "validation.json", to_jsonable(result.validation))
    write_json(output_dir / "risks.json", to_jsonable(result.risks))
    write_json(output_dir / "review_tasks.json", to_jsonable(result.review_tasks))
    write_json(output_dir / "runtime_policy.json", to_jsonable(result.metadata["runtime_policy"]))
    write_json(output_dir / "audit_input.json", to_jsonable(result.metadata["audit_input"]))
    write_json(output_dir / "agent_execution_report.json", to_jsonable(result.metadata["agent_execution_report"]))
    write_json(output_dir / "missing_item_report.json", to_jsonable(result.metadata["missing_item_report"]))
    write_json(output_dir / "missing_fields.json", to_jsonable(result.metadata["missing_item_report"]["missing_fields"]))
    write_json(output_dir / "missing_tables.json", to_jsonable(result.metadata["missing_item_report"]["missing_tables"]))
    write_json(output_dir / "output_contract_validation_report.json", to_jsonable(result.metadata["output_contract_validation_report"]))
    write_json(output_dir / "mvp_acceptance_metrics.json", to_jsonable(result.metadata["mvp_acceptance_metrics"]))
    write_json(output_dir / "visual_document_graph.json", to_jsonable(result.metadata["visual_document_graph"]))
    write_json(output_dir / "schema_audit.json", to_jsonable(result.metadata["schema_audit"]))
    write_json(output_dir / "structure_audit.json", to_jsonable(result.metadata["structure_audit"]))
    write_json(output_dir / "table_layers.json", to_jsonable(result.metadata["table_parser"]["table_layers"]))
    write_json(output_dir / "table_quality_report.json", to_jsonable(result.metadata["table_parser"]["table_quality_report"]))
    write_json(output_dir / "repair_plan_patches.json", to_jsonable(result.metadata["repair_loop"]["repair_plan_patches"]))
    if (run_dir / "standard_items.json").exists():
        shutil.copyfile(run_dir / "standard_items.json", output_dir / "standard_items.json")
    if (run_dir / "quality_report.json").exists():
        shutil.copyfile(run_dir / "quality_report.json", output_dir / "quality_report.json")
    (output_dir / "score_report.md").write_text(_score_report(summary), encoding="utf-8")


def _items_by_key(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get("semantic_key")), []).append(item)
        grouped.setdefault(str(item.get("field")), []).append(item)
    return grouped


def _first_text_match(candidates: list[dict[str, Any]], expected_text: Any) -> dict[str, Any] | None:
    if expected_text is None:
        return candidates[0] if candidates else None
    expected_norm = _normalize(str(expected_text))
    for candidate in candidates:
        actual_norm = _normalize(str(candidate.get("text", "")))
        if actual_norm == expected_norm or expected_norm in actual_norm or actual_norm in expected_norm:
            return candidate
    return None


def _boundary_char_f1(expected: dict[str, Any], actual: dict[str, Any]) -> float | None:
    expected_range = _expected_char_range(expected)
    actual_range = _actual_char_range(actual)
    if expected_range is None or actual_range is None:
        return None
    expected_start, expected_end = expected_range
    actual_start, actual_end = actual_range
    expected_len = max(expected_end - expected_start, 0)
    actual_len = max(actual_end - actual_start, 0)
    if expected_len == 0 or actual_len == 0:
        return 0.0
    overlap = max(min(expected_end, actual_end) - max(expected_start, actual_start), 0)
    if overlap == 0:
        return 0.0
    precision = overlap / actual_len
    recall = overlap / expected_len
    return round((2 * precision * recall) / (precision + recall), 4)


def _expected_char_range(expected: dict[str, Any]) -> tuple[int, int] | None:
    source = expected.get("source")
    if isinstance(source, dict):
        start = source.get("char_start")
        end = source.get("char_end")
    else:
        start = expected.get("char_start")
        end = expected.get("char_end")
    if start is None or end is None:
        return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None


def _actual_char_range(actual: dict[str, Any]) -> tuple[int, int] | None:
    source = actual.get("source")
    if not isinstance(source, dict):
        return None
    start = source.get("char_start")
    end = source.get("char_end")
    if start is None or end is None:
        return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None


def _label_check(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any] | None:
    if "label" not in expected:
        return None
    expected_label = str(expected.get("label") or "")
    actual_label = str(actual.get("label") or "")
    return {
        "semantic_key": expected.get("semantic_key") or expected.get("field"),
        "expected_label": expected_label,
        "actual_label": actual_label,
        "matched": _normalize(expected_label) == _normalize(actual_label),
    }


def _type_count_diff(expected: list[dict[str, Any]], actual: list[dict[str, Any]], key: str) -> dict[str, Any]:
    expected_counts = _count_by_key(expected, key)
    actual_counts = _count_by_key(actual, key)
    matched_count = sum(min(expected_count, actual_counts.get(item_key, 0)) for item_key, expected_count in expected_counts.items())
    expected_total = sum(expected_counts.values())
    actual_total = sum(actual_counts.values())
    missing = {
        item_key: max(expected_count - actual_counts.get(item_key, 0), 0)
        for item_key, expected_count in expected_counts.items()
        if actual_counts.get(item_key, 0) < expected_count
    }
    unexpected = {
        item_key: max(actual_count - expected_counts.get(item_key, 0), 0)
        for item_key, actual_count in actual_counts.items()
        if expected_counts and actual_count > expected_counts.get(item_key, 0)
    }
    return {
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "expected_total": expected_total,
        "actual_total": actual_total,
        "matched_count": matched_count,
        "recall": round(matched_count / expected_total, 4) if expected_total else None,
        "precision": round(matched_count / actual_total, 4) if expected_total and actual_total else None,
        "missing": missing,
        "unexpected": unexpected,
    }


def _count_checks(
    expected_items: list[dict[str, Any]],
    actual_items: list[dict[str, Any]],
    type_key: str,
    actual_list_key: str,
    exact_key: str,
    min_key: str,
) -> list[dict[str, Any]]:
    actual_by_type: dict[str, list[dict[str, Any]]] = {}
    for item in actual_items:
        actual_by_type.setdefault(str(item.get(type_key, "unknown")), []).append(item)

    checks = []
    for expected in expected_items:
        item_type = str(expected.get(type_key, "unknown"))
        if exact_key not in expected and min_key not in expected:
            continue
        actual = actual_by_type.get(item_type, [{}])[0]
        actual_count = _actual_item_count(actual, actual_list_key)
        expected_exact = expected.get(exact_key)
        expected_min = expected.get(min_key)
        passed = True
        if expected_exact is not None:
            passed = actual_count == int(expected_exact)
        if expected_min is not None:
            passed = passed and actual_count >= int(expected_min)
        checks.append(
            {
                type_key: item_type,
                "actual_count": actual_count,
                exact_key: expected_exact,
                min_key: expected_min,
                "result": "passed" if passed else "failed",
            }
        )
    return checks


def _actual_item_count(actual: dict[str, Any], actual_list_key: str) -> int:
    if not actual:
        return 0
    if isinstance(actual.get(actual_list_key), list):
        return len(actual[actual_list_key])
    if isinstance(actual.get("item_count"), int):
        return int(actual["item_count"])
    return 0


def _count_by_key(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        item_key = str(item.get(key, "unknown"))
        counts[item_key] = counts.get(item_key, 0) + 1
    return counts


def _normalize(text: str) -> str:
    return "".join(text.split())


def _score_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Harness Score Report",
        "",
        f"- harness_status: {summary['harness_status']}",
        f"- quality_status: {summary['quality_status']}",
        f"- downstream_allowed: {summary['downstream_allowed']}",
        f"- field_recall: {summary['field_recall']}",
        f"- field_precision: {summary['field_precision']}",
        f"- boundary_char_f1: {summary['boundary_char_f1']}",
        f"- label_accuracy: {summary['label_accuracy']}",
        f"- hallucination_count: {summary['hallucination_count']}",
        f"- anchor_coverage: {summary['anchor_coverage']}",
        f"- missing_anchor_count: {summary['missing_anchor_count']}",
        f"- sequence_gap_count: {summary['sequence_gap_count']}",
        f"- group_issue_count: {summary['group_issue_count']}",
        f"- table_issue_count: {summary['table_issue_count']}",
        f"- required_prefix_issue_count: {summary['required_prefix_issue_count']}",
        f"- container_duplicate_issue_count: {summary['container_duplicate_issue_count']}",
        f"- agent_override_issue_count: {summary['agent_override_issue_count']}",
        f"- duplicate_coverage_issue_count: {summary['duplicate_coverage_issue_count']}",
        f"- table_quality_status: {summary['table_quality_status']}",
        f"- parser_agreement: {summary['parser_agreement']}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
