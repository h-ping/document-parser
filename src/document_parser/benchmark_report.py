from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_benchmark_report(path: Path, evaluation: dict[str, Any], diff: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(evaluation, diff), encoding="utf-8")


def _render(evaluation: dict[str, Any], diff: dict[str, Any]) -> str:
    metrics = [
        ("Status", evaluation.get("status")),
        ("Field exact", f'{evaluation.get("field_exact_count", 0)}/{evaluation.get("field_expected_count", 0)}'),
        ("Critical recall", _percent(evaluation.get("critical_field_exact_recall"))),
        ("Entity recall", _percent(evaluation.get("repeated_entity_group_recall"))),
        ("Nutrition tables", _percent(evaluation.get("nutrition_table_recall"))),
        ("Nutrition rows", _percent(evaluation.get("nutrition_row_value_accuracy"))),
        ("Cell boundary", _percent(evaluation.get("nutrition_cell_boundary_conformance"))),
        ("High risk", evaluation.get("unresolved_high_risk_count", 0)),
    ]
    metric_html = "".join(f'<div class="metric"><span>{_e(label)}</span><strong>{_e(value)}</strong></div>' for label, value in metrics)
    field_rows = "".join(_field_row(item) for item in diff.get("field_matches", []))
    missing_rows = "".join(_missing_row(item) for item in diff.get("missing_fields", []))
    unexpected_rows = "".join(_unexpected_row(item) for item in diff.get("unexpected_fields", []))
    entity_rows = "".join(_entity_row(item) for item in diff.get("entity_groups", []))
    table_rows = "".join(_table_row(item) for item in diff.get("table_matches", []))
    diagnostics = json.dumps(diff.get("diagnostics", {}), ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark Diff · {_e(evaluation.get('case_id'))}</title>
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#17191d;--muted:#6b7280;--line:#dfe3e8;--good:#18794e;--bad:#b42318;--near:#9a6700}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1480px;margin:auto;padding:24px}}h1{{font-size:24px;margin:0 0 16px}}h2{{font-size:16px;margin:0;padding:14px 16px;border-bottom:1px solid var(--line)}}
.metrics{{display:grid;grid-template-columns:repeat(8,minmax(110px,1fr));gap:8px;margin-bottom:16px}}.metric{{background:var(--panel);border:1px solid var(--line);padding:12px;border-radius:6px}}
.metric span{{display:block;color:var(--muted);font-size:12px}}.metric strong{{font-size:18px}}section{{background:var(--panel);border:1px solid var(--line);border-radius:6px;margin:12px 0;overflow:hidden}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}}th,td{{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left;overflow-wrap:anywhere}}th{{color:var(--muted);font-size:12px;background:#fafbfc}}
.exact{{color:var(--good)}}.near{{color:var(--near)}}.mismatch,.missing,.detail_mismatch{{color:var(--bad)}}code,pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{color:var(--muted);padding:12px}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}main{{padding:12px}}section{{overflow-x:auto}}table{{min-width:900px}}}}
</style></head><body><main>
<h1>Benchmark Diff · {_e(evaluation.get('case_id'))}</h1><div class="metrics">{metric_html}</div>
<section><h2>Field comparison</h2><table><thead><tr><th style="width:14%">Status</th><th style="width:20%">Field</th><th>Expected</th><th>Current</th><th style="width:18%">Evidence / bbox</th></tr></thead><tbody>{field_rows}{missing_rows}{unexpected_rows}</tbody></table></section>
<section><h2>Repeated entities</h2><table><thead><tr><th>Status</th><th>Role</th><th>Expected name</th><th>Current name</th><th>Detail mismatches</th></tr></thead><tbody>{entity_rows}</tbody></table></section>
<section><h2>Nutrition tables</h2><table><thead><tr><th>Expected</th><th>Current</th><th>Rows</th><th>Boundary</th><th>Issues</th></tr></thead><tbody>{table_rows}</tbody></table></section>
<section><h2>Unknown, conflict and repair history</h2><details open><summary>Pipeline diagnostics</summary><pre><code>{_e(diagnostics)}</code></pre></details></section>
</main></body></html>"""


def _field_row(item: dict[str, Any]) -> str:
    expected = item.get("expected", {})
    current = item.get("current", {})
    evidence = current.get("benchmark_evidence", [])
    evidence_text = json.dumps(
        [{"page": value.get("page"), "bbox": value.get("bbox_pdf") or value.get("bbox_normalized")} for value in evidence],
        ensure_ascii=False,
    )
    return "<tr>" + "".join(
        (
            f'<td class="{_e(item.get("status"))}">{_e(item.get("status"))}<br>{_e(item.get("similarity"))}</td>',
            f'<td>{_e(item.get("semantic_key"))}</td>',
            f'<td><code>{_e(_value(expected))}</code></td>',
            f'<td><code>{_e(_value(current))}</code></td>',
            f'<td><details><summary>{len(evidence)} refs</summary><code>{_e(evidence_text)}</code></details></td>',
        )
    ) + "</tr>"


def _missing_row(item: dict[str, Any]) -> str:
    return f'<tr><td class="missing">missing</td><td>{_e(item.get("semantic_key"))}</td><td><code>{_e(_value(item))}</code></td><td>—</td><td>—</td></tr>'


def _unexpected_row(item: dict[str, Any]) -> str:
    evidence = item.get("benchmark_evidence", [])
    evidence_text = json.dumps(
        [{"page": value.get("page"), "bbox": value.get("bbox_pdf") or value.get("bbox_normalized")} for value in evidence],
        ensure_ascii=False,
    )
    return f'<tr><td class="mismatch">unexpected</td><td>{_e(item.get("semantic_key"))}</td><td>—</td><td><code>{_e(_value(item))}</code></td><td><details><summary>{len(evidence)} refs</summary><code>{_e(evidence_text)}</code></details></td></tr>'


def _table_row(item: dict[str, Any]) -> str:
    issues = json.dumps({"boundary": item.get("boundary_issues", []), "rows": item.get("row_diffs", []), "bbox_pdf": item.get("current_bbox_pdf")}, ensure_ascii=False)
    boundary = "pass" if item.get("boundary_conforms") else "failed"
    return f'<tr><td>{_e(item.get("expected_title") or item.get("expected_table_id"))}</td><td>{_e(item.get("current_title") or item.get("current_table_id"))}</td><td>{_e(item.get("exact_row_count"))}/{_e(item.get("expected_row_count"))}</td><td class="{_e(boundary)}">{_e(boundary)}</td><td><details><summary>rows / bbox</summary><code>{_e(issues)}</code></details></td></tr>'


def _entity_row(item: dict[str, Any]) -> str:
    expected = item.get("expected") or {}
    current = item.get("current") or {}
    mismatches = json.dumps(item.get("detail_mismatches", []), ensure_ascii=False)
    status = str(item.get("status") or "missing")
    return f'<tr><td class="{_e(status)}">{_e(status)}</td><td>{_e(expected.get("role"))}</td><td><code>{_e(expected.get("name"))}</code></td><td><code>{_e(current.get("name", ""))}</code></td><td><code>{_e(mismatches)}</code></td></tr>'


def _value(field: dict[str, Any]) -> str:
    return str(field.get("normalized_value") or field.get("clean_value") or field.get("raw_value") or field.get("value") or "")


def _percent(value: Any) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))
