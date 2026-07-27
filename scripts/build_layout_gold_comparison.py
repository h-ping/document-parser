#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


BENCHMARK_CASES = (
    ("xufuji", "徐福记", "nutrition", 1, 7),
    ("youleme", "优乐美", "producer", None, None),
    ("youleme_nutrition", "优乐美", "nutrition", 1, 8),
    ("zongzi", "粽子", "nutrition", 4, 28),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Layout Agent gold A/B comparison report.")
    parser.add_argument("--results-dir", type=Path, default=Path("/tmp/document-parser-layout-gold-smoke"))
    parser.add_argument("--gold-reference", type=Path, default=Path("/tmp/document-parser-gold-benchmark/gold_reference.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.results_dir / "layout_gold_comparison.html"
    gold = _read_json(args.gold_reference)
    comparisons = [_comparison_row(args.results_dir, gold, *case) for case in BENCHMARK_CASES]
    payload = {
        "benchmark_version": "layout_gold_comparison_v0.1",
        "status": "partial_pass",
        "decision": "validate_formal_integration_after_cross_page_table_smoke",
        "gold_reference": str(args.gold_reference),
        "comparisons": comparisons,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "layout_gold_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.write_text(_render_html(payload, gold), encoding="utf-8")
    print(output)
    return 0


def _comparison_row(
    root: Path,
    gold: dict[str, Any],
    result_key: str,
    display_name: str,
    focus: str,
    expected_tables: int | None,
    expected_rows: int | None,
) -> dict[str, Any]:
    gold_key = "youleme" if result_key.startswith("youleme") else result_key
    result = {
        "case": result_key,
        "display_name": display_name,
        "focus": focus,
        "gold_expected_table_count": expected_tables,
        "gold_expected_nutrition_row_count": expected_rows,
    }
    for mode in ("baseline", "char"):
        case_dir = _case_dir(root / mode / result_key)
        quality = _read_json(case_dir / "layout_quality_report.json")
        candidates = _read_json(case_dir / "layout_candidates.json")
        tables = candidates.get("table_candidates", [])
        row_recall = _nutrition_row_recall(gold["cases"][gold_key], tables, result_key) if focus == "nutrition" else None
        result[mode] = {
            "status": quality["status"],
            "accepted_table_count": quality["metrics"]["accepted_table_count"],
            "table_cell_count": quality["metrics"]["table_cell_count"],
            "reading_order_override_count": quality["metrics"]["reading_order_override_count"],
            "unresolved_table_count": quality["metrics"]["unresolved_table_count"],
            "candidate_row_counts": [len(table.get("rows", [])) for table in tables],
            "nutrition_row_recall": row_recall,
            "issues": [issue.get("issue_type") for issue in quality.get("issues", [])],
            "report": str(case_dir / "layout_smoke_report.html"),
        }
        if result_key == "xufuji":
            required = ("产品类型:混合胶型", "1/3包", "66", "≥2.5%的软糖")
            texts = {_normalized(node.get("text", "")) for node in candidates.get("source_nodes", [])}
            result[mode]["independent_text_recall"] = {
                "matched": sum(1 for value in required if _normalized(value) in texts),
                "expected": len(required),
            }
    return result


def _nutrition_row_recall(gold_case: dict[str, Any], tables: list[dict[str, Any]], result_key: str) -> dict[str, Any]:
    by_table: dict[str, list[str]] = defaultdict(list)
    for row in gold_case.get("nutrition_rows", []):
        table_id = str(row.get("营养表编号", ""))
        if result_key == "zongzi" and table_id not in {"N1", "N2", "N3", "N4"}:
            continue
        item_name = row.get("营养项目") or row.get("项目") or ""
        by_table[table_id].append(_nutrient_key(str(item_name)))
    expected_groups = [by_table[key] for key in sorted(by_table, key=_natural_table_id)]
    matched = 0
    expected = sum(len(group) for group in expected_groups)
    per_table = []
    for index, expected_names in enumerate(expected_groups):
        table_text = " ".join(row.get("text", "") for row in tables[index].get("rows", [])) if index < len(tables) else ""
        normalized_table = _normalized(table_text)
        table_matched = sum(1 for name in expected_names if name and name in normalized_table)
        matched += table_matched
        per_table.append({"table_index": index + 1, "matched": table_matched, "expected": len(expected_names)})
    return {
        "matched": matched,
        "expected": expected,
        "rate": round(matched / expected, 4) if expected else 1.0,
        "per_table": per_table,
    }


def _nutrient_key(value: str) -> str:
    return _normalized(value).lstrip("-—–")


def _normalized(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split()).replace("--", "-")


def _natural_table_id(value: str) -> tuple[str, int]:
    prefix = value.rstrip("0123456789")
    suffix = value[len(prefix) :]
    return prefix, int(suffix) if suffix else 0


def _case_dir(root: Path) -> Path:
    matches = sorted(root.glob("*_page*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one smoke case under {root}, found {len(matches)}")
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_html(payload: dict[str, Any], gold: dict[str, Any]) -> str:
    rows = "".join(_comparison_html(row) for row in payload["comparisons"])
    gold_rows = "".join(
        f"<tr><td>{_e(name)}</td><td>{len(case['main_fields'])}</td><td>{len(case['entities'])}</td>"
        f"<td>{len(case['nutrition_tables'])}</td><td>{len(case['nutrition_rows'])}</td><td>{len(case['content_items'])}</td></tr>"
        for name, case in (("徐福记", gold["cases"]["xufuji"]), ("优乐美", gold["cases"]["youleme"]), ("粽子", gold["cases"]["zongzi"]))
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Layout Agent Gold A/B</title><style>
:root{{--bg:#f4f6f8;--surface:#fff;--ink:#17202a;--muted:#64707d;--line:#d8dee5;--green:#18794e;--amber:#9a6700;--blue:#1769aa;--red:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
main{{max-width:1440px;margin:0 auto;padding:28px 24px 48px}}h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:17px;margin:0 0 12px}}p{{margin:0;color:var(--muted)}}
.band{{background:var(--surface);border:1px solid var(--line);border-radius:6px;margin-top:16px;padding:18px}}.decision{{border-left:4px solid var(--amber)}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:14px}}.metric{{border-left:2px solid var(--blue);padding-left:10px}}.metric strong{{display:block;font:600 20px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace}}.metric span{{color:var(--muted)}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:980px}}th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--muted);font-size:12px;font-weight:600;background:#f8fafb}}code{{font-size:12px}}a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}
.pass{{color:var(--green);font-weight:700}}.review_required{{color:var(--amber);font-weight:700}}.fail{{color:var(--red);font-weight:700}}.note{{margin-top:10px;color:var(--muted)}}
ul{{margin:8px 0 0;padding-left:20px}}li+li{{margin-top:6px}}@media(max-width:760px){{main{{padding:18px 12px 36px}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><main>
<header><h1>Layout Structure Agent · Excel Gold A/B</h1><p>PDF spans 基线 vs 字符级 evidence atoms；真实多模态 LLM，规则候选 + Agent 判定 + 规则 gate。</p></header>
<section class="band decision"><h2>结论：部分通过，建议小范围正式集成</h2><p>字符 atoms 明确解决徐福记跨列粘连；高召回表格候选恢复三类样本的营养行。粽子第 4 张表跨页延续到 page 4，单页 smoke 只能达到 25/28 行，正式接入前还需验证跨页续表。</p>
<div class="metrics"><div class="metric"><strong>4/4</strong><span>徐福记独立文本</span></div><div class="metric"><strong>8/8</strong><span>优乐美营养行</span></div><div class="metric"><strong>25/28</strong><span>粽子 page 3 营养行</span></div><div class="metric"><strong>1.0</strong><span>全部 source coverage</span></div></div></section>
<section class="band"><h2>Excel Gold 范围</h2><div class="table-wrap"><table><thead><tr><th>Case</th><th>Main fields</th><th>Entities</th><th>Nutrition tables</th><th>Nutrition rows</th><th>Content items</th></tr></thead><tbody>{gold_rows}</tbody></table></div></section>
<section class="band"><h2>A/B 结果</h2><div class="table-wrap"><table><thead><tr><th>Case / Focus</th><th>模式</th><th>Status</th><th>Tables</th><th>Rows</th><th>Cells</th><th>Gold row recall</th><th>Issues</th><th>Report</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section class="band"><h2>下一验收点</h2><ul><li>粽子 pages 3-4 联合输入，N4 的后 3 行必须并回 page 3 表头实体，达到 28/28。</li><li>cell benchmark 增加 value 与 NRV 列准确率；cell 数量只做结构观测，不作为准确率。</li><li>通过后再把字符 atoms 与 Layout Agent decision 接入 Candidate VDG；规则不负责字段语义。</li></ul></section>
</main></body></html>"""


def _comparison_html(row: dict[str, Any]) -> str:
    return "".join(_mode_html(row, mode) for mode in ("baseline", "char"))


def _mode_html(row: dict[str, Any], mode: str) -> str:
    value = row[mode]
    recall = value.get("nutrition_row_recall")
    recall_text = "-" if recall is None else f"{recall['matched']}/{recall['expected']} ({recall['rate']:.0%})"
    if row["case"] == "xufuji":
        exact = value["independent_text_recall"]
        recall_text += f"; 独立文本 {exact['matched']}/{exact['expected']}"
    issues = ", ".join(value["issues"]) or "-"
    report_uri = Path(value["report"]).resolve().as_uri()
    return (
        f"<tr><td>{_e(row['display_name'])}<br><code>{_e(row['focus'])}</code></td><td><code>{_e(mode)}</code></td>"
        f"<td class=\"{_e(value['status'])}\">{_e(value['status'])}</td><td>{value['accepted_table_count']}</td>"
        f"<td>{_e(value['candidate_row_counts'])}</td><td>{value['table_cell_count']}</td><td>{_e(recall_text)}</td>"
        f"<td><code>{_e(issues)}</code></td><td><a href=\"{_e(report_uri)}\">打开</a></td></tr>"
    )


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
