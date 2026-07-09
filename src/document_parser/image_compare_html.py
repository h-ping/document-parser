from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any


def write_image_compare_html(
    output_path: Path,
    *,
    image_path: Path,
    image_width: int,
    image_height: int,
    package_ocr_lines: list[dict[str, Any]],
    comparison_result: dict[str, Any],
    standard_targets: list[dict[str, Any]] | None = None,
    unmatched_print_text: dict[str, Any] | None = None,
    ocr_quality_report: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_image_compare_html(
            output_path=output_path,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            package_ocr_lines=package_ocr_lines,
            comparison_result=comparison_result,
            standard_targets=standard_targets,
            unmatched_print_text=unmatched_print_text,
            ocr_quality_report=ocr_quality_report,
        ),
        encoding="utf-8",
    )


def build_image_compare_html(
    *,
    output_path: Path,
    image_path: Path,
    image_width: int,
    image_height: int,
    package_ocr_lines: list[dict[str, Any]],
    comparison_result: dict[str, Any],
    standard_targets: list[dict[str, Any]] | None = None,
    unmatched_print_text: dict[str, Any] | None = None,
    ocr_quality_report: dict[str, Any] | None = None,
) -> str:
    results = [item for item in _as_list(comparison_result.get("results")) if isinstance(item, dict)]
    extra_items = _extra_text_items(unmatched_print_text)
    standard_items = _standard_doc_items(results, standard_targets)
    anchors = {
        str(item.get("target_id")): _as_dict(item.get("selected_candidate")).get("source_ocr_line_ids", [])
        for item in results
    }
    anchors.update(_extra_anchors(extra_items))
    image_src = _relative_path(output_path.parent, image_path)
    boxes = "".join(_ocr_box(line) for line in package_ocr_lines)
    standard_doc = _standard_doc_panel(standard_items)
    cards = _grouped_result_cards(results, extra_items)
    summary = _summary(comparison_result)
    quality = _quality_banner(ocr_quality_report)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>包装图一致性对比</title>",
            "<style>",
            _CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>包装图一致性对比</h1>",
            summary,
            quality,
            '<section class="compare-shell">',
            '<div class="pane standard-pane">',
            '<div class="pane-header"><h2>标准文档</h2><span>点击右侧结果定位标准项</span></div>',
            '<div class="standard-doc">',
            standard_doc,
            "</div>",
            "</div>",
            '<div class="pane image-pane">',
            '<div class="pane-header"><h2>包装图</h2><span>点击右侧结果高亮识别位置</span></div>',
            f'<div class="image-wrap" style="aspect-ratio:{image_width} / {image_height}">',
            f'<img src="{_attr(image_src)}" alt="包装图">',
            boxes,
            "</div>",
            "</div>",
            '<div class="pane result-pane">',
            '<div class="pane-header"><h2>一致性结果</h2><span id="filter-label">全部结果</span></div>',
            '<div class="result-list">',
            cards,
            '<div class="empty-state" hidden>没有匹配当前筛选条件的列表项。</div>',
            "</div>",
            "</div>",
            "</section>",
            "</main>",
            "<script>",
            f"const TARGET_ANCHORS = {json.dumps(anchors, ensure_ascii=False)};",
            _SCRIPT,
            "</script>",
            "</body>",
            "</html>",
        ]
    )


def _summary(result: dict[str, Any]) -> str:
    rows = [
        ("检查结论", _overall_status_label(result.get("status")), "all"),
        ("检查项", result.get("target_count"), "all"),
        ("通过", result.get("pass_count"), "pass"),
        ("不一致/缺失", result.get("critical_count"), "critical"),
        ("需人工复核", result.get("manual_review_count"), "manual_review"),
        ("包装图多出文字", result.get("info_extra_text_count"), "info_extra_text"),
    ]
    return (
        '<section class="summary" aria-label="结果筛选">'
        + "".join(
            '<button type="button" class="summary-card" '
            f'data-filter-status="{_attr(filter_status)}" '
            f'aria-label="筛选{_attr(label)}">'
            f"<span>{_html(label)}</span><strong>{_html(value)}</strong>"
            "</button>"
            for label, value, filter_status in rows
        )
        + "</section>"
    )


def _quality_banner(report: dict[str, Any] | None) -> str:
    report = _as_dict(report)
    if not report:
        return ""
    status = str(report.get("bbox_overlay_status") or "unknown")
    if status == "aligned":
        return (
            '<section class="quality-banner ok">'
            "文字识别位置已与包装图对齐。"
            "</section>"
        )
    return (
        '<section class="quality-banner warn">'
        "包装图位置标记可能不准确，请以文字结果为准。"
        "</section>"
    )


def _ocr_box(line: dict[str, Any]) -> str:
    bbox = _as_dict(line.get("bbox_normalized"))
    if not bbox:
        return ""
    x1 = float(bbox.get("x1") or 0)
    y1 = float(bbox.get("y1") or 0)
    x2 = float(bbox.get("x2") or 0)
    y2 = float(bbox.get("y2") or 0)
    return (
        '<button class="ocr-box" type="button" '
        f'data-line-id="{_attr(line.get("ocr_line_id"))}" '
        f'title="{_attr(line.get("text"))}" '
        f'style="left:{x1 * 100:.4f}%;top:{y1 * 100:.4f}%;width:{(x2 - x1) * 100:.4f}%;height:{(y2 - y1) * 100:.4f}%;">'
        "</button>"
    )


def _standard_doc_items(results: list[dict[str, Any]], standard_targets: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    result_by_target = {str(item.get("target_id")): item for item in results if item.get("target_id")}
    if standard_targets:
        items = []
        for target in standard_targets:
            if not isinstance(target, dict):
                continue
            target_id = str(target.get("target_id") or "")
            result = result_by_target.get(target_id, {})
            items.append(
                {
                    **target,
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "issue_category": result.get("issue_category"),
                }
            )
        return items
    return results


def _standard_doc_panel(items: list[dict[str, Any]]) -> str:
    groups = [
        ("main", "标签主文字"),
        ("enterprise", "企业信息"),
        ("content", "多内容物组合装"),
        ("nutrition", "营养成分表"),
        ("other", "其他标签文字"),
    ]
    buckets = {key: [] for key, _ in groups}
    for item in items:
        buckets[_standard_bucket(item)].append(item)

    html = []
    for key, label in groups:
        group_items = buckets[key]
        if not group_items:
            continue
        html.append(f'<section class="standard-group" data-standard-group="{_attr(key)}"><h3>{_html(label)} <span>{len(group_items)}</span></h3>')
        for item in group_items:
            html.append(_standard_doc_item(item))
        html.append("</section>")
    return "".join(html)


def _standard_bucket(item: dict[str, Any]) -> str:
    category = str(item.get("category") or "")
    if category in {"main", "enterprise", "content", "nutrition", "other"}:
        return category
    semantic_key = str(item.get("semantic_key") or "")
    if semantic_key.startswith(("principal.", "manufacturer.", "distributor.", "enterprise.")):
        return "enterprise"
    if semantic_key.startswith("content_item."):
        return "content"
    if semantic_key.startswith("nutrition."):
        return "nutrition"
    return "other" if "other" in semantic_key else "main"


def _standard_doc_item(item: dict[str, Any]) -> str:
    status = str(item.get("status") or "not_compared")
    return (
        '<button class="standard-item" type="button" '
        f'data-target-id="{_attr(item.get("target_id"))}" '
        f'data-status="{_attr(status)}">'
        '<span class="standard-item-head">'
        f'<strong>{_html(item.get("label"))}</strong>'
        f'<span class="status {status}">{_html(_status_label(status))}</span>'
        '</span>'
        f'<span class="standard-text">{_html_multiline(item.get("expected_text"))}</span>'
        "</button>"
    )


def _result_card(index: int, result: dict[str, Any]) -> str:
    candidate = _as_dict(result.get("selected_candidate"))
    status = str(result.get("status") or "")
    candidate_text = candidate.get("text") if candidate else "未在包装图中找到对应文字"
    hint = _customer_hint(result)
    hint_html = f'<small class="hint">{_html(hint)}</small>' if hint else ""
    return (
        '<button class="result-card" type="button" '
        f'data-status="{_attr(status)}" '
        f'data-target-id="{_attr(result.get("target_id"))}">'
        f'<span class="index">{index}</span>'
        '<span class="body">'
        '<span class="result-head">'
        f'<strong>{_html(result.get("label"))}</strong>'
        f'<span class="status {status}">{_html(_status_label(status))}</span>'
        '</span>'
        f'<span class="expected"><b>标准文档</b>{_html_multiline(result.get("expected_text"))}</span>'
        f'<span class="actual"><b>包装图文字</b>{_html_multiline(candidate_text)}</span>'
        f"{hint_html}"
        "</span>"
        "</button>"
    )


def _grouped_result_cards(results: list[dict[str, Any]], extra_items: list[dict[str, Any]]) -> str:
    groups = [
        ("problem", "不一致或缺失"),
        ("review", "需人工复核"),
        ("extra", "包装图多出文字"),
        ("pass", "通过"),
    ]
    buckets = {key: [] for key, _ in groups}
    for result in results:
        buckets[_result_bucket(result)].append(result)
    buckets["extra"].extend(extra_items)

    html = []
    index = 1
    for key, label in groups:
        items = buckets[key]
        if not items:
            continue
        html.append(f'<section class="result-group" data-result-group="{_attr(key)}"><h3>{_html(label)} <span>{len(items)}</span></h3>')
        for result in items:
            if key == "extra":
                html.append(_extra_text_card(index, result))
            else:
                html.append(_result_card(index, result))
            index += 1
        html.append("</section>")
    return "".join(html)


def _result_bucket(result: dict[str, Any]) -> str:
    if result.get("status") == "pass":
        return "pass"
    if result.get("status") == "manual_review":
        return "review"
    return "problem"


def _extra_text_items(unmatched_print_text: dict[str, Any] | None) -> list[dict[str, Any]]:
    items = _as_list(_as_dict(unmatched_print_text).get("items"))
    normalized = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        normalized.append({**item, "target_id": f"extra_{index:04d}"})
    return normalized


def _extra_anchors(extra_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        str(item.get("target_id")): [str(item.get("ocr_line_id"))]
        for item in extra_items
        if item.get("target_id") and item.get("ocr_line_id")
    }


def _extra_text_card(index: int, item: dict[str, Any]) -> str:
    return (
        '<button class="result-card extra-card" type="button" '
        'data-status="info_extra_text" '
        f'data-target-id="{_attr(item.get("target_id"))}">'
        f'<span class="index">{index}</span>'
        '<span class="body">'
        '<span class="result-head">'
        "<strong>包装图多出文字</strong>"
        '<span class="status info_extra_text">包装图多出文字</span>'
        '</span>'
        '<span class="expected"><b>标准文档</b>无对应要求</span>'
        f'<span class="actual"><b>包装图文字</b>{_html_multiline(item.get("text"))}</span>'
        '<small class="hint">请确认该文字是否允许出现在包装图上。</small>'
        "</span>"
        "</button>"
    )


def _overall_status_label(status: Any) -> str:
    return {
        "pass": "全部通过",
        "fail": "未通过",
        "manual_review": "需人工复核",
    }.get(str(status or ""), "未通过")


def _status_label(status: Any) -> str:
    return {
        "pass": "通过",
        "critical_missing": "缺失",
        "critical_mismatch": "不一致",
        "manual_review": "需人工复核",
        "info_extra_text": "包装图多出文字",
        "not_compared": "未检查",
    }.get(str(status or ""), "需人工复核")


def _customer_hint(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "")
    if status == "pass":
        return ""
    if status == "critical_missing":
        return "请确认包装图是否漏印该字段，或图片是否清晰完整。"
    if status == "critical_mismatch":
        return "请核对标准文档与包装图文字，前缀、数字、单位和关键文字需一致。"
    if status == "manual_review":
        return "该项识别结果不够稳定，请人工核对标准文档和包装图文字。"
    return "请人工确认该项是否符合标准文档要求。"


def _relative_path(base_dir: Path, path: Path) -> str:
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return str(path)


def _html(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=False)


def _attr(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _html_multiline(value: Any) -> str:
    return "<br>".join(_html(value).splitlines())


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


_SCRIPT = """
function cardMatchesFilter(card, filter) {
  const status = card.dataset.status || '';
  if (filter === 'all') return true;
  if (filter === 'critical') return status === 'critical_missing' || status === 'critical_mismatch';
  return status === filter;
}
function updateGroupVisibility() {
  document.querySelectorAll('.result-group').forEach(group => {
    const visibleCards = Array.from(group.querySelectorAll('.result-card')).filter(card => !card.hidden);
    group.hidden = visibleCards.length === 0;
  });
  const anyVisible = Array.from(document.querySelectorAll('.result-card')).some(card => !card.hidden);
  const emptyState = document.querySelector('.empty-state');
  if (emptyState) emptyState.hidden = anyVisible;
}
function applyFilter(filter, label) {
  clearActive();
  document.querySelectorAll('.summary-card').forEach(card => {
    card.classList.toggle('active', (card.dataset.filterStatus || 'all') === filter);
  });
  document.querySelectorAll('.result-card').forEach(card => {
    card.hidden = !cardMatchesFilter(card, filter);
  });
  const filterLabel = document.getElementById('filter-label');
  if (filterLabel) filterLabel.textContent = filter === 'all' ? '全部结果' : `当前筛选：${label || '全部结果'}`;
  updateGroupVisibility();
}
function clearActive() {
  document.querySelectorAll('.result-card.active, .standard-item.active, .ocr-box.active').forEach(el => el.classList.remove('active'));
}
function activateTarget(targetId) {
  clearActive();
  const card = document.querySelector(`.result-card[data-target-id="${CSS.escape(targetId)}"]`);
  if (card) card.classList.add('active');
  const standardItem = document.querySelector(`.standard-item[data-target-id="${CSS.escape(targetId)}"]`);
  if (standardItem) {
    standardItem.classList.add('active');
    standardItem.scrollIntoView({block: 'center', inline: 'nearest', behavior: 'smooth'});
  }
  const lineIds = TARGET_ANCHORS[targetId] || [];
  lineIds.forEach(lineId => {
    document.querySelectorAll(`.ocr-box[data-line-id="${CSS.escape(lineId)}"]`).forEach(el => el.classList.add('active'));
  });
  const first = lineIds.map(lineId => document.querySelector(`.ocr-box[data-line-id="${CSS.escape(lineId)}"]`)).find(Boolean);
  if (first) first.scrollIntoView({block: 'center', inline: 'center', behavior: 'smooth'});
}
document.querySelectorAll('.result-card').forEach(card => {
  card.addEventListener('click', () => activateTarget(card.dataset.targetId || ''));
});
document.querySelectorAll('.standard-item').forEach(item => {
  item.addEventListener('click', () => activateTarget(item.dataset.targetId || ''));
});
document.querySelectorAll('.summary-card').forEach(card => {
  card.addEventListener('click', () => {
    const label = card.querySelector('span')?.textContent || '';
    applyFilter(card.dataset.filterStatus || 'all', label);
  });
});
applyFilter('all', '全部');
"""


_CSS = """
:root {
  color-scheme: light;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #20242a;
  background: #f5f7fa;
}
body {
  margin: 0;
}
[hidden] {
  display: none !important;
}
main {
  padding: 20px;
}
h1 {
  margin: 0 0 14px;
  font-size: 24px;
}
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}
.summary-card {
  border: 1px solid #d8dee9;
  background: #fff;
  padding: 10px;
  text-align: left;
  cursor: pointer;
  color: inherit;
}
.summary-card.active {
  border-color: #2b6ef6;
  box-shadow: 0 0 0 2px rgba(43, 110, 246, 0.12);
}
.summary-card span {
  display: block;
  color: #687182;
  font-size: 12px;
}
.summary-card strong {
  display: block;
  margin-top: 4px;
  font-size: 16px;
}
.quality-banner {
  border: 1px solid #d8dee9;
  background: #fff;
  padding: 10px 12px;
  margin-bottom: 16px;
  font-size: 13px;
  line-height: 1.45;
}
.quality-banner.ok {
  color: #1f7a4d;
}
.quality-banner.warn {
  border-color: #f0c36a;
  background: #fff8e6;
  color: #7a4c00;
}
.compare-shell {
  display: grid;
  grid-template-columns: minmax(300px, 0.72fr) minmax(460px, 1.08fr) minmax(360px, 0.82fr);
  gap: 16px;
  align-items: start;
}
.pane {
  border: 1px solid #d8dee9;
  background: #fff;
  min-height: 400px;
}
.pane-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 12px 14px;
  border-bottom: 1px solid #e6ebf2;
  position: sticky;
  top: 0;
  background: #fff;
  z-index: 5;
}
.pane-header h2 {
  margin: 0;
  font-size: 16px;
}
.pane-header span {
  color: #687182;
  font-size: 12px;
}
.standard-pane {
  max-height: calc(100vh - 140px);
  overflow: auto;
}
.standard-doc {
  display: grid;
  gap: 10px;
  padding: 12px;
}
.standard-group {
  display: grid;
  gap: 8px;
}
.standard-group h3 {
  margin: 8px 0 2px;
  font-size: 13px;
  color: #394150;
}
.standard-group h3 span {
  color: #687182;
  font-weight: 500;
}
.standard-item {
  display: grid;
  gap: 6px;
  border: 1px solid #d8dee9;
  background: #fff;
  color: inherit;
  text-align: left;
  padding: 10px;
  cursor: pointer;
}
.standard-item.active {
  border-color: #ff3b30;
  box-shadow: 0 0 0 2px rgba(255, 59, 48, 0.12);
}
.standard-item-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.standard-item-head strong {
  font-size: 14px;
}
.standard-item small {
  color: #687182;
  line-height: 1.35;
}
.standard-text {
  display: block;
  line-height: 1.45;
  word-break: break-word;
}
.image-pane {
  overflow: auto;
  max-height: calc(100vh - 140px);
}
.image-wrap {
  position: relative;
  width: 100%;
  min-width: 520px;
}
.image-wrap img {
  display: block;
  width: 100%;
  height: auto;
}
.ocr-box {
  position: absolute;
  border: 1px solid rgba(43, 110, 246, 0.22);
  background: rgba(43, 110, 246, 0.04);
  padding: 0;
}
.ocr-box.active {
  border: 2px solid #ff3b30;
  background: rgba(255, 59, 48, 0.18);
  z-index: 4;
}
.result-pane {
  max-height: calc(100vh - 140px);
  overflow: auto;
}
.result-list {
  display: grid;
  gap: 8px;
  padding: 12px;
}
.result-group {
  display: grid;
  gap: 8px;
}
.result-group h3 {
  margin: 8px 0 2px;
  font-size: 13px;
  color: #394150;
}
.result-group h3 span {
  color: #687182;
  font-weight: 500;
}
.result-card {
  display: grid;
  grid-template-columns: 32px 1fr;
  gap: 8px;
  border: 1px solid #d8dee9;
  background: #fff;
  text-align: left;
  padding: 10px;
  cursor: pointer;
}
.result-card.active {
  border-color: #ff3b30;
  box-shadow: 0 0 0 2px rgba(255, 59, 48, 0.12);
}
.empty-state {
  border: 1px dashed #c8d0dd;
  color: #687182;
  padding: 16px;
  text-align: center;
}
.index {
  color: #687182;
  font-size: 12px;
}
.body {
  display: grid;
  gap: 6px;
}
.result-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.body strong {
  font-size: 14px;
}
.body small {
  color: #687182;
}
.hint {
  color: #394150;
}
.expected, .actual {
  display: block;
  border-top: 1px solid #eef2f7;
  padding-top: 6px;
  line-height: 1.45;
}
.expected b, .actual b {
  display: block;
  color: #687182;
  font-size: 12px;
  margin-bottom: 3px;
}
.status.pass { color: #1f7a4d; }
.status.critical_missing, .status.critical_mismatch { color: #c5281c; }
.status.manual_review { color: #9a5b00; }
.status.info_extra_text { color: #687182; }
pre {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
}
@media (max-width: 900px) {
  .compare-shell {
    grid-template-columns: 1fr;
  }
}
"""
