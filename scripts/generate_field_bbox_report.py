from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import struct
import unicodedata
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a visual field/value/bbox HTML report from parser result JSON.")
    parser.add_argument("--result", type=Path, required=True, help="Path to parser result.json.")
    parser.add_argument("--page-images-dir", type=Path, required=True, help="Directory containing page_001.png style page images.")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path.")
    args = parser.parse_args()

    data = json.loads(args.result.read_text(encoding="utf-8"))
    pages = _page_images(args.page_images_dir)
    evidence_by_id = {item.get("evidence_id"): item for item in data.get("evidence", []) if isinstance(item, dict)}
    fields = _field_rows(data, evidence_by_id)
    tables = _table_rows(data, evidence_by_id)
    scope_nodes = _scope_nodes(data)
    page_boxes = _page_boxes(fields, tables, scope_nodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_html(data, pages, fields, tables, page_boxes), encoding="utf-8")


def _page_images(page_images_dir: Path) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for path in sorted(page_images_dir.glob("page_*.png")):
        page = _page_number(path)
        if page is None:
            continue
        width, height = _png_size(path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        pages[page] = {
            "path": str(path),
            "width": width,
            "height": height,
            "src": f"data:{media_type};base64,{encoded}",
        }
    return pages


def _page_number(path: Path) -> int | None:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return (1, 1)


def _field_rows(data: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fields = data.get("extracted_data", {}).get("fields", {})
    rows: list[dict[str, Any]] = []
    for index, (field_key, field) in enumerate(fields.items(), start=1):
        if not isinstance(field, dict):
            continue
        evidence_refs = [str(ref) for ref in field.get("evidence_refs", [])]
        evidences = [evidence_by_id[ref] for ref in evidence_refs if ref in evidence_by_id]
        rows.append(
            {
                "index": index,
                "field_key": field_key,
                "field_id": str(field.get("field_id") or field_key),
                "semantic_key": str(field.get("semantic_key") or ""),
                "display_name": str(field.get("display_name") or ""),
                "value": str(field.get("raw_value") or ""),
                "status": str(field.get("status") or ""),
                "risk_level": str(field.get("risk_level") or ""),
                "review_required": bool(field.get("review_required")),
                "confidence": field.get("confidence", {}).get("overall") if isinstance(field.get("confidence"), dict) else None,
                "reason": field.get("reason"),
                "evidence_refs": evidence_refs,
                "evidences": evidences,
            }
        )
    return rows


def _table_rows(data: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    raw_tables = data.get("extracted_data", {}).get("tables", [])
    tables: list[dict[str, Any]] = []
    for index, table in enumerate(raw_tables, start=1):
        if not isinstance(table, dict):
            continue
        columns = [column for column in table.get("columns", []) if isinstance(column, dict)]
        rows = [row for row in table.get("rows", []) if isinstance(row, dict)]
        row_items = []
        cell_count = 0
        non_empty_cell_count = 0
        table_evidence_refs = _string_refs(table.get("evidence_refs"))
        table_evidences = [evidence_by_id[ref] for ref in table_evidence_refs if ref in evidence_by_id]
        for row_index, row in enumerate(rows, start=1):
            cells = [cell for cell in row.get("cells", []) if isinstance(cell, dict)]
            row_evidence_refs = _unique(_string_refs(row.get("evidence_refs")))
            rendered_cells = []
            for cell_index, cell in enumerate(cells, start=1):
                refs = _unique(_string_refs(cell.get("evidence_refs")) or row_evidence_refs)
                value = str(cell.get("raw_value") or cell.get("normalized_value") or "")
                cell_count += 1
                if value.strip():
                    non_empty_cell_count += 1
                rendered_cells.append(
                    {
                        "column_id": str(cell.get("column_id") or f"col_{cell_index:03d}"),
                        "value": value,
                        "evidence_refs": refs,
                        "bbox_count": sum(1 for ref in refs if _normalized_bbox(evidence_by_id.get(ref, {}))),
                    }
                )
            row_evidences = [evidence_by_id[ref] for ref in row_evidence_refs if ref in evidence_by_id]
            row_items.append(
                {
                    "index": row_index,
                    "row_id": str(row.get("row_id") or f"row_{row_index:04d}"),
                    "row_key": str(row.get("row_key") or ""),
                    "evidence_refs": row_evidence_refs,
                    "evidences": row_evidences,
                    "bbox_count": sum(1 for evidence in row_evidences if _normalized_bbox(evidence)),
                    "cells": rendered_cells,
                }
            )
        tables.append(
            {
                "index": index,
                "table_id": str(table.get("table_id") or f"tbl_{index:04d}"),
                "table_type": str(table.get("table_type") or ""),
                "title": str(table.get("title") or table.get("display_name") or ""),
                "status": str(table.get("status") or ""),
                "risk_level": str(table.get("risk_level") or ""),
                "review_required": bool(table.get("review_required")),
                "bbox_status": str(table.get("bbox_status") or ""),
                "source": str(table.get("source") or table.get("parser") or ""),
                "columns": columns,
                "rows": row_items,
                "row_count": len(row_items),
                "column_count": len(columns),
                "cell_count": cell_count,
                "non_empty_cell_count": non_empty_cell_count,
                "evidence_refs": table_evidence_refs,
                "evidences": table_evidences,
            }
        )
    return tables


def _scope_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    report = data.get("metadata", {}).get("label_text_scope_report", {})
    nodes = []
    for key in ("ignored_noise_nodes", "unknown_scope_nodes"):
        for item in report.get(key, []):
            if isinstance(item, dict):
                nodes.append(item)
    return nodes


def _page_boxes(fields: list[dict[str, Any]], tables: list[dict[str, Any]], scope_nodes: list[dict[str, Any]] | None = None) -> dict[int, list[dict[str, Any]]]:
    pages: dict[int, list[dict[str, Any]]] = {}
    for row in fields:
        for evidence in row["evidences"]:
            bbox = _normalized_bbox(evidence)
            if bbox is None:
                continue
            page = int(evidence.get("page") or 1)
            pages.setdefault(page, []).append(
                {
                    "field_id": row["field_id"],
                    "semantic_key": row["semantic_key"],
                    "value": row["value"],
                    "evidence_id": evidence.get("evidence_id"),
                    "source_text": evidence.get("source_text"),
                    "kind": "field",
                    **bbox,
                }
            )
    for table in tables:
        for evidence in table["evidences"]:
            bbox = _normalized_bbox(evidence)
            if bbox is None:
                continue
            page = int(evidence.get("page") or 1)
            pages.setdefault(page, []).append(
                {
                    "kind": "table",
                    "table_id": table["table_id"],
                    "table_type": table["table_type"],
                    "value": table["title"] or table["table_type"],
                    "evidence_id": evidence.get("evidence_id"),
                    "source_text": evidence.get("source_text"),
                    **bbox,
                }
            )
        for row in table["rows"]:
            for evidence in row["evidences"]:
                bbox = _normalized_bbox(evidence)
                if bbox is None:
                    continue
                page = int(evidence.get("page") or 1)
                pages.setdefault(page, []).append(
                    {
                        "kind": "table-row",
                        "table_id": table["table_id"],
                        "row_id": row["row_id"],
                        "table_type": table["table_type"],
                        "value": row["row_key"],
                        "evidence_id": evidence.get("evidence_id"),
                        "source_text": evidence.get("source_text"),
                        **bbox,
                    }
                )
    for node in scope_nodes or []:
        bbox = node.get("bbox_normalized")
        if not isinstance(bbox, dict):
            continue
        try:
            page = int(node.get("page") or 1)
            pages.setdefault(page, []).append(
                {
                    "kind": "scope-node",
                    "node_id": node.get("node_id"),
                    "scope_status": node.get("scope_status"),
                    "scope_category": node.get("scope_category"),
                    "value": node.get("text"),
                    "source_text": node.get("reason"),
                    "x1": float(bbox["x1"]),
                    "y1": float(bbox["y1"]),
                    "x2": float(bbox["x2"]),
                    "y2": float(bbox["y2"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return pages


def _normalized_bbox(evidence: dict[str, Any]) -> dict[str, float] | None:
    normalized = evidence.get("bbox_normalized")
    if isinstance(normalized, dict):
        try:
            return {
                "x1": float(normalized["x1"]),
                "y1": float(normalized["y1"]),
                "x2": float(normalized["x2"]),
                "y2": float(normalized["y2"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    bbox = evidence.get("bbox_pdf")
    if not isinstance(bbox, dict):
        return None
    try:
        page_width = float(bbox["page_width"])
        page_height = float(bbox["page_height"])
        x1 = float(bbox["x"]) / page_width
        y1 = float(bbox["y"]) / page_height
        x2 = (float(bbox["x"]) + float(bbox["width"])) / page_width
        y2 = (float(bbox["y"]) + float(bbox["height"])) / page_height
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _render_html(
    data: dict[str, Any],
    pages: dict[int, dict[str, Any]],
    fields: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    page_boxes: dict[int, list[dict[str, Any]]],
) -> str:
    contract = data.get("metadata", {}).get("output_contract_validation_report", {})
    quality = data.get("metadata", {}).get("standard_artifacts", {}).get("quality_report", {})
    agent = data.get("metadata", {}).get("agent_harness", {})
    metrics = _metrics(data, fields, tables)
    summary = {
        "Fields": len(fields),
        "Tables": len(tables),
        "Nutrition rows": metrics["nutrition_row_count"],
        "Cell fill": metrics["cell_fill_rate"],
        "Evidence boxes": sum(len(boxes) for boxes in page_boxes.values()),
        "Contract": contract.get("status", "unknown"),
        "Quality": quality.get("status", "unknown"),
        "Agent fields": agent.get("agent_plan_field_count", 0),
        "Rule fallback fields": agent.get("rule_fallback_field_count", 0),
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Field BBox Report</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #667085;
      --line: #d0d7de;
      --accent: #0b6bcb;
      --warn: #b54708;
      --danger: #b42318;
      --ok: #027a48;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.94);
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      background: #fff;
    }}
    .metric b {{ margin-right: 4px; }}
    h2 {{ margin: 0; font-size: 15px; }}
    h3 {{ margin: 10px 0 6px; font-size: 14px; }}
    main.report {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      padding: 14px;
      align-items: start;
    }}
    .report-section {{
      width: 100%;
    }}
    .field-bbox-grid {{
      display: grid;
      grid-template-columns: minmax(420px, 1fr) minmax(520px, 1.35fr);
      gap: 14px;
      align-items: start;
      content-visibility: auto;
      contain-intrinsic-block-size: 1200px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }}
    input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      font: inherit;
    }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{
      border-bottom: 1px solid #eaedf0;
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 3;
      color: #475467;
      font-weight: 600;
    }}
    tr[data-field-id] {{ cursor: pointer; }}
    tr[data-field-id]:hover, tr.active {{ background: #eef6ff; }}
    .field-table-wrap {{ max-height: calc(100vh - 126px); overflow: auto; }}
    .section {{
      padding: 10px;
      border-bottom: 1px solid var(--line);
    }}
    .metrics-row {{
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(320px, .8fr);
      gap: 12px;
      align-items: start;
      margin-top: 8px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
      min-height: 58px;
      min-width: 0;
    }}
    .metric-card .label {{ color: var(--muted); font-size: 12px; }}
    .metric-card .value {{
      font-size: 14px;
      font-weight: 700;
      line-height: 1.25;
      margin-top: 2px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #fff;
      font-size: 12px;
    }}
    .chip.pass {{ border-color: #75c7a1; background: #ecfdf3; color: var(--ok); }}
    .chip.fail {{ border-color: #fecdca; background: #fef3f2; color: var(--danger); }}
    .required-panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .required-panel h3 {{ margin-top: 0; }}
    .vdg-grid {{
      display: grid;
      grid-template-columns: minmax(560px, 1fr) minmax(320px, .38fr);
      gap: 12px;
      align-items: start;
    }}
    .vdg-graph-wrap {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      overflow-x: auto;
      padding: 8px;
      contain: paint;
      isolation: isolate;
    }}
    .vdg-svg {{
      display: block;
      width: 100%;
      min-width: 980px;
      height: auto;
      font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .vdg-edge {{ stroke: rgb(152, 162, 179); stroke-width: 1.25; fill: none; opacity: .72; }}
    .vdg-edge.strong {{ stroke: #0b6bcb; stroke-width: 1.8; opacity: .9; }}
    .vdg-node rect {{
      stroke: #98a2b3;
      stroke-width: 1;
      fill: #fff;
      rx: 6;
    }}
    .vdg-node.page rect {{ fill: #eff8ff; stroke: #84caff; }}
    .vdg-node.region rect {{ fill: #f4f3ff; stroke: #bdb4fe; }}
    .vdg-node.anchor rect {{ fill: #ecfdf3; stroke: #75c7a1; }}
    .vdg-node.table rect {{ fill: #fff7ed; stroke: #fdba74; }}
    .vdg-node.status rect {{ fill: #fef3f2; stroke: #fecdca; }}
    .vdg-node text.title {{ font-weight: 700; fill: #1f2933; }}
    .vdg-node text.meta {{ fill: #667085; }}
    .vdg-side {{
      display: grid;
      gap: 10px;
    }}
    .vdg-side .section {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .table-block {{
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 10px 0;
      overflow: hidden;
      background: #fff;
    }}
    .table-head {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }}
    .table-meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 5px; }}
    .nutrition-table td, .nutrition-table th {{ font-size: 13px; }}
    .nutrition-table tr[data-table-row-id] {{ cursor: pointer; }}
    .nutrition-table tr[data-table-row-id]:hover, .nutrition-table tr.active {{ background: #f4f3ff; }}
    .status {{ font-weight: 600; }}
    .status.verified {{ color: var(--ok); }}
    .status.manual_review_required {{ color: var(--danger); }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .doc-view {{ padding: 10px; max-height: calc(100vh - 96px); overflow: auto; }}
    .page {{
      position: relative;
      margin: 0 auto 18px;
      border: 1px solid var(--line);
      background: #fff;
      width: min(100%, 900px);
    }}
    .page img {{ display: block; width: 100%; height: auto; }}
    .box {{
      position: absolute;
      border: 2px solid var(--accent);
      background: rgba(11,107,203,.12);
      min-width: 6px;
      min-height: 6px;
      pointer-events: auto;
    }}
    .box.review {{ border-color: var(--danger); background: rgba(180,35,24,.16); }}
    .box.table {{
      border-color: #12b76a;
      background: rgba(18,183,106,.14);
    }}
    .box.table-row {{
      border-color: #7a5af8;
      background: rgba(122,90,248,.14);
    }}
    .box.scope-node {{
      border-color: #b42318;
      background: rgba(180,35,24,.18);
      border-style: dashed;
    }}
    .box.active {{
      border-color: #f79009;
      background: rgba(247,144,9,.22);
      box-shadow: 0 0 0 2px rgba(247,144,9,.28);
      z-index: 5;
    }}
    .box-label {{
      position: absolute;
      left: -2px;
      top: -20px;
      display: none;
      max-width: 320px;
      padding: 2px 5px;
      border-radius: 4px;
      color: #fff;
      background: rgba(17,24,39,.88);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 12px;
    }}
    .box:hover .box-label, .box.active .box-label {{ display: block; }}
    .page-title {{ margin: 0 0 8px; font-weight: 700; color: #344054; }}
    @media (max-width: 980px) {{
      .field-bbox-grid, .metrics-row, .vdg-grid {{ grid-template-columns: 1fr; }}
      .metrics-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .field-table-wrap, .doc-view {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Field BBox Report</h1>
    <div class="summary">{_render_summary(summary)}</div>
  </header>
  <main class="report">
    <section class="panel report-section metrics-section">
      <div class="section">
        <h2>Metrics + Required Nutrition Rows</h2>
        {_render_metrics(metrics)}
      </div>
    </section>
    <section class="panel report-section vdg-section">
      <div class="section">
        <h2>VDG</h2>
        {_render_vdg_section(data, metrics)}
      </div>
    </section>
    <section class="report-section field-bbox-grid">
    <section class="panel">
      <div class="toolbar">
        <input id="filter" placeholder="Filter field, table row, value, status, evidence id">
      </div>
      <div class="field-table-wrap">
        <div class="section"><h2>Fields</h2></div>
        <table>
          <thead>
            <tr>
              <th style="width: 46px;">#</th>
              <th style="width: 180px;">Field</th>
              <th>Value</th>
              <th style="width: 128px;">Status</th>
              <th style="width: 110px;">BBox</th>
            </tr>
          </thead>
          <tbody>{_render_field_rows(fields)}</tbody>
        </table>
        <div class="section">
          <h2>Tables</h2>
          {_render_tables(tables)}
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="doc-view">{_render_pages(pages, page_boxes)}</div>
    </section>
    </section>
  </main>
  <script>
    const rows = Array.from(document.querySelectorAll('tr[data-field-id]'));
    const boxes = Array.from(document.querySelectorAll('.box[data-field-id]'));
    const tableRows = Array.from(document.querySelectorAll('tr[data-table-row-id]'));
    const tableBoxes = Array.from(document.querySelectorAll('.box[data-table-id]'));
    function setActive(fieldId, boxId) {{
      rows.forEach(row => row.classList.toggle('active', row.dataset.fieldId === fieldId));
      boxes.forEach(box => box.classList.toggle('active', box.dataset.boxId === boxId));
      const activeBox = boxes.find(box => box.dataset.boxId === boxId);
      if (activeBox) activeBox.scrollIntoView({{block: 'center', inline: 'center', behavior: 'smooth'}});
    }}
    function setActiveTable(tableId, rowId, boxId) {{
      tableRows.forEach(row => row.classList.toggle('active', row.dataset.tableRowId === rowId));
      tableBoxes.forEach(box => box.classList.toggle('active', box.dataset.boxId === boxId));
      const activeBox = tableBoxes.find(box => box.dataset.boxId === boxId);
      if (activeBox) activeBox.scrollIntoView({{block: 'center', inline: 'center', behavior: 'smooth'}});
    }}
    rows.forEach(row => {{
      const selectFirstBox = () => {{
        const firstBox = boxes.find(box => box.dataset.fieldId === row.dataset.fieldId);
        setActive(row.dataset.fieldId, firstBox && firstBox.dataset.boxId);
      }};
      row.addEventListener('mouseenter', selectFirstBox);
      row.addEventListener('click', selectFirstBox);
    }});
    boxes.forEach(box => {{
      box.addEventListener('mouseenter', () => setActive(box.dataset.fieldId, box.dataset.boxId));
      box.addEventListener('click', () => setActive(box.dataset.fieldId, box.dataset.boxId));
    }});
    tableRows.forEach(row => {{
      const selectFirstBox = () => {{
        const firstBox = tableBoxes.find(box => box.dataset.tableRowId === row.dataset.tableRowId);
        setActiveTable(row.dataset.tableId, row.dataset.tableRowId, firstBox && firstBox.dataset.boxId);
      }};
      row.addEventListener('mouseenter', selectFirstBox);
      row.addEventListener('click', selectFirstBox);
    }});
    tableBoxes.forEach(box => {{
      box.addEventListener('mouseenter', () => setActiveTable(box.dataset.tableId, box.dataset.tableRowId, box.dataset.boxId));
      box.addEventListener('click', () => setActiveTable(box.dataset.tableId, box.dataset.tableRowId, box.dataset.boxId));
    }});
    const filter = document.getElementById('filter');
    filter.addEventListener('input', () => {{
      const q = filter.value.trim().toLowerCase();
      [...rows, ...tableRows].forEach(row => {{
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""


def _render_summary(summary: dict[str, Any]) -> str:
    return "".join(f'<span class="metric"><b>{_e(key)}</b>{_e(value)}</span>' for key, value in summary.items())


def _metrics(data: dict[str, Any], fields: list[dict[str, Any]], tables: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = data.get("metadata", {})
    missing = metadata.get("missing_item_report", {})
    quality = metadata.get("standard_artifacts", {}).get("quality_report", {})
    table_quality = metadata.get("table_parser", {}).get("table_quality_report", {})
    vdg_quality = metadata.get("vdg_quality_report", {})
    vdg_consumption = metadata.get("vdg_consumption_report", {})
    label_text_scope = metadata.get("label_text_scope_report", {})
    layout_quality = metadata.get("layout_quality_report", {})
    risks = data.get("risks", [])
    review_tasks = data.get("review_tasks", [])
    total_cells = sum(table["cell_count"] for table in tables)
    non_empty_cells = sum(table["non_empty_cell_count"] for table in tables)
    nutrition_tables = [table for table in tables if table["table_type"] == "nutrition_facts"]
    nutrition_rows = [row for table in nutrition_tables for row in table["rows"]]
    required_nutrients = ["能量", "蛋白质", "脂肪", "碳水化合物", "钠"]
    required_coverage = {
        nutrient: _nutrient_present(nutrient, nutrition_rows)
        for nutrient in required_nutrients
    }
    return {
        "field_count": len(fields),
        "table_count": len(tables),
        "nutrition_table_count": len(nutrition_tables),
        "nutrition_row_count": len(nutrition_rows),
        "table_row_count": sum(table["row_count"] for table in tables),
        "table_cell_count": total_cells,
        "non_empty_cell_count": non_empty_cells,
        "cell_fill_rate": f"{(non_empty_cells / total_cells * 100):.0f}%" if total_cells else "0%",
        "critical_missing_count": missing.get("missing_count", 0),
        "missing_field_count": missing.get("missing_field_count", 0),
        "missing_table_count": missing.get("missing_table_count", 0),
        "high_risk_count": sum(1 for risk in risks if isinstance(risk, dict) and risk.get("risk_level") == "high"),
        "review_task_count": len(review_tasks),
        "quality_status": quality.get("status", "unknown"),
        "table_quality_status": table_quality.get("status", "unknown"),
        "table_quality_issue_count": table_quality.get("issue_count", 0),
        "vdg_quality_status": vdg_quality.get("status", "unknown"),
        "vdg_source_span_coverage_rate": vdg_quality.get("source_span_coverage_rate", "unknown"),
        "vdg_unknown_important_node_count": vdg_consumption.get("unknown_important_node_count", 0),
        "vdg_conflict_node_count": vdg_consumption.get("conflict_node_count", 0),
        "vdg_boundary_issue_count": vdg_quality.get("boundary_issue_count", 0),
        "nutrition_table_candidate_status": vdg_quality.get("nutrition_table_candidate_status", "unknown"),
        "label_text_scope_status": label_text_scope.get("status", "unknown"),
        "extracted_out_of_scope_count": label_text_scope.get("extracted_out_of_scope_count", 0),
        "ignored_noise_node_count": label_text_scope.get("ignored_noise_node_count", 0),
        "unknown_scope_node_count": label_text_scope.get("unknown_scope_node_count", 0),
        "scope_gate_rejected_count": label_text_scope.get("scope_gate_rejected_count", 0),
        "scope_ignored_nodes": label_text_scope.get("ignored_noise_nodes", []),
        "scope_unknown_nodes": label_text_scope.get("unknown_scope_nodes", []),
        "layout_mode": layout_quality.get("mode", "legacy"),
        "layout_quality_status": layout_quality.get("status", "disabled"),
        "pdf_character_atom_count": layout_quality.get("pdf_character_atom_count", 0),
        "layout_candidate_count": layout_quality.get("layout_candidate_count", 0),
        "nutrition_layout_candidate_count": layout_quality.get("nutrition_layout_candidate_count", 0),
        "producer_layout_candidate_count": layout_quality.get("producer_layout_candidate_count", 0),
        "layout_boundary_issue_count": layout_quality.get("layout_boundary_issue_count", 0),
        "cross_page_candidate_count": layout_quality.get("cross_page_candidate_count", 0),
        "layout_fallback_used": layout_quality.get("fallback_used", False),
        "layout_issues": layout_quality.get("issues", []),
        "vdg_issues": vdg_quality.get("issues", []),
        "vdg_unknown_nodes": vdg_consumption.get("unknown_important_nodes", []),
        "vdg_conflict_nodes": vdg_consumption.get("conflict_nodes", []),
        "required_nutrition_coverage": required_coverage,
        "missing_fields": missing.get("missing_fields", []),
        "missing_tables": missing.get("missing_tables", []),
        "failed_gate_checks": [check for check in quality.get("gate_checks", []) if isinstance(check, dict) and check.get("result") != "passed"],
    }


def _nutrient_present(nutrient: str, rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        haystack = row.get("row_key", "")
        for cell in row.get("cells", []):
            haystack += " " + str(cell.get("value") or "")
        if nutrient in haystack:
            return True
    return False


def _render_metrics(metrics: dict[str, Any]) -> str:
    cards = [
        ("Fields", metrics["field_count"]),
        ("Tables", metrics["table_count"]),
        ("Nutrition rows", metrics["nutrition_row_count"]),
        ("Cell fill", metrics["cell_fill_rate"]),
        ("Missing fields", metrics["missing_field_count"]),
        ("Missing tables", metrics["missing_table_count"]),
        ("High risks", metrics["high_risk_count"]),
        ("Review tasks", metrics["review_task_count"]),
        ("Table quality", metrics["table_quality_status"]),
        ("VDG quality", metrics["vdg_quality_status"]),
        ("VDG span coverage", metrics["vdg_source_span_coverage_rate"]),
        ("VDG unknown nodes", metrics["vdg_unknown_important_node_count"]),
        ("VDG conflicts", metrics["vdg_conflict_node_count"]),
        ("VDG boundary issues", metrics["vdg_boundary_issue_count"]),
        ("Nutrition candidate", metrics["nutrition_table_candidate_status"]),
        ("Scope status", metrics["label_text_scope_status"]),
        ("Out-of-scope extracted", metrics["extracted_out_of_scope_count"]),
        ("Scope ignored nodes", metrics["ignored_noise_node_count"]),
        ("Scope unknown nodes", metrics["unknown_scope_node_count"]),
        ("Scope rejected", metrics["scope_gate_rejected_count"]),
        ("Layout mode", metrics["layout_mode"]),
        ("Layout quality", metrics["layout_quality_status"]),
        ("PDF atoms", metrics["pdf_character_atom_count"]),
        ("Layout candidates", metrics["layout_candidate_count"]),
        ("Nutrition candidates", metrics["nutrition_layout_candidate_count"]),
        ("Producer candidates", metrics["producer_layout_candidate_count"]),
        ("Layout issues", metrics["layout_boundary_issue_count"]),
        ("Cross-page candidates", metrics["cross_page_candidate_count"]),
    ]
    card_html = "".join(
        f'<div class="metric-card"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div></div>'
        for label, value in cards
    )
    coverage = metrics["required_nutrition_coverage"]
    coverage_html = "".join(
        f'<span class="chip {"pass" if passed else "fail"}">{_e(key)} {"ok" if passed else "missing"}</span>'
        for key, passed in coverage.items()
    )
    missing_html = _render_missing_items(metrics)
    gates_html = _render_failed_gate_checks(metrics)
    return f"""
      <div class="metrics-row">
        <div>
          <div class="metrics-grid">{card_html}</div>
          {missing_html}
          {gates_html}
        </div>
        <div class="required-panel">
          <h3>Required Nutrition Rows</h3>
          <div class="chips">{coverage_html}</div>
        </div>
      </div>
    """


def _render_vdg_section(data: dict[str, Any], metrics: dict[str, Any]) -> str:
    return f"""
      <div class="vdg-grid">
        <div class="vdg-graph-wrap">{_render_vdg_graph(data)}</div>
        <div class="vdg-side">
          <div class="section">{_render_vdg_issues(metrics)}</div>
          <div class="section">{_render_vdg_nodes(metrics)}</div>
          <div class="section">{_render_scope_nodes(metrics)}</div>
        </div>
      </div>
    """


def _render_vdg_graph(data: dict[str, Any]) -> str:
    model = _vdg_graph_model(data)
    nodes = model["nodes"]
    edges = model["edges"]
    width = model["width"]
    height = model["height"]
    if not nodes:
        return '<div class="muted">No VDG graph available.</div>'
    edge_html = []
    for edge in edges:
        source = model["node_by_id"].get(edge["source"])
        target = model["node_by_id"].get(edge["target"])
        if not source or not target:
            continue
        x1 = source["x"] + source["w"]
        y1 = source["y"] + source["h"] / 2
        x2 = target["x"]
        y2 = target["y"] + target["h"] / 2
        cls = " strong" if edge.get("edge_type") in {"belongs_to_table", "belongs_to_region"} else ""
        edge_html.append(
            f'<line class="vdg-edge{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">'
            f'<title>{_e(edge.get("edge_type"))}: {_e(edge.get("source"))} -> {_e(edge.get("target"))}</title></line>'
        )
    node_html = []
    for node in nodes:
        node_html.append(
            f"""
            <g class="vdg-node {_css_class(node['kind'])}" transform="translate({node['x']},{node['y']})">
              <rect width="{node['w']}" height="{node['h']}"></rect>
              <text class="title" x="10" y="19">{_e(_truncate_display(node['title'], 24))}</text>
              <text class="meta" x="10" y="38">{_e(_truncate_display(node['meta'], 28))}</text>
              <title>{_e(node['node_id'])} | {_e(node['title'])} | {_e(node['meta'])}</title>
            </g>
            """
        )
    legend = (
        '<div class="chips">'
        '<span class="chip">page</span><span class="chip">region</span><span class="chip">field anchor</span>'
        '<span class="chip">table/cell</span><span class="chip">status</span>'
        '</div>'
    )
    return f"""
      {legend}
      <svg class="vdg-svg" viewBox="0 0 {width} {height}" role="img" aria-label="VDG graph summary">
        {''.join(edge_html)}
        {''.join(node_html)}
      </svg>
    """


def _vdg_graph_model(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata", {})
    graph = metadata.get("visual_document_graph") or metadata.get("candidate_visual_document_graph") or {}
    graph_nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    graph_edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    raw_by_id = {str(node.get("node_id")): node for node in graph_nodes if node.get("node_id")}
    vdg_quality = metadata.get("vdg_quality_report", {})
    vdg_consumption = metadata.get("vdg_consumption_report", {})
    scope_report = metadata.get("label_text_scope_report", {})
    layout_report = metadata.get("layout_quality_report", {})

    page_nodes = sorted(
        [node for node in graph_nodes if node.get("node_type") == "page"],
        key=lambda node: int(node.get("page") or 0),
    )[:6]
    region_nodes = _select_vdg_nodes(graph_nodes, "region", 10)
    anchor_nodes = _select_anchor_nodes(graph_nodes, vdg_quality, vdg_consumption, 14)
    table_nodes = _select_vdg_nodes(graph_nodes, "table", 8)
    status_nodes = [
        {
            "node_id": "status_vdg_quality",
            "node_type": "status",
            "text": f"VDG {vdg_quality.get('status', 'unknown')}",
            "meta": f"coverage {vdg_quality.get('source_span_coverage_rate', 'unknown')}",
        },
        {
            "node_id": "status_consumption",
            "node_type": "status",
            "text": "Node consumption",
            "meta": f"unknown {vdg_consumption.get('unknown_important_node_count', 0)} / conflict {vdg_consumption.get('conflict_node_count', 0)}",
        },
        {
            "node_id": "status_scope",
            "node_type": "status",
            "text": f"Scope {scope_report.get('status', 'unknown')}",
            "meta": f"unknown {scope_report.get('unknown_scope_node_count', 0)} / rejected {scope_report.get('scope_gate_rejected_count', 0)}",
        },
        {
            "node_id": "status_layout",
            "node_type": "status",
            "text": f"Layout {layout_report.get('status', 'disabled')}",
            "meta": f"atoms {layout_report.get('pdf_character_atom_count', 0)} / candidates {layout_report.get('layout_candidate_count', 0)}",
        },
    ]

    columns = [
        ("page", page_nodes),
        ("region", region_nodes),
        ("anchor", anchor_nodes),
        ("table", table_nodes),
        ("status", status_nodes),
    ]
    nodes: list[dict[str, Any]] = []
    x_positions = {"page": 24, "region": 244, "anchor": 464, "table": 684, "status": 904}
    w = 184
    h = 54
    y_gap = 70
    for kind, raw_nodes in columns:
        for index, raw_node in enumerate(raw_nodes):
            node_id = str(raw_node.get("node_id"))
            nodes.append(
                {
                    "node_id": node_id,
                    "kind": kind,
                    "title": _vdg_node_title(raw_node, kind),
                    "meta": _vdg_node_meta(raw_node),
                    "x": x_positions[kind],
                    "y": 34 + index * y_gap,
                    "w": w,
                    "h": h,
                }
            )
    node_by_id = {node["node_id"]: node for node in nodes}
    display_ids = set(node_by_id)
    edges = []
    preferred_edge_types = {"contains", "belongs_to_region", "belongs_to_table", "reading_order_next"}
    for edge in graph_edges:
        source = str(edge.get("source_node_id"))
        target = str(edge.get("target_node_id"))
        edge_type = str(edge.get("edge_type") or "")
        if source in display_ids and target in display_ids and edge_type in preferred_edge_types:
            edges.append({"source": source, "target": target, "edge_type": edge_type})
        if len(edges) >= 70:
            break
    page_ids = {str(node.get("node_id")) for node in page_nodes}
    for node in nodes:
        if node["kind"] in {"region", "anchor", "table"}:
            raw = raw_by_id.get(node["node_id"], {})
            page_id = f"page_{int(raw.get('page') or 1):04d}"
            if page_id in page_ids and not _edge_exists(edges, page_id, node["node_id"]):
                edges.append({"source": page_id, "target": node["node_id"], "edge_type": "page_contains"})
    for table in table_nodes:
        table_id = str(table.get("node_id"))
        for anchor in anchor_nodes:
            if _same_page(anchor, table) and _text_mentions_table(anchor, table):
                edges.append({"source": str(anchor.get("node_id")), "target": table_id, "edge_type": "table_anchor"})
                break
    if table_nodes:
        table_id = str(table_nodes[0].get("node_id"))
        for status in status_nodes:
            edges.append({"source": table_id, "target": str(status.get("node_id")), "edge_type": "status"})
    elif anchor_nodes:
        anchor_id = str(anchor_nodes[0].get("node_id"))
        for status in status_nodes:
            edges.append({"source": anchor_id, "target": str(status.get("node_id")), "edge_type": "status"})
    max_rows = max((len(items) for _kind, items in columns), default=1)
    return {
        "nodes": nodes,
        "edges": edges[:48],
        "node_by_id": node_by_id,
        "width": 1118,
        "height": max(330, 50 + max_rows * y_gap),
    }


def _select_vdg_nodes(nodes: list[dict[str, Any]], node_type: str, limit: int) -> list[dict[str, Any]]:
    selected = [node for node in nodes if node.get("node_type") == node_type]
    return selected[:limit]


def _select_anchor_nodes(
    nodes: list[dict[str, Any]],
    vdg_quality: dict[str, Any],
    vdg_consumption: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    important_ids = []
    for issue in vdg_quality.get("issues", []):
        if isinstance(issue, dict) and issue.get("node_id"):
            important_ids.append(str(issue["node_id"]))
    for key in ("unknown_important_nodes", "conflict_nodes"):
        for node in vdg_consumption.get(key, []):
            if isinstance(node, dict) and node.get("node_id"):
                important_ids.append(str(node["node_id"]))
    by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id")}
    selected: list[dict[str, Any]] = []
    for node_id in important_ids:
        node = by_id.get(node_id)
        if node and node.get("node_type") == "text_span" and node not in selected:
            selected.append(node)
    anchors = ("营养成分表", "配料", "产品标准", "净含量", "保质期", "贮存", "商品条码", "外箱条码")
    for node in nodes:
        if node.get("node_type") != "text_span" or node in selected:
            continue
        text = str(node.get("text") or "")
        if any(anchor in text for anchor in anchors) or node.get("status") in {"extracted", "conflict"}:
            selected.append(node)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _vdg_node_title(node: dict[str, Any], kind: str) -> str:
    if kind == "page":
        return f"Page {node.get('page')}"
    if kind == "table":
        return str(node.get("table_type") or node.get("text") or node.get("node_id"))
    if kind == "status":
        return str(node.get("text") or node.get("node_id"))
    return str(node.get("display_name") or node.get("text") or node.get("node_id"))


def _vdg_node_meta(node: dict[str, Any]) -> str:
    if node.get("meta"):
        return str(node.get("meta"))
    parts = []
    if node.get("node_type"):
        parts.append(str(node.get("node_type")))
    if node.get("page"):
        parts.append(f"p{node.get('page')}")
    if node.get("status"):
        parts.append(str(node.get("status")))
    return " | ".join(parts)


def _same_page(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("page") or "") == str(right.get("page") or "")


def _text_mentions_table(anchor: dict[str, Any], table: dict[str, Any]) -> bool:
    text = str(anchor.get("text") or "")
    table_text = str(table.get("text") or table.get("table_type") or "")
    return "营养成分表" in text or bool(table_text and table_text in text)


def _edge_exists(edges: list[dict[str, Any]], source: str, target: str) -> bool:
    return any(edge.get("source") == source and edge.get("target") == target for edge in edges)


def _render_missing_items(metrics: dict[str, Any]) -> str:
    items = []
    for item in metrics.get("missing_fields", []):
        if isinstance(item, dict):
            items.append(f"{item.get('semantic_key')}: {item.get('label')}")
    for item in metrics.get("missing_tables", []):
        if isinstance(item, dict):
            items.append(f"{item.get('table_type')}: {item.get('label')}")
    if not items:
        return '<h3>Missing Critical Items</h3><div class="muted">None</div>'
    return "<h3>Missing Critical Items</h3><div class=\"chips\">" + "".join(f'<span class="chip fail">{_e(item)}</span>' for item in items) + "</div>"


def _render_failed_gate_checks(metrics: dict[str, Any]) -> str:
    checks = metrics.get("failed_gate_checks", [])
    if not checks:
        return '<h3>Failed Quality Gates</h3><div class="muted">None</div>'
    chips = []
    for check in checks:
        chips.append(f'<span class="chip fail">{_e(check.get("check"))}: {_e(check.get("actual"))}</span>')
    return '<h3>Failed Quality Gates</h3><div class="chips">' + "".join(chips) + "</div>"


def _render_vdg_issues(metrics: dict[str, Any]) -> str:
    issues = [issue for issue in metrics.get("vdg_issues", []) if isinstance(issue, dict)]
    if not issues:
        return '<h3>VDG Issues</h3><div class="muted">None</div>'
    chips = []
    for issue in issues[:20]:
        chips.append(
            f'<span class="chip fail">{_e(issue.get("issue_type"))}: {_e(issue.get("severity"))}</span>'
        )
    return '<h3>VDG Issues</h3><div class="chips">' + "".join(chips) + "</div>"


def _render_vdg_nodes(metrics: dict[str, Any]) -> str:
    sections = []
    for title, key in (("VDG Unknown Important Nodes", "vdg_unknown_nodes"), ("VDG Conflict Nodes", "vdg_conflict_nodes")):
        nodes = [node for node in metrics.get(key, []) if isinstance(node, dict)]
        if not nodes:
            sections.append(f"<h3>{_e(title)}</h3><div class=\"muted\">None</div>")
            continue
        rows = []
        for node in nodes[:30]:
            bbox = node.get("bbox_normalized") if isinstance(node.get("bbox_normalized"), dict) else {}
            bbox_text = (
                f"{bbox.get('x1')},{bbox.get('y1')} - {bbox.get('x2')},{bbox.get('y2')}"
                if bbox
                else "missing"
            )
            rows.append(
                "<tr>"
                f"<td>{_e(node.get('node_id'))}</td>"
                f"<td>{_e(node.get('node_type'))}</td>"
                f"<td>{_e(node.get('page'))}</td>"
                f"<td>{_e(bbox_text)}</td>"
                f"<td>{_e(_truncate(str(node.get('text') or ''), 160))}</td>"
                "</tr>"
            )
        sections.append(
            f"<h3>{_e(title)}</h3>"
            "<table class=\"node-table\"><thead><tr><th>Node</th><th>Type</th><th>Page</th><th>BBox</th><th>Text</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    return "".join(sections)


def _render_scope_nodes(metrics: dict[str, Any]) -> str:
    sections = []
    for title, key in (("Scope Ignored Noise Nodes", "scope_ignored_nodes"), ("Scope Unknown Nodes", "scope_unknown_nodes")):
        nodes = [node for node in metrics.get(key, []) if isinstance(node, dict)]
        if not nodes:
            sections.append(f"<h3>{_e(title)}</h3><div class=\"muted\">None</div>")
            continue
        rows = []
        for node in nodes[:40]:
            bbox = node.get("bbox_normalized") if isinstance(node.get("bbox_normalized"), dict) else {}
            bbox_text = (
                f"{bbox.get('x1')},{bbox.get('y1')} - {bbox.get('x2')},{bbox.get('y2')}"
                if bbox
                else "missing"
            )
            rows.append(
                "<tr>"
                f"<td>{_e(node.get('node_id'))}</td>"
                f"<td>{_e(node.get('scope_status'))}</td>"
                f"<td>{_e(node.get('scope_category'))}</td>"
                f"<td>{_e(node.get('page'))}</td>"
                f"<td>{_e(bbox_text)}</td>"
                f"<td>{_e(_truncate(str(node.get('reason') or ''), 100))}</td>"
                f"<td>{_e(_truncate(str(node.get('text') or ''), 160))}</td>"
                "</tr>"
            )
        sections.append(
            f"<h3>{_e(title)}</h3>"
            "<table class=\"node-table\"><thead><tr><th>Node</th><th>Status</th><th>Category</th><th>Page</th><th>BBox</th><th>Reason</th><th>Text</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    return "".join(sections)


def _render_field_rows(fields: list[dict[str, Any]]) -> str:
    rows = []
    for row in fields:
        bbox_count = sum(1 for evidence in row["evidences"] if _normalized_bbox(evidence))
        status_class = _css_class(row["status"])
        reason = f'<div class="muted">{_e(row["reason"])}</div>' if row.get("reason") else ""
        refs = ", ".join(row["evidence_refs"])
        rows.append(
            f"""
            <tr data-field-id="{_e(row['field_id'])}">
              <td>{row['index']}</td>
              <td>
                <div><b>{_e(row['semantic_key'])}</b></div>
                <div class="muted">{_e(row['field_id'])}</div>
              </td>
              <td>{_e(row['value'])}{reason}<div class="muted">{_e(refs)}</div></td>
              <td><span class="status {status_class}">{_e(row['status'])}</span><div class="muted">risk: {_e(row['risk_level'])}</div></td>
              <td>{bbox_count}/{len(row['evidence_refs'])}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def _render_tables(tables: list[dict[str, Any]]) -> str:
    if not tables:
        return '<div class="muted">No tables extracted.</div>'
    return "\n".join(_render_table(table) for table in tables)


def _render_table(table: dict[str, Any]) -> str:
    status_class = _css_class(table["status"])
    fill_rate = f"{(table['non_empty_cell_count'] / table['cell_count'] * 100):.0f}%" if table["cell_count"] else "0%"
    head = f"""
      <div class="table-head">
        <div>
          <b>{_e(table['table_type'])}</b>
          <div class="muted">{_e(table['table_id'])} | {_e(table['title'])}</div>
          <div class="table-meta">
            <span class="chip">rows: {_e(table['row_count'])}</span>
            <span class="chip">cols: {_e(table['column_count'])}</span>
            <span class="chip">cell fill: {_e(fill_rate)}</span>
            <span class="chip">bbox: {_e(table['bbox_status'])}</span>
            <span class="chip">source: {_e(table['source'] or 'unknown')}</span>
          </div>
        </div>
        <div><span class="status {status_class}">{_e(table['status'])}</span><div class="muted">risk: {_e(table['risk_level'])}</div></div>
      </div>
    """
    if not table["rows"]:
        body = '<div class="section muted">No rows recovered.</div>'
    else:
        body = f"""
          <table class="nutrition-table">
            <thead>
              <tr>
                <th style="width: 42px;">#</th>
                <th style="width: 140px;">Row key</th>
                <th>Cells</th>
                <th style="width: 120px;">Evidence</th>
                <th style="width: 80px;">BBox</th>
              </tr>
            </thead>
            <tbody>{''.join(_render_table_row(table, row) for row in table['rows'])}</tbody>
          </table>
        """
    return f'<div class="table-block" data-table-id="{_e(table["table_id"])}">{head}{body}</div>'


def _render_table_row(table: dict[str, Any], row: dict[str, Any]) -> str:
    cell_text = "; ".join(
        f"{cell['column_id']}={cell['value'] or '<empty>'}"
        for cell in row["cells"]
    )
    refs = ", ".join(row["evidence_refs"])
    return f"""
      <tr data-table-id="{_e(table['table_id'])}" data-table-row-id="{_e(row['row_id'])}">
        <td>{row['index']}</td>
        <td>{_e(row['row_key'])}</td>
        <td>{_e(cell_text)}</td>
        <td><span class="muted">{_e(refs)}</span></td>
        <td>{row['bbox_count']}/{len(row['evidence_refs'])}</td>
      </tr>
    """


def _render_pages(pages: dict[int, dict[str, Any]], page_boxes: dict[int, list[dict[str, Any]]]) -> str:
    output = []
    for page, image in pages.items():
        boxes = page_boxes.get(page, [])
        output.append(
            f"""
            <div class="page-title">Page {page} | {len(boxes)} boxes</div>
            <div class="page" style="aspect-ratio: {image['width']} / {image['height']};">
              <img src="{image['src']}" alt="Page {page}">
              {_render_boxes(boxes, page)}
            </div>
            """
        )
    if not output:
        return '<div class="muted">No page images found.</div>'
    return "\n".join(output)


def _render_boxes(boxes: list[dict[str, Any]], page: int) -> str:
    output = []
    for index, box in enumerate(boxes, start=1):
        left = max(0, min(1, box["x1"])) * 100
        top = max(0, min(1, box["y1"])) * 100
        width = max(0, min(1, box["x2"]) - max(0, min(1, box["x1"]))) * 100
        height = max(0, min(1, box["y2"]) - max(0, min(1, box["y1"]))) * 100
        kind = box.get("kind", "field")
        title = _box_title(box)
        attrs = f'data-box-id="box_p{page:04d}_{index:04d}"'
        if kind == "field":
            attrs += f' data-field-id="{_e(box["field_id"])}"'
        elif kind == "table":
            attrs += f' data-table-id="{_e(box["table_id"])}"'
        elif kind == "table-row":
            attrs += f' data-table-id="{_e(box["table_id"])}" data-table-row-id="{_e(box["row_id"])}"'
        elif kind == "scope-node":
            attrs += f' data-scope-node-id="{_e(box["node_id"])}"'
        output.append(
            f"""
            <div class="box {_e(kind)}" {attrs} title="{_e(title)}"
                 style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;">
              <div class="box-label">{_e(title)}</div>
            </div>
            """
        )
    return "\n".join(output)


def _box_title(box: dict[str, Any]) -> str:
    if box.get("kind") == "field":
        return f"field: {box.get('semantic_key')} | {box.get('source_text') or box.get('value')}"
    if box.get("kind") == "table-row":
        return f"table row: {box.get('table_type')} | {box.get('source_text') or box.get('value')}"
    if box.get("kind") == "scope-node":
        return f"scope: {box.get('scope_status')} / {box.get('scope_category')} | {box.get('source_text') or box.get('value')}"
    return f"table: {box.get('table_type')} | {box.get('source_text') or box.get('value')}"


def _string_refs(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _css_class(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "..."


def _truncate_display(value: str, width_limit: int) -> str:
    if sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in value) <= width_limit:
        return value
    width = 0
    output = []
    for char in value:
        char_width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if width + char_width > width_limit - 3:
            break
        output.append(char)
        width += char_width
    return "".join(output) + "..."


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


if __name__ == "__main__":
    main()
