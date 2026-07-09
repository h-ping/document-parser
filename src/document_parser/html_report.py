from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any

from .models import to_jsonable


def write_result_preview_html(
    result: Any,
    output_path: Path,
    artifact_root: Path | None = None,
    workbook_structure: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_result_preview_html(
            result,
            output_path=output_path,
            artifact_root=artifact_root,
            workbook_structure=workbook_structure,
        ),
        encoding="utf-8",
    )


def build_result_preview_html(
    result: Any,
    output_path: Path | None = None,
    artifact_root: Path | None = None,
    workbook_structure: dict[str, Any] | None = None,
) -> str:
    data = to_jsonable(result)
    metadata = _as_dict(data.get("metadata"))
    standard_artifacts = _as_dict(metadata.get("standard_artifacts"))
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in _as_list(data.get("evidence"))
        if isinstance(item, dict) and item.get("evidence_id")
    }
    items = [item for item in _as_list(standard_artifacts.get("standard_items")) if isinstance(item, dict)]
    tables = _as_list(standard_artifacts.get("tables"))
    groups = _as_list(standard_artifacts.get("field_groups"))
    quality_report = _as_dict(standard_artifacts.get("quality_report"))
    contract = _as_dict(metadata.get("output_contract_validation_report"))
    document = _as_dict(data.get("document"))
    job = _as_dict(data.get("job"))
    source_layers = _as_dict(metadata.get("source_layers"))
    anchors = _item_anchor_map(items, evidence_by_id)
    artifact_root = artifact_root or (output_path.parent if output_path else Path("."))
    output_dir = output_path.parent if output_path else artifact_root

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_html(document.get('file_name') or '结构化结果预览')}</title>",
            "<style>",
            _CSS,
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>结构化标签内容预览</h1>",
            _summary_section(job, document, quality_report, contract, items, tables),
            '<section class="compare-shell">',
            _standard_document_pane(document, source_layers, items, evidence_by_id, anchors, output_dir, artifact_root, workbook_structure),
            _extracted_items_pane(items, groups, tables, evidence_by_id, anchors),
            "</section>",
            _tables_section(tables),
            _field_groups_section(groups),
            _validation_section(contract),
            "</main>",
            "<script>",
            _interaction_script(anchors),
            "</script>",
            "</body>",
            "</html>",
        ]
    )


def _summary_section(
    job: dict[str, Any],
    document: dict[str, Any],
    quality_report: dict[str, Any],
    contract: dict[str, Any],
    items: list[dict[str, Any]],
    tables: list[Any],
) -> str:
    rows = [
        ("文件", document.get("file_name")),
        ("任务类型", job.get("job_type")),
        ("任务状态", job.get("status")),
        ("解析状态", document.get("parse_status")),
        ("质量状态", quality_report.get("status")),
        ("输出契约", f"{contract.get('status')} / failed={contract.get('failed_count', 0)}"),
        ("标准项数量", len(items)),
        ("表格数量", len(tables)),
    ]
    cells = "".join(f"<div><span>{_html(label)}</span><strong>{_html(value)}</strong></div>" for label, value in rows)
    return f'<section class="summary">{cells}</section>'


def _standard_document_pane(
    document: dict[str, Any],
    source_layers: dict[str, Any],
    items: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
    output_dir: Path,
    artifact_root: Path,
    workbook_structure: dict[str, Any] | None,
) -> str:
    if str(document.get("file_name") or "").lower().endswith(".xlsx"):
        body = _xlsx_source_view(source_layers, evidence_by_id, workbook_structure)
    else:
        body = _pdf_source_view(document, items, anchors, output_dir, artifact_root)
        if not body:
            body = _generic_source_view(source_layers, evidence_by_id)
    return (
        '<div class="pane doc-pane">'
        '<div class="pane-header"><h2>标准文档</h2><span class="hint">点击右侧提取项后，这里会高亮对应 bbox 或 source node</span></div>'
        f"{body}"
        "</div>"
    )


def _pdf_source_view(
    document: dict[str, Any],
    items: list[dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
    output_dir: Path,
    artifact_root: Path,
) -> str:
    page_count = int(document.get("page_count") or 0)
    if page_count <= 0:
        return ""
    pages = []
    for page in range(1, page_count + 1):
        image_path = artifact_root / "page_images" / f"page_{page:03d}.png"
        image_src = _relative_path(output_dir, image_path)
        boxes = []
        for item in items:
            item_id = str(item.get("id") or "")
            for bbox in anchors.get(item_id, {}).get("boxes", []):
                if int(bbox.get("page") or 1) != page:
                    continue
                normalized = _as_dict(bbox.get("bbox_normalized"))
                if not normalized:
                    continue
                boxes.append(
                    '<button class="bbox-box" type="button" '
                    f'data-item-id="{_attr(item_id)}" '
                    f'title="{_attr(item.get("label") or item.get("field"))}" '
                    "style="
                    f'"left:{_pct(normalized.get("x1"))};top:{_pct(normalized.get("y1"))};'
                    f'width:{_pct(float(normalized.get("x2", 0)) - float(normalized.get("x1", 0)))};'
                    f'height:{_pct(float(normalized.get("y2", 0)) - float(normalized.get("y1", 0)))};"'
                    "></button>"
                )
        pages.append(
            f'<article class="page-frame" data-page="{page}">'
            f"<h3>Page {page}</h3>"
            '<div class="page-image-wrap">'
            f'<img src="{_attr(image_src)}" alt="Page {page}">'
            f"{''.join(boxes)}"
            "</div>"
            "</article>"
        )
    return "".join(pages)


def _xlsx_source_view(
    source_layers: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    workbook_structure: dict[str, Any] | None,
) -> str:
    if workbook_structure:
        sheets = _as_dict(workbook_structure.get("sheets"))
        sections = []
        for sheet_name in workbook_structure.get("read_sheets", []):
            sheet = _as_dict(sheets.get(sheet_name))
            columns = [
                column
                for column in _as_list(sheet.get("display_columns"))
                if isinstance(column, dict)
            ]
            rows = [
                row
                for row in _as_list(sheet.get("rows"))
                if isinstance(row, dict)
            ]
            if not columns:
                continue
            header = "".join(f"<th>{_html(column.get('header'))}</th>" for column in columns)
            body_rows = []
            for row in rows:
                cells = _as_dict(row.get("cells"))
                body_cells = []
                for column in columns:
                    cell = _as_dict(cells.get(str(column.get("canonical_header"))))
                    body_cells.append(
                        '<td class="source-node" '
                        f'data-source-node="{_attr(cell.get("source_node_id"))}">'
                        f'{_html_multiline(cell.get("text"))}</td>'
                    )
                body_rows.append(f"<tr>{''.join(body_cells)}</tr>")
            sections.append(
                '<article class="sheet-card">'
                f"<h3>{_html(sheet_name)}</h3>"
                '<div class="table-wrap nested"><table>'
                f"<thead><tr>{header}</tr></thead>"
                f"<tbody>{''.join(body_rows)}</tbody>"
                "</table></div>"
                "</article>"
            )
        if sections:
            return "".join(sections)

    spans = [
        span
        for span in _as_list(source_layers.get("spans"))
        if isinstance(span, dict) and str(span.get("span_id", "")).startswith("xlsx:")
    ]
    if not spans:
        return _generic_source_view(source_layers, evidence_by_id)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        sheet, coord = _xlsx_sheet_coord(str(span.get("span_id")))
        grouped.setdefault(sheet, []).append({**span, "coord": coord})
    sections = []
    for sheet, sheet_spans in grouped.items():
        rows = []
        for span in sorted(sheet_spans, key=lambda item: _coord_sort_key(str(item.get("coord") or ""))):
            rows.append(
                '<tr class="source-node" '
                f'data-source-node="{_attr(span.get("span_id"))}">'
                f"<td><code>{_html(span.get('coord'))}</code></td>"
                f'<td class="text-cell">{_html_multiline(span.get("text"))}</td>'
                "</tr>"
            )
        sections.append(
            '<article class="sheet-card">'
            f"<h3>{_html(sheet)}</h3>"
            '<div class="table-wrap nested"><table>'
            "<thead><tr><th>单元格</th><th>原文</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table></div>"
            "</article>"
        )
    return "".join(sections)


def _generic_source_view(source_layers: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> str:
    spans = [span for span in _as_list(source_layers.get("spans")) if isinstance(span, dict)]
    if not spans:
        spans = [
            {
                "span_id": node_id,
                "page": evidence.get("page"),
                "text": evidence.get("source_text"),
            }
            for evidence in evidence_by_id.values()
            for node_id in _as_list(evidence.get("source_node_ids"))
        ]
    rows = []
    for span in spans:
        rows.append(
            '<tr class="source-node" '
            f'data-source-node="{_attr(span.get("span_id"))}">'
            f"<td>{_html(span.get('page'))}</td>"
            f"<td><code>{_html(span.get('span_id'))}</code></td>"
            f'<td class="text-cell">{_html_multiline(span.get("text"))}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap nested"><table>'
        "<thead><tr><th>页</th><th>source node</th><th>原文</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _extracted_items_pane(
    items: list[dict[str, Any]],
    groups: list[Any],
    tables: list[Any],
    evidence_by_id: dict[str, dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
) -> str:
    cards = []
    grouped_ids = {
        str(group.get("group_id"))
        for group in groups
        if isinstance(group, dict) and group.get("group_id") and group.get("group_type") in {"principal", "manufacturer", "distributor", "enterprise", "content_item"}
    }
    table_item_ids = {str(item.get("id")) for item in items if item.get("table_id") or item.get("semantic_key") == "product.nutrition_table"}
    visible_items = [
        item
        for item in items
        if str(item.get("id")) not in table_item_ids and not (item.get("group_id") and str(item.get("group_id")) in grouped_ids)
    ]
    index = 1
    for item in visible_items:
        cards.append(_standard_item_card(index, item, evidence_by_id, anchors))
        index += 1
    for group in groups:
        if not isinstance(group, dict) or str(group.get("group_id")) not in grouped_ids:
            continue
        cards.append(_group_extract_card(index, group, items, anchors))
        index += 1
    for table in tables:
        if not isinstance(table, dict):
            continue
        cards.append(_table_extract_card(index, table, items, anchors))
        index += 1
    return (
        '<div class="pane items-pane">'
        '<div class="pane-header"><h2>提取项</h2><span class="hint">点击任一项定位左侧原文</span></div>'
        '<div class="extract-list">'
        f"{''.join(cards)}"
        "</div>"
        "</div>"
    )


def _standard_item_card(
    index: int,
    item: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
) -> str:
    item_id = str(item.get("id") or f"item_{index}")
    evidence_items = _item_evidence(item, evidence_by_id)
    source_nodes = anchors.get(item_id, {}).get("nodes", [])
    bbox_count = len(anchors.get(item_id, {}).get("boxes", []))
    return (
        '<button class="extract-card" type="button" '
        f'data-item-id="{_attr(item_id)}">'
        '<span class="extract-index">'
        f"{index}"
        "</span>"
        '<span class="extract-body">'
        f"<strong>{_html(item.get('label') or item.get('field'))}</strong>"
        f"<small><code>{_html(item.get('field'))}</code> · {_html(item.get('semantic_key'))}</small>"
        f'<span class="extract-text">{_html_multiline(item.get("text"))}</span>'
        f'<small>source nodes: {len(source_nodes)} · bbox: {bbox_count} · comparison: {_yes_no(item.get("comparison_required"))}</small>'
        f"{_source_details(item, evidence_items)}"
        "</span>"
        "</button>"
    )


def _group_extract_card(
    index: int,
    group: dict[str, Any],
    items: list[dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
) -> str:
    group_id = str(group.get("group_id") or f"group_{index}")
    anchor_id = f"group:{group_id}"
    group_items = _items_by_field_ref(group, items)
    rows = []
    for field in _as_list(group.get("fields")):
        if not isinstance(field, dict):
            continue
        linked_item = group_items.get(str(field.get("standard_item_id")))
        label = linked_item.get("label") if linked_item else field.get("semantic_key")
        rows.append(
            "<tr>"
            f"<th>{_html(label)}</th>"
            f'<td class="text-cell">{_html_multiline(field.get("text"))}</td>'
            "</tr>"
        )
    source_nodes = anchors.get(anchor_id, {}).get("nodes", [])
    bbox_count = len(anchors.get(anchor_id, {}).get("boxes", []))
    return (
        '<button class="extract-card group-card" type="button" '
        f'data-item-id="{_attr(anchor_id)}">'
        f'<span class="extract-index">{index}</span>'
        '<span class="extract-body">'
        f"<strong>{_html(_group_title(group))}</strong>"
        f"<small><code>{_html(group_id)}</code> · {_html(group.get('group_type'))}</small>"
        '<span class="extract-table mini-table"><table><tbody>'
        f"{''.join(rows)}"
        "</tbody></table></span>"
        f"<small>source nodes: {len(source_nodes)} · bbox: {bbox_count}</small>"
        f"{_json_details('group source', {'group_id': group_id, 'linked_table_ids': group.get('linked_table_ids', []), 'field_count': len(rows)})}"
        "</span>"
        "</button>"
    )


def _table_extract_card(
    index: int,
    table: dict[str, Any],
    items: list[dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
) -> str:
    table_id = str(table.get("table_id") or f"table_{index}")
    anchor_id = f"table:{table_id}"
    columns = [column for column in _as_list(table.get("columns")) if isinstance(column, dict)]
    header = "".join(f"<th>{_html(column.get('name') or column.get('column_id'))}</th>" for column in columns)
    rows = []
    for table_row in _as_list(table.get("rows")):
        if not isinstance(table_row, dict):
            continue
        cells_by_column = {
            str(cell.get("column_id")): cell
            for cell in _as_list(table_row.get("cells"))
            if isinstance(cell, dict)
        }
        row_cells = "".join(
            f'<td>{_html_multiline(_as_dict(cells_by_column.get(str(column.get("column_id")))).get("raw_value", ""))}</td>'
            for column in columns
        )
        rows.append(f"<tr>{row_cells}</tr>")
    footnotes = _as_list(table.get("footnotes"))
    source_nodes = anchors.get(anchor_id, {}).get("nodes", [])
    bbox_count = len(anchors.get(anchor_id, {}).get("boxes", []))
    linked_item = next((item for item in items if item.get("table_id") == table_id), None)
    return (
        '<button class="extract-card table-extract-card" type="button" '
        f'data-item-id="{_attr(anchor_id)}">'
        f'<span class="extract-index">{index}</span>'
        '<span class="extract-body">'
        f"<strong>{_html(table.get('title') or '营养成分表')}</strong>"
        f"<small><code>{_html(table_id)}</code> · {_html(table.get('table_type'))}</small>"
        '<span class="extract-table nutrition-mini"><table>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></span>"
        f"{'<span class=\"extract-footnote\">' + _html_multiline(chr(10).join(str(item) for item in footnotes)) + '</span>' if footnotes else ''}"
        f"<small>source nodes: {len(source_nodes)} · bbox: {bbox_count} · standard item: {_html(linked_item.get('id') if linked_item else '')}</small>"
        f"{_json_details('table source', _table_source_summary(table))}"
        "</span>"
        "</button>"
    )


def _tables_section(tables: list[Any]) -> str:
    if not tables:
        return '<section><h2>表格</h2><p class="empty">无表格。</p></section>'
    blocks = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        columns = [column for column in _as_list(table.get("columns")) if isinstance(column, dict)]
        header = "".join(f"<th>{_html(column.get('name') or column.get('column_id'))}</th>" for column in columns)
        rows = []
        for table_row in _as_list(table.get("rows")):
            if not isinstance(table_row, dict):
                continue
            cells_by_column = {
                str(cell.get("column_id")): cell
                for cell in _as_list(table_row.get("cells"))
                if isinstance(cell, dict)
            }
            row_cells = "".join(
                f'<td class="text-cell">{_html_multiline(_as_dict(cells_by_column.get(str(column.get("column_id")))).get("raw_value", ""))}</td>'
                for column in columns
            )
            rows.append(f"<tr><td>{_html(table_row.get('row_key'))}</td>{row_cells}</tr>")
        blocks.append(
            '<article class="table-card">'
            f"<h3>{_html(table.get('title'))}</h3>"
            f"<p><code>{_html(table.get('table_id'))}</code> · {_html(table.get('table_type'))} · status={_html(table.get('status'))}</p>"
            f"{_json_details('table source / bbox', _table_source_summary(table))}"
            '<div class="table-wrap nested"><table>'
            f"<thead><tr><th>row_key</th>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table></div>"
            "</article>"
        )
    return f"<section><h2>表格</h2>{''.join(blocks)}</section>"


def _field_groups_section(groups: list[Any]) -> str:
    if not groups:
        return '<section><h2>字段分组</h2><p class="empty">无重复分组。</p></section>'
    rows = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        fields = [
            f"{field.get('semantic_key')}: {field.get('text')}"
            for field in _as_list(group.get("fields"))
            if isinstance(field, dict)
        ]
        rows.append(
            "<tr>"
            f"<td><code>{_html(group.get('group_id'))}</code></td>"
            f"<td>{_html(group.get('group_type'))}</td>"
            f"<td>{_html(group.get('instance_index'))}</td>"
            f'<td class="text-cell">{_html_multiline(chr(10).join(fields))}</td>'
            f"<td>{_html(', '.join(str(item) for item in _as_list(group.get('linked_table_ids'))))}</td>"
            "</tr>"
        )
    return (
        '<section><h2>字段分组</h2><div class="table-wrap"><table>'
        "<thead><tr><th>group_id</th><th>类型</th><th>序号</th><th>字段</th><th>关联表</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _validation_section(contract: dict[str, Any]) -> str:
    failed = [
        check
        for check in _as_list(contract.get("checks"))
        if isinstance(check, dict) and check.get("result") == "failed"
    ]
    if not failed:
        return '<section><h2>契约校验</h2><p class="pass">全部通过。</p></section>'
    rows = "".join(
        "<tr>"
        f"<td>{_html(check.get('check_type'))}</td>"
        f"<td>{_html(check.get('target'))}</td>"
        f"<td>{_html(check.get('message'))}</td>"
        f"<td>{_json_details('details', check.get('details'))}</td>"
        "</tr>"
        for check in failed
    )
    return (
        '<section><h2>契约校验</h2><div class="table-wrap"><table>'
        "<thead><tr><th>类型</th><th>目标</th><th>消息</th><th>详情</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def _item_anchor_map(items: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    anchor_map: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        nodes: list[str] = []
        boxes: list[dict[str, Any]] = []
        source = _as_dict(item.get("source"))
        if source.get("span_id"):
            nodes.append(str(source["span_id"]))
        if source.get("bbox_normalized"):
            boxes.append(
                {
                    "page": source.get("page") or 1,
                    "bbox_normalized": source.get("bbox_normalized"),
                    "bbox_pdf": source.get("bbox_pdf"),
                }
            )
        for evidence in _item_evidence(item, evidence_by_id):
            nodes.extend(str(node_id) for node_id in _as_list(evidence.get("source_node_ids")))
            if evidence.get("bbox_normalized"):
                boxes.append(
                    {
                        "page": evidence.get("page") or 1,
                        "bbox_normalized": evidence.get("bbox_normalized"),
                        "bbox_pdf": evidence.get("bbox_pdf"),
                    }
                )
        anchor = {"nodes": _ordered_unique(nodes), "boxes": boxes}
        anchor_map[item_id] = anchor
        if item.get("group_id"):
            _merge_anchor(anchor_map, f"group:{item.get('group_id')}", anchor)
        if item.get("table_id"):
            _merge_anchor(anchor_map, f"table:{item.get('table_id')}", anchor)
    return anchor_map


def _merge_anchor(anchor_map: dict[str, dict[str, Any]], key: str, anchor: dict[str, Any]) -> None:
    target = anchor_map.setdefault(key, {"nodes": [], "boxes": []})
    target["nodes"] = _ordered_unique([*target.get("nodes", []), *anchor.get("nodes", [])])
    target["boxes"].extend(anchor.get("boxes", []))


def _items_by_field_ref(group: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    item_by_id = {str(item.get("id")): item for item in items if item.get("id")}
    refs = [
        str(field.get("standard_item_id"))
        for field in _as_list(group.get("fields"))
        if isinstance(field, dict) and field.get("standard_item_id")
    ]
    return {ref: item_by_id[ref] for ref in refs if ref in item_by_id}


def _group_title(group: dict[str, Any]) -> str:
    group_type = str(group.get("group_type") or "")
    label = {
        "principal": "委托方信息",
        "manufacturer": "受委托方/生产者信息",
        "distributor": "经销方信息",
        "enterprise": "企业信息",
        "content_item": "内容物信息",
    }.get(group_type, group_type or "字段分组")
    container_text = group.get("container_text")
    return f"{label}：{container_text}" if container_text else label


def _item_evidence(item: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [evidence_by_id[ref] for ref in (str(ref) for ref in _as_list(item.get("evidence_refs"))) if ref in evidence_by_id]


def _source_details(item: dict[str, Any], evidence_items: list[dict[str, Any]]) -> str:
    source = _as_dict(item.get("source"))
    source_summary = {
        "page": source.get("page"),
        "section": source.get("section"),
        "span_id": source.get("span_id"),
        "bbox": source.get("bbox"),
        "bbox_pdf": source.get("bbox_pdf"),
        "bbox_normalized": source.get("bbox_normalized"),
        "source_node_ids": [node_id for evidence in evidence_items for node_id in _as_list(evidence.get("source_node_ids"))],
        "evidence_refs": item.get("evidence_refs", []),
        "extraction_method": item.get("extraction_method"),
    }
    return _json_details("source / bbox", source_summary)


def _table_source_summary(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "table_id": table.get("table_id"),
        "bbox_status": table.get("bbox_status"),
        "bbox_pdf": table.get("bbox_pdf"),
        "bbox_normalized": table.get("bbox_normalized"),
        "source_span_ids": table.get("source_span_ids", []),
        "evidence_refs": table.get("evidence_refs", []),
    }


def _interaction_script(anchors: dict[str, dict[str, Any]]) -> str:
    anchor_json = json.dumps(anchors, ensure_ascii=False)
    return f"""
const ITEM_ANCHORS = {anchor_json};
function clearActive() {{
  document.querySelectorAll('.extract-card.active, .source-node.active, .bbox-box.active').forEach(el => el.classList.remove('active'));
}}
function activateItem(itemId) {{
  clearActive();
  const card = document.querySelector(`.extract-card[data-item-id="${{CSS.escape(itemId)}}"]`);
  if (card) card.classList.add('active');
  const anchors = ITEM_ANCHORS[itemId] || {{}};
  (anchors.nodes || []).forEach(nodeId => {{
    document.querySelectorAll(`.source-node[data-source-node="${{CSS.escape(nodeId)}}"]`).forEach(el => el.classList.add('active'));
  }});
  document.querySelectorAll(`.bbox-box[data-item-id="${{CSS.escape(itemId)}}"]`).forEach(el => el.classList.add('active'));
  const target = document.querySelector(`.bbox-box[data-item-id="${{CSS.escape(itemId)}}"]`) ||
    (anchors.nodes || []).map(nodeId => document.querySelector(`.source-node[data-source-node="${{CSS.escape(nodeId)}}"]`)).find(Boolean);
  if (target) target.scrollIntoView({{behavior: 'smooth', block: 'center', inline: 'center'}});
}}
document.querySelectorAll('.extract-card').forEach(card => {{
  card.addEventListener('click', () => activateItem(card.dataset.itemId));
}});
document.querySelectorAll('.bbox-box').forEach(box => {{
  box.addEventListener('click', event => {{
    event.stopPropagation();
    activateItem(box.dataset.itemId);
  }});
}});
const firstCard = document.querySelector('.extract-card');
if (firstCard) activateItem(firstCard.dataset.itemId);
"""


def _xlsx_sheet_coord(span_id: str) -> tuple[str, str]:
    match = re.match(r"^xlsx:(.+)!([A-Z]+[0-9]+)$", span_id)
    if not match:
        return "Workbook", span_id
    return match.group(1), match.group(2)


def _coord_sort_key(coord: str) -> tuple[int, int, str]:
    match = re.match(r"^([A-Z]+)([0-9]+)$", coord)
    if not match:
        return (10**9, 10**9, coord)
    column = 0
    for char in match.group(1):
        column = column * 26 + ord(char) - ord("A") + 1
    return (int(match.group(2)), column, coord)


def _relative_path(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, from_dir).replace(os.sep, "/")


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return f"{max(0.0, min(100.0, number * 100.0)):.4f}%"


def _json_details(summary: str, payload: Any) -> str:
    return (
        f"<details><summary>{_html(summary)}</summary>"
        f"<pre>{_html(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))}</pre>"
        "</details>"
    )


def _yes_no(value: Any) -> str:
    return "是" if value is True else "否" if value is False else _html(value)


def _html(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _attr(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _html_multiline(value: Any) -> str:
    return _html(value).replace("\n", "<br>")


def _ordered_unique(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


_CSS = """
:root {
  color-scheme: light;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7f9;
  color: #20242a;
}
body {
  margin: 0;
}
main {
  max-width: 1680px;
  margin: 0 auto;
  padding: 20px;
}
h1 {
  margin: 0 0 14px;
  font-size: 24px;
}
h2 {
  margin: 0;
  font-size: 17px;
}
h3 {
  margin: 0 0 8px;
  font-size: 15px;
}
section, .table-card, .sheet-card {
  border: 1px solid #d9dee7;
  border-radius: 8px;
  background: #fff;
}
section {
  margin-top: 16px;
  padding: 14px;
}
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
}
.summary div {
  border: 1px solid #e2e6ee;
  border-radius: 6px;
  padding: 9px;
  background: #fbfcfe;
}
.summary span, .hint, small {
  color: #667085;
}
.summary span {
  display: block;
  font-size: 12px;
}
.summary strong {
  display: block;
  margin-top: 4px;
  font-size: 13px;
  word-break: break-word;
}
.compare-shell {
  display: grid;
  grid-template-columns: minmax(360px, 1fr) minmax(360px, 0.92fr);
  gap: 14px;
  padding: 0;
  border: 0;
  background: transparent;
}
.pane {
  min-height: 72vh;
  max-height: 82vh;
  overflow: auto;
  border: 1px solid #d9dee7;
  border-radius: 8px;
  background: #fff;
}
.pane-header {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  padding: 12px;
  border-bottom: 1px solid #e7ebf1;
  background: #fff;
}
.doc-pane {
  padding-bottom: 12px;
}
.items-pane {
  display: flex;
  flex-direction: column;
}
.extract-list {
  padding: 10px;
}
.extract-card {
  width: 100%;
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 10px;
  margin: 0 0 8px;
  padding: 10px;
  border: 1px solid #e0e5ee;
  border-radius: 7px;
  background: #fff;
  color: inherit;
  text-align: left;
  cursor: pointer;
}
.extract-card:hover {
  border-color: #9db5ff;
}
.extract-card.active {
  border-color: #1f5eff;
  background: #f4f7ff;
  box-shadow: 0 0 0 2px rgba(31, 94, 255, 0.14);
}
.extract-index {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #edf2ff;
  color: #1f5eff;
  font-weight: 700;
}
.extract-body {
  display: grid;
  gap: 4px;
}
.extract-text {
  font-size: 13px;
  line-height: 1.45;
}
.page-frame {
  margin: 12px;
  padding: 10px;
  border: 1px solid #e0e5ee;
  border-radius: 8px;
  background: #fbfcfe;
}
.page-image-wrap {
  position: relative;
  width: max-content;
  max-width: 100%;
}
.page-image-wrap img {
  display: block;
  max-width: 100%;
  height: auto;
  border: 1px solid #cfd6e3;
  background: #fff;
}
.bbox-box {
  position: absolute;
  display: none;
  border: 2px solid #ff3b30;
  background: rgba(255, 59, 48, 0.14);
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.75);
  cursor: pointer;
}
.bbox-box.active {
  display: block;
}
.sheet-card {
  margin: 12px;
  padding: 10px;
}
.source-node.active {
  background: #fff3cd;
  outline: 2px solid #ffbf00;
  outline-offset: -2px;
}
.table-wrap {
  overflow: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th, td {
  vertical-align: top;
  text-align: left;
  border-bottom: 1px solid #e7ebf1;
  padding: 8px;
}
th {
  position: sticky;
  top: 0;
  background: #f0f3f8;
  z-index: 1;
}
code {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
}
.text-cell {
  min-width: 210px;
  max-width: 520px;
  word-break: break-word;
}
details {
  max-width: 420px;
}
summary {
  cursor: pointer;
  color: #1f5eff;
}
pre {
  overflow: auto;
  max-height: 280px;
  padding: 8px;
  border-radius: 6px;
  background: #f5f7fb;
  font-size: 12px;
  line-height: 1.45;
}
.table-card {
  margin-top: 10px;
  padding: 12px;
}
.nested th {
  position: static;
}
.pass {
  margin-bottom: 0;
  color: #087443;
}
.empty {
  color: #667085;
}
@media (max-width: 980px) {
  .compare-shell {
    grid-template-columns: 1fr;
  }
  .pane {
    max-height: none;
  }
}
"""
