from __future__ import annotations

import importlib.util
import re
from typing import Any

from .models import TextSpan
from .utils import stable_id


NUTRITION_ROW_LABELS = (
    "能量",
    "蛋白质",
    "蛋⽩质",
    "脂肪",
    "饱和脂肪",
    "--饱和脂肪",
    "—饱和脂肪",
    "碳水化合物",
    "碳⽔化合物",
    "糖",
    "--糖",
    "—糖",
    "钠",
    "钙",
)


def build_table_parser_outputs(spans: list[TextSpan], pdf_path: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    table_layers = build_table_layers(spans, pdf_path)
    table_quality_report = build_table_quality_report(table_layers)
    return table_layers, table_quality_report


def build_table_layers(spans: list[TextSpan], pdf_path: str | None = None) -> dict[str, Any]:
    tables = []
    parsers = ["text_span_nutrition"]
    title_indexes = [index for index, span in enumerate(spans) if "营养成分表" in span.text]
    for table_index, title_index in enumerate(title_indexes, start=1):
        title_span = spans[title_index]
        row_spans = _nutrition_row_spans(spans, title_index)
        rows = []
        columns = _columns_from_rows(row_spans)
        for row_index, row_span in enumerate(row_spans, start=1):
            row_text = row_span.text.strip()
            if "项目" in row_text and row_index == 1:
                rows.append(_row_from_cells(row_index, "header", _header_cells(row_text, row_span)))
                continue
            if not _looks_like_nutrition_row(row_text):
                continue
            item, amount, nrv = _split_nutrition_row(row_text)
            rows.append(
                _row_from_cells(
                    row_index,
                    "data",
                    [
                        _cell(row_index, 0, item, row_span),
                        _cell(row_index, 1, amount, row_span),
                        _cell(row_index, 2, nrv, row_span),
                    ],
                    row_key=_nutrition_row_key(item),
                )
            )

        source_span_ids = [title_span.span_id] + [row_span.span_id for row_span in row_spans]
        tables.append(
            {
                "table_layer_id": stable_id("tl_tbl", table_index),
                "parser": "text_span_nutrition",
                "table_type": "nutrition_facts",
                "page": title_span.page,
                "title": title_span.text,
                "columns": columns,
                "rows": rows,
                "source_span_ids": source_span_ids,
                **_table_bbox_metadata([title_span] + row_spans),
                "confidence": 0.96 if any(row["row_type"] == "data" for row in rows) else 0.50,
            }
        )

    pdfplumber_tables, pdfplumber_issues = _extract_pdfplumber_tables(pdf_path)
    if pdfplumber_tables or pdfplumber_issues:
        parsers.append("pdfplumber")
    tables.extend(pdfplumber_tables)

    return {
        "parsers": parsers,
        "tables": tables,
        "parser_issues": pdfplumber_issues,
    }


def build_table_quality_report(table_layers: dict[str, Any]) -> dict[str, Any]:
    dependency_preflight = _dependency_preflight()
    tables = table_layers.get("tables", [])
    issues = []
    primary_tables = [table for table in tables if table.get("parser") == "text_span_nutrition"]
    pdfplumber_tables = [table for table in tables if table.get("parser") == "pdfplumber"]
    for table in primary_tables:
        data_rows = [row for row in table.get("rows", []) if row.get("row_type") == "data"]
        if not data_rows:
            issues.append(
                {
                    "issue_type": "nutrition_table_rows_incomplete",
                    "table_layer_id": table["table_layer_id"],
                    "message": "营养成分表没有恢复出数据行。",
                    "severity": "high",
                }
            )
        if len(table.get("columns", [])) < 3:
            issues.append(
                {
                    "issue_type": "column_count_unstable",
                    "table_layer_id": table["table_layer_id"],
                    "message": "营养成分表列数不足。",
                    "severity": "high",
                }
            )

    status = "pass" if not issues else "review_required"
    return {
        "status": status,
        "dependency_preflight": dependency_preflight,
        "parser_agreement": {
            "status": _parser_agreement_status(primary_tables, pdfplumber_tables, dependency_preflight),
            "available_parsers": _available_parsers(table_layers, dependency_preflight),
            "unavailable_parsers": [
                name for name, info in dependency_preflight.items() if name in {"pdfplumber", "camelot"} and not info["available"]
            ],
            "text_span_table_count": len(primary_tables),
            "pdfplumber_table_count": len(pdfplumber_tables),
        },
        "table_count": len(primary_tables),
        "candidate_table_count": len(tables),
        "nutrition_table_count": sum(1 for table in primary_tables if table.get("table_type") == "nutrition_facts"),
        "parser_issues": table_layers.get("parser_issues", []),
        "issues": issues,
        "issue_count": len(issues),
    }


def _dependency_preflight() -> dict[str, dict[str, Any]]:
    modules = {
        "pdfplumber": "pdfplumber",
        "camelot": "camelot",
        "pandas": "pandas",
        "opencv_python": "cv2",
    }
    return {
        name: {
            "available": importlib.util.find_spec(module) is not None,
            "module": module,
            "required_for": "optional_parser" if name in {"pdfplumber", "camelot"} else "optional_parser_dependency",
        }
        for name, module in modules.items()
    }


def _extract_pdfplumber_tables(pdf_path: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pdf_path:
        return [], []
    if importlib.util.find_spec("pdfplumber") is None:
        return [], [
            {
                "parser": "pdfplumber",
                "issue_type": "parser_unavailable",
                "message": "pdfplumber is not installed.",
                "severity": "info",
            }
        ]

    try:
        import pdfplumber  # type: ignore
    except Exception as exc:  # pragma: no cover - import environment dependent
        return [], [
            {
                "parser": "pdfplumber",
                "issue_type": "parser_import_failed",
                "message": exc.__class__.__name__,
                "severity": "medium",
            }
        ]

    tables: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for raw_table_index, raw_table in enumerate(page.extract_tables() or [], start=1):
                    rows = _pdfplumber_rows(raw_table)
                    if not rows:
                        continue
                    tables.append(
                        {
                            "table_layer_id": stable_id("pdfplumber_tbl", len(tables) + 1),
                            "parser": "pdfplumber",
                            "table_type": _pdfplumber_table_type(raw_table),
                            "page": page_index,
                            "title": _pdfplumber_title(raw_table, page_index, raw_table_index),
                            "columns": _pdfplumber_columns(raw_table),
                            "rows": rows,
                            "source_span_ids": [],
                            "bbox_status": "missing",
                            "confidence": 0.70,
                            "raw_table_index": raw_table_index,
                        }
                    )
    except Exception as exc:
        issues.append(
            {
                "parser": "pdfplumber",
                "issue_type": "parser_failed",
                "message": exc.__class__.__name__,
                "severity": "medium",
            }
        )

    return tables, issues


def _pdfplumber_rows(raw_table: list[list[Any]]) -> list[dict[str, Any]]:
    rows = []
    for row_index, raw_row in enumerate(raw_table or [], start=1):
        cells = [
            {
                "cell_id": f"pdfplumber_cell_{row_index:03d}_{col_index + 1:03d}",
                "row_index": row_index,
                "col_index": col_index,
                "text": _clean_cell_text(cell),
                "source_span_ids": [],
                "page": None,
                "bbox_status": "missing",
            }
            for col_index, cell in enumerate(raw_row or [])
        ]
        if not any(cell["text"] for cell in cells):
            continue
        row_text = " ".join(cell["text"] for cell in cells)
        row_type = "header" if row_index == 1 and "项目" in row_text else "data"
        item = cells[0]["text"] if cells else ""
        rows.append(
            {
                "row_index": row_index,
                "row_type": row_type,
                "row_key": _nutrition_row_key(item) if row_type == "data" else None,
                "cells": cells,
                "source_span_ids": [],
            }
        )
    return rows


def _pdfplumber_table_type(raw_table: list[list[Any]]) -> str:
    text = "\n".join(" ".join(_clean_cell_text(cell) for cell in row or []) for row in raw_table or [])
    if "营养" in text or any(label in text for label in NUTRITION_ROW_LABELS):
        return "nutrition_facts"
    return "unknown"


def _pdfplumber_title(raw_table: list[list[Any]], page_index: int, raw_table_index: int) -> str:
    table_type = _pdfplumber_table_type(raw_table)
    if table_type == "nutrition_facts":
        return "营养成分表"
    return f"pdfplumber_table_p{page_index}_{raw_table_index}"


def _pdfplumber_columns(raw_table: list[list[Any]]) -> list[dict[str, Any]]:
    if not raw_table:
        return []
    first_row = raw_table[0] or []
    return [
        {"column_id": f"col_{index + 1:03d}", "name": _clean_cell_text(cell) or f"Column {index + 1}"}
        for index, cell in enumerate(first_row)
    ]


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _parser_agreement_status(primary_tables: list[dict[str, Any]], pdfplumber_tables: list[dict[str, Any]], dependency_preflight: dict[str, dict[str, Any]]) -> str:
    if not dependency_preflight.get("pdfplumber", {}).get("available"):
        return "single_parser_only"
    if not pdfplumber_tables:
        return "pdfplumber_no_tables"
    primary_count = len(primary_tables)
    pdfplumber_nutrition_count = sum(1 for table in pdfplumber_tables if table.get("table_type") == "nutrition_facts")
    if primary_count == pdfplumber_nutrition_count:
        return "table_count_match"
    return "table_count_conflict"


def _available_parsers(table_layers: dict[str, Any], dependency_preflight: dict[str, dict[str, Any]]) -> list[str]:
    available = ["text_span_nutrition"]
    if dependency_preflight.get("pdfplumber", {}).get("available"):
        available.append("pdfplumber")
    if dependency_preflight.get("camelot", {}).get("available"):
        available.append("camelot")
    return [parser for parser in available if parser in table_layers.get("parsers", available)]
    return {
        name: {
            "available": importlib.util.find_spec(module) is not None,
            "module": module,
            "required_for": "optional_parser" if name in {"pdfplumber", "camelot"} else "optional_parser_dependency",
        }
        for name, module in modules.items()
    }


def _nutrition_row_spans(spans: list[TextSpan], title_index: int) -> list[TextSpan]:
    title_page = spans[title_index].page
    rows: list[TextSpan] = []
    for span in spans[title_index + 1 :]:
        text = span.text.strip()
        if span.page != title_page:
            break
        if "营养成分表" in text:
            break
        if not rows and "项目" not in text and not _looks_like_nutrition_row(text):
            continue
        if rows and not _looks_like_nutrition_row(text) and "项目" not in text:
            break
        rows.append(span)
    return rows


def _columns_from_rows(row_spans: list[TextSpan]) -> list[dict[str, Any]]:
    for span in row_spans:
        if "项目" in span.text:
            parts = [part for part in re.split(r"\s{1,}", span.text.strip()) if part]
            if len(parts) >= 3:
                return [
                    {"column_id": "col_001", "name": parts[0]},
                    {"column_id": "col_002", "name": " ".join(parts[1:-1])},
                    {"column_id": "col_003", "name": parts[-1]},
                ]
    return [
        {"column_id": "col_001", "name": "项目"},
        {"column_id": "col_002", "name": "含量"},
        {"column_id": "col_003", "name": "NRV%"},
    ]


def _header_cells(text: str, span: TextSpan) -> list[dict[str, Any]]:
    columns = _columns_from_rows([span])
    return [
        _cell(0, index, column["name"], span)
        for index, column in enumerate(columns)
    ]


def _looks_like_nutrition_row(text: str) -> bool:
    return any(text.startswith(label) for label in NUTRITION_ROW_LABELS)


def _split_nutrition_row(text: str) -> tuple[str, str, str]:
    parts = [part for part in re.split(r"\s+", text.strip()) if part]
    if len(parts) <= 1:
        return text.strip(), "", ""
    nrv = parts[-1] if re.search(r"(%|-)$", parts[-1]) else ""
    amount_parts = parts[1:-1] if nrv else parts[1:]
    return parts[0], " ".join(amount_parts), nrv


def _nutrition_row_key(item: str) -> str:
    mapping = {
        "能量": "energy",
        "蛋白质": "protein",
        "蛋⽩质": "protein",
        "脂肪": "fat",
        "碳水化合物": "carbohydrate",
        "碳⽔化合物": "carbohydrate",
        "钠": "sodium",
        "钙": "calcium",
    }
    normalized = item.strip("—- ")
    return mapping.get(normalized, re.sub(r"\W+", "_", normalized).strip("_") or "unknown")


def _row_from_cells(row_index: int, row_type: str, cells: list[dict[str, Any]], row_key: str | None = None) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "row_type": row_type,
        "row_key": row_key,
        "cells": cells,
        "source_span_ids": sorted({source_id for cell in cells for source_id in cell["source_span_ids"]}),
    }


def _cell(row_index: int, col_index: int, text: str, span: TextSpan) -> dict[str, Any]:
    return {
        "cell_id": f"cell_{row_index:03d}_{col_index + 1:03d}",
        "row_index": row_index,
        "col_index": col_index,
        "text": text,
        "source_span_ids": [span.span_id],
        "page": span.page,
        "bbox_status": "available" if span.bbox_pdf else "missing",
        "bbox_pdf": span.bbox_pdf,
        "bbox_normalized": span.bbox_normalized,
    }


def _table_bbox_metadata(spans: list[TextSpan]) -> dict[str, Any]:
    if not spans or not all(span.bbox_pdf for span in spans):
        return {"bbox_status": "missing"}
    x1 = min(float(span.bbox_pdf.x) for span in spans if span.bbox_pdf)
    y1 = min(float(span.bbox_pdf.y) for span in spans if span.bbox_pdf)
    x2 = max(float(span.bbox_pdf.x + span.bbox_pdf.width) for span in spans if span.bbox_pdf)
    y2 = max(float(span.bbox_pdf.y + span.bbox_pdf.height) for span in spans if span.bbox_pdf)
    first_bbox = next(span.bbox_pdf for span in spans if span.bbox_pdf)
    metadata: dict[str, Any] = {
        "bbox_status": "available",
        "bbox_pdf": {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "page_width": first_bbox.page_width,
            "page_height": first_bbox.page_height,
            "unit": first_bbox.unit,
            "origin": first_bbox.origin,
        },
    }
    normalized = [span.bbox_normalized for span in spans if span.bbox_normalized]
    if len(normalized) == len(spans):
        metadata["bbox_normalized"] = {
            "x1": min(float(bbox.x1) for bbox in normalized),
            "y1": min(float(bbox.y1) for bbox in normalized),
            "x2": max(float(bbox.x2) for bbox in normalized),
            "y2": max(float(bbox.y2) for bbox in normalized),
        }
    return metadata
