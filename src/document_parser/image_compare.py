from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import BBoxNormalized, OcrLine, to_jsonable
from .utils import stable_id


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LONG_TEXT_KEYS = {
    "product.net_content",
    "product.directions",
    "product.warning",
    "custom.symbol_text",
}
KNOWN_FIELD_PREFIXES = (
    "配料",
    "配料表",
    "致敏物质提示",
    "致敏原提示",
    "贮存条件",
    "保质期",
    "生产日期",
    "生产日期、保质期到期日及工厂代码",
    "产品标准号",
    "产品标准代号",
    "产品类型",
    "产品分类",
    "品名",
    "冲调方法",
    "食用方法",
    "净含量",
    "净含量/规格",
    "委托方",
    "生产者",
    "受托方1",
    "受托方2",
    "受托方3",
    "受托方4",
    "受托方5",
    "被委托方",
    "地址",
    "产地",
    "公司地址",
    "食品生产许可证编号",
    "食品生产许可证",
    "电话",
    "客户服务热线",
    "消费者服务电话",
    "内容物1",
    "内容物2",
    "内容物3",
    "内容物4",
    "内容物5",
    "内容物6",
    "内容物7",
    "内容物8",
    "内容物9",
)
PARAGRAPH_ANCHOR_PREFIXES = {
    "配料",
    "配料表",
    "贮存条件",
    "冲调方法",
    "食用方法",
    "温馨提示",
    "净含量",
    "净含量/规格",
    "产品标准号",
    "产品标准代号",
    "生产者",
    "公司地址",
    "地址",
    "食品生产许可证编号",
    "客户服务热线",
    "消费者服务电话",
}


@dataclass(frozen=True)
class CompareCandidate:
    candidate_id: str
    candidate_type: str
    text: str
    normalized_text: str
    source_ocr_line_ids: list[str]
    bbox_normalized: dict[str, float] | None
    score: float
    reason: str
    region_id: str | None = None
    metadata: dict[str, Any] | None = None


class ImageCompareError(RuntimeError):
    pass


def load_standard_artifacts(standard_dir: Path) -> dict[str, Any]:
    required = ["standard_items.json"]
    missing = [name for name in required if not (standard_dir / name).exists()]
    if missing:
        raise ImageCompareError(f"Standard artifact directory is missing required file(s): {', '.join(missing)}")
    return {
        "standard_items": _read_json(standard_dir / "standard_items.json"),
        "field_groups": _read_json_if_exists(standard_dir / "field_groups.json", []),
        "tables": _read_json_if_exists(standard_dir / "tables.json", []),
        "lists": _read_json_if_exists(standard_dir / "lists.json", []),
        "comparison_index": _read_json_if_exists(standard_dir / "comparison_index.json", {}),
    }


def build_standard_targets(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in _as_list(artifacts.get("standard_items")):
        if not isinstance(item, dict):
            continue
        if item.get("semantic_key") == "product.nutrition_table":
            continue
        target = _target_from_standard_item(len(targets) + 1, item)
        if target:
            targets.append(target)

    for table in _as_list(artifacts.get("tables")):
        if not isinstance(table, dict) or table.get("table_type") != "nutrition_facts":
            continue
        targets.extend(_targets_from_nutrition_table(table, len(targets) + 1))
    return targets


def build_package_layout(ocr_lines: list[OcrLine], tables: list[Any] | None = None) -> dict[str, Any]:
    line_items = [_ocr_line_item(line) for line in ocr_lines]
    columns = _build_columns(line_items)
    content_segments = _build_content_segments(line_items)
    enterprise_segments = _build_enterprise_segments(line_items)
    table_regions = _build_table_regions(line_items, [table for table in _as_list(tables) if isinstance(table, dict)])
    return {
        "artifact_version": "package_layout_v0.1",
        "line_count": len(line_items),
        "columns": columns,
        "content_segments": content_segments,
        "enterprise_segments": enterprise_segments,
        "nutrition_table_regions": table_regions,
    }


def compare_standard_to_ocr(
    artifacts: dict[str, Any],
    ocr_lines: list[OcrLine],
    image_path: Path,
) -> dict[str, Any]:
    targets = build_standard_targets(artifacts)
    layout = build_package_layout(ocr_lines, artifacts.get("tables"))
    layout["barcode_decode"] = _decode_barcodes(image_path)
    line_items = [_ocr_line_item(line) for line in ocr_lines]
    _assign_target_scopes(targets, layout, line_items)
    candidate_pool = _build_candidate_pool(line_items, layout)
    candidate_pool.extend(_barcode_candidates(layout.get("barcode_decode")))
    candidates_by_target: dict[str, list[dict[str, Any]]] = {}
    extracted_items = []
    results = []
    used_line_ids: set[str] = set()

    for target in targets:
        result, target_candidates = _compare_target(target, line_items, layout, candidate_pool)
        candidates_by_target[target["target_id"]] = [to_jsonable(candidate) for candidate in target_candidates]
        if result.get("selected_candidate"):
            used_line_ids.update(result["selected_candidate"].get("source_ocr_line_ids", []))
            extracted_items.append(
                {
                    "target_id": target["target_id"],
                    "target_type": target["target_type"],
                    "semantic_key": target.get("semantic_key"),
                    "label": target.get("label"),
                    "text": result["selected_candidate"].get("text"),
                    "normalized_text": result["selected_candidate"].get("normalized_text"),
                    "source_ocr_line_ids": result["selected_candidate"].get("source_ocr_line_ids", []),
                    "bbox_normalized": result["selected_candidate"].get("bbox_normalized"),
                    "status": result["status"],
                }
            )
        results.append(result)

    unmatched = _unmatched_print_text(line_items, used_line_ids)
    critical_count = sum(1 for result in results if result["status"] in {"critical_missing", "critical_mismatch"})
    manual_review_count = sum(1 for result in results if result["status"] == "manual_review")
    pass_count = sum(1 for result in results if result["status"] == "pass")
    return {
        "comparison_result": {
            "artifact_version": "package_image_comparison_result_v0.1",
            "status": "fail" if critical_count else ("manual_review" if manual_review_count else "pass"),
            "image_path": str(image_path),
            "target_count": len(targets),
            "pass_count": pass_count,
            "critical_count": critical_count,
            "manual_review_count": manual_review_count,
            "info_extra_text_count": len(unmatched),
            "results": results,
        },
        "standard_targets": targets,
        "package_layout": layout,
        "package_candidates": {
            "artifact_version": "package_candidates_v0.1",
            "target_count": len(targets),
            "candidates_by_target": candidates_by_target,
        },
        "package_extracted_items": {
            "artifact_version": "package_extracted_items_v0.1",
            "item_count": len(extracted_items),
            "items": extracted_items,
        },
        "unmatched_print_text": {
            "artifact_version": "unmatched_print_text_v0.1",
            "severity": "info_extra_text",
            "item_count": len(unmatched),
            "items": unmatched,
        },
    }


def normalize_compare_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    replacements = {
        "：": ":",
        "﹕": ":",
        "∶": ":",
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "、": ",",
        "；": ";",
        "％": "%",
        "×": "x",
        "X": "x",
        "*": "x",
        "－": "-",
        "—": "-",
        "–": "-",
        "≤": "<=",
        "≥": ">=",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"-{2,}", "-", value)
    value = re.sub(r"([0-9])\.0(?=[^0-9]|$)", r"\1", value)
    return value.lower()


def normalize_ppocr_fixture_page(fixture_path: Path, fallback_width: int, fallback_height: int) -> tuple[int, int]:
    body = _read_json(fixture_path)
    data_info = body.get("dataInfo") if isinstance(body, dict) else {}
    data_info_dict = _as_dict(data_info)
    first_page = _as_dict(_as_list(data_info_dict.get("pages"))[0]) if _as_list(data_info_dict.get("pages")) else {}
    width = _int_or_none(data_info_dict.get("width")) or _int_or_none(first_page.get("width")) or fallback_width
    height = _int_or_none(data_info_dict.get("height")) or _int_or_none(first_page.get("height")) or fallback_height
    return width, height


def build_ocr_quality_report(
    *,
    fixture_path: Path | None,
    input_image_width: int,
    input_image_height: int,
    ocr_page_width: int,
    ocr_page_height: int,
    ocr_lines: list[OcrLine],
) -> dict[str, Any]:
    fixture_data_info = None
    if fixture_path:
        body = _read_json(fixture_path)
        fixture_data_info = _as_dict(body.get("dataInfo")) if isinstance(body, dict) else {}
    bbox_range = _ocr_bbox_range(ocr_lines)
    input_ratio = _safe_ratio(input_image_width, input_image_height)
    ocr_ratio = _safe_ratio(ocr_page_width, ocr_page_height)
    ratio_delta = abs(input_ratio - ocr_ratio) / input_ratio if input_ratio else 0.0
    bbox_overlay_status = "aligned" if ratio_delta <= 0.08 else "source_image_mismatch"
    return {
        "artifact_version": "package_ocr_quality_report_v0.1",
        "fixture_path": str(fixture_path.resolve()) if fixture_path else None,
        "fixture_data_info": fixture_data_info,
        "input_image_size": {"width": input_image_width, "height": input_image_height},
        "ocr_page_size": {"width": ocr_page_width, "height": ocr_page_height},
        "aspect_ratio": {
            "input_image": round(input_ratio, 6) if input_ratio else None,
            "ocr_page": round(ocr_ratio, 6) if ocr_ratio else None,
            "relative_delta": round(ratio_delta, 6),
            "matches": ratio_delta <= 0.08,
        },
        "bbox_overlay_status": bbox_overlay_status,
        "bbox_range": bbox_range,
        "line_count": len(ocr_lines),
        "bbox_available_count": sum(1 for line in ocr_lines if line.bbox_normalized is not None),
        "quality_flags": ([] if bbox_overlay_status == "aligned" else ["source_image_mismatch"]),
    }


def _target_from_standard_item(index: int, item: dict[str, Any]) -> dict[str, Any] | None:
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    semantic_key = str(item.get("semantic_key") or "")
    return {
        "target_id": stable_id("target", index),
        "target_type": "field",
        "category": _target_category(item),
        "semantic_key": semantic_key,
        "label": item.get("label") or item.get("field") or semantic_key,
        "expected_text": text,
        "normalized_expected_text": normalize_compare_text(text),
        "source_standard_item_ids": [item.get("id")],
        "group_id": item.get("group_id"),
        "table_id": item.get("table_id"),
        "row_key": item.get("row_key"),
        "comparison_required": bool(item.get("comparison_required", True)),
        "review_required": bool(item.get("review_required", False)),
        "severity_if_missing": "critical",
        "severity_if_mismatch": "critical",
    }


def _targets_from_nutrition_table(table: dict[str, Any], start_index: int) -> list[dict[str, Any]]:
    targets = []
    table_id = str(table.get("table_id") or "")
    title = str(table.get("title") or "营养成分表").strip()
    target_index = start_index
    targets.append(_nutrition_target(target_index, "nutrition_title", table_id, "营养表标题", title, None))
    target_index += 1

    columns = [column for column in _as_list(table.get("columns")) if isinstance(column, dict)]
    header_text = " ".join(str(column.get("name") or column.get("column_id") or "").strip() for column in columns if column)
    if header_text:
        targets.append(_nutrition_target(target_index, "nutrition_header", table_id, "营养表表头", header_text, None))
        target_index += 1

    column_names = {str(column.get("column_id")): str(column.get("name") or column.get("column_id") or "") for column in columns}
    for row in _as_list(table.get("rows")):
        if not isinstance(row, dict):
            continue
        cells = [cell for cell in _as_list(row.get("cells")) if isinstance(cell, dict)]
        cell_values = [str(cell.get("raw_value") or cell.get("normalized_value") or "").strip() for cell in cells]
        row_key = str(row.get("row_key") or (cell_values[0] if cell_values else "")).strip()
        expected_text = " ".join(value for value in cell_values if value)
        targets.append(
            {
                **_nutrition_target(target_index, "nutrition_row", table_id, f"{title} - {row_key}", expected_text, row_key),
                "expected_parts": [
                    {
                        "column_id": str(cell.get("column_id") or ""),
                        "column_name": column_names.get(str(cell.get("column_id") or ""), ""),
                        "text": str(cell.get("raw_value") or cell.get("normalized_value") or "").strip(),
                    }
                    for cell in cells
                ],
            }
        )
        target_index += 1

    for footnote_index, footnote in enumerate(_as_list(table.get("footnotes")), start=1):
        text = str(footnote).strip()
        if text:
            targets.append(_nutrition_target(target_index, "nutrition_footnote", table_id, f"{title} - 脚注 {footnote_index}", text, None))
            target_index += 1
    return targets


def _nutrition_target(index: int, target_type: str, table_id: str, label: str, expected_text: str, row_key: str | None) -> dict[str, Any]:
    return {
        "target_id": stable_id("target", index),
        "target_type": target_type,
        "category": "nutrition",
        "semantic_key": f"nutrition.{target_type.removeprefix('nutrition_')}",
        "label": label,
        "expected_text": expected_text,
        "normalized_expected_text": normalize_compare_text(expected_text),
        "source_standard_item_ids": [],
        "group_id": None,
        "table_id": table_id,
        "row_key": row_key,
        "comparison_required": True,
        "review_required": False,
        "severity_if_missing": "critical",
        "severity_if_mismatch": "critical",
    }


def _target_category(item: dict[str, Any]) -> str:
    semantic_key = str(item.get("semantic_key") or "")
    if semantic_key.startswith(("principal.", "manufacturer.", "distributor.", "enterprise.")):
        return "enterprise"
    if semantic_key.startswith("content_item."):
        return "content"
    source = _as_dict(item.get("source"))
    section = str(source.get("section") or "")
    if "other" in section:
        return "other"
    return "main"


def _build_columns(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {
        "left": [line for line in line_items if _line_center_x(line) < 0.48],
        "right": [line for line in line_items if _line_center_x(line) >= 0.48],
    }
    columns = []
    for column_id, lines in grouped.items():
        ordered = sorted(lines, key=lambda line: (_line_center_y(line), _line_center_x(line)))
        bbox = _union_bbox([line.get("bbox_normalized") for line in ordered])
        columns.append(
            {
                "column_id": column_id,
                "line_count": len(ordered),
                "source_ocr_line_ids": [line["ocr_line_id"] for line in ordered],
                "bbox_normalized": bbox,
            }
        )
    return columns


def _build_content_segments(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors = []
    for line in line_items:
        if _line_center_x(line) < 0.45:
            continue
        match = re.search(r"内容\s*物?\s*(\d+)", line["text"])
        if match:
            anchors.append((int(match.group(1)), line))
    anchors.sort(key=lambda item: _line_center_y(item[1]))
    segments = []
    for index, (content_index, anchor) in enumerate(anchors):
        y1 = _line_center_y(anchor) - 0.004
        y2 = _line_center_y(anchors[index + 1][1]) - 0.004 if index + 1 < len(anchors) else 1.0
        segment_lines = [
            line
            for line in line_items
            if _line_center_x(line) >= 0.45 and y1 <= _line_center_y(line) < y2
        ]
        segments.append(
            {
                "segment_id": f"content_c{content_index}",
                "content_index": content_index,
                "anchor_ocr_line_id": anchor["ocr_line_id"],
                "source_ocr_line_ids": [line["ocr_line_id"] for line in segment_lines],
                "bbox_normalized": _union_bbox([line.get("bbox_normalized") for line in segment_lines]),
            }
        )
    return segments


def _build_enterprise_segments(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors = [line for line in line_items if _is_enterprise_anchor(line)]
    bands = _enterprise_anchor_bands(anchors)
    segments = []
    for band_index, band in enumerate(bands, start=1):
        band_anchors = sorted(band["anchors"], key=lambda item: (_line_center_y(item), _line_center_x(item)))
        x1, x2 = band["x1"], band["x2"]
        for anchor_index, anchor in enumerate(band_anchors):
            next_anchor_y = _line_center_y(band_anchors[anchor_index + 1]) if anchor_index + 1 < len(band_anchors) else None
            y1 = _enterprise_segment_y1(anchor, line_items, x1, x2, include_previous_fields=anchor_index == 0)
            y2 = next_anchor_y - 0.006 if next_anchor_y is not None else min(1.0, _line_center_y(anchor) + 0.13)
            segment_lines = [
                line
                for line in line_items
                if x1 <= _line_center_x(line) <= x2 and y1 <= _line_center_y(line) < y2
            ]
            segments.append(
                {
                    "segment_id": f"enterprise_{len(segments) + 1}",
                    "subcolumn_id": f"enterprise_band_{band_index}",
                    "group_id": None,
                    "anchor_ocr_line_id": anchor["ocr_line_id"],
                    "anchor_text": anchor["text"],
                    "source_ocr_line_ids": [line["ocr_line_id"] for line in sorted(segment_lines, key=lambda item: (_line_center_y(item), _line_center_x(item)))],
                    "bbox_normalized": _union_bbox([line.get("bbox_normalized") for line in segment_lines]),
                }
            )
    return segments


def _is_enterprise_anchor(line: dict[str, Any]) -> bool:
    text = normalize_compare_text(str(line.get("text") or ""))
    return bool(re.match(r"^(委托方|被委托方|受委托方|受托方\d*|生产者|生产商|经销方|经销商)", text))


def _enterprise_anchor_bands(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not anchors:
        return []
    clusters: list[list[dict[str, Any]]] = []
    for anchor in sorted(anchors, key=_line_center_x):
        center = _line_center_x(anchor)
        if clusters and abs(center - _enterprise_cluster_center(clusters[-1])) <= 0.18:
            clusters[-1].append(anchor)
        else:
            clusters.append([anchor])

    centers = [_enterprise_cluster_center(cluster) for cluster in clusters]
    bands = []
    for index, cluster in enumerate(clusters):
        bbox = _union_bbox([line.get("bbox_normalized") for line in cluster]) or {}
        left = max(0.0, float(bbox.get("x1", centers[index])) - 0.08)
        right = min(1.0, float(bbox.get("x2", centers[index])) + 0.1)
        if index > 0:
            left = max(left, (centers[index - 1] + centers[index]) / 2)
        if index + 1 < len(clusters):
            right = min(right, (centers[index] + centers[index + 1]) / 2)
        bands.append({"anchors": cluster, "x1": left, "x2": right})
    return bands


def _enterprise_cluster_center(cluster: list[dict[str, Any]]) -> float:
    return sum(_line_center_x(line) for line in cluster) / len(cluster)


def _enterprise_segment_y1(
    anchor: dict[str, Any],
    line_items: list[dict[str, Any]],
    x1: float,
    x2: float,
    *,
    include_previous_fields: bool,
) -> float:
    anchor_y = _line_center_y(anchor)
    if not include_previous_fields:
        return max(0.0, anchor_y - 0.006)
    pre_field_ys = [
        _line_center_y(line)
        for line in line_items
        if x1 <= _line_center_x(line) <= x2
        and 0 < anchor_y - _line_center_y(line) <= 0.13
        and _is_enterprise_field_prefix_line(line)
    ]
    if pre_field_ys:
        return max(0.0, min(pre_field_ys) - 0.006)
    return max(0.0, anchor_y - 0.006)


def _is_enterprise_field_prefix_line(line: dict[str, Any]) -> bool:
    text = normalize_compare_text(str(line.get("text") or ""))
    return bool(re.match(r"^(地址|公司地址|生产地址|产地|食品生产许可证|食品生产许可证编号|电话|客户服务热线|消费者服务电话)", text))


def _build_table_regions(line_items: list[dict[str, Any]], tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    title_lines = [line for line in line_items if "营养成分表" in line["text"]]
    title_lines.sort(key=lambda line: (_line_center_y(line), _line_center_x(line)))
    regions = []
    for line in title_lines:
        matched_table = _best_table_for_title(line["text"], tables)
        if not matched_table:
            continue
        bbox = _as_dict(line.get("bbox_normalized"))
        x1, x2 = _table_region_x_bounds(line, title_lines)
        y1 = max(0.0, float(bbox.get("y1", 0)) - 0.005)
        next_y = _next_table_title_y(line, title_lines, x1, x2)
        y2 = next_y - 0.006 if next_y is not None else min(1.0, y1 + 0.22)
        region_lines = [
            candidate
            for candidate in line_items
            if x1 <= _line_center_x(candidate) <= x2 and y1 <= _line_center_y(candidate) <= y2
        ]
        regions.append(
            {
                "region_id": f"region_{matched_table.get('table_id')}",
                "table_id": matched_table.get("table_id"),
                "title": matched_table.get("title"),
                "title_ocr_line_id": line["ocr_line_id"],
                "source_ocr_line_ids": [item["ocr_line_id"] for item in region_lines],
                "bbox_normalized": _union_bbox([item.get("bbox_normalized") for item in region_lines]),
            }
        )
    return regions


def _table_region_x_bounds(title_line: dict[str, Any], title_lines: list[dict[str, Any]]) -> tuple[float, float]:
    bbox = _as_dict(title_line.get("bbox_normalized"))
    default_x1 = max(0.0, float(bbox.get("x1", 0)) - 0.35)
    default_x2 = min(1.0, float(bbox.get("x2", 1)) + 0.35)
    same_band = [
        line
        for line in title_lines
        if abs(_line_center_y(line) - _line_center_y(title_line)) <= 0.03
    ]
    same_band.sort(key=_line_center_x)
    index = same_band.index(title_line) if title_line in same_band else -1
    if index == -1:
        return default_x1, default_x2
    x1 = default_x1
    x2 = default_x2
    current_x = _line_center_x(title_line)
    if index > 0:
        x1 = (current_x + _line_center_x(same_band[index - 1])) / 2
    if index + 1 < len(same_band):
        x2 = (current_x + _line_center_x(same_band[index + 1])) / 2
    return max(0.0, x1), min(1.0, x2)


def _build_candidate_pool(line_items: list[dict[str, Any]], layout: dict[str, Any]) -> list[CompareCandidate]:
    candidates: list[CompareCandidate] = []
    for line in line_items:
        candidates.append(_candidate(len(candidates) + 1, "line", [line], "single OCR line", None))
        for split_candidate in _split_line_candidates(line, len(candidates) + 1):
            candidates.append(split_candidate)

    for paragraph in _anchored_paragraph_candidates(line_items, len(candidates) + 1):
        candidates.append(paragraph)

    lines_by_id = {line["ocr_line_id"]: line for line in line_items}
    for column in _as_list(layout.get("columns")):
        column_lines = [lines_by_id[line_id] for line_id in column.get("source_ocr_line_ids", []) if line_id in lines_by_id]
        for start in range(len(column_lines)):
            for length in range(2, min(4, len(column_lines) - start) + 1):
                window = column_lines[start : start + length]
                if _window_has_noise(window):
                    continue
                candidates.append(_candidate(len(candidates) + 1, "window", window, f"{column.get('column_id')} column {length}-line window", None))

    for segment in _as_list(layout.get("content_segments")):
        segment_lines = [lines_by_id[line_id] for line_id in segment.get("source_ocr_line_ids", []) if line_id in lines_by_id]
        if segment_lines:
            candidates.append(_candidate(len(candidates) + 1, "content_segment", segment_lines, "content item segment", str(segment.get("segment_id"))))

    for segment in _as_list(layout.get("enterprise_segments")):
        segment_lines = [lines_by_id[line_id] for line_id in segment.get("source_ocr_line_ids", []) if line_id in lines_by_id]
        if not segment_lines:
            continue
        region_id = str(segment.get("segment_id"))
        candidates.append(_candidate(len(candidates) + 1, "enterprise_segment", segment_lines, "enterprise scoped segment", region_id))
        for start in range(len(segment_lines)):
            for length in range(1, min(6, len(segment_lines) - start) + 1):
                window = segment_lines[start : start + length]
                candidates.append(_candidate(len(candidates) + 1, "enterprise_window", window, "enterprise subcolumn window", region_id))
    return candidates


def _split_line_candidates(line: dict[str, Any], start_index: int) -> list[CompareCandidate]:
    spans = _known_prefix_spans(str(line.get("text") or ""))
    if len(spans) < 2:
        return []
    candidates = []
    for offset, (char_start, prefix) in enumerate(spans):
        char_end = spans[offset + 1][0] if offset + 1 < len(spans) else len(str(line.get("text") or ""))
        text = str(line.get("text") or "")[char_start:char_end].strip()
        if not text:
            continue
        candidates.append(
            _text_candidate(
                start_index + len(candidates),
                "split_line",
                text,
                [str(line.get("ocr_line_id"))],
                _as_dict(line.get("bbox_normalized")) or None,
                "prefix-aware split from multi-field OCR line",
                None,
                {"char_start": char_start, "char_end": char_end, "prefix": prefix},
            )
        )
    return candidates


def _known_prefix_spans(text: str) -> list[tuple[int, str]]:
    spans: list[tuple[int, str]] = []
    for prefix in sorted(KNOWN_FIELD_PREFIXES, key=len, reverse=True):
        pattern = re.compile(re.escape(prefix) + r"\s*[:：]")
        for match in pattern.finditer(text):
            spans.append((match.start(), prefix))
    return sorted(set(spans), key=lambda item: item[0])


def _anchored_paragraph_candidates(line_items: list[dict[str, Any]], start_index: int) -> list[CompareCandidate]:
    ordered = sorted(line_items, key=lambda item: (_line_center_y(item), _line_center_x(item)))
    candidates = []
    for anchor in ordered:
        if not _starts_paragraph_anchor(anchor):
            continue
        lines = _paragraph_lines_from_anchor(anchor, ordered)
        if len(lines) <= 1:
            continue
        candidates.append(_candidate(start_index + len(candidates), "anchored_paragraph", lines, "known-prefix anchored paragraph", None))
        strict_lines = _strict_indent_lines(anchor, lines)
        if 1 < len(strict_lines) < len(lines):
            candidates.append(_candidate(start_index + len(candidates), "anchored_paragraph_strict", strict_lines, "known-prefix anchored paragraph with strict indent", None))
    return candidates


def _starts_paragraph_anchor(line: dict[str, Any]) -> bool:
    text = normalize_compare_text(str(line.get("text") or ""))
    return any(text.startswith(normalize_compare_text(prefix) + ":") for prefix in PARAGRAPH_ANCHOR_PREFIXES)


def _paragraph_lines_from_anchor(anchor: dict[str, Any], ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_y = _line_center_y(anchor)
    anchor_x = _line_center_x(anchor)
    anchor_bbox = _as_dict(anchor.get("bbox_normalized"))
    x1 = max(0.0, float(anchor_bbox.get("x1", anchor_x)) - 0.035)
    x2 = min(1.0, float(anchor_bbox.get("x2", anchor_x)) + 0.06)
    lines = [anchor]
    last_y = anchor_y
    for line in ordered:
        if line is anchor:
            continue
        y = _line_center_y(line)
        if y <= anchor_y:
            continue
        if y - last_y > 0.045:
            break
        text = str(line.get("text") or "")
        norm = normalize_compare_text(text)
        if _is_noise_line(line):
            continue
        cx = _line_center_x(line)
        if _starts_known_prefix(line) and (x1 <= cx <= x2 or abs(cx - anchor_x) <= 0.11) and line not in lines:
            break
        if x1 <= cx <= x2 or abs(cx - anchor_x) <= 0.11:
            lines.append(line)
            last_y = y
            line_bbox = _as_dict(line.get("bbox_normalized"))
            if line_bbox:
                x1 = min(x1, max(0.0, float(line_bbox.get("x1", x1)) - 0.02))
                x2 = max(x2, min(1.0, float(line_bbox.get("x2", x2)) + 0.02))
        elif lines and len(norm) <= 2:
            continue
    return sorted(lines, key=lambda item: (_line_center_y(item), _line_center_x(item)))


def _strict_indent_lines(anchor: dict[str, Any], lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_left = float(_as_dict(anchor.get("bbox_normalized")).get("x1", _line_center_x(anchor)))
    strict = [anchor]
    for line in lines:
        if line is anchor:
            continue
        left = float(_as_dict(line.get("bbox_normalized")).get("x1", _line_center_x(line)))
        if abs(left - anchor_left) <= 0.025:
            strict.append(line)
    return sorted(strict, key=lambda item: (_line_center_y(item), _line_center_x(item)))


def _starts_known_prefix(line: dict[str, Any]) -> bool:
    text = normalize_compare_text(str(line.get("text") or ""))
    return any(text.startswith(normalize_compare_text(prefix) + ":") for prefix in KNOWN_FIELD_PREFIXES)


def _window_has_noise(lines: list[dict[str, Any]]) -> bool:
    return any(_is_noise_line(line) for line in lines)


def _is_noise_line(line: dict[str, Any]) -> bool:
    text = normalize_compare_text(str(line.get("text") or ""))
    if text in {"刀纹", "涂胶", "打牙", "区域", "r", "0", "1", "110", "100", "奶茶", "优乐美"}:
        return True
    return len(text) <= 2 and not re.search(r"[\u4e00-\u9fff]{2,}|[0-9]{2,}", text)


def _candidate_has_noise(candidate: CompareCandidate) -> bool:
    return any(normalize_compare_text(line) in {"刀纹", "涂胶", "打牙", "区域", "r", "0", "1", "110", "100", "奶茶", "优乐美"} for line in candidate.text.splitlines())


def _compare_target(
    target: dict[str, Any],
    line_items: list[dict[str, Any]],
    layout: dict[str, Any],
    candidate_pool: list[CompareCandidate],
) -> tuple[dict[str, Any], list[CompareCandidate]]:
    if not target.get("comparison_required", True):
        return _manual_review_result(target, "comparison_not_required", None), []
    if target.get("review_required"):
        return _manual_review_result(target, "standard_item_requires_review", None), []
    if str(target.get("target_type", "")).startswith("nutrition_"):
        return _compare_nutrition_target(target, line_items, layout)
    return _compare_field_target(target, layout, candidate_pool)


def _compare_field_target(
    target: dict[str, Any],
    layout: dict[str, Any],
    candidate_pool: list[CompareCandidate],
) -> tuple[dict[str, Any], list[CompareCandidate]]:
    expected_norm = _match_normalize_for_target(target, str(target.get("expected_text") or ""))
    pool = _candidate_pool_for_target(target, layout, candidate_pool)
    long_text = _is_long_target(target)
    exact = []
    for candidate in pool:
        candidate_norm = _match_normalize_for_target(target, candidate.text)
        if _candidate_matches_for_target(target, expected_norm, candidate_norm, long_text):
            reason = "normalized_text_exact_match"
            if candidate_norm != candidate.normalized_text:
                reason = "separator_insensitive_normalized_match"
            exact.append(_with_score(candidate, 1.0, reason))
    if exact:
        selected = _best_candidate(exact)
        return _result(target, "pass", "matched", selected), exact[:5]

    sequence_candidate = _sequence_candidate_for_target(target, pool, expected_norm)
    if sequence_candidate:
        return _result(target, "pass", "matched_sequence_lines", sequence_candidate), [sequence_candidate]

    scored = [
        _with_score(
            candidate,
            _candidate_score(expected_norm, _match_normalize_for_target(target, candidate.text), long_text),
            candidate.reason,
        )
        for candidate in pool
    ]
    scored = sorted(scored, key=lambda item: item.score, reverse=True)
    viable = [candidate for candidate in scored if _is_viable_mismatch_candidate(target, expected_norm, candidate)]
    if viable:
        selected = viable[0]
        return _result(target, "critical_mismatch", "candidate_found_but_text_differs", selected), viable[:5]
    if _is_barcode_target(target):
        barcode_decode = _as_dict(layout.get("barcode_decode"))
        decode_status = str(barcode_decode.get("status") or "unavailable")
        reason = "barcode_decoder_unavailable" if decode_status == "unavailable" else "barcode_not_decoded"
        return _result(target, "critical_missing", reason, None), scored[:5]
    return _result(target, "critical_missing", "no_reliable_candidate_found", None), scored[:5]


def _compare_nutrition_target(
    target: dict[str, Any],
    line_items: list[dict[str, Any]],
    layout: dict[str, Any],
) -> tuple[dict[str, Any], list[CompareCandidate]]:
    table_id = str(target.get("table_id") or "")
    region = next((item for item in _as_list(layout.get("nutrition_table_regions")) if item.get("table_id") == table_id), None)
    if not isinstance(region, dict):
        return _result(target, "critical_missing", "nutrition_table_region_not_found", None), []
    lines_by_id = {line["ocr_line_id"]: line for line in line_items}
    region_lines = [lines_by_id[line_id] for line_id in region.get("source_ocr_line_ids", []) if line_id in lines_by_id]
    target_type = str(target.get("target_type") or "")
    if target_type == "nutrition_title":
        title_line = lines_by_id.get(str(region.get("title_ocr_line_id")))
        candidate = _candidate(1, "nutrition_title", [title_line] if title_line else [], "nutrition title line", str(region.get("region_id")))
        status = "pass" if candidate.normalized_text == target.get("normalized_expected_text") else "critical_mismatch"
        return _result(target, status, "nutrition_title_match" if status == "pass" else "nutrition_title_differs", candidate), [candidate]
    if target_type == "nutrition_header":
        candidate_lines = _nutrition_header_lines(region_lines)
        candidate = _candidate(1, "nutrition_header", candidate_lines, "nutrition header row", str(region.get("region_id")))
        expected_parts = [normalize_compare_text(part) for part in str(target.get("expected_text") or "").split() if part.strip()]
        pass_status = all(_nutrition_header_part_present(part, candidate.normalized_text) for part in expected_parts)
        return _result(target, "pass" if pass_status else "critical_mismatch", "nutrition_header_checked", candidate), [candidate]
    if target_type == "nutrition_row":
        return _compare_nutrition_row(target, region_lines, str(region.get("region_id")))
    if target_type == "nutrition_footnote":
        candidate = _best_text_candidate(target, region_lines, str(region.get("region_id")))
        if candidate and _candidate_matches(str(target.get("normalized_expected_text")), candidate.normalized_text, True):
            return _result(target, "pass", "nutrition_footnote_match", candidate), [candidate]
        review_candidate = candidate if candidate and candidate.score >= 0.75 else None
        return _result(target, "critical_missing", "nutrition_footnote_not_found", review_candidate), [candidate] if candidate else []
    return _result(target, "manual_review", "unsupported_nutrition_target_type", None), []


def _compare_nutrition_row(target: dict[str, Any], region_lines: list[dict[str, Any]], region_id: str) -> tuple[dict[str, Any], list[CompareCandidate]]:
    row_key = normalize_compare_text(str(target.get("row_key") or ""))
    structured = _structured_nutrition_row_candidate(target, region_lines, region_id)
    if structured:
        status, reason, candidate = structured
        return _result(target, status, reason, candidate), [candidate]
    row_groups = _nutrition_row_groups(region_lines)
    candidates = []
    for group in row_groups:
        candidate = _candidate(len(candidates) + 1, "nutrition_row", group, "nutrition row geometry group", region_id)
        candidates.append(candidate)
    for candidate in candidates:
        if _candidate_matches(str(target.get("normalized_expected_text")), candidate.normalized_text, False):
            return _result(target, "pass", "nutrition_row_match", candidate), candidates

    merged_candidates = [candidate for candidate in candidates if "糖钠" in normalize_compare_text(candidate.text)]
    if row_key in {"糖", "-糖", "钠"} and merged_candidates:
        if row_key == "钠":
            reconstructed = _reconstruct_sodium_candidate(target, merged_candidates, region_lines, region_id)
            if reconstructed:
                return _result(target, "pass", "nutrition_sodium_row_reconstructed_from_merged_label", reconstructed), candidates + [reconstructed]
        return _manual_review_result(target, "nutrition_sugar_sodium_rows_merged_by_ocr", merged_candidates[0]), candidates

    for candidate in candidates:
        if row_key and row_key in candidate.normalized_text:
            return _result(target, "critical_mismatch", "nutrition_row_found_but_values_differ", candidate), candidates
    return _result(target, "critical_missing", "nutrition_row_not_found", None), candidates


def _nutrition_header_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headers = []
    for line in lines:
        text = normalize_compare_text(line["text"])
        if _is_nutrition_header_text(text):
            headers.append(line)
    return sorted(headers, key=lambda item: (_line_center_y(item), _line_center_x(item)))


def _structured_nutrition_row_candidate(
    target: dict[str, Any],
    region_lines: list[dict[str, Any]],
    region_id: str,
) -> tuple[str, str, CompareCandidate] | None:
    row_key = normalize_compare_text(str(target.get("row_key") or ""))
    label_lines = [line for line in region_lines if _line_matches_row_label(line, row_key)]
    if not label_lines:
        return None
    expected_values = _nutrition_expected_value_norms(target)
    if not expected_values:
        return None
    for label_line in sorted(label_lines, key=lambda item: (_line_center_y(item), _line_center_x(item))):
        label_text = normalize_compare_text(str(label_line.get("text") or ""))
        if row_key in {"糖", "-糖"} and "糖钠" in label_text:
            candidate = _candidate(8000, "nutrition_row_structured", [label_line], "nutrition merged sugar/sodium label", region_id)
            return "manual_review", "nutrition_sugar_sodium_rows_merged_by_ocr", candidate
        nearby = _nearby_nutrition_lines(label_line, region_lines)
        selected = [label_line]
        for value in expected_values:
            matched = _find_nutrition_value_line(value, nearby, selected)
            if matched is None:
                selected = []
                break
            selected.append(matched)
        if selected:
            candidate = _candidate(8001, "nutrition_row_structured", _dedupe_lines(selected), "nutrition row reconstructed by column/value tracks", region_id)
            return "pass", "nutrition_row_structured_match", _with_score(candidate, 1.0, "nutrition_row_structured_match")
    return None


def _nearby_nutrition_lines(label_line: dict[str, Any], region_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_y = _line_center_y(label_line)
    return [
        line
        for line in region_lines
        if abs(_line_center_y(line) - label_y) <= 0.014
        and "营养成分表" not in str(line.get("text") or "")
        and not _is_nutrition_footnote_text(line)
    ]


def _find_nutrition_value_line(value_norm: str, lines: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any] | None:
    for line in sorted(lines, key=lambda item: (_line_center_y(item), _line_center_x(item))):
        if line in selected:
            continue
        text = normalize_compare_text(str(line.get("text") or ""))
        if value_norm == text or value_norm in text:
            return line
    return None


def _line_matches_row_label(line: dict[str, Any], row_key_norm: str) -> bool:
    text = normalize_compare_text(str(line.get("text") or ""))
    if not text or _is_nutrition_header_text(text) or _is_nutrition_footnote_text(line):
        return False
    variants = _row_label_variants(row_key_norm)
    return text in variants or any(variant and variant in text and len(text) <= len(variant) + 2 for variant in variants)


def _row_label_variants(row_key_norm: str) -> set[str]:
    variants = {row_key_norm}
    if row_key_norm.startswith("-"):
        bare = row_key_norm.lstrip("-")
        variants.update({bare, f"一{bare}", f"-{bare}"})
    if row_key_norm.startswith("—"):
        bare = row_key_norm.lstrip("—")
        variants.update({bare, f"一{bare}", f"-{bare}"})
    return {variant for variant in variants if variant}


def _is_nutrition_header_text(text: str) -> bool:
    norm = normalize_compare_text(text)
    return norm in {"项目", "营养项目", "每100克", "每100克(g)", "每份(65g)", "营养素参考值%", "nrv%"}


def _nutrition_header_part_present(part_norm: str, candidate_norm: str) -> bool:
    if not part_norm:
        return True
    variants = {part_norm}
    if part_norm == "营养项目":
        variants.add("项目")
    if part_norm == "营养素参考值%":
        variants.add("nrv%")
    return any(variant in candidate_norm for variant in variants)


def _is_nutrition_footnote_text(line: dict[str, Any]) -> bool:
    text = normalize_compare_text(str(line.get("text") or ""))
    return "儿童青少年应避免过量摄入盐油糖" in text


def _nutrition_row_groups(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    row_lines = [
        line
        for line in lines
        if "营养成分表" not in line["text"]
        and not _is_nutrition_header_text(str(line.get("text") or ""))
        and not _is_nutrition_footnote_text(line)
    ]
    groups: list[list[dict[str, Any]]] = []
    for line in sorted(row_lines, key=lambda item: (_line_center_y(item), _line_center_x(item))):
        y = _line_center_y(line)
        group = next((item for item in groups if abs(_line_center_y(item[0]) - y) <= 0.009), None)
        if group is None:
            groups.append([line])
        else:
            group.append(line)
    return [sorted(group, key=_line_center_x) for group in groups]


def _reconstruct_sodium_candidate(
    target: dict[str, Any],
    merged_candidates: list[CompareCandidate],
    region_lines: list[dict[str, Any]],
    region_id: str,
) -> CompareCandidate | None:
    expected_values = _nutrition_expected_value_norms(target)
    if not expected_values:
        return None
    lines_by_id = {str(line.get("ocr_line_id")): line for line in region_lines}
    for merged in merged_candidates:
        merged_lines = [lines_by_id[line_id] for line_id in merged.source_ocr_line_ids if line_id in lines_by_id]
        if not merged_lines:
            continue
        y_min = min(_line_center_y(line) for line in merged_lines) - 0.018
        y_max = max(_line_center_y(line) for line in merged_lines) + 0.018
        nearby_lines = [line for line in region_lines if y_min <= _line_center_y(line) <= y_max]
        selected_lines = [line for line in nearby_lines if "糖钠" in normalize_compare_text(str(line.get("text") or ""))]
        for expected_value in expected_values:
            matched = next(
                (
                    line
                    for line in nearby_lines
                    if expected_value == normalize_compare_text(str(line.get("text") or ""))
                    or expected_value in normalize_compare_text(str(line.get("text") or ""))
                ),
                None,
            )
            if matched is None:
                selected_lines = []
                break
            selected_lines.append(matched)
        if selected_lines:
            unique_lines = _dedupe_lines(selected_lines)
            candidate = _candidate(9000, "nutrition_row_reconstructed", unique_lines, "nutrition sodium row reconstructed from merged OCR label", region_id)
            return _with_score(candidate, 1.0, "nutrition_sodium_reconstructed")
    return None


def _nutrition_expected_value_norms(target: dict[str, Any]) -> list[str]:
    row_key = normalize_compare_text(str(target.get("row_key") or ""))
    values = []
    for part in _as_list(target.get("expected_parts")):
        text = normalize_compare_text(str(_as_dict(part).get("text") or ""))
        if not text or text in {row_key, row_key.lstrip("-"), "-"}:
            continue
        values.append(text)
    if values:
        return values
    return [
        normalize_compare_text(token)
        for token in str(target.get("expected_text") or "").split()
        if normalize_compare_text(token) not in {row_key, row_key.lstrip("-"), "-"}
    ]


def _dedupe_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for line in sorted(lines, key=lambda item: (_line_center_y(item), _line_center_x(item))):
        line_id = str(line.get("ocr_line_id") or "")
        if line_id in seen:
            continue
        seen.add(line_id)
        unique.append(line)
    return unique


def _best_text_candidate(target: dict[str, Any], lines: list[dict[str, Any]], region_id: str) -> CompareCandidate | None:
    candidates = [_candidate(index, "nutrition_text", lines[start : start + length], "nutrition text window", region_id) for index, (start, length) in enumerate(_window_ranges(len(lines), 8), start=1)]
    if not candidates:
        return None
    expected_norm = str(target.get("normalized_expected_text") or "")
    return max((_with_score(candidate, _candidate_score(expected_norm, candidate.normalized_text, True), candidate.reason) for candidate in candidates), key=lambda item: item.score)


def _candidate_pool_for_target(
    target: dict[str, Any],
    layout: dict[str, Any],
    candidate_pool: list[CompareCandidate],
) -> list[CompareCandidate]:
    scope_id = str(target.get("scope_id") or "")
    if scope_id:
        line_ids = _line_ids_for_scope(layout, scope_id)
        return [
            candidate
            for candidate in candidate_pool
            if candidate.region_id == scope_id
            or (line_ids and candidate.source_ocr_line_ids and all(line_id in line_ids for line_id in candidate.source_ocr_line_ids))
        ]

    group_id = str(target.get("group_id") or "")
    if group_id.startswith("content_c"):
        segment_ids = {
            str(segment.get("segment_id"))
            for segment in _as_list(layout.get("content_segments"))
            if str(segment.get("segment_id")) == group_id
        }
        line_ids = {
            line_id
            for segment in _as_list(layout.get("content_segments"))
            if str(segment.get("segment_id")) == group_id
            for line_id in _as_list(segment.get("source_ocr_line_ids"))
        }
        return [
            candidate
            for candidate in candidate_pool
            if candidate.region_id in segment_ids
            or (line_ids and candidate.source_ocr_line_ids and all(line_id in line_ids for line_id in candidate.source_ocr_line_ids))
        ]
    return candidate_pool


def _sequence_candidate_for_target(target: dict[str, Any], pool: list[CompareCandidate], expected_norm: str) -> CompareCandidate | None:
    if not expected_norm or len(expected_norm) < _sequence_min_length(target):
        return None
    atomic = [
        candidate
        for candidate in pool
        if candidate.candidate_type in {"line", "split_line"}
        and candidate.text
        and not _candidate_has_noise(candidate)
    ]
    atomic.sort(key=_candidate_position)
    best: CompareCandidate | None = None
    for start, candidate in enumerate(atomic):
        selected: list[CompareCandidate] = []
        cursor = 0
        for item in atomic[start:]:
            norm = _match_normalize_for_target(target, item.text)
            if not norm or norm not in expected_norm:
                continue
            pos = expected_norm.find(norm, cursor)
            if pos < 0:
                continue
            if not selected and pos > 0:
                continue
            selected.append(item)
            cursor = pos + len(norm)
            combined = "".join(_match_normalize_for_target(target, part.text) for part in selected)
            if combined == expected_norm:
                best = _combine_candidates_as_sequence(target, selected)
                break
        if best:
            break
    return best


def _sequence_min_length(target: dict[str, Any]) -> int:
    if str(target.get("semantic_key") or "") == "custom.other_label_text":
        return 6
    return 12


def _combine_candidates_as_sequence(target: dict[str, Any], selected: list[CompareCandidate]) -> CompareCandidate:
    text = "\n".join(candidate.text for candidate in selected)
    return _text_candidate(
        8500,
        "sequence_recall",
        text,
        [line_id for candidate in selected for line_id in candidate.source_ocr_line_ids],
        _union_bbox([candidate.bbox_normalized for candidate in selected]),
        "target text reconstructed from ordered OCR lines",
        None,
        {"semantic_key": target.get("semantic_key")},
    )


def _candidate_position(candidate: CompareCandidate) -> tuple[float, float]:
    bbox = _as_dict(candidate.bbox_normalized)
    return ((float(bbox.get("y1", 0)) + float(bbox.get("y2", 0))) / 2, (float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2)


def _barcode_candidates(barcode_decode: Any) -> list[CompareCandidate]:
    decoded = _as_dict(barcode_decode)
    candidates = []
    for item in _as_list(decoded.get("items")):
        value = str(_as_dict(item).get("text") or "").strip()
        if not value:
            continue
        candidates.append(
            _text_candidate(
                len(candidates) + 7000,
                "barcode_decoder",
                value,
                [],
                None,
                "barcode decoder result",
                None,
                {"decoder": _as_dict(item).get("decoder"), "format": _as_dict(item).get("format")},
            )
        )
    return candidates


def _decode_barcodes(image_path: Path) -> dict[str, Any]:
    try:
        import zxingcpp  # type: ignore
    except Exception:
        return {
            "status": "unavailable",
            "decoder": "zxingcpp",
            "items": [],
            "quality_flags": ["barcode_decoder_unavailable"],
        }
    try:
        decoded = zxingcpp.read_barcodes(str(image_path))
    except Exception as exc:
        return {
            "status": "failed",
            "decoder": "zxingcpp",
            "error": str(exc),
            "items": [],
            "quality_flags": ["barcode_decoder_failed"],
        }
    items = [
        {
            "text": str(item.text),
            "format": str(getattr(item, "format", "")),
            "decoder": "zxingcpp",
        }
        for item in decoded
        if str(getattr(item, "text", "")).strip()
    ]
    return {
        "status": "decoded" if items else "not_decoded",
        "decoder": "zxingcpp",
        "items": items,
        "quality_flags": [] if items else ["barcode_not_decoded"],
    }


def _assign_target_scopes(targets: list[dict[str, Any]], layout: dict[str, Any], line_items: list[dict[str, Any]]) -> None:
    for target in targets:
        group_id = str(target.get("group_id") or "")
        target_type = str(target.get("target_type") or "")
        table_id = str(target.get("table_id") or "")
        if group_id.startswith("content_c"):
            target["scope_id"] = group_id
        elif target_type.startswith("nutrition_") and table_id:
            target["scope_id"] = f"region_{table_id}"

    enterprise_group_names: dict[str, dict[str, Any]] = {}
    for target in targets:
        if target.get("category") != "enterprise":
            continue
        group_id = str(target.get("group_id") or "")
        semantic_key = str(target.get("semantic_key") or "")
        if group_id and semantic_key.endswith(".name"):
            enterprise_group_names[group_id] = target

    used_segments: set[str] = set()
    lines_by_id = {str(line.get("ocr_line_id")): line for line in line_items}
    for group_id, name_target in enterprise_group_names.items():
        expected = str(name_target.get("normalized_expected_text") or normalize_compare_text(name_target.get("expected_text", "")))
        best_segment: tuple[float, dict[str, Any]] | None = None
        for segment in _as_list(layout.get("enterprise_segments")):
            if not isinstance(segment, dict):
                continue
            segment_id = str(segment.get("segment_id") or "")
            if segment_id in used_segments:
                continue
            score = _enterprise_scope_score(expected, segment, lines_by_id)
            if best_segment is None or score > best_segment[0]:
                best_segment = (score, segment)
        if best_segment and best_segment[0] >= 0.72:
            segment = best_segment[1]
            segment["group_id"] = group_id
            used_segments.add(str(segment.get("segment_id")))
            for target in targets:
                if str(target.get("group_id") or "") == group_id:
                    target["scope_id"] = str(segment.get("segment_id"))


def _enterprise_scope_score(expected_norm: str, segment: dict[str, Any], lines_by_id: dict[str, dict[str, Any]]) -> float:
    segment_lines = [
        lines_by_id[line_id]
        for line_id in _as_list(segment.get("source_ocr_line_ids"))
        if line_id in lines_by_id
    ]
    segment_lines.sort(key=lambda item: (_line_center_y(item), _line_center_x(item)))
    anchor_id = str(segment.get("anchor_ocr_line_id") or "")
    anchor_index = next((index for index, line in enumerate(segment_lines) if str(line.get("ocr_line_id")) == anchor_id), 0)
    candidate_texts = [str(segment.get("anchor_text") or "")]
    for length in range(2, min(5, len(segment_lines) - anchor_index) + 1):
        candidate_texts.append("\n".join(str(line.get("text") or "") for line in segment_lines[anchor_index : anchor_index + length]))
    scores = [_scope_text_similarity(expected_norm, normalize_compare_text(text)) for text in candidate_texts if text.strip()]
    return max(scores) if scores else 0.0


def _scope_text_similarity(expected_norm: str, candidate_norm: str) -> float:
    if not expected_norm or not candidate_norm:
        return 0.0
    if expected_norm == candidate_norm:
        return 1.0
    return difflib.SequenceMatcher(None, expected_norm, candidate_norm).ratio()


def _line_ids_for_scope(layout: dict[str, Any], scope_id: str) -> set[str]:
    for key in ("content_segments", "enterprise_segments", "nutrition_table_regions"):
        for segment in _as_list(layout.get(key)):
            if str(_as_dict(segment).get("segment_id") or _as_dict(segment).get("region_id") or "") == scope_id:
                return {str(line_id) for line_id in _as_list(_as_dict(segment).get("source_ocr_line_ids"))}
    return set()


def _candidate_matches(expected_norm: str, candidate_norm: str, long_text: bool) -> bool:
    if not expected_norm:
        return False
    if long_text:
        return expected_norm == candidate_norm or expected_norm in candidate_norm
    return expected_norm == candidate_norm


def _candidate_matches_for_target(target: dict[str, Any], expected_norm: str, candidate_norm: str, long_text: bool) -> bool:
    if not expected_norm or not _prefix_compatible(expected_norm, candidate_norm):
        return False
    return _candidate_matches(expected_norm, candidate_norm, long_text)


def _is_viable_mismatch_candidate(target: dict[str, Any], expected_norm: str, candidate: CompareCandidate) -> bool:
    candidate_norm = _match_normalize_for_target(target, candidate.text)
    if candidate.score >= (0.55 if _is_long_target(target) else 0.72):
        if expected_norm in candidate_norm and len(candidate_norm) > max(len(expected_norm) * 2, len(expected_norm) + 8):
            return False
        return True
    prefix = _expected_prefix(expected_norm)
    return bool(prefix and prefix in candidate_norm)


def _candidate_score(expected_norm: str, candidate_norm: str, long_text: bool) -> float:
    if not expected_norm or not candidate_norm:
        return 0.0
    if expected_norm == candidate_norm:
        return 1.0
    if expected_norm in candidate_norm:
        return 0.88 if long_text else 0.65
    if candidate_norm in expected_norm:
        return 0.7 if long_text else 0.45
    return difflib.SequenceMatcher(None, expected_norm, candidate_norm).ratio()


def _is_long_target(target: dict[str, Any]) -> bool:
    semantic_key = str(target.get("semantic_key") or "")
    expected = str(target.get("expected_text") or "")
    return semantic_key in LONG_TEXT_KEYS or semantic_key.endswith(".ingredients") or len(normalize_compare_text(expected)) > 40


def _is_barcode_target(target: dict[str, Any]) -> bool:
    return str(target.get("semantic_key") or "").startswith("barcode.")


def _match_normalize_for_target(target: dict[str, Any], text: str) -> str:
    normalized = normalize_compare_text(text)
    if _allows_separator_insensitive_match(target):
        prefix, value = _prefix_parts(normalized)
        value = _strip_long_text_separators(value)
        return f"{prefix}:{value}" if prefix else value
    return normalized


def _allows_separator_insensitive_match(target: dict[str, Any]) -> bool:
    semantic_key = str(target.get("semantic_key") or "")
    return semantic_key.endswith(".ingredients") or semantic_key in {"product.warning", "product.directions"}


def _strip_long_text_separators(value: str) -> str:
    value = re.sub(r"(?<!\d)[,.;](?!\d)", "", value)
    return value


def _prefix_parts(normalized_text: str) -> tuple[str, str]:
    if ":" not in normalized_text:
        return "", normalized_text
    prefix, value = normalized_text.split(":", 1)
    return prefix, value


def _prefix_compatible(expected_norm: str, candidate_norm: str) -> bool:
    expected_prefix, _ = _prefix_parts(expected_norm)
    if not expected_prefix:
        return True
    candidate_prefix, _ = _prefix_parts(candidate_norm)
    return expected_prefix == candidate_prefix


def _expected_prefix(expected_norm: str) -> str:
    if ":" not in expected_norm:
        return ""
    prefix = expected_norm.split(":", 1)[0]
    return prefix if len(prefix) >= 2 else ""


def _result(target: dict[str, Any], status: str, reason: str, candidate: CompareCandidate | None) -> dict[str, Any]:
    issue_category = _issue_category(target, status, reason, candidate)
    issue_origin = _issue_origin(issue_category, status, reason)
    return {
        "target_id": target.get("target_id"),
        "target_type": target.get("target_type"),
        "category": target.get("category"),
        "semantic_key": target.get("semantic_key"),
        "label": target.get("label"),
        "expected_text": target.get("expected_text"),
        "normalized_expected_text": target.get("normalized_expected_text"),
        "status": status,
        "severity": _severity_for_status(status),
        "reason": reason,
        "issue_category": issue_category,
        "issue_origin": issue_origin,
        "resolution_hint": _resolution_hint(issue_category, status, reason),
        "match_quality_flags": _match_quality_flags(target, status, reason, candidate),
        "selected_candidate": to_jsonable(candidate) if candidate else None,
        "source_standard_item_ids": target.get("source_standard_item_ids", []),
        "group_id": target.get("group_id"),
        "scope_id": target.get("scope_id"),
        "table_id": target.get("table_id"),
        "row_key": target.get("row_key"),
    }


def _manual_review_result(target: dict[str, Any], reason: str, candidate: CompareCandidate | None) -> dict[str, Any]:
    return _result(target, "manual_review", reason, candidate)


def _severity_for_status(status: str) -> str:
    return {
        "pass": "none",
        "critical_missing": "critical",
        "critical_mismatch": "critical",
        "manual_review": "review",
        "info_extra_text": "info",
    }.get(status, "review")


def _issue_category(target: dict[str, Any], status: str, reason: str, candidate: CompareCandidate | None) -> str:
    if status == "pass":
        if candidate and candidate.candidate_type == "barcode_decoder":
            return "barcode_decode"
        return "none"
    if reason in {"barcode_decoder_unavailable", "barcode_not_decoded"}:
        return "barcode_decode"
    if candidate and candidate.candidate_type == "split_line":
        return "multi_field_ocr_line"
    if candidate and _candidate_has_noise(candidate):
        return "candidate_contamination"
    if "sugar_sodium" in reason or "table" in reason or "nutrition_row" in reason:
        return "table_reconstruction"
    if "nutrition_footnote_not_found" in reason:
        return "ocr_or_print_missing"
    if reason in {"no_reliable_candidate_found", "nutrition_table_region_not_found"} or status == "critical_missing":
        return "candidate_recall"
    if status == "critical_mismatch":
        return "text_difference"
    if status == "manual_review":
        return "manual_review"
    return "matcher_uncertain"


def _issue_origin(issue_category: str, status: str, reason: str) -> str:
    if issue_category == "barcode_decode":
        return "ocr"
    if issue_category in {"table_reconstruction", "ocr_or_print_missing"}:
        return "ocr"
    if issue_category in {"candidate_recall", "text_difference", "multi_field_ocr_line"}:
        return "package_image"
    return "matcher"


def _resolution_hint(issue_category: str, status: str, reason: str) -> str:
    if status == "pass":
        return ""
    if issue_category == "table_reconstruction":
        return "请人工核对营养表区域；OCR 可能存在行名粘连或表格列轨道不稳定。"
    if issue_category == "barcode_decode":
        return "请检查条码是否清晰可解码；当前环境可能缺少条码解码依赖或图像中未识别到条码。"
    if issue_category == "candidate_contamination":
        return "候选文字混入了相邻区域或无关短文本，请复核段落边界。"
    if issue_category == "multi_field_ocr_line":
        return "OCR 同一行包含多个字段，已按字段前缀拆分；请核对拆分边界。"
    if issue_category == "ocr_or_print_missing":
        return "请检查包装图中是否实际印刷该文字；若已印刷，需复核 OCR 识别结果。"
    if issue_category == "candidate_recall":
        return "请确认包装图是否缺少该标准字段，或字段是否位于未覆盖/低清晰区域。"
    if issue_category == "text_difference":
        return "请核对包装图候选文字与标准文字，前缀、数字、单位和关键文字必须一致。"
    if issue_category == "manual_review":
        return "该项不适合自动判定，请人工复核标准项与包装图候选。"
    return "请复核候选召回与匹配边界。"


def _match_quality_flags(target: dict[str, Any], status: str, reason: str, candidate: CompareCandidate | None) -> list[str]:
    flags = []
    if target.get("scope_id"):
        flags.append("scope_constrained")
    if candidate and candidate.candidate_type == "split_line":
        flags.append("multi_field_ocr_line")
    if candidate and candidate.candidate_type == "barcode_decoder":
        flags.append("barcode_decoder_match")
    if candidate and "separator_insensitive" in candidate.reason:
        flags.append("separator_insensitive_match")
    if candidate and _candidate_has_noise(candidate):
        flags.append("candidate_contamination")
    if "sugar_sodium" in reason or (candidate and "糖钠" in normalize_compare_text(candidate.text)):
        flags.append("merged_ocr_row_label")
    if status == "pass" and candidate and len(candidate.source_ocr_line_ids) == 1:
        flags.append("single_line_exact_boundary")
    if status == "critical_mismatch" and candidate:
        flags.append("candidate_found_text_differs")
    if status == "critical_missing":
        flags.append("no_reliable_candidate")
    if reason in {"barcode_decoder_unavailable", "barcode_not_decoded"}:
        flags.append(reason)
    return flags


def _candidate(index: int, candidate_type: str, lines: list[dict[str, Any]], reason: str, region_id: str | None) -> CompareCandidate:
    clean_lines = [line for line in lines if isinstance(line, dict)]
    text = "\n".join(str(line.get("text") or "") for line in clean_lines).strip()
    return CompareCandidate(
        candidate_id=stable_id("cand", index),
        candidate_type=candidate_type,
        text=text,
        normalized_text=normalize_compare_text(text),
        source_ocr_line_ids=[str(line.get("ocr_line_id")) for line in clean_lines if line.get("ocr_line_id")],
        bbox_normalized=_union_bbox([line.get("bbox_normalized") for line in clean_lines]),
        score=0.0,
        reason=reason,
        region_id=region_id,
        metadata=None,
    )


def _text_candidate(
    index: int,
    candidate_type: str,
    text: str,
    line_ids: list[str],
    bbox_normalized: dict[str, float] | None,
    reason: str,
    region_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> CompareCandidate:
    return CompareCandidate(
        candidate_id=stable_id("cand", index),
        candidate_type=candidate_type,
        text=text.strip(),
        normalized_text=normalize_compare_text(text),
        source_ocr_line_ids=line_ids,
        bbox_normalized=bbox_normalized,
        score=0.0,
        reason=reason,
        region_id=region_id,
        metadata=metadata,
    )


def _with_score(candidate: CompareCandidate, score: float, reason: str) -> CompareCandidate:
    return CompareCandidate(
        candidate_id=candidate.candidate_id,
        candidate_type=candidate.candidate_type,
        text=candidate.text,
        normalized_text=candidate.normalized_text,
        source_ocr_line_ids=candidate.source_ocr_line_ids,
        bbox_normalized=candidate.bbox_normalized,
        score=round(score, 4),
        reason=reason,
        region_id=candidate.region_id,
        metadata=candidate.metadata,
    )


def _best_candidate(candidates: list[CompareCandidate]) -> CompareCandidate:
    return sorted(candidates, key=lambda item: (item.score, -len(item.source_ocr_line_ids)), reverse=True)[0]


def _unmatched_print_text(line_items: list[dict[str, Any]], used_line_ids: set[str]) -> list[dict[str, Any]]:
    items = []
    for line in line_items:
        if line["ocr_line_id"] in used_line_ids:
            continue
        items.append(
            {
                "status": "info_extra_text",
                "severity": "info",
                "text": line["text"],
                "ocr_line_id": line["ocr_line_id"],
                "bbox_normalized": line.get("bbox_normalized"),
                "confidence": line.get("confidence"),
            }
        )
    return items


def _ocr_line_item(line: OcrLine) -> dict[str, Any]:
    return {
        "ocr_line_id": line.ocr_line_id,
        "page": line.page,
        "text": line.text,
        "normalized_text": normalize_compare_text(line.text),
        "confidence": line.confidence,
        "bbox_normalized": to_jsonable(line.bbox_normalized),
        "bbox_pdf": to_jsonable(line.bbox_pdf),
        "block_id": line.block_id,
        "metadata": to_jsonable(line.metadata),
    }


def _best_table_for_title(title_text: str, tables: list[dict[str, Any]]) -> dict[str, Any] | None:
    title_norm = normalize_compare_text(title_text)
    best: tuple[float, dict[str, Any]] | None = None
    for table in tables:
        candidate_norm = normalize_compare_text(str(table.get("title") or ""))
        score = _candidate_score(candidate_norm, title_norm, False)
        if best is None or score > best[0]:
            best = (score, table)
    return best[1] if best and best[0] >= 0.75 else None


def _next_table_title_y(current: dict[str, Any], titles: list[dict[str, Any]], x1: float, x2: float) -> float | None:
    current_y = _line_center_y(current)
    candidates = [
        _line_center_y(line)
        for line in titles
        if _line_center_y(line) > current_y and x1 <= _line_center_x(line) <= x2
    ]
    return min(candidates) if candidates else None


def _window_ranges(size: int, max_len: int) -> list[tuple[int, int]]:
    ranges = []
    for start in range(size):
        for length in range(1, min(max_len, size - start) + 1):
            ranges.append((start, length))
    return ranges


def _line_center_x(line: dict[str, Any]) -> float:
    bbox = _as_dict(line.get("bbox_normalized"))
    return (float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2


def _line_center_y(line: dict[str, Any]) -> float:
    bbox = _as_dict(line.get("bbox_normalized"))
    return (float(bbox.get("y1", 0)) + float(bbox.get("y2", 0))) / 2


def _union_bbox(boxes: list[Any]) -> dict[str, float] | None:
    normalized = []
    for box in boxes:
        item = _as_dict(box)
        if not item:
            continue
        try:
            normalized.append((float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not normalized:
        return None
    return {
        "x1": round(min(item[0] for item in normalized), 6),
        "y1": round(min(item[1] for item in normalized), 6),
        "x2": round(max(item[2] for item in normalized), 6),
        "y2": round(max(item[3] for item in normalized), 6),
    }


def _ocr_bbox_range(ocr_lines: list[OcrLine]) -> dict[str, float] | None:
    return _union_bbox([to_jsonable(line.bbox_normalized) for line in ocr_lines if line.bbox_normalized is not None])


def _safe_ratio(width: int, height: int) -> float:
    return float(width) / float(height) if height else 0.0


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ImageCompareError(f"Invalid JSON: {path}") from exc


def _read_json_if_exists(path: Path, default: Any) -> Any:
    return _read_json(path) if path.exists() else default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
