from __future__ import annotations

import fnmatch
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SEMANTIC_KEY_ALIASES = {
    "custom.allergen_statement": "custom.allergen_notice",
    "custom.usage_instruction": "product.directions",
    "custom.warning_statement": "product.warning",
    "product.brand_text": "custom.brand_text",
    "product.production_date_mark": "product.date_marking",
    "manufacturer.food_production_license": "manufacturer.license_number",
}

LABEL_PREFIXES = {
    "barcode.commodity": ("商品条码", "条码"),
    "barcode.outer_case": ("外箱条码",),
    "content_item.ingredients": ("配料表", "配料"),
    "content_item.name": ("内容物名称",),
    "content_item.product_category": ("产品分类",),
    "manufacturer.address": ("生产地址", "地址"),
    "manufacturer.contact": ("客户服务热线", "消费者服务电话", "电话", "联系方式"),
    "manufacturer.license_number": ("食品生产许可证编号", "许可证编号"),
    "manufacturer.name": ("被委托方", "受托方", "生产者", "生产商"),
    "manufacturer.origin": ("产地",),
    "principal.address": ("地址",),
    "principal.contact": ("消费者服务电话", "电话", "联系方式"),
    "principal.name": ("委托方",),
    "product.date_marking": ("生产日期",),
    "product.directions": ("烹调加工方法", "冲调方法", "食用方法"),
    "product.ingredients": ("配料表", "配料"),
    "product.name": ("产品名称", "品名"),
    "product.net_content": ("净含量/规格", "净含量", "规格"),
    "product.product_type": ("产品类型",),
    "product.shelf_life": ("礼盒保质期", "保质期"),
    "product.standard_code": ("产品标准代号", "产品标准号"),
    "product.storage_condition": ("贮存条件",),
    "product.warning": ("温馨提示", "提示", "警示语"),
}

NUTRIENT_KEYS = {
    "能量",
    "蛋白质",
    "脂肪",
    "饱和脂肪",
    "反式脂肪酸",
    "碳水化合物",
    "糖",
    "钠",
}


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    pdf_pattern: str
    reference_xlsx: Path


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    version: str
    cases: tuple[BenchmarkCase, ...]

    def match(self, pdf_name: str) -> BenchmarkCase | None:
        return next((case for case in self.cases if fnmatch.fnmatch(pdf_name, case.pdf_pattern)), None)


@dataclass(frozen=True, slots=True)
class BenchmarkThresholds:
    critical_field_exact_recall: float = 0.95
    field_exact_recall: float = 0.90
    repeated_entity_group_recall: float = 0.95
    nutrition_table_recall: float = 1.0
    nutrition_row_value_accuracy: float = 1.0
    nutrition_cell_boundary_conformance: float = 1.0
    extracted_out_of_scope_count: int = 0
    unresolved_high_risk_count: int = 0

    @classmethod
    def disabled(cls) -> BenchmarkThresholds:
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2**31 - 1, 2**31 - 1)


def load_benchmark_manifest(path: Path) -> BenchmarkManifest:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Cannot read benchmark manifest {path}: {exc}") from exc
    if body.get("version") != "benchmark_manifest_v0.1" or not isinstance(body.get("cases"), list):
        raise BenchmarkError("Benchmark manifest must use version benchmark_manifest_v0.1 and contain cases.")
    cases = []
    for item in body["cases"]:
        if not isinstance(item, dict) or not all(item.get(key) for key in ("case_id", "pdf_pattern", "reference_xlsx")):
            raise BenchmarkError("Every benchmark case requires case_id, pdf_pattern and reference_xlsx.")
        reference = Path(str(item["reference_xlsx"])).expanduser()
        if not reference.exists():
            raise BenchmarkError(f"Benchmark XLSX does not exist: {reference}")
        cases.append(BenchmarkCase(str(item["case_id"]), str(item["pdf_pattern"]), reference))
    return BenchmarkManifest(str(body["version"]), tuple(cases))


def canonical_semantic_key(semantic_key: str) -> str:
    return SEMANTIC_KEY_ALIASES.get(semantic_key, semantic_key)


def normalize_benchmark_value(value: str, semantic_key: str) -> str:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))
    key = canonical_semantic_key(semantic_key)
    prefixes = sorted(LABEL_PREFIXES.get(key, ()), key=len, reverse=True)
    for prefix in prefixes:
        match = re.match(rf"^\s*{re.escape(prefix)}\s*[:：]?\s*", normalized, re.IGNORECASE)
        if match and match.end() > 0:
            normalized = normalized[match.end() :]
            break
    return normalized


def evaluate_benchmark(
    *,
    case_id: str,
    current: dict[str, Any],
    reference: dict[str, Any],
    thresholds: BenchmarkThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_fields = [_field_with_evidence(field, current) for field in _fields(current)]
    reference_fields = [field for field in _fields(reference) if canonical_semantic_key(str(field.get("semantic_key", ""))) != "product.nutrition_table"]
    current_entities = _entity_types(current)
    reference_entities = _entity_types(reference)
    matches, missing, unexpected = _match_fields(current_fields, reference_fields, current_entities, reference_entities)
    exact = [item for item in matches if item["status"] == "exact"]
    near = [item for item in matches if item["status"] == "near"]
    critical_expected = [field for field in reference_fields if field.get("criticality") == "critical"]
    critical_exact = [item for item in exact if item["expected"].get("criticality") == "critical"]
    entity_recall, entity_diff = _repeated_entity_recall(current_fields, reference_fields, current_entities, reference_entities)
    table_metrics, table_diff = _evaluate_tables(_tables(current), _tables(reference))
    out_of_scope = int(_nested(current, "metadata", "label_text_scope_report", "extracted_out_of_scope_count") or 0)
    high_risk = _unresolved_high_risk_count(current)
    field_expected_count = len(reference_fields)
    field_exact_recall = _ratio(len(exact), field_expected_count)
    critical_exact_recall = _ratio(len(critical_exact), len(critical_expected))
    checks = {
        "critical_field_exact_recall": critical_exact_recall >= thresholds.critical_field_exact_recall,
        "field_exact_recall": field_exact_recall >= thresholds.field_exact_recall,
        "repeated_entity_group_recall": entity_recall >= thresholds.repeated_entity_group_recall,
        "nutrition_table_recall": table_metrics["nutrition_table_recall"] >= thresholds.nutrition_table_recall,
        "nutrition_row_value_accuracy": table_metrics["nutrition_row_value_accuracy"] >= thresholds.nutrition_row_value_accuracy,
        "nutrition_cell_boundary_conformance": table_metrics["nutrition_cell_boundary_conformance"] >= thresholds.nutrition_cell_boundary_conformance,
        "extracted_out_of_scope_count": out_of_scope <= thresholds.extracted_out_of_scope_count,
        "unresolved_high_risk_count": high_risk <= thresholds.unresolved_high_risk_count,
    }
    evaluation = {
        "benchmark_version": "benchmark_evaluation_v0.1",
        "case_id": case_id,
        "status": "pass" if all(checks.values()) else "failed",
        "field_expected_count": field_expected_count,
        "field_current_count": len(current_fields),
        "field_exact_count": len(exact),
        "field_near_count": len(near),
        "field_missing_count": len(missing),
        "field_exact_recall": field_exact_recall,
        "critical_field_exact_recall": critical_exact_recall,
        "repeated_entity_group_recall": entity_recall,
        **table_metrics,
        "extracted_out_of_scope_count": out_of_scope,
        "unresolved_high_risk_count": high_risk,
        "checks": checks,
    }
    diff = {
        "benchmark_diff_version": "benchmark_diff_v0.1",
        "case_id": case_id,
        "field_matches": matches,
        "missing_fields": missing,
        "unexpected_fields": unexpected,
        "entity_groups": entity_diff,
        "diagnostics": {
            "unknown_nodes": _nested(current, "metadata", "vdg_consumption_report", "unknown_important_nodes") or [],
            "conflict_nodes": _nested(current, "metadata", "vdg_consumption_report", "conflict_nodes") or [],
            "semantic_review": _nested(current, "metadata", "semantic_review_report") or {},
            "repair_history": _nested(current, "metadata", "repair_loop", "trace") or {},
        },
        **table_diff,
    }
    return evaluation, diff


def _match_fields(current: list[dict[str, Any]], expected: list[dict[str, Any]], current_entities: dict[str, str], expected_entities: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    for expected_index, expected_field in enumerate(expected):
        expected_key = canonical_semantic_key(str(expected_field.get("semantic_key", "")))
        expected_role = expected_entities.get(str(expected_field.get("entity_id") or ""), _key_role(expected_key))
        expected_value = normalize_benchmark_value(_field_value(expected_field), expected_key)
        for current_index, current_field in enumerate(current):
            current_key = canonical_semantic_key(str(current_field.get("semantic_key", "")))
            current_role = current_entities.get(str(current_field.get("entity_id") or ""), _key_role(current_key))
            if current_key != expected_key or current_role != expected_role:
                continue
            current_value = normalize_benchmark_value(_field_value(current_field), current_key)
            score = SequenceMatcher(None, expected_value, current_value).ratio()
            status = "exact" if expected_value == current_value else ("near" if score >= 0.85 else "mismatch")
            candidates.append((status == "exact", score, expected_index, current_index, status))
    used_expected: set[int] = set()
    used_current: set[int] = set()
    matches = []
    for _, score, expected_index, current_index, status in sorted(candidates, reverse=True):
        if expected_index in used_expected or current_index in used_current:
            continue
        used_expected.add(expected_index)
        used_current.add(current_index)
        matches.append(
            {
                "status": status,
                "similarity": round(score, 4),
                "semantic_key": canonical_semantic_key(str(expected[expected_index].get("semantic_key", ""))),
                "expected": expected[expected_index],
                "current": current[current_index],
            }
        )
    missing = [field for index, field in enumerate(expected) if index not in used_expected]
    unexpected = [field for index, field in enumerate(current) if index not in used_current]
    return matches, missing, unexpected


def _evaluate_tables(current: list[dict[str, Any]], expected: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    current_tables = [table for table in current if table.get("table_type") == "nutrition_facts"]
    expected_tables = [table for table in expected if table.get("table_type") == "nutrition_facts"]
    candidates = []
    for expected_index, expected_table in enumerate(expected_tables):
        expected_rows = _nutrition_rows(expected_table)
        for current_index, current_table in enumerate(current_tables):
            current_rows = _nutrition_rows(current_table)
            if not expected_rows or not current_rows:
                continue
            exact_rows = sum(current_rows.get(key) == value for key, value in expected_rows.items())
            title_match = normalize_benchmark_value(str(expected_table.get("title", "")), "table.title") == normalize_benchmark_value(str(current_table.get("title", "")), "table.title")
            candidates.append((exact_rows, title_match, expected_index, current_index, len(expected_rows)))
    used_expected: set[int] = set()
    used_current: set[int] = set()
    matches = []
    exact_row_count = 0
    conforming_tables = 0
    for exact_rows, title_match, expected_index, current_index, expected_row_count in sorted(candidates, reverse=True):
        if expected_index in used_expected or current_index in used_current:
            continue
        if not title_match and exact_rows == 0:
            continue
        used_expected.add(expected_index)
        used_current.add(current_index)
        exact_row_count += exact_rows
        expected_row_values = _nutrition_rows(expected_tables[expected_index])
        current_row_values = _nutrition_rows(current_tables[current_index])
        boundary_conforms, boundary_issues = _table_boundary_conformance(current_tables[current_index])
        conforming_tables += int(boundary_conforms)
        matches.append(
            {
                "expected_table_id": expected_tables[expected_index].get("table_id"),
                "current_table_id": current_tables[current_index].get("table_id"),
                "expected_title": expected_tables[expected_index].get("title"),
                "current_title": current_tables[current_index].get("title"),
                "exact_row_count": exact_rows,
                "expected_row_count": expected_row_count,
                "boundary_conforms": boundary_conforms,
                "boundary_issues": boundary_issues,
                "row_diffs": [
                    {
                        "row_label": label,
                        "status": "exact" if current_row_values.get(label) == value else "mismatch" if label in current_row_values else "missing",
                        "expected": value,
                        "current": current_row_values.get(label),
                    }
                    for label, value in expected_row_values.items()
                ],
                "current_bbox_pdf": current_tables[current_index].get("bbox_pdf"),
                "current_evidence_refs": current_tables[current_index].get("evidence_refs", []),
            }
        )
    expected_row_total = sum(len(_nutrition_rows(table)) for table in expected_tables)
    expected_table_count = len(expected_tables)
    metrics = {
        "nutrition_table_expected_count": expected_table_count,
        "nutrition_table_current_count": len(current_tables),
        "nutrition_table_recall": _ratio(len(matches), expected_table_count),
        "nutrition_row_expected_count": expected_row_total,
        "nutrition_row_exact_count": exact_row_count,
        "nutrition_row_value_accuracy": _ratio(exact_row_count, expected_row_total),
        "nutrition_cell_boundary_conformance": _ratio(conforming_tables, expected_table_count),
    }
    diff = {
        "table_matches": matches,
        "missing_tables": [table for index, table in enumerate(expected_tables) if index not in used_expected],
    }
    return metrics, diff


def _nutrition_rows(table: dict[str, Any]) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for row in table.get("rows", []):
        cells = [str(cell.get("raw_value", "")) for cell in row.get("cells", []) if isinstance(cell, dict)]
        if not cells:
            continue
        label_index = 1 if _is_marker(cells[0]) and len(cells) > 1 else 0
        label = re.sub(r"^[-—]+", "", normalize_benchmark_value(cells[label_index], "nutrition.label"))
        if label not in NUTRIENT_KEYS:
            continue
        values = cells[label_index + 1 :]
        nrv = normalize_benchmark_value(values[-1], "nutrition.nrv") if values and ("%" in values[-1] or values[-1].strip() in {"", "-", "—"}) else ""
        amount_cells = values[:-1] if nrv or (values and values[-1].strip() == "") else values
        amount = normalize_benchmark_value("".join(amount_cells), "nutrition.amount")
        rows[label] = (amount, nrv)
    return rows


def _table_boundary_conformance(table: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    issues = []
    column_count = len(table.get("columns", []))
    for row in table.get("rows", []):
        cells = row.get("cells", [])
        if not isinstance(cells, list) or not cells:
            continue
        first_value = str(cells[0].get("raw_value", "")) if isinstance(cells[0], dict) else ""
        if _is_marker(first_value):
            issues.append({"row_key": row.get("row_key"), "reason": "row_marker_in_own_cell"})
        if column_count and len(cells) != column_count:
            issues.append({"row_key": row.get("row_key"), "reason": "row_column_count_mismatch", "actual": len(cells), "expected": column_count})
    return not issues, issues


def _fields(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fields = _nested(payload, "extracted_data", "fields")
    return list(fields.values()) if isinstance(fields, dict) else (fields if isinstance(fields, list) else [])


def _tables(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tables = _nested(payload, "extracted_data", "tables")
    return tables if isinstance(tables, list) else list(tables.values()) if isinstance(tables, dict) else []


def _entity_types(payload: dict[str, Any]) -> dict[str, str]:
    entities = _nested(payload, "extracted_data", "entities")
    values = list(entities.values()) if isinstance(entities, dict) else entities if isinstance(entities, list) else []
    return {str(entity.get("entity_id")): str(entity.get("entity_type")) for entity in values if isinstance(entity, dict) and entity.get("entity_id")}


def _repeated_entity_recall(
    current_fields: list[dict[str, Any]],
    expected_fields: list[dict[str, Any]],
    current_types: dict[str, str],
    expected_types: dict[str, str],
) -> tuple[float, list[dict[str, Any]]]:
    expected_groups = _entity_groups(expected_fields, expected_types)
    current_groups = _entity_groups(current_fields, current_types)
    expected_groups = [group for group in expected_groups if group["role"] != "product"]
    if not expected_groups:
        return 1.0, []
    used_current: set[int] = set()
    diff = []
    matched_count = 0
    for expected in expected_groups:
        candidates = []
        for index, current in enumerate(current_groups):
            if index in used_current or current["role"] != expected["role"] or current["name"] != expected["name"]:
                continue
            expected_details = expected["details"]
            exact_details = sum(current["details"].get(key) == value for key, value in expected_details.items())
            candidates.append((exact_details, index, current))
        if not candidates:
            diff.append({"status": "missing", "expected": expected, "current": None, "detail_mismatches": []})
            continue
        exact_details, current_index, current = max(candidates, key=lambda item: item[0])
        used_current.add(current_index)
        detail_mismatches = [
            {"semantic_key": key, "expected": value, "current": current["details"].get(key, "")}
            for key, value in expected["details"].items()
            if current["details"].get(key) != value
        ]
        status = "exact" if not detail_mismatches else "detail_mismatch"
        matched_count += int(status == "exact")
        diff.append({"status": status, "expected": expected, "current": current, "detail_mismatches": detail_mismatches})
    return _ratio(matched_count, len(expected_groups)), diff


def _entity_groups(fields: list[dict[str, Any]], entity_types: dict[str, str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        entity_id = str(field.get("entity_id") or "")
        if entity_id:
            grouped.setdefault(entity_id, []).append(field)
    groups = []
    for entity_id, entity_fields in grouped.items():
        role = entity_types.get(entity_id) or _key_role(canonical_semantic_key(str(entity_fields[0].get("semantic_key") or "")))
        if role not in {"principal", "manufacturer", "content_item", "barcode", "requirement", "product"}:
            continue
        values = {
            canonical_semantic_key(str(field.get("semantic_key") or "")): normalize_benchmark_value(
                _field_value(field),
                canonical_semantic_key(str(field.get("semantic_key") or "")),
            )
            for field in entity_fields
        }
        name_key = {
            "principal": "principal.name",
            "manufacturer": "manufacturer.name",
            "content_item": "content_item.name",
            "product": "product.name",
        }.get(role, f"{role}.name")
        detail_prefixes = {
            "principal": ("principal.address", "principal.contact"),
            "manufacturer": (
                "manufacturer.address",
                "manufacturer.license_number",
                "manufacturer.origin",
                "manufacturer.contact",
                "manufacturer.factory_code",
            ),
            "content_item": ("content_item.product_category", "content_item.ingredients"),
        }.get(role, ())
        name, embedded_factory_code = _split_entity_name(values.get(name_key, ""))
        details = {key: values[key] for key in detail_prefixes if values.get(key)}
        if role == "manufacturer" and embedded_factory_code:
            details.setdefault("manufacturer.factory_code", embedded_factory_code)
        groups.append(
            {
                "entity_id": entity_id,
                "role": role,
                "name": name,
                "details": details,
            }
        )
    return groups


def _split_entity_name(value: str) -> tuple[str, str]:
    without_index = re.sub(r"^\d+[:：]", "", value)
    match = re.search(r"[（(]工厂代码[:：]([^）)]+)[）)]", without_index)
    if not match:
        return without_index, ""
    name = (without_index[: match.start()] + without_index[match.end() :]).strip()
    return name, match.group(1).strip()


def _field_value(field: dict[str, Any]) -> str:
    return str(field.get("normalized_value") or field.get("clean_value") or field.get("raw_value") or field.get("value") or "")


def _field_with_evidence(field: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("evidence", {})
    evidence_items = list(evidence.values()) if isinstance(evidence, dict) else evidence if isinstance(evidence, list) else []
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in evidence_items
        if isinstance(item, dict) and item.get("evidence_id")
    }
    result = dict(field)
    result["benchmark_evidence"] = [
        evidence_by_id[ref]
        for ref in field.get("evidence_refs", [])
        if ref in evidence_by_id
    ]
    return result


def _key_role(key: str) -> str:
    prefix = key.split(".", 1)[0]
    return "product" if prefix == "custom" else prefix


def _is_marker(value: str) -> bool:
    return bool(re.fullmatch(r"\s*[-—]{1,3}\s*", value))


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 1.0


def _unresolved_high_risk_count(payload: dict[str, Any]) -> int:
    risks = payload.get("risks", [])
    reviews = payload.get("review_tasks", [])
    unresolved = {
        (str(item.get("target_id") or "document"), str(item.get("message") or item.get("reason") or item.get("risk_type") or "high_risk"))
        for item in [*risks, *reviews]
        if isinstance(item, dict) and (item.get("risk_level") == "high" or item.get("severity") == "high")
    }
    return len(unresolved)
