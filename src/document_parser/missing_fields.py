from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import CompiledField, Risk
from .utils import stable_id


@dataclass(frozen=True)
class RequiredField:
    semantic_key: str
    field: str
    label: str
    field_type: str


REQUIRED_CRITICAL_FIELDS: tuple[RequiredField, ...] = (
    RequiredField("product.name", "product_name", "品名", "string"),
    RequiredField("product.ingredients", "ingredients", "配料", "long_text"),
    RequiredField("product.net_content", "net_content", "净含量", "string"),
    RequiredField("product.standard_code", "standard_code", "产品标准代号", "string"),
    RequiredField("product.shelf_life", "shelf_life", "保质期", "string"),
    RequiredField("product.storage_condition", "storage", "贮存条件", "string"),
    RequiredField("manufacturer.name", "manufacturer", "生产商", "string"),
    RequiredField("manufacturer.address", "address", "地址", "long_text"),
    RequiredField("manufacturer.license_number", "license", "许可证编号", "string"),
    RequiredField("barcode.commodity", "barcode", "商品条码", "barcode"),
)


REQUIRED_CRITICAL_TABLES = (
    {
        "table_type": "nutrition_facts",
        "label": "营养成分表",
        "criticality": "critical",
    },
)


def build_missing_item_report(
    compiled_fields: dict[str, CompiledField],
    tables: list[dict[str, Any]],
) -> dict[str, Any]:
    extracted_field_keys = {
        field.semantic_key
        for field in compiled_fields.values()
        if field.raw_value and field.status not in {"missing", "cannot_verify", "rejected"}
    }
    extracted_table_types = {
        str(table.get("table_type"))
        for table in tables
        if _table_is_evidence_bound(table)
    }

    missing_fields = [
        _missing_field(index, required)
        for index, required in enumerate(REQUIRED_CRITICAL_FIELDS, start=1)
        if required.semantic_key not in extracted_field_keys
    ]
    missing_tables = [
        _missing_table(index, required)
        for index, required in enumerate(REQUIRED_CRITICAL_TABLES, start=1)
        if required["table_type"] not in extracted_table_types
    ]
    missing_count = len(missing_fields) + len(missing_tables)
    return {
        "status": "review_required" if missing_count else "pass",
        "expectation_source": "packaging_label_mvp_core_requirements",
        "missing_count": missing_count,
        "missing_field_count": len(missing_fields),
        "missing_table_count": len(missing_tables),
        "missing_fields": missing_fields,
        "missing_tables": missing_tables,
    }


def _table_is_evidence_bound(table: dict[str, Any]) -> bool:
    if not table.get("rows"):
        return False
    if table.get("status") in {"missing", "cannot_verify", "rejected"}:
        return False
    if table.get("evidence_refs"):
        return True
    return any(row.get("evidence_refs") for row in table.get("rows", []) if isinstance(row, dict))


def missing_item_validation_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for item in report.get("missing_fields", []):
        checks.append(
            {
                "validation_id": stable_id("missing_val", len(checks) + 1),
                "target_id": item["missing_id"],
                "check_type": "missing_required_field",
                "semantic_key": item["semantic_key"],
                "result": "failed",
                "severity": "high",
                "message": item["reason"],
                "evidence_refs": [],
            }
        )
    for item in report.get("missing_tables", []):
        checks.append(
            {
                "validation_id": stable_id("missing_val", len(checks) + 1),
                "target_id": item["missing_id"],
                "check_type": "missing_required_table",
                "table_type": item["table_type"],
                "result": "failed",
                "severity": "high",
                "message": item["reason"],
                "evidence_refs": [],
            }
        )
    return checks


def risks_from_missing_item_report(report: dict[str, Any]) -> list[Risk]:
    risks: list[Risk] = []
    for item in report.get("missing_fields", []):
        risks.append(
            Risk(
                risk_id=stable_id("missing_risk", len(risks) + 1),
                target_type="missing_field",
                target_id=item["missing_id"],
                risk_level="high",
                risk_type="critical_field_missing",
                message=item["reason"],
            )
        )
    for item in report.get("missing_tables", []):
        risks.append(
            Risk(
                risk_id=stable_id("missing_risk", len(risks) + 1),
                target_type="missing_table",
                target_id=item["missing_id"],
                risk_level="high",
                risk_type="critical_table_missing",
                message=item["reason"],
            )
        )
    return risks


def _missing_field(index: int, required: RequiredField) -> dict[str, Any]:
    return {
        "missing_id": stable_id("missing_field", index),
        "semantic_key": required.semantic_key,
        "field": required.field,
        "label": required.label,
        "field_type": required.field_type,
        "criticality": "critical",
        "status": "missing",
        "evidence_refs": [],
        "reason": f"未从原文证据中抽取到 MVP 关键字段：{required.label}。",
        "comparison_required": False,
    }


def _missing_table(index: int, required: dict[str, Any]) -> dict[str, Any]:
    return {
        "missing_id": stable_id("missing_table", index),
        "table_type": required["table_type"],
        "label": required["label"],
        "criticality": required["criticality"],
        "status": "missing",
        "evidence_refs": [],
        "reason": f"未从原文证据中恢复出 MVP 关键表格：{required['label']}。",
        "comparison_required": False,
    }
