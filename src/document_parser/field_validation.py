from __future__ import annotations

import re
from typing import Any

from .models import CompiledField


NET_CONTENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:kg|g|ml|mL|L|千克|克|毫升|升|公斤)", re.IGNORECASE)
SHELF_LIFE_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*(?:个月|月|天|日|年|小时)|见(?:包装|喷码|标示|标签))")
STANDARD_CODE_RE = re.compile(r"(?:GB|GB/T|Q/|QB/T|SB/T|T/)[A-Z0-9\u4e00-\u9fff/.\-\s]+", re.IGNORECASE)
LICENSE_RE = re.compile(r"\bSC\d{14}\b", re.IGNORECASE)


def format_validation_confidence(semantic_key: str, field_type: str, normalized_value: str) -> float | None:
    if field_type == "unknown":
        return None
    result = validate_field_format_value(semantic_key, normalized_value)
    if result["result"] == "failed":
        return 0.85
    return 1.0


def format_checks_for_field(field: CompiledField) -> list[dict[str, Any]]:
    result = validate_field_format_value(field.semantic_key, field.normalized_value)
    check_result = "passed" if result["result"] == "skipped" else result["result"]
    return [
        {
            "target_id": field.field_id,
            "check_type": "format_check",
            "result": check_result,
            "format_rule_status": "skipped_no_rule" if result["result"] == "skipped" else "evaluated",
            "semantic_key": field.semantic_key,
            "expected": result["expected"],
            "actual": field.normalized_value,
            "severity": result["severity"],
            "message": result["message"],
            "evidence_refs": field.evidence_refs,
        }
    ]


def validate_field_format_value(semantic_key: str, value: str) -> dict[str, Any]:
    normalized = value.strip()
    if semantic_key in {"barcode.commodity", "barcode.outer_case"}:
        return _validate_barcode(normalized)
    if semantic_key == "manufacturer.license_number":
        return _regex_result(
            LICENSE_RE.search(normalized) is not None,
            "SC followed by 14 digits",
            "high",
            "食品生产许可证编号格式不符合 SC + 14 位数字。",
        )
    if semantic_key == "product.net_content":
        return _regex_result(
            NET_CONTENT_RE.search(normalized) is not None,
            "number plus metric unit",
            "medium",
            "净含量未识别到数值和计量单位。",
        )
    if semantic_key == "product.shelf_life":
        return _regex_result(
            SHELF_LIFE_RE.search(normalized) is not None,
            "duration or see-package date marking",
            "medium",
            "保质期格式未识别到持续时间或见包装标示。",
        )
    if semantic_key == "product.standard_code":
        return _regex_result(
            STANDARD_CODE_RE.search(normalized) is not None,
            "GB/GB-T/Q/QB/SB/T standard code",
            "medium",
            "产品标准代号格式未识别到常见标准编号。",
        )
    return {
        "result": "skipped",
        "expected": None,
        "severity": "info",
        "message": "No deterministic format rule for this field.",
    }


def _validate_barcode(value: str) -> dict[str, Any]:
    digits = re.sub(r"\D", "", value)
    if digits != value and not re.fullmatch(r"[\d\s-]+", value):
        return _failed("8/12/13/14 digit barcode", "high", "条码包含非数字字符。")
    if len(digits) not in {8, 12, 13, 14}:
        return _failed("8/12/13/14 digit barcode", "high", "条码长度不是 8、12、13 或 14 位。")
    if not _gtin_check_digit_valid(digits):
        return _failed("valid GTIN check digit", "high", "条码校验位不通过。")
    return _passed("valid 8/12/13/14 digit barcode")


def _gtin_check_digit_valid(digits: str) -> bool:
    body = digits[:-1]
    expected = int(digits[-1])
    total = 0
    reversed_body = list(reversed(body))
    for index, char in enumerate(reversed_body, start=1):
        weight = 3 if index % 2 == 1 else 1
        total += int(char) * weight
    calculated = (10 - (total % 10)) % 10
    return calculated == expected


def _regex_result(passed: bool, expected: str, failed_severity: str, failed_message: str) -> dict[str, Any]:
    return _passed(expected) if passed else _failed(expected, failed_severity, failed_message)


def _passed(expected: str) -> dict[str, Any]:
    return {
        "result": "passed",
        "expected": expected,
        "severity": "info",
        "message": "Format check passed.",
    }


def _failed(expected: str, severity: str, message: str) -> dict[str, Any]:
    return {
        "result": "failed",
        "expected": expected,
        "severity": severity,
        "message": message,
    }
