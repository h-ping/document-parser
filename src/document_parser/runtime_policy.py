from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_GLM_OCR_MODEL, RuntimeConfig


DEFAULT_RUNTIME_OPTIONS = {
    "layout_mode": "legacy",
    "quality_gate": "strict",
    "extraction_mode": "agent_plus_rule",
    "ocr_mode": "glm_ocr",
    "repair_mode": "execute_plan",
    "table_parser_mode": "validate_only",
    "block_downstream_on_quality_failure": True,
    "cloud_ocr_consent": True,
    "allow_scanned_pdf": True,
    "require_table_quality_pass": True,
}


class RuntimePolicyError(RuntimeError):
    pass


def runtime_options_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "layout_mode": _string_option(manifest, "layout_mode"),
        "quality_gate": _string_option(manifest, "quality_gate"),
        "extraction_mode": _string_option(manifest, "extraction_mode"),
        "ocr_mode": _string_option(manifest, "ocr_mode"),
        "repair_mode": _string_option(manifest, "repair_mode"),
        "table_parser_mode": _string_option(manifest, "table_parser_mode"),
        "block_downstream_on_quality_failure": _bool_option(manifest, "block_downstream_on_quality_failure"),
        "cloud_ocr_consent": _bool_option(manifest, "cloud_ocr_consent"),
        "allow_scanned_pdf": _bool_option(manifest, "allow_scanned_pdf"),
        "require_table_quality_pass": _bool_option(manifest, "require_table_quality_pass"),
    }


def default_runtime_options() -> dict[str, Any]:
    return dict(DEFAULT_RUNTIME_OPTIONS)


def build_runtime_policy(
    *,
    source: str,
    options: dict[str, Any],
    config: RuntimeConfig,
    required_env_vars: list[str],
    ocr_fixture_path: Path | None,
    agent_items_path: Path | None,
    use_llm_agent: bool,
    max_repair_rounds: int,
) -> dict[str, Any]:
    effective = _effective_options(options)
    cloud_ocr_enabled = ocr_fixture_path is None and effective["ocr_mode"] == "glm_ocr"
    llm_mode = "online" if use_llm_agent else "disabled"
    provided_env_vars = _provided_env_vars(config)
    checks = _policy_checks(effective, required_env_vars, provided_env_vars, cloud_ocr_enabled, use_llm_agent)
    failed_count = sum(1 for check in checks if check["result"] == "failed")

    return {
        "policy_version": "mvp_runtime_policy_v0.1",
        "status": "pass" if failed_count == 0 else "review_required",
        "failed_count": failed_count,
        "source": source,
        "requested_options": options,
        "effective_options": effective,
        "secrets": {
            "required_env_vars": required_env_vars,
            "provided_env_vars": provided_env_vars,
            "optional_env_vars": ["GLM_OCR_MODEL"],
            "values_redacted": True,
        },
        "ocr": {
            "provider": "glm_ocr",
            "mode": "recorded_fixture" if ocr_fixture_path else "cloud",
            "cloud_ocr_enabled": cloud_ocr_enabled,
            "fixture_path": str(ocr_fixture_path) if ocr_fixture_path else None,
            "api_key_env_var": "GLM_OCR_API_KEY" if cloud_ocr_enabled else None,
            "model": config.glm_ocr_model,
            "model_is_default": config.glm_ocr_model == DEFAULT_GLM_OCR_MODEL,
        },
        "llm_agent": {
            "mode": llm_mode,
            "enabled": use_llm_agent,
            "required_env_vars": ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"] if use_llm_agent else [],
            "agent_items_path": str(agent_items_path) if agent_items_path else None,
            "runtime_managed_online_llm": use_llm_agent,
            "base_url_configured": bool(config.llm_base_url),
            "model_configured": bool(config.llm_model),
        },
        "repair": {
            "mode": effective["repair_mode"],
            "max_repair_rounds": max_repair_rounds,
            "execution_policy": "agent_repair_execute_plan_recompile_validate",
            "deterministic_plan_repair_enabled": True,
            "auto_apply_final_json": False,
        },
        "table_parser": {
            "mode": effective["table_parser_mode"],
            "execution_policy": "feed_structure_when_quality_pass" if effective["table_parser_mode"] == "feed_structure" else "validate_only",
            "feeds_standard_items": effective["table_parser_mode"] == "feed_structure",
        },
        "layout": {
            "mode": effective["layout_mode"],
            "enhanced": effective["layout_mode"] == "char_atoms_high_recall",
            "requires_llm_agent": effective["layout_mode"] == "char_atoms_high_recall",
            "failure_policy": "fail_closed_no_legacy_fallback" if effective["layout_mode"] == "char_atoms_high_recall" else "legacy",
        },
        "quality_gate": {
            "mode": effective["quality_gate"],
            "block_downstream_on_quality_failure": effective["block_downstream_on_quality_failure"],
            "critical_confidence_threshold": 0.95,
            "require_table_quality_pass": effective["require_table_quality_pass"],
        },
        "checks": checks,
    }


def assert_runtime_policy_passed(policy: dict[str, Any]) -> None:
    if policy.get("status") == "pass":
        return
    failed_checks = [
        str(check.get("check_type"))
        for check in policy.get("checks", [])
        if isinstance(check, dict) and check.get("result") == "failed"
    ]
    failed_summary = ", ".join(failed_checks) if failed_checks else "unknown_policy_check"
    raise RuntimePolicyError(f"Runtime policy failed: {failed_summary}")


def _effective_options(options: dict[str, Any]) -> dict[str, Any]:
    effective = dict(DEFAULT_RUNTIME_OPTIONS)
    for key, default in DEFAULT_RUNTIME_OPTIONS.items():
        value = options.get(key)
        effective[key] = default if value is None else value
    return effective


def _policy_checks(
    effective: dict[str, Any],
    required_env_vars: list[str],
    provided_env_vars: list[str],
    cloud_ocr_enabled: bool,
    use_llm_agent: bool,
) -> list[dict[str, Any]]:
    checks = []
    supported_modes = {
        "layout_mode": {"legacy", "char_atoms_high_recall"},
        "quality_gate": {"strict"},
        "extraction_mode": {"agent_plus_rule", "rule_only"},
        "ocr_mode": {"glm_ocr"},
        "repair_mode": {"execute_plan", "plan_only"},
        "table_parser_mode": {"validate_only", "feed_structure"},
    }
    for key, allowed in supported_modes.items():
        _add_check(
            checks,
            f"{key}_supported",
            effective.get(key) in allowed,
            f"{key} is supported by this MVP runtime.",
            {"requested": effective.get(key), "allowed": sorted(allowed)},
        )
    missing_env_vars = [name for name in required_env_vars if name not in provided_env_vars]
    _add_check(
        checks,
        "required_env_vars_available",
        not missing_env_vars,
        "Required secret environment variables are available by name.",
        {"missing_env_vars": missing_env_vars},
    )
    _add_check(
        checks,
        "cloud_ocr_key_policy",
        (not cloud_ocr_enabled) or ("GLM_OCR_API_KEY" in required_env_vars),
        "Cloud OCR requires the canonical GLM_OCR_API_KEY secret name; ZAI_API_KEY and ZHIPUAI_API_KEY are accepted as env aliases by config.",
        {"cloud_ocr_enabled": cloud_ocr_enabled, "required_env_vars": required_env_vars},
    )
    _add_check(
        checks,
        "cloud_ocr_consent",
        (not cloud_ocr_enabled) or bool(effective.get("cloud_ocr_consent")),
        "Cloud OCR may run only when cloud_ocr_consent is true.",
        {"cloud_ocr_enabled": cloud_ocr_enabled, "cloud_ocr_consent": effective.get("cloud_ocr_consent")},
    )
    _add_check(
        checks,
        "llm_secret_policy",
        (not use_llm_agent) or all(name in required_env_vars for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")),
        "Online LLM agent mode requires LLM_API_KEY, LLM_BASE_URL and LLM_MODEL.",
        {"llm_agent_enabled": use_llm_agent, "required_env_vars": required_env_vars},
    )
    _add_check(
        checks,
        "enhanced_layout_requires_llm_agent",
        effective.get("layout_mode") != "char_atoms_high_recall" or use_llm_agent,
        "char_atoms_high_recall layout mode requires the LLM agent production path.",
        {"layout_mode": effective.get("layout_mode"), "llm_agent_enabled": use_llm_agent},
    )
    return checks


def _add_check(
    checks: list[dict[str, Any]],
    check_type: str,
    passed: bool,
    message: str,
    details: dict[str, Any],
) -> None:
    checks.append(
        {
            "check_id": f"runtime_policy_{len(checks) + 1:04d}",
            "check_type": check_type,
            "result": "passed" if passed else "failed",
            "message": message,
            "details": details,
        }
    )


def _provided_env_vars(config: RuntimeConfig) -> list[str]:
    provided = []
    if config.glm_ocr_api_key:
        provided.append("GLM_OCR_API_KEY")
    if config.llm_api_key:
        provided.append("LLM_API_KEY")
    if config.llm_base_url:
        provided.append("LLM_BASE_URL")
    if config.llm_model:
        provided.append("LLM_MODEL")
    return provided


def _string_option(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    if value is None:
        return None
    return str(value)


def _bool_option(manifest: dict[str, Any], key: str) -> bool | None:
    value = manifest.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)
