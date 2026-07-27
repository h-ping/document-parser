from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid manifest JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest root must be a JSON object.")
    return manifest


def manifest_input_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    raw_path = _first_path(
        manifest,
        [
            ("input_xlsx",),
            ("input_pdf",),
            ("input_path",),
            ("file_path",),
            ("source_file",),
            ("document", "path"),
            ("document", "file_path"),
            ("input", "path"),
            ("input", "file_path"),
        ],
    )
    if raw_path is None:
        inputs = manifest.get("inputs")
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
            raw_path = inputs[0].get("path") or inputs[0].get("file_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ManifestError("Manifest must include an input file path.")
    return _resolve_path(raw_path, manifest_path.parent)


def manifest_input_pdf(manifest: dict[str, Any], manifest_path: Path) -> Path:
    return manifest_input_path(manifest, manifest_path)


def manifest_agent_items_path(manifest: dict[str, Any], manifest_path: Path) -> Path | None:
    raw_path = _first_path(
        manifest,
        [
            ("agent_items_path",),
            ("agent", "items_path"),
            ("llm_agent", "items_path"),
        ],
    )
    if not raw_path:
        return None
    if not isinstance(raw_path, str):
        raise ManifestError("agent_items_path must be a string when provided.")
    return _resolve_path(raw_path, manifest_path.parent)


def manifest_ocr_fixture_path(manifest: dict[str, Any], manifest_path: Path) -> Path | None:
    raw_path = _first_path(
        manifest,
        [
            ("ocr_fixture",),
            ("ocr_fixture_path",),
            ("recorded_ocr_fixture",),
            ("ocr", "fixture_path"),
        ],
    )
    if not raw_path:
        return None
    if not isinstance(raw_path, str):
        raise ManifestError("ocr_fixture_path must be a string when provided.")
    return _resolve_path(raw_path, manifest_path.parent)


def manifest_use_llm_agent(manifest: dict[str, Any]) -> bool:
    if bool(manifest.get("use_llm_agent")):
        return True
    llm_mode = manifest.get("llm_mode")
    if isinstance(llm_mode, str) and llm_mode.lower() == "agent":
        return True
    agent = manifest.get("llm_agent") or manifest.get("agent")
    if isinstance(agent, dict) and bool(agent.get("enabled")):
        return True
    return False


def manifest_max_repair_rounds(manifest: dict[str, Any], default: int = 2) -> int:
    value = manifest.get("max_repair_rounds", default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError("max_repair_rounds must be an integer.") from exc
    if parsed < 0:
        raise ManifestError("max_repair_rounds must be >= 0.")
    return parsed


def redacted_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return _redact(manifest)


def _first_path(manifest: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = manifest
        for part in path:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current:
            return current
    return None


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            lowered = key.lower()
            if any(token in lowered for token in ("key", "token", "secret", "password")):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
