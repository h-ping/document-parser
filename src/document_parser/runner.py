from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import ConfigError, RuntimeConfig
from .failure_result import build_failure_result, failure_stage_for_exception
from .llm import OpenAICompatibleLlmClient
from .llm_agents import SpanGroundedFieldAgent
from .manifest import (
    ManifestError,
    load_manifest,
    manifest_agent_items_path,
    manifest_input_path,
    manifest_max_repair_rounds,
    manifest_ocr_fixture_path,
    manifest_use_llm_agent,
    redacted_manifest,
)
from .models import ParseResult
from .ocr import GLMOcrClient, RecordedOcrClient
from .pipeline import DocumentParser, ParseError, write_artifact_index
from .runtime_policy import assert_runtime_policy_passed, build_runtime_policy, runtime_options_from_manifest
from .schema_artifacts import write_schema_artifacts
from .utils import sha256_file, write_json


def run_manifest(manifest_path: Path, output_dir: Path) -> ParseResult:
    manifest = load_manifest(manifest_path)
    input_path = manifest_input_path(manifest, manifest_path)
    agent_items_path = manifest_agent_items_path(manifest, manifest_path)
    ocr_fixture_path = manifest_ocr_fixture_path(manifest, manifest_path)
    use_llm_agent = manifest_use_llm_agent(manifest)
    max_repair_rounds = manifest_max_repair_rounds(manifest)
    suffix = input_path.suffix.lower()
    if suffix == ".xlsx":
        if use_llm_agent:
            raise ManifestError("LLM agent mode is not supported for structured XLSX input.")
        if agent_items_path:
            raise ManifestError("agent_items_path is not supported for structured XLSX input.")
        from .standard_xlsx import StandardXlsxParser

        result = StandardXlsxParser().parse(input_path, debug_dir=output_dir)
        _write_input_artifacts(output_dir, manifest_path, manifest, input_path, "input_xlsx", None, ocr_fixture_path)
        write_artifact_index(output_dir)
        return result
    if suffix != ".pdf":
        raise ManifestError(f"Unsupported input file type for extract-structure: {suffix or '<none>'}")

    runtime_options = runtime_options_from_manifest(manifest)
    if use_llm_agent and runtime_options.get("extraction_mode") == "rule_only":
        raise ManifestError("rule_only extraction is not allowed when online LLM agent mode is enabled.")

    required_env_vars: list[str] = []
    if ocr_fixture_path is None and _manifest_requests_cloud_ocr(runtime_options):
        required_env_vars.append("GLM_OCR_API_KEY")
    if use_llm_agent:
        required_env_vars.extend(["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"])

    config = RuntimeConfig.from_env(require_secrets=bool(required_env_vars), required_env_vars=required_env_vars)
    runtime_policy = build_runtime_policy(
        source="manifest",
        options=runtime_options,
        config=config,
        required_env_vars=required_env_vars,
        ocr_fixture_path=ocr_fixture_path,
        agent_items_path=agent_items_path,
        use_llm_agent=use_llm_agent,
        max_repair_rounds=max_repair_rounds,
    )
    assert_runtime_policy_passed(runtime_policy)
    ocr_client = RecordedOcrClient(ocr_fixture_path) if ocr_fixture_path else GLMOcrClient(config)
    llm_agent = SpanGroundedFieldAgent(OpenAICompatibleLlmClient(config)) if use_llm_agent else None
    result = DocumentParser(
        ocr_client=ocr_client,
        llm_agent=llm_agent,
        max_repair_rounds=max_repair_rounds,
    ).parse(
        input_path,
        debug_dir=output_dir,
        agent_items_path=agent_items_path,
        use_llm_agent=use_llm_agent,
        runtime_policy=runtime_policy,
    )
    _write_input_artifacts(output_dir, manifest_path, manifest, input_path, "input_pdf", agent_items_path, ocr_fixture_path)
    write_artifact_index(output_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract structured packaging label standard artifacts from a manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="Input manifest JSON path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output artifact directory.")
    args = parser.parse_args(argv)

    try:
        run_manifest(args.manifest, args.output_dir)
    except (ConfigError, ManifestError, ParseError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        write_json(
            args.output_dir / "failure_result.json",
            build_failure_result(
                input_path=None,
                stage=failure_stage_for_exception(exc),
                reason=str(exc),
                error_type=exc.__class__.__name__,
                extra_metadata={"manifest_path": str(args.manifest)},
            ),
        )
        write_schema_artifacts(args.output_dir)
        write_artifact_index(args.output_dir)
        print(f"extract-structure failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _manifest_requests_cloud_ocr(runtime_options: dict[str, Any]) -> bool:
    return (runtime_options.get("ocr_mode") or "glm_ocr") == "glm_ocr"


def _write_input_artifacts(
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    input_path: Path,
    input_role: str,
    agent_items_path: Path | None,
    ocr_fixture_path: Path | None,
) -> None:
    write_json(output_dir / "00_inputs" / "input_manifest.redacted.json", redacted_manifest(manifest))
    files = [
        _file_inventory_item("manifest", manifest_path),
        _file_inventory_item(input_role, input_path),
    ]
    if agent_items_path:
        files.append(_file_inventory_item("agent_items", agent_items_path))
    if ocr_fixture_path:
        files.append(_file_inventory_item("ocr_fixture", ocr_fixture_path))
    write_json(
        output_dir / "00_inputs" / "file_inventory.json",
        {
            "files": files,
        },
    )


def _file_inventory_item(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
