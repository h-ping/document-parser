from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, RuntimeConfig
from .failure_result import build_failure_result, failure_stage_for_exception
from .llm import OpenAICompatibleLlmClient
from .llm_agents import SpanGroundedFieldAgent
from .models import to_jsonable
from .ocr import GLMOcrClient, RecordedOcrClient
from .pipeline import DocumentParser, ParseError
from .runtime_policy import assert_runtime_policy_passed, build_runtime_policy, default_runtime_options
from .schema_artifacts import write_schema_artifacts
from .utils import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse packaging label standard PDF into evidence-bound JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Input standard PDF path.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--debug-dir", type=Path, help="Optional debug artifact directory.")
    parser.add_argument("--ocr-fixture", type=Path, help="Recorded OCR response fixture for offline tests.")
    parser.add_argument("--agent-items-path", type=Path, help="Optional span-grounded agent candidate JSON.")
    parser.add_argument("--use-llm-agent", action="store_true", help="Call the configured online LLM agent for span-grounded candidates.")
    args = parser.parse_args(argv)

    runtime_policy = None
    try:
        required_env_vars: list[str] = []
        if args.ocr_fixture is None:
            required_env_vars.append("GLM_OCR_API_KEY")
        if args.use_llm_agent:
            required_env_vars.extend(["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"])
        config = RuntimeConfig.from_env(require_secrets=bool(required_env_vars), required_env_vars=required_env_vars)
        runtime_policy = build_runtime_policy(
            source="cli",
            options=default_runtime_options(),
            config=config,
            required_env_vars=required_env_vars,
            ocr_fixture_path=args.ocr_fixture,
            agent_items_path=args.agent_items_path,
            use_llm_agent=args.use_llm_agent,
            max_repair_rounds=2,
        )
        assert_runtime_policy_passed(runtime_policy)
        ocr_client = RecordedOcrClient(args.ocr_fixture) if args.ocr_fixture else GLMOcrClient(config)
        llm_agent = SpanGroundedFieldAgent(OpenAICompatibleLlmClient(config)) if args.use_llm_agent else None
        result = DocumentParser(ocr_client=ocr_client, llm_agent=llm_agent).parse(
            args.input,
            debug_dir=args.debug_dir,
            agent_items_path=args.agent_items_path,
            use_llm_agent=args.use_llm_agent,
            runtime_policy=runtime_policy,
        )
        _write_verified_json(args.output, to_jsonable(result))
        write_schema_artifacts(args.output.parent)
    except (ConfigError, ParseError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        _write_failure_json(args.output, args.input, exc, runtime_policy)
        write_schema_artifacts(args.output.parent)
        print(f"parse-pdf failed: {exc}", file=sys.stderr)
        return 1

    return 0


def _write_verified_json(output_path: Path, payload: object) -> None:
    write_json(output_path, payload)
    with output_path.open(encoding="utf-8") as handle:
        json.load(handle)


def _write_failure_json(output_path: Path, input_path: Path, exc: BaseException, runtime_policy: dict | None) -> None:
    failure = build_failure_result(
        input_path=input_path,
        stage=failure_stage_for_exception(exc),
        reason=str(exc),
        error_type=exc.__class__.__name__,
        runtime_policy=runtime_policy,
    )
    _write_verified_json(output_path, failure)


if __name__ == "__main__":
    raise SystemExit(main())
