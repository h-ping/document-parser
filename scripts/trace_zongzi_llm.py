from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from document_parser.agents import SchemaInductionAgent
from document_parser.config import RuntimeConfig
from document_parser.label_text_scope import build_label_text_scope_agent_context, load_label_text_scope_reference
from document_parser.llm import OpenAICompatibleLlmClient
from document_parser.llm_agents import SpanGroundedFieldAgent
from document_parser.ocr import GLMOcrClient
from document_parser.pdf import PdfPerceptionReader
from document_parser.pipeline import _merge_spans, _schema_from_agent_body
from document_parser.structures import detect_regions
from document_parser.table_parser import build_table_parser_outputs
from document_parser.utils import write_json
from document_parser.vdg_quality import build_pre_agent_vdg_artifacts


class TracingLlmClient(OpenAICompatibleLlmClient):
    def __init__(self, config: RuntimeConfig, trace_dir: Path, timeout_seconds: int) -> None:
        super().__init__(config, timeout_seconds=timeout_seconds)
        self.trace_dir = trace_dir
        self.stage = "unknown"
        self.request_index = 0

    def structured_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        write_json(
            self.trace_dir / f"{self.stage}.input.json",
            {
                "stage": self.stage,
                "system": system,
                "user": user,
                "schema": schema,
                "system_char_count": len(system),
                "user_char_count": len(user),
            },
        )
        try:
            result = super().structured_json(system, user, schema)
        except Exception as exc:
            write_json(
                self.trace_dir / f"{self.stage}.parsed_error.json",
                {
                    "stage": self.stage,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        write_json(self.trace_dir / f"{self.stage}.parsed_output.json", result)
        return result

    def _post(self, payload: dict[str, Any]):  # type: ignore[no-untyped-def]
        self.request_index += 1
        request_id = f"{self.request_index:02d}_{self.stage}"
        write_json(
            self.trace_dir / f"{request_id}.request_payload.json",
            {
                "stage": self.stage,
                "request_index": self.request_index,
                "payload": payload,
            },
        )
        try:
            response = super()._post(payload)
        except Exception as exc:
            write_json(
                self.trace_dir / f"{request_id}.response_error.json",
                {
                    "stage": self.stage,
                    "request_index": self.request_index,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        response_text = response.text
        try:
            response_json: Any = response.json()
        except ValueError:
            response_json = None
        write_json(
            self.trace_dir / f"{request_id}.response.json",
            {
                "stage": self.stage,
                "request_index": self.request_index,
                "status_code": response.status_code,
                "response_text": response_text,
                "response_text_char_count": len(response_text),
                "response_json": response_json,
            },
        )
        return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("test-documents/Q2-1.4千克粽粽有礼粽子礼盒标签信息 26.3.4.pdf"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/document-parser-zongzi-llm-trace"))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig.from_env(
        require_secrets=True,
        required_env_vars=["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"] if args.skip_ocr else ["GLM_OCR_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"],
    )

    perception = PdfPerceptionReader().read(args.pdf)
    ocr_lines = [] if args.skip_ocr else GLMOcrClient(config).recognize_pdf(args.pdf, perception.pages)
    spans = _merge_spans(perception.text_spans, ocr_lines)
    regions = detect_regions(spans)
    table_layers, _table_quality_report = build_table_parser_outputs(spans, str(args.pdf))
    candidate_graph, vdg_quality_report, vdg_agent_context = build_pre_agent_vdg_artifacts(
        perception.pages,
        spans,
        regions,
        table_layers,
    )
    label_text_scope_reference = load_label_text_scope_reference()
    vdg_agent_context = {
        **vdg_agent_context,
        "label_text_scope": build_label_text_scope_agent_context(label_text_scope_reference),
    }
    effective_vdg_context = None if vdg_quality_report.get("status") == "fail" else vdg_agent_context
    rule_schema = SchemaInductionAgent().generate(spans)
    write_json(
        args.output_dir / "trace_context_summary.json",
        {
            "pdf": str(args.pdf),
            "span_count": len(spans),
            "pdf_text_span_count": len(perception.text_spans),
            "ocr_line_count": len(ocr_lines),
            "region_count": len(regions),
            "candidate_vdg_node_count": candidate_graph.get("node_count"),
            "candidate_vdg_edge_count": candidate_graph.get("edge_count"),
            "vdg_quality_status": vdg_quality_report.get("status"),
            "nutrition_table_candidate_status": vdg_quality_report.get("nutrition_table_candidate_status"),
            "nutrition_table_row_count": vdg_quality_report.get("nutrition_table_row_count"),
            "nutrition_table_cell_count": vdg_quality_report.get("nutrition_table_cell_count"),
        },
    )

    client = TracingLlmClient(config, args.output_dir, timeout_seconds=args.timeout_seconds)
    agent = SpanGroundedFieldAgent(client)

    client.stage = "schema"
    schema_body = agent.generate_schema(spans, vdg_context=effective_vdg_context)
    generated_schema = _schema_from_agent_body(schema_body, spans, rule_schema)

    client.stage = "extraction"
    agent.generate_extraction_plan(generated_schema, spans, vdg_context=effective_vdg_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
