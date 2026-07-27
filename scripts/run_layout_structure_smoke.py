#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import shutil
import struct
import unicodedata
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import requests

from document_parser.config import RuntimeConfig
from document_parser.layout_candidates import build_layout_candidates as build_shared_layout_candidates
from document_parser.models import BBoxNormalized, BBoxPdf, PageInfo, TextSpan
from document_parser.page_render import render_page_images
from document_parser.pdf import PdfPerceptionReader
from document_parser.pdf_atoms import read_pdf_character_atoms
from document_parser.utils import stable_id, write_json


DECISION_ACTIONS = {
    "accept_table_candidate",
    "reject_table_candidate",
    "classify_node",
    "split_node",
    "merge_nodes",
    "override_reading_order",
    "mark_table_structure_unresolved",
    "mark_unknown_block",
    "mark_excluded_content",
}

NODE_TYPES = {
    "heading",
    "paragraph",
    "table",
    "table_row",
    "table_cell",
    "excluded_content",
    "unknown_block",
}

SMOKE_CASES = [
    {
        "case_id": "youleme_page2",
        "display_name": "优乐美 page 2",
        "pdf": "test-documents/04-374-提箱-65克优乐美红豆奶茶15杯装彩色箱（有配料文字）（8）.pdf",
        "page": 2,
        "focus": "producer_info",
    },
    {
        "case_id": "zongzi_page3",
        "display_name": "粽子 page 3",
        "pdf": "test-documents/Q2-1.4千克粽粽有礼粽子礼盒标签信息 26.3.4.pdf",
        "page": 3,
        "focus": "nutrition_tables",
        "expected_table_count": 4,
        "expected_row_count": 7,
    },
]

LAYOUT_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["pass", "review_required", "fail"]},
        "summary": {"type": "string"},
        "decisions": {
            "type": "array",
            "maxItems": 120,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision_id": {"type": "string"},
                    "action": {"type": "string", "enum": sorted(DECISION_ACTIONS)},
                    "node_type": {"type": "string", "enum": sorted(NODE_TYPES) + [""]},
                    "table_candidate_id": {"type": "string"},
                    "source_span_ids": {"type": "array", "maxItems": 40, "items": {"type": "string"}},
                    "target_node_ids": {"type": "array", "maxItems": 40, "items": {"type": "string"}},
                    "reading_order_after": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "decision_id",
                    "action",
                    "node_type",
                    "table_candidate_id",
                    "source_span_ids",
                    "target_node_ids",
                    "reading_order_after",
                    "reason",
                    "confidence",
                ],
            },
        },
    },
    "required": ["status", "summary", "decisions"],
}

PREFLIGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["pass"]},
        "image_visible": {"type": "boolean"},
    },
    "required": ["status", "image_visible"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a standalone multimodal Layout Structure Agent smoke test.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/document-parser-layout-smoke"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip vision preflight and directly run cases.")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--pdf", type=Path, help="Run smoke test for a custom PDF instead of the built-in cases.")
    parser.add_argument("--pages", type=int, nargs="*", help="Specific pages for --pdf. Defaults to all pages.")
    parser.add_argument("--focus", choices=["auto", "nutrition_tables", "producer_info"], default="auto")
    parser.add_argument("--source-mode", choices=["pdf_spans", "pdf_char_atoms"], default="pdf_spans")
    parser.add_argument("--expected-table-count", type=int)
    parser.add_argument("--expected-row-count", type=int)
    parser.add_argument("--required-independent-text", nargs="*", default=[])
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig.from_env(require_secrets=True, required_env_vars=["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"])
    client = VisionChatJsonClient(config, timeout_seconds=args.timeout_seconds)
    cases = _cases_from_args(args)

    summary: dict[str, Any] = {
        "status": "pass",
        "output_dir": str(args.output_dir),
        "cases": [],
    }

    preflight = {"status": "skipped", "reason": "skip_preflight_requested"} if args.skip_preflight else _run_vision_preflight(client, args.output_dir, cases)
    write_json(args.output_dir / "vision_preflight.json", preflight)
    if preflight.get("status") == "blocked":
        summary["status"] = "blocked"
        summary["preflight"] = preflight
        for case in cases:
            case_dir = args.output_dir / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_blocked_case(case, case_dir, preflight)
            summary["cases"].append({"case_id": case["case_id"], "status": "blocked", "report": str(case_dir / "layout_smoke_report.html")})
        write_json(args.output_dir / "layout_smoke_summary.json", summary)
        return 2

    for case in cases:
        case_summary = _run_case(case, client, args.output_dir)
        summary["cases"].append(case_summary)
        summary["status"] = _aggregate_status(str(summary["status"]), str(case_summary.get("layout_status", "fail")))

    write_json(args.output_dir / "layout_smoke_summary.json", summary)
    return 0 if summary["status"] == "pass" else 1


def _aggregate_status(current: str, case_status: str) -> str:
    priority = {"pass": 0, "review_required": 1, "fail": 2, "blocked": 3}
    return case_status if priority.get(case_status, 2) > priority.get(current, 2) else current


def _cases_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.pdf:
        return [{**case, "source_mode": args.source_mode} for case in SMOKE_CASES]
    if not args.pdf.exists():
        raise FileNotFoundError(f"Custom PDF does not exist: {args.pdf}")
    perception = PdfPerceptionReader().read(args.pdf)
    pages = args.pages if args.pages else [page.page for page in perception.pages]
    spans_by_page: dict[int, list[TextSpan]] = {}
    for span in perception.text_spans:
        spans_by_page.setdefault(span.page, []).append(span)
    cases = []
    for page in pages:
        focus = args.focus if args.focus != "auto" else _infer_focus(spans_by_page.get(page, []))
        cases.append(
            {
                "case_id": f"{_safe_case_id(args.pdf.stem)}_page{page}",
                "display_name": f"{args.pdf.stem} page {page}",
                "pdf": str(args.pdf),
                "page": page,
                "focus": focus,
                "source_mode": args.source_mode,
                "expected_table_count": args.expected_table_count,
                "expected_row_count": args.expected_row_count,
                "required_independent_text": args.required_independent_text,
            }
        )
    return cases


def _infer_focus(spans: list[TextSpan]) -> str:
    if any("营养成分表" in span.text or "NRV" in span.text for span in spans):
        return "nutrition_tables"
    if any(any(anchor in span.text for anchor in ("委托", "受托", "地址", "许可证编号", "生产者", "生产商")) for span in spans):
        return "producer_info"
    return "nutrition_tables"


def _safe_case_id(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value).strip("_")
    return safe or "custom_pdf"


class VisionChatJsonClient:
    def __init__(self, config: RuntimeConfig, timeout_seconds: int = 180) -> None:
        self._api_key = config.llm_api_key
        self._base_url = config.llm_base_url.rstrip("/")
        self._model = config.llm_model
        self._timeout_seconds = timeout_seconds

    def structured_json_with_image(self, system: str, user: str, schema: dict[str, Any], image_path: Path) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": _llm_max_tokens(),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "layout_structure_smoke",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        response = self._post(payload)
        if response.status_code == 400:
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = {"type": "json_object"}
            response = self._post(fallback_payload)
        if response.status_code != 200:
            raise RuntimeError(f"vision_llm_http_{response.status_code}:{_safe_error_text(response)}")
        try:
            return _structured_json_from_chat_response(response.json())
        except Exception as schema_parse_exc:
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = {"type": "json_object"}
            fallback_response = self._post(fallback_payload)
            if fallback_response.status_code != 200:
                raise RuntimeError(f"vision_llm_http_{fallback_response.status_code}:{_safe_error_text(fallback_response)}") from schema_parse_exc
            try:
                return _structured_json_from_chat_response(fallback_response.json())
            except Exception as fallback_parse_exc:
                raise RuntimeError(
                    f"vision_llm_json_parse_failed:{fallback_parse_exc}; first_parse_error:{schema_parse_exc}; "
                    f"raw:{_safe_error_text(fallback_response)}"
                ) from fallback_parse_exc

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            _chat_completions_url(self._base_url),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self._timeout_seconds,
        )


def _run_vision_preflight(client: VisionChatJsonClient, output_dir: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    image_dir = output_dir / "vision_preflight_page_images"
    preflight_case = cases[0] if cases else SMOKE_CASES[0]
    page_images = render_page_images(Path(preflight_case["pdf"]), image_dir)
    try:
        image_path = _page_image_path(page_images, int(preflight_case["page"]))
    except Exception as exc:
        return {"status": "blocked", "reason": "vision_preflight_render_failed", "error": str(exc), "page_images": page_images}
    try:
        result = client.structured_json_with_image(
            "Return strict JSON only.",
            "This is a vision preflight. A PDF page screenshot is attached. "
            "Return {\"status\":\"pass\",\"image_visible\":true} if you can see the attached page image. "
            "Do not use any external knowledge.",
            PREFLIGHT_SCHEMA,
            image_path,
        )
    except Exception as exc:
        return {"status": "blocked", "reason": "vision_preflight_failed", "error": str(exc)}
    if result.get("status") == "pass" and result.get("image_visible") is True:
        return {"status": "pass", "result": result}
    return {"status": "blocked", "reason": "vision_preflight_unexpected_response", "result": result}


def _run_case(case: dict[str, Any], client: VisionChatJsonClient, output_dir: Path) -> dict[str, Any]:
    case_dir = output_dir / case["case_id"]
    image_dir = case_dir / "page_images"
    case_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(case["pdf"])
    page = int(case["page"])

    perception = PdfPerceptionReader().read(pdf_path)
    page_images = render_page_images(pdf_path, image_dir)
    page_image = _page_image_path(page_images, page)
    source_mode = str(case.get("source_mode", "pdf_spans"))
    if source_mode == "pdf_char_atoms":
        page_spans = read_pdf_character_atoms(pdf_path, pages={page})
    else:
        page_spans = [span for span in perception.text_spans if span.page == page]
    candidates = _build_layout_candidates(case, page_spans)
    agent_input = _build_agent_input(case, page_spans, candidates, page_image)

    write_json(case_dir / "layout_candidates.json", candidates)
    write_json(case_dir / "layout_agent_input.json", agent_input)

    try:
        decisions = client.structured_json_with_image(
            _layout_agent_system_prompt(),
            _layout_agent_user_prompt(agent_input),
            LAYOUT_AGENT_SCHEMA,
            page_image,
        )
    except Exception as exc:
        decisions = {
            "status": "blocked",
            "summary": "Layout Agent request failed.",
            "decisions": [],
            "agent_error": str(exc),
        }
    decisions = _normalize_agent_decisions(decisions, candidates)
    write_json(case_dir / "layout_agent_decisions.json", decisions)

    if decisions.get("status") == "blocked":
        structure_tree = {"schema_version": "layout_smoke_v0.1", "case_id": case["case_id"], "status": "blocked", "page": page, "nodes": [], "metrics": _empty_metrics()}
        quality_report = {
            "status": "blocked",
            "case_id": case["case_id"],
            "agent_status": "blocked",
            "agent_summary": decisions.get("summary"),
            "metrics": _empty_metrics(),
            "issues": [
                {
                    "issue_type": "layout_agent_blocked",
                    "severity": "high",
                    "message": decisions.get("agent_error", "Layout Agent request failed."),
                }
            ],
            "issue_count": 1,
            "gate_checks": [_check("layout_agent_completed", False, decisions.get("agent_error"))],
        }
        write_json(case_dir / "layout_structure_tree.json", structure_tree)
        write_json(case_dir / "layout_quality_report.json", quality_report)
        report_path = case_dir / "layout_smoke_report.html"
        report_path.write_text(_render_case_html(case, page_spans, candidates, decisions, structure_tree, quality_report, page_image), encoding="utf-8")
        return {
            "case_id": case["case_id"],
            "layout_status": "blocked",
            "report": str(report_path),
            "layout_quality_report": str(case_dir / "layout_quality_report.json"),
        }

    structure_tree, quality_report = _build_structure_tree_and_quality(case, page_spans, candidates, decisions)
    write_json(case_dir / "layout_structure_tree.json", structure_tree)
    write_json(case_dir / "layout_quality_report.json", quality_report)

    html_report = _render_case_html(case, page_spans, candidates, decisions, structure_tree, quality_report, page_image)
    report_path = case_dir / "layout_smoke_report.html"
    report_path.write_text(html_report, encoding="utf-8")

    return {
        "case_id": case["case_id"],
        "layout_status": quality_report["status"],
        "source_mode": source_mode,
        "report": str(report_path),
        "layout_quality_report": str(case_dir / "layout_quality_report.json"),
    }


def _write_blocked_case(case: dict[str, Any], case_dir: Path, preflight: dict[str, Any]) -> None:
    for name, data in (
        ("layout_candidates.json", {"status": "blocked", "case": case}),
        ("layout_agent_input.json", {"status": "blocked", "case": case}),
        ("layout_agent_decisions.json", {"status": "blocked", "decisions": [], "preflight": preflight}),
        ("layout_structure_tree.json", {"status": "blocked", "nodes": []}),
        ("layout_quality_report.json", {"status": "blocked", "reason": "vision_preflight_blocked", "preflight": preflight}),
    ):
        write_json(case_dir / name, data)
    (case_dir / "layout_smoke_report.html").write_text(
        _html_page(
            "Layout Smoke Blocked",
            f"<main><h1>{_e(case['display_name'])}</h1><p>Vision preflight blocked: {_e(preflight.get('error') or preflight.get('reason'))}</p></main>",
            "",
        ),
        encoding="utf-8",
    )


def _build_layout_candidates(case: dict[str, Any], spans: list[TextSpan]) -> dict[str, Any]:
    candidates = build_shared_layout_candidates(spans, _page_infos(spans))
    candidates.update({"case_id": case["case_id"], "page": int(case["page"])})
    table_type = "nutrition_facts" if case["focus"] == "nutrition_tables" else "producer_info_repeated_rows"
    candidates["table_candidates"] = [
        candidate for candidate in candidates["table_candidates"] if candidate.get("table_type") == table_type
    ]
    if not candidates["table_candidates"]:
        candidates["quality_issues"].append(
            {
                "issue_type": "table_candidate_missing",
                "severity": "high",
                "message": "No table candidate was generated by rule layout candidates.",
            }
        )
    return candidates


def _nutrition_table_candidates(spans: list[TextSpan]) -> list[dict[str, Any]]:
    artifact = build_shared_layout_candidates(spans, _page_infos(spans))
    return [item for item in artifact["table_candidates"] if item.get("table_type") == "nutrition_facts"]


def _producer_info_candidates(spans: list[TextSpan]) -> list[dict[str, Any]]:
    artifact = build_shared_layout_candidates(spans, _page_infos(spans))
    return [item for item in artifact["table_candidates"] if item.get("table_type") == "producer_info_repeated_rows"]


def _page_infos(spans: list[TextSpan]) -> list[PageInfo]:
    dimensions: dict[int, tuple[float, float]] = {}
    for span in spans:
        if span.bbox_pdf:
            dimensions[span.page] = (span.bbox_pdf.page_width, span.bbox_pdf.page_height)
    return [PageInfo(page=page, width=width, height=height) for page, (width, height) in sorted(dimensions.items())]


def _rows_from_spans(spans: list[TextSpan]) -> list[dict[str, Any]]:
    rows = []
    for row_index, group in enumerate(_group_by_visual_row(spans), start=1):
        cells = []
        for col_index, span in enumerate(group, start=1):
            cells.append(
                {
                    "cell_id": f"cand_cell_{row_index:03d}_{col_index:03d}",
                    "row_index": row_index,
                    "col_index": col_index,
                    "text": span.text,
                    "source_span_ids": [span.span_id],
                    "bbox_normalized": _to_jsonable(span.bbox_normalized),
                }
            )
        rows.append(
            {
                "row_index": row_index,
                "row_type": _row_type(group),
                "text": " | ".join(span.text for span in group),
                "source_span_ids": [span.span_id for span in group],
                "cells": cells,
            }
        )
    return rows


def _group_by_visual_row(spans: list[TextSpan]) -> list[list[TextSpan]]:
    groups: list[list[TextSpan]] = []
    for span in sorted([span for span in spans if span.bbox_pdf], key=lambda item: (item.bbox_pdf.y, item.bbox_pdf.x)):
        assert span.bbox_pdf is not None
        matched: list[TextSpan] | None = None
        for group in groups:
            first = group[0]
            if not first.bbox_pdf:
                continue
            tolerance = max(first.bbox_pdf.height, span.bbox_pdf.height, 2.0) * 0.75
            if abs(_bbox_center(first.bbox_pdf)[1] - _bbox_center(span.bbox_pdf)[1]) <= tolerance:
                matched = group
                break
        if matched is None:
            groups.append([span])
        else:
            matched.append(span)
    return [sorted(group, key=lambda item: item.bbox_pdf.x if item.bbox_pdf else 0) for group in groups]


def _row_type(group: list[TextSpan]) -> str:
    text = " ".join(span.text for span in group)
    if any(anchor in text for anchor in ("营养成分表", "项目", "每100", "NRV", "营养素参考值")):
        return "header"
    if any(anchor in text for anchor in ("能量", "蛋白质", "蛋⽩质", "脂肪", "碳水化合物", "碳⽔化合物", "钠")):
        return "data"
    return "data" if "：" in text or ":" in text else "unknown"


def _reading_order_edges(spans: list[TextSpan]) -> list[dict[str, Any]]:
    ordered = sorted(spans, key=lambda span: (span.bbox_pdf.y if span.bbox_pdf else 0, span.bbox_pdf.x if span.bbox_pdf else 0))
    edges = []
    for index, (left, right) in enumerate(zip(ordered, ordered[1:]), start=1):
        edges.append({"edge_id": stable_id("reading_order", index), "source_span_id": left.span_id, "target_span_id": right.span_id})
    return edges


def _side_marker_candidates(spans: list[TextSpan]) -> list[dict[str, Any]]:
    markers = []
    for span in spans:
        if not span.bbox_pdf:
            continue
        text = span.text.strip()
        if len(text) <= 2 and span.bbox_pdf.x < 40:
            markers.append(
                {
                    "node_id": f"side_marker_{span.span_id}",
                    "source_span_ids": [span.span_id],
                    "text": span.text,
                    "bbox_normalized": _to_jsonable(span.bbox_normalized),
                    "candidate_type": "side_marker_or_vertical_label",
                }
            )
    return markers


def _build_agent_input(case: dict[str, Any], spans: list[TextSpan], candidates: dict[str, Any], image_path: Path) -> dict[str, Any]:
    focus_ids = _focus_source_span_ids(candidates)
    source_nodes = [node for node in candidates["source_nodes"] if node.get("span_id") in focus_ids]
    reading_order_candidates = [
        edge
        for edge in candidates["reading_order_candidates"]
        if edge.get("source_span_id") in focus_ids and edge.get("target_span_id") in focus_ids
    ]
    return {
        "case_id": case["case_id"],
        "display_name": case["display_name"],
        "page": case["page"],
        "focus": case["focus"],
        "source_mode": case.get("source_mode", "pdf_spans"),
        "image_path": str(image_path),
        "instructions": {
            "goal": "Validate whether a multimodal Layout Agent can recover layout structure better than rule-only candidates.",
            "agent_must_not_generate_bbox": True,
            "allowed_decision_actions": sorted(DECISION_ACTIONS),
            "max_decisions": 30,
            "decision_style": "compact_id_only_no_prose",
        },
        "source_nodes": source_nodes,
        "table_candidates": candidates["table_candidates"],
        "reading_order_candidates": reading_order_candidates[:120],
        "side_marker_candidates": candidates["side_marker_candidates"],
        "quality_issues": candidates["quality_issues"],
        "acceptance_focus": _case_acceptance_focus(case["focus"]),
    }


def _focus_source_span_ids(candidates: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for table in candidates.get("table_candidates", []):
        ids.update(str(span_id) for span_id in table.get("source_span_ids", []))
    for marker in candidates.get("side_marker_candidates", []):
        ids.update(str(span_id) for span_id in marker.get("source_span_ids", []))
    return ids


def _case_acceptance_focus(focus: str) -> list[str]:
    if focus == "producer_info":
        return [
            "Keep side marker text such as 标 independent from producer information rows.",
            "Order 委托方 before its 地址 when both are on the same visual row.",
            "Classify producer/entrusted manufacturer repeated rows as table or repeated row group.",
        ]
    return [
        "Return one accept_table_candidate decision for each nutrition table candidate when the table is visually valid.",
        "Use table_candidate_id exactly as provided, such as nutrition_candidate_0001.",
        "Identify each nutrition facts table.",
        "Recover title/header/data rows and cells when possible.",
        "Mark unresolved table structure instead of flattening nutrition rows into paragraphs.",
    ]


def _layout_agent_system_prompt() -> str:
    return (
        "You are a multimodal PDF layout structure agent for Chinese packaging label standards. "
        "Return strict JSON only. Do not include hidden reasoning, markdown, XML tags, or prose. "
        "Use the page image plus source nodes and rule candidates. "
        "Do not invent text, bbox, span ids, or table ids. All decisions must reference provided source_span_ids "
        "or table_candidate_id values. Prefer explicit review decisions when uncertain. "
        "Never copy source text, addresses, company names, phone numbers, or any page content into summary or reason. "
        "Use only ids and generic reason codes such as visual_table, side_marker, reading_order, unresolved_table."
    )


def _layout_agent_user_prompt(agent_input: dict[str, Any]) -> str:
    compact = dict(agent_input)
    compact["image_path"] = Path(str(compact["image_path"])).name
    return (
        "Analyze this single PDF page image and the evidence-bound layout candidates.\n"
        "Return only the JSON object matching the schema. Do not think aloud. Do not use <think> tags. "
        "Do not return final field values. Keep decisions compact and id-only.\n\n"
        "Output safety rule: do not quote or repeat source text. Use source_span_ids/table_candidate_id only. "
        "The reason field must be a short generic code, not page text.\n\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )


def _normalize_agent_decisions(decisions: dict[str, Any], candidates: dict[str, Any]) -> dict[str, Any]:
    if decisions.get("status") == "blocked":
        return decisions
    candidate_list = [str(candidate.get("table_candidate_id")) for candidate in candidates.get("table_candidates", [])]
    candidate_ids = set(candidate_list)
    source_ids = {node.get("span_id") for node in candidates.get("source_nodes", [])}
    normalized = {
        "status": decisions.get("status") if decisions.get("status") in {"pass", "review_required", "fail"} else None,
        "summary": decisions.get("summary"),
        "decisions": [],
        "raw_agent_output": decisions,
    }
    raw_decisions = decisions.get("decisions") if isinstance(decisions.get("decisions"), list) else []
    target_sources: dict[str, list[str]] = {}
    next_table_candidate_index = 0
    side_marker_source_ids = [
        source_id
        for candidate in candidates.get("side_marker_candidates", [])
        for source_id in candidate.get("source_span_ids", [])
        if source_id in source_ids
    ]
    for index, decision in enumerate(raw_decisions, start=1):
        if not isinstance(decision, dict):
            continue
        raw_targets = decision.get("target_node_ids")
        if not isinstance(raw_targets, list):
            raw_targets = decision.get("target_ids")
        if not isinstance(raw_targets, list):
            raw_target = decision.get("target_id")
            raw_targets = [raw_target] if raw_target else []
        target_node_ids = [str(value) for value in raw_targets if value]
        source_span_ids = _decision_source_span_ids(decision, target_node_ids, source_ids, target_sources)
        action = str(decision.get("action") or "")
        reason = str(decision.get("reason") or decision.get("reason_code") or "unspecified")
        if not source_span_ids and action == "mark_excluded_content" and "side_marker" in reason:
            source_span_ids = side_marker_source_ids
        for target in target_node_ids:
            if source_span_ids:
                target_sources[target] = list(source_span_ids)

        table_candidate_id = _decision_table_candidate_id(decision, target_node_ids, candidate_ids)
        if not table_candidate_id:
            if action in {"accept_table_candidate", "reject_table_candidate", "mark_table_structure_unresolved"} and next_table_candidate_index < len(candidate_list):
                table_candidate_id = candidate_list[next_table_candidate_index]
                next_table_candidate_index += 1
        node_type = decision.get("node_type") or decision.get("node_class") or ""
        if node_type not in NODE_TYPES:
            if "side_marker" in str(node_type):
                node_type = "excluded_content"
            elif "nutrition_table" in str(node_type):
                node_type = "table"
            else:
                node_type = ""
        normalized["decisions"].append(
            {
                "decision_id": str(decision.get("decision_id") or stable_id("layout_decision", index)),
                "action": str(decision.get("action") or ""),
                "node_type": node_type,
                "table_candidate_id": table_candidate_id,
                "source_span_ids": source_span_ids,
                "target_node_ids": target_node_ids,
                "reading_order_after": str(decision.get("reading_order_after") or ""),
                "reason": reason,
                "confidence": _confidence(decision),
            }
        )
    return normalized


def _decision_table_candidate_id(decision: dict[str, Any], target_node_ids: list[str], candidate_ids: set[str]) -> str:
    keys = ("table_candidate_id", "candidate_id", "target_candidate_id", "target_table_candidate_id", "table_id")
    for key in keys:
        value = decision.get(key)
        if isinstance(value, str) and value in candidate_ids:
            return value
    target = decision.get("target")
    if isinstance(target, dict):
        for key in keys:
            value = target.get(key)
            if isinstance(value, str) and value in candidate_ids:
                return value
    return next((target for target in target_node_ids if target in candidate_ids), "")


def _decision_source_span_ids(
    decision: dict[str, Any],
    target_node_ids: list[str],
    source_ids: set[str],
    target_sources: dict[str, list[str]],
) -> list[str]:
    raw = decision.get("source_span_ids")
    if isinstance(raw, list):
        values = [str(value) for value in raw if value in source_ids]
        if values:
            return values
    target = decision.get("target")
    if isinstance(target, dict):
        for key in ("source_span_ids", "excluded_span_ids"):
            raw = target.get(key)
            if isinstance(raw, list):
                values = [str(value) for value in raw if value in source_ids]
                if values:
                    return values
    values: list[str] = []
    for target in target_node_ids:
        if target in source_ids:
            values.append(target)
        if target.startswith("side_marker_") and target[len("side_marker_") :] in source_ids:
            values.append(target[len("side_marker_") :])
        values.extend(target_sources.get(target, []))
    return sorted(set(values), key=values.index)


def _empty_metrics() -> dict[str, Any]:
    return {
        "source_span_coverage_rate": 0.0,
        "accepted_table_count": 0,
        "table_cell_count": 0,
        "reading_order_override_count": 0,
        "unresolved_table_count": 0,
        "unknown_block_count": 0,
        "excluded_content_count": 0,
    }


def _build_structure_tree_and_quality(
    case: dict[str, Any],
    spans: list[TextSpan],
    candidates: dict[str, Any],
    decisions: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    span_ids = {span.span_id for span in spans}
    candidate_ids = {candidate["table_candidate_id"] for candidate in candidates.get("table_candidates", [])}
    candidate_by_id = {candidate["table_candidate_id"]: candidate for candidate in candidates.get("table_candidates", [])}
    issues = []
    accepted_candidates: set[str] = set()
    rejected_candidates: set[str] = set()
    unresolved_candidates: set[str] = set()
    covered_spans: set[str] = set()
    nodes: list[dict[str, Any]] = []
    reading_order_override_count = 0
    excluded_content_count = 0
    unknown_block_count = 0

    for index, decision in enumerate(decisions.get("decisions", []) if isinstance(decisions.get("decisions"), list) else [], start=1):
        action = decision.get("action")
        if action not in DECISION_ACTIONS:
            issues.append(_issue("invalid_agent_action", "high", decision_id=decision.get("decision_id"), action=action))
            continue
        source_refs = [ref for ref in decision.get("source_span_ids", []) if ref]
        missing_refs = sorted(ref for ref in source_refs if ref not in span_ids)
        if missing_refs:
            issues.append(_issue("unresolved_source_ref", "high", decision_id=decision.get("decision_id"), missing_refs=missing_refs))
            continue
        table_candidate_id = decision.get("table_candidate_id")
        if table_candidate_id and table_candidate_id not in candidate_ids:
            issues.append(_issue("unresolved_table_candidate_ref", "high", decision_id=decision.get("decision_id"), table_candidate_id=table_candidate_id))
            continue
        if action == "accept_table_candidate" and table_candidate_id:
            accepted_candidates.add(table_candidate_id)
            candidate = candidate_by_id[table_candidate_id]
            nodes.append(_table_node_from_candidate(candidate, index, decision))
            covered_spans.update(candidate.get("source_span_ids", []))
        elif action == "reject_table_candidate" and table_candidate_id:
            rejected_candidates.add(table_candidate_id)
        elif action == "mark_table_structure_unresolved" and table_candidate_id:
            unresolved_candidates.add(table_candidate_id)
            candidate = candidate_by_id[table_candidate_id]
            nodes.append(_unresolved_table_node(candidate, index, decision))
            covered_spans.update(candidate.get("source_span_ids", []))
        elif action == "override_reading_order":
            reading_order_override_count += 1
        elif action in {"classify_node", "merge_nodes", "split_node", "mark_unknown_block", "mark_excluded_content"}:
            if not source_refs:
                continue
            node_type = decision.get("node_type") or ("unknown_block" if action == "mark_unknown_block" else "excluded_content" if action == "mark_excluded_content" else "paragraph")
            if node_type not in NODE_TYPES:
                issues.append(_issue("invalid_node_type", "medium", decision_id=decision.get("decision_id"), node_type=node_type))
                node_type = "unknown_block"
            node_spans = [span for span in spans if span.span_id in source_refs]
            if action == "mark_excluded_content":
                excluded_content_count += 1
            if action == "mark_unknown_block" or node_type == "unknown_block":
                unknown_block_count += 1
            covered_spans.update(source_refs)
            nodes.append(
                {
                    "node_id": stable_id("layout_node", len(nodes) + 1),
                    "node_type": node_type,
                    "page": case["page"],
                    "text": " ".join(span.text for span in node_spans),
                    "source_span_ids": source_refs,
                    "bbox_normalized": _union_normalized_bbox(node_spans),
                    "children": [],
                    "reading_order_index": len(nodes) + 1,
                    "confidence": _confidence(decision),
                    "issues": [],
                    "agent_decision_id": decision.get("decision_id"),
                    "reason": decision.get("reason"),
                }
            )

    for candidate_id, candidate in candidate_by_id.items():
        if candidate_id not in accepted_candidates and candidate_id not in rejected_candidates and candidate_id not in unresolved_candidates:
            unresolved_candidates.add(candidate_id)
            nodes.append(_unresolved_table_node(candidate, len(nodes) + 1, {"reason": "Agent did not explicitly accept or reject this table candidate.", "confidence": 0.0}))
            covered_spans.update(candidate.get("source_span_ids", []))

    for span in spans:
        if span.span_id in covered_spans:
            continue
        nodes.append(
            {
                "node_id": stable_id("layout_unknown", len(nodes) + 1),
                "node_type": "unknown_block",
                "page": span.page,
                "text": span.text,
                "source_span_ids": [span.span_id],
                "bbox_normalized": _to_jsonable(span.bbox_normalized),
                "children": [],
                "reading_order_index": len(nodes) + 1,
                "confidence": 0.20,
                "issues": [{"issue_type": "not_classified_by_layout_agent", "severity": "info"}],
            }
        )
        unknown_block_count += 1
        covered_spans.add(span.span_id)

    table_nodes = [node for node in nodes if node["node_type"] == "table"]
    table_cell_count = sum(1 for table in table_nodes for row in table.get("children", []) for cell in row.get("children", []))
    source_span_coverage_rate = round(len(covered_spans & span_ids) / len(span_ids), 4) if span_ids else 1.0

    acceptance_issues = _case_acceptance_issues(case, spans, nodes, accepted_candidates, unresolved_candidates, reading_order_override_count)
    issues.extend(acceptance_issues)
    hard_fail = any(issue["severity"] == "high" for issue in issues if issue["issue_type"].startswith("unresolved_") or issue["issue_type"] == "invalid_agent_action")
    structure_nodes_have_bbox = all(node.get("bbox_normalized") for node in nodes)
    if not structure_nodes_have_bbox:
        issues.append(_issue("structure_node_bbox_missing", "medium", message="Some layout structure nodes are missing bbox."))
    status = "fail" if hard_fail else "review_required" if issues or unresolved_candidates else "pass"

    tree = {
        "schema_version": "layout_smoke_v0.1",
        "case_id": case["case_id"],
        "status": status,
        "page": case["page"],
        "nodes": nodes,
        "metrics": {
            "source_span_coverage_rate": source_span_coverage_rate,
            "accepted_table_count": len(accepted_candidates),
            "table_cell_count": table_cell_count,
            "reading_order_override_count": reading_order_override_count,
            "unresolved_table_count": len(unresolved_candidates),
            "unknown_block_count": unknown_block_count,
            "excluded_content_count": excluded_content_count,
        },
    }
    quality = {
        "status": status,
        "case_id": case["case_id"],
        "agent_status": decisions.get("status"),
        "agent_summary": decisions.get("summary"),
        "metrics": tree["metrics"],
        "issues": issues,
        "issue_count": len(issues),
        "gate_checks": [
            _check("source_span_coverage", source_span_coverage_rate == 1.0, source_span_coverage_rate),
            _check("agent_decision_refs_resolve", not any(issue["issue_type"] == "unresolved_source_ref" for issue in issues), None),
            _check("table_candidates_accounted_for", not candidate_ids - accepted_candidates - rejected_candidates - unresolved_candidates, sorted(candidate_ids - accepted_candidates - rejected_candidates - unresolved_candidates)),
            _check("structure_nodes_have_bbox", structure_nodes_have_bbox, None),
        ],
    }
    return tree, quality


def _case_acceptance_issues(
    case: dict[str, Any],
    spans: list[TextSpan],
    nodes: list[dict[str, Any]],
    accepted_candidates: set[str],
    unresolved_candidates: set[str],
    reading_order_override_count: int,
) -> list[dict[str, Any]]:
    issues = []
    if case["focus"] == "producer_info":
        sticky = any(node.get("text", "").startswith("标 ") and ("地址" in node.get("text", "") or "许可证" in node.get("text", "")) for node in nodes)
        if sticky:
            issues.append(_issue("side_marker_sticky", "high", message="Side marker 标 is still sticky with producer fields."))
        text_order = [span.text for span in sorted(spans, key=lambda span: _node_order_key(span))]
        delegate_index = next((idx for idx, text in enumerate(text_order) if "委托" in text), None)
        address_index = next((idx for idx, text in enumerate(text_order) if "地址：深圳" in text), None)
        if delegate_index is not None and address_index is not None and delegate_index > address_index and reading_order_override_count == 0:
            issues.append(_issue("producer_reading_order_unresolved", "medium", message="委托方 still appears after 深圳地址 in source visual order."))
        if not accepted_candidates and not unresolved_candidates:
            issues.append(_issue("producer_table_not_accounted_for", "medium", message="Producer info candidate was neither accepted nor unresolved."))
    if case["focus"] == "nutrition_tables":
        table_nodes = [node for node in nodes if node["node_type"] == "table"]
        expected_table_count = int(case.get("expected_table_count") or 4)
        if len(table_nodes) < expected_table_count:
            issues.append(
                _issue(
                    "nutrition_table_count_below_acceptance",
                    "medium",
                    expected_count=expected_table_count,
                    observed_count=len(table_nodes),
                )
            )
        expected_row_count = case.get("expected_row_count")
        if expected_row_count is not None:
            for table_node in table_nodes:
                observed_row_count = len(table_node.get("children", []))
                if observed_row_count < int(expected_row_count):
                    issues.append(
                        _issue(
                            "nutrition_row_count_below_acceptance",
                            "medium",
                            table_candidate_id=table_node.get("table_candidate_id"),
                            expected_count=int(expected_row_count),
                            observed_count=observed_row_count,
                        )
                    )
        for required in ("能量", "蛋白", "脂肪", "钠"):
            if not any(_normalized_text(required) in _normalized_text(_node_full_text(node)) for node in table_nodes):
                issues.append(_issue("nutrition_required_row_missing_from_tables", "medium", required_text=required))
    for required_text in case.get("required_independent_text", []):
        if not any(_normalized_text(span.text.strip()) == _normalized_text(required_text) for span in spans):
            issues.append(
                _issue(
                    "required_independent_text_missing",
                    "medium",
                    required_text=required_text,
                )
            )
    return issues


def _node_full_text(node: dict[str, Any]) -> str:
    parts = [str(node.get("text", ""))]
    for child in node.get("children", []) if isinstance(node.get("children"), list) else []:
        parts.append(_node_full_text(child))
    return " ".join(parts)


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _table_node_from_candidate(candidate: dict[str, Any], index: int, decision: dict[str, Any]) -> dict[str, Any]:
    row_nodes = []
    for row in candidate.get("rows", []):
        cell_nodes = [
            {
                "node_id": f"{candidate['table_candidate_id']}_{cell['cell_id']}",
                "node_type": "table_cell",
                "page": candidate.get("page"),
                "text": cell.get("text", ""),
                "source_span_ids": cell.get("source_span_ids", []),
                "bbox_normalized": cell.get("bbox_normalized"),
                "children": [],
                "row_index": cell.get("row_index"),
                "col_index": cell.get("col_index"),
                "confidence": _confidence(decision),
                "issues": [],
            }
            for cell in row.get("cells", [])
        ]
        row_nodes.append(
            {
                "node_id": f"{candidate['table_candidate_id']}_row_{int(row.get('row_index', len(row_nodes) + 1)):03d}",
                "node_type": "table_row",
                "page": candidate.get("page"),
                "text": row.get("text", ""),
                "source_span_ids": row.get("source_span_ids", []),
                "bbox_normalized": _bbox_from_child_cells(cell_nodes),
                "children": cell_nodes,
                "row_index": row.get("row_index"),
                "row_type": row.get("row_type"),
                "confidence": _confidence(decision),
                "issues": [],
            }
        )
    return {
        "node_id": stable_id("layout_table", index),
        "node_type": "table",
        "page": candidate.get("page"),
        "text": candidate.get("title", ""),
        "table_type": candidate.get("table_type"),
        "source_span_ids": candidate.get("source_span_ids", []),
        "bbox_normalized": candidate.get("bbox_normalized"),
        "children": row_nodes,
        "reading_order_index": index,
        "confidence": _confidence(decision),
        "issues": [],
        "table_candidate_id": candidate.get("table_candidate_id"),
        "agent_decision_id": decision.get("decision_id"),
        "reason": decision.get("reason"),
    }


def _unresolved_table_node(candidate: dict[str, Any], index: int, decision: dict[str, Any]) -> dict[str, Any]:
    node = _table_node_from_candidate(candidate, index, decision)
    node["node_id"] = stable_id("layout_unresolved_table", index)
    node["confidence"] = min(node.get("confidence", 0.5), 0.49)
    node["issues"] = [{"issue_type": "table_structure_unresolved", "severity": "high", "message": decision.get("reason", "")}]
    return node


def _render_case_html(
    case: dict[str, Any],
    spans: list[TextSpan],
    candidates: dict[str, Any],
    decisions: dict[str, Any],
    structure_tree: dict[str, Any],
    quality_report: dict[str, Any],
    page_image: Path,
) -> str:
    image = _html_image(page_image)
    boxes = "".join(_bbox_div(_to_jsonable(span.bbox_normalized), span.span_id, span.text, "source") for span in spans if span.bbox_normalized)
    table_boxes = "".join(
        _bbox_div(candidate.get("bbox_normalized"), candidate["table_candidate_id"], candidate.get("title", ""), "table")
        for candidate in candidates.get("table_candidates", [])
        if candidate.get("bbox_normalized")
    )
    metrics = quality_report.get("metrics", {})
    body = f"""
<main>
  <header>
    <h1>{_e(case["display_name"])} Layout Smoke</h1>
    <div class="metrics">
      {_metric("layout_status", quality_report.get("status"))}
      {_metric("source_span_coverage_rate", metrics.get("source_span_coverage_rate"))}
      {_metric("accepted_table_count", metrics.get("accepted_table_count"))}
      {_metric("table_cell_count", metrics.get("table_cell_count"))}
      {_metric("reading_order_override_count", metrics.get("reading_order_override_count"))}
      {_metric("unresolved_table_count", metrics.get("unresolved_table_count"))}
      {_metric("unknown_block_count", metrics.get("unknown_block_count"))}
      {_metric("excluded_content_count", metrics.get("excluded_content_count"))}
    </div>
  </header>
  <section class="grid">
    <aside>
      <h2>Source Spans</h2>
      <div class="list">{''.join(_span_row(span) for span in spans)}</div>
      <h2>Table Candidates</h2>
      <div class="list">{''.join(_candidate_row(candidate) for candidate in candidates.get("table_candidates", []))}</div>
      <h2>Agent Decisions</h2>
      <pre>{_e(json.dumps(decisions, ensure_ascii=False, indent=2))}</pre>
    </aside>
    <section>
      <h2>PDF + BBox</h2>
      <div class="page" style="aspect-ratio:{image['width']} / {image['height']}">
        <img src="{image['src']}" alt="page image">
        {boxes}
        {table_boxes}
      </div>
      <h2>Structure Tree</h2>
      <pre>{_e(json.dumps(structure_tree, ensure_ascii=False, indent=2))}</pre>
      <h2>Quality Issues</h2>
      <pre>{_e(json.dumps(quality_report.get("issues", []), ensure_ascii=False, indent=2))}</pre>
    </section>
  </section>
</main>
"""
    css = """
body{margin:0;background:#f7f7f4;color:#191919;font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1600px;margin:0 auto;padding:18px}
h1{font-size:22px;margin:0 0 12px}h2{font-size:14px;margin:18px 0 8px}
.metrics{display:flex;gap:8px;flex-wrap:wrap}.metric{background:white;border:1px solid #ddd;border-radius:6px;padding:6px 8px}
.grid{display:grid;grid-template-columns:460px 1fr;gap:16px;align-items:start}
aside,.page,pre{background:white;border:1px solid #ddd;border-radius:8px}
aside{padding:12px;max-height:calc(100vh - 120px);overflow:auto;position:sticky;top:12px}
.row{border-bottom:1px solid #eee;padding:7px 0}.id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#666}.text{margin-top:3px}
.page{position:relative;overflow:hidden}.page img{display:block;width:100%;height:100%;object-fit:contain}
.box{position:absolute;border:1.5px solid rgba(26,115,232,.85);background:rgba(26,115,232,.08);box-sizing:border-box}
.box.table{border-color:rgba(218,124,0,.95);background:rgba(218,124,0,.08)}
.box:hover{background:rgba(255,0,0,.18);border-color:#c00}
pre{white-space:pre-wrap;word-break:break-word;padding:12px;max-height:520px;overflow:auto}
"""
    return _html_page(f"{case['display_name']} Layout Smoke", body, css)


def _span_node(span: TextSpan) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "page": span.page,
        "text": span.text,
        "source": span.source,
        "bbox_normalized": _to_jsonable(span.bbox_normalized),
    }


def _span_row(span: TextSpan) -> str:
    return f'<div class="row"><div class="id">{_e(span.span_id)}</div><div class="text">{_e(span.text)}</div></div>'


def _candidate_row(candidate: dict[str, Any]) -> str:
    return (
        f'<div class="row"><div class="id">{_e(candidate.get("table_candidate_id"))}</div>'
        f'<div>{_e(candidate.get("table_type"))} rows={len(candidate.get("rows", []))}</div>'
        f'<div class="text">{_e(candidate.get("title"))}</div></div>'
    )


def _metric(label: str, value: Any) -> str:
    return f'<span class="metric"><strong>{_e(label)}</strong>: {_e(value)}</span>'


def _bbox_div(bbox: dict[str, Any] | None, node_id: str, text: str, cls: str) -> str:
    if not bbox:
        return ""
    try:
        x1 = float(bbox["x1"]) * 100
        y1 = float(bbox["y1"]) * 100
        x2 = float(bbox["x2"]) * 100
        y2 = float(bbox["y2"]) * 100
    except (KeyError, TypeError, ValueError):
        return ""
    return (
        f'<div class="box {cls}" title="{_e(node_id)} {_e(text)}" '
        f'style="left:{x1:.4f}%;top:{y1:.4f}%;width:{max(0.1, x2-x1):.4f}%;height:{max(0.1, y2-y1):.4f}%"></div>'
    )


def _html_image(path: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return {
        "width": _png_size(path)[0],
        "height": _png_size(path)[1],
        "src": f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}",
    }


def _html_page(title: str, body: str, css: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>{css}</style>
</head>
<body>{body}</body>
</html>
"""


def _page_image_path(page_images: dict[str, Any], page: int) -> Path:
    for item in page_images.get("pages", []):
        if item.get("page") == page and item.get("render_status") == "rendered" and item.get("path"):
            return Path(item["path"])
    raise RuntimeError(f"Page image not rendered for page {page}: {page_images.get('status')}")


def _union_normalized_bbox(spans: list[TextSpan]) -> dict[str, float] | None:
    boxes = [span.bbox_normalized for span in spans if span.bbox_normalized]
    if not boxes:
        return None
    return {
        "x1": round(min(box.x1 for box in boxes), 6),
        "y1": round(min(box.y1 for box in boxes), 6),
        "x2": round(max(box.x2 for box in boxes), 6),
        "y2": round(max(box.y2 for box in boxes), 6),
    }


def _bbox_from_child_cells(cells: list[dict[str, Any]]) -> dict[str, float] | None:
    boxes = [cell.get("bbox_normalized") for cell in cells if isinstance(cell.get("bbox_normalized"), dict)]
    if not boxes:
        return None
    return {
        "x1": round(min(float(box["x1"]) for box in boxes), 6),
        "y1": round(min(float(box["y1"]) for box in boxes), 6),
        "x2": round(max(float(box["x2"]) for box in boxes), 6),
        "y2": round(max(float(box["y2"]) for box in boxes), 6),
    }


def _node_order_key(span: TextSpan) -> tuple[float, float]:
    if not span.bbox_pdf:
        return (0, 0)
    return (span.bbox_pdf.y, span.bbox_pdf.x)


def _bbox_center(bbox: BBoxPdf) -> tuple[float, float]:
    return (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)


def _confidence(decision: dict[str, Any]) -> float:
    try:
        value = float(decision.get("confidence", 0.5))
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, value))


def _check(check_type: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"check_type": check_type, "status": "pass" if passed else "fail", "details": details}


def _issue(issue_type: str, severity: str, **kwargs: Any) -> dict[str, Any]:
    return {"issue_type": issue_type, "severity": severity, **kwargs}


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (BBoxPdf, BBoxNormalized)):
        return asdict(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return (1, 1)


def _write_preflight_png(path: Path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGP8z8Dwn4GBgYGJAQoAHxcCAsg77S4AAAAASUVORK5CYII="
        )
    )


def _structured_json_from_chat_response(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("LLM response did not include choices[0].message.")
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response message content is empty.")
    content = _json_object_text(content)
    parsed_content = json.loads(content)
    if not isinstance(parsed_content, dict):
        raise RuntimeError("LLM response JSON root must be an object.")
    return parsed_content


def _json_object_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _chat_completions_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    version_segment = base_url.rsplit("/", 1)[-1]
    if len(version_segment) > 1 and version_segment[0] == "v" and version_segment[1:].isdigit():
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _llm_max_tokens() -> int:
    raw_value = os.getenv("LLM_MAX_TOKENS", "8192")
    try:
        value = int(raw_value)
    except ValueError:
        return 8192
    return value if value > 0 else 8192


def _safe_error_text(response: requests.Response) -> str:
    try:
        return response.text[:500]
    except Exception:
        return ""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
