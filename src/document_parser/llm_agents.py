from __future__ import annotations

import copy
from typing import Any

from .llm import LlmClient
from .models import GeneratedSchema, TextSpan, to_jsonable


TABLE_AGENT_MAX_TOKENS = 32768


SEMANTIC_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "issue_type": {"type": "string"},
                    "block_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "source_span_ids": {"type": "array", "maxItems": 40, "items": {"type": "string"}},
                    "message": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "repair_required": {"type": "boolean"},
                },
                "required": [
                    "issue_type",
                    "block_id",
                    "target_id",
                    "source_span_ids",
                    "message",
                    "severity",
                    "repair_required",
                ],
            },
        }
    },
    "required": ["findings"],
}


SCHEMA_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sections": {"type": "array", "maxItems": 12, "items": {"type": "object", "additionalProperties": True}},
        "entity_types": {"type": "array", "maxItems": 12, "items": {"type": "object", "additionalProperties": True}},
        "field_definitions": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "semantic_key": {"type": "string"},
                    "display_name": {"type": "string"},
                    "field_type": {"type": "string"},
                    "criticality": {"type": "string"},
                    "repeatable": {"type": "boolean"},
                    "source_span_ids": {"type": "array", "maxItems": 20, "items": {"type": "string"}},
                },
                "required": ["semantic_key", "display_name", "field_type", "criticality", "repeatable", "source_span_ids"],
            },
        },
        "table_definitions": {"type": "array", "maxItems": 20, "items": {"type": "object", "additionalProperties": True}},
        "requirement_definitions": {"type": "array", "maxItems": 30, "items": {"type": "object", "additionalProperties": True}},
    },
    "required": ["sections", "entity_types", "field_definitions", "table_definitions", "requirement_definitions"],
}


AGENT_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fields": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "semantic_key": {"type": "string"},
                    "display_name": {"type": "string"},
                    "field_type": {"type": "string"},
                    "span_id": {"type": "string"},
                    "start_offset": {"type": "integer"},
                    "end_offset": {"type": "integer"},
                    "text": {"type": "string"},
                    "ranges": {
                        "type": "array",
                        "maxItems": 64,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "span_id": {"type": "string"},
                                "start_offset": {"type": "integer"},
                                "end_offset": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["span_id", "start_offset", "end_offset", "text"],
                        },
                    },
                    "confidence": {"type": "number"},
                    "entity_id": {"type": ["string", "null"]},
                    "section_id": {"type": ["string", "null"]},
                    "supporting_node_ids": {"type": "array", "items": {"type": "string"}},
                    "boundary_edge_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "semantic_key",
                    "display_name",
                    "field_type",
                    "confidence",
                    "entity_id",
                    "section_id",
                ],
                "anyOf": [
                    {"required": ["span_id", "start_offset", "end_offset", "text"]},
                    {"required": ["ranges"]},
                ],
            },
        },
        "entities": {"type": "array", "maxItems": 20, "items": {"type": "object", "additionalProperties": True}},
        "tables": {"type": "array", "maxItems": 12, "items": {"type": "object", "additionalProperties": True}},
        "requirements": {"type": "array", "maxItems": 20, "items": {"type": "object", "additionalProperties": True}},
        "ignored_nodes": {"type": "array", "maxItems": 40, "items": {"type": "string"}},
        "unknown_nodes": {"type": "array", "maxItems": 40, "items": {"type": "string"}},
        "node_scope_decisions": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node_id": {"type": "string"},
                    "scope_status": {"type": "string"},
                    "scope_category": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["node_id", "scope_status", "scope_category", "reason", "confidence"],
            },
        },
        "layout_candidate_decisions": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "layout_candidate_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["accept", "reject", "unresolved"]},
                    "source_span_ids": {"type": "array", "maxItems": 80, "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["layout_candidate_id", "decision", "source_span_ids", "reason", "confidence"],
            },
        },
    },
    "required": ["fields", "entities", "tables", "requirements", "ignored_nodes", "unknown_nodes", "layout_candidate_decisions"],
}

GLOBAL_RECONCILIATION_SCHEMA = copy.deepcopy(AGENT_ITEMS_SCHEMA)
GLOBAL_RECONCILIATION_SCHEMA["properties"]["fields"]["maxItems"] = 160
GLOBAL_RECONCILIATION_SCHEMA["properties"]["entities"]["maxItems"] = 80
GLOBAL_RECONCILIATION_SCHEMA["properties"]["tables"]["maxItems"] = 30


class SpanGroundedFieldAgent:
    def __init__(self, llm_client: LlmClient, max_spans: int = 240) -> None:
        self._llm_client = llm_client
        self._max_spans = max_spans

    def generate_schema(self, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        visible_spans = spans[: self._max_spans]
        return self._llm_client.structured_json(
            system=_schema_system_prompt(),
            user=_schema_user_prompt(visible_spans, vdg_context),
            schema=SCHEMA_AGENT_SCHEMA,
        )

    def generate_extraction_plan(self, schema: GeneratedSchema, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        visible_spans = spans[: self._max_spans]
        return self._llm_client.structured_json(
            system=_extraction_system_prompt(),
            user=_extraction_user_prompt(schema, visible_spans, vdg_context),
            schema=AGENT_ITEMS_SCHEMA,
        )

    def generate_field_extraction_plan(self, schema: GeneratedSchema, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        visible_spans = spans[: self._max_spans]
        return self._llm_client.structured_json(
            system=_field_extraction_system_prompt(),
            user=_field_extraction_user_prompt(schema, visible_spans, vdg_context),
            schema=AGENT_ITEMS_SCHEMA,
        )

    def generate_table_extraction_plan(self, schema: GeneratedSchema, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        visible_spans = spans[: self._max_spans]
        return self._llm_client.structured_json_with_max_tokens(
            system=_table_extraction_system_prompt(),
            user=_table_extraction_user_prompt(schema, visible_spans, vdg_context),
            schema=AGENT_ITEMS_SCHEMA,
            max_tokens=TABLE_AGENT_MAX_TOKENS,
        )

    def review_compiled_blocks(self, review_input: dict[str, Any]) -> dict[str, Any]:
        return self._llm_client.structured_json(
            system=(
                "You are an independent semantic review agent for packaging-label extraction. "
                "Review compiled results against only the supplied source spans. Return findings only. "
                "Do not generate, repair, rewrite, normalize, or infer final values. Check missing schema anchors, "
                "repeatable entities, truncated long fields, field adhesion, identifier completeness, entity/address "
                "ownership, nutrition row/cell consistency, and unresolved important text. Every finding must cite "
                "a supplied block_id and source_span_ids. Return compact JSON only."
            ),
            user=f"Compiled block review input:\n{to_jsonable(review_input)}",
            schema=SEMANTIC_REVIEW_SCHEMA,
        )

    def reconcile_extraction_plan(
        self,
        schema: GeneratedSchema,
        spans: list[TextSpan],
        proposals: dict[str, Any],
        vdg_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        span_by_id = {span.span_id: span for span in spans}
        referenced_ids = _proposal_span_ids(proposals)
        source_index = [
            {
                "span_id": span_id,
                "page": span_by_id[span_id].page,
                "text": span_by_id[span_id].text,
                "bbox_pdf": to_jsonable(span_by_id[span_id].bbox_pdf),
            }
            for span_id in referenced_ids
            if span_id in span_by_id
        ]
        return self._llm_client.structured_json_with_max_tokens(
            system=(
                "You are the global reconciliation agent for packaging-label extraction proposals. "
                "Return compact JSON only. Resolve duplicate PDF/OCR observations, unify repeated entity ownership, "
                "and select one authoritative proposal per non-repeatable semantic slot. Preserve genuinely repeated "
                "content items, manufacturers, claims, and tables. You may combine existing ranges, but every output "
                "range must be copied exactly from the supplied proposals. Do not create text, spans, offsets, values, "
                "or table cells. When ownership cannot be resolved, omit the field and put its source span in unknown_nodes."
            ),
            user=(
                f"Document schema:\n{_schema_prompt_view(schema)}\n\n"
                f"Candidate proposals:\n{to_jsonable(proposals)}\n\n"
                f"Referenced source index:\n{source_index}\n\n"
                f"Global structure context:\n{_reconciliation_context(vdg_context)}"
            ),
            schema=GLOBAL_RECONCILIATION_SCHEMA,
            max_tokens=TABLE_AGENT_MAX_TOKENS,
        )

    def generate_candidates(self, schema: GeneratedSchema, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        plan = self.generate_extraction_plan(schema, spans, vdg_context)
        return {"items": plan.get("fields", plan.get("items", []))}


def _schema_system_prompt() -> str:
    return (
        "You are a packaging-label schema agent. "
        "Return a compact valid JSON object only, with no markdown and no explanation. "
        "Generate the document schema from the provided source spans. "
        "Return field definitions, entity types, table definitions, and requirement definitions only. "
        "Do not output final field values, summaries, inferred facts, or rewritten text. "
        "Every definition must cite source_span_ids from the provided spans when an anchor exists."
    )


def _schema_user_prompt(spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> str:
    span_lines = "\n".join(
        f"- span_id={span.span_id}; page={span.page}; text={span.text!r}"
        for span in spans
    )
    return (
        "Source spans:\n"
        f"{span_lines}\n\n"
        f"{_vdg_context_text(vdg_context)}"
        "Build a dynamic packaging-label schema. Include repeated manufacturers, content items, barcodes, "
        "nutrition_facts tables, and requirement/note text when present. "
        "Return reusable definitions only, not instance rows or per-flavor duplicates. "
        "For multiple content items or nutrition tables, create repeatable definitions with source anchors; "
        "leave instance extraction to the extraction and table agents."
    )


def _extraction_system_prompt() -> str:
    return (
        "You are a packaging-label extraction planning agent. "
        "Return a compact valid JSON object only, with no markdown and no explanation. "
        "Return an evidence-bound extraction plan, not final JSON values. "
        "Every field must cite either one provided span_id or an ordered ranges array with zero-based Python string offsets. "
        "Use ranges for values that continue across multiple atoms. Every range text must be an exact substring of its span. "
        "Decide field boundaries, entity grouping, table/list semantics, and section assignment. "
        "For every provided layout candidate, return an accept, reject, or unresolved layout_candidate_decision. "
        "A layout candidate is only accepted when its cited atom spans support the proposed structure. "
        "In this main extraction plan, do not expand nutrition table rows or cells; table reconstruction is handled by "
        "the table agent or parser candidates. "
        "Keep the output compact: prefer high-confidence label fields, and do not return more than 40 fields. "
        "Do not infer, summarize, translate, normalize, or fill missing values. "
        "If unsure, omit the field or mark the source span as unknown_nodes."
    )


def _field_extraction_system_prompt() -> str:
    return (
        "You are the field-boundary planning agent for packaging labels. "
        "Return a compact valid JSON object only, with no markdown and no explanation. "
        "Return only evidence-bound field plans in the fields array; leave tables, entities, requirements, "
        "ignored_nodes, and unknown_nodes as arrays. "
        "Also return accept, reject, or unresolved layout_candidate_decisions for every provided layout candidate. "
        "Each field must use a semantic_key from the generated schema when available and cite one span_id or ordered ranges. "
        "Every cited range must include zero-based Python string start_offset/end_offset and exact source text. "
        "Do not infer, summarize, normalize, translate, or write the final JSON output."
    )


def _table_extraction_system_prompt() -> str:
    return (
        "You are the table/list semantics planning agent for packaging labels. "
        "Return a compact valid JSON object only, with no markdown and no explanation. "
        "Return evidence-bound table plans in the tables array; leave fields, entities, requirements, "
        "ignored_nodes, and unknown_nodes as arrays. "
        "Also return accept, reject, or unresolved layout_candidate_decisions for every provided layout candidate. "
        "For nutrition facts, reconstruct rows and cells only from provided spans. "
        "In this retry, return only nutrition_facts tables from the provided VDG table candidates; do not expand producer or operator tables. "
        "For every nutrition candidate, return all visible title, header, and data rows. A title-only table is invalid; "
        "if rows cannot be completed from evidence, omit that table and mark its layout candidate unresolved. "
        "Treat a same-row --/— marker plus 饱和脂肪、反式脂肪酸或糖 as one nutrient label cell; do not emit the marker as its own cell. "
        "Use consistent item, amount-with-unit, and NRV column semantics for every data row. "
        "Each cell must cite span_id plus zero-based Python string start_offset/end_offset or exact text from that span. "
        "Do not infer missing cells, normalize values, translate, summarize, or write final JSON."
    )


def _extraction_user_prompt(schema: GeneratedSchema, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> str:
    span_lines = "\n".join(
        f"- span_id={span.span_id}; page={span.page}; text={span.text!r}"
        for span in spans
    )
    return (
        "Generated schema summary:\n"
        f"{_schema_prompt_view(schema)}\n\n"
        "Source spans:\n"
        f"{span_lines}\n\n"
        f"{_vdg_context_text(vdg_context)}"
        "Create the extraction plan. Use schema semantic keys where possible. "
        "Emit each visually distinct unlabeled printed claim, pack marker, serving marker, or qualification text as a separate "
        "repeatable custom.other_label_text field; do not concatenate non-adjacent printed text into one field. "
        "Return field and entity plans first. Do not expand nutrition table rows/cells in this main plan; "
        "leave large nutrition tables to the table agent or parser/VDG candidates. "
        "In unknown_nodes, ignored_nodes, and node_scope_decisions, list only important or ambiguous nodes, "
        "not every unextracted span."
    )


def _field_extraction_user_prompt(schema: GeneratedSchema, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> str:
    span_lines = "\n".join(
        f"- span_id={span.span_id}; page={span.page}; text={span.text!r}"
        for span in spans
    )
    field_definitions = [
        {
            "semantic_key": definition.semantic_key,
            "display_name": definition.display_name,
            "field_type": definition.field_type,
            "criticality": definition.criticality,
            "repeatable": definition.repeatable,
            "source_span_ids": definition.source_span_ids,
        }
        for definition in schema.field_definitions
    ]
    return (
        "Generated field definitions:\n"
        f"{field_definitions}\n\n"
        "Source spans:\n"
        f"{span_lines}\n\n"
        f"{_vdg_context_text(vdg_context)}"
        "Create field boundary plans for all schema fields that are explicitly present in the source spans. "
        "Keep repeatable custom.other_label_text instances separate by visual boundary and source ranges. "
        "Return no guessed fields. If a schema field is absent, omit it."
    )


def _schema_prompt_view(schema: GeneratedSchema) -> dict[str, Any]:
    return {
        "sections": schema.sections[:12],
        "entity_types": schema.entity_types[:12],
        "field_definitions": [
            {
                "semantic_key": definition.semantic_key,
                "display_name": definition.display_name,
                "field_type": definition.field_type,
                "criticality": definition.criticality,
                "repeatable": definition.repeatable,
                "source_span_ids": definition.source_span_ids[:8],
            }
            for definition in schema.field_definitions[:80]
        ],
        "table_definitions": schema.table_definitions[:20],
        "requirement_definitions": schema.requirement_definitions[:20],
    }


def _table_extraction_user_prompt(schema: GeneratedSchema, spans: list[TextSpan], vdg_context: dict[str, Any] | None = None) -> str:
    span_lines = "\n".join(
        f"- span_id={span.span_id}; page={span.page}; text={span.text!r}"
        for span in spans
    )
    return (
        "Generated table definitions:\n"
        f"{schema.table_definitions}\n\n"
        "Source spans:\n"
        f"{span_lines}\n\n"
        f"{_vdg_context_text(vdg_context)}"
        "Create table/list extraction plans for tables that are explicitly present. "
        "For a nutrition_facts table, return rows with row_key and cells. Use column ids such as col_001, col_002, col_003, col_004."
    )


def _vdg_context_text(vdg_context: dict[str, Any] | None) -> str:
    if not vdg_context:
        return ""
    return (
        "VDG context for visual/semantic boundaries:\n"
        f"{to_jsonable(vdg_context)}\n\n"
        "Use VDG regions, table cells, reading order and same-row/same-column relationships to avoid field adhesion. "
        "When useful, include supporting_node_ids and boundary_edge_ids for each field plan.\n\n"
        "If label_text_scope context is present, use it to decide whether each text node is final printed packaging label text, "
        "out-of-scope noise, or unknown_scope. Put only important uncertain nodes in unknown_nodes; "
        "do not enumerate node_scope_decisions for every node.\n\n"
    )


def _proposal_span_ids(proposals: dict[str, Any]) -> list[str]:
    span_ids: list[str] = []
    for field in proposals.get("fields", []):
        if not isinstance(field, dict):
            continue
        if field.get("span_id"):
            span_ids.append(str(field["span_id"]))
        for span_range in field.get("ranges", []):
            if isinstance(span_range, dict) and span_range.get("span_id"):
                span_ids.append(str(span_range["span_id"]))
    return list(dict.fromkeys(span_ids))


def _reconciliation_context(vdg_context: dict[str, Any] | None) -> dict[str, Any]:
    if not vdg_context:
        return {}
    return {
        "regions": vdg_context.get("regions", []),
        "table_candidates": vdg_context.get("table_candidates", []),
        "source_fusion": vdg_context.get("source_fusion", {}),
        "quality_issues": vdg_context.get("quality_issues", []),
        "label_text_scope": vdg_context.get("label_text_scope", {}),
        "semantic_review_findings": vdg_context.get("semantic_review_findings", []),
        "previous_reconciled_plan": vdg_context.get("previous_reconciled_plan", {}),
    }
