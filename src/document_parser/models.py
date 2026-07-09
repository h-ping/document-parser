from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


RiskLevel = Literal["high", "medium", "low", "info"]
FieldStatus = Literal[
    "extracted",
    "verified",
    "normalized",
    "missing",
    "uncertain",
    "conflict",
    "not_applicable",
    "cannot_verify",
    "manual_review_required",
    "compiled",
    "rejected",
]


@dataclass(frozen=True)
class BBoxPdf:
    x: float
    y: float
    width: float
    height: float
    page_width: float
    page_height: float
    unit: str = "pt"
    origin: str = "top_left"


@dataclass(frozen=True)
class BBoxNormalized:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class PageInfo:
    page: int
    width: float
    height: float


@dataclass(frozen=True)
class TextSpan:
    span_id: str
    page: int
    text: str
    source: str
    bbox_pdf: BBoxPdf | None = None
    bbox_normalized: BBoxNormalized | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class OcrLine:
    ocr_line_id: str
    page: int
    text: str
    confidence: float
    bbox_pdf: BBoxPdf | None = None
    bbox_normalized: BBoxNormalized | None = None
    block_id: str | None = None
    tokens: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VdgNode:
    node_id: str
    node_type: str
    page: int
    text: str = ""
    source_span_ids: list[str] = field(default_factory=list)
    bbox_pdf: BBoxPdf | None = None
    bbox_normalized: BBoxNormalized | None = None


@dataclass(frozen=True)
class FieldDefinition:
    field_def_id: str
    semantic_key: str
    display_name: str
    field_type: str
    criticality: str
    repeatable: bool = False
    semantic_key_type: str = "canonical"
    source_span_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratedSchema:
    schema_id: str
    auto_generated: bool
    schema_version: str
    sections: list[dict[str, Any]]
    entity_types: list[dict[str, Any]]
    field_definitions: list[FieldDefinition]
    table_definitions: list[dict[str, Any]] = field(default_factory=list)
    requirement_definitions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SpanRange:
    span_id: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class ValueSource:
    mode: str
    ranges: list[SpanRange] = field(default_factory=list)


@dataclass(frozen=True)
class FieldPlan:
    field_plan_id: str
    semantic_key: str
    display_name: str
    field_type: str
    section_id: str | None
    entity_id: str | None
    value_source: ValueSource
    criticality: str
    confidence: dict[str, float | None]
    boundary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionPlan:
    plan_id: str
    schema_id: str
    fields: list[FieldPlan]
    entities: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    requirements: list[dict[str, Any]] = field(default_factory=list)
    ignored_nodes: list[str] = field(default_factory=list)
    unknown_nodes: list[str] = field(default_factory=list)
    ignored_node_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_text: str
    page: int
    extraction_methods: list[str]
    bbox_status: str
    source_node_ids: list[str] = field(default_factory=list)
    bbox_pdf: BBoxPdf | None = None
    bbox_normalized: BBoxNormalized | None = None


@dataclass(frozen=True)
class CompiledField:
    field_id: str
    semantic_key: str
    display_name: str
    field_type: str
    raw_value: str
    clean_value: str
    normalized_value: str
    value_hash: str
    status: FieldStatus
    criticality: str
    confidence: dict[str, float | None]
    risk_level: RiskLevel
    review_required: bool
    section_id: str | None
    entity_id: str | None
    table_id: str | None
    row_key: str | None
    evidence_refs: list[str]
    normalization: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True)
class Risk:
    risk_id: str
    target_type: str
    target_id: str
    risk_level: RiskLevel
    risk_type: str
    message: str
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewTask:
    task_id: str
    target_type: str
    target_id: str
    risk_level: RiskLevel
    reason: str
    required: bool
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParseResult:
    job: dict[str, Any]
    document: dict[str, Any]
    generated_schema: GeneratedSchema
    extracted_data: dict[str, Any]
    evidence: list[Evidence]
    cross_validation: dict[str, Any]
    coverage: dict[str, Any]
    validation: list[dict[str, Any]]
    quality: dict[str, Any]
    risks: list[Risk]
    review_tasks: list[ReviewTask]
    metadata: dict[str, Any]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
