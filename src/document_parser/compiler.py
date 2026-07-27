from __future__ import annotations

from .agents import clean_field_value, normalize_value
from .field_validation import format_validation_confidence
from .models import CompiledField, Evidence, ExtractionPlan, TextSpan
from .pdf import normalized_bbox_from_pdf
from .utils import sha256_text, stable_id


class CompilerError(RuntimeError):
    pass


class DeterministicCompiler:
    def compile(self, plan: ExtractionPlan, spans: list[TextSpan]) -> tuple[dict[str, CompiledField], list[Evidence]]:
        span_by_id = {span.span_id: span for span in spans}
        fields: dict[str, CompiledField] = {}
        evidence: list[Evidence] = []
        evidence_by_key: dict[tuple[str, int, int], str] = {}

        for field_index, field_plan in enumerate(plan.fields, start=1):
            raw_parts: list[str] = []
            raw_part_spans: list[TextSpan] = []
            evidence_refs: list[str] = []
            evidence_confidence = 1.0
            has_bbox = True

            for span_range in field_plan.value_source.ranges:
                source_span = span_by_id.get(span_range.span_id)
                if source_span is None:
                    raise CompilerError(f"Extraction plan references missing span_id: {span_range.span_id}")
                if span_range.start_offset < 0 or span_range.end_offset > len(source_span.text):
                    raise CompilerError(
                        f"Extraction plan range is outside span {span_range.span_id}: "
                        f"{span_range.start_offset}:{span_range.end_offset} for length {len(source_span.text)}"
                    )
                if span_range.start_offset >= span_range.end_offset:
                    raise CompilerError(
                        f"Extraction plan range must be non-empty and ordered for span {span_range.span_id}: "
                        f"{span_range.start_offset}:{span_range.end_offset}"
                    )
                raw_text = source_span.text[span_range.start_offset : span_range.end_offset]
                raw_parts.append(raw_text)
                raw_part_spans.append(source_span)
                evidence_key = (span_range.span_id, span_range.start_offset, span_range.end_offset)
                evidence_id = evidence_by_key.get(evidence_key)
                if evidence_id is None:
                    evidence_id = stable_id("ev", len(evidence) + 1)
                    evidence_by_key[evidence_key] = evidence_id
                    bbox_normalized = source_span.bbox_normalized
                    if source_span.bbox_pdf and not bbox_normalized:
                        bbox_normalized = normalized_bbox_from_pdf(source_span.bbox_pdf)
                    evidence.append(
                        Evidence(
                            evidence_id=evidence_id,
                            source_text=raw_text,
                            page=source_span.page,
                            extraction_methods=[source_span.source],
                            bbox_status="available" if source_span.bbox_pdf and bbox_normalized else "missing",
                            source_node_ids=[source_span.span_id],
                            bbox_pdf=source_span.bbox_pdf,
                            bbox_normalized=bbox_normalized,
                        )
                    )
                evidence_refs.append(evidence_id)
                if not source_span.bbox_pdf:
                    has_bbox = False
                    evidence_confidence = min(evidence_confidence, 0.80)

            raw_value = _join_raw_parts(raw_parts, raw_part_spans).strip()
            clean_value, clean_normalization = clean_field_value(raw_value, field_plan.display_name)
            normalized_value, value_normalization = normalize_value(clean_value, field_plan.field_type)
            normalization = clean_normalization + value_normalization

            confidence = dict(field_plan.confidence)
            confidence["evidence_confidence"] = evidence_confidence
            confidence["format_validation_confidence"] = format_validation_confidence(
                field_plan.semantic_key,
                field_plan.field_type,
                normalized_value,
            )
            confidence["overall"] = min(value for value in confidence.values() if isinstance(value, float))

            critical_low_confidence = field_plan.criticality == "critical" and confidence["overall"] < 0.95
            critical_missing_bbox = field_plan.criticality == "critical" and not has_bbox
            uncertain_normalization = bool(normalization) and confidence["overall"] < 0.95
            review_required = critical_low_confidence or critical_missing_bbox or uncertain_normalization
            if critical_low_confidence or critical_missing_bbox:
                risk_level = "high"
            elif uncertain_normalization:
                risk_level = "medium"
            elif normalization:
                risk_level = "low"
            else:
                risk_level = "info"
            status = "manual_review_required" if review_required else ("normalized" if normalization else "verified")
            reason = None
            if critical_missing_bbox:
                reason = "关键字段缺少 bbox"
            elif critical_low_confidence:
                reason = "关键字段置信度低于0.95"
            elif uncertain_normalization:
                reason = "归一化字段置信度低于0.95"

            field_id = stable_id("fld", field_index)
            fields[field_id] = CompiledField(
                field_id=field_id,
                semantic_key=field_plan.semantic_key,
                display_name=field_plan.display_name,
                field_type=field_plan.field_type,
                raw_value=raw_value,
                clean_value=clean_value,
                normalized_value=normalized_value,
                value_hash=sha256_text(normalized_value),
                status=status,
                criticality=field_plan.criticality,
                confidence=confidence,
                risk_level=risk_level,
                review_required=review_required,
                section_id=field_plan.section_id,
                entity_id=field_plan.entity_id,
                table_id=None,
                row_key=None,
                evidence_refs=evidence_refs,
                normalization=normalization,
                reason=reason,
            )

        return fields, evidence


def _join_raw_parts(parts: list[str], spans: list[TextSpan]) -> str:
    if not parts:
        return ""
    value = parts[0]
    for index in range(1, len(parts)):
        value += _span_separator(spans[index - 1], spans[index], parts[index - 1], parts[index]) + parts[index]
    return value


def _span_separator(previous: TextSpan, current: TextSpan, previous_text: str, current_text: str) -> str:
    if previous.source != "pdf_char_atom" or current.source != "pdf_char_atom":
        return "\n"
    if previous.page != current.page or previous.bbox_pdf is None or current.bbox_pdf is None:
        return "\n"
    previous_center = previous.bbox_pdf.y + previous.bbox_pdf.height / 2
    current_center = current.bbox_pdf.y + current.bbox_pdf.height / 2
    same_line_tolerance = max(1.25, min(previous.bbox_pdf.height, current.bbox_pdf.height) * 0.35)
    if abs(previous_center - current_center) > same_line_tolerance:
        return "\n"
    gap = current.bbox_pdf.x - (previous.bbox_pdf.x + previous.bbox_pdf.width)
    if gap > 1.0 and previous_text[-1:].isascii() and previous_text[-1:].isalnum() and current_text[:1].isascii() and current_text[:1].isalnum():
        return " "
    return ""
