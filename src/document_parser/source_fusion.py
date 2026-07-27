from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Final

from .models import BBoxPdf, OcrLine, TextSpan
from .utils import stable_id


MIN_TEXT_SIMILARITY: Final = 0.72
MIN_CONTAINMENT_RATIO: Final = 0.6
MIN_VERTICAL_OVERLAP: Final = 0.45


@dataclass(frozen=True, slots=True)
class SourceAlignment:
    alignment_id: str
    page: int
    pdf_span_ids: list[str]
    ocr_line_id: str
    pdf_text: str
    ocr_text: str
    text_relation: str
    text_similarity: float


@dataclass(frozen=True, slots=True)
class SourceFusionResult:
    canonical_spans: list[TextSpan]
    alignments: list[SourceAlignment]
    unmatched_ocr_line_ids: list[str]
    report: dict[str, int | float | str | bool]

    def agent_context(self) -> dict[str, list[dict[str, str | float | list[str]]]]:
        return {
            "alternate_readings": [
                {
                    "alignment_id": alignment.alignment_id,
                    "pdf_span_ids": alignment.pdf_span_ids,
                    "ocr_line_id": alignment.ocr_line_id,
                    "pdf_text": alignment.pdf_text,
                    "ocr_text": alignment.ocr_text,
                    "text_relation": alignment.text_relation,
                    "text_similarity": alignment.text_similarity,
                }
                for alignment in self.alignments
                if alignment.text_relation != "equivalent"
            ]
        }


def build_source_fusion(
    pdf_spans: list[TextSpan],
    ocr_lines: list[OcrLine],
    *,
    enabled: bool,
) -> SourceFusionResult:
    if not enabled:
        canonical_spans = [*pdf_spans, *_ocr_spans(ocr_lines)]
        return SourceFusionResult(
            canonical_spans=canonical_spans,
            alignments=[],
            unmatched_ocr_line_ids=[line.ocr_line_id for line in ocr_lines],
            report={
                "artifact_version": "source_fusion_report_v0.1",
                "status": "disabled",
                "enabled": False,
                "pdf_span_count": len(pdf_spans),
                "ocr_line_count": len(ocr_lines),
                "aligned_ocr_line_count": 0,
                "unmatched_ocr_line_count": len(ocr_lines),
                "canonical_span_count": len(canonical_spans),
                "duplicate_span_count_prevented": 0,
                "superseded_adhesion_span_count": 0,
            },
        )

    superseded_ids = _superseded_adhesion_span_ids(pdf_spans, ocr_lines)
    active_pdf_spans = [span for span in pdf_spans if span.span_id not in superseded_ids]
    by_page: dict[int, list[TextSpan]] = {}
    for span in active_pdf_spans:
        by_page.setdefault(span.page, []).append(span)

    alignments: list[SourceAlignment] = []
    unmatched: list[OcrLine] = []
    for line in ocr_lines:
        matched_spans = _matching_pdf_spans(line, by_page.get(line.page, []))
        if not matched_spans:
            unmatched.append(line)
            continue
        pdf_text = "".join(span.text for span in matched_spans)
        similarity = _text_similarity(pdf_text, line.text)
        alignments.append(
            SourceAlignment(
                alignment_id=stable_id("source_alignment", len(alignments) + 1),
                page=line.page,
                pdf_span_ids=[span.span_id for span in matched_spans],
                ocr_line_id=line.ocr_line_id,
                pdf_text=pdf_text,
                ocr_text=line.text,
                text_relation="equivalent" if _normalized_text(pdf_text) == _normalized_text(line.text) else "alternate",
                text_similarity=round(similarity, 4),
            )
        )

    canonical_spans = [*active_pdf_spans, *_ocr_spans(unmatched)]
    return SourceFusionResult(
        canonical_spans=canonical_spans,
        alignments=alignments,
        unmatched_ocr_line_ids=[line.ocr_line_id for line in unmatched],
        report={
            "artifact_version": "source_fusion_report_v0.1",
            "status": "pass",
            "enabled": True,
            "pdf_span_count": len(pdf_spans),
            "ocr_line_count": len(ocr_lines),
            "aligned_ocr_line_count": len(alignments),
            "unmatched_ocr_line_count": len(unmatched),
            "canonical_span_count": len(canonical_spans),
            "duplicate_span_count_prevented": len(alignments),
            "superseded_adhesion_span_count": len(superseded_ids),
        },
    )


def _ocr_spans(lines: list[OcrLine]) -> list[TextSpan]:
    return [
        TextSpan(
            span_id=stable_id("ocr_span", index),
            page=line.page,
            text=line.text,
            source="ocr",
            bbox_pdf=line.bbox_pdf,
            bbox_normalized=line.bbox_normalized,
            confidence=line.confidence,
        )
        for index, line in enumerate(lines, start=1)
    ]


def _matching_pdf_spans(line: OcrLine, page_spans: list[TextSpan]) -> list[TextSpan]:
    normalized_ocr = _normalized_text(line.text)
    exact = [span for span in page_spans if _normalized_text(span.text) == normalized_ocr]
    exact_on_line = [span for span in exact if _same_visual_line(span.bbox_pdf, line.bbox_pdf)]
    if exact_on_line:
        return [max(exact_on_line, key=lambda span: _bbox_overlap_area(span.bbox_pdf, line.bbox_pdf))]
    if len(exact) == 1:
        return exact

    geometric = [span for span in page_spans if _same_visual_line(span.bbox_pdf, line.bbox_pdf)]
    if not geometric:
        return []
    geometric.sort(key=lambda span: (span.bbox_pdf.x if span.bbox_pdf else 0.0, span.span_id))
    combined = "".join(span.text for span in geometric)
    if _texts_match(combined, line.text):
        return geometric

    individually_matching = [span for span in geometric if _texts_match(span.text, line.text)]
    return individually_matching[:1]


def _superseded_adhesion_span_ids(pdf_spans: list[TextSpan], ocr_lines: list[OcrLine]) -> set[str]:
    lines_by_page: dict[int, list[OcrLine]] = {}
    for line in ocr_lines:
        lines_by_page.setdefault(line.page, []).append(line)

    superseded: set[str] = set()
    for span in pdf_spans:
        target = _normalized_text(span.text)
        segments = [
            line
            for line in lines_by_page.get(span.page, [])
            if 0 < len(_normalized_text(line.text)) < len(target)
            and _normalized_text(line.text) in target
            and _same_visual_line(span.bbox_pdf, line.bbox_pdf)
        ]
        segments.sort(key=lambda line: (line.bbox_pdf.x if line.bbox_pdf else 0.0, line.ocr_line_id))
        if _has_exact_ocr_partition(target, segments):
            superseded.add(span.span_id)
    return superseded


def _has_exact_ocr_partition(target: str, segments: list[OcrLine]) -> bool:
    for start in range(len(segments)):
        combined = ""
        for end in range(start, len(segments)):
            combined += _normalized_text(segments[end].text)
            candidate = segments[start : end + 1]
            if end > start and combined == target and _has_visual_gaps(candidate):
                return True
            if not target.startswith(combined):
                break
    return False


def _has_visual_gaps(segments: list[OcrLine]) -> bool:
    for left, right in zip(segments, segments[1:]):
        if left.bbox_pdf is None or right.bbox_pdf is None:
            return False
        if right.bbox_pdf.x - (left.bbox_pdf.x + left.bbox_pdf.width) < 4.0:
            return False
    return True


def _bbox_overlap_area(first: BBoxPdf | None, second: BBoxPdf | None) -> float:
    if first is None or second is None:
        return 0.0
    width = max(0.0, min(first.x + first.width, second.x + second.width) - max(first.x, second.x))
    height = max(0.0, min(first.y + first.height, second.y + second.height) - max(first.y, second.y))
    return width * height


def _same_visual_line(first: BBoxPdf | None, second: BBoxPdf | None) -> bool:
    if first is None or second is None:
        return False
    vertical_overlap = max(0.0, min(first.y + first.height, second.y + second.height) - max(first.y, second.y))
    minimum_height = min(first.height, second.height)
    if minimum_height <= 0 or vertical_overlap / minimum_height < MIN_VERTICAL_OVERLAP:
        return False
    horizontal_overlap = max(0.0, min(first.x + first.width, second.x + second.width) - max(first.x, second.x))
    return horizontal_overlap > 0


def _texts_match(first: str, second: str) -> bool:
    normalized_first = _normalized_text(first)
    normalized_second = _normalized_text(second)
    if not normalized_first or not normalized_second:
        return False
    shorter, longer = sorted((normalized_first, normalized_second), key=len)
    containment_ratio = len(shorter) / len(longer)
    return (
        _text_similarity(normalized_first, normalized_second) >= MIN_TEXT_SIMILARITY
        or (shorter in longer and containment_ratio >= MIN_CONTAINMENT_RATIO)
    )


def _text_similarity(first: str, second: str) -> float:
    return SequenceMatcher(None, _normalized_text(first), _normalized_text(second)).ratio()


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[\s:：,，。.;；()（）\[\]【】]+", "", normalized)
