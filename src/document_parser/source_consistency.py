from __future__ import annotations

import re
from typing import Any

from .models import BBoxPdf, OcrLine, TextSpan
from .utils import stable_id


IMPORTANT_LABEL_RE = re.compile(
    r"(品名|产品名称|配料|净含量|规格|产品标准|执行标准|许可证编号|生产者|生产商|地址|产地|"
    r"保质期|贮存条件|储存条件|商品条码|外箱条码|营养成分表|内容物\s*\d+)"
)


def build_source_consistency_report(
    pdf_spans: list[TextSpan],
    ocr_lines: list[OcrLine],
    low_confidence_threshold: float = 0.80,
) -> dict[str, Any]:
    if not ocr_lines:
        return {
            "status": "skipped_no_ocr",
            "pdf_text_span_count": len(pdf_spans),
            "ocr_line_count": 0,
            "matched_ocr_line_count": 0,
            "issue_count": 0,
            "issues": [],
            "page_reports": [],
        }

    pdf_by_page = _group_by_page(pdf_spans)
    ocr_by_page = _group_by_page(ocr_lines)
    issues: list[dict[str, Any]] = []
    page_reports: list[dict[str, Any]] = []
    matched_ocr_line_count = 0

    for page in sorted(set(pdf_by_page) | set(ocr_by_page)):
        page_pdf_spans = pdf_by_page.get(page, [])
        page_ocr_lines = ocr_by_page.get(page, [])
        matched_ocr_ids: set[str] = set()
        matched_pdf_ids: set[str] = set()
        page_conflicts = []

        for ocr_line in page_ocr_lines:
            best_match = _best_text_match(ocr_line.text, page_pdf_spans)
            if best_match:
                matched_ocr_ids.add(ocr_line.ocr_line_id)
                matched_pdf_ids.add(best_match.span_id)
            elif page_pdf_spans:
                overlapping_span = _best_bbox_overlap(ocr_line.bbox_pdf, page_pdf_spans)
                if overlapping_span:
                    issue = _issue(
                        issues,
                        "pdf_ocr_text_conflict",
                        "medium" if _is_important(ocr_line.text) or _is_important(overlapping_span.text) else "low",
                        page,
                        "OCR text and PDF text overlap geometrically but do not match.",
                        {"ocr_line_id": ocr_line.ocr_line_id, "pdf_span_id": overlapping_span.span_id},
                    )
                    page_conflicts.append(issue)
                elif _is_important(ocr_line.text):
                    issue = _issue(
                        issues,
                        "ocr_important_text_unmatched",
                        "medium",
                        page,
                        "Important OCR text did not match any PDF text span.",
                        {"ocr_line_id": ocr_line.ocr_line_id, "text": ocr_line.text},
                    )
                    page_conflicts.append(issue)

            if ocr_line.confidence < low_confidence_threshold:
                issue = _issue(
                    issues,
                    "ocr_low_confidence",
                    "medium" if _is_important(ocr_line.text) else "low",
                    page,
                    "OCR line confidence is below threshold.",
                    {
                        "ocr_line_id": ocr_line.ocr_line_id,
                        "confidence": ocr_line.confidence,
                        "threshold": low_confidence_threshold,
                    },
                )
                page_conflicts.append(issue)

        if page_ocr_lines:
            for pdf_span in page_pdf_spans:
                if pdf_span.span_id in matched_pdf_ids:
                    continue
                if _is_important(pdf_span.text):
                    issue = _issue(
                        issues,
                        "pdf_important_text_unconfirmed_by_ocr",
                        "low",
                        page,
                        "Important PDF text was not confirmed by OCR.",
                        {"pdf_span_id": pdf_span.span_id, "text": pdf_span.text},
                    )
                    page_conflicts.append(issue)

        matched_ocr_line_count += len(matched_ocr_ids)
        page_reports.append(
            {
                "page": page,
                "pdf_text_span_count": len(page_pdf_spans),
                "ocr_line_count": len(page_ocr_lines),
                "matched_ocr_line_count": len(matched_ocr_ids),
                "unmatched_ocr_line_count": len(page_ocr_lines) - len(matched_ocr_ids),
                "unconfirmed_important_pdf_span_count": sum(
                    1 for pdf_span in page_pdf_spans if pdf_span.span_id not in matched_pdf_ids and _is_important(pdf_span.text)
                )
                if page_ocr_lines
                else 0,
                "issue_count": len(page_conflicts),
                "issues": page_conflicts,
            }
        )

    blocking_issue_count = sum(1 for issue in issues if issue["severity"] in {"high", "medium"})
    return {
        "status": "review_required" if blocking_issue_count else "pass",
        "pdf_text_span_count": len(pdf_spans),
        "ocr_line_count": len(ocr_lines),
        "matched_ocr_line_count": matched_ocr_line_count,
        "match_rate": round(matched_ocr_line_count / len(ocr_lines), 4) if ocr_lines else 0.0,
        "issue_count": len(issues),
        "blocking_issue_count": blocking_issue_count,
        "issues": issues,
        "page_reports": page_reports,
    }


def _group_by_page(items: list[Any]) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {}
    for item in items:
        grouped.setdefault(int(item.page), []).append(item)
    return grouped


def _best_text_match(text: str, spans: list[TextSpan]) -> TextSpan | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for span in spans:
        span_normalized = _normalize_text(span.text)
        if not span_normalized:
            continue
        if normalized == span_normalized or normalized in span_normalized or span_normalized in normalized:
            return span
    return None


def _best_bbox_overlap(bbox: BBoxPdf | None, spans: list[TextSpan]) -> TextSpan | None:
    if bbox is None:
        return None
    candidates = [
        (span, _overlap_ratio(bbox, span.bbox_pdf))
        for span in spans
        if span.bbox_pdf is not None
    ]
    if not candidates:
        return None
    span, ratio = max(candidates, key=lambda item: item[1])
    return span if ratio >= 0.20 else None


def _overlap_ratio(left: BBoxPdf, right: BBoxPdf | None) -> float:
    if right is None:
        return 0.0
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    overlap_width = max(x2 - x1, 0.0)
    overlap_height = max(y2 - y1, 0.0)
    overlap_area = overlap_width * overlap_height
    left_area = max(left.width * left.height, 1.0)
    right_area = max(right.width * right.height, 1.0)
    return overlap_area / min(left_area, right_area)


def _normalize_text(text: str) -> str:
    return re.sub(r"[\s，,。；;：:（）()\[\]【】%-]+", "", text).lower()


def _is_important(text: str) -> bool:
    return bool(IMPORTANT_LABEL_RE.search(text))


def _issue(
    issues: list[dict[str, Any]],
    issue_type: str,
    severity: str,
    page: int,
    message: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    issue = {
        "issue_id": stable_id("source_consistency_issue", len(issues) + 1),
        "issue_type": issue_type,
        "severity": severity,
        "page": page,
        "message": message,
        "detail": detail,
    }
    issues.append(issue)
    return issue
