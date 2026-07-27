from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import BBoxPdf, BBoxNormalized, PageInfo, TextSpan
from .utils import stable_id


class PdfDependencyError(RuntimeError):
    pass


class PdfReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PdfPerception:
    pages: list[PageInfo]
    text_spans: list[TextSpan]
    text_layer_available: bool
    warnings: list[str]


class PdfPerceptionReader:
    def read(self, path: Path) -> PdfPerception:
        try:
            return self._read_with_pdfplumber(path)
        except Exception as pdfplumber_exc:
            try:
                fallback = self._read_with_pypdf(path)
            except Exception as pypdf_exc:
                raise PdfReadError(
                    f"Failed to read PDF {path}: "
                    f"pdfplumber failed with {pdfplumber_exc.__class__.__name__}: {pdfplumber_exc}; "
                    f"pypdf failed with {pypdf_exc.__class__.__name__}: {pypdf_exc}"
                ) from pypdf_exc
            return PdfPerception(
                pages=fallback.pages,
                text_spans=fallback.text_spans,
                text_layer_available=fallback.text_layer_available,
                warnings=[*fallback.warnings, f"pdfplumber_text_extract_failed:{pdfplumber_exc.__class__.__name__}"],
            )

    def _read_with_pdfplumber(self, path: Path) -> PdfPerception:
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:
            raise PdfDependencyError("Missing dependency: pdfplumber. Install project dependencies first.") from exc

        pages: list[PageInfo] = []
        spans: list[TextSpan] = []
        warnings: list[str] = []

        with pdfplumber.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                width = float(page.width)
                height = float(page.height)
                page_info = PageInfo(page=page_index, width=width, height=height)
                pages.append(page_info)
                try:
                    words = page.extract_words(
                        keep_blank_chars=False,
                        use_text_flow=True,
                        extra_attrs=[],
                    )
                except Exception as exc:
                    warnings.append(f"page_{page_index}_pdfplumber_words_failed:{exc.__class__.__name__}")
                    words = []

                line_index = 0
                for line_words in _group_words_into_lines(words):
                    text = " ".join(word["text"] for word in line_words).strip()
                    if not text:
                        continue
                    line_index += 1
                    bbox_pdf = _bbox_from_pdfplumber_words(line_words, page_info)
                    spans.append(
                        TextSpan(
                            span_id=stable_id(f"span_p{page_index}", line_index),
                            page=page_index,
                            text=text,
                            source="pdf_text",
                            bbox_pdf=bbox_pdf,
                            bbox_normalized=normalized_bbox_from_pdf(bbox_pdf),
                            confidence=1.0,
                        )
                    )

        return PdfPerception(
            pages=pages,
            text_spans=spans,
            text_layer_available=bool(spans),
            warnings=warnings,
        )

    def _read_with_pypdf(self, path: Path) -> PdfPerception:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise PdfDependencyError("Missing dependency: pypdf. Install project dependencies first.") from exc

        reader = PdfReader(str(path))
        pages: list[PageInfo] = []
        spans: list[TextSpan] = []
        warnings: list[str] = []

        for page_index, page in enumerate(reader.pages, start=1):
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            pages.append(PageInfo(page=page_index, width=width, height=height))

            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - depends on malformed PDFs
                warnings.append(f"page_{page_index}_text_extract_failed:{exc.__class__.__name__}")
                text = ""

            line_index = 0
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                line_index += 1
                # pypdf text extraction does not reliably expose bbox through this
                # path, so mark line geometry as missing instead of guessing.
                spans.append(
                    TextSpan(
                        span_id=stable_id(f"span_p{page_index}", line_index),
                        page=page_index,
                        text=line,
                        source="pdf_text",
                        confidence=1.0,
                    )
                )

        return PdfPerception(
            pages=pages,
            text_spans=spans,
            text_layer_available=bool(spans),
            warnings=warnings,
        )


def normalized_bbox_from_pdf(bbox: BBoxPdf) -> BBoxNormalized:
    return BBoxNormalized(
        x1=round(bbox.x / bbox.page_width, 6),
        y1=round(bbox.y / bbox.page_height, 6),
        x2=round((bbox.x + bbox.width) / bbox.page_width, 6),
        y2=round((bbox.y + bbox.height) / bbox.page_height, 6),
    )


def bbox_from_points(points: list[list[float]], page: PageInfo, source_width: float | None = None, source_height: float | None = None) -> tuple[BBoxPdf, BBoxNormalized]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if source_width and source_height:
        x = min_x / source_width * page.width
        y = min_y / source_height * page.height
        width = (max_x - min_x) / source_width * page.width
        height = (max_y - min_y) / source_height * page.height
    else:
        x = min_x
        y = min_y
        width = max_x - min_x
        height = max_y - min_y

    x2 = x + width
    y2 = y + height
    x = _clamp(x, 0.0, page.width)
    y = _clamp(y, 0.0, page.height)
    x2 = _clamp(x2, 0.0, page.width)
    y2 = _clamp(y2, 0.0, page.height)
    width = max(0.0, x2 - x)
    height = max(0.0, y2 - y)

    bbox = BBoxPdf(
        x=round(x, 3),
        y=round(y, 3),
        width=round(width, 3),
        height=round(height, 3),
        page_width=page.width,
        page_height=page.height,
    )
    return bbox, normalized_bbox_from_pdf(bbox)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _group_words_into_lines(words: list[dict]) -> list[list[dict]]:
    """Group pdfplumber words into visual text spans.

    The historical function name says "lines", but mixed label artwork often
    contains table cells, side labels, and multi-column fields on the same
    visual row. Returning whole rows here pollutes source evidence before VDG
    or the extraction agents can reason about field boundaries.
    """

    sorted_words = sorted(
        (word for word in words if word.get("text")),
        key=lambda word: (float(word.get("top", 0.0)), float(word.get("x0", 0.0))),
    )
    line_clusters: list[list[dict]] = []
    for word in sorted_words:
        best_index: int | None = None
        best_distance: float | None = None
        for index, line in enumerate(line_clusters):
            if not _same_visual_line(line, word):
                continue
            distance = abs(_word_center_y(word) - _line_center_y(line))
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            line_clusters.append([word])
        else:
            line_clusters[best_index].append(word)

    line_clusters.sort(key=lambda line: (_line_center_y(line), min(float(word.get("x0", 0.0)) for word in line)))

    visual_spans: list[list[dict]] = []
    for line in line_clusters:
        current_span: list[dict] = []
        for word in sorted(line, key=lambda item: float(item.get("x0", 0.0))):
            if not current_span:
                current_span.append(word)
                continue
            if _same_visual_span(current_span[-1], word):
                current_span.append(word)
            else:
                visual_spans.append(current_span)
                current_span = [word]
        if current_span:
            visual_spans.append(current_span)
    return visual_spans


def _same_visual_line(line: list[dict], word: dict) -> bool:
    line_top = min(float(item.get("top", 0.0)) for item in line)
    line_bottom = max(float(item.get("bottom", item.get("top", 0.0))) for item in line)
    word_top = float(word.get("top", 0.0))
    word_bottom = float(word.get("bottom", word_top))
    overlap = max(0.0, min(line_bottom, word_bottom) - max(line_top, word_top))
    min_height = max(0.1, min(line_bottom - line_top, word_bottom - word_top))
    if overlap / min_height >= 0.55:
        return True

    center_distance = abs(_word_center_y(word) - _line_center_y(line))
    tolerance = max(2.0, min(_line_height(line), _word_height(word)) * 0.45)
    return center_distance <= tolerance


def _same_visual_span(previous: dict, word: dict) -> bool:
    gap = float(word.get("x0", 0.0)) - float(previous.get("x1", previous.get("x0", 0.0)))
    if gap <= 0:
        return True
    tolerance = max(4.0, min(_word_height(previous), _word_height(word)) * 0.75)
    return gap <= tolerance


def _line_center_y(line: list[dict]) -> float:
    return sum(_word_center_y(word) for word in line) / len(line)


def _word_center_y(word: dict) -> float:
    top = float(word.get("top", 0.0))
    return top + _word_height(word) / 2


def _line_height(line: list[dict]) -> float:
    return max(_word_height(word) for word in line)


def _word_height(word: dict) -> float:
    top = float(word.get("top", 0.0))
    bottom = float(word.get("bottom", top))
    return max(0.1, bottom - top)


def _bbox_from_pdfplumber_words(words: list[dict], page: PageInfo) -> BBoxPdf:
    x0 = min(float(word["x0"]) for word in words)
    top = min(float(word["top"]) for word in words)
    x1 = max(float(word["x1"]) for word in words)
    bottom = max(float(word["bottom"]) for word in words)
    return BBoxPdf(
        x=round(x0, 3),
        y=round(top, 3),
        width=round(x1 - x0, 3),
        height=round(bottom - top, 3),
        page_width=page.width,
        page_height=page.height,
    )
