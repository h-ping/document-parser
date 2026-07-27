from __future__ import annotations

import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import BBoxPdf, PageInfo, TextSpan
from .pdf import PdfDependencyError, normalized_bbox_from_pdf
from .utils import stable_id


_BASELINE_TOLERANCE_PT = 1.25


@dataclass(frozen=True)
class PdfCharacterAtomRead:
    atoms: list[TextSpan]
    source_char_count: int
    dropped_control_char_count: int


def read_pdf_character_atoms(path: Path, pages: set[int] | None = None) -> list[TextSpan]:
    return read_pdf_character_atoms_with_report(path, pages).atoms


def read_pdf_character_atoms_with_report(path: Path, pages: set[int] | None = None) -> PdfCharacterAtomRead:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise PdfDependencyError("Missing dependency: pdfplumber. Install project dependencies first.") from exc

    atoms: list[TextSpan] = []
    source_char_count = 0
    dropped_control_char_count = 0
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if pages is not None and page_number not in pages:
                continue
            page_info = PageInfo(page=page_number, width=float(page.width), height=float(page.height))
            page_read = character_atoms_from_chars_with_report(page.chars, page_info)
            atoms.extend(page_read.atoms)
            source_char_count += page_read.source_char_count
            dropped_control_char_count += page_read.dropped_control_char_count
    return PdfCharacterAtomRead(atoms, source_char_count, dropped_control_char_count)


def character_atoms_from_chars(chars: list[dict[str, Any]], page: PageInfo) -> list[TextSpan]:
    return character_atoms_from_chars_with_report(chars, page).atoms


def character_atoms_from_chars_with_report(chars: list[dict[str, Any]], page: PageInfo) -> PdfCharacterAtomRead:
    visible_chars = [char for char in chars if _has_usable_geometry(char) and not _is_control_text(str(char.get("text", "")))]
    line_groups = _group_chars_by_baseline(visible_chars)
    raw_atoms: list[list[dict[str, Any]]] = []
    for line in line_groups:
        raw_atoms.extend(_split_line_into_atoms(line))

    spans: list[TextSpan] = []
    for index, atom_chars in enumerate(raw_atoms, start=1):
        text = "".join(str(char["text"]) for char in atom_chars)
        if not text:
            continue
        bbox = _bbox_from_chars(atom_chars, page)
        spans.append(
            TextSpan(
                span_id=stable_id(f"atom_p{page.page}", index),
                page=page.page,
                text=text,
                source="pdf_char_atom",
                bbox_pdf=bbox,
                bbox_normalized=normalized_bbox_from_pdf(bbox),
                confidence=1.0,
            )
        )
    dropped_control_char_count = sum(1 for char in chars if _is_control_text(str(char.get("text", ""))))
    return PdfCharacterAtomRead(spans, len(chars), dropped_control_char_count)


def _group_chars_by_baseline(chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    group_tops: list[float] = []
    for char in sorted(chars, key=lambda item: (_number(item, "top"), _number(item, "x0"))):
        top = _number(char, "top")
        group_index = next(
            (index for index, group_top in enumerate(group_tops) if abs(top - group_top) <= _BASELINE_TOLERANCE_PT),
            None,
        )
        if group_index is None:
            groups.append([char])
            group_tops.append(top)
            continue
        groups[group_index].append(char)
        group_tops[group_index] = statistics.fmean(_number(item, "top") for item in groups[group_index])
    return [sorted(group, key=lambda item: (_number(item, "x0"), _number(item, "x1"))) for group in groups]


def _split_line_into_atoms(chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    printable = [char for char in chars if str(char.get("text", "")).strip()]
    widths = [_number(char, "x1") - _number(char, "x0") for char in printable]
    positive_widths = [width for width in widths if width > 0]
    gap_threshold = max(4.0, statistics.median(positive_widths) * 1.25) if positive_widths else 4.0

    atoms: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for char in chars:
        text = str(char.get("text", ""))
        if not text.strip():
            if current:
                atoms.append(current)
                current = []
            previous = None
            continue
        gap = _number(char, "x0") - _number(previous, "x1") if previous is not None else 0.0
        if current and gap > gap_threshold:
            atoms.append(current)
            current = []
        current.append(char)
        previous = char
    if current:
        atoms.append(current)
    return atoms


def _bbox_from_chars(chars: list[dict[str, Any]], page: PageInfo) -> BBoxPdf:
    x1 = min(_number(char, "x0") for char in chars)
    y1 = min(_number(char, "top") for char in chars)
    x2 = max(_number(char, "x1") for char in chars)
    y2 = max(_number(char, "bottom") for char in chars)
    return BBoxPdf(
        x=round(x1, 3),
        y=round(y1, 3),
        width=round(x2 - x1, 3),
        height=round(y2 - y1, 3),
        page_width=page.width,
        page_height=page.height,
    )


def _has_usable_geometry(char: dict[str, Any]) -> bool:
    return bool(str(char.get("text", ""))) and all(key in char for key in ("x0", "x1", "top", "bottom"))


def _is_control_text(text: str) -> bool:
    return bool(text) and all(unicodedata.category(char).startswith("C") for char in text)


def _number(value: dict[str, Any], key: str) -> float:
    return float(value[key])
