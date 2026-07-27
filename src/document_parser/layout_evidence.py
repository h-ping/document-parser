from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .layout_candidates import build_candidate_table_layers, build_layout_candidates, validate_layout_candidates
from .models import TextSpan
from .pdf import PdfPerception
from .pdf_atoms import PdfCharacterAtomRead, read_pdf_character_atoms_with_report


LEGACY_LAYOUT_MODE = "legacy"
ENHANCED_LAYOUT_MODE = "char_atoms_high_recall"


class LayoutEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayoutEvidence:
    canonical_pdf_spans: list[TextSpan]
    character_atoms: list[TextSpan]
    candidate_table_layers: dict[str, Any]
    layout_candidates: dict[str, Any]
    layout_quality_report: dict[str, Any]
    mode: str
    fallback_used: bool
    failure_reason: str | None
    dropped_control_char_count: int = 0


def build_layout_evidence(
    input_pdf: Path,
    perception: PdfPerception,
    mode: str,
    *,
    atom_reader: Callable[[Path], PdfCharacterAtomRead] = read_pdf_character_atoms_with_report,
) -> LayoutEvidence:
    if mode == LEGACY_LAYOUT_MODE:
        return LayoutEvidence(
            canonical_pdf_spans=perception.text_spans,
            character_atoms=[],
            candidate_table_layers={"parsers": [], "tables": [], "parser_issues": [], "candidate_only": True},
            layout_candidates=_disabled_candidates(),
            layout_quality_report={
                "report_version": "layout_quality_v0.1",
                "status": "disabled",
                "mode": mode,
                "source_span_coverage_rate": 0.0,
                "pdf_character_atom_count": 0,
                "dropped_control_char_count": 0,
                "layout_candidate_count": 0,
                "nutrition_layout_candidate_count": 0,
                "producer_layout_candidate_count": 0,
                "layout_boundary_issue_count": 0,
                "cross_page_candidate_count": 0,
                "fallback_used": False,
                "issues": [],
            },
            mode=mode,
            fallback_used=False,
            failure_reason=None,
        )
    if mode != ENHANCED_LAYOUT_MODE:
        raise LayoutEvidenceError(f"Unsupported layout mode: {mode}")
    if not perception.text_layer_available:
        raise LayoutEvidenceError("Enhanced layout mode requires a PDF character layer.")

    atom_read = atom_reader(input_pdf)
    if not atom_read.atoms:
        raise LayoutEvidenceError("Enhanced layout mode produced no character atoms; legacy fallback is disabled.")
    candidates = build_layout_candidates(atom_read.atoms, perception.pages)
    quality = validate_layout_candidates(candidates, atom_read.atoms)
    quality.update(
        {
            "mode": mode,
            "pdf_character_atom_count": len(atom_read.atoms),
            "dropped_control_char_count": atom_read.dropped_control_char_count,
            "layout_candidate_count": len(candidates.get("table_candidates", [])),
            "nutrition_layout_candidate_count": sum(1 for item in candidates.get("table_candidates", []) if item.get("table_type") == "nutrition_facts"),
            "producer_layout_candidate_count": sum(1 for item in candidates.get("table_candidates", []) if item.get("table_type") == "producer_info_repeated_rows"),
            "cross_page_candidate_count": candidates.get("cross_page_candidate_count", 0),
            "fallback_used": False,
        }
    )
    if quality["status"] == "fail":
        raise LayoutEvidenceError("Enhanced layout evidence quality failed; legacy fallback is disabled.")
    return LayoutEvidence(
        canonical_pdf_spans=atom_read.atoms,
        character_atoms=atom_read.atoms,
        candidate_table_layers=build_candidate_table_layers(candidates),
        layout_candidates=candidates,
        layout_quality_report=quality,
        mode=mode,
        fallback_used=False,
        failure_reason=None,
        dropped_control_char_count=atom_read.dropped_control_char_count,
    )


def _disabled_candidates() -> dict[str, Any]:
    return {
        "artifact_version": "layout_candidates_v0.1",
        "status": "disabled",
        "source_nodes": [],
        "table_candidates": [],
        "reading_order_candidates": [],
        "side_marker_candidates": [],
        "quality_issues": [],
        "cross_page_candidate_count": 0,
    }
