import unittest
from pathlib import Path

from document_parser.layout_evidence import LayoutEvidenceError, build_layout_evidence
from document_parser.models import BBoxNormalized, BBoxPdf, PageInfo, TextSpan
from document_parser.pdf import PdfPerception
from document_parser.pdf_atoms import PdfCharacterAtomRead


def _span(span_id: str, source: str, text: str = "营养成分表") -> TextSpan:
    bbox = BBoxPdf(x=10, y=10, width=80, height=10, page_width=600, page_height=800)
    return TextSpan(
        span_id=span_id,
        page=1,
        text=text,
        source=source,
        bbox_pdf=bbox,
        bbox_normalized=BBoxNormalized(10 / 600, 10 / 800, 90 / 600, 20 / 800),
    )


class LayoutEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.perception = PdfPerception(
            pages=[PageInfo(page=1, width=600, height=800)],
            text_spans=[_span("legacy", "pdf_text")],
            text_layer_available=True,
            warnings=[],
        )

    def test_legacy_keeps_pdf_spans_and_writes_disabled_artifacts(self) -> None:
        evidence = build_layout_evidence(Path("unused.pdf"), self.perception, "legacy")

        self.assertEqual([span.span_id for span in evidence.canonical_pdf_spans], ["legacy"])
        self.assertEqual(evidence.character_atoms, [])
        self.assertEqual(evidence.layout_quality_report["status"], "disabled")

    def test_enhanced_replaces_legacy_spans_with_atoms(self) -> None:
        atom = _span("atom", "pdf_char_atom")

        evidence = build_layout_evidence(
            Path("unused.pdf"),
            self.perception,
            "char_atoms_high_recall",
            atom_reader=lambda _path: PdfCharacterAtomRead([atom], 1, 0),
        )

        self.assertEqual([span.span_id for span in evidence.canonical_pdf_spans], ["atom"])
        self.assertNotIn("legacy", {span.span_id for span in evidence.canonical_pdf_spans})
        self.assertEqual(evidence.layout_quality_report["status"], "review_required")

    def test_enhanced_fails_closed_without_character_atoms(self) -> None:
        with self.assertRaises(LayoutEvidenceError):
            build_layout_evidence(
                Path("unused.pdf"),
                self.perception,
                "char_atoms_high_recall",
                atom_reader=lambda _path: PdfCharacterAtomRead([], 0, 0),
            )

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(LayoutEvidenceError):
            build_layout_evidence(Path("unused.pdf"), self.perception, "unknown")


if __name__ == "__main__":
    unittest.main()
