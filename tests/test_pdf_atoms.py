import unittest

from document_parser.models import PageInfo
from document_parser.pdf_atoms import character_atoms_from_chars, character_atoms_from_chars_with_report


def _char(text: str, x0: float, x1: float, top: float, bottom: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom}


class PdfCharacterAtomTests(unittest.TestCase):
    def test_filters_control_fillers_and_splits_visually_separate_text(self) -> None:
        chars = [
            _char("产", 10, 20, 20, 30),
            _char("品", 20, 30, 20, 30),
            _char("类", 30, 40, 20, 30),
            _char("型", 40, 50, 20, 30),
            _char(":", 50, 55, 20, 30),
            _char("混", 55, 65, 20, 30),
            _char("合", 65, 75, 20, 30),
            _char("胶", 75, 85, 20, 30),
            _char("型", 85, 95, 20, 30),
            _char("\x00", 95, 160, 20, 30),
            _char("6", 120, 126, 20, 30),
            _char("6", 126, 132, 20, 30),
        ]

        atoms = character_atoms_from_chars(chars, PageInfo(page=1, width=200, height=100))

        self.assertEqual([atom.text for atom in atoms], ["产品类型:混合胶型", "66"])
        self.assertEqual(atoms[0].bbox_pdf.x, 10)
        self.assertEqual(atoms[0].bbox_pdf.width, 85)
        self.assertEqual(atoms[1].bbox_pdf.x, 120)
        self.assertEqual(atoms[1].bbox_pdf.width, 12)

        report = character_atoms_from_chars_with_report(chars, PageInfo(page=1, width=200, height=100))
        self.assertEqual(report.dropped_control_char_count, 1)
        self.assertEqual(report.source_char_count, 12)

    def test_keeps_nearby_baselines_as_separate_atoms(self) -> None:
        chars = [
            _char("上", 10, 20, 20.0, 30.0),
            _char("行", 20, 30, 20.0, 30.0),
            _char("下", 10, 20, 22.0, 32.0),
            _char("行", 20, 30, 22.0, 32.0),
        ]

        atoms = character_atoms_from_chars(chars, PageInfo(page=1, width=200, height=100))

        self.assertEqual([atom.text for atom in atoms], ["上行", "下行"])

    def test_whitespace_creates_an_atom_boundary(self) -> None:
        chars = [
            _char("1", 10, 16, 20, 30),
            _char("/", 16, 20, 20, 30),
            _char("3", 20, 26, 20, 30),
            _char("包", 26, 36, 20, 30),
            _char(" ", 36, 46, 20, 30),
            _char("≥", 46, 56, 20, 30),
            _char("2", 56, 62, 20, 30),
            _char(".", 62, 65, 20, 30),
            _char("5", 65, 71, 20, 30),
            _char("%", 71, 78, 20, 30),
            _char("的", 78, 88, 20, 30),
            _char("软", 88, 98, 20, 30),
            _char("糖", 98, 108, 20, 30),
        ]

        atoms = character_atoms_from_chars(chars, PageInfo(page=1, width=200, height=100))

        self.assertEqual([atom.text for atom in atoms], ["1/3包", "≥2.5%的软糖"])


if __name__ == "__main__":
    unittest.main()
