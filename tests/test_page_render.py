import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from document_parser.page_render import render_page_images
from document_parser.pipeline import _risks_from_page_images


ROOT = Path(__file__).resolve().parents[1]


class _FakePdfDocument:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        if index == 1:
            raise RuntimeError("page render failed")
        return _FakePage()

    def close(self) -> None:
        pass


class _FakePage:
    def render(self, scale: float):
        del scale
        return _FakeBitmap()


class _FakeBitmap:
    def to_pil(self):
        return _FakeImage()


class _FakeImage:
    width = 100
    height = 200

    def save(self, path: Path) -> None:
        path.write_bytes(b"fake png")


class PageRenderTests(unittest.TestCase):
    def test_render_page_images_writes_png_metadata(self) -> None:
        sample_pdf = ROOT / "test-documents" / "识别文字.pdf"
        if not sample_pdf.exists():
            self.skipTest("sample PDF not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "page_images"
            manifest = render_page_images(sample_pdf, output_dir, dpi=72)

            self.assertEqual(manifest["status"], "rendered")
            self.assertGreaterEqual(manifest["page_count"], 1)
            first_page = manifest["pages"][0]
            self.assertTrue(Path(first_page["path"]).exists())
            self.assertEqual(first_page["format"], "png")
            self.assertEqual(first_page["render_status"], "rendered")
            self.assertTrue(first_page["sha256"].startswith("sha256:"))
            self.assertGreater(first_page["width_px"], 0)
            self.assertGreater(first_page["height_px"], 0)

    def test_render_page_images_can_be_skipped_without_output_dir(self) -> None:
        manifest = render_page_images(Path("sample.pdf"), None)

        self.assertEqual(manifest["status"], "not_rendered")
        self.assertEqual(manifest["reason"], "output_dir_not_provided")
        self.assertEqual(manifest["pages"], [])

    def test_failed_page_rendering_creates_risk(self) -> None:
        risks = _risks_from_page_images({"status": "failed", "reason": "render_failed"})

        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].risk_type, "page_image_render_failed")
        self.assertEqual(risks[0].risk_level, "medium")

    def test_page_render_continues_after_one_page_fails(self) -> None:
        fake_pdfium = types.SimpleNamespace(PdfDocument=lambda path: _FakePdfDocument())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "page_images"
            with patch.dict(sys.modules, {"pypdfium2": fake_pdfium}):
                manifest = render_page_images(Path("sample.pdf"), output_dir, dpi=72)

            self.assertEqual(manifest["status"], "partial_failed")
            self.assertEqual(manifest["page_count"], 3)
            self.assertEqual(manifest["rendered_page_count"], 2)
            self.assertEqual(manifest["failed_page_count"], 1)
            self.assertEqual(manifest["failed_pages"][0]["page"], 2)
            self.assertTrue((output_dir / "page_001.png").exists())
            self.assertTrue((output_dir / "page_003.png").exists())


if __name__ == "__main__":
    unittest.main()
