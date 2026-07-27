from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import sha256_file


def render_page_images(pdf_path: Path, output_dir: Path | None, dpi: int = 144) -> dict[str, Any]:
    if output_dir is None:
        return {
            "status": "not_rendered",
            "reason": "output_dir_not_provided",
            "dpi": dpi,
            "pages": [],
        }

    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:
        return {
            "status": "failed",
            "reason": "missing_dependency:pypdfium2",
            "error": exc.__class__.__name__,
            "dpi": dpi,
            "pages": [],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:  # pragma: no cover - depends on malformed PDFs/render backend
        return {
            "status": "failed",
            "reason": "render_failed",
            "error": exc.__class__.__name__,
            "dpi": dpi,
            "pages": pages,
            "failed_pages": failed_pages,
        }

    try:
        for page_index in range(len(document)):
            page_number = page_index + 1
            image_path = output_dir / f"page_{page_number:03d}.png"
            try:
                page = document[page_index]
                bitmap = page.render(scale=dpi / 72)
                image = bitmap.to_pil()
                image.save(image_path)
                pages.append(
                    {
                        "page": page_number,
                        "path": str(image_path),
                        "format": "png",
                        "dpi": dpi,
                        "width_px": image.width,
                        "height_px": image.height,
                        "sha256": sha256_file(image_path),
                        "render_status": "rendered",
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on page renderer internals
                failed_page = {
                    "page": page_number,
                    "render_status": "failed",
                    "reason": "render_failed",
                    "error": exc.__class__.__name__,
                }
                pages.append(failed_page)
                failed_pages.append(failed_page)
    finally:
        try:
            document.close()
        except Exception:
            pass

    if failed_pages and any(page.get("render_status") == "rendered" for page in pages):
        status = "partial_failed"
        reason = "page_render_partial_failed"
    elif failed_pages:
        status = "failed"
        reason = "render_failed"
    else:
        status = "rendered" if pages else "empty"
        reason = None

    return {
        "status": status,
        "reason": reason,
        "dpi": dpi,
        "page_count": len(pages),
        "rendered_page_count": sum(1 for page in pages if page.get("render_status") == "rendered"),
        "failed_page_count": len(failed_pages),
        "pages": pages,
        "failed_pages": failed_pages,
    }
