from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import struct
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF text-layer span/bbox HTML report from perception.json.")
    parser.add_argument("--perception", type=Path, required=True, help="Path to perception.json.")
    parser.add_argument("--page-images-dir", type=Path, required=True, help="Directory containing rendered page_001.png images.")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path.")
    args = parser.parse_args()

    data = json.loads(args.perception.read_text(encoding="utf-8"))
    pages = _page_images(args.page_images_dir)
    spans = [span for span in data.get("text_spans", []) if isinstance(span, dict)]
    page_boxes = _page_boxes(spans)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_html(data, pages, spans, page_boxes), encoding="utf-8")


def _page_images(page_images_dir: Path) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for path in sorted(page_images_dir.glob("page_*.png")):
        page = _page_number(path)
        if page is None:
            continue
        width, height = _png_size(path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        pages[page] = {
            "width": width,
            "height": height,
            "src": f"data:{media_type};base64,{encoded}",
        }
    return pages


def _page_number(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return None


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return (1, 1)


def _page_boxes(spans: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    pages: dict[int, list[dict[str, Any]]] = {}
    for span in spans:
        bbox = _normalized_bbox(span)
        if bbox is None:
            continue
        page = int(span.get("page") or 1)
        pages.setdefault(page, []).append({**span, **bbox})
    return pages


def _normalized_bbox(span: dict[str, Any]) -> dict[str, float] | None:
    normalized = span.get("bbox_normalized")
    if isinstance(normalized, dict):
        try:
            return {
                "x1": float(normalized["x1"]),
                "y1": float(normalized["y1"]),
                "x2": float(normalized["x2"]),
                "y2": float(normalized["y2"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    bbox = span.get("bbox_pdf")
    if not isinstance(bbox, dict):
        return None
    try:
        page_width = float(bbox["page_width"])
        page_height = float(bbox["page_height"])
        x1 = float(bbox["x"]) / page_width
        y1 = float(bbox["y"]) / page_height
        x2 = (float(bbox["x"]) + float(bbox["width"])) / page_width
        y2 = (float(bbox["y"]) + float(bbox["height"])) / page_height
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _render_html(
    data: dict[str, Any],
    pages: dict[int, dict[str, Any]],
    spans: list[dict[str, Any]],
    page_boxes: dict[int, list[dict[str, Any]]],
) -> str:
    text_layer_available = data.get("text_layer_available")
    bbox_count = sum(1 for span in spans if _normalized_bbox(span))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDF Text Layer Span BBox Report</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #fff;
      --text: #1f2933;
      --muted: #667085;
      --line: #d0d7de;
      --accent: #0b6bcb;
      --active: #f79009;
      --missing: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.96);
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 8px; font-size: 18px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: #fff;
      font-size: 12px;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(380px, .86fr) minmax(560px, 1.35fr);
      gap: 14px;
      padding: 14px;
      align-items: start;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
    }}
    input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      font: inherit;
    }}
    .span-list {{
      max-height: calc(100vh - 112px);
      overflow: auto;
    }}
    .span-row {{
      display: grid;
      grid-template-columns: 78px 1fr 74px;
      gap: 8px;
      padding: 8px 10px;
      border-bottom: 1px solid #eaedf0;
      cursor: pointer;
    }}
    .span-row:hover,
    .span-row.active {{ background: #eef6ff; }}
    .span-id {{ font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #344054; }}
    .span-text {{ word-break: break-word; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .bbox-missing {{ color: var(--missing); font-weight: 700; }}
    .doc-view {{
      padding: 10px;
      max-height: calc(100vh - 96px);
      overflow: auto;
    }}
    .page-title {{ margin: 0 0 8px; font-weight: 700; color: #344054; }}
    .page {{
      position: relative;
      margin: 0 auto 18px;
      border: 1px solid var(--line);
      background: #fff;
      width: min(100%, 900px);
    }}
    .page img {{ display: block; width: 100%; height: auto; }}
    .box {{
      position: absolute;
      border: 2px solid var(--accent);
      background: rgba(11,107,203,.12);
      min-width: 5px;
      min-height: 5px;
      pointer-events: auto;
    }}
    .box.active {{
      border-color: var(--active);
      background: rgba(247,144,9,.24);
      box-shadow: 0 0 0 2px rgba(247,144,9,.28);
      z-index: 5;
    }}
    .box-label {{
      position: absolute;
      left: -2px;
      top: -20px;
      display: none;
      max-width: 360px;
      padding: 2px 5px;
      border-radius: 4px;
      color: #fff;
      background: rgba(17,24,39,.9);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 12px;
    }}
    .box:hover .box-label,
    .box.active .box-label {{ display: block; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      .span-list, .doc-view {{ max-height: none; }}
      header {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PDF Text Layer Span BBox Report</h1>
    <div class="summary">
      <span class="metric">text_layer_available: {_e(text_layer_available)}</span>
      <span class="metric">pages: {_e(len(data.get("pages", [])))}</span>
      <span class="metric">pdf_spans: {_e(len(spans))}</span>
      <span class="metric">spans_with_bbox: {_e(bbox_count)}</span>
      <span class="metric">warnings: {_e(len(data.get("warnings", [])))}</span>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="toolbar">
        <input id="filter" placeholder="Filter span id or text">
      </div>
      <div class="span-list">{_render_span_rows(spans)}</div>
    </section>
    <section class="panel">
      <div class="doc-view">{_render_pages(pages, page_boxes)}</div>
    </section>
  </main>
  <script>
    const rows = Array.from(document.querySelectorAll('.span-row'));
    const boxes = Array.from(document.querySelectorAll('.box'));
    function setActive(spanId) {{
      rows.forEach(row => row.classList.toggle('active', row.dataset.spanId === spanId));
      boxes.forEach(box => box.classList.toggle('active', box.dataset.spanId === spanId));
      const firstBox = boxes.find(box => box.dataset.spanId === spanId);
      if (firstBox) firstBox.scrollIntoView({{block: 'center', inline: 'center', behavior: 'smooth'}});
    }}
    rows.forEach(row => {{
      row.addEventListener('mouseenter', () => setActive(row.dataset.spanId));
      row.addEventListener('click', () => setActive(row.dataset.spanId));
    }});
    boxes.forEach(box => {{
      box.addEventListener('mouseenter', () => setActive(box.dataset.spanId));
      box.addEventListener('click', () => setActive(box.dataset.spanId));
    }});
    document.getElementById('filter').addEventListener('input', event => {{
      const q = event.target.value.trim().toLowerCase();
      rows.forEach(row => {{
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""


def _render_span_rows(spans: list[dict[str, Any]]) -> str:
    rows = []
    for span in spans:
        span_id = str(span.get("span_id") or "")
        bbox = _normalized_bbox(span)
        bbox_text = "bbox" if bbox else "missing"
        bbox_class = "" if bbox else " bbox-missing"
        rows.append(
            f"""
            <div class="span-row" data-span-id="{_e(span_id)}">
              <div>
                <div class="span-id">{_e(span_id)}</div>
                <div class="muted">p{_e(span.get('page'))}</div>
              </div>
              <div class="span-text">{_e(span.get('text'))}</div>
              <div class="muted{bbox_class}">{_e(bbox_text)}</div>
            </div>
            """
        )
    return "\n".join(rows)


def _render_pages(pages: dict[int, dict[str, Any]], page_boxes: dict[int, list[dict[str, Any]]]) -> str:
    html_parts = []
    for page_number, image in sorted(pages.items()):
        boxes = page_boxes.get(page_number, [])
        html_parts.append(
            f"""
            <div class="page-title">Page {_e(page_number)} | spans with bbox: {_e(len(boxes))}</div>
            <div class="page" style="aspect-ratio: {image['width']} / {image['height']};">
              <img src="{image['src']}" alt="page {page_number}">
              {_render_boxes(boxes)}
            </div>
            """
        )
    return "\n".join(html_parts) if html_parts else '<div class="muted">No page images found.</div>'


def _render_boxes(boxes: list[dict[str, Any]]) -> str:
    parts = []
    for box in boxes:
        x = box["x1"] * 100
        y = box["y1"] * 100
        width = max((box["x2"] - box["x1"]) * 100, 0.2)
        height = max((box["y2"] - box["y1"]) * 100, 0.2)
        span_id = str(box.get("span_id") or "")
        title = f"{span_id}: {box.get('text') or ''}"
        parts.append(
            f"""
            <div class="box" data-span-id="{_e(span_id)}" title="{_e(title)}"
                 style="left:{x:.4f}%; top:{y:.4f}%; width:{width:.4f}%; height:{height:.4f}%;">
              <div class="box-label">{_e(title)}</div>
            </div>
            """
        )
    return "\n".join(parts)


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


if __name__ == "__main__":
    main()
