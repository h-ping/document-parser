from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None

from .config import RuntimeConfig
from .models import BBoxNormalized, BBoxPdf, OcrLine, PageInfo
from .utils import stable_id


class OcrError(RuntimeError):
    pass


class OcrClient:
    def recognize_pdf(self, pdf_path: Path, pages: list[PageInfo]) -> list[OcrLine]:
        raise NotImplementedError

    def recognize_image(self, image_path: Path, page: PageInfo) -> list[OcrLine]:
        raise NotImplementedError


class GLMOcrClient(OcrClient):
    def __init__(self, config: RuntimeConfig, timeout_seconds: int = 180) -> None:
        self._api_key = config.glm_ocr_api_key
        self._model = config.glm_ocr_model
        self._timeout_seconds = timeout_seconds

    def recognize_pdf(self, pdf_path: Path, pages: list[PageInfo]) -> list[OcrLine]:
        return normalize_glm_ocr_response(self._recognize_file(pdf_path), pages)

    def recognize_image(self, image_path: Path, page: PageInfo) -> list[OcrLine]:
        return normalize_glm_ocr_response(self._recognize_file(image_path), [page])

    def _recognize_file(self, path: Path) -> dict[str, Any]:
        if ZhipuAiClient is None:
            raise OcrError("GLM-OCR requires the zai-sdk package.")
        client = ZhipuAiClient(api_key=self._api_key)
        response = client.layout_parsing.create(
            model=self._model,
            file=base64.b64encode(path.read_bytes()).decode("ascii"),
            return_crop_images=False,
            need_layout_visualization=False,
            timeout=self._timeout_seconds,
        )
        return _response_to_dict(response)


class RecordedOcrClient(OcrClient):
    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path

    def recognize_pdf(self, pdf_path: Path, pages: list[PageInfo]) -> list[OcrLine]:
        del pdf_path
        body = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        return normalize_ppocrv6_any_response(body, pages)

    def recognize_image(self, image_path: Path, page: PageInfo) -> list[OcrLine]:
        del image_path
        body = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        return normalize_ppocrv6_any_response(body, [page])


def normalize_glm_ocr_response(body: dict[str, Any], pages: list[PageInfo]) -> list[OcrLine]:
    layout_pages = _glm_layout_pages(body)
    data_pages = _as_list(_as_dict(body.get("data_info")).get("pages"))
    lines: list[OcrLine] = []

    for page_zero_index, raw_page_details in enumerate(layout_pages):
        page_info = pages[min(page_zero_index, len(pages) - 1)] if pages else PageInfo(page=page_zero_index + 1, width=1, height=1)
        page_details = [detail for detail in _as_list(raw_page_details) if isinstance(detail, dict)]
        data_page = _as_dict(data_pages[page_zero_index]) if page_zero_index < len(data_pages) else {}
        page_line_index = 0
        for detail_index, detail in enumerate(page_details, start=1):
            detail_lines = _glm_detail_text_lines(str(detail.get("content") or ""))
            if not detail_lines:
                continue
            source_width = _first_float(detail, ["width"]) or _first_float(data_page, ["width"]) or page_info.width
            source_height = _first_float(detail, ["height"]) or _first_float(data_page, ["height"]) or page_info.height
            bbox_pdf, bbox_normalized = _glm_bbox(detail, page_info, source_width, source_height)
            block_id = stable_id(f"ocr_p{page_info.page}_block", detail_index)
            for text in detail_lines:
                page_line_index += 1
                lines.append(
                    OcrLine(
                        ocr_line_id=stable_id(f"ocr_p{page_info.page}", page_line_index),
                        page=page_info.page,
                        text=text,
                        confidence=1.0,
                        bbox_pdf=bbox_pdf,
                        bbox_normalized=bbox_normalized,
                        block_id=block_id,
                        tokens=[],
                        metadata={
                            "provider": "glm_ocr",
                            "line_index": page_line_index,
                            "detail_index": detail_index,
                            "detail_label": detail.get("label"),
                            "block_id": block_id,
                            "source_size": {
                                "width": source_width,
                                "height": source_height,
                            },
                        },
                    )
                )
    return lines


def normalize_ppocrv6_any_response(body: dict[str, Any], pages: list[PageInfo]) -> list[OcrLine]:
    if isinstance(body.get("result"), dict):
        return normalize_ppocrv6_response(body, pages)
    if isinstance(body.get("ocrResults"), list):
        return normalize_ppocrv6_response({"result": {"ocrResults": body["ocrResults"]}}, pages)
    return normalize_ppocrv6_response(body, pages)


def normalize_ppocrv6_response(body: dict[str, Any], pages: list[PageInfo]) -> list[OcrLine]:
    result = body.get("result") or {}
    ocr_results = result.get("ocrResults") or []
    lines: list[OcrLine] = []

    for page_zero_index, page_result in enumerate(ocr_results):
        page_info = pages[min(page_zero_index, len(pages) - 1)] if pages else PageInfo(page=page_zero_index + 1, width=1, height=1)
        pruned = page_result.get("prunedResult") or page_result.get("res") or page_result
        source_width, source_height = _extract_source_size(pruned)

        texts = _first_list(pruned, ["rec_texts", "recTexts", "texts", "text"])
        scores = _first_list(pruned, ["rec_scores", "recScores", "scores", "confidences"])
        boxes = _first_list(pruned, ["dt_polys", "dt_polygons", "rec_polys", "boxes", "polys"])

        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            texts = []

        for text_index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                continue
            score = _score_at(scores, text_index)
            box = _box_at(boxes, text_index)
            bbox_pdf = None
            bbox_normalized = None
            if box:
                bbox_pdf, bbox_normalized = _bbox_from_points(box, page_info, source_width, source_height)
                if bbox_pdf.width <= 0 or bbox_pdf.height <= 0:
                    bbox_pdf = None
                    bbox_normalized = None
            block_id = _block_id_at(pruned, text_index, page_info.page)
            lines.append(
                OcrLine(
                    ocr_line_id=stable_id(f"ocr_p{page_info.page}", text_index + 1),
                    page=page_info.page,
                    text=text.strip(),
                    confidence=score,
                    bbox_pdf=bbox_pdf,
                    bbox_normalized=bbox_normalized,
                    block_id=block_id,
                    tokens=_tokens_for_line(pruned, text_index, len(texts), page_info, source_width, source_height),
                    metadata={
                        "line_index": text_index + 1,
                        "block_id": block_id,
                        "source_size": {
                            "width": source_width,
                            "height": source_height,
                        },
                    },
                )
            )

    return lines


def normalize_ppocrv6_jsonl(text: str, pages: list[PageInfo]) -> list[OcrLine]:
    ocr_results: list[Any] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        body = json.loads(raw_line)
        result = body.get("result")
        if not isinstance(result, dict):
            continue
        line_results = result.get("ocrResults")
        if isinstance(line_results, list):
            ocr_results.extend(line_results)
    return normalize_ppocrv6_response({"result": {"ocrResults": ocr_results}}, pages)


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        value = response.model_dump()
        if isinstance(value, dict):
            return value
    if hasattr(response, "to_dict"):
        value = response.to_dict()
        if isinstance(value, dict):
            return value
    if hasattr(response, "dict"):
        value = response.dict()
        if isinstance(value, dict):
            return value
    raise OcrError("GLM-OCR response could not be converted to a JSON object.")


def _glm_layout_pages(body: dict[str, Any]) -> list[Any]:
    layout_details = body.get("layout_details")
    if not isinstance(layout_details, list):
        return []
    if layout_details and isinstance(layout_details[0], dict):
        return [layout_details]
    return layout_details


def _glm_detail_text_lines(content: str) -> list[str]:
    lines = []
    for raw_line in content.splitlines():
        text = raw_line.strip()
        if not text:
            continue
        if _is_markdown_separator_row(text):
            continue
        lines.append(_markdown_table_row_text(text))
    return lines or ([content.strip()] if content.strip() else [])


def _is_markdown_separator_row(text: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", text))


def _markdown_table_row_text(text: str) -> str:
    if not text.startswith("|") or "|" not in text.strip("|"):
        return text
    cells = [cell.strip() for cell in text.strip("|").split("|") if cell.strip()]
    return " ".join(cells) if cells else text


def _glm_bbox(
    detail: dict[str, Any],
    page_info: PageInfo,
    source_width: float | None,
    source_height: float | None,
) -> tuple[BBoxPdf | None, BBoxNormalized | None]:
    bbox = detail.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(item, (int, float)) for item in bbox):
        return None, None
    x1, y1, x2, y2 = [float(item) for item in bbox]
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    bbox_pdf, bbox_normalized = _bbox_from_points(points, page_info, source_width, source_height)
    if bbox_pdf.width <= 0 or bbox_pdf.height <= 0:
        return None, None
    return bbox_pdf, bbox_normalized


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_list(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return []


def _score_at(scores: Any, index: int) -> float:
    if isinstance(scores, list) and index < len(scores):
        try:
            return float(scores[index])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _box_at(boxes: Any, index: int) -> list[list[float]] | None:
    if not isinstance(boxes, list) or index >= len(boxes):
        return None
    return _box_from_value(boxes[index])


def _box_from_value(box: Any) -> list[list[float]] | None:
    if not isinstance(box, list):
        return None
    if len(box) == 4 and all(isinstance(item, (int, float)) for item in box):
        x, y, width, height = [float(item) for item in box]
        return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    if all(isinstance(point, list) and len(point) >= 2 for point in box):
        return [[float(point[0]), float(point[1])] for point in box]
    return None


def _block_id_at(data: dict[str, Any], line_index: int, page: int) -> str:
    block_ids = _first_list(data, ["block_ids", "blockIds", "rec_block_ids", "recBlockIds", "block_indexes", "blockIndexes"])
    if isinstance(block_ids, list) and line_index < len(block_ids) and block_ids[line_index] not in (None, ""):
        return str(block_ids[line_index])
    return stable_id(f"ocr_p{page}_block", 1)


def _tokens_for_line(
    data: dict[str, Any],
    line_index: int,
    line_count: int,
    page_info: PageInfo,
    source_width: float | None,
    source_height: float | None,
) -> list[dict[str, Any]]:
    raw_tokens = _first_list(data, ["rec_word_infos", "recWordInfos", "word_infos", "wordInfos", "tokens", "words"])
    if not isinstance(raw_tokens, list):
        return []

    candidates: list[Any] = []
    if line_index < len(raw_tokens) and isinstance(raw_tokens[line_index], list):
        candidates = raw_tokens[line_index]
    elif line_count == 1:
        candidates = raw_tokens
    else:
        line_index_base = _line_index_base(raw_tokens)
        candidates = [
            item
            for item in raw_tokens
            if isinstance(item, dict) and _matches_line_index(item, line_index, line_index_base)
        ]

    tokens = []
    for token_index, raw_token in enumerate(candidates, start=1):
        token = _normalize_token(
            raw_token,
            stable_id(f"ocr_p{page_info.page}_l{line_index + 1}_tok", token_index),
            page_info,
            source_width,
            source_height,
        )
        if token:
            tokens.append(token)
    return tokens


def _line_index_base(raw_tokens: list[Any]) -> int:
    values = []
    for item in raw_tokens:
        if not isinstance(item, dict):
            continue
        value = _line_index_value(item)
        if value is not None:
            values.append(value)
    return 0 if 0 in values else 1


def _matches_line_index(item: dict[str, Any], zero_based_line_index: int, line_index_base: int) -> bool:
    value = _line_index_value(item)
    if value is None:
        return False
    return value == zero_based_line_index + line_index_base


def _line_index_value(item: dict[str, Any]) -> int | None:
    for key in ("line_index", "lineIndex", "text_index", "textIndex", "rec_index", "recIndex"):
        if key not in item:
            continue
        try:
            return int(item[key])
        except (TypeError, ValueError):
            continue
    return None


def _normalize_token(
    raw_token: Any,
    token_id: str,
    page_info: PageInfo,
    source_width: float | None,
    source_height: float | None,
) -> dict[str, Any] | None:
    if isinstance(raw_token, str):
        text = raw_token.strip()
        if not text:
            return None
        return {
            "token_id": token_id,
            "page": page_info.page,
            "text": text,
            "bbox_status": "missing",
        }

    if not isinstance(raw_token, dict):
        return None

    text = _first_scalar(raw_token, ["text", "word", "value", "rec_text", "recText"])
    if not isinstance(text, str) or not text.strip():
        return None
    box = _first_scalar(raw_token, ["bbox", "box", "poly", "points", "dt_poly", "dtPoly"])
    bbox_pdf = None
    bbox_normalized = None
    normalized_box = _box_from_value(box)
    if normalized_box:
        bbox_pdf, bbox_normalized = _bbox_from_points(normalized_box, page_info, source_width, source_height)

    return {
        "token_id": token_id,
        "page": page_info.page,
        "text": text.strip(),
        "confidence": _first_float(raw_token, ["confidence", "score", "rec_score", "recScore"]),
        "bbox_status": "available" if bbox_pdf else "missing",
        "bbox_pdf": bbox_pdf,
        "bbox_normalized": bbox_normalized,
    }


def _first_scalar(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _first_float(data: dict[str, Any], keys: list[str]) -> float | None:
    value = _first_scalar(data, keys)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_source_size(data: dict[str, Any]) -> tuple[float | None, float | None]:
    for key in ("input_img_shape", "inputImageShape", "image_shape", "imageShape"):
        shape = data.get(key)
        if isinstance(shape, list) and len(shape) >= 2:
            height = float(shape[0])
            width = float(shape[1])
            return width, height
    width = data.get("width") or data.get("image_width")
    height = data.get("height") or data.get("image_height")
    if width and height:
        return float(width), float(height)
    return None, None


def _bbox_from_points(
    points: list[list[float]],
    page: PageInfo,
    source_width: float | None = None,
    source_height: float | None = None,
) -> tuple[BBoxPdf, BBoxNormalized]:
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

    bbox_pdf = BBoxPdf(
        x=round(x, 3),
        y=round(y, 3),
        width=round(width, 3),
        height=round(height, 3),
        page_width=page.width,
        page_height=page.height,
    )
    return bbox_pdf, _normalized_bbox_from_pdf(bbox_pdf)


def _normalized_bbox_from_pdf(bbox: BBoxPdf) -> BBoxNormalized:
    return BBoxNormalized(
        x1=round(bbox.x / bbox.page_width, 6),
        y1=round(bbox.y / bbox.page_height, 6),
        x2=round((bbox.x + bbox.width) / bbox.page_width, 6),
        y2=round((bbox.y + bbox.height) / bbox.page_height, 6),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
