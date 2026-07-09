from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

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


class PPOCRV6OcrClient(OcrClient):
    def __init__(self, config: RuntimeConfig, timeout_seconds: int = 180, poll_interval_seconds: float = 5.0) -> None:
        self._api_key = config.ppocrv6_api_key
        self._api_url = config.ppocrv6_api_url
        self._model = config.ppocrv6_model
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def recognize_pdf(self, pdf_path: Path, pages: list[PageInfo]) -> list[OcrLine]:
        return normalize_ppocrv6_jsonl(self._recognize_file(pdf_path), pages)

    def recognize_image(self, image_path: Path, page: PageInfo) -> list[OcrLine]:
        return normalize_ppocrv6_jsonl(self._recognize_file(image_path), [page])

    def _recognize_file(self, path: Path) -> str:
        headers = {
            "Authorization": f"bearer {self._api_key}",
        }
        optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useTextlineOrientation": False,
        }
        data = {
            "model": self._model,
            "optionalPayload": json.dumps(optional_payload),
        }
        with path.open("rb") as handle:
            response = requests.post(
                self._api_url,
                headers=headers,
                data=data,
                files={"file": handle},
                timeout=self._timeout_seconds,
            )
        if response.status_code != 200:
            raise OcrError(f"PPOCRV6 job submission failed with HTTP {response.status_code}")

        job_id = _job_id_from_response(response.json())
        result_url = self._poll_job(job_id, headers)
        jsonl_response = requests.get(result_url, timeout=self._timeout_seconds)
        if jsonl_response.status_code != 200:
            raise OcrError(f"PPOCRV6 result download failed with HTTP {jsonl_response.status_code}")
        return jsonl_response.text

    def _poll_job(self, job_id: str, headers: dict[str, str]) -> str:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            response = requests.get(f"{self._api_url}/{job_id}", headers=headers, timeout=self._timeout_seconds)
            if response.status_code != 200:
                raise OcrError(f"PPOCRV6 job polling failed with HTTP {response.status_code}")
            body = response.json()
            data = body.get("data") or {}
            state = data.get("state")
            if state == "done":
                result_url = ((data.get("resultUrl") or {}).get("jsonUrl"))
                if not result_url:
                    raise OcrError("PPOCRV6 job completed without jsonUrl result.")
                return str(result_url)
            if state == "failed":
                raise OcrError(f"PPOCRV6 job failed: {data.get('errorMsg', 'unknown error')}")
            if state not in {"pending", "running"}:
                raise OcrError(f"PPOCRV6 job returned unexpected state: {state}")
            time.sleep(self._poll_interval_seconds)
        raise OcrError("PPOCRV6 job polling timed out.")


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


def _job_id_from_response(body: dict[str, Any]) -> str:
    data = body.get("data") or {}
    job_id = data.get("jobId")
    if not job_id:
        raise OcrError("PPOCRV6 job submission response did not include data.jobId.")
    return str(job_id)


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
