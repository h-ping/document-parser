import unittest

import document_parser.ocr as ocr_module
from document_parser.config import RuntimeConfig
from document_parser.models import PageInfo
from document_parser.ocr import GLMOcrClient, PPOCRV6Client, normalize_glm_ocr_response, normalize_ppocrv6_jsonl, normalize_ppocrv6_response


class PPOCRV6NormalizeTests(unittest.TestCase):
    def test_normalizes_glm_ocr_layout_details(self) -> None:
        response = {
            "layout_details": [
                [
                    {
                        "index": 1,
                        "label": "text",
                        "bbox_2d": [10, 20, 210, 60],
                        "content": "品名：红豆奶茶",
                        "width": 1000,
                        "height": 500,
                    },
                    {
                        "index": 2,
                        "label": "table",
                        "bbox_2d": [20, 80, 420, 180],
                        "content": "|项目|每100克|NRV%|\n|---|---|---|\n|能量|100千焦|1%|",
                        "width": 1000,
                        "height": 500,
                    },
                ]
            ],
            "data_info": {"pages": [{"width": 1000, "height": 500}]},
        }

        lines = normalize_glm_ocr_response(response, [PageInfo(page=1, width=500, height=250)])

        self.assertEqual([line.text for line in lines], ["品名：红豆奶茶", "项目 每100克 NRV%", "能量 100千焦 1%"])
        self.assertEqual(lines[0].bbox_pdf.x, 5.0)
        self.assertEqual(lines[0].bbox_pdf.y, 10.0)
        self.assertEqual(lines[0].bbox_pdf.width, 100.0)
        self.assertEqual(lines[0].metadata["provider"], "glm_ocr")
        self.assertEqual(lines[1].block_id, lines[2].block_id)

    def test_glm_ocr_client_uses_layout_parsing_sdk(self) -> None:
        calls = []
        original_client = ocr_module.ZhipuAiClient

        class Response:
            def model_dump(self):
                return {
                    "layout_details": [
                        [
                            {
                                "index": 1,
                                "label": "text",
                                "bbox_2d": [0, 0, 100, 20],
                                "content": "品名：牛奶",
                                "width": 100,
                                "height": 100,
                            }
                        ]
                    ]
                }

        class LayoutParsing:
            def create(self, **kwargs):
                calls.append(kwargs)
                return Response()

        class FakeZhipuAiClient:
            def __init__(self, api_key):
                calls.append({"api_key": api_key})
                self.layout_parsing = LayoutParsing()

        try:
            ocr_module.ZhipuAiClient = FakeZhipuAiClient
            client = GLMOcrClient(RuntimeConfig(glm_ocr_api_key="glm-key"), timeout_seconds=12)
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                image = Path(temp_dir) / "input.png"
                image.write_bytes(b"abc")
                lines = client.recognize_image(image, PageInfo(page=1, width=100, height=100))
        finally:
            ocr_module.ZhipuAiClient = original_client

        self.assertEqual(lines[0].text, "品名：牛奶")
        self.assertEqual(calls[0]["api_key"], "glm-key")
        self.assertEqual(calls[1]["model"], "glm-ocr")
        self.assertEqual(calls[1]["file"], "data:image/png;base64,YWJj")
        self.assertFalse(calls[1]["return_crop_images"])
        self.assertFalse(calls[1]["need_layout_visualization"])
        self.assertEqual(calls[1]["timeout"], 12)

    def test_glm_ocr_client_retries_transient_request_failure(self) -> None:
        api_calls = []
        sleeps = []
        original_client = ocr_module.ZhipuAiClient
        original_sleep = ocr_module.time.sleep

        class Response:
            def model_dump(self):
                return {
                    "layout_details": [
                        [
                            {
                                "index": 1,
                                "label": "text",
                                "bbox_2d": [0, 0, 100, 20],
                                "content": "品名：牛奶",
                                "width": 100,
                                "height": 100,
                            }
                        ]
                    ]
                }

        class LayoutParsing:
            def create(self, **kwargs):
                api_calls.append(kwargs)
                if len(api_calls) == 1:
                    raise ocr_module.ZaiError('Error code: 400, with error text {"error":{"code":"1210"}}')
                return Response()

        class FakeZhipuAiClient:
            def __init__(self, api_key):
                self.layout_parsing = LayoutParsing()

        try:
            ocr_module.ZhipuAiClient = FakeZhipuAiClient
            ocr_module.time.sleep = sleeps.append
            client = GLMOcrClient(RuntimeConfig(glm_ocr_api_key="glm-key"), timeout_seconds=12)
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                image = Path(temp_dir) / "input.png"
                image.write_bytes(b"abc")
                lines = client.recognize_image(image, PageInfo(page=1, width=100, height=100))
        finally:
            ocr_module.ZhipuAiClient = original_client
            ocr_module.time.sleep = original_sleep

        self.assertEqual(lines[0].text, "品名：牛奶")
        self.assertEqual(len(api_calls), 2)
        self.assertEqual(sleeps, [1.5])

    def test_ppocrv6_client_submits_polls_downloads_and_normalizes_jsonl(self) -> None:
        requests_calls = []
        original_post = ocr_module.requests.post
        original_get = ocr_module.requests.get

        class Response:
            def __init__(self, status_code, body, text="", content: bytes | None = None) -> None:
                self.status_code = status_code
                self._body = body
                self.text = text
                self.content = content if content is not None else text.encode("utf-8")

            def json(self):
                return self._body

        def fake_post(url, headers, data, files, timeout):
            requests_calls.append({"method": "post", "url": url, "headers": headers, "data": data, "files": files, "timeout": timeout})
            return Response(200, {"data": {"jobId": "job-1"}})

        def fake_get(url, headers=None, timeout=None):
            requests_calls.append({"method": "get", "url": url, "headers": headers, "timeout": timeout})
            if url.endswith("/job-1"):
                return Response(200, {"data": {"state": "done", "resultUrl": {"jsonUrl": "https://result.example.test/ocr.jsonl"}}})
            jsonl = '{"result":{"ocrResults":[{"prunedResult":{"rec_texts":["品名：牛奶"],"rec_scores":[0.98],"dt_polys":[[[0,0],[100,0],[100,20],[0,20]]]}}]}}\n'
            return Response(
                200,
                {},
                jsonl.encode("utf-8").decode("ptcp154"),
                jsonl.encode("utf-8"),
            )

        try:
            ocr_module.requests.post = fake_post
            ocr_module.requests.get = fake_get
            client = PPOCRV6Client(RuntimeConfig(ppocrv6_api_key="pp-token", ppocrv6_job_url="https://pp.example.test/jobs", ppocrv6_model="PP-OCRv6-test"))
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                image = Path(temp_dir) / "input.png"
                image.write_bytes(b"abc")
                lines = client.recognize_image(image, PageInfo(page=1, width=100, height=100))
        finally:
            ocr_module.requests.post = original_post
            ocr_module.requests.get = original_get

        self.assertEqual(lines[0].text, "品名：牛奶")
        self.assertEqual(lines[0].metadata["provider"], "ppocrv6")
        self.assertEqual(requests_calls[0]["method"], "post")
        self.assertEqual(requests_calls[0]["url"], "https://pp.example.test/jobs")
        self.assertEqual(requests_calls[0]["headers"]["Authorization"], "bearer pp-token")
        self.assertEqual(requests_calls[0]["data"]["model"], "PP-OCRv6-test")
        self.assertIn("optionalPayload", requests_calls[0]["data"])
        self.assertEqual(requests_calls[1]["url"], "https://pp.example.test/jobs/job-1")
        self.assertEqual(requests_calls[2]["url"], "https://result.example.test/ocr.jsonl")

    def test_ppocrv6_submit_retries_only_safe_failures_and_reopens_file(self) -> None:
        submit_calls = []
        uploaded_payloads = []
        sleeps = []
        original_post = ocr_module.requests.post
        original_sleep = ocr_module.time.sleep

        class Response:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code

            def json(self):
                return {"data": {"jobId": "job-safe-retry"}}

        def fake_post(url, headers, data, files, timeout):
            del url, headers, data, timeout
            submit_calls.append(len(submit_calls) + 1)
            uploaded_payloads.append(files["file"].read())
            if len(submit_calls) == 1:
                raise ocr_module.requests.ConnectTimeout("connect timeout")
            if len(submit_calls) == 2:
                return Response(429)
            return Response(200)

        try:
            ocr_module.requests.post = fake_post
            ocr_module.time.sleep = sleeps.append
            client = PPOCRV6Client(RuntimeConfig(ppocrv6_api_key="pp-token"))
            client._poll_result = lambda job_id: {"job_id": job_id}
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                image = Path(temp_dir) / "input.png"
                image.write_bytes(b"complete-image")
                result = client._recognize_file(image)
        finally:
            ocr_module.requests.post = original_post
            ocr_module.time.sleep = original_sleep

        self.assertEqual(result, {"job_id": "job-safe-retry"})
        self.assertEqual(submit_calls, [1, 2, 3])
        self.assertEqual(uploaded_payloads, [b"complete-image", b"complete-image", b"complete-image"])
        self.assertEqual(sleeps, [1.5, 3.0])

    def test_ppocrv6_submit_does_not_retry_ambiguous_read_timeout(self) -> None:
        submit_calls = []
        sleeps = []
        original_post = ocr_module.requests.post
        original_sleep = ocr_module.time.sleep

        def fake_post(url, headers, data, files, timeout):
            del url, headers, data, files, timeout
            submit_calls.append(1)
            raise ocr_module.requests.ReadTimeout("response lost after upload")

        try:
            ocr_module.requests.post = fake_post
            ocr_module.time.sleep = sleeps.append
            client = PPOCRV6Client(RuntimeConfig(ppocrv6_api_key="pp-token"))
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                image = Path(temp_dir) / "input.png"
                image.write_bytes(b"complete-image")
                with self.assertRaisesRegex(ocr_module.OcrError, "PP-OCRv6 job submit failed: ReadTimeout"):
                    client._recognize_file(image)
        finally:
            ocr_module.requests.post = original_post
            ocr_module.time.sleep = original_sleep

        self.assertEqual(len(submit_calls), 1)
        self.assertEqual(sleeps, [])

    def test_ppocrv6_poll_retries_transient_failures(self) -> None:
        poll_calls = []
        sleeps = []
        original_get = ocr_module.requests.get
        original_sleep = ocr_module.time.sleep

        class Response:
            def __init__(self, status_code: int, body: dict | None = None, content: bytes = b"") -> None:
                self.status_code = status_code
                self._body = body or {}
                self.content = content

            def json(self):
                return self._body

        def fake_get(url, headers=None, timeout=None):
            del headers, timeout
            if url.endswith("/job-retry"):
                poll_calls.append(url)
                if len(poll_calls) == 1:
                    raise ocr_module.requests.ConnectionError("temporary disconnect")
                if len(poll_calls) == 2:
                    return Response(503)
                return Response(200, {"data": {"state": "done", "resultUrl": {"jsonUrl": "https://result.example.test/ocr.jsonl"}}})
            jsonl = b'{"result":{"ocrResults":[{"prunedResult":{"rec_texts":["recovered"]}}]}}\n'
            return Response(200, content=jsonl)

        try:
            ocr_module.requests.get = fake_get
            ocr_module.time.sleep = sleeps.append
            client = PPOCRV6Client(
                RuntimeConfig(ppocrv6_api_key="pp-token", ppocrv6_job_url="https://pp.example.test/jobs"),
                poll_interval_seconds=5,
            )
            result = client._poll_result("job-retry")
        finally:
            ocr_module.requests.get = original_get
            ocr_module.time.sleep = original_sleep

        self.assertEqual(len(poll_calls), 3)
        self.assertEqual(sleeps, [1.5, 3.0])
        self.assertEqual(result["result"]["ocrResults"][0]["prunedResult"]["rec_texts"], ["recovered"])

    def test_ppocrv6_poll_does_not_retry_auth_failure(self) -> None:
        poll_calls = []
        sleeps = []
        original_get = ocr_module.requests.get
        original_sleep = ocr_module.time.sleep

        class Response:
            status_code = 401

        def fake_get(url, headers=None, timeout=None):
            del url, headers, timeout
            poll_calls.append(1)
            return Response()

        try:
            ocr_module.requests.get = fake_get
            ocr_module.time.sleep = sleeps.append
            client = PPOCRV6Client(RuntimeConfig(ppocrv6_api_key="pp-token"))
            with self.assertRaisesRegex(ocr_module.OcrError, "PP-OCRv6 job polling failed with HTTP 401"):
                client._poll_result("job-auth")
        finally:
            ocr_module.requests.get = original_get
            ocr_module.time.sleep = original_sleep

        self.assertEqual(len(poll_calls), 1)
        self.assertEqual(sleeps, [])

    def test_ppocrv6_result_download_retries_transient_failures(self) -> None:
        calls = []
        sleeps = []
        original_get = ocr_module.requests.get
        original_sleep = ocr_module.time.sleep

        class Response:
            def __init__(self, status_code: int, content: bytes = b"") -> None:
                self.status_code = status_code
                self.content = content

        def fake_get(url, timeout):
            calls.append({"url": url, "timeout": timeout})
            if len(calls) == 1:
                raise ocr_module.requests.Timeout("temporary timeout")
            if len(calls) == 2:
                return Response(503)
            jsonl = b'{"result":{"ocrResults":[{"prunedResult":{"rec_texts":["product name"]}}]}}\n'
            return Response(200, jsonl)

        try:
            ocr_module.requests.get = fake_get
            ocr_module.time.sleep = sleeps.append

            result = ocr_module._download_ppocrv6_jsonl("https://result.example.test/ocr.jsonl")
        finally:
            ocr_module.requests.get = original_get
            ocr_module.time.sleep = original_sleep

        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1.5, 3.0])
        self.assertEqual(result["result"]["ocrResults"][0]["prunedResult"]["rec_texts"], ["product name"])

    def test_ppocrv6_result_download_raises_after_three_failures(self) -> None:
        calls = []
        sleeps = []
        original_get = ocr_module.requests.get
        original_sleep = ocr_module.time.sleep

        def fake_get(url, timeout):
            calls.append({"url": url, "timeout": timeout})
            raise ocr_module.requests.ConnectionError("connection reset")

        try:
            ocr_module.requests.get = fake_get
            ocr_module.time.sleep = sleeps.append

            with self.assertRaisesRegex(ocr_module.OcrError, "PP-OCRv6 result download failed: ConnectionError"):
                ocr_module._download_ppocrv6_jsonl("https://result.example.test/ocr.jsonl")
        finally:
            ocr_module.requests.get = original_get
            ocr_module.time.sleep = original_sleep

        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1.5, 3.0])

    def test_normalizes_common_ppocrv6_fields(self) -> None:
        response = {
            "result": {
                "ocrResults": [
                    {
                        "prunedResult": {
                            "input_img_shape": [1000, 500, 3],
                            "rec_texts": ["品名：红豆奶茶"],
                            "rec_scores": [0.98],
                            "dt_polys": [[[10, 20], [210, 20], [210, 50], [10, 50]]],
                            "block_ids": ["block_001"],
                            "rec_word_infos": [
                                [
                                    {"text": "品名", "confidence": 0.99, "bbox": [10, 20, 40, 30]},
                                    {"text": "红豆奶茶", "confidence": 0.97, "bbox": [50, 20, 160, 30]},
                                ]
                            ],
                        }
                    }
                ]
            }
        }
        lines = normalize_ppocrv6_response(response, [PageInfo(page=1, width=250, height=500)])
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "品名：红豆奶茶")
        self.assertAlmostEqual(lines[0].confidence, 0.98)
        self.assertIsNotNone(lines[0].bbox_pdf)
        self.assertIsNotNone(lines[0].bbox_normalized)
        self.assertEqual(lines[0].bbox_pdf.x, 5.0)
        self.assertEqual(lines[0].bbox_pdf.y, 10.0)
        self.assertEqual(lines[0].block_id, "block_001")
        self.assertEqual(len(lines[0].tokens), 2)
        self.assertEqual(lines[0].tokens[0]["text"], "品名")
        self.assertEqual(lines[0].tokens[0]["bbox_status"], "available")
        self.assertEqual(lines[0].metadata["line_index"], 1)

    def test_normalizes_flat_line_indexed_tokens(self) -> None:
        response = {
            "result": {
                "ocrResults": [
                    {
                        "prunedResult": {
                            "rec_texts": ["品名：牛奶", "净含量：250mL"],
                            "rec_scores": [0.99, 0.98],
                            "tokens": [
                                {"text": "品名", "lineIndex": 0},
                                {"text": "牛奶", "lineIndex": 0},
                                {"text": "净含量", "lineIndex": 1},
                                {"text": "250mL", "lineIndex": 1},
                            ],
                        }
                    }
                ]
            }
        }

        lines = normalize_ppocrv6_response(response, [PageInfo(page=1, width=250, height=500)])

        self.assertEqual([token["text"] for token in lines[0].tokens], ["品名", "牛奶"])
        self.assertEqual([token["text"] for token in lines[1].tokens], ["净含量", "250mL"])

    def test_normalizes_ppocrv6_jsonl_pages_in_order(self) -> None:
        jsonl = "\n".join(
            [
                '{"result":{"ocrResults":[{"prunedResult":{"rec_texts":["品名：牛奶"],"rec_scores":[0.99]}}]}}',
                '{"result":{"ocrResults":[{"prunedResult":{"rec_texts":["净含量：250mL"],"rec_scores":[0.98]}}]}}',
            ]
        )

        lines = normalize_ppocrv6_jsonl(
            jsonl,
            [PageInfo(page=1, width=250, height=500), PageInfo(page=2, width=250, height=500)],
        )

        self.assertEqual([line.page for line in lines], [1, 2])
        self.assertEqual([line.text for line in lines], ["品名：牛奶", "净含量：250mL"])

    def test_zero_area_clamped_ocr_bbox_is_marked_missing(self) -> None:
        response = {
            "result": {
                "ocrResults": [
                    {
                        "prunedResult": {
                            "input_img_shape": [100, 100, 3],
                            "rec_texts": ["边缘文字"],
                            "rec_scores": [0.98],
                            "dt_polys": [[[120, 10], [130, 10], [130, 20], [120, 20]]],
                        }
                    }
                ]
            }
        }

        lines = normalize_ppocrv6_response(response, [PageInfo(page=1, width=100, height=100)])

        self.assertIsNone(lines[0].bbox_pdf)
        self.assertIsNone(lines[0].bbox_normalized)

    def test_infers_integer_render_scale_when_source_shape_is_missing(self) -> None:
        response = {
            "result": {
                "ocrResults": [
                    {
                        "prunedResult": {
                            "rec_texts": ["左列", "页底"],
                            "rec_scores": [0.99, 0.99],
                            "dt_polys": [
                                [[20, 40], [60, 40], [60, 60], [20, 60]],
                                [[20, 160], [60, 160], [60, 180], [20, 180]],
                            ],
                        }
                    }
                ]
            }
        }

        lines = normalize_ppocrv6_response(response, [PageInfo(page=1, width=100, height=100)])

        self.assertEqual(lines[0].bbox_pdf.x, 10.0)
        self.assertEqual(lines[0].bbox_pdf.y, 20.0)
        self.assertEqual(lines[1].bbox_pdf.y, 80.0)
        self.assertEqual(lines[0].metadata["source_size"], {"width": 200, "height": 200})

if __name__ == "__main__":
    unittest.main()
