import unittest

import document_parser.llm as llm_module
from document_parser.llm_agents import AGENT_ITEMS_SCHEMA
from document_parser.config import RuntimeConfig
from document_parser.llm import LlmClient
from document_parser.llm import OpenAICompatibleLlmClient
from document_parser.llm_agents import SpanGroundedFieldAgent
from document_parser.models import FieldDefinition, GeneratedSchema, TextSpan


class FakeLlmClient(LlmClient):
    def __init__(self) -> None:
        self.system = ""
        self.user = ""
        self.schema = {}

    def structured_json(self, system, user, schema):
        self.system = system
        self.user = user
        self.schema = schema
        return {
            "items": [
                {
                    "semantic_key": "custom.design_note",
                    "display_name": "设计注意",
                    "field_type": "requirement",
                    "span_id": "span_0001",
                    "start_offset": 0,
                    "end_offset": 9,
                    "text": "设计注意：保留",
                    "confidence": 0.91,
                    "entity_id": None,
                    "section_id": "sec_label_text",
                }
            ]
        }


class MaxTokenFakeLlmClient(FakeLlmClient):
    def __init__(self) -> None:
        super().__init__()
        self.max_tokens = None

    def structured_json_with_max_tokens(self, system, user, schema, max_tokens):
        self.max_tokens = max_tokens
        return self.structured_json(system, user, schema)


class LlmAgentTests(unittest.TestCase):
    def test_field_schema_accepts_ordered_multi_ranges(self) -> None:
        field_schema = AGENT_ITEMS_SCHEMA["properties"]["fields"]["items"]

        self.assertIn("ranges", field_schema["properties"])
        self.assertNotIn("span_id", field_schema["required"])

    def test_generates_span_grounded_agent_plan_with_schema(self) -> None:
        fake = FakeLlmClient()
        schema = GeneratedSchema(
            schema_id="schema_dynamic_001",
            auto_generated=True,
            schema_version="dynamic_v1",
            sections=[],
            entity_types=[],
            field_definitions=[
                FieldDefinition(
                    field_def_id="fdef_0001",
                    semantic_key="requirement.text",
                    display_name="文字要求",
                    field_type="requirement",
                    criticality="non_critical",
                )
            ],
        )
        spans = [TextSpan("span_0001", 1, "设计注意：保留", "pdf_text")]
        body = SpanGroundedFieldAgent(fake).generate_candidates(schema, spans)

        self.assertEqual(body["items"][0]["span_id"], "span_0001")
        self.assertIn("span_id=span_0001", fake.user)
        self.assertIn("fields", fake.schema["required"])

    def test_openai_compatible_client_requests_structured_json(self) -> None:
        calls = []
        original_post = llm_module.requests.post

        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": '{"items":[]}'}}]}

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return Response()

        try:
            llm_module.requests.post = fake_post
            client = OpenAICompatibleLlmClient(
                RuntimeConfig(
                    glm_ocr_api_key="ocr-key",
                    llm_api_key="llm-key",
                    llm_base_url="https://llm.test/v1",
                    llm_model="model-name",
                )
            )
            result = client.structured_json("system", "user", {"type": "object"})
        finally:
            llm_module.requests.post = original_post

        self.assertEqual(result, {"items": []})
        self.assertEqual(calls[0]["url"], "https://llm.test/v1/chat/completions")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer llm-key")
        self.assertEqual(calls[0]["json"]["model"], "model-name")
        self.assertEqual(calls[0]["json"]["max_tokens"], 8192)
        self.assertEqual(calls[0]["json"]["response_format"]["type"], "json_schema")

    def test_table_agent_uses_larger_output_budget_without_changing_other_agents(self) -> None:
        fake = MaxTokenFakeLlmClient()
        schema = GeneratedSchema("schema", True, "v1", [], [], [], [], [])

        SpanGroundedFieldAgent(fake).generate_table_extraction_plan(schema, [])

        self.assertEqual(fake.max_tokens, 32768)

    def test_openai_compatible_client_falls_back_to_json_object_format(self) -> None:
        calls = []
        original_post = llm_module.requests.post

        class Response:
            def __init__(self, status_code):
                self.status_code = status_code

            def json(self):
                return {"choices": [{"message": {"content": '{"items":[]}'}}]}

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            return Response(400 if len(calls) == 1 else 200)

        try:
            llm_module.requests.post = fake_post
            client = OpenAICompatibleLlmClient(
                RuntimeConfig(
                    glm_ocr_api_key="ocr-key",
                    llm_api_key="llm-key",
                    llm_base_url="https://llm.test",
                    llm_model="model-name",
                )
            )
            result = client.structured_json("system", "user", {"type": "object"})
        finally:
            llm_module.requests.post = original_post

        self.assertEqual(result, {"items": []})
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(calls[1]["response_format"]["type"], "json_object")

    def test_openai_compatible_client_accepts_fenced_json_content(self) -> None:
        calls = []
        original_post = llm_module.requests.post

        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": '```json\n{"fields":[]}\n```'}}]}

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            return Response()

        try:
            llm_module.requests.post = fake_post
            client = OpenAICompatibleLlmClient(
                RuntimeConfig(
                    glm_ocr_api_key="ocr-key",
                    llm_api_key="llm-key",
                    llm_base_url="https://llm.test",
                    llm_model="model-name",
                )
            )
            result = client.structured_json("system", "user", {"type": "object"})
        finally:
            llm_module.requests.post = original_post

        self.assertEqual(result, {"fields": []})

    def test_openai_compatible_client_raises_on_non_json_response(self) -> None:
        calls = []
        original_post = llm_module.requests.post

        class Response:
            status_code = 200

            def __init__(self, content):
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            return Response("Here are the fields: none.")

        try:
            llm_module.requests.post = fake_post
            client = OpenAICompatibleLlmClient(
                RuntimeConfig(
                    glm_ocr_api_key="ocr-key",
                    llm_api_key="llm-key",
                    llm_base_url="https://llm.test",
                    llm_model="model-name",
                )
            )
            with self.assertRaises(llm_module.LlmError):
                client.structured_json("system", "user", {"type": "object"})
        finally:
            llm_module.requests.post = original_post

        self.assertEqual(len(calls), 1)

    def test_openai_compatible_client_raises_on_empty_message_content(self) -> None:
        calls = []
        original_post = llm_module.requests.post

        class Response:
            status_code = 200

            def __init__(self, content):
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            if len(calls) == 1:
                return Response("")
            return Response('{"fields":[]}')

        try:
            llm_module.requests.post = fake_post
            client = OpenAICompatibleLlmClient(
                RuntimeConfig(
                    glm_ocr_api_key="ocr-key",
                    llm_api_key="llm-key",
                    llm_base_url="https://llm.test",
                    llm_model="model-name",
                )
            )
            with self.assertRaises(llm_module.LlmError):
                client.structured_json("system", "user", {"type": "object"})
        finally:
            llm_module.requests.post = original_post

        self.assertEqual(len(calls), 1)

    def test_openai_compatible_client_raises_when_json_is_invalid(self) -> None:
        calls = []
        original_post = llm_module.requests.post

        class Response:
            status_code = 200

            def __init__(self, content):
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            return Response("not json")

        try:
            llm_module.requests.post = fake_post
            client = OpenAICompatibleLlmClient(
                RuntimeConfig(
                    glm_ocr_api_key="ocr-key",
                    llm_api_key="llm-key",
                    llm_base_url="https://llm.test",
                    llm_model="model-name",
                )
            )
            with self.assertRaises(llm_module.LlmError):
                client.structured_json("system", "user", {"type": "object"})
        finally:
            llm_module.requests.post = original_post

        self.assertEqual(len(calls), 1)

    def test_openai_compatible_client_retries_once_when_json_object_is_malformed(self) -> None:
        calls = []
        original_post = llm_module.requests.post

        class Response:
            status_code = 200

            def __init__(self, content):
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            return Response('{"fields":[}') if len(calls) == 1 else Response('{"fields":[]}')

        try:
            llm_module.requests.post = fake_post
            client = OpenAICompatibleLlmClient(
                RuntimeConfig(
                    glm_ocr_api_key="ocr-key",
                    llm_api_key="llm-key",
                    llm_base_url="https://llm.test",
                    llm_model="model-name",
                )
            )
            result = client.structured_json("system", "user", {"type": "object"})
        finally:
            llm_module.requests.post = original_post

        self.assertEqual(result, {"fields": []})
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["response_format"]["type"], "json_object")

    def test_openai_compatible_client_retries_one_timeout(self) -> None:
        client = OpenAICompatibleLlmClient(
            RuntimeConfig(
                glm_ocr_api_key="ocr-key",
                llm_api_key="llm-key",
                llm_base_url="https://llm.test",
                llm_model="model-name",
            )
        )
        calls = []

        class Response:
            status_code = 200

        def fake_post(payload):
            calls.append(payload)
            if len(calls) == 1:
                raise llm_module.LlmError("LLM request exceeded 180 seconds.")
            return Response()

        client._post = fake_post

        response = client._post_with_timeout_retry({"model": "model-name"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
