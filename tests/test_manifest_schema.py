import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_MANIFEST_SCHEMA = ROOT / "schemas" / "input_manifest.schema.json"
README = ROOT / "README.md"


class InputManifestSchemaTests(unittest.TestCase):
    def test_manifest_schema_matches_single_key_runtime_policy(self) -> None:
        schema = _load_schema()
        properties = schema["properties"]

        self.assertEqual(properties["ocr_mode"]["enum"], ["glm_ocr"])
        self.assertEqual(properties["use_llm_agent"]["type"], "boolean")
        self.assertEqual(properties["llm_mode"]["enum"], ["disabled", "agent"])
        self.assertEqual(properties["llm_agent"]["properties"]["enabled"]["type"], "boolean")
        self.assertEqual(properties["agent"]["properties"]["enabled"]["type"], "boolean")
        self.assertEqual(properties["table_parser_mode"]["enum"], ["validate_only", "feed_structure"])
        self.assertEqual(properties["repair_mode"]["enum"], ["execute_plan", "plan_only"])

    def test_manifest_schema_keeps_offline_agent_items_path(self) -> None:
        schema = _load_schema()
        agent_items = schema["properties"]["agent_items_path"]

        self.assertEqual(agent_items["type"], "string")
        self.assertIn("offline", agent_items["description"])
        self.assertIn("without calling an online LLM", agent_items["description"])

    def test_public_readme_documents_external_consistency_cli_only(self) -> None:
        text = README.read_text(encoding="utf-8")

        self.assertIn("GLM_OCR_API_KEY", text)
        self.assertIn("check-package-consistency", text)
        self.assertIn("标准模板 Excel", text)
        self.assertNotIn("LLM_API_KEY", text)
        self.assertNotIn("--use-llm-agent", text)
        self.assertNotIn("动态", text)
        self.assertNotIn("Agent", text)
        self.assertNotIn("OPENAI_API_KEY", text)
        self.assertNotIn("OPENAI_MODEL", text)


def _load_schema() -> dict:
    return json.loads(INPUT_MANIFEST_SCHEMA.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
