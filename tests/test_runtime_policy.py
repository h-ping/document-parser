import unittest

from document_parser.config import RuntimeConfig
from document_parser.runtime_policy import RuntimePolicyError, assert_runtime_policy_passed, build_runtime_policy, runtime_options_from_manifest


class RuntimePolicyTests(unittest.TestCase):
    def test_cloud_ocr_base_path_requires_only_glm_ocr_key(self) -> None:
        policy = build_runtime_policy(
            source="cli",
            options={},
            config=_config(glm_key="ocr-key"),
            required_env_vars=["GLM_OCR_API_KEY"],
            ocr_fixture_path=None,
            agent_items_path=None,
            use_llm_agent=False,
            max_repair_rounds=2,
        )

        self.assertEqual(policy["status"], "pass")
        self.assertTrue(policy["ocr"]["cloud_ocr_enabled"])
        self.assertEqual(policy["secrets"]["required_env_vars"], ["GLM_OCR_API_KEY"])
        self.assertEqual(policy["secrets"]["optional_env_vars"], ["GLM_OCR_MODEL"])
        self.assertEqual(policy["ocr"]["provider"], "glm_ocr")
        self.assertNotIn("OPENAI_API_KEY", policy["secrets"]["required_env_vars"])
        self.assertEqual(policy["repair"]["execution_policy"], "agent_repair_execute_plan_recompile_validate")

    def test_runtime_policy_rejects_missing_required_env_names(self) -> None:
        policy = build_runtime_policy(
            source="cli",
            options={},
            config=_config(glm_key="ocr-key"),
            required_env_vars=["GLM_OCR_API_KEY", "OTHER_API_KEY"],
            ocr_fixture_path=None,
            agent_items_path=None,
            use_llm_agent=False,
            max_repair_rounds=2,
        )

        self.assertEqual(policy["status"], "review_required")
        failed_types = {check["check_type"] for check in policy["checks"] if check["result"] == "failed"}
        self.assertIn("required_env_vars_available", failed_types)

    def test_fixture_path_does_not_require_ocr_secret(self) -> None:
        policy = build_runtime_policy(
            source="cli",
            options={},
            config=_config(),
            required_env_vars=[],
            ocr_fixture_path=__file__,
            agent_items_path=None,
            use_llm_agent=False,
            max_repair_rounds=2,
        )

        self.assertEqual(policy["status"], "pass")
        self.assertEqual(policy["ocr"]["mode"], "recorded_fixture")
        self.assertFalse(policy["ocr"]["cloud_ocr_enabled"])
        self.assertEqual(policy["secrets"]["required_env_vars"], [])

    def test_table_feed_structure_mode_is_supported(self) -> None:
        policy = build_runtime_policy(
            source="manifest",
            options={"table_parser_mode": "feed_structure"},
            config=_config(glm_key="ocr-key"),
            required_env_vars=["GLM_OCR_API_KEY"],
            ocr_fixture_path=None,
            agent_items_path=None,
            use_llm_agent=False,
            max_repair_rounds=2,
        )

        self.assertEqual(policy["status"], "pass")
        self.assertEqual(policy["table_parser"]["mode"], "feed_structure")
        self.assertEqual(policy["table_parser"]["execution_policy"], "feed_structure_when_quality_pass")
        self.assertTrue(policy["table_parser"]["feeds_standard_items"])

    def test_online_llm_agent_requires_runtime_config(self) -> None:
        policy = build_runtime_policy(
            source="manifest",
            options={},
            config=_config(glm_key="ocr-key", llm_key="llm-key", llm_base_url="https://llm.test/v1", llm_model="model"),
            required_env_vars=["GLM_OCR_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"],
            ocr_fixture_path=None,
            agent_items_path=None,
            use_llm_agent=True,
            max_repair_rounds=1,
        )

        self.assertEqual(policy["status"], "pass")
        self.assertTrue(policy["llm_agent"]["enabled"])
        self.assertEqual(policy["llm_agent"]["mode"], "online")
        self.assertEqual(policy["llm_agent"]["required_env_vars"], ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"])
        self.assertTrue(policy["llm_agent"]["runtime_managed_online_llm"])

    def test_cloud_ocr_requires_consent_before_runtime_can_pass(self) -> None:
        policy = build_runtime_policy(
            source="manifest",
            options={"cloud_ocr_consent": False},
            config=_config(glm_key="ocr-key"),
            required_env_vars=["GLM_OCR_API_KEY"],
            ocr_fixture_path=None,
            agent_items_path=None,
            use_llm_agent=False,
            max_repair_rounds=2,
        )

        self.assertEqual(policy["status"], "review_required")
        failed_types = {check["check_type"] for check in policy["checks"] if check["result"] == "failed"}
        self.assertIn("cloud_ocr_consent", failed_types)
        with self.assertRaises(RuntimePolicyError):
            assert_runtime_policy_passed(policy)

    def test_manifest_boolean_strings_are_parsed(self) -> None:
        options = runtime_options_from_manifest(
            {
                "cloud_ocr_consent": "false",
                "block_downstream_on_quality_failure": "true",
            }
        )

        self.assertFalse(options["cloud_ocr_consent"])
        self.assertTrue(options["block_downstream_on_quality_failure"])

    def test_enhanced_layout_requires_llm_agent(self) -> None:
        policy = build_runtime_policy(
            source="manifest",
            options={"layout_mode": "char_atoms_high_recall"},
            config=_config(glm_key="ocr-key"),
            required_env_vars=["GLM_OCR_API_KEY"],
            ocr_fixture_path=None,
            agent_items_path=None,
            use_llm_agent=False,
            max_repair_rounds=2,
        )

        self.assertEqual(policy["status"], "review_required")
        failed_types = {check["check_type"] for check in policy["checks"] if check["result"] == "failed"}
        self.assertIn("enhanced_layout_requires_llm_agent", failed_types)


def _config(glm_key: str = "", llm_key: str = "", llm_base_url: str = "", llm_model: str = "") -> RuntimeConfig:
    return RuntimeConfig(
        glm_ocr_api_key=glm_key,
        llm_api_key=llm_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )


if __name__ == "__main__":
    unittest.main()
