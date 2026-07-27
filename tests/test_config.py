import os
import tempfile
import unittest
from pathlib import Path

from document_parser.config import ConfigError, RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_default_requires_only_glm_ocr_key(self) -> None:
        old_env = os.environ.copy()
        try:
            for key in ("GLM_OCR_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY", "GLM_OCR_MODEL", "PPOCRV6_API_KEY", "PPOCRV6_TOKEN", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
                os.environ.pop(key, None)
            with tempfile.TemporaryDirectory() as temp_dir:
                with self.assertRaises(ConfigError):
                    RuntimeConfig.from_env(require_secrets=True, dotenv_path=Path(temp_dir) / ".env")

                os.environ["GLM_OCR_API_KEY"] = "ocr-key"
                config = RuntimeConfig.from_env(require_secrets=True, dotenv_path=Path(temp_dir) / ".env")
            self.assertEqual(config.glm_ocr_api_key, "ocr-key")
            self.assertEqual(config.glm_ocr_model, "glm-ocr")
            self.assertEqual(config.ppocrv6_api_key, "")
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_zai_token_alias_is_accepted(self) -> None:
        old_env = os.environ.copy()
        try:
            for key in ("GLM_OCR_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY"):
                os.environ.pop(key, None)
            os.environ["ZAI_API_KEY"] = "ocr-token"
            with tempfile.TemporaryDirectory() as temp_dir:
                config = RuntimeConfig.from_env(require_secrets=True, dotenv_path=Path(temp_dir) / ".env")
            self.assertEqual(config.glm_ocr_api_key, "ocr-token")
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_ppocrv6_token_alias_is_accepted(self) -> None:
        old_env = os.environ.copy()
        try:
            for key in ("GLM_OCR_API_KEY", "PPOCRV6_API_KEY", "PPOCRV6_TOKEN"):
                os.environ.pop(key, None)
            os.environ["GLM_OCR_API_KEY"] = "glm-token"
            os.environ["PPOCRV6_TOKEN"] = "pp-token"
            with tempfile.TemporaryDirectory() as temp_dir:
                config = RuntimeConfig.from_env(require_secrets=True, dotenv_path=Path(temp_dir) / ".env")
            self.assertEqual(config.ppocrv6_api_key, "pp-token")
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_dotenv_loads_without_overriding_existing_env(self) -> None:
        old_env = os.environ.copy()
        try:
            os.environ["GLM_OCR_MODEL"] = "custom-glm-ocr"
            os.environ.pop("GLM_OCR_API_KEY", None)
            os.environ.pop("ZAI_API_KEY", None)
            os.environ.pop("ZHIPUAI_API_KEY", None)
            with tempfile.TemporaryDirectory() as temp_dir:
                dotenv = Path(temp_dir) / ".env"
                dotenv.write_text(
                    "GLM_OCR_API_KEY=ocr-key\nGLM_OCR_MODEL=dotenv-glm-ocr\n",
                    encoding="utf-8",
                )
                config = RuntimeConfig.from_env(require_secrets=True, dotenv_path=dotenv)
            self.assertEqual(config.glm_ocr_api_key, "ocr-key")
            self.assertEqual(config.glm_ocr_model, "custom-glm-ocr")
        finally:
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
