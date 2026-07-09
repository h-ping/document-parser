from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PPOCRV6_API_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_PPOCRV6_MODEL = "PP-OCRv6"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    ppocrv6_api_key: str
    ppocrv6_api_url: str = DEFAULT_PPOCRV6_API_URL
    ppocrv6_model: str = DEFAULT_PPOCRV6_MODEL
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    @classmethod
    def from_env(
        cls,
        require_secrets: bool = True,
        dotenv_path: Path | None = None,
        required_env_vars: list[str] | None = None,
    ) -> "RuntimeConfig":
        if dotenv_path is None:
            dotenv_path = Path.cwd() / ".env"
        load_dotenv(dotenv_path)

        ppocrv6_api_key = os.getenv("PPOCRV6_API_KEY") or os.getenv("PPOCRV6_TOKEN", "")
        ppocrv6_api_url = os.getenv("PPOCRV6_API_URL", DEFAULT_PPOCRV6_API_URL)
        ppocrv6_model = os.getenv("PPOCRV6_MODEL", DEFAULT_PPOCRV6_MODEL)
        llm_api_key = os.getenv("LLM_API_KEY", "")
        llm_base_url = os.getenv("LLM_BASE_URL", "")
        llm_model = os.getenv("LLM_MODEL", "")

        env_values = {
            "PPOCRV6_API_KEY": ppocrv6_api_key,
            "PPOCRV6_TOKEN": ppocrv6_api_key,
            "LLM_API_KEY": llm_api_key,
            "LLM_BASE_URL": llm_base_url,
            "LLM_MODEL": llm_model,
        }
        if require_secrets:
            names = required_env_vars or ["PPOCRV6_API_KEY"]
            missing = [name for name in names if not env_values.get(name)]
            if missing:
                raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            ppocrv6_api_key=ppocrv6_api_key,
            ppocrv6_api_url=ppocrv6_api_url,
            ppocrv6_model=ppocrv6_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
        )


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
