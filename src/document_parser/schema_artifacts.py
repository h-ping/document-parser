from __future__ import annotations

import json
from pathlib import Path

from .utils import write_json


FINAL_RESULT_SCHEMA_ARTIFACT = "schemas/final_result.schema.json"


def write_schema_artifacts(output_dir: Path) -> None:
    schema = json.loads(_source_schema_path().read_text(encoding="utf-8"))
    write_json(output_dir / FINAL_RESULT_SCHEMA_ARTIFACT, schema)


def _source_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / FINAL_RESULT_SCHEMA_ARTIFACT
