import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from document_parser.failure_result import build_failure_result
from document_parser.output_contract import FINAL_JSON_ROOT_KEYS, JOB_STATUSES, PARSE_STATUSES, RISK_TARGET_TYPES
from document_parser.pipeline import _json_export_manifest
from document_parser.schema_artifacts import write_schema_artifacts


ROOT = Path(__file__).resolve().parents[1]
FINAL_RESULT_SCHEMA = ROOT / "schemas" / "final_result.schema.json"


class FinalResultJsonSchemaTests(unittest.TestCase):
    def test_final_result_schema_documents_current_root_contract(self) -> None:
        schema = _load_schema()

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["required"], FINAL_JSON_ROOT_KEYS)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), set(FINAL_JSON_ROOT_KEYS))

    def test_final_result_schema_status_enums_match_output_contract(self) -> None:
        schema = _load_schema()

        self.assertEqual(set(schema["properties"]["job"]["properties"]["status"]["enum"]), JOB_STATUSES)
        self.assertEqual(set(schema["properties"]["document"]["properties"]["parse_status"]["enum"]), PARSE_STATUSES)

    def test_final_result_schema_target_type_enums_match_output_contract(self) -> None:
        schema = _load_schema()

        self.assertEqual(set(schema["$defs"]["risk"]["properties"]["target_type"]["enum"]), RISK_TARGET_TYPES)
        self.assertEqual(set(schema["$defs"]["review_task"]["properties"]["target_type"]["enum"]), RISK_TARGET_TYPES)

    def test_json_export_manifest_points_to_final_result_schema(self) -> None:
        manifest = _json_export_manifest()

        self.assertEqual(manifest["schema_artifact"], "schemas/final_result.schema.json")

    def test_failure_result_points_to_final_result_schema(self) -> None:
        result = build_failure_result(
            input_path=Path("broken.pdf"),
            stage="pdf_read",
            reason="Failed to read PDF broken.pdf",
            error_type="PdfReadError",
        )

        self.assertEqual(list(result.keys()), FINAL_JSON_ROOT_KEYS)
        self.assertEqual(result["metadata"]["json_export"]["schema_artifact"], "schemas/final_result.schema.json")

    def test_schema_artifact_is_copied_into_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_schema_artifacts(output_dir)
            copied = json.loads((output_dir / "schemas" / "final_result.schema.json").read_text(encoding="utf-8"))

        self.assertEqual(copied["required"], FINAL_JSON_ROOT_KEYS)

    def test_schema_artifact_is_packaged_as_importlib_resource(self) -> None:
        packaged = resources.files("document_parser").joinpath("schemas/final_result.schema.json")

        self.assertTrue(packaged.is_file())
        self.assertEqual(json.loads(packaged.read_text(encoding="utf-8")), _load_schema())


def _load_schema() -> dict:
    return json.loads(FINAL_RESULT_SCHEMA.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
