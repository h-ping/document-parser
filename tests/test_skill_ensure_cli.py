from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
ENSURE_CLI = ROOT / ".agents" / "skills" / "check-package-consistency" / "scripts" / "ensure_cli.py"


class SkillEnsureCliTests(unittest.TestCase):
    def test_required_help_flags_include_llm_mode(self) -> None:
        module = _load_ensure_cli()

        self.assertIn("--llm-mode", module.REQUIRED_HELP_FLAGS)

    def test_project_candidates_prefer_explicit_and_current_sources(self) -> None:
        module = _load_ensure_cli()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            explicit_project = _fake_project(root / "explicit")
            cwd_project = _fake_project(root / "cwd")
            skill_project = _fake_project(root / "skill-project")
            workspace = root / "workspace"
            workspace_project = _fake_project(workspace / "document-parser")
            script_file = skill_project / ".agents" / "skills" / "check-package-consistency" / "scripts" / "ensure_cli.py"
            script_file.parent.mkdir(parents=True)
            script_file.touch()

            context = module.DiscoveryContext(
                workspace=workspace,
                cwd=cwd_project,
                script_file=script_file,
                env_source_dir=str(explicit_project),
            )

            candidates = module._candidate_project_dirs(context)

        self.assertEqual(
            [(candidate.source, candidate.path) for candidate in candidates],
            [
                ("source_env_project", explicit_project),
                ("cwd_project", cwd_project),
                ("skill_project", skill_project),
                ("workspace_project", workspace_project),
            ],
        )


def _load_ensure_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ensure_cli_under_test", ENSURE_CLI)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load {ENSURE_CLI}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_project(path: Path) -> Path:
    (path / "src" / "document_parser").mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname = \"document-parser\"\n", encoding="utf-8")
    (path / "src" / "document_parser" / "consistency_cli.py").write_text("", encoding="utf-8")
    return path.resolve()


if __name__ == "__main__":
    unittest.main()
