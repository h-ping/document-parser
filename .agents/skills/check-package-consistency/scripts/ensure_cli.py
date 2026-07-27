#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


GITHUB_URL = "git+https://github.com/h-ping/document-parser.git"
REQUIRED_HELP_FLAGS = (
    "--standard",
    "--image",
    "--publish-cos",
    "--cos-dry-run",
    "--ocr-mode",
    "--llm-mode",
    "--ppocr-fixture",
    "--glm-ocr-fixture",
)


@dataclass(frozen=True, slots=True)
class DiscoveryContext:
    workspace: Path
    cwd: Path
    script_file: Path
    env_source_dir: str


@dataclass(frozen=True, slots=True)
class ProjectCandidate:
    source: str
    path: Path


def main() -> int:
    home = Path.home()
    workspace = Path(os.getenv("DOCUMENT_PARSER_WORKSPACE", home / "workspace")).expanduser()
    managed_venv = workspace / ".document-parser-cli-venv"
    context = DiscoveryContext(
        workspace=workspace,
        cwd=Path.cwd(),
        script_file=Path(__file__).resolve(),
        env_source_dir=os.getenv("DOCUMENT_PARSER_SOURCE_DIR", ""),
    )

    for candidate in _candidate_project_dirs(context):
        cli = _install_project(candidate.path)
        if _help_ok([str(cli)]):
            _emit("ready", candidate.source, str(cli), f"initialized from {candidate.path}")
            return 0
        _emit("failed", candidate.source, str(cli), f"installed from {candidate.path}, but --help failed")
        return 1

    path_cli = shutil.which("check-package-consistency")
    if path_cli and _help_ok([path_cli]):
        _emit("ready", "path", path_cli, "found check-package-consistency on PATH")
        return 0

    workspace.mkdir(parents=True, exist_ok=True)
    cli = _install_from_github(managed_venv)
    if _help_ok([str(cli)]):
        _emit("ready", "github_install", str(cli), f"installed from {GITHUB_URL}")
        return 0

    _emit("failed", "github_install", str(cli), "installed from GitHub, but --help failed")
    return 1


def _candidate_project_dirs(context: DiscoveryContext) -> list[ProjectCandidate]:
    candidates = []
    if context.env_source_dir:
        candidates.append(ProjectCandidate("source_env_project", Path(context.env_source_dir).expanduser()))
    candidates.append(ProjectCandidate("cwd_project", context.cwd))
    script_project = _project_containing(context.script_file)
    if script_project:
        candidates.append(ProjectCandidate("skill_project", script_project))
    candidates.append(ProjectCandidate("workspace_project", context.workspace / "document-parser"))
    return _dedupe_existing_projects(candidates)


def _project_containing(path: Path) -> Path | None:
    for parent in path.parents:
        if _looks_like_project(parent):
            return parent
    return None


def _dedupe_existing_projects(candidates: list[ProjectCandidate]) -> list[ProjectCandidate]:
    result = []
    seen = set()
    for candidate in candidates:
        path = candidate.path.expanduser().resolve()
        if str(path) in seen or not _looks_like_project(path):
            continue
        seen.add(str(path))
        result.append(ProjectCandidate(candidate.source, path))
    return result


def _looks_like_project(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "pyproject.toml").is_file()
        and (path / "src" / "document_parser" / "consistency_cli.py").is_file()
    )


def _install_project(project_dir: Path) -> Path:
    venv = project_dir / ".venv"
    _ensure_venv(venv)
    _run([str(venv / "bin" / "python"), "-m", "pip", "install", "-e", str(project_dir)])
    return venv / "bin" / "check-package-consistency"


def _install_from_github(venv: Path) -> Path:
    _ensure_venv(venv)
    _run([str(venv / "bin" / "python"), "-m", "pip", "install", "--upgrade", GITHUB_URL])
    return venv / "bin" / "check-package-consistency"


def _ensure_venv(venv: Path) -> None:
    if not (venv / "bin" / "python").exists():
        _run([sys.executable, "-m", "venv", str(venv)])


def _help_ok(command: list[str]) -> bool:
    completed = subprocess.run(
        [*command, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and all(flag in completed.stdout for flag in REQUIRED_HELP_FLAGS)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _emit(status: str, source: str, cli: str, message: str) -> None:
    print(json.dumps({"status": status, "source": source, "cli": cli, "message": message}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
