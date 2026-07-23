from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .config import ConfigError
from .cos_publish import CosPublishError, load_local_runtime_env, publish_report_to_cos
from .image_compare import ImageCompareError, SUPPORTED_IMAGE_SUFFIXES
from .image_compare_cli import run_compare_package_image
from .standard_xlsx import ParseError as StandardXlsxParseError
from .standard_xlsx import StandardXlsxParser
from .utils import sha256_file, write_json


class ConsistencyPipelineError(RuntimeError):
    def __init__(self, stage: str, reason: str, error_type: str = "ConsistencyPipelineError") -> None:
        super().__init__(reason)
        self.stage = stage
        self.error_type = error_type


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成包装设计图文字一致性报告。")
    parser.add_argument("--standard", required=True, type=Path, help="标准模板 Excel 路径，仅支持 .xlsx。")
    parser.add_argument("--image", required=True, type=Path, help="包装设计图路径，支持 PNG/JPG/JPEG。")
    parser.add_argument("--output-dir", required=True, type=Path, help="报告输出目录。")
    parser.add_argument("--ocr-fixture", type=Path, help="可选，包装图文字识别离线测试结果。")
    parser.add_argument("--publish-cos", action="store_true", help="将客户报告发布到腾讯云 COS，并在 summary 中写入公开链接。")
    parser.add_argument("--cos-key-prefix", help="可选，COS 对象前缀模板，支持 {run_id} 和 {timestamp}。")
    parser.add_argument("--cos-dry-run", action="store_true", help="只生成公开发布包，不实际上传 COS。")
    parser.add_argument("--cos-config", type=Path, help="可选，COS 本机配置文件路径，默认 ~/.config/packaging-consistency-check/secrets.env。")
    args = parser.parse_args(argv)

    try:
        run_consistency_report(
            standard_path=args.standard,
            image_path=args.image,
            output_dir=args.output_dir,
            ocr_fixture_path=args.ocr_fixture,
            publish_cos=args.publish_cos,
            cos_key_prefix=args.cos_key_prefix,
            cos_dry_run=args.cos_dry_run,
            cos_config_path=args.cos_config,
        )
    except (
        ConfigError,
        ConsistencyPipelineError,
        CosPublishError,
        ImageCompareError,
        StandardXlsxParseError,
        OSError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stage = str(getattr(exc, "stage", "package_consistency"))
        error_type = str(getattr(exc, "error_type", exc.__class__.__name__))
        write_json(args.output_dir / "failure_result.json", _failure_result(stage, error_type, str(exc)))
        if not (args.output_dir / "pipeline_summary.json").exists():
            write_json(args.output_dir / "pipeline_summary.json", _failure_summary(args, stage, error_type, str(exc)))
        _write_artifact_index(args.output_dir)
        print(f"一致性报告生成失败：{exc}", file=sys.stderr)
        return 1
    return 0


def run_consistency_report(
    *,
    standard_path: Path,
    image_path: Path,
    output_dir: Path,
    ocr_fixture_path: Path | None = None,
    publish_cos: bool = False,
    cos_key_prefix: str | None = None,
    cos_dry_run: bool = False,
    cos_config_path: Path | None = None,
) -> dict[str, Any]:
    pipeline_started_at = _utc_now()
    pipeline_started_monotonic = time.perf_counter()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    standard_path = _existing_path(standard_path, "standard_structure")
    _assert_standard_template_xlsx(standard_path)
    image_path = _existing_path(image_path, "package_image_comparison")
    ocr_fixture_path = _existing_path(ocr_fixture_path, "package_image_comparison") if ocr_fixture_path else None
    cos_config_path = cos_config_path.expanduser().resolve() if cos_config_path else None
    load_local_runtime_env(cos_config_path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ConsistencyPipelineError("package_image_comparison", f"不支持的包装设计图格式：{image_path.suffix or '<none>'}", "ImageCompareError")

    standard_dir = output_dir / "standard_structure"
    manifest_path = _write_generated_manifest(standard_path, output_dir)
    summary = _base_summary(
        standard_path=standard_path,
        image_path=image_path,
        output_dir=output_dir,
        standard_manifest_path=manifest_path,
        ocr_fixture_path=ocr_fixture_path,
        generated_manifest=True,
        created_at=pipeline_started_at,
    )
    write_json(output_dir / "pipeline_summary.json", summary)

    standard_started_at = _utc_now()
    standard_started_monotonic = time.perf_counter()
    summary["stages"]["standard_structure"] = _stage_running(output_dir / "standard_structure", standard_started_at)
    write_json(output_dir / "pipeline_summary.json", summary)
    try:
        StandardXlsxParser().parse(standard_path, debug_dir=standard_dir)
    except (StandardXlsxParseError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        summary = _with_stage_failure(
            summary,
            "standard_structure",
            exc.__class__.__name__,
            str(exc),
            started_at=standard_started_at,
            duration_seconds=_elapsed_seconds(standard_started_monotonic),
        )
        summary = _complete_summary(summary, pipeline_started_monotonic)
        write_json(output_dir / "pipeline_summary.json", summary)
        _write_artifact_index(output_dir)
        raise ConsistencyPipelineError("standard_structure", str(exc), exc.__class__.__name__) from exc

    quality_report = _read_quality_report(standard_dir)
    standard_duration_seconds = _elapsed_seconds(standard_started_monotonic)
    if not _standard_quality_allows_downstream(quality_report):
        reason = (
            "标准文档结构化质量未通过，已停止包装图一致性比对。"
            f" quality_report.status={quality_report.get('status')!r}, downstream_allowed={quality_report.get('downstream_allowed')!r}"
        )
        summary["stages"]["standard_structure"] = _stage_success(
            artifacts_dir=standard_dir,
            started_at=standard_started_at,
            duration_seconds=standard_duration_seconds,
            extra={"quality_report": _quality_summary(quality_report), "downstream_allowed": False},
        )
        summary = _with_stage_failure(
            summary,
            "standard_structure",
            "QualityGateError",
            reason,
            started_at=standard_started_at,
            duration_seconds=standard_duration_seconds,
        )
        summary = _complete_summary(summary, pipeline_started_monotonic)
        write_json(output_dir / "pipeline_summary.json", summary)
        _write_artifact_index(output_dir)
        raise ConsistencyPipelineError("standard_structure", reason, "QualityGateError")

    summary["stages"]["standard_structure"] = _stage_success(
        artifacts_dir=standard_dir,
        started_at=standard_started_at,
        duration_seconds=standard_duration_seconds,
        extra={"quality_report": _quality_summary(quality_report), "downstream_allowed": True},
    )
    write_json(output_dir / "pipeline_summary.json", summary)

    comparison_started_at = _utc_now()
    comparison_started_monotonic = time.perf_counter()
    summary["stages"]["package_image_comparison"] = _stage_running(output_dir, comparison_started_at)
    write_json(output_dir / "pipeline_summary.json", summary)
    try:
        _ensure_ocr_token_for_real_ocr(ocr_fixture_path)
        comparison_result = run_compare_package_image(
            standard_dir=standard_dir,
            image_path=image_path,
            output_dir=output_dir,
            ocr_fixture_path=ocr_fixture_path,
        )
    except (ConfigError, ImageCompareError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        summary = _with_stage_failure(
            summary,
            "package_image_comparison",
            exc.__class__.__name__,
            str(exc),
            started_at=comparison_started_at,
            duration_seconds=_elapsed_seconds(comparison_started_monotonic),
        )
        summary = _complete_summary(summary, pipeline_started_monotonic)
        write_json(output_dir / "pipeline_summary.json", summary)
        _write_artifact_index(output_dir)
        raise ConsistencyPipelineError("package_image_comparison", str(exc), exc.__class__.__name__) from exc

    summary["status"] = "completed"
    summary["comparison_status"] = comparison_result.get("status")
    summary["stages"]["package_image_comparison"] = _stage_success(
        artifacts_dir=output_dir,
        started_at=comparison_started_at,
        duration_seconds=_elapsed_seconds(comparison_started_monotonic),
        extra={
            "comparison_result": {
                "status": comparison_result.get("status"),
                "target_count": comparison_result.get("target_count"),
                "pass_count": comparison_result.get("pass_count"),
                "critical_count": comparison_result.get("critical_count"),
                "manual_review_count": comparison_result.get("manual_review_count"),
                "info_extra_text_count": comparison_result.get("info_extra_text_count"),
            }
        },
    )
    if publish_cos:
        publish_started_at = _utc_now()
        publish_started_monotonic = time.perf_counter()
        summary["stages"]["publish"] = _stage_running(output_dir / "artifacts" / "06_publish", publish_started_at)
        write_json(output_dir / "pipeline_summary.json", summary)
        try:
            publish_result = publish_report_to_cos(
                output_dir=output_dir,
                run_id=str(summary["run_id"]),
                key_prefix_template=cos_key_prefix,
                dry_run=cos_dry_run,
                config_path=cos_config_path,
            )
        except CosPublishError as exc:
            summary = _with_stage_failure(
                summary,
                "publish",
                "CosPublishError",
                str(exc),
                started_at=publish_started_at,
                duration_seconds=_elapsed_seconds(publish_started_monotonic),
            )
            summary = _complete_summary(summary, pipeline_started_monotonic)
            write_json(output_dir / "pipeline_summary.json", summary)
            _write_artifact_index(output_dir)
            raise ConsistencyPipelineError("publish", str(exc), "CosPublishError") from exc
        summary["publish"] = publish_result
        summary["stages"]["publish"] = _stage_success(
            artifacts_dir=output_dir / "artifacts" / "06_publish",
            started_at=publish_started_at,
            duration_seconds=_elapsed_seconds(publish_started_monotonic),
            extra={"publish": {"status": publish_result.get("status"), "public_url": publish_result.get("public_url"), "dry_run": publish_result.get("dry_run")}},
        )
    summary["key_artifacts"] = _key_artifacts(output_dir)
    if isinstance(summary.get("publish"), dict) and summary["publish"].get("public_url"):
        summary["key_artifacts"]["published_report_html"] = summary["publish"].get("public_url")
    summary = _complete_summary(summary, pipeline_started_monotonic)
    write_json(output_dir / "pipeline_summary.json", summary)
    _write_artifact_index(output_dir)
    return summary


def _write_generated_manifest(standard_path: Path, output_dir: Path) -> Path:
    _assert_standard_template_xlsx(standard_path)
    manifest = {"input_xlsx": str(standard_path)}
    manifest_path = output_dir / "00_inputs" / "standard_manifest.generated.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _assert_standard_template_xlsx(path: Path) -> None:
    if path.suffix.lower() != ".xlsx":
        raise ConsistencyPipelineError(
            "standard_structure",
            f"对外一致性 CLI 仅支持标准模板 Excel（.xlsx），暂不支持其他标准文档格式：{path.suffix or '<none>'}",
            "UnsupportedStandardInputError",
        )


def _read_quality_report(standard_dir: Path) -> dict[str, Any]:
    path = standard_dir / "quality_report.json"
    if not path.exists():
        raise ConsistencyPipelineError("standard_structure", f"标准文档结构化结果缺少 quality_report.json：{path}", "QualityGateError")
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ConsistencyPipelineError("standard_structure", "quality_report.json 必须是 JSON 对象。", "QualityGateError")
    return body


def _standard_quality_allows_downstream(report: dict[str, Any]) -> bool:
    return report.get("status") == "pass" and report.get("downstream_allowed") is True


def _ensure_ocr_token_for_real_ocr(ocr_fixture_path: Path | None) -> None:
    if ocr_fixture_path:
        return
    if os.getenv("GLM_OCR_API_KEY") or os.getenv("ZAI_API_KEY") or os.getenv("ZHIPUAI_API_KEY"):
        return
    if not sys.stdin.isatty():
        raise ConsistencyPipelineError(
            "package_image_comparison",
            "缺少 GLM-OCR token。请设置 GLM_OCR_API_KEY 环境变量，或在交互式终端运行后按提示输入。",
            "MissingOcrTokenError",
        )
    token = getpass.getpass("请输入 GLM_OCR_API_KEY（输入不会显示，且只在本次运行中使用）：").strip()
    if not token:
        raise ConsistencyPipelineError("package_image_comparison", "未输入 OCR token，已停止包装图文字识别。", "MissingOcrTokenError")
    os.environ["GLM_OCR_API_KEY"] = token


def _base_summary(
    *,
    standard_path: Path,
    image_path: Path,
    output_dir: Path,
    standard_manifest_path: Path,
    ocr_fixture_path: Path | None,
    generated_manifest: bool,
    created_at: str,
) -> dict[str, Any]:
    return {
        "artifact_version": "package_consistency_pipeline_summary_v0.1",
        "run_id": f"pkg_consistency_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "status": "running",
        "comparison_status": None,
        "created_at": created_at,
        "completed_at": None,
        "duration_seconds": None,
        "inputs": {
            "standard": _file_item("standard", standard_path),
            "package_image": _file_item("package_image", image_path),
            "standard_manifest": _file_item("standard_manifest", standard_manifest_path),
            "standard_manifest_generated": generated_manifest,
            "ocr_fixture": _file_item("ocr_fixture", ocr_fixture_path) if ocr_fixture_path else None,
        },
        "stages": {
            "standard_structure": {"status": "not_started", "artifacts_dir": str(output_dir / "standard_structure")},
            "package_image_comparison": {"status": "not_started", "artifacts_dir": str(output_dir)},
            "publish": {"status": "not_started", "artifacts_dir": str(output_dir / "artifacts" / "06_publish")},
        },
        "key_artifacts": _key_artifacts(output_dir),
    }


def _stage_running(artifacts_dir: Path, started_at: str) -> dict[str, Any]:
    return {
        "status": "running",
        "artifacts_dir": str(artifacts_dir),
        "started_at": started_at,
        "completed_at": None,
        "duration_seconds": None,
    }


def _stage_success(
    *,
    artifacts_dir: Path,
    started_at: str,
    duration_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "pass",
        "artifacts_dir": str(artifacts_dir),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": duration_seconds,
    }
    if extra:
        payload.update(extra)
    return payload


def _with_stage_failure(
    summary: dict[str, Any],
    stage: str,
    error_type: str,
    reason: str,
    *,
    started_at: str | None = None,
    duration_seconds: float | None = None,
    not_started: bool = True,
) -> dict[str, Any]:
    updated = dict(summary)
    updated["status"] = "failed"
    updated["failed_stage"] = stage
    stages = dict(updated.get("stages") or {})
    current = dict(stages.get(stage) or {})
    current.update({"status": "failed", "error_type": error_type, "reason": reason})
    current.setdefault("started_at", started_at)
    current["completed_at"] = _utc_now()
    current["duration_seconds"] = duration_seconds
    stages[stage] = current
    if not_started:
        for name, stage_payload in stages.items():
            if name != stage and stage_payload.get("status") == "running":
                stage_payload["status"] = "not_started"
    updated["stages"] = stages
    updated["key_artifacts"] = summary.get("key_artifacts", {})
    return updated


def _complete_summary(summary: dict[str, Any], started_monotonic: float) -> dict[str, Any]:
    updated = dict(summary)
    updated["completed_at"] = _utc_now()
    updated["duration_seconds"] = _elapsed_seconds(started_monotonic)
    return updated


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _elapsed_seconds(started_monotonic: float) -> float:
    return round(time.perf_counter() - started_monotonic, 3)


def _quality_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "downstream_allowed": report.get("downstream_allowed"),
        "issue_count": report.get("issue_count"),
        "overall_status": report.get("overall_status"),
    }


def _key_artifacts(output_dir: Path) -> dict[str, str]:
    return {
        "standard_items": str(output_dir / "standard_structure" / "standard_items.json"),
        "standard_quality_report": str(output_dir / "standard_structure" / "quality_report.json"),
        "comparison_result": str(output_dir / "comparison_result.json"),
        "html_report": str(output_dir / "result_preview.html"),
        "pipeline_summary": str(output_dir / "pipeline_summary.json"),
    }


def _failure_result(stage: str, error_type: str, reason: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "stage": stage,
        "error_type": error_type,
        "reason": reason,
    }


def _failure_summary(args: argparse.Namespace, stage: str, error_type: str, reason: str) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    return {
        "artifact_version": "package_consistency_pipeline_summary_v0.1",
        "run_id": f"pkg_consistency_failed_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "status": "failed",
        "failed_stage": stage,
        "comparison_status": None,
        "created_at": _utc_now(),
        "completed_at": _utc_now(),
        "duration_seconds": None,
        "inputs": {
            "standard": _maybe_file_item("standard", args.standard),
            "package_image": _maybe_file_item("package_image", args.image),
            "standard_manifest": None,
            "standard_manifest_generated": True,
            "ocr_fixture": _maybe_file_item("ocr_fixture", args.ocr_fixture),
        },
        "stages": {
            "standard_structure": {
                "status": "failed" if stage == "standard_structure" else "not_started",
                **({"error_type": error_type, "reason": reason} if stage == "standard_structure" else {}),
            },
            "package_image_comparison": {
                "status": "failed" if stage == "package_image_comparison" else "not_started",
                **({"error_type": error_type, "reason": reason} if stage == "package_image_comparison" else {}),
            },
            "publish": {
                "status": "failed" if stage == "publish" else "not_started",
                **({"error_type": error_type, "reason": reason} if stage == "publish" else {}),
            },
        },
        "key_artifacts": _key_artifacts(output_dir),
    }


def _existing_path(path: Path | None, stage: str) -> Path:
    if path is None:
        raise ConsistencyPipelineError(stage, "缺少必需文件路径。", "FileNotFoundError")
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ConsistencyPipelineError(stage, f"文件不存在：{resolved}", "FileNotFoundError")
    return resolved


def _file_item(role: str, path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _maybe_file_item(role: str, path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    return _file_item(role, resolved)


def _write_artifact_index(output_dir: Path) -> None:
    index_path = output_dir / "artifacts" / "index.json"
    suffixes = {".json", ".html", ".png", ".jpg", ".jpeg", ".md"}
    artifacts = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.suffix.lower() in suffixes):
        if path == index_path:
            continue
        artifacts.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(index_path, {"artifact_count": len(artifacts), "artifacts": artifacts})


if __name__ == "__main__":
    raise SystemExit(main())
