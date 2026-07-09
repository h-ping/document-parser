from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import sha256_file, write_json


DEFAULT_COS_CONFIG_PATH = Path.home() / ".config" / "packaging-consistency-check" / "secrets.env"
COS_ENV_KEYS = {
    "PACKAGING_COS_SECRET_ID",
    "PACKAGING_COS_SECRET_KEY",
    "PACKAGING_COS_BUCKET_URL",
    "PACKAGING_COS_CDN_DOMAIN",
    "PPOCRV6_TOKEN",
}
SECRET_KEY_PARTS = ("secret", "token", "password", "authorization")
PATH_KEY_PARTS = ("path", "dir", "file")


class CosPublishError(RuntimeError):
    pass


def load_local_runtime_env(path: Path | None = None) -> None:
    config_path = (path or DEFAULT_COS_CONFIG_PATH).expanduser()
    if not config_path.exists():
        return
    for key, value in _parse_env_file(config_path).items():
        if key in COS_ENV_KEYS and key not in os.environ:
            os.environ[key] = value


def publish_report_to_cos(
    *,
    output_dir: Path,
    run_id: str,
    key_prefix_template: str | None = None,
    dry_run: bool = False,
    config_path: Path | None = None,
) -> dict[str, Any]:
    load_local_runtime_env(config_path)
    publish_dir = output_dir / "artifacts" / "06_publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    result_path = publish_dir / "cos_upload_result.json"
    errors_path = publish_dir / "cos_upload_errors.json"

    try:
        cos_config = _load_cos_config()
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        key_prefix = _render_key_prefix(key_prefix_template or "document-parser/{run_id}/{timestamp}", run_id, timestamp)
        public_url = f"{cos_config['cdn_domain']}/{key_prefix}/result_preview.html"
        publish_info = {
            "enabled": True,
            "provider": "tencent_cos",
            "status": "pending",
            "public_url": public_url,
            "bucket": cos_config["bucket"],
            "region": cos_config["region"],
            "key_prefix": key_prefix,
            "dry_run": dry_run,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json(publish_dir / "publish_manifest.json", redact_data({"config": cos_config, "publish": publish_info}))
        bundle_dir, files = _build_public_bundle(output_dir, publish_dir, publish_info)
        offenders = _bundle_sensitive_path_offenders(bundle_dir)
        write_json(
            publish_dir / "public_bundle_manifest.json",
            {
                "bundle_dir": str(bundle_dir),
                "file_count": len(files),
                "total_bytes": sum(int(item.get("size") or 0) for item in files),
                "files": files,
                "sensitive_path_offenders": offenders,
            },
        )
        if offenders:
            raise CosPublishError(f"公开报告包仍包含本地路径或敏感字段：{', '.join(offenders[:10])}")
        upload_result = _upload_bundle(bundle_dir, files, cos_config, key_prefix, dry_run=dry_run)
        verification = _verify_public_url(public_url, dry_run=dry_run)
        status = "success" if upload_result.get("status") == "success" and verification.get("status") in {"success", "skipped"} else "failed"
        final = {
            **publish_info,
            "status": status,
            "uploaded_file_count": upload_result.get("uploaded_file_count", 0),
            "uploaded_total_bytes": upload_result.get("uploaded_total_bytes", 0),
            "verification_status": verification.get("status"),
            "verification": verification,
            "result_path": str(result_path),
            "errors_path": str(errors_path),
        }
        write_json(result_path, final)
        write_json(errors_path, {"errors": []})
        return final
    except Exception as exc:
        failed = {
            "enabled": True,
            "provider": "tencent_cos",
            "status": "failed",
            "public_url": None,
            "error": str(exc),
            "dry_run": dry_run,
            "result_path": str(result_path),
            "errors_path": str(errors_path),
        }
        write_json(result_path, failed)
        write_json(errors_path, {"errors": [str(exc)]})
        raise CosPublishError(str(exc)) from exc


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _load_cos_config() -> dict[str, str]:
    secret_id = os.getenv("PACKAGING_COS_SECRET_ID")
    secret_key = os.getenv("PACKAGING_COS_SECRET_KEY")
    bucket_url = os.getenv("PACKAGING_COS_BUCKET_URL")
    cdn_domain = os.getenv("PACKAGING_COS_CDN_DOMAIN")
    missing = [
        key
        for key, value in (
            ("PACKAGING_COS_SECRET_ID", secret_id),
            ("PACKAGING_COS_SECRET_KEY", secret_key),
            ("PACKAGING_COS_BUCKET_URL", bucket_url),
            ("PACKAGING_COS_CDN_DOMAIN", cdn_domain),
        )
        if not value
    ]
    if missing:
        raise CosPublishError(f"COS 配置缺失：{', '.join(missing)}")
    bucket, region = _parse_bucket_url(str(bucket_url))
    return {
        "secret_id": str(secret_id),
        "secret_key": str(secret_key),
        "bucket": bucket,
        "region": region,
        "cdn_domain": str(cdn_domain).rstrip("/"),
    }


def _parse_bucket_url(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    bucket = (query.get("bucket") or [""])[0]
    region = (query.get("region") or [""])[0]
    if not bucket or not region:
        raise CosPublishError("PACKAGING_COS_BUCKET_URL 必须包含 bucket 和 region 查询参数")
    return bucket, region


def _render_key_prefix(template: str, run_id: str, timestamp: str) -> str:
    value = template.format(run_id=run_id, timestamp=timestamp)
    value = re.sub(r"[^A-Za-z0-9._/-]+", "-", value)
    return value.strip("/") or f"document-parser/{run_id}/{timestamp}"


def _build_public_bundle(output_dir: Path, publish_dir: Path, publish_info: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    bundle_dir = publish_dir / "public_bundle"
    if bundle_dir.exists():
        import shutil

        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _write_redacted_text(output_dir / "result_preview.html", bundle_dir / "result_preview.html")
    for image_path in sorted(output_dir.glob("package_image.*")):
        if image_path.is_file():
            (bundle_dir / image_path.name).write_bytes(image_path.read_bytes())
            break
    for name in ("comparison_result.json", "package_ocr_quality_report.json"):
        src = output_dir / name
        if src.exists():
            write_json(bundle_dir / name, redact_data(json.loads(src.read_text(encoding="utf-8"))))
    summary_path = output_dir / "pipeline_summary.json"
    if summary_path.exists():
        summary = redact_data(json.loads(summary_path.read_text(encoding="utf-8")))
        if isinstance(summary, dict):
            summary["publish"] = redact_data(publish_info)
            summary.setdefault("key_artifacts", {})
            if isinstance(summary["key_artifacts"], dict):
                summary["key_artifacts"]["published_report_html"] = publish_info.get("public_url")
        write_json(bundle_dir / "pipeline_summary.public.json", summary)

    files = _bundle_files(bundle_dir)
    write_json(
        bundle_dir / "artifacts" / "index.public.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "publish": redact_data(publish_info),
            "files": files,
        },
    )
    return bundle_dir, _bundle_files(bundle_dir)


def _write_redacted_text(src: Path, dst: Path) -> None:
    if not src.exists():
        raise CosPublishError(f"缺少待发布文件：{src.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(redact_text(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")


def _bundle_files(bundle_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(bundle_dir).as_posix()
            files.append(
                {
                    "path": rel,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "content_type": _content_type_for(path),
                }
            )
    return files


def _upload_bundle(bundle_dir: Path, files: list[dict[str, Any]], cos_config: dict[str, str], key_prefix: str, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "success",
            "dry_run": True,
            "uploaded_file_count": len(files),
            "uploaded_total_bytes": sum(int(item.get("size") or 0) for item in files),
            "uploaded_files": files,
        }
    from qcloud_cos import CosConfig, CosS3Client  # type: ignore

    config = CosConfig(Region=cos_config["region"], SecretId=cos_config["secret_id"], SecretKey=cos_config["secret_key"], Scheme="https")
    client = CosS3Client(config)
    uploaded = []
    for item in files:
        rel = str(item["path"])
        local_path = bundle_dir / rel
        key = f"{key_prefix}/{rel}".strip("/")
        client.upload_file(
            Bucket=cos_config["bucket"],
            LocalFilePath=str(local_path),
            Key=key,
            PartSize=1,
            MAXThread=4,
            EnableMD5=False,
            ContentType=item.get("content_type") or _content_type_for(local_path),
        )
        uploaded.append({**item, "key": key})
    return {
        "status": "success",
        "dry_run": False,
        "uploaded_file_count": len(uploaded),
        "uploaded_total_bytes": sum(int(item.get("size") or 0) for item in uploaded),
        "uploaded_files": uploaded,
    }


def _verify_public_url(public_url: str, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "skipped", "reason": "dry_run"}
    last_error = ""
    for _ in range(3):
        try:
            req = urllib.request.Request(public_url, headers={"User-Agent": "document-parser/0.1"})
            with urllib.request.urlopen(req, timeout=12) as response:
                body = response.read(4096).decode("utf-8", errors="ignore")
                ok = response.status == 200 and ("一致性" in body or "<html" in body.lower())
                return {"status": "success" if ok else "failed", "http_status": response.status}
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2)
    return {"status": "failed", "error": last_error}


def _content_type_for(path: Path) -> str:
    if path.suffix.lower() == ".html":
        return "text/html; charset=utf-8"
    if path.suffix.lower() == ".json":
        return "application/json; charset=utf-8"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _bundle_sensitive_path_offenders(bundle_dir: Path) -> list[str]:
    offenders = []
    patterns = (
        "/Users/",
        "/private/tmp/",
        "/tmp/",
        "/var/folders/",
        "file://",
        "SecretId",
        "SecretKey",
        "PACKAGING_COS_SECRET",
        "PPOCRV6_API_KEY",
        "PPOCRV6_TOKEN",
    )
    for path in bundle_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".html", ".json", ".txt", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in patterns):
            offenders.append(path.relative_to(bundle_dir).as_posix())
    return offenders


def redact_data(data: Any) -> Any:
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            if any(part in key_lower for part in SECRET_KEY_PARTS):
                result[key] = "***"
            elif isinstance(value, str) and any(part in key_lower for part in PATH_KEY_PARTS) and _looks_like_local_path(value):
                result[key] = "[local_path_redacted]"
            else:
                result[key] = redact_data(value)
        return result
    if isinstance(data, list):
        return [redact_data(item) for item in data]
    if isinstance(data, str):
        return redact_text(data)
    return data


def redact_text(text: str) -> str:
    text = re.sub(r"/Users/[^\"'\n\r\t<]+", "[local_path_redacted]", text)
    text = re.sub(r"/private/tmp/[^\"'\n\r\t<]+", "[local_path_redacted]", text)
    text = re.sub(r"/tmp/[^\"'\n\r\t<]+", "[local_path_redacted]", text)
    text = re.sub(r"/var/folders/[^\"'\n\r\t<]+", "[local_path_redacted]", text)
    text = re.sub(r"file://[^\"'\n\r\t<]+", "[local_file_url_redacted]", text)
    return text


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "file://")) and not value.startswith(("http://", "https://"))
