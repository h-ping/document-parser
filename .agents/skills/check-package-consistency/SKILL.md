---
name: check-package-consistency
description: Bootstrap and run the document-parser `check-package-consistency` CLI for packaging label text consistency checks, publishing the customer report to Tencent COS by default. Use when comparing a standard template Excel (.xlsx) with a package artwork image (.png/.jpg/.jpeg), installing or locating the CLI, configuring COS publish, generating or explaining `result_preview.html`, `comparison_result.json`, `pipeline_summary.json`, `publish.public_url`, using an OCR fixture, or diagnosing failures such as `UnsupportedStandardInputError`, `MissingOcrTokenError`, `CosPublishError`, and quality gate failures. 中文触发词：包装文字一致性校验、标准模板 Excel 和包装图比对、默认发布到 COS、安装 check-package-consistency、包装设计稿文案核对、客户报告生成。
---

# 包装文字一致性检查

使用 `check-package-consistency` CLI 作为事实源。能运行 CLI 时，不要手工推断包装图文字是否匹配标准。

## 适用边界

- 标准输入只接受标准模板 Excel：`.xlsx`。如果用户给 PDF/Word/TXT，说明当前 CLI 不直接支持，并要求提供模板 Excel；只有用户明确要求时才先做上游转换。
- 包装设计图只接受 `.png`、`.jpg`、`.jpeg`。
- CLI 一次处理一份标准模板和一张包装图。多组文件要分别运行，并使用独立输出目录。
- 真实 OCR 默认使用 hybrid 双引擎，需要 `GLM_OCR_API_KEY`（也兼容 `ZAI_API_KEY`/`ZHIPUAI_API_KEY`）和 `PPOCRV6_API_KEY`（也兼容 `PPOCRV6_TOKEN`）。离线回归可用 fixture，不需要 OCR 密钥。
- 正式任务默认发布到腾讯云 COS，命令必须带 `--publish-cos`。只有用户明确要求本地报告、离线验证或测试发布包时，才使用 `--cos-dry-run` 或不发布。
- 不要打印、记录或复述 OCR token 或 COS 密钥。不要把密钥放进命令参数、报告目录、日志或最终回复。
- 以 CLI 生成的 JSON/HTML 产物为审核依据；LLM/VLM 只能辅助解释，不能替代 OCR 或比对结果。

## 快速入口

不要假设用户已经安装 `document-parser`。先初始化 CLI，并把后续命令中的 `check-package-consistency` 替换为初始化输出里的 `cli`。

优先运行 skill 自带脚本。将脚本路径解析为本 `SKILL.md` 所在目录下的 `scripts/ensure_cli.py`：

```bash
python3 scripts/ensure_cli.py
```

脚本固定执行以下环境检查：

1. 优先检查 `DOCUMENT_PARSER_SOURCE_DIR`、当前工作目录、skill 所在仓库、`~/workspace/document-parser` 是否存在且像源码项目。
2. 如果找到源码项目，创建/复用该项目的 `.venv`，安装 `-e`，再验证 CLI。
3. 如果未找到源码项目，检查当前环境是否支持 `check-package-consistency --help`。
4. 如果 PATH 上也没有可用 CLI，创建/复用 `~/workspace/.document-parser-cli-venv`，从 GitHub 安装，再验证 CLI。
5. 验证 CLI 帮助中包含 `--publish-cos`、`--cos-dry-run`、`--ocr-mode`、`--llm-mode`、`--ppocr-fixture` 和 `--glm-ocr-fixture`；缺少这些参数视为旧版本，不可用于本 skill。

成功输出示例：

```json
{"status":"ready","source":"skill_project","cli":"/path/to/document-parser/.venv/bin/check-package-consistency","message":"initialized from /path/to/document-parser"}
```

如果不能运行 skill 脚本，按同样顺序手工执行：

```bash
DOCUMENT_PARSER_SOURCE_DIR=/path/to/document-parser python3 scripts/ensure_cli.py
cd /path/to/document-parser
python3 -m venv .venv
".venv/bin/python" -m pip install -e .
".venv/bin/check-package-consistency" --help
```

```bash
check-package-consistency --help
mkdir -p "$HOME/workspace"
python3 -m venv "$HOME/workspace/.document-parser-cli-venv"
"$HOME/workspace/.document-parser-cli-venv/bin/python" -m pip install --upgrade git+https://github.com/h-ping/document-parser.git
"$HOME/workspace/.document-parser-cli-venv/bin/check-package-consistency" --help
```

成功后只使用已验证过的 CLI 路径，例如：

```bash
/Users/name/workspace/.document-parser-cli-venv/bin/check-package-consistency
```

如果网络不可用、`python3` 不可用，或平台策略禁止安装，停止并说明需要用户提供可运行的 `check-package-consistency` 路径、`DOCUMENT_PARSER_SOURCE_DIR` 指向的源码目录、`~/workspace/document-parser` 源码目录，或允许安装依赖。

## 执行流程

1. 预检输入路径存在，后缀符合边界，输出目录不会覆盖用户仍要保留的报告。
2. 确认真实 OCR 是否同时有 GLM-OCR 和 PP-OCR 密钥。非交互环境缺少密钥时，不要盲跑真实 OCR；要求用户提供环境变量，或改用离线 fixture。
3. 确认 COS 发布配置存在。默认读取 `~/.config/packaging-consistency-check/secrets.env`，也可使用同名环境变量或 `--cos-config /path/to/secrets.env`。
4. COS 配置必须包含：

```text
PACKAGING_COS_SECRET_ID=...
PACKAGING_COS_SECRET_KEY=...
PACKAGING_COS_BUCKET_URL=...?bucket=your-bucket&region=ap-guangzhou
PACKAGING_COS_CDN_DOMAIN=https://your-cdn-domain
```

5. 运行正式检查并默认发布 COS：

```bash
check-package-consistency \
  --standard /path/to/standard.xlsx \
  --image /path/to/package.jpg \
  --output-dir /path/to/report_dir \
  --publish-cos
```

离线 PP-OCR 回归但仍发布 COS。旧参数 `--ocr-fixture` 是 `--ppocr-fixture` 的兼容别名，会按 PP-OCR 单引擎运行：

```bash
check-package-consistency \
  --standard /path/to/standard.xlsx \
  --image /path/to/package.jpg \
  --ocr-fixture /path/to/recorded_ocr.json \
  --output-dir /path/to/report_dir \
  --publish-cos
```

离线 hybrid 回归但仍发布 COS：

```bash
check-package-consistency \
  --standard /path/to/standard.xlsx \
  --image /path/to/package.jpg \
  --ppocr-fixture /path/to/recorded_ppocr.json \
  --glm-ocr-fixture /path/to/recorded_glm_ocr.json \
  --output-dir /path/to/report_dir \
  --publish-cos
```

只生成脱敏公开发布包，不实际上传：

```bash
check-package-consistency \
  --standard /path/to/standard.xlsx \
  --image /path/to/package.jpg \
  --ocr-fixture /path/to/recorded_ocr.json \
  --output-dir /path/to/report_dir \
  --publish-cos \
  --cos-dry-run
```

6. 运行失败也要检查输出目录。优先读取 `failure_result.json` 和 `pipeline_summary.json`，不要只看 stderr。
7. 运行成功后读取 `pipeline_summary.json`、`comparison_result.json`，并确认 `publish.public_url` 或 `key_artifacts.published_report_html` 存在。
8. 最终回复只给审核结论、关键计数、COS 公开报告链接、失败阶段或需人工复核点。

## 本地报告例外

仅当用户明确说“不发布 COS”“本地检查”“dry-run”或“离线测试发布包”时，才改用本地报告流程：

```bash
check-package-consistency \
  --standard /path/to/standard.xlsx \
  --image /path/to/package.jpg \
  --output-dir /path/to/report_dir
```

## 旧版命令

不要把下面这种不带 `--publish-cos` 的命令作为正式默认流程：

```bash
check-package-consistency \
  --standard /path/to/standard.xlsx \
  --image /path/to/package.jpg \
  --output-dir /path/to/report_dir
```

## 结果解释

关键产物：

- `result_preview.html`：客户可读一致性报告。
- `comparison_result.json`：字段级比对结果，含 `status`、`target_count`、`pass_count`、`critical_count`、`manual_review_count`、`info_extra_text_count`。
- `package_ppocr_lines.json`：PP-OCR 识别行，用于普通字段和精确位置。
- `package_glm_lines.json`：GLM-OCR 识别行，用于版面结构和营养表。
- `package_fusion_evidence.json`：双 OCR 融合证据来源。
- `package_fusion_quality_report.json`：双 OCR 融合质量提示。
- `pipeline_summary.json`：运行 ID、输入文件、阶段状态、耗时、关键产物路径。
- `pipeline_summary.json.publish.public_url`：COS 发布后的公开报告链接。
- `pipeline_summary.json.key_artifacts.published_report_html`：客户可打开的公开 HTML 报告链接。
- `failure_result.json`：失败时输出，含 `stage`、`error_type`、`reason`。
- `standard_structure/quality_report.json`：标准模板结构化质量门禁。
- `artifacts/06_publish/cos_upload_result.json`：COS 上传结果。
- `artifacts/06_publish/public_bundle/`：脱敏后的公开报告包。

状态含义：

- `pass`：通过。
- `critical_missing`：标准模板有字段，但包装图未找到对应文字。
- `critical_mismatch`：标准模板文字和包装图文字不一致。
- `manual_review`：需要人工复核。
- `info_extra_text`：包装图多出文字提示，默认不是严重问题。

## 失败处理

- `UnsupportedStandardInputError`：标准文件不是 `.xlsx`。停止，让用户提供标准模板 Excel。
- `MissingOcrTokenError`：真实 OCR 缺少密钥。hybrid 默认需要 GLM-OCR 和 PP-OCR 两类密钥；让用户通过环境变量提供，或提供离线 fixture。
- `QualityGateError` 或 `standard_structure` 阶段失败：标准模板结构化质量不允许下游比对。引用 `standard_structure/quality_report.json` 的状态和原因，不要继续包装图比对。
- `package_image_comparison` 阶段失败：检查包装图格式、OCR 配置、`package_ocr_quality_report.json` 和 `failure_result.json`。
- `CosPublishError` 或 `publish` 阶段失败：检查 COS 配置文件、同名环境变量、`PACKAGING_COS_BUCKET_URL` 是否包含 `bucket` 和 `region`，以及 `artifacts/06_publish/cos_upload_errors.json`。

## 多平台使用

本 skill 使用 Agent Skills 标准的 `SKILL.md` 目录格式。平台安装路径和迁移规则见 `references/platform-install.md`；只有在用户询问安装、迁移、发布或平台兼容性时才读取它。
