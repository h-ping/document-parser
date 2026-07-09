# 包装标签文字一致性报告 CLI

本项目提供一个对外命令：`check-package-consistency`。

输入一份标准模板 Excel 和一张包装设计图，输出客户可读的一致性检查报告。对外入口只支持标准模板 Excel，不开放非模板标准文档能力。

## 安装

```bash
python3 -m pip install -e .
```

安装后确认命令可用：

```bash
check-package-consistency --help
```

## 环境变量

真实识别包装图文字时需要配置 PP-OCRv6 密钥：

```bash
PPOCRV6_API_KEY=...
```

也兼容 `PPOCRV6_TOKEN`。如供应商地址或模型名变更，可选配置：

```bash
PPOCRV6_API_URL=...
PPOCRV6_MODEL=PP-OCRv6
```

## 使用

```bash
check-package-consistency \
  --standard path/to/standard.xlsx \
  --image path/to/package.jpg \
  --output-dir out/package_consistency_report
```

离线回归或测试时，可以传入已录制的包装图文字识别结果；这种模式不需要配置 OCR 密钥：

```bash
check-package-consistency \
  --standard path/to/standard.xlsx \
  --image path/to/package.jpg \
  --ocr-fixture test_ocr/zongzi_ocr_result.json \
  --output-dir out/package_consistency_report
```

## 输入限制

- 标准文档：只支持标准模板 Excel，后缀必须是 `.xlsx`
- 包装设计图：支持 `.png`、`.jpg`、`.jpeg`
- 默认按单张完整包装设计图处理；如果只上传局部图，标准模板中存在但局部图未出现的字段会被判为缺失

如果 `--standard` 不是 `.xlsx`，流程会在标准结构化阶段停止，不会继续检查包装图。

## 输出目录

输出目录会包含：

- `result_preview.html`：客户可读的一致性报告
- `comparison_result.json`：字段级一致性结果
- `package_ocr_lines.json`：包装图文字识别行
- `package_ocr_quality_report.json`：文字识别位置质量报告
- `pipeline_summary.json`：两阶段状态、运行 ID、输入文件、关键产物和耗时
- `failure_result.json`：失败时输出，记录失败阶段和原因
- `standard_structure/`：标准模板结构化产物，包括 `standard_items.json`、`tables.json`、`field_groups.json`、`quality_report.json`

## 质量门禁

- 标准模板结构化失败时停止，不继续跑包装图比对
- `standard_structure/quality_report.json` 中 `status != "pass"` 或 `downstream_allowed != true` 时停止
- 包装图多印的标准外文字默认只提示，不作为严重问题

## 结果状态

- `pass`：通过
- `critical_missing`：标准模板有该字段，但包装图未找到对应文字
- `critical_mismatch`：标准模板文字和包装图文字不一致
- `manual_review`：需要人工复核
- `info_extra_text`：包装图多出文字提示

## 验证

```bash
PYTHONPATH=src /tmp/document-parser-venv/bin/python -m unittest tests.test_consistency_cli
PYTHONPATH=src /tmp/document-parser-venv/bin/python -m unittest discover -s tests
```
