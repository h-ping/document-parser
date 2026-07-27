# 多平台安装与兼容性

保持一个权威 skill 目录，不要为不同 Agent 平台维护多份不同工作流。需要平台专属入口时，复制或软链同一个 `check-package-consistency` 目录，并让 `SKILL.md` 保持一致。

## Codex

- 仓库级：`.agents/skills/check-package-consistency/SKILL.md`
- 用户级：`${CODEX_HOME:-~/.codex}/skills/check-package-consistency/SKILL.md`
- `agents/openai.yaml` 只作为 Codex UI 元数据，其他平台可忽略。

## Claude Code

- 项目级：`.claude/skills/check-package-consistency/SKILL.md`
- 用户级：`~/.claude/skills/check-package-consistency/SKILL.md`
- 推荐复制或软链本目录。不要新增 Claude 专用动态 shell 注入，避免不同平台执行行为分叉。

## Hermes Agent

- 默认目录：`~/.hermes/skills/check-package-consistency/SKILL.md`
- 也可通过 Hermes 外部 skill 目录加载本仓库的 `.agents/skills` 或导出的 skill 目录。
- Hermes 支持额外 frontmatter，但本 skill 为跨平台兼容只使用 `name` 和 `description`。

## OpenClaw

- 工作区：`<workspace>/skills/check-package-consistency/SKILL.md`
- 项目 Agent：`<workspace>/.agents/skills/check-package-consistency/SKILL.md`
- 个人 Agent：`~/.agents/skills/check-package-consistency/SKILL.md`
- 托管本机：`~/.openclaw/skills/check-package-consistency/SKILL.md`

当前仓库位置 `.agents/skills/check-package-consistency` 可被 OpenClaw 作为项目 Agent skill 发现。

如需 OpenClaw gating，可在发布版 frontmatter 加 `metadata.openclaw.requires.bins`，但不要强制要求 `PPOCRV6_API_KEY`：离线 `--ocr-fixture` 不需要密钥，真实 OCR 也可能由运行时环境注入。
