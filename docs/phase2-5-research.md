# Phase 2–5 调研：抄什么、不抄什么

调研时间 **2026-09-28**。目标：为 HTTP/WebSocket 服务层、沙箱、向量记忆、多 Agent、YAML 工作流、A2A、Web 控制台七个方向各找到业界最好的实现，并明确采纳哪些决策。

**三条改变方案的结论（先说）：**

1. **macOS 上沙箱的正解不是 Docker，是 Seatbelt（`sandbox-exec`）** —— 这是 Codex 和 Gemini CLI 实际在用的东西。Docker Desktop 在 macOS 上每条命令几秒开销，结果是开发期关掉它，然后它等于不存在。
2. **前端协议不要自己发明 WebSocket 格式，直接实现 AG-UI** —— 已经有开放标准，而且 LangGraph / Mastra / Pydantic AI / Microsoft Agent Framework 都在上面。自己造一套的收益是零，成本是永远追不上生态。
3. **多 Agent 对「编程」是负收益，对「研究」是正收益** —— Anthropic 自己的数据：多 Agent 比 chat 多用 **15×** tokens，且明确说编程任务**不适合**多 Agent。所以不能无条件加，必须做成**显式选择 + 预算闸门**。

---

## 一、服务层：AG-UI 而不是自造 WebSocket 协议

### 候选

| 方案 | 做法 | 判断 |
|---|---|---|
| 自造 WebSocket JSON | 最常见的选择 | **不采纳**。收益为零，且要自己解决重连、状态同步、HITL 表示 |
| SSE + 自定义事件名 | 比 WebSocket 简单 | 不采纳，理由同上 |
| **AG-UI**（CopilotKit + 多家框架） | 开放的事件协议，HTTP/WebSocket 传输，SSE 是常见承载 | **采纳** |

### AG-UI 的关键设计（照抄）

所有事件共享基础字段：`type` / `timestamp?` / `rawEvent?` / `metadata?` / `subagentRunId?`。

三类模式，覆盖了 Agent 前端的全部需求：

**① Start-Content-End（流式内容）**
```
RUN_STARTED        { threadId, runId, parentRunId?, input? }
STEP_STARTED       { stepName }
TEXT_MESSAGE_START { messageId, role }
TEXT_MESSAGE_CONTENT { messageId, delta }      ← 逐 token
TEXT_MESSAGE_END   { messageId }
TOOL_CALL_START    { toolCallId, toolCallName, parentMessageId? }
TOOL_CALL_ARGS     { toolCallId, delta }
TOOL_CALL_END      { toolCallId }
TOOL_CALL_RESULT   { messageId, toolCallId, content, role? }
REASONING_*        取代了已废弃的 THINKING_*
RUN_FINISHED       { outcome?, result? }
RUN_ERROR          { message, code? }
```

**② Snapshot-Delta（状态同步）**
```
STATE_SNAPSHOT     { snapshot }        ← 整体替换，不是合并
STATE_DELTA        { delta }           ← RFC 6902 JSON Patch
MESSAGES_SNAPSHOT  { messages }
ACTIVITY_SNAPSHOT  { messageId, activityType, content, replace? }
```
`activityType` 直接用 `"PLAN"` 就能把计划渲染成卡片——**这正是本项目的 PlanStep 需要的形状**。

**③ 中断（HITL）**

这个设计值得专门抄：中断**不**是单独的事件，而是 `RUN_FINISHED` 的 `outcome`：
```json
{ "type": "interrupt", "interrupts": [ ... ] }
```
恢复时**起一个新的 run**，在输入里带 `resume` 数组去寻址每个未关闭的中断。

> 为什么这个设计好：它让「暂停」和「结束」共用同一个生命周期出口，前端只需要处理一个终止事件，而不是「等 RUN_FINISHED，但如果它不来了呢」。

**采纳决定**：实现 AG-UI 兼容的编码器，`RUN_FINISHED.outcome.interrupts` 直接映射本项目的 `PendingConfirmation`。同时保留 REST 端点（REST 适合脚本和 A2A 内部调用，SSE 适合 UI）。

---

## 二、沙箱：macOS 用 Seatbelt，Docker 作为可选后端

### 隔离强度阶梯（调研数据）

| 隔离模型 | 启动时间 | 强度 | 开销 |
|---|---|---|---|
| 纯容器（Docker） | 50–200ms | 弱（共享内核，内核漏洞即逃逸） | 最低 |
| gVisor | 100–300ms | 强（用户态内核过滤 syscall） | 低–中 |
| Firecracker microVM | ~125ms–1s | **最强（自有内核）** | 中 |

反直觉的一点：microVM 现在启动几乎和容器一样快（~125ms），「VM 太慢做不了按请求隔离」这个老假设已经破了。

### 但本项目的场景不同

托管方案（E2B / Modal / Daytona）解决的是**云端跑不可信代码**。本项目是**开发者自己机器上的本地工具**，威胁模型不一样：

- 代码来源是**用户自己的仓库**（半可信）
- 真实威胁是 **prompt injection 把 Agent 变成攻击者的 shell**，而不是「用户故意攻击自己」
- 用户已经有一台机器，不想为每条 `pytest` 付 3 秒

**macOS 上的正确答案是 Seatbelt。** Codex、Gemini CLI 都在用。`sandbox-exec` 被 Apple 标记为 deprecated，但底层 Seatbelt 子系统完全可用，且是 macOS 标准的进程沙箱。

### 抄它的威胁模型（这句话是关键）

> **Threat model: anti-tampering, not anti-exfiltration.**
> 读是全开的；目标是阻止 Agent **修改**它不该改的东西，不是造一个滴水不漏的数据外泄边界。

这个取舍非常务实，而且理由充分：Agent 必须读代码、读配置、读工具链；如果读也锁死，Agent 就废了。而「写」是可以精确白名单的。

配套的**写白名单**（照抄它的分类）：
- 工作区目录（唯一可编辑的项目树）
- AI 配置目录（`~/.claude`、`~/.codex`、`~/.config/{openai,anthropic}`…）
- 缓存（`~/Library/Caches`、`~/.cache`、`~/.cargo`、`~/.npm`…）
- 工具链（`~/.nvm`、`~/.pyenv`、`~/go`…）
- 临时目录与设备（`/private/tmp`、`/dev/null`）

**只读**（可读不可写，让 git-over-ssh、AWS SDK 能用，但改不了）：
`~/.ssh`、`~/.aws`、`~/.gnupg`、`~/.docker`、`~/.kube`、`~/.config/gcloud`、`~/.netrc`、`~/.npmrc`、`~/Library/Keychains`。

**明确的坑（要写进文档，别让用户踩）：**
- 网络是放开的（Agent 要访问模型 API），**这不是出口防火墙**，被注入的命令仍然可以外联
- 登录类操作会失败（`npm login`、`gh auth login`、`aws configure` 都要**写**凭据文件）
- `brew install` 默认被挡（`/opt/homebrew` 不可写）
- Keychain 经 Mach IPC 的写入**挡不住**（挡了会破坏太多东西）
- `sandbox-exec` 已被 Apple 标记弃用；未来若移除，退化为无沙箱

**还有一个特别值得抄的东西**：它提供**只读模式**（`sb-ro`）——项目目录**含 `.git` 全部只读**。`git log/diff/show/blame` 能用，`commit/checkout/stash/fetch/pull` 全部失败。

这对本项目是直接可用的：**「分析这个项目」和「修改这个项目」应该是两个不同的沙箱档位**，而不是同一个权限配置。我会把它做成 `sandbox: read-only | workspace-write | full` 三档，正好对应「只读分析 / 正常开发 / 需要装依赖」。

**采纳决定**：实现可插拔沙箱后端，按优先级探测：
1. **Seatbelt**（macOS，首选，零开销）
2. **Docker**（跨平台，可选，明确告知开销）
3. **None**（无沙箱，但路径围栏 + 命令白名单仍然生效）

---

## 三、记忆：抄 Mem0 的抽取 + Zep 的失效语义，存储用 sqlite-vec

### 三家到底在做什么

| | Mem0 | Letta | Zep |
|---|---|---|---|
| 核心抽象 | 向量库里的**抽取式原子事实** | **可编辑记忆块** + 归档 + 召回 | **时序知识图谱** |
| 抽取 | 抽取器 LLM 提显著事实 | LLM 自主决策（记忆是工具） | 图抽取，事实存为带**有效期窗口的边** |
| 矛盾处理 | LLM judge 判 update/replace/add | 手动工具编辑 | **旧边失效而非覆盖** |
| 写延迟 p50 | ~80–200ms（异步） | ~150–400ms（同步 core 更新） | ~300–800ms（异步图抽取） |
| 读延迟 | 50–200ms | 50–200ms | 50–200ms |
| 基准 | LOCOMO 比强 RAG 基线 **+26%**，比塞全量历史**省约 90% token** | — | — |

### 关键洞察（这段话决定设计）

> 朴素 RAG 的问题：纯语义搜索检索出的结果噪声大，且**无法处理矛盾** —— 向量搜索会同时返回旧事实与新事实，除非框架主动调和。

这正是本项目 FTS5 记忆现在的状态。所以升级的重点**不是「加向量」，是「加矛盾处理」**。

**从 Zep 抄的那一条最有价值**：矛盾事实出现时，**旧边失效（invalidated）而非覆盖**，所以 Agent 仍然能回答「用户上个季度偏好什么？」

好消息：本项目的 `memories` 表**已经有 `superseded_by` 列**（当时是为了预留）。现在把它用起来——这就是 Zep 的 temporal invalidation 的轻量版，成本几乎为零。

### 向量存储选型

| | sqlite-vec | LanceDB |
|---|---|---|
| stars | 8.1k | 12k |
| 语言 | C | Rust |
| 定位 | **SQLite 里的向量搜索**（单文件，就是你已经在用的那个库） | 嵌入式向量数据库（列式 Lance 格式，可版本化，大于内存的检索） |
| 检索 | 精确暴力 KNN | 索引 + ANN |

**采纳 sqlite-vec 的定位，但先不引入扩展。** 理由：
- 个人记忆库的规模是**几百到几千条**，暴力余弦在 Python 里就是微秒级；引入 ANN 索引解决的是不存在的问题
- sqlite-vec 是 C 扩展，要按平台装二进制；LanceDB 是 Rust，要装 wheel。两者都给「零外部服务依赖」增加一个依赖
- 做法：向量存 BLOB，暴力检索用纯 Python。**留一个 `VectorIndex` 接口**，条目数超过阈值（比如 5 万）或用户显式开启时再接 sqlite-vec

**采纳决定**：
- 保留 FTS5（精确词 + 中文短词），**加**向量（语义），做**混合检索**
- 加 LLM 抽取：把对话/任务结果提炼成原子事实，而不是存原始文本
- 加 LLM judge 的 ADD / UPDATE / **INVALIDATE**（invalidate 优先于 overwrite，保时间线）
- 嵌入模型：优先复用用户的模型 provider（OpenAI 兼容的 `/embeddings`），**无网络时降级为哈希向量**（够用于去重和粗略聚类，不够用于语义搜索——文档里说清楚）

---

## 四、多 Agent：抄 Anthropic 的编排者-工作者，但加预算闸门

### 必须记住的数字

| 指标 | 数值 |
|---|---|
| Agent vs chat token | **约 4×** |
| **多 Agent vs chat token** | **约 15×** |
| 性能方差解释度 | 三因素解释 **95%**，其中 **token 用量单独解释 80%** |
| 多 Agent vs 单 Agent（Opus4 lead + Sonnet4 worker） | 内部评测 **+90.2%** |
| 并行化对研究时间的改善 | **最多 −90%** |

### 它对「编程任务」的结论（原话意思）

> 大多数编程任务中真正可并行的子任务**比研究少**；LLM Agent 目前**还不擅长**实时协调和委派。

多 Agent 真正擅长的：**高度并行 + 信息量超出单上下文窗口 + 需与大量复杂工具交互**。

**所以本项目不能无条件加多 Agent。** 采纳的形态是：**显式开启 + 预算闸门 + 角色可配**，默认关闭。

### 照抄的四条工程经验

1. **把规模规则写进提示词。** Agent 自己判断不了该投入多少。Anthropic 的规则表直接抄：

   | 任务复杂度 | 子 Agent 数 | 工具调用数 |
   |---|---|---|
   | 简单事实查找 | 1 | 3–10 |
   | 直接对比 | 2–4 | 每个 10–15 |
   | 复杂研究 | >10 | 职责清晰划分 |

   早期失败模式就是「给简单查询生成 50 个子 Agent」。

2. **教编排者怎么委派。** 每个子 Agent 的任务必须带：**目标 + 输出格式 + 工具/来源指引 + 清晰边界**。只给「研究半导体短缺」这种短指令，会出现多个子 Agent 做**完全相同的搜索**。

3. **子 Agent 的输出写文件系统，不要塞回对话历史。** 原文称之为避免「传话游戏（telephone game）」。这条对实现影响很大：子 Agent 的完整报告落盘，返回给编排者的只是**路径 + 摘要**。

4. **并行工具调用 + 并行子 Agent（3–5 个）** 是最大的性能杠杆（研究时间 −90%）。

### 失败模式（要在实现里防）

- 为简单查询生成 50 个子 Agent → **硬上限 + 规模规则**
- 无止境地搜索不存在的来源 → **每个子 Agent 的步数/成本预算**
- 子 Agent 用过多更新互相干扰 → **编排者是唯一写共享状态的人**
- 同步执行子 Agent 是瓶颈 → 首版可以同步（Anthropic 也这么做），但**留异步接口**

### 生产可靠性（正好对上本项目已有的东西）

Anthropic 说的挑战，本项目 Phase 1 已经解决了一半：
- 「Agent 有状态、错误会复利放大，不能从头重启」→ **本项目的事件溯源 + 写前台账就是解这个的**
- 「调试需要新方法，要全链路追踪」→ **本项目的 JSONL 事件流 + 事件表**
- 「部署需要 rainbow deployments」→ 暂不需要（本地单进程）

**采纳决定**：`orchestrator-worker` 模式，子 Agent = 独立 session 的完整 AgentRuntime（**天然上下文隔离**，这是多 Agent 唯一的真收益），并行度可配（默认 3），每个子 Agent 有独立预算，输出落盘。

---

## 五、YAML 工作流：抄 Dify 的 DSL 形状

### Dify DSL 的结构（照抄的部分）

```yaml
version: "0.1.0"
name: "Customer Support Bot"
app:
  mode: workflow          # 或 chatflow
graph:
  nodes:
    - id: start
      type: start
      data:
        title: "User Input"
        variables:
          - variable: query
            type: string
            required: true
    - id: llm_1
      type: llm
      data:
        title: "Generate Response"
        prompt_template: "{{#start.query#}}"
  edges:
    - source: start
      source_handle: source
      target: llm_1
      target_handle: target
```

### 三个值得抄的设计

1. **`{{#node_id.field#}}` 显式数据流。** 节点之间不共享可变状态，靠**显式引用上游节点的输出字段**。这比 LangGraph 的共享 TypedDict 状态更容易静态检查——**引用了不存在的节点或字段可以在加载时直接报错，而不是运行时 KeyError**。这是本项目要采纳的。

2. **`error_strategy` 挂在节点上**（不是全局）：
   ```yaml
   error_strategy:
     retry: { enabled: true, max_retries: 3 }
     error_handling: { error_branch_enabled: false }
   ```
   每个节点的失败语义不同，这是对的。

3. **节点类型划分**（入口 / 处理 / 控制流 / 输出）比 LangGraph 的「万物皆节点」更有指导性。

### 不抄的部分

- **40+ 节点类型**。Dify 是可视化产品，节点类型多是为了覆盖 UI 场景。本项目只需要：`start` / `agent` / `tool` / `ifelse` / `iteration` / `code` / `end`。**七个够了。**
- **`viewport` / UI 坐标**。那是给画布用的，本项目的工作流是手写 YAML。
- `app.mode: chatflow` 那套 UI 特性（开场白、建议问题）。

**采纳决定**：Dify 形状的 DSL（`version`/`name`/`graph{nodes,edges}`），`{{#node.field#}}` 变量引用，节点级 `error_strategy`，七种节点类型。**加载时做完整静态校验**：DAG 无环、变量引用可解析、节点类型已知、必填字段齐全。

---

## 六、A2A：实现 v1.0，别自造 Agent 间协议

### 事实

- Linux Foundation 治理（Google 捐赠），**2026-04 达到 v1.0**，150+ 组织，22k stars
- 与 MCP 互补而非竞争：**MCP 是 Agent 触达自己的工具；A2A 是 Agent 跨边界触达另一个 Agent**

### 要实现的形状

**Agent Card** 发布在 `/.well-known/agent-card.json`：
身份（名称/描述/提供方）、endpoint URL、**skills**（供编排器匹配的能力标签）、输入输出模态、支持的认证方案（广告，非强制）、extensions。

**5 个核心对象**：Task、Message（带 `role` + Part 数组）、Part（文本/文件/URL 引用/结构化 JSON）、Artifact、Agent Card。

**8 个任务状态**（严格有序）：
```
SUBMITTED → WORKING → INPUT_REQUIRED / AUTH_REQUIRED
                    → COMPLETED / FAILED / CANCELED / REJECTED
```
> 注意 `INPUT_REQUIRED` —— **本项目的 `waiting_confirmation` 正好映射到它**。这是 HITL 的跨组织标准表示。

**3 种传输绑定**（功能等价）：JSON-RPC 2.0（默认）、gRPC、HTTP+JSON/REST。SSE 是默认流式机制。

### 必须抄的安全项（规范里点名的风险）

| 风险 | 缓解 |
|---|---|
| **Webhook SSRF** | **webhook 主机 allowlist + 屏蔽私有 IP 段** |
| Agent 冒充 | 强制 Signed Agent Cards（JWS），固定可信签发方 |
| Card 篡改 | 每次获取验证 JWS，不缓存未签名卡片 |
| 重放 | 每请求 nonce + 时间戳，短生命周期 token |
| **上下文投毒** | **把 Card 字段当不可信输入，按 schema 验证** |

最后一条尤其重要：**别人给你的 Agent Card 是不可信输入**。`description` 字段里可以塞 prompt injection。本项目已经有 prompt-injection 检测器（技能安全审查那套），复用它。

**采纳决定**：实现 A2A v1.0 的 JSON-RPC 绑定 + Agent Card + SSE 流式 + webhook（带 SSRF 防护）。**不自造 `AgentMessage`**（原规格 §13.4 想自造，作废）。

---

## 七、Web 控制台：零构建，直接吃 AG-UI

### 选项

| 方案 | 判断 |
|---|---|
| Next.js + CopilotKit | **不采纳**。引入整条 npm 工具链，而本项目的核心卖点之一是「零外部服务依赖」 |
| Chainlit / Streamlit | 不采纳。会变成「另一个应用」，而不是这个运行时的控制台 |
| Open WebUI | 不采纳。它是独立产品，不是可嵌入的控制台 |
| **单页原生 JS，消费 AG-UI SSE** | **采纳** |

**理由**：AG-UI 的编码器已经写好（第一节），控制台就是一个 SSE 消费者 + 几个渲染函数。用原生 JS + CSS 可以做到**无构建步骤、无 npm 依赖**，和本项目其他部分保持一致。真要上 React 生态，AG-UI 兼容意味着随时能换成 CopilotKit 的组件，不用改后端。

控制台要展示的东西（都是本项目已经有的）：
- 流式回答（TEXT_MESSAGE_CONTENT）
- 计划卡片（ACTIVITY_SNAPSHOT with activityType=PLAN）
- 工具调用与结果（TOOL_CALL_*）
- **审批按钮**（RUN_FINISHED.outcome.interrupts → 调 approve/deny）
- 事件时间线（读 `/api/v1/tasks/{id}/events`）

---

## 八、实施顺序（按依赖，不按你列的顺序）

你列的七项里，有三项是**互相依赖**的，顺序错了会返工：

```
1. 事件总线 + 流式回调        ← 服务层和多 Agent 都要用，必须先做
2. 沙箱后端（Seatbelt 优先）   ← 独立，价值高，先做
3. AG-UI 编码器 + FastAPI     ← 依赖 1
4. 向量记忆 + 矛盾处理         ← 独立
5. 工作流引擎                  ← 依赖 1（节点要能跑 Agent）
6. 多 Agent                    ← 依赖 1 + 5（编排者本身可以是一个工作流节点）
7. A2A                        ← 依赖 3（复用 HTTP 层）+ 6（远程 Agent 就是一个子 Agent）
8. Web 控制台                  ← 依赖 3
```

**先做 1、2、3**：它们让整个系统第一次「可被外部使用」，也是后面所有东西的地基。

---

## 附：不采纳清单（明确记下来，免得以后又想做）

| 不采纳 | 理由 |
|---|---|
| 自造 WebSocket 事件格式 | AG-UI 已是开放标准，自己造永远追不上生态 |
| 自造 Agent 间消息协议 | A2A v1.0 已是 Linux Foundation 标准 |
| Dify 的 40+ 节点类型 | 那是可视化产品的需求；七个节点覆盖本项目全部场景 |
| LanceDB / Chroma / Milvus | 个人记忆库规模用暴力检索就够；引入它们是解决不存在的问题 |
| 无条件开启多 Agent | Anthropic 数据：15× token，且编程任务不适合 |
| Docker 作为 macOS 首选沙箱 | 每条命令数秒开销 → 开发期被关掉 → 等于不存在 |
| Letta 式「LLM 自主管理记忆」 | 让模型自己决定记什么，在编程 Agent 上噪声太大；外部管线更可控 |
| Next.js 控制台 | 破坏「零外部服务依赖」的核心卖点 |


---

## 实施状态（2026-09-28 当日完成）

按第八节的依赖顺序，**1 → 2 → 3 已完成**（事件总线 / 沙箱 / AG-UI + FastAPI + 控制台）。

| 方向 | 状态 | 落点 |
|---|---|---|
| 事件总线 + 流式回调 | ✅ | `observability/bus.py`，durable 事件与 ephemeral token 分流 |
| 沙箱后端 | ✅ | `sandbox/base.py`：Seatbelt（macOS 首选）/ Docker / None，**探测式可用性判定 + 回退必留痕** |
| AG-UI 编码器 | ✅ | `api/agui.py`，事件名与基础字段照规范 |
| FastAPI + SSE + WebSocket | ✅ | `api/app.py`，SSE 与 WS 共用同一个事件生成器 |
| Web 控制台 | ✅ | `api/console/`，零构建原生 JS，直接消费 AG-UI |
| 向量记忆 + 矛盾处理 | ⏸ | 设计已定（Mem0 抽取 + Zep 失效语义 + sqlite-vec 定位），未实现 |
| YAML 工作流 | ⏸ | 设计已定（Dify 形状 + `{{#node.field#}}` + 七种节点），未实现 |
| 多 Agent | ⏸ | 设计已定（orchestrator-worker + 规模规则 + 预算闸门），未实现 |
| A2A v1.0 | ⏸ | 设计已定（Agent Card + JSON-RPC + SSE + webhook SSRF 防护），未实现 |

### 实施中发现的三条修正

1. **Seatbelt 的可用性必须探测，不能靠「二进制存在」。** 在已被沙箱化的进程里，macOS **拒绝**安装一个更窄的 profile（`sandbox_apply: Operation not permitted`）。所以 `available` 会真的去应用一个限制性 profile 试试 —— 用 `(allow default)` 探测是无效的，那在任何环境都能通过，正好掩盖了要防的那种失败。
2. **沙箱不可用时 `wrap()` 静默直通，这是对的（不能让坏沙箱废掉所有命令），但必须由调用方出声音。** 于是有了 `SandboxSelection.warning()`，`uaa run`/`chat`/`serve`/`doctor` 都会打。否则用户配了 Seatbelt 却在裸跑。
3. **工具结果必须写进 `TOOL_COMPLETED` 事件，不能只放在 `LOG_APPENDED` 里。** 否则 UI 流出来的 `TOOL_CALL_RESULT` 是空的，`uaa task events` 也只能看到「某个工具跑了」而看不到它说了什么。
