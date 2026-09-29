/* unified-ai-agent console.
 *
 * Zero build step, zero npm. It consumes the AG-UI event stream over SSE and
 * renders it. Because the backend speaks AG-UI rather than a bespoke format,
 * this file can be replaced by CopilotKit's React components later without
 * touching the server.
 *
 * SSE via fetch + ReadableStream, not EventSource: EventSource can only do
 * GET, and AG-UI runs carry a request body.
 */

const $ = (id) => document.getElementById(id);

const state = {
  threadId: "console-" + Math.random().toString(36).slice(2, 8),
  taskId: null,
  // Per-run render targets, reset on each RUN_STARTED.
  current: null,
  _running: false,
  get running() {
    return this._running;
  },
  /* The send button is a property of "is a run going", not something to
   * remember at each of the six places that change it. Binding it here means
   * a new exit path cannot leave the button saying 发送 while a run streams. */
  set running(value) {
    this._running = value;
    setSendMode(value);
  },
};

/* The desktop launcher hands us a per-launch session token in the URL
 * fragment. A fragment is never sent to the server and never appears in a
 * Referer, so it cannot leak through the page itself -- which is the whole
 * reason it is not a query parameter. */
const TOKEN = (() => {
  const raw = location.hash.replace(/^#/, "");
  if (!raw) return "";
  // Keep it out of the address bar once we have it.
  history.replaceState(null, "", location.pathname + location.search);
  return raw;
})();

/* ---------------------------------------------------------------- helpers */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* The one argument worth showing on the collapsed line.
 *
 * A tool call is identified by what it acted on -- `read_file orders/calc.py`
 * -- not by its JSON. Showing the JSON is what made a run of eight reads take
 * eight screens. */
const ARG_KEYS = ["path", "command", "target", "url", "pattern", "query", "name", "id"];

function summariseArgs(raw) {
  if (!raw) return "";
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return ""; // still streaming; the full JSON is in the expanded body
  }
  if (!parsed || typeof parsed !== "object") return "";
  for (const key of ARG_KEYS) {
    const value = parsed[key];
    if (typeof value === "string" && value) {
      return value.length > 58 ? `${value.slice(0, 55)}…` : value;
    }
  }
  const first = Object.values(parsed).find((value) => typeof value === "string" && value);
  return first ? String(first).slice(0, 58) : "";
}

/* Follow the tail, but only once per frame, and never while the reader is
 * somewhere else.
 *
 * This used to run on every streamed token, reading `scrollHeight` each time --
 * a forced synchronous layout per token, which is the classic way to make a
 * streaming UI stutter. It also yanked the view back to the bottom while
 * someone was scrolling up to read what had already happened.
 */
let scrollQueued = false;

function scrollToBottom(force = false) {
  if (scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(() => {
    scrollQueued = false;
    const box = $("transcript");
    if (!box) return;
    const distance = box.scrollHeight - box.scrollTop - box.clientHeight;
    if (force || distance < 120) box.scrollTop = box.scrollHeight;
  });
}

function addTurn(kind, title) {
  $("empty")?.classList.add("hidden");
  const turn = el("div", `turn ${kind}`);
  const head = el("div", "turn-head");
  head.append(el("span", "", title));
  turn.append(head);
  $("transcript").append(turn);
  scrollToBottom(true);
  return turn;
}

function clearTranscript() {
  $("transcript").replaceChildren($("empty"));
  $("empty")?.classList.remove("hidden");
}

/* `state` is a key, not a label.
 *
 * The colour and the logic key off it, and the label is looked up separately.
 * Passing the display string here is how a translated interface loses all its
 * status colours: `[data-state="completed"]` stops matching the moment the
 * text becomes "已完成".
 */
function setStatus(state, busy = false) {
  const node = $("status");
  setText(node, STATUS_LABEL[state] || state);
  node.dataset.state = state;
  node.classList.toggle("busy", busy);

  // The header chip tracks the live state too. Otherwise it only moves when
  // the task list polls, so the header could say "运行中" for five seconds
  // after the run had already stopped -- and the two status lines on screen
  // would disagree.
  const chip = $("topbar-status");
  if (chip) {
    setClass(chip, `chip ${STATUS_TONE[state] || ""}`.trim());
    setText(chip, STATUS_LABEL[state] || state);
    setTitle(chip, state);
  }
}

/* Effects the UI may grant. SYSTEM_ADMIN is absent on purpose: the server
   refuses it too, and offering a control that always fails is worse than not
   offering it. */
const GRANTABLE = [
  "read_only",
  "write_local",
  "execute_local",
  "network",
  "external_side_effect",
];

function buildApprovalToggles() {
  const box = $("approve");
  box.replaceChildren();
  for (const effect of GRANTABLE) {
    const button = el("button", "toggle", effect);
    button.type = "button";
    // aria-pressed, not a class: the state has to be visible to a screen
    // reader and to the CSS, and a native multi-select could not say it at all.
    button.setAttribute("aria-pressed", "false");
    button.onclick = () => {
      const on = button.getAttribute("aria-pressed") === "true";
      button.setAttribute("aria-pressed", on ? "false" : "true");
    };
    box.append(button);
  }
}

function approvedEffects() {
  return [...$("approve").querySelectorAll('.toggle[aria-pressed="true"]')].map(
    (button) => button.textContent,
  );
}

const STATUS_TONE = {
  completed: "chip-ok",
  failed: "chip-danger",
  cancelled: "chip-danger",
  waiting_confirmation: "chip-warn",
  running: "chip-accent",
  planning: "chip-accent",
  pending: "chip-accent",
};

// Short label, full value in the tooltip. `waiting_confirmation` is a wire
// value; as a UI label it is both long and jargon, and it wrapped the row.
const STATUS_LABEL = {
  pending: "排队中",
  planning: "规划中",
  running: "运行中",
  waiting: "等待批准",
  waiting_confirmation: "等待批准",
  cancelling: "正在取消",
  resyncing: "重新同步",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  idle: "空闲",
};

const taskNodes = new Map();

function taskRow(task) {
  const item = el("div", "task-item");
  const goal = el("div", "task-goal");
  const meta = el("div", "task-meta");
  const chip = el("span", "chip");
  const stats = el("span", "task-stats");
  meta.append(chip, stats);
  item.append(goal, meta);
  item.onclick = async () => {
    state.taskId = task.id;
    clearTranscript();
    state.current = newRun();
    // Order matters: `attach` opens an SSE stream and does not resolve until
    // the stream ends, so anything awaited after it never runs. Read the
    // approval state first, then start streaming.
    await syncApproval(task.id);
    attach(task.id).catch((error) => console.warn("attach failed", error));
    refreshTasks();
  };
  return { item, goal, chip, stats };
}

/* Write only when the value actually changed.
 *
 * Assigning `textContent` replaces the text node even when the string is
 * identical, which is still a mutation: the browser drops and rebuilds the
 * node, and anything observing the subtree sees a change. Polling every five
 * seconds meant doing that to every row forever, for no reason -- and a
 * MutationObserver watching the list could never tell "nothing happened" from
 * "everything happened".
 */
function setText(node, value) {
  if (node.textContent !== value) node.textContent = value;
}

function setClass(node, value) {
  if (node.className !== value) node.className = value;
}

function setTitle(node, value) {
  if (node.title !== value) node.title = value;
}

function paintTask(node, task) {
  setText(node.goal, task.goal);
  setTitle(node.goal, task.goal);
  setClass(node.chip, `chip ${STATUS_TONE[task.status] || ""}`.trim());
  setText(node.chip, STATUS_LABEL[task.status] || task.status);
  setTitle(node.chip, task.status);
  setText(node.stats, `${task.steps_used} 步 · ${task.tokens_in + task.tokens_out} token`);
  node.item.classList.toggle("active", task.id === state.taskId);
}

/* The header says which task you are looking at and what it is doing. Without
 * it the content floated in the middle of a dark rectangle with nothing to
 * anchor it -- and no way to tell at a glance whether the thing on screen was
 * still running. */
function renderTopbar(task) {
  setText($("topbar-title"), task ? task.id : "控制台");
  setTitle($("topbar-title"), task ? task.goal : "");

  const chip = $("topbar-status");
  setClass(chip, `chip ${STATUS_TONE[task?.status] || ""}`.trim());
  setText(chip, task ? STATUS_LABEL[task.status] || task.status : STATUS_LABEL.idle);
  setTitle(chip, task?.status || "");

  const meta = $("topbar-meta");
  const wanted = task
    ? `${task.steps_used} 步 · ${task.tokens_in + task.tokens_out} token`
    : "";
  if (meta.textContent !== wanted) meta.textContent = wanted;
}

async function refreshTasks() {
  try {
    const tasks = await json("/api/v1/tasks?limit=15");
    const list = $("tasks");

    if (!tasks.length) {
      taskNodes.clear();
      list.replaceChildren(el("div", "task-empty", "还没有任务。"));
      $("task-count").textContent = "";
      return;
    }

    const ids = tasks.map((task) => task.id).join();
    if (ids !== [...taskNodes.keys()].join()) {
      // The *set* changed, so rebuild. Doing this on every poll -- which is
      // what it used to do -- destroys the element under the pointer every
      // five seconds: hover flickers, and a click lands on a node that is
      // already gone.
      const next = new Map();
      list.replaceChildren();
      for (const task of tasks) {
        const node = taskNodes.get(task.id) || taskRow(task);
        paintTask(node, task);
        list.append(node.item);
        next.set(task.id, node);
      }
      taskNodes.clear();
      for (const [id, node] of next) taskNodes.set(id, node);
    } else {
      for (const task of tasks) paintTask(taskNodes.get(task.id), task);
    }

    $("task-count").textContent = String(tasks.length);
    // The most recent task is the one a resume would target.
    if (!state.taskId) state.taskId = tasks[0].id;
    renderTopbar(tasks.find((task) => task.id === state.taskId) || null);
  } catch (error) {
    console.warn("task list unavailable", error);
  }
}

/* Show the approval panel for a task that is *already* waiting.
 *
 * The panel used to appear only when a live run reported an interrupt, so a
 * task paused before you opened the page showed its plan and nothing else --
 * no prompt, no buttons, no way to approve it. Attaching to a task is a
 * different path from running one, and it needs the same affordance. */
async function syncApproval(taskId) {
  if (!taskId) return;
  try {
    const task = await json(`/api/v1/tasks/${taskId}`);
    const pending = task.pending_confirmation;
    if (pending) {
      showApproval({
        effect: pending.effect,
        preview: pending.preview,
        detail: pending.reason,
        id: pending.request_id,
      });
      setStatus("waiting_confirmation", false);
      $("btn-cancel").disabled = false;
      // The panel is content. Leaving "No task running" above it contradicts
      // what the panel says and reads as a bug.
      $("empty")?.classList.add("hidden");
    } else {
      hideApproval();
    }
  } catch {
    /* the task detail is a nicety; failing to read it must not break attach */
  }
}

function authHeaders(extra) {
  const headers = { ...(extra || {}) };
  if (TOKEN) headers["X-UAA-Token"] = TOKEN;
  return headers;
}

async function json(url, options) {
  const request = { ...(options || {}) };
  request.headers = authHeaders(request.headers);
  const response = await fetch(url, request);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail.slice(0, 300)}`);
  }
  return response.json();
}

/* ------------------------------------------------------------ run context */

function newRun() {
  return {
    assistant: null,       // the growing text node for the final answer
    messageText: new Map(), // messageId -> element
    tools: new Map(),       // toolCallId -> element
    plan: null,             // plan element
    interrupted: false,
  };
}

/* --------------------------------------------------------------- renderers */

function renderEvent(event) {
  const run = state.current;
  switch (event.type) {
    case "RUN_STARTED":
      run.assistant = addTurn("assistant", "Agent");
      setStatus("running", true);
      break;

    case "TEXT_MESSAGE_START": {
      const body = el("div", "body");
      run.assistant.append(body);
      run.messageText.set(event.messageId, body);
      break;
    }

    case "TEXT_MESSAGE_CONTENT": {
      const body = run.messageText.get(event.messageId);
      if (body) {
        body.textContent += event.delta || "";
        scrollToBottom();
      }
      break;
    }

    case "TOOL_CALL_START": {
      // One line per call, expandable. A card per call with its arguments
      // always visible turns a run of eight reads into eight screens of
      // JSON -- which is what a log file looks like, not a tool.
      const node = el("details", "tool running");
      const head = el("summary", "tool-head");
      const mark = el("span", "tool-mark", "·");
      head.append(mark, el("span", "tool-name", event.toolCallName));
      node.append(head);

      const args = el("pre", "tool-args");
      node.append(args);
      node.dataset.args = "";

      run.assistant.append(node);
      run.tools.set(event.toolCallId, node);
      break;
    }

    case "TOOL_CALL_ARGS": {
      const node = run.tools.get(event.toolCallId);
      if (node) {
        node.dataset.args += event.delta || "";
        const args = node.querySelector(".tool-args");
        if (args) args.textContent = node.dataset.args;
        const hint = summariseArgs(node.dataset.args);
        const label = node.querySelector(".tool-arg");
        if (hint) {
          if (label) setText(label, hint);
          else node.querySelector(".tool-head")?.append(el("span", "tool-arg", hint));
        }
      }
      break;
    }

    case "TOOL_CALL_RESULT": {
      const node = run.tools.get(event.toolCallId);
      if (node) {
        const ok = !event.metadata || event.metadata.success !== false;
        node.classList.remove("running");
        node.classList.add(ok ? "ok" : "failed");
        setText(node.querySelector(".tool-mark"), ok ? "✓" : "✕");
        const result = el("pre", "tool-result");
        result.textContent = String(event.content ?? "").slice(0, 4000);
        node.append(result);
      }
      break;
    }

    case "ACTIVITY_SNAPSHOT": {
      if (event.activityType !== "PLAN") break;
      if (!run.plan) {
        const wrapper = el("section", "plan-card");
        wrapper.append(el("div", "plan-label", "计划"));
        run.plan = el("div", "plan");
        wrapper.append(run.plan);
        run.assistant.append(wrapper);
      }
      run.plan.replaceChildren();
      const marks = { pending: "○", running: "◐", completed: "●", failed: "✕", skipped: "—" };
      for (const [i, step] of (event.content.steps || []).entries()) {
        const status = step.status || "pending";
        const row = el("div", `plan-step ${status}`);
        // The glyph carries the status; the number carries the order. Both
        // are fixed-width so the descriptions line up in a column.
        row.append(el("span", "mark", marks[status] || "○"));
        row.append(el("span", "num", String(i + 1)));
        row.append(el("span", "step-text", step.description));
        run.plan.append(row);
      }
      break;
    }

    case "RUN_FINISHED": {
      const outcome = event.outcome || { type: "success" };
      if (outcome.type === "interrupt") {
        run.interrupted = true;
        showApproval(outcome.interrupts[0]);
        setStatus("waiting", false);
      } else {
        setStatus("idle", false);
        state.running = false;
      }
      $("btn-cancel").disabled = true;
      break;
    }

    case "RUN_ERROR":
      addTurn("error", "错误").append(el("div", "body", event.message || "未知错误"));
      setStatus("failed", false);
      state.running = false;
      $("btn-cancel").disabled = true;
      break;

    case "CUSTOM":
      renderCustom(event);
      break;

    default:
      break;
  }
}

function renderCustom(event) {
  const run = state.current;
  if (event.name === "usage" && run.assistant) {
    // One chip that updates, not one per model call. A run makes a call per
    // step, and appending a chip each time buried the turn header under a row
    // of near-identical numbers.
    const value = event.value || {};
    run.usage = (run.usage || 0) + (value.totalTokens || 0);
    const head = run.assistant.querySelector(".turn-head");
    if (head) {
      let chip = head.querySelector(".usage-chip");
      if (!chip) {
        chip = el("span", "chip usage-chip");
        head.append(chip);
      }
      setText(chip, `${value.model || "?"} · ${run.usage} tok`);
    }
  }
  if (event.name === "context_compacted" && run.assistant) {
    run.assistant.append(el("div", "dim", "上下文已压缩，早期步骤已摘要"));
  }
  if (event.name === "tool_ambiguous" && run.assistant) {
    const warn = el("div", "tool");
    warn.append(el("div", "tool-head", "结果未知"));
    warn.append(
      el(
        "div",
        "dim",
        `\`${event.value.tool}\` 在中断时正在执行，无法确定它是否已经跑过。` +
          "继续之前请先确认当前状态。"
      )
    );
    run.assistant.append(warn);
  }
  if (event.name === "stream_idle") {
    setStatus("waiting", true);
  }
  if (event.name === "stream_notice") {
    setStatus("resyncing", true);
  }
}

/* ---------------------------------------------------------------- approval */

function showApproval(interrupt) {
  if (!interrupt) return;
  $("approval-effect").textContent = interrupt.effect || "";
  $("approval-preview").textContent = interrupt.preview || "";
  $("approval-detail").textContent = interrupt.detail || "";
  $("approval").classList.remove("hidden");
  $("approval").dataset.requestId = interrupt.id || "";
}

function hideApproval() {
  $("approval").classList.add("hidden");
}

async function decide(verb) {
  if (!state.taskId) return;
  hideApproval();
  const run = newRun();
  state.current = run;
  run.assistant = addTurn("assistant", "agent");
  setStatus(verb === "approve" ? "running" : "idle", true);
  try {
    await json(`/api/v1/tasks/${state.taskId}/${verb}`, { method: "POST" });
    await attach(state.taskId);
  } catch (error) {
    addTurn("error", "error").append(el("div", "body", String(error)));
    setStatus("failed", false);
  }
}

/* -------------------------------------------------------------- SSE client */

async function streamRun(body) {
  const response = await fetch("/agui", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${response.status}: ${(await response.text()).slice(0, 300)}`);
  }
  await consume(response);
}

/* Attach to a task, and re-attach if the connection drops while it is still
 * going.
 *
 * The server holds the stream open for the whole run and only closes it when
 * the task reaches a terminal state, so under normal conditions this loop
 * runs once. What it is for is the abnormal case: the server restarts, the
 * machine sleeps, the connection is reset. Without it the transcript simply
 * stops updating, with no error and no indication that anything is wrong --
 * the page looks like the task is still thinking.
 */
async function attach(taskId) {
  const live = new Set(["pending", "planning", "running"]);
  for (;;) {
    if (state.taskId !== taskId) return; // the user moved to another task

    let response;
    try {
      response = await fetch(`/api/v1/tasks/${taskId}/stream`, {
        headers: authHeaders(),
      });
    } catch (error) {
      if (state.taskId !== taskId) return;
      console.warn("stream connect failed, retrying", error);
      await new Promise((r) => setTimeout(r, 1500));
      continue;
    }
    if (!response.ok) throw new Error(`${response.status} attaching to ${taskId}`);

    await consume(response);
    if (state.taskId !== taskId) return;

    const task = await json(`/api/v1/tasks/${taskId}`).catch(() => null);
    if (!task || !live.has(task.status)) return;
    console.warn(`stream ended while ${taskId} is ${task.status}; re-attaching`);
    await new Promise((r) => setTimeout(r, 800));
  }
}

async function consume(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line.
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          renderEvent(JSON.parse(payload));
        } catch (error) {
          console.warn("bad SSE payload", payload, error);
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ actions */

async function send(goal) {
  if (state.running) return;
  state.running = true;
  hideApproval();
  $("btn-cancel").disabled = false;

  addTurn("user", "你").append(el("div", "body", goal));
  state.current = newRun();

  const approve = approvedEffects();
  try {
    await streamRun({
      threadId: state.threadId,
      messages: [{ role: "user", content: goal }],
      model: $("model").value.trim() || undefined,
      approve,
    });
  } catch (error) {
    addTurn("error", "error").append(el("div", "body", String(error)));
    setStatus("failed", false);
  } finally {
    if (!state.current.interrupted) {
      state.running = false;
      $("btn-cancel").disabled = true;
    }
    refreshTasks();
  }
}

async function cancel() {
  if (!state.taskId) return;
  // Read the answer. A paused task is cancelled outright and a running one is
  // only *asked* to stop, so "cancelling…" is wrong half the time -- and a
  // status line that lies is worse than no status line.
  try {
    const response = await fetch(`/api/v1/tasks/${state.taskId}/cancel`, {
      method: "POST",
      headers: authHeaders(),
    });
    const result = await response.json();
    if (result.outcome === "cancelled" || result.outcome === "already_terminal") {
      state.running = false;
      state.current = newRun();
      hideApproval();
      $("btn-cancel").disabled = true;
      setStatus(result.status || "cancelled", false);
    } else {
      setStatus("cancelling", true);
    }
  } catch (error) {
    setStatus("failed", false);
  }
}

/* -------------------------------------------------------------------- boot */

function sandboxRow(key, value, tone) {
  const row = el("div", "sandbox-row");
  row.append(el("span", "sandbox-key", key));
  row.append(tone ? el("span", `chip ${tone}`, value) : el("span", "sandbox-val", value));
  return row;
}

function renderSandbox(sandbox) {
  const box = $("sandbox");
  box.replaceChildren();
  const isolated = sandbox.backend && sandbox.backend !== "none";
  box.append(sandboxRow("后端", sandbox.backend || "none", isolated ? "chip-ok" : "chip-warn"));
  box.append(sandboxRow("隔离", sandbox.isolation || "none"));

  const notes = sandbox.notes || [];
  if (!isolated) {
    // The one-line version of the caveat. It is the thing a user needs to
    // know, and it has to be visible without expanding anything.
    box.append(
      el("div", "sandbox-alert", "命令在无操作系统隔离下运行。路径围栏与命令守卫仍然生效。"),
    );
  }
  if (notes.length) {
    // Collapsed: these are three paragraphs of explanation, and a sidebar is
    // not the place to read them. Available, not in the way.
    const details = el("details", "sandbox-why");
    details.append(el("summary", "", `为什么？(${notes.length})`));
    const list = el("ul");
    for (const note of notes) list.append(el("li", "", note));
    details.append(list);
    box.append(details);
  }
}

async function boot() {
  buildApprovalToggles();

  try {
    const health = await json("/api/v1/health");
    $("version").textContent = health.version;
    if (health.default_model) $("model").value = health.default_model;
    if (health.token_required && !TOKEN) {
      // Say it here rather than letting every action fail with a 403.
      addTurn("error", "错误").append(
        el(
          "div",
          "body",
          "这个服务需要会话令牌，而当前页面没有。\n" +
            "用 `uaa desktop` 启动（令牌会传给窗口），或打开 " +
            "`uaa serve --token` 打印出来的那个地址。"
        )
      );
    }
  } catch { /* the console still works without /health */ }

  try {
    renderSandbox(await json("/api/v1/sandbox"));
  } catch {
    $("sandbox").replaceChildren(el("div", "sandbox-val", "不可用"));
  }

  await refreshTasks();
  // Open the most recent task on load, so the page shows where you left off
  // instead of an empty pane above a task list that has things in it.
  if (state.taskId) {
    state.current = newRun();
    // `attach` is a long-lived SSE stream and does not resolve until it ends,
    // so it is deliberately not awaited: awaiting it here meant everything
    // after it -- including the approval panel -- never ran.
    await syncApproval(state.taskId);
    attach(state.taskId).catch((error) => console.warn("attach failed", error));
  }
  setInterval(refreshTasks, 5000);
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const goal = $("goal").value.trim();
  if (!goal) return;
  $("goal").value = "";
  send(goal);
});

/* ------------------------------------------------------------------ composer */

/* Whether an input method is mid-composition.
 *
 * This matters more for Chinese than for English: typing 你好 means typing
 * `nihao` and pressing Enter to pick a candidate. Without this check that
 * Enter submits the form, and the agent starts on a half-typed sentence. Both
 * the flag and `event.isComposing` are consulted, because some browsers fire
 * `keydown` before `compositionend` and others after. */
let composing = false;

$("goal").addEventListener("compositionstart", () => {
  composing = true;
});
$("goal").addEventListener("compositionend", () => {
  composing = false;
});

$("goal").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  if (event.isComposing || composing) return;
  event.preventDefault();
  $("composer").requestSubmit();
});

/* Grow with the text. `field-sizing: content` does this natively where it is
 * supported, so the script only fills in where it is not -- otherwise the two
 * fight over the height. */
const nativeSizing =
  typeof CSS !== "undefined" && CSS.supports && CSS.supports("field-sizing", "content");

$("goal").addEventListener("input", () => {
  if (nativeSizing) return;
  const box = $("goal");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 220)}px`;
});

/* One button, two jobs: while a run is going, the thing you want is to stop
 * it, and hunting for a separate control to do that is how people end up
 * closing the tab instead. */
function setSendMode(running) {
  const button = $("btn-send");
  if (running) {
    button.type = "button";
    button.textContent = "停止";
    button.className = "btn btn-danger";
    button.onclick = () => cancel();
  } else {
    button.type = "submit";
    button.textContent = "发送";
    button.className = "btn btn-primary";
    button.onclick = null;
  }
}

for (const button of document.querySelectorAll(".example")) {
  button.onclick = () => {
    $("goal").value = button.textContent;
    $("goal").focus();
  };
}

$("btn-approve").onclick = () => decide("approve");
$("btn-deny").onclick = () => decide("deny");
$("btn-cancel").onclick = cancel;

boot();
