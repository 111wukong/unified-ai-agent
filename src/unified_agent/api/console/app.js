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
  running: false,
  // Per-run render targets, reset on each RUN_STARTED.
  current: null,
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

function addTurn(kind, title) {
  const turn = el("div", `turn ${kind}`);
  const head = el("div", "turn-head");
  head.append(el("span", "", title));
  turn.append(head);
  $("transcript").append(turn);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return turn;
}

function setStatus(text, busy = false) {
  const node = $("status");
  node.textContent = text;
  node.classList.toggle("busy", busy);
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
      run.assistant = addTurn("assistant", "agent");
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
        $("transcript").scrollTop = $("transcript").scrollHeight;
      }
      break;
    }

    case "TOOL_CALL_START": {
      const node = el("div", "tool running");
      const head = el("div", "tool-head");
      head.append(el("span", "chip", "tool"), el("span", "", event.toolCallName));
      node.append(head);
      run.assistant.append(node);
      run.tools.set(event.toolCallId, node);
      break;
    }

    case "TOOL_CALL_ARGS": {
      const node = run.tools.get(event.toolCallId);
      if (node) {
        const args = node.querySelector("pre") || node.appendChild(el("pre"));
        args.textContent += event.delta || "";
      }
      break;
    }

    case "TOOL_CALL_RESULT": {
      const node = run.tools.get(event.toolCallId);
      if (node) {
        node.classList.remove("running");
        const ok = !event.metadata || event.metadata.success !== false;
        node.classList.add(ok ? "ok" : "bad");
        const details = el("details");
        details.append(el("summary", "", ok ? "result" : "failed"));
        details.append(el("pre", "", String(event.content ?? "").slice(0, 4000)));
        node.append(details);
      }
      break;
    }

    case "ACTIVITY_SNAPSHOT": {
      if (event.activityType !== "PLAN") break;
      if (!run.plan) {
        const wrapper = el("div", "tool");
        wrapper.append(el("div", "tool-head", "plan"));
        run.plan = el("div", "plan");
        wrapper.append(run.plan);
        run.assistant.append(wrapper);
      }
      run.plan.replaceChildren();
      const marks = { pending: " ", running: "~", completed: "x", failed: "!", skipped: "-" };
      for (const [i, step] of (event.content.steps || []).entries()) {
        const row = el("div", `plan-step ${step.status || "pending"}`);
        row.append(el("span", "mark", marks[step.status] || " "));
        row.append(el("span", "", `${i + 1}. ${step.description}`));
        run.plan.append(row);
      }
      break;
    }

    case "RUN_FINISHED": {
      const outcome = event.outcome || { type: "success" };
      if (outcome.type === "interrupt") {
        run.interrupted = true;
        showApproval(outcome.interrupts[0]);
        setStatus("waiting for approval", false);
      } else {
        setStatus("idle", false);
        state.running = false;
      }
      $("btn-cancel").disabled = true;
      break;
    }

    case "RUN_ERROR":
      addTurn("error", "error").append(el("div", "body", event.message || "unknown error"));
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
    const value = event.value || {};
    run.assistant
      .querySelector(".turn-head")
      ?.append(
        el(
          "span",
          "chip",
          `${value.model || "?"} · ${value.totalTokens || 0} tok · $${(value.costUsd || 0).toFixed(4)}`
        )
      );
  }
  if (event.name === "context_compacted" && run.assistant) {
    run.assistant.append(el("div", "dim", "context compacted — older steps summarised"));
  }
  if (event.name === "tool_ambiguous" && run.assistant) {
    const warn = el("div", "tool");
    warn.append(el("div", "tool-head", "outcome unknown"));
    warn.append(
      el(
        "div",
        "dim",
        `${event.value.tool} was interrupted mid-flight. It may or may not have run — ` +
          "check the current state before continuing."
      )
    );
    run.assistant.append(warn);
  }
  if (event.name === "stream_idle") {
    setStatus("waiting…", true);
  }
  if (event.name === "stream_notice") {
    setStatus("stream fell behind — re-syncing", true);
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
  setStatus(`${verb}…`, true);
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

async function attach(taskId) {
  const response = await fetch(`/api/v1/tasks/${taskId}/stream`, {
    headers: authHeaders(),
  });
  if (!response.ok) throw new Error(`${response.status} attaching to ${taskId}`);
  await consume(response);
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

  addTurn("user", "you").append(el("div", "body", goal));
  state.current = newRun();

  const approve = [...$("approve").selectedOptions].map((o) => o.value);
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
  await fetch(`/api/v1/tasks/${state.taskId}/cancel`, {
    method: "POST",
    headers: authHeaders(),
  });
  setStatus("cancelling…", true);
}

/* -------------------------------------------------------------------- boot */

async function refreshTasks() {
  try {
    const tasks = await json("/api/v1/tasks?limit=15");
    const list = $("tasks");
    list.replaceChildren();
    for (const task of tasks) {
      const item = el("div", "task-item" + (task.id === state.taskId ? " active" : ""));
      item.append(el("span", "goal", task.goal));
      item.append(
        el(
          "span",
          "meta",
          `${task.status} · ${task.steps_used} steps · ${task.tokens_in + task.tokens_out} tok`
        )
      );
      item.onclick = async () => {
        state.taskId = task.id;
        $("transcript").replaceChildren();
        state.current = newRun();
        await attach(task.id);
        refreshTasks();
      };
      list.append(item);
    }
    // The most recent task is the one a resume would target.
    if (!state.taskId && tasks.length) state.taskId = tasks[0].id;
  } catch (error) {
    console.warn("task list unavailable", error);
  }
}

async function boot() {
  try {
    const health = await json("/api/v1/health");
    $("version").textContent = health.version;
    if (health.default_model) $("model").value = health.default_model;
    if (health.token_required && !TOKEN) {
      // Say it here rather than letting every action fail with a 403.
      addTurn("error", "error").append(
        el(
          "div",
          "body",
          "This server requires a session token and this page does not have one.\n" +
            "Launch it with `uaa desktop` (the token is passed to the window), or " +
            "open the URL printed by `uaa serve --token`."
        )
      );
    }
  } catch { /* the console still works without /health */ }

  try {
    const sandbox = await json("/api/v1/sandbox");
    const lines = [
      `backend: ${sandbox.backend}`,
      `isolation: ${sandbox.isolation}`,
      sandbox.fell_back ? `requested ${sandbox.requested} — fell back` : null,
      ...(sandbox.notes || []),
    ].filter(Boolean);
    $("sandbox").textContent = lines.join("\n");
    $("sandbox").classList.toggle("warn", sandbox.backend === "none");
  } catch {
    $("sandbox").textContent = "unavailable";
  }

  refreshTasks();
  setInterval(refreshTasks, 5000);
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const goal = $("goal").value.trim();
  if (!goal) return;
  $("goal").value = "";
  send(goal);
});

$("goal").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

$("btn-approve").onclick = () => decide("approve");
$("btn-deny").onclick = () => decide("deny");
$("btn-cancel").onclick = cancel;

boot();
