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
  $("empty")?.classList.add("hidden");
  const turn = el("div", `turn ${kind}`);
  const head = el("div", "turn-head");
  head.append(el("span", "", title));
  turn.append(head);
  $("transcript").append(turn);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return turn;
}

function clearTranscript() {
  $("transcript").replaceChildren($("empty"));
  $("empty")?.classList.remove("hidden");
}

function setStatus(text, busy = false) {
  const node = $("status");
  node.textContent = text;
  node.classList.toggle("busy", busy);
  // Also exposed as a data attribute so the colour can follow the state
  // rather than the wording -- "waiting_confirmation" should look like a
  // warning, and it should keep looking like one if the label is reworded.
  node.dataset.state = String(text || "idle");
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

function statusChip(status) {
  const tone =
    {
      completed: "chip-ok",
      failed: "chip-danger",
      cancelled: "chip-danger",
      waiting_confirmation: "chip-warn",
      running: "chip-accent",
      planning: "chip-accent",
      pending: "chip-accent",
    }[status] || "";
  // Short label, full value in the tooltip. `waiting_confirmation` is a wire
  // value; as a UI label it is both long and jargon, and it wrapped the row.
  const label =
    {
      waiting_confirmation: "waiting",
      completed: "done",
    }[status] || status;
  const chip = el("span", `chip ${tone}`.trim(), label);
  chip.title = status;
  return chip;
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
      // The name is the label. A chip reading "tool" next to the tool's own
      // name says nothing and costs a line of visual noise on every call.
      head.append(el("span", "chip", event.toolCallName));
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
        node.classList.add(ok ? "ok" : "failed");
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
      setStatus("cancelling…", true);
    }
  } catch (error) {
    setStatus(`cancel failed: ${error}`, false);
  }
}

/* -------------------------------------------------------------------- boot */

async function refreshTasks() {
  try {
    const tasks = await json("/api/v1/tasks?limit=15");
    const list = $("tasks");
    list.replaceChildren();
    $("task-count").textContent = tasks.length ? String(tasks.length) : "";
    if (!tasks.length) {
      list.append(el("div", "task-empty", "Nothing yet."));
    }
    for (const task of tasks) {
      const item = el("div", "task-item" + (task.id === state.taskId ? " active" : ""));
      item.append(el("div", "task-goal", task.goal));
      const meta = el("div", "task-meta");
      meta.append(statusChip(task.status));
      meta.append(
        el("span", "", `${task.steps_used} steps · ${task.tokens_in + task.tokens_out} tok`),
      );
      item.append(meta);
      item.onclick = async () => {
        state.taskId = task.id;
        clearTranscript();
        state.current = newRun();
        // Order matters: `attach` opens an SSE stream and does not resolve
        // until the stream ends, so anything awaited after it never runs.
        // Read the approval state first, then start streaming.
        await syncApproval(task.id);
        attach(task.id).catch((error) => console.warn("attach failed", error));
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

/* The sandbox report used to be one joined string, which wrapped into a wall
   of prose that pushed everything else off the panel. It is structured data:
   show it as rows, and put the caveats in a block that reads as a caveat. */

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
  box.append(sandboxRow("Backend", sandbox.backend || "none", isolated ? "chip-ok" : "chip-warn"));
  box.append(sandboxRow("Isolation", sandbox.isolation || "none"));

  const notes = sandbox.notes || [];
  if (!isolated) {
    // The one-line version of the caveat. It is the thing a user needs to
    // know, and it has to be visible without expanding anything.
    box.append(
      el("div", "sandbox-alert", "Commands run without OS isolation. The path fence and command guard still apply."),
    );
  }
  if (notes.length) {
    // Collapsed: these are three paragraphs of explanation, and a sidebar is
    // not the place to read them. Available, not in the way.
    const details = el("details", "sandbox-why");
    details.append(el("summary", "", `why? (${notes.length})`));
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
    renderSandbox(await json("/api/v1/sandbox"));
  } catch {
    $("sandbox").replaceChildren(el("div", "sandbox-val", "unavailable"));
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

$("goal").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

/* Grow with the text instead of scrolling inside two rows. A goal worth
   typing is usually longer than one line. */
$("goal").addEventListener("input", () => {
  const box = $("goal");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 200)}px`;
});

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
