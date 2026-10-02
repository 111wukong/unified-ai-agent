/* wukong console.
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
  // Preview pane. The text and the lock live here rather than in the DOM, so
  // a save sends back what it opened instead of what the view happens to say.
  previewPath: null,
  previewText: "",
  previewSha: null,
  previewEditable: false,
  // Tasks running anywhere in this process, from /health. The tab's own run is
  // tracked locally; this catches the ones it did not start -- the agent is a
  // single in-process object, so a run started from the CLI holds the
  // workspace too.
  runningTasks: 0,
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

/* ---------------------------------------------------------------- markdown
 *
 * The model writes Markdown; this renders it. Hand-rolled rather than pulled
 * in -- the project has no dependencies, and a renderer that builds DOM nodes
 * cannot inject markup the way `innerHTML` can.
 *
 * Deliberately partial: fenced code, inline code, bold, italic, headings,
 * lists and links. Everything else falls through as literal text, which is
 * the right failure mode -- a console that mangles a table is worse than one
 * that shows it plainly.
 */

const MD_BLOCK = /^(?:```|#{1,4}\s|\s*[-*+]\s|\s*\d+[.)]\s)/;

function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  const lines = String(text ?? "").split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (/^```/.test(line)) {
      const lang = line.slice(3).trim();
      const body = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      i += 1; // the closing fence, or the end of an unterminated block
      const pre = el("pre", "md-pre");
      const code = el("code", "", body.join("\n"));
      if (lang) code.dataset.lang = lang;
      pre.append(code);
      frag.append(pre);
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const node = el("div", `md-head md-h${heading[1].length}`);
      appendInline(node, heading[2]);
      frag.append(node);
      i += 1;
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line) || /^\s*\d+[.)]\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const list = el(ordered ? "ol" : "ul", "md-list");
      while (
        i < lines.length &&
        (/^\s*[-*+]\s+/.test(lines[i]) || /^\s*\d+[.)]\s+/.test(lines[i]))
      ) {
        const item = el("li", "");
        appendInline(item, lines[i].replace(/^\s*(?:[-*+]|\d+[.)])\s+/, ""));
        list.append(item);
        i += 1;
      }
      frag.append(list);
      continue;
    }

    if (!line.trim()) {
      i += 1;
      continue;
    }

    const para = [];
    while (i < lines.length && lines[i].trim() && !MD_BLOCK.test(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    const node = el("div", "md-p");
    appendInline(node, para.join("\n"));
    frag.append(node);
  }

  return frag;
}

/* Inline code is split out first, so `**` inside a span stays literal. */
function appendInline(parent, text) {
  for (const part of String(text).split(/(`[^`]+`)/g)) {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      parent.append(el("code", "md-code", part.slice(1, -1)));
    } else {
      appendEmphasis(parent, part);
    }
  }
}

function appendEmphasis(parent, text) {
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;
  let last = 0;
  let match;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      parent.append(document.createTextNode(text.slice(last, match.index)));
    }
    const token = match[0];
    if (token.startsWith("**")) {
      parent.append(el("strong", "", token.slice(2, -2)));
    } else if (token.startsWith("[")) {
      const link = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/.exec(token);
      const anchor = el("a", "", link[1]);
      anchor.href = link[2];
      anchor.target = "_blank";
      anchor.rel = "noreferrer";
      parent.append(anchor);
    } else {
      parent.append(el("em", "", token.slice(1, -1)));
    }
    last = match.index + token.length;
  }
  if (last < text.length) {
    parent.append(document.createTextNode(text.slice(last)));
  }
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

/* The gist of a result, for the collapsed row.
 *
 * The first line that has anything in it, because tool output usually leads
 * with the answer and follows with detail -- `list_directory` returns the
 * listing, `read_file` returns the file. Picking the *last* line would show
 * the trailing blank. */
function summariseResult(content) {
  const text = String(content ?? "").trim();
  if (!text) return "";
  const line = text.split("\n").find((candidate) => candidate.trim()) || "";
  const clean = line.trim().replace(/\s+/g, " ");
  return clean.length > 64 ? `${clean.slice(0, 64)}…` : clean;
}

/* One glyph per *family*, not per tool.
 *
 * Twenty-two bespoke icons is a drawing exercise; five is a vocabulary you
 * learn in one glance. The point is to let a column of calls be scanned for
 * shape -- a run of document glyphs is reading, a chevron is a side effect --
 * without reading a single tool name. */
const TOOL_ICON = {
  read: "M5 3h7l5 5v13H5zM12 3v5h5",
  write: "M4 20h4L19 9a2 2 0 0 0-3-3L5 17zM14 6l3 3",
  exec: "M5 7l5 5-5 5M13 17h6",
  vcs: "M6 4v16M6 8h9a3 3 0 0 1 3 3v9M18 17l-2-2M18 17l2-2",
  plan: "M4 7h16M4 12h16M4 17h10",
  other: "M12 6v12M6 12h12",
};

const TOOL_FAMILY = [
  [/^(read_file|list_directory|file_info|search_files|grep|recall|list_memory|search_memory)$/, "read"],
  [/^(write_file|apply_patch|delete_file|delete_memory|save_memory|update_plan)$/, "write"],
  [/^run_command$/, "exec"],
  [/^git_/, "vcs"],
  [/^(load_skill|finish)$/, "plan"],
];

function toolFamily(name) {
  for (const [pattern, family] of TOOL_FAMILY) {
    if (pattern.test(name)) return family;
  }
  return "other";
}

function toolIcon(name) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "13");
  svg.setAttribute("height", "13");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.7");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  const path = document.createElementNS(NS, "path");
  path.setAttribute("d", TOOL_ICON[toolFamily(name)]);
  svg.append(path);
  return svg;
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

/* Parents before their children, children indented.
 *
 * A fan-out run creates one task per sub-agent. The list used to show them as
 * peers of the task that spawned them, so "fan out to five" read as six
 * unrelated runs -- and the sub-agent that failed was indistinguishable from
 * the one that called it. The relationship was already in the data
 * (`parent_task_id`); nothing was reading it.
 *
 * Returns a flat list with a depth, rather than a nested structure, because
 * the renderer already rebuilds the list in one pass and a tree of DOM nodes
 * would only add a second shape to keep in sync. */
function flattenTasks(tasks) {
  const byId = new Map(tasks.map((task) => [task.id, task]));
  const kids = new Map();
  const roots = [];

  for (const task of tasks) {
    const parent = task.parent_task_id;
    // A parent outside this page (older than the limit) is not a parent here.
    // Showing the child at the top level beats dropping it off the list.
    if (parent && byId.has(parent)) {
      if (!kids.has(parent)) kids.set(parent, []);
      kids.get(parent).push(task);
    } else {
      roots.push(task);
    }
  }

  const out = [];
  const walk = (task, depth) => {
    out.push({ task, depth });
    for (const child of kids.get(task.id) || []) walk(child, depth + 1);
  };
  for (const root of roots) walk(root, 0);
  return out;
}

function taskRow(task, depth = 0) {
  const item = el("div", `task-item depth-${Math.min(depth, 3)}`);
  const goal = el("div", "task-goal");
  if (depth > 0) goal.append(el("span", "task-child-mark", "↳ "));
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
  // The status *key*, not its label: the colour is chosen from this and the
  // text is looked up separately, so a translated label cannot lose the colour.
  node.item.dataset.state = task.status;
  node.item.classList.toggle("active", task.id === state.taskId);
}

/* The header says which task you are looking at and what it is doing. Without
 * it the content floated in the middle of a dark rectangle with nothing to
 * anchor it -- and no way to tell at a glance whether the thing on screen was
 * still running. */
/* How much of the budget is gone.
 *
 * A run that ends on `steps budget exhausted` should not come as a surprise:
 * the limit was known from the first step, and watching it climb is the
 * difference between "it stopped" and "it ran out". Drawn as a bar rather than
 * a fraction because the question is "how close", which a ratio answers at a
 * glance and a pair of integers does not.
 *
 * The limits live on the task detail (they are part of the task's own record),
 * not on the list rows, so they are fetched once per selected task and cached
 * -- the list polls every few seconds and must not drag a detail call with it.
 */
const budgetCache = { taskId: null, limits: {} };

async function renderBudget(task) {
  const meta = $("topbar-meta");
  if (!task) {
    if (meta.dataset.wanted !== "") {
      meta.dataset.wanted = "";
      meta.replaceChildren();
    }
    return;
  }

  if (budgetCache.taskId !== task.id) {
    budgetCache.taskId = task.id;
    budgetCache.limits = {};
    try {
      budgetCache.limits = (await json(`/api/v1/tasks/${task.id}`)).budgets || {};
    } catch {
      /* the bar is a nicety; the run does not depend on it */
    }
  }

  const steps = Number(task.steps_used || 0);
  const tokens = Number(task.tokens_in || 0) + Number(task.tokens_out || 0);
  const cost = Number(task.cost_usd || 0);
  const limit = Number(budgetCache.limits.max_steps || 0);

  const wanted = limit
    ? `${steps}/${limit} 步 · ${tokens} token · $${cost.toFixed(4)}`
    : `${steps} 步 · ${tokens} token`;
  if (meta.dataset.wanted === wanted) return;
  meta.dataset.wanted = wanted;

  meta.replaceChildren(el("span", "budget-text", wanted));
  if (!limit) return;

  const ratio = Math.min(1, steps / limit);
  // The last fifth is where a run is about to stop, and it is the only part of
  // the bar anyone needs to notice.
  const tone = ratio >= 0.8 ? "danger" : ratio >= 0.5 ? "warn" : "ok";
  const bar = el("span", `budget-bar ${tone}`);
  const fill = el("span", "budget-fill");
  fill.style.width = `${Math.round(ratio * 100)}%`;
  bar.append(fill);
  meta.append(bar);
}

function renderTopbar(task) {
  setText($("topbar-title"), task ? task.id : "控制台");
  setTitle($("topbar-title"), task ? task.goal : "");

  const chip = $("topbar-status");
  setClass(chip, `chip ${STATUS_TONE[task?.status] || ""}`.trim());
  setText(chip, task ? STATUS_LABEL[task.status] || task.status : STATUS_LABEL.idle);
  setTitle(chip, task?.status || "");

  renderBudget(task).catch((error) => console.warn("budget unavailable", error));
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

    const ordered = flattenTasks(tasks);
    const ids = ordered.map((entry) => entry.task.id).join();
    if (ids !== [...taskNodes.keys()].join()) {
      // The *set* changed, so rebuild. Doing this on every poll -- which is
      // what it used to do -- destroys the element under the pointer every
      // five seconds: hover flickers, and a click lands on a node that is
      // already gone.
      const next = new Map();
      list.replaceChildren();
      for (const { task, depth } of ordered) {
        const node = taskNodes.get(task.id) || taskRow(task, depth);
        paintTask(node, task);
        list.append(node.item);
        next.set(task.id, node);
      }
      taskNodes.clear();
      for (const [id, node] of next) taskNodes.set(id, node);
    } else {
      for (const { task } of ordered) paintTask(taskNodes.get(task.id), task);
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
  if (TOKEN) headers["X-WUKONG-Token"] = TOKEN;
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
      // Which task this run is. The stream is the only place the client can
      // learn it -- `threadId` is the client's own id, not the task's. Without
      // this, approving an approval prompt posted to whatever task the list
      // had selected, which is a 409 the moment anything has been run before.
      if (event.taskId) {
        state.taskId = event.taskId;
        refreshTasks();
      }
      run.assistant = addTurn("assistant", "Agent");
      setStatus("running", true);
      break;

    case "TEXT_MESSAGE_START": {
      const body = el("div", "body md");
      run.assistant.append(body);
      run.messageText.set(event.messageId, body);
      break;
    }

    case "TEXT_MESSAGE_CONTENT": {
      const body = run.messageText.get(event.messageId);
      if (body) {
        // Re-render the whole message instead of appending to the DOM: a
        // fenced block that arrives one line at a time has to be able to
        // change its mind about what it is. `dataset.raw` keeps the source.
        body.dataset.raw = (body.dataset.raw || "") + (event.delta || "");
        body.replaceChildren(renderMarkdown(body.dataset.raw));
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
      const icon = el("span", "tool-icon");
      icon.append(toolIcon(event.toolCallName));
      head.append(mark, icon, el("span", "tool-name", event.toolCallName));
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

        // A one-line gist on the collapsed row. Without it, knowing whether a
        // call found anything meant opening all of them -- which is the thing
        // collapsing was supposed to avoid.
        const gist = summariseResult(event.content);
        if (gist) node.querySelector(".tool-head")?.append(el("span", "tool-gist", gist));

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

/* Colour a unified diff so the eye lands on the changed lines.
 *
 * The +/- prefixes are the only thing separating a diff from prose, and
 * uncoloured on a dark background they are nearly invisible -- which loses the
 * entire point of showing a diff instead of the payload, because the reader
 * still has to hunt for the change.
 *
 * Colourising is conditional. A wall of one colour is not a diff, so text that
 * does not look like one (a command, a JSON payload) keeps the plain treatment
 * the panel already gave it. */
function renderPreview(node, text) {
  node.replaceChildren();
  const lines = String(text).split("\n");
  const looksLikeDiff =
    lines.some((line) => line.startsWith("+++") || line.startsWith("---")) &&
    lines.some((line) => /^[+-][^+-]/.test(line));
  if (!looksLikeDiff) {
    node.textContent = text;
    return;
  }
  for (const line of lines) {
    let cls = "diff-line";
    if (/^(\+\+\+|---)/.test(line)) cls += " diff-meta";
    else if (line.startsWith("@@")) cls += " diff-hunk";
    else if (line.startsWith("+")) cls += " diff-add";
    else if (line.startsWith("-")) cls += " diff-del";
    node.append(el("div", cls, line));
  }
}

function showApproval(interrupt) {
  if (!interrupt) return;
  $("approval-effect").textContent = interrupt.effect || "";
  renderPreview($("approval-preview"), interrupt.preview || "");
  $("approval-detail").textContent = interrupt.detail || "";
  $("approval").classList.remove("hidden");
  $("approval").dataset.requestId = interrupt.id || "";
}

function hideApproval() {
  $("approval").classList.add("hidden");
}

async function decide(verb) {
  if (!state.taskId) {
    setStatus("failed", false);
    addTurn("error", "错误").append(
      el("div", "body", "不知道该批准哪个任务 —— 界面上没有选中的任务。"),
    );
    return;
  }
  hideApproval();
  const run = newRun();
  state.current = run;
  run.assistant = addTurn("assistant", "Agent");
  setStatus(verb === "approve" ? "running" : "idle", true);
  try {
    await json(`/api/v1/tasks/${state.taskId}/${verb}`, { method: "POST" });
    await attach(state.taskId);
  } catch (error) {
    // A 409 here means the request went to a task that was not waiting, which
    // is a stale id rather than anything the user did wrong. Naming the task
    // makes it diagnosable instead of just "409".
    addTurn("error", "错误").append(
      el(
        "div",
        "body",
        `${verb === "approve" ? "批准" : "拒绝"}失败：${error}\n` +
          `打到的任务是 ${state.taskId}`,
      ),
    );
    setStatus("failed", false);
  }
}

/* ----------------------------------------------------------------- skills */

/* The human gate.
 *
 * `candidate → validated → approved → active` is enforced in the runtime, and
 * that ladder *is* the safety mechanism: an agent able to promote its own
 * skill would have no gate at all. A gate reachable only from a terminal is
 * one that gets bypassed -- by editing the database, or by moving files
 * around -- so it belongs here.
 *
 * The list comes from the registry, not from the `skills` table. The table
 * only holds skills promoted at least once, so a fresh candidate was
 * invisible -- which made the gate unreachable, because the one thing needing
 * a decision was the one thing this pane could not show. */

const PENDING_STATUS = new Set(["candidate", "validated", "approved"]);

/* The next rung. Derived from the ladder rather than from the current status
 * alone, so adding a rung later is one edit here and nothing else. */
const NEXT_STATUS = {
  candidate: "validated",
  validated: "approved",
  approved: "active",
};

const SKILL_TONE = {
  active: "chip-ok",
  approved: "chip-accent",
  validated: "chip-warn",
  candidate: "chip-warn",
  deprecated: "",
};

async function loadSkills() {
  const list = $("skills");
  let skills;
  try {
    skills = await json("/api/v1/skills");
  } catch (error) {
    list.replaceChildren(el("div", "skill-error", `读取失败：${error.message || error}`));
    return;
  }

  setText($("skill-count"), skills.length ? String(skills.length) : "");
  // A closed drawer hides the one thing that needs a decision. It opens for
  // anything waiting, and only then -- a drawer that opens itself for no
  // reason is noise.
  if (skills.some((skill) => PENDING_STATUS.has(skill.status))) {
    $("skills-fold").open = true;
  }

  list.replaceChildren();
  if (!skills.length) {
    list.append(el("div", "skill-empty", "还没有技能。"));
    return;
  }
  for (const skill of skills) list.append(skillRow(skill));
}

function skillRow(skill) {
  const row = el("div", `skill-row status-${skill.status}`);
  const head = el("div", "skill-head");
  head.append(el("span", "skill-name", skill.name));
  head.append(el("span", `chip ${SKILL_TONE[skill.status] || ""}`.trim(), skill.status));
  row.append(head);
  if (skill.description) row.append(el("div", "skill-desc", skill.description));

  const next = NEXT_STATUS[skill.status];
  if (!next) return row;

  const actions = el("div", "skill-actions");
  const button = el("button", "btn btn-ghost skill-promote", `晋级为 ${next}`);
  button.type = "button";
  button.onclick = async (event) => {
    // The row is not clickable today, but stopping the event here means it can
    // become so without this button silently triggering it as well.
    event.stopPropagation();
    button.disabled = true;
    setText(button, "处理中…");
    try {
      await json(`/api/v1/skills/${encodeURIComponent(skill.name)}/promote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: next }),
      });
    } catch (error) {
      setText(button, `失败：${error.message || error}`);
      button.disabled = false;
      return;
    }
    loadSkills();
  };
  actions.append(button);
  row.append(actions);
  return row;
}

/* ----------------------------------------------------------------- preview */

/* A read-only look at one file.
 *
 * Read-only is the design, not a limitation. The agent may be mid-run holding
 * a file it has already read, and an editor here would open a question nobody
 * has answered -- whose version wins, and what happens to the agent's belief
 * about the file when it loses. A window cannot race anything. */
async function openPreview(path) {
  const body = $("preview-body");
  $("preview").hidden = false;
  $("app").classList.add("with-preview");
  setText($("preview-path"), path);
  setText($("preview-meta"), "");
  body.replaceChildren(el("div", "preview-note", "读取中…"));

  let payload;
  try {
    payload = await json(`/api/v1/files/content?path=${encodeURIComponent(path)}`);
  } catch (error) {
    state.previewPath = null;
    body.replaceChildren(
      el("div", "preview-note error", `读取失败：${error.message || error}`),
    );
    renderPreviewActions();
    return;
  }

  // The source and the lock are kept here rather than read back out of the
  // DOM: the rendered view has line numbers woven through it, and un-weaving
  // them is a parser where a variable will do.
  state.previewPath = path;
  state.previewText = payload.text || "";
  state.previewSha = payload.sha || null;
  state.previewEditable = !payload.binary && !payload.reason;

  setText($("preview-meta"), formatBytes(payload.size));
  // Asked now rather than cached from boot: a task may have started since,
  // and the answer decides whether the edit button is usable at all.
  await refreshRunningCount();
  renderPreviewActions();
  if (!payload.text) {
    body.replaceChildren(el("div", "preview-note", payload.reason || "（空文件）"));
    return;
  }
  body.replaceChildren(renderCode(payload.text));
}

/* Which buttons belong in the header right now.
 *
 * The edit button is disabled while a task is running, and the tooltip says
 * why. A disabled control with no explanation is a bug report waiting to
 * happen -- and this particular disabled control is load-bearing, so the
 * reason is the most important thing on screen.
 *
 * The count comes from the server, not from the local run flag. The agent is
 * one in-process object, so a run started from the CLI holds the workspace
 * just as much as one started here, and the server is the one that will
 * refuse the write. Consulting the local flag would show an enabled button
 * for a save that cannot land. */
function renderPreviewActions(editing = false) {
  const busy = state.runningTasks > 0;
  const canEdit = Boolean(state.previewPath) && state.previewEditable;

  const edit = $("preview-edit");
  edit.hidden = editing || !canEdit;
  edit.disabled = busy;
  setTitle(edit, busy ? "有任务正在运行，工作区归它使用" : "编辑这个文件");

  $("preview-save").hidden = !editing;
  $("preview-cancel").hidden = !editing;
}

async function refreshRunningCount() {
  try {
    state.runningTasks = (await json("/api/v1/health")).running_tasks || 0;
  } catch {
    // If health is unreachable the button stays as it was. The server refuses
    // the write either way, so a stale count is a hint, not a hole.
  }
}

function startEditing() {
  const area = el("textarea", "preview-editor");
  area.value = state.previewText;
  area.spellcheck = false;
  $("preview-body").replaceChildren(area);
  area.focus();
  renderPreviewActions(true);
}

function cancelEditing() {
  openPreview(state.previewPath).catch((error) => console.warn("reload failed", error));
}

async function saveEditing() {
  const area = $("preview-body").querySelector(".preview-editor");
  if (!area || !state.previewPath) return;

  const save = $("preview-save");
  save.disabled = true;
  setText(save, "保存中…");
  try {
    await json("/api/v1/files/content", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: state.previewPath,
        text: area.value,
        // Sent back so a save over a file that moved in the meantime is
        // refused instead of silently winning.
        base_sha: state.previewSha,
      }),
    });
  } catch (error) {
    // Both 409s mean the same thing to a reader: this save should not land.
    // The server's sentence already says which case it is.
    setText(save, "保存失败，见下方");
    $("preview-body").prepend(
      el("div", "preview-note error", String(error.message || error)),
    );
    save.disabled = false;
    return;
  }
  setText(save, "保存");
  save.disabled = false;
  await openPreview(state.previewPath);
}

function closePreview() {
  $("preview").hidden = true;
  $("app").classList.remove("with-preview");
}

/* Line numbers, because a file without them is a wall.
 *
 * No syntax highlighting: that means shipping a tokeniser per language, and
 * what actually helps you find your place in a file you did not write is
 * knowing which line you are on. */
function renderCode(text) {
  const pre = el("pre", "code-view");
  const gutter = el("span", "code-gutter");
  const body = el("span", "code-body");
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    gutter.append(el("span", "code-ln", String(i + 1)));
    body.append(el("span", "code-line", lines[i]));
  }
  pre.append(gutter, body);
  return pre;
}

/* -------------------------------------------------------------------- tree */

/* The workspace, one directory at a time.
 *
 * Lazy on purpose: a directory is fetched when it is opened, not when it is
 * listed. The eager version walks the whole project on every page load, which
 * on a repository with a `node_modules` in it is a slow way to populate a
 * sidebar nobody asked to expand. */
async function loadTree(path = "", container = $("tree"), depth = 0) {
  let payload;
  try {
    payload = await json(`/api/v1/files?path=${encodeURIComponent(path)}`);
  } catch (error) {
    container.append(el("div", "tree-error", `读取失败：${error.message || error}`));
    return;
  }
  if (depth === 0) {
    container.replaceChildren();
    setText($("tree-count"), String(payload.entries.length));
  }
  container.dataset.loaded = path;

  if (!payload.entries.length) {
    container.append(el("div", "tree-empty", "（空目录）"));
    return;
  }

  for (const entry of payload.entries) {
    const isDir = entry.type === "dir";
    const row = el("div", `tree-row ${isDir ? "dir" : "file"}`);
    const label = el("button", "tree-name");
    label.type = "button";
    label.append(el("span", "tree-icon", isDir ? "▸" : "·"));
    label.append(el("span", "tree-text", entry.name));
    if (!isDir && entry.size !== null) {
      label.append(el("span", "tree-size", formatBytes(entry.size)));
    }
    const childPath = path ? `${path}/${entry.name}` : entry.name;
    row.append(label);
    container.append(row);

    if (!isDir) {
      label.addEventListener("click", () => {
        openPreview(childPath).catch((error) => console.warn("preview failed", error));
      });
      continue;
    }
    const kids = el("div", "tree-children");
    kids.hidden = true;
    container.append(kids);
    label.addEventListener("click", async () => {
      const opening = kids.hidden;
      kids.hidden = !opening;
      setText(label.querySelector(".tree-icon"), opening ? "▾" : "▸");
      if (opening && !kids.dataset.loaded) {
        await loadTree(childPath, kids, depth + 1);
      }
    });
  }
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
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
    // The agent is one in-process object, so a run started from the CLI holds
    // the workspace just as much as one started here. Read it once at boot and
    // let the server be the authority on the rest -- it refuses writes while
    // anything is running, which is the guarantee; this is only the hint.
    state.runningTasks = health.running_tasks || 0;
    if (health.token_required && !TOKEN) {
      // Say it here rather than letting every action fail with a 403.
      addTurn("error", "错误").append(
        el(
          "div",
          "body",
          "这个服务需要会话令牌，而当前页面没有。\n" +
            "用 `wukong desktop` 启动（令牌会传给窗口），或打开 " +
            "`wukong serve --token` 打印出来的那个地址。"
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
  // Loaded after the task list, not before: the list is what the page is for,
  // and a slow directory read must not hold it up.
  loadTree().catch((error) => console.warn("tree unavailable", error));
  loadSkills().catch((error) => console.warn("skills unavailable", error));
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
$("preview-close").onclick = closePreview;
$("preview-edit").onclick = startEditing;
$("preview-save").onclick = () => saveEditing().catch((error) => console.warn(error));
$("preview-cancel").onclick = cancelEditing;
// Escape closes the preview, but not while a run is asking for approval --
// dismissing the panel must never be the same gesture as answering it.
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("preview").hidden && $("approval").classList.contains("hidden")) {
    closePreview();
  }
});

boot();
