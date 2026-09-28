#!/usr/bin/env node
/**
 * Screenshot the web console, so UI work can be checked by looking at it.
 *
 *   node scripts/console-shot.mjs [url] [outfile]
 *
 * Why this exists: the console was once served from `/` while its assets lived
 * under `/console/`, so the page arrived with no stylesheet and no script and
 * rendered as unstyled HTML. Every check said it was fine -- `GET /` returned
 * 200, the tests passed, the API worked. The only thing that would have caught
 * it is looking at the page, and there was no way to do that from a terminal.
 *
 * Uses the browser already on the machine and Node's built-in WebSocket, so it
 * adds no dependency to a project that deliberately has none.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const URL_UNDER_TEST = process.argv[2] || "http://127.0.0.1:8765/";
const OUT = process.argv[3] || "/tmp/console.png";
const WIDTH = Number(process.env.SHOT_WIDTH || 1440);
const HEIGHT = Number(process.env.SHOT_HEIGHT || 960);

function findBrowser() {
  const candidates = [
    ...(process.env.HOME
      ? [
          path.join(
            process.env.HOME,
            "Library/Caches/ms-playwright",
          ),
        ]
      : []),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/opt/google/chrome/chrome",
  ];
  for (const candidate of candidates) {
    if (candidate.includes("ms-playwright")) {
      if (!fs.existsSync(candidate)) continue;
      for (const entry of fs.readdirSync(candidate)) {
        if (!entry.startsWith("chromium")) continue;
        for (const arch of fs.readdirSync(path.join(candidate, entry))) {
          const bin = path.join(
            candidate,
            entry,
            arch,
            "chrome-headless-shell",
          );
          if (fs.existsSync(bin)) return { bin, kind: "shell" };
        }
      }
      continue;
    }
    if (fs.existsSync(candidate)) return { bin: candidate, kind: "chrome" };
  }
  return null;
}

function cdp(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let seq = 0;
  const waiting = new Map();
  const listeners = [];
  ws.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message.id && waiting.has(message.id)) {
      const { resolve, reject } = waiting.get(message.id);
      waiting.delete(message.id);
      message.error
        ? reject(new Error(message.error.message))
        : resolve(message.result);
    } else if (message.method) {
      listeners.forEach((fn) => fn(message));
    }
  });
  const ready = new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve);
    ws.addEventListener("error", () => reject(new Error("CDP connect failed")));
  });
  return {
    ready,
    send(method, params) {
      const id = ++seq;
      return new Promise((resolve, reject) => {
        waiting.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params: params || {} }));
        setTimeout(() => {
          if (waiting.has(id)) {
            waiting.delete(id);
            reject(new Error(`${method} timed out`));
          }
        }, 20000);
      });
    },
    on(fn) {
      listeners.push(fn);
    },
    close() {
      try {
        ws.close();
      } catch {
        /* already closed */
      }
    },
  };
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
  const found = findBrowser();
  if (!found) {
    console.log("no Chromium-based browser found; nothing to screenshot");
    process.exit(0);
  }

  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "uaa-shot-"));
  const args = [
    // Without these Chrome cannot start its own sandbox inside a sandboxed
    // environment, and every CDP command then hangs with no error at all.
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    `--window-size=${WIDTH},${HEIGHT}`,
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "about:blank",
  ];
  if (found.kind === "chrome") args.unshift("--headless=new");

  const proc = spawn(found.bin, args, { stdio: ["ignore", "ignore", "pipe"] });
  proc.stderr.on("data", (chunk) => {
    const text = String(chunk);
    if (/error|fail/i.test(text)) process.stderr.write(`[chrome] ${text}`);
  });

  let client;
  try {
    const portFile = path.join(profile, "DevToolsActivePort");
    const deadline = Date.now() + 20000;
    while (!fs.existsSync(portFile)) {
      if (Date.now() > deadline) throw new Error("browser did not start");
      await sleep(150);
    }
    const port = fs.readFileSync(portFile, "utf8").trim().split("\n")[0];
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = targets.find((t) => t.type === "page");
    if (!page) throw new Error("no page target");

    client = cdp(page.webSocketDebuggerUrl);
    await client.ready;
    await client.send("Page.enable");
    await client.send("Runtime.enable");

    const problems = [];
    client.on((message) => {
      if (message.method === "Runtime.exceptionThrown") {
        problems.push(
          message.params.exceptionDetails?.exception?.description ||
            message.params.exceptionDetails?.text ||
            "uncaught exception",
        );
      }
      if (
        message.method === "Runtime.consoleAPICalled" &&
        message.params.type === "error"
      ) {
        problems.push(
          (message.params.args || [])
            .map((a) => a.value ?? a.description ?? "")
            .join(" "),
        );
      }
    });

    await client.send("Page.navigate", { url: URL_UNDER_TEST });

    // Wait for the app to be live, not just for the document to load: the
    // console boots from /api/v1/health, so "the script ran" is the condition
    // that matters.
    const readyBy = Date.now() + 15000;
    let booted = false;
    while (Date.now() < readyBy) {
      const probe = await client.send("Runtime.evaluate", {
        expression:
          "document.readyState + '|' + (typeof window.fetch) + '|' + document.querySelectorAll('#tasks .task-item, #tasks a, #tasks div').length",
        returnByValue: true,
      });
      const value = String(probe.result?.value || "");
      if (value.startsWith("complete|function|")) {
        booted = true;
        break;
      }
      await sleep(150);
    }

    // Let the health call land and the task list render.
    await sleep(1200);

    // Optionally open the most recent task, so the transcript, plan and tool
    // cards are populated. Without this every screenshot is the empty state,
    // which is the one view that says nothing about the design.
    if (process.env.SHOT_OPEN_TASK) {
      await client.send("Runtime.evaluate", {
        expression: `(function () {
          var item = document.querySelector("#tasks .task-item");
          if (!item) return "no task";
          item.click();
          return "clicked";
        })()`,
        returnByValue: true,
      });
      // Give the SSE replay time to render the plan and the tool cards.
      await sleep(Number(process.env.SHOT_SETTLE_MS || 2500));
    }

    const report = await client.send("Runtime.evaluate", {
      expression: `(function () {
        try {
          var cs = getComputedStyle(document.body);
          var aside = document.getElementById("sidebar");
          var composer = document.getElementById("composer");
          return JSON.stringify({
            stylesheetLoaded: cs.fontFamily !== "" && cs.margin !== "",
            bodyFont: cs.fontFamily.slice(0, 40),
            bodyBackground: cs.backgroundColor,
            sidebarWidth: aside ? getComputedStyle(aside).width : null,
            sidebarDisplay: aside ? getComputedStyle(aside).display : null,
            composerVisible: composer ? composer.getBoundingClientRect().height > 0 : false,
            taskRows: document.querySelectorAll("#tasks *").length,
            innerWidth: window.innerWidth,
            scrollHeight: document.documentElement.scrollHeight
          });
        } catch (e) {
          return "probe failed: " + e.message;
        }
      })()`,
      returnByValue: true,
    });

    const shot = await client.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
    });
    fs.writeFileSync(OUT, Buffer.from(shot.data, "base64"));

    console.log(`booted: ${booted}`);
    console.log(`report: ${report.result?.value}`);
    console.log(`console errors: ${problems.length}`);
    problems.slice(0, 5).forEach((p) => console.log(`  - ${p.slice(0, 200)}`));
    console.log(`wrote ${OUT} (${fs.statSync(OUT).size} bytes)`);
  } finally {
    client?.close();
    try {
      proc.kill("SIGKILL");
    } catch {
      /* already gone */
    }
    try {
      fs.rmSync(profile, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  }
}

main().catch((error) => {
  console.error(`failed: ${error.message}`);
  process.exit(1);
});
