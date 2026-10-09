// Drives the blades page in a headless Chromium over the DevTools protocol,
// against the Vite dev server (so the page runs on the mock in src/mock.ts).
// No dependencies beyond Node 22+ (global fetch and WebSocket).
//
// The browser is the first of BLADES_BROWSER or the usual macOS install
// paths. The clock is frozen and reduced motion is on, so a screenshot of
// the same state is the same bytes run to run.

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BROWSERS = [
  process.env.BLADES_BROWSER,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Helium.app/Contents/MacOS/Helium",
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
].filter(Boolean);

// 2026-10-08 16:20 local: the header clock reads the same every run.
const FROZEN_TIME = new Date(2026, 9, 8, 16, 20, 0).getTime();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(check, what, timeoutMs = 15000) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    try {
      const value = await check();
      if (value) return value;
    } catch {
      // Not up yet.
    }
    if (Date.now() > until) throw new Error(`timed out waiting for ${what}`);
    await sleep(150);
  }
}

function startViteServer(port) {
  const server = spawn("npx", ["vite", "--port", String(port), "--strictPort"], {
    cwd: ROOT,
    stdio: "ignore",
    detached: true
  });
  return server;
}

function startBrowser(port) {
  const path = BROWSERS.find((candidate) => existsSync(candidate));
  if (!path) throw new Error("no Chromium-based browser found; set BLADES_BROWSER");
  const profile = mkdtempSync(join(tmpdir(), "blades-test-"));
  const browser = spawn(
    path,
    [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--force-device-scale-factor=1",
      `--user-data-dir=${profile}`,
      `--remote-debugging-port=${port}`,
      "about:blank"
    ],
    { stdio: "ignore", detached: true }
  );
  return { browser, profile };
}

class Page {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        this.pending.get(message.id)(message);
        this.pending.delete(message.id);
      }
    });
  }

  call(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      this.pending.set(id, (message) =>
        message.error ? reject(new Error(`${method}: ${message.error.message}`)) : resolve(message.result)
      );
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async setup({ colorScheme = "dark" } = {}) {
    await this.call("Emulation.setDeviceMetricsOverride", { width: 1280, height: 752, deviceScaleFactor: 1, mobile: false });
    await this.media(colorScheme);
    await this.call("Page.enable");
    await this.call("Page.addScriptToEvaluateOnNewDocument", {
      source: `(() => {
        const frozen = ${FROZEN_TIME};
        const RealDate = Date;
        class FrozenDate extends RealDate {
          constructor(...args) { super(...(args.length ? args : [frozen])); }
          static now() { return frozen; }
        }
        globalThis.Date = FrozenDate;
      })();`
    });
  }

  async media(colorScheme) {
    await this.call("Emulation.setEmulatedMedia", {
      features: [
        { name: "prefers-color-scheme", value: colorScheme },
        { name: "prefers-reduced-motion", value: "reduce" }
      ]
    });
  }

  /** Loads the page at a hash (#home, #all-games...) from a clean slate. */
  async open(base, hash = "") {
    await this.call("Page.navigate", { url: "about:blank" });
    await this.call("Page.navigate", { url: `${base}/${hash}` });
    const rendered = () => this.eval("!!document.querySelector('.deck .blade')");
    try {
      await waitFor(rendered, "the blades to render", 30000);
    } catch {
      // Vite's first page of a run can stall while it prepares dependencies
      // and reloads; one more load gets the prepared page.
      await this.call("Page.navigate", { url: `${base}/${hash}` });
      await waitFor(rendered, "the blades to render", 60000);
    }
    await sleep(400);
  }

  async eval(expression) {
    const result = await this.call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(`${expression}: ${result.exceptionDetails.text}`);
    return result.result.value;
  }

  /** A key as the page sees it from the keyboard or, via gamepad.ts, a pad. */
  async key(key) {
    const codes = { ArrowUp: 38, ArrowDown: 40, ArrowLeft: 37, ArrowRight: 39, Enter: 13, Escape: 27, End: 35, Home: 36 };
    const keyCode = codes[key] ?? key.toUpperCase().charCodeAt(0);
    const text = key.length === 1 ? key : undefined;
    await this.call("Input.dispatchKeyEvent", { type: "keyDown", key, windowsVirtualKeyCode: keyCode, text });
    await this.call("Input.dispatchKeyEvent", { type: "keyUp", key, windowsVirtualKeyCode: keyCode });
    await sleep(120);
  }

  async screenshot() {
    const { data } = await this.call("Page.captureScreenshot", { format: "png" });
    return Buffer.from(data, "base64");
  }
}

/** Runs `body(page, base)` with a dev server and a browser, then stops both. */
export async function withBlades(body, options = {}) {
  const vitePort = 5300 + Math.floor(Math.random() * 400);
  const debugPort = 9300 + Math.floor(Math.random() * 400);
  const server = startViteServer(vitePort);
  const { browser, profile } = startBrowser(debugPort);
  const base = `http://localhost:${vitePort}`;
  try {
    await waitFor(async () => (await fetch(base)).ok, "the Vite dev server", 30000);
    const targets = await waitFor(
      async () => (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((t) => t.type === "page"),
      "the browser"
    );
    const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve);
      ws.addEventListener("error", reject);
    });
    const page = new Page(ws);
    await page.setup(options);
    try {
      return await body(page, base);
    } finally {
      ws.close();
    }
  } finally {
    for (const child of [browser, server]) {
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {
        // Already gone.
      }
    }
    rmSync(profile, { recursive: true, force: true });
  }
}
