// Layout check for translations: npm run layout [-- <out-dir>] [codes...]
//
// For each language (all of them, or those named), opens every blade and
// every row's pane, All Games and Game Folders on the mock, at the test
// window size (1280 x 752), and lists text that doesn't fit: text cut off by
// a box that clips it (overflow hidden, an ellipsis), text running past the
// window's edge, and option sheets, tabs and legend buttons that wrapped onto
// a second line. English runs too, as the baseline: what English already
// does isn't the translation's fault. With an out-dir, writes layout.json and
// a screenshot of each blade per language there.
//
// Needs the same Chromium as the smoke tests (tests/harness.mjs).

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { withBlades } from "./harness.mjs";

const BLADES = ["home", "graphics", "controller", "audio", "profiles", "system", "advanced"];
const SUB_BLADES = ["all-games", "game-folders"];
const args = process.argv.slice(2);
const outDir = args[0] && !/^[a-z]{2}(-[A-Za-z]+)?$/.test(args[0]) ? args.shift() : null;

// In the page: every visible piece of text that doesn't fit, with where it is.
const SCAN = `(() => {
  const found = [];
  const visible = (el) => {
    for (let e = el; e; e = e.parentElement) {
      if (e.hidden) return false;
      const cs = getComputedStyle(e);
      if (cs.display === "none" || cs.visibility === "hidden" || Number(cs.opacity) === 0) return false;
      if (e.matches(".blade:not([data-state=open]) .blade-body, [aria-hidden=true]")) return false;
    }
    return true;
  };
  const where = (el) => {
    const parts = [];
    for (let e = el; e && e !== document.body && parts.length < 3; e = e.parentElement) {
      const cls = [...e.classList].filter((c) => /^(blade|menu|legend|pref|pane|subblade|game|tab|pad|hotkey|sound|toast)/.test(c))[0];
      if (cls) parts.unshift(cls);
    }
    return parts.join(" > ") || el.tagName.toLowerCase();
  };
  const SINGLE_LINE = ".blade-tab, .legend-entry, .legend-toggle, .pref-option, .pane-button, .menu-row, .subblade-tab, .pref-tag";
  const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent.trim();
    const el = node.parentElement;
    // The straight tab labels are screen-reader copies of the curved ones.
    if (!text || !el || seen.has(el) || el.closest(".sr-only, svg") || !visible(el)) continue;
    seen.add(el);
    const range = document.createRange();
    range.selectNodeContents(node);
    const r = range.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const label = text.length > 80 ? text.slice(0, 77) + "…" : text;
    // Cut off sideways by an ancestor that clips. (Up and down, the
    // blades scroll, and text below the fold is not cut off.)
    for (let a = el; a && a !== document.documentElement; a = a.parentElement) {
      const cs = getComputedStyle(a);
      if (cs.overflowX === "visible" || cs.overflowX === "auto" || cs.overflowX === "scroll") continue;
      const b = a.getBoundingClientRect();
      if (r.bottom < b.top || r.top > b.bottom) break;
      const over = Math.max(r.right - b.right, b.left - r.left);
      if (over > 1.5) {
        found.push({ kind: "clipped", text: label, where: where(el), px: Math.round(over) });
        break;
      }
    }
    if (getComputedStyle(el).textOverflow === "ellipsis" && el.scrollWidth > el.clientWidth + 1) {
      found.push({ kind: "ellipsis", text: label, where: where(el), px: el.scrollWidth - el.clientWidth });
    }
    if (r.right > innerWidth + 1) found.push({ kind: "offscreen", text: label, where: where(el), px: Math.round(r.right - innerWidth) });
    const single = el.closest(SINGLE_LINE);
    if (single) {
      const lines = new Set([...range.getClientRects()].filter((q) => q.width > 1).map((q) => Math.round(q.top)));
      if (lines.size > 1) found.push({ kind: "wrapped", text: label, where: where(el), px: lines.size });
    }
  }
  // The tabs' labels run along a curve (tab-label.tsx): they don't fit when
  // they run past either end of their tab.
  for (const text of document.querySelectorAll(".blade-tab svg text, .subblade svg text")) {
    if (!visible(text)) continue;
    const tab = text.closest(".blade-tab, .blade-tab-wrap, button, section")?.getBoundingClientRect();
    const r = text.getBoundingClientRect();
    if (!tab || !r.height) continue;
    const over = Math.max(r.bottom - Math.min(tab.bottom, innerHeight), Math.max(tab.top, 0) - r.top);
    if (over > 1.5) found.push({ kind: "tab", text: text.textContent, where: "blade-tab", px: Math.round(over) });
  }
  return found;
})()`;

const codes = args.length
  ? args
  : (await import("node:fs")).readdirSync(new URL("../src/i18n/catalogs/", import.meta.url))
      .filter((f) => f.endsWith(".ts"))
      .map((f) => f.slice(0, -3));
const languages = ["en", ...codes.filter((c) => c !== "en")];
const report = {};

await withBlades(async (page, base) => {
  await page.call("Emulation.setFocusEmulationEnabled", { enabled: true });
  for (const code of languages) {
    const issues = new Map();
    const note = (state, list) => {
      for (const issue of list) {
        const key = `${issue.kind}|${issue.where}|${issue.text}`;
        if (!issues.has(key)) issues.set(key, { ...issue, state });
      }
    };
    const setLanguage = async (hash) => {
      await page.open(base, hash);
      if ((await page.eval("document.documentElement.lang")) !== code) {
        await page.eval(`localStorage.setItem("macxenos-language", ${JSON.stringify(code)})`);
        await page.open(base, hash);
      }
    };
    for (const blade of BLADES) {
      await setLanguage(`#${blade}`);
      const rows = await page.eval("document.querySelectorAll('.blade[data-state=open] .menu-row').length");
      if (outDir) {
        mkdirSync(join(outDir, code), { recursive: true });
        writeFileSync(join(outDir, code, `${blade}.png`), await page.screenshot());
      }
      for (let i = 0; i < rows; i++) {
        await page.eval(`document.querySelectorAll('.blade[data-state=open] .menu-row')[${i}]?.focus()`);
        await page.eval("new Promise((r) => setTimeout(r, 60))");
        const row = await page.eval(`document.querySelectorAll('.blade[data-state=open] .menu-row')[${i}]?.textContent`);
        note(`#${blade} / ${row}`, await page.eval(SCAN));
      }
    }
    for (const sub of SUB_BLADES) {
      await setLanguage(`#${sub}`);
      await page.eval("new Promise((r) => setTimeout(r, 400))");
      note(`#${sub}`, await page.eval(SCAN));
      if (outDir) writeFileSync(join(outDir, code, `${sub}.png`), await page.screenshot());
    }
    report[code] = [...issues.values()];
    // Keep completed languages even if a later browser navigation fails.
    if (outDir) writeFileSync(join(outDir, "layout.json"), JSON.stringify(report, null, 2));
    console.log(`${code}: ${report[code].length} pieces of text that don't fit`);
  }
});

// What a translation adds over English.
const english = new Set((report.en ?? []).map((i) => `${i.kind}|${i.where}`));
for (const code of languages.filter((c) => c !== "en")) {
  const extra = report[code].filter((i) => !english.has(`${i.kind}|${i.where}`) || i.kind === "wrapped");
  if (!extra.length) continue;
  console.log(`\n${code}:`);
  for (const i of extra) console.log(`  ${i.kind.padEnd(9)} ${String(i.px).padStart(4)}  ${i.where}  "${i.text}"  (${i.state})`);
}
if (outDir) writeFileSync(join(outDir, "layout.json"), JSON.stringify(report, null, 2));
