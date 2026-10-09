// npm run check: the things that can go stale without anything failing.
//
//   1. src/mock-cvars.json matches the emulator's DEFINE_* settings (it is
//      the list CvarName checks every setting name in the page against).
//   2. The page type-checks.
//   3. The dashboard's text (tools/i18n.mjs): every English key used, every
//      translation keeps English's placeholders, and no file gains English
//      written straight into it instead of through t().
//   4. dist/, which is checked in and is what the app ships, is what the
//      sources build to now.
//
// Each failure says how to fix it. Exit status 1 if any failed.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { report as textReport } from "./i18n.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCES = join(ROOT, "..", "..");
let failed = 0;

function step(name, body) {
  try {
    const problem = body();
    if (problem) {
      failed++;
      console.log(`FAIL  ${name}\n      ${problem}`);
    } else {
      console.log(`ok    ${name}`);
    }
  } catch (error) {
    failed++;
    console.log(`FAIL  ${name}\n      ${String(error.stdout ?? error.message).trim().split("\n").slice(0, 12).join("\n      ")}`);
  }
}

function files(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...files(path));
    else out.push(path);
  }
  return out;
}

const scratch = mkdtempSync(join(tmpdir(), "blades-check-"));
try {
  step("settings list matches the emulator's DEFINE_* settings", () => {
    const fresh = join(scratch, "cvars.json");
    execFileSync("python3", [join(ROOT, "tools", "gen-mock-cvars.py"), SOURCES, fresh], { stdio: "pipe" });
    const want = JSON.parse(readFileSync(fresh, "utf8"));
    const have = JSON.parse(readFileSync(join(ROOT, "src", "mock-cvars.json"), "utf8"));
    const added = Object.keys(want).filter((name) => !(name in have));
    const removed = Object.keys(have).filter((name) => !(name in want));
    const changed = Object.keys(want).filter(
      (name) => name in have && JSON.stringify(want[name]) !== JSON.stringify(have[name])
    );
    if (added.length || removed.length || changed.length) {
      return `out of date (new: ${added.join(", ") || "none"}; gone: ${removed.join(", ") || "none"}; changed: ${changed.join(", ") || "none"}). Run npm run gen:cvars.`;
    }
  });

  step("page type-checks", () => {
    execFileSync("npx", ["tsc", "--noEmit"], { cwd: ROOT, stdio: "pipe" });
  });

  const textProblems = await textReport({ log: (line) => console.log(`      ${line}`) });
  if (textProblems.length) {
    failed++;
    console.log(`FAIL  the dashboard's text\n      ${textProblems.join("\n      ")}`);
  } else {
    console.log("ok    the dashboard's text");
  }

  step("dist/ is what the sources build to", () => {
    const out = join(scratch, "dist");
    execFileSync("npx", ["vite", "build", "--outDir", out, "--emptyOutDir", "--logLevel", "error"], {
      cwd: ROOT,
      stdio: "pipe"
    });
    const built = files(out).map((path) => relative(out, path)).sort();
    const shipped = files(join(ROOT, "dist")).map((path) => relative(join(ROOT, "dist"), path)).sort();
    const missing = built.filter((path) => !shipped.includes(path));
    const extra = shipped.filter((path) => !built.includes(path));
    const differ = built.filter(
      (path) => shipped.includes(path) && !readFileSync(join(out, path)).equals(readFileSync(join(ROOT, "dist", path)))
    );
    if (missing.length || extra.length || differ.length) {
      return `stale (${[...missing, ...extra, ...differ].join(", ")}). Run npm run build and commit dist/.`;
    }
  });
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
