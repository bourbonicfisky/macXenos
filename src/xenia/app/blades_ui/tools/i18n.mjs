// The dashboard's text: what is translatable, what still isn't, and how far
// each translation has got. Run by npm run check; on its own:
//
//   node tools/i18n.mjs                  the report, exit 1 on a problem
//   node tools/i18n.mjs --update-baseline   record today's hard-coded counts
//
// Checks:
//   - every key in src/i18n/en.ts is used somewhere (a key nothing reads is
//     text a translator would translate for nothing);
//   - every translation (src/i18n/catalogs/<code>.ts) uses only English's
//     keys and keeps each message's placeholders;
//   - the English still written straight into the sources (not through t())
//     never grows: tools/i18n-baseline.json holds each file's count, and a
//     file may only go down. New files start at 0.
// Reports each language's coverage, and how much text is still hard-coded.

import { readFileSync, readdirSync, statSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(ROOT, "src");
const BASELINE = join(ROOT, "tools", "i18n-baseline.json");

function files(dir, test) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...files(path, test));
    else if (test(path)) out.push(path);
  }
  return out;
}

/** "{a} and {n, plural, one {# x} other {# y}}" -> ["a", "n"] (format.ts). */
function placeholders(message) {
  const names = new Set();
  let depth = 0;
  for (let i = 0; i < message.length; i++) {
    if (message[i] === "{") {
      if (depth === 0) {
        const match = /^\{\s*([A-Za-z_][A-Za-z0-9_]*)/.exec(message.slice(i));
        if (match) names.add(match[1]);
      }
      depth++;
    } else if (message[i] === "}") depth--;
  }
  return [...names].sort().join(",");
}

// --- English text written straight into a source file -----------------------

const TEXT_PROPS = ["label", "short", "heading", "title", "section", "placeholder", "aria-label", "alt", "description"];

function stripComments(source) {
  return source
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:"'`\\])\/\/.*$/gm, "$1");
}

const WORDS = /[A-Za-z]{2,}/;
// Names that read the same in every language.
// (BACK and START are printed on the controller itself, as the drawing shows.)
const UNTRANSLATED = new Set(["macXenos", "macxenos.com", "Pseudo", "Metal", "Vulkan", "MoltenVK", "MetalFX", "FSR", "CAS", "BACK", "START"]);
// What code looks like, as opposed to words between tags.
const CODE = /[=;()]|&&|\|\||[,:]\s*\n|^[,:.?]|\breturn\b|\bconst\b/;
const looksLikeKey = (text) => /^[a-z][\w-]*(\.[\w-]+)+$/.test(text.trim());

/** The pieces of user-facing English in a .tsx file that bypass t(). */
export function hardCoded(source) {
  const text = stripComments(source);
  const found = [];
  // JSX text: between a tag's ">" (not an arrow) and the next "<" or "{".
  for (const match of text.matchAll(/(?<![=-])>([^<>{}]+?)(?=[<{])/g)) {
    const piece = match[1].trim();
    if (!piece || !WORDS.test(piece) || CODE.test(match[1]) || UNTRANSLATED.has(piece)) continue;
    found.push(piece);
  }
  // Text after an expression and before a tag: "{x} games<".
  for (const match of text.matchAll(/\}([^<>{}]+?)(?=<\/)/g)) {
    const piece = match[1].trim();
    if (!piece || !WORDS.test(piece) || CODE.test(match[1]) || UNTRANSLATED.has(piece)) continue;
    found.push(piece);
  }
  // Text props and object fields: label: "…", title="…", aria-label={`…`}.
  const props = TEXT_PROPS.map((p) => p.replace("-", "\\-")).join("|");
  const prop = new RegExp(`(?:^|[\\s{,(])(?:${props})\\s*[:=]\\s*\\{?\\s*(["'\`])((?:(?!\\1)[^\\\\]|\\\\.)*)\\1`, "gm");
  for (const match of text.matchAll(prop)) {
    const piece = match[2].replace(/\$\{[^}]*\}/g, "").trim();
    if (WORDS.test(piece) && !looksLikeKey(piece) && !UNTRANSLATED.has(piece)) found.push(match[2]);
  }
  // Toasts and announcements given English directly.
  for (const match of text.matchAll(/\b(?:showToast|announceSetting)\(\s*(["'`])((?:(?!\1).)*)\1/g)) {
    if (WORDS.test(match[2])) found.push(match[2]);
  }
  return found;
}

// --- The report -----------------------------------------------------------------

export async function report({ updateBaseline = false, log = console.log } = {}) {
  const problems = [];
  const { en } = await import(join(SRC, "i18n", "en.ts"));
  const keys = Object.keys(en);

  // Every key used somewhere outside en.ts.
  const sources = files(SRC, (p) => /\.(ts|tsx)$/.test(p) && !p.endsWith(join("i18n", "en.ts")));
  const corpus = sources.map((p) => readFileSync(p, "utf8")).join("\n");
  const unused = keys.filter((key) => !corpus.includes(`"${key}"`) && !corpus.includes(`'${key}'`));
  if (unused.length) problems.push(`keys nothing uses (remove them from en.ts): ${unused.join(", ")}`);

  // Translations: known keys, same placeholders; coverage.
  const catalogDir = join(SRC, "i18n", "catalogs");
  const catalogs = existsSync(catalogDir) ? files(catalogDir, (p) => p.endsWith(".ts")) : [];
  log(`Languages: en (source, ${keys.length} messages)`);
  for (const path of catalogs) {
    const code = path.replace(/^.*\/(.+)\.ts$/, "$1");
    const catalog = (await import(path)).default ?? {};
    const unknown = Object.keys(catalog).filter((key) => !(key in en));
    if (unknown.length) problems.push(`${code}: keys English doesn't have: ${unknown.join(", ")}`);
    const mismatched = Object.keys(catalog).filter(
      (key) => key in en && placeholders(catalog[key]) !== placeholders(en[key])
    );
    if (mismatched.length) problems.push(`${code}: placeholders differ from English in ${mismatched.join(", ")}`);
    const done = keys.filter((key) => typeof catalog[key] === "string" && catalog[key].trim()).length;
    log(`           ${code} ${done}/${keys.length} (${Math.round((100 * done) / keys.length)}%)`);
  }

  // Hard-coded English: may only go down.
  const counts = {};
  for (const path of files(SRC, (p) => p.endsWith(".tsx"))) {
    const n = hardCoded(readFileSync(path, "utf8")).length;
    if (n) counts[relative(ROOT, path)] = n;
  }
  const baseline = existsSync(BASELINE) ? JSON.parse(readFileSync(BASELINE, "utf8")) : {};
  if (updateBaseline) {
    writeFileSync(BASELINE, JSON.stringify(counts, Object.keys(counts).sort(), 2) + "\n");
    log(`Recorded ${Object.keys(counts).length} files in tools/i18n-baseline.json`);
  } else {
    for (const [file, n] of Object.entries(counts)) {
      const allowed = baseline[file] ?? 0;
      if (n > allowed) {
        const pieces = hardCoded(readFileSync(join(ROOT, file), "utf8")).slice(0, 6).map((p) => `"${p}"`);
        problems.push(
          `${file}: ${n} hard-coded strings, was ${allowed}; put new text in src/i18n/en.ts and use t(). Some: ${pieces.join(", ")}`
        );
      }
    }
    const lower = Object.entries(baseline).filter(([file, n]) => (counts[file] ?? 0) < n);
    if (lower.length) log(`Fewer hard-coded strings than recorded in ${lower.length} files: run npm run i18n:baseline to lock that in.`);
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const tsx = files(SRC, (p) => p.endsWith(".tsx")).length;
  log(`Hard-coded English: ${total} strings in ${Object.keys(counts).length} of ${tsx} files (all text through t() in the other ${tsx - Object.keys(counts).length})`);
  return problems;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const problems = await report({ updateBaseline: process.argv.includes("--update-baseline") });
  for (const problem of problems) console.log(`FAIL  ${problem}`);
  process.exit(problems.length ? 1 : 0);
}
