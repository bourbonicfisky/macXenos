// Unit tests for the dashboard's text: node --test tests/*.test.mjs (npm test).
// The i18n modules are plain TypeScript, which Node runs directly.

import { test } from "node:test";
import assert from "node:assert/strict";
import { format, placeholders, pseudo } from "../src/i18n/format.ts";
import { localeForTag, matchLocale, LOCALES } from "../src/i18n/locales.ts";
import { en } from "../src/i18n/en.ts";
import { hardCoded } from "../tools/i18n.mjs";

test("the sixteen console languages", () => {
  assert.equal(LOCALES.length, 16);
  assert.equal(new Set(LOCALES.map((l) => l.code)).size, 16);
});

test("a Mac language tag picks our locale", () => {
  assert.equal(localeForTag("en-GB"), "en");
  assert.equal(localeForTag("fr-CA"), "fr");
  assert.equal(localeForTag("pt-BR"), "pt");
  assert.equal(localeForTag("zh-Hant-TW"), "zh-Hant");
  assert.equal(localeForTag("zh-TW"), "zh-Hant");
  assert.equal(localeForTag("zh-HK"), "zh-Hant");
  assert.equal(localeForTag("zh-Hans-CN"), "zh-Hans");
  assert.equal(localeForTag("zh-CN"), "zh-Hans");
  assert.equal(localeForTag("zh"), "zh-Hans");
  assert.equal(localeForTag("no"), "nb");
  assert.equal(localeForTag("nn-NO"), "nb");
  assert.equal(localeForTag("ja_JP"), "ja");
  assert.equal(localeForTag("fi-FI"), null);
});

test("the first Mac language with a translation wins, else English", () => {
  assert.equal(matchLocale(["fi-FI", "de-DE", "en-US"], ["en", "de"]), "de");
  assert.equal(matchLocale(["fi-FI", "de-DE"], ["en"]), "en");
  assert.equal(matchLocale([], ["en", "de"]), "en");
  assert.equal(matchLocale(["zh-TW"], ["en", "zh-Hans"]), "en");
});

test("placeholders and plurals", () => {
  assert.equal(format("{a} of {b}", { a: 3, b: 50 }), "3 of 50");
  assert.equal(format("Hi {name}", {}), "Hi {name}");
  const games = "{count, plural, one {# game} other {# games}}";
  assert.equal(format(games, { count: 1 }), "1 game");
  assert.equal(format(games, { count: 0 }), "0 games");
  assert.equal(format(games, { count: 1500 }), "1,500 games");
  const ru = "{n, plural, one {# файл} few {# файла} many {# файлов} other {# файла}}";
  assert.equal(format(ru, { n: 1 }, "ru"), "1 файл");
  assert.equal(format(ru, { n: 3 }, "ru"), "3 файла");
  assert.equal(format(ru, { n: 11 }, "ru"), "11 файлов");
  assert.equal(format("{n, plural, =0 {none} other {#}}", { n: 0 }), "none");
  assert.deepEqual(placeholders("{a} {n, plural, one {# x {b}} other {y}}"), ["a", "n"]);
});

test("the pseudo-locale accents words and keeps the syntax", () => {
  assert.equal(pseudo("Back"), "[Ɓåçķ]");
  const message = "{count, plural, one {# game} other {# games}} from {name}";
  assert.equal(format(pseudo(message), { count: 2, name: "N" }), "[2 ĝåɱéš ƒŕöɱ N]");
  assert.deepEqual(placeholders(pseudo(message)), placeholders(message));
});

test("every English message formats", () => {
  for (const [key, message] of Object.entries(en)) {
    assert.doesNotThrow(() => format(message, {}), key);
  }
});

test("the hard-coded text scanner sees text but not code", () => {
  const found = hardCoded(`
    export function X() {
      const ok = items.map((i) => <b key={i}>{i}</b>);
      return (<p className="x" title="Hello there">Plain words here</p>);
    }
    const label = { label: "Menu label", glyph: "folder" };
    showToast("Saved", value);
    const fine = t("legend.back");
  `);
  assert.deepEqual(found.sort(), ["Hello there", "Menu label", "Plain words here", "Saved"].sort());
});
