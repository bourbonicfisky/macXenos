// Unit tests for the dashboard's text: node --test tests/*.test.mjs (npm test).
// The i18n modules are plain TypeScript, which Node runs directly.

import { test } from "node:test";
import assert from "node:assert/strict";
import { format, placeholders, pseudo } from "../src/i18n/format.ts";
import { localeForTag, matchLocale, LOCALES } from "../src/i18n/locales.ts";
import { en } from "../src/i18n/en.ts";
import ru from "../src/i18n/catalogs/ru.ts";
import { catalogProblems, hardCoded, pluralCategories, pluralRules } from "../tools/i18n.mjs";
import { toFormat } from "../tools/native-strings.mjs";
import { readdirSync } from "node:fs";

const CATALOG_DIR = new URL("../src/i18n/catalogs/", import.meta.url);
const catalogs = await Promise.all(
  readdirSync(CATALOG_DIR)
    .filter((file) => file.endsWith(".ts"))
    .map(async (file) => [file.slice(0, -3), (await import(new URL(file, CATALOG_DIR))).default])
);

test("the console's sixteen languages, then Danish and Finnish", () => {
  assert.equal(LOCALES.length, 18);
  assert.equal(new Set(LOCALES.map((l) => l.code)).size, 18);
  assert.deepEqual(LOCALES.slice(16).map((l) => l.code), ["da", "fi"]);
});

test("a Mac language tag picks our locale", () => {
  assert.equal(localeForTag("en-GB"), "en");
  assert.equal(localeForTag("fr-CA"), "fr");
  assert.equal(localeForTag("pt-BR"), "pt-BR");
  assert.equal(localeForTag("pt-PT"), "pt-BR");
  assert.equal(localeForTag("pt"), "pt-BR");
  assert.equal(localeForTag("da-DK"), "da");
  assert.equal(localeForTag("fi-FI"), "fi");
  assert.equal(localeForTag("pl-PL"), "pl");
  assert.equal(localeForTag("ja-JP"), "ja");
  assert.equal(localeForTag("nb-NO"), "nb");
  assert.equal(localeForTag("zh-SG"), "zh-Hans");
  assert.equal(localeForTag("zh-Hant-TW"), "zh-Hant");
  assert.equal(localeForTag("zh-TW"), "zh-Hant");
  assert.equal(localeForTag("zh-HK"), "zh-Hant");
  assert.equal(localeForTag("zh-Hans-CN"), "zh-Hans");
  assert.equal(localeForTag("zh-CN"), "zh-Hans");
  assert.equal(localeForTag("zh"), "zh-Hans");
  assert.equal(localeForTag("no"), "nb");
  assert.equal(localeForTag("nn-NO"), "nb");
  assert.equal(localeForTag("ja_JP"), "ja");
  assert.equal(localeForTag("hu-HU"), null);
  // Regional variants read their language's catalog.
  assert.equal(localeForTag("es-MX"), "es");
  assert.equal(localeForTag("es-419"), "es");
  assert.equal(localeForTag("de-CH"), "de");
  assert.equal(localeForTag("nl-BE"), "nl");
  assert.equal(localeForTag("sv-FI"), "sv");
  assert.equal(localeForTag("it-CH"), "it");
  assert.equal(localeForTag("ko-KR"), "ko");
  assert.equal(localeForTag("zh-MO"), "zh-Hant");
  assert.equal(localeForTag("zh-Hant-HK"), "zh-Hant");
  assert.equal(localeForTag("zh-Hans-HK"), "zh-Hans");
});

test("Traditional Chinese is never what Simplified or plain Chinese read", () => {
  const available = ["en", "zh-Hant"];
  for (const tag of ["zh", "zh-CN", "zh-Hans", "zh-Hans-CN", "zh-SG"]) {
    assert.equal(matchLocale([tag], available), "en", tag);
  }
  for (const tag of ["zh-TW", "zh-HK", "zh-MO", "zh-Hant", "zh-Hant-TW"]) {
    assert.equal(matchLocale([tag], available), "zh-Hant", tag);
  }
  assert.equal(matchLocale(["zh-CN", "zh-TW"], available), "zh-Hant");
  // With a Simplified catalog, the rest of Chinese reads it.
  const both = ["en", "zh-Hant", "zh-Hans"];
  for (const tag of ["zh", "zh-CN", "zh-SG", "zh-Hans", "zh-Hans-HK"]) {
    assert.equal(matchLocale([tag], both), "zh-Hans", tag);
  }
  for (const tag of ["zh-TW", "zh-HK", "zh-MO", "zh-Hant-CN"]) {
    assert.equal(matchLocale([tag], both), "zh-Hant", tag);
  }
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

test("Russian covers the catalog, preserves placeholders and uses Russian plurals", () => {
  assert.deepEqual(Object.keys(ru).sort(), Object.keys(en).sort());
  for (const key of Object.keys(en)) {
    assert.deepEqual(placeholders(ru[key]), placeholders(en[key]), key);
    assert.doesNotThrow(() => format(ru[key], {}, "ru"), key);
  }
  for (const [count, expected] of [[0, "0 папок"], [1, "1 папка"], [2, "2 папки"], [5, "5 папок"], [11, "11 папок"], [21, "21 папка"], [22, "22 папки"], [25, "25 папок"], [111, "111 папок"], [1.5, "1,5 папки"]]) {
    assert.equal(format(ru["home.gameFolders.count"], { count }, "ru"), expected);
  }
  assert.equal(format(ru["profiles.pictures.installed"], {
    pictures: 22, titles: 5, collection: "Collection", site: "Site"
  }, "ru"), "22 аватара из 5 игр: официальная библиотека аватаров Xbox 360 из коллекции Collection на Site (archive.org).");
  assert.equal(matchLocale(["ru-RU", "en-US"], ["en", "ru"]), "ru");
});

test("every catalog is complete and correct for its language", () => {
  assert.ok(catalogs.length > 0);
  for (const [code, catalog] of catalogs) {
    assert.ok(LOCALES.some((locale) => locale.code === code), `${code} is a dashboard language`);
    assert.deepEqual(catalogProblems(code, catalog, en), [], code);
    for (const key of Object.keys(en)) {
      assert.doesNotThrow(() => format(catalog[key], { count: 2, days: 3 }, code), `${code} ${key}`);
    }
  }
});

test("plural categories follow Intl.PluralRules", () => {
  assert.deepEqual(pluralCategories("{n, plural, =0 {none} one {# a {b}} other {# as}} and {m, plural, other {#}}"), [["one", "other"], ["other"]]);
  assert.deepEqual([...pluralRules("ko").required], ["other"]);
  assert.deepEqual([...pluralRules("zh-Hant").allowed], ["other"]);
  assert.deepEqual([...pluralRules("ru").required].sort(), ["few", "many", "one", "other"]);
  // French has "many" (for millions) but whole numbers below that never pick it.
  assert.deepEqual([...pluralRules("fr").required].sort(), ["one", "other"]);
  const en1 = { "x.count": "{count, plural, one {# game} other {# games}}" };
  assert.deepEqual(catalogProblems("ko", { "x.count": "{count, plural, other {게임 #개}}" }, en1), []);
  assert.match(catalogProblems("ko", { "x.count": "{count, plural, one {#} other {#}}" }, en1).join(), /doesn't have: one/);
  assert.match(catalogProblems("ru", { "x.count": "{count, plural, one {#} other {#}}" }, en1).join(), /lacks few, many/);
  assert.deepEqual([...pluralRules("pl").required], ["one", "few", "many", "other"]);
  assert.match(catalogProblems("pl", { "x.count": "{count, plural, one {#} few {#} other {#}}" }, en1).join(), /lacks many/);
  assert.deepEqual([...pluralRules("fi").allowed], ["one", "other"]);
  assert.match(catalogProblems("fi", { "x.count": "{count, plural, one {#} few {#} other {#}}" }, en1).join(), /doesn't have: few/);
  assert.deepEqual([...pluralRules("ja").allowed], ["other"]);
  assert.match(catalogProblems("de", {}, en1).join(), /1 keys missing/);
  assert.match(catalogProblems("de", { "x.count": "{n}", "x.extra": "" }, en1).join(), /English doesn't have: x\.extra.*placeholders differ/);
});

test("native messages become Localizable.strings formats", () => {
  assert.equal(toFormat("Hide Others"), "Hide Others");
  assert.equal(toFormat("About {app}"), "About %@");
  assert.equal(toFormat("{app} beenden", ["app"]), "%@ beenden");
  assert.equal(toFormat("{b} 100% {a}", ["a", "b"]), "%2$@ 100%% %1$@");
  // A message without placeholders isn't a format, so its % stays.
  assert.equal(toFormat("100%"), "100%");
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

// Exercise the actual catalogs, including plural branches missed by a single
// count=2 sample and reordered placeholders in CJK translations.
test("all translated messages retain parameters across plural boundaries", () => {
  for (const [code, catalog] of [["en", en], ...catalogs]) {
    for (const [key, message] of Object.entries(catalog)) {
      const names = placeholders(message);
      const pluralNames = new Set([...message.matchAll(/\{(\w+),\s*plural,/g)].map(m => m[1]));
      for (const count of pluralNames.size ? [0, 1, 2, 3, 5, 11, 21, 22, 101, 1.5, 1000000] : [1]) {
        const params = Object.fromEntries(names.map(name => [name, pluralNames.has(name) ? count : `⟦${name}⟧`]));
        const result = format(message, params, code);
        const context = `${code} ${key} (${count})`;
        assert.ok(result.trim(), context);
        assert.doesNotMatch(result, /[{}]/, context);
        for (const name of names) {
          if (!pluralNames.has(name)) assert.ok(result.includes(params[name]), `${context}: lost ${name}`);
        }
        if (pluralNames.size) assert.ok(result.includes(new Intl.NumberFormat(code).format(count)), `${context}: lost count`);
      }
    }
  }
});
