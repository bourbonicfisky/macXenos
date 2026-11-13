import { useSyncExternalStore } from "react";
import { AVAILABLE, CATALOGS } from "./catalogs";
import { en, type MessageKey } from "./en";
import { format, pseudo, type MessageParams } from "./format";
import { matchLocale, SOURCE_LOCALE, type LocaleCode } from "./locales";

/**
 * The dashboard's text in the reader's language.
 *
 *   t("legend.back")                      "Back"
 *   t("allGames.achievements", { unlocked: 3, total: 50 })
 *
 * The language is the Dashboard language preference (System > Dashboard):
 * "Same as my Mac" by default, which picks the first of the Mac's preferred
 * languages that has a translation, else English. Any key a translation
 * lacks falls back to English.
 *
 * The pseudo-locale (PSEUDO), offered only in development and used by the
 * tests, accents every message so text that bypasses t() stands out.
 */

export type { MessageKey } from "./en";
export { AVAILABLE } from "./catalogs";
export { LOCALES, type LocaleCode } from "./locales";

export const PSEUDO = "en-XA";
export type LanguageChoice = "system" | LocaleCode | typeof PSEUDO;
export type ActiveLanguage = LocaleCode | typeof PSEUDO;

let choice: LanguageChoice = "system";
// The Mac's preferred languages, most preferred first: the app reports them
// (NSLocale); in a browser, the browser's.
let systemLanguages: readonly string[] = typeof navigator === "undefined" ? [] : navigator.languages;
let active: ActiveLanguage = resolve();
const listeners = new Set<() => void>();

function resolve(): ActiveLanguage {
  if (choice === PSEUDO) return PSEUDO;
  if (choice !== "system") return AVAILABLE.includes(choice) ? choice : SOURCE_LOCALE;
  return matchLocale(systemLanguages, AVAILABLE);
}

function update() {
  const next = resolve();
  const changed = next !== active;
  active = next;
  if (typeof document !== "undefined") {
    document.documentElement.lang = next === PSEUDO ? "en" : next;
    document.title = t("dashboard.pageTitle");
  }
  if (!changed) return;
  for (const listener of listeners) listener();
}

/** Called by preferences.ts when the Dashboard language preference changes. */
export function setLanguageChoice(next: LanguageChoice) {
  choice = next;
  update();
}

/** Called with the app's state: the Mac's languages, when it reports them. */
export function setSystemLanguages(tags: readonly string[]) {
  if (!tags.length || tags.join() === systemLanguages.join()) return;
  systemLanguages = tags;
  update();
}

export function getLanguage(): ActiveLanguage {
  return active;
}

/** The text for `key` in the current language. */
export function t(key: MessageKey, params?: MessageParams): string {
  if (active === PSEUDO) return format(pseudo(en[key]), params, "en");
  const message = CATALOGS[active]?.[key] ?? en[key];
  return format(message, params, active);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The current language; a component using it redraws when it changes. */
export function useLanguage(): ActiveLanguage {
  return useSyncExternalStore(subscribe, getLanguage, getLanguage);
}
