import { en, type MessageKey } from "./en";
import { LOCALES, type LocaleCode } from "./locales";

/**
 * Every language's text. English is en.ts; a translation is
 * catalogs/<code>.ts (zh-Hant.ts, pt.ts...) exporting its messages as
 * `export default { "legend.back": "…" } satisfies Catalog`, and is picked up
 * here without further wiring. A translation may leave keys out; English
 * fills them.
 */
export type Catalog = Partial<Record<MessageKey, string>>;

const modules = import.meta.glob<{ default: Catalog }>("./catalogs/*.ts", { eager: true });

export const CATALOGS: Partial<Record<LocaleCode, Catalog>> = { en };
for (const [path, module] of Object.entries(modules)) {
  const code = path.replace(/^.*\/(.+)\.ts$/, "$1");
  const locale = LOCALES.find((candidate) => candidate.code === code);
  if (locale) CATALOGS[locale.code] = module.default;
}

/** The languages with text to show, in the console's order. */
export const AVAILABLE: LocaleCode[] = LOCALES.filter((locale) => CATALOGS[locale.code]).map(
  (locale) => locale.code
);
