/**
 * The dashboard's languages: the console's own sixteen (the Language setting
 * under System > Language and region), each named in itself. Pure, so the
 * tests can run it in Node.
 */

export const LOCALES = [
  { code: "en", name: "English" },
  { code: "ja", name: "日本語" },
  { code: "de", name: "Deutsch" },
  { code: "fr", name: "Français" },
  { code: "es", name: "Español" },
  { code: "it", name: "Italiano" },
  { code: "ko", name: "한국어" },
  { code: "zh-Hant", name: "繁體中文" },
  { code: "pt", name: "Português" },
  { code: "pl", name: "Polski" },
  { code: "ru", name: "Русский" },
  { code: "sv", name: "Svenska" },
  { code: "tr", name: "Türkçe" },
  { code: "nb", name: "Norsk" },
  { code: "nl", name: "Nederlands" },
  { code: "zh-Hans", name: "简体中文" }
] as const;

export type LocaleCode = (typeof LOCALES)[number]["code"];

export const SOURCE_LOCALE: LocaleCode = "en";

/** Which of our locales a BCP 47 tag (en-GB, zh-TW, pt-BR, no) means. */
export function localeForTag(tag: string): LocaleCode | null {
  const parts = tag.toLowerCase().replace(/_/g, "-").split("-");
  const language = parts[0];
  if (language === "zh") {
    // Chinese by script when the tag says it, else by region: Taiwan, Hong
    // Kong and Macau write Traditional, everywhere else Simplified.
    if (parts.includes("hant")) return "zh-Hant";
    if (parts.includes("hans")) return "zh-Hans";
    return parts.some((part) => ["tw", "hk", "mo"].includes(part)) ? "zh-Hant" : "zh-Hans";
  }
  // Norwegian: Bokmål, Nynorsk and plain "no" all read the Norwegian text.
  if (language === "nb" || language === "nn" || language === "no") return "nb";
  const match = LOCALES.find((locale) => locale.code === language);
  return match ? match.code : null;
}

/**
 * The first of `preferred` (the Mac's languages, most preferred first) that
 * has text to show, else English.
 */
export function matchLocale(preferred: readonly string[], available: readonly LocaleCode[]): LocaleCode {
  for (const tag of preferred) {
    const locale = localeForTag(tag);
    if (locale && available.includes(locale)) return locale;
  }
  return SOURCE_LOCALE;
}
