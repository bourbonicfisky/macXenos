import { useSyncExternalStore } from "react";
import { AVAILABLE, LOCALES, PSEUDO, setLanguageChoice, t, type LanguageChoice, type MessageKey } from "../i18n";

/**
 * The Preferences blade's settings, kept in localStorage and applied to the
 * root element. layout.tsx's no-flash script reads the same keys before first
 * paint; this store takes over once the page is running.
 *
 *   theme   "system" follows the OS (stored as no key at all, which is what
 *           the site has always done), or a fixed "light" / "dark"
 *   motion  "fast" is the dashboard's own speed; "slow" eases a little;
 *           "reduced" switches blades instantly (data-motion, globals.css)
 *   sound   on unless turned off (sound.ts)
 *   volume  "normal" and three softer levels; "soft" by default
 *   notifications  where the toast appears; bottom center, as on the 360
 *   motionOverride  "on" lets the chosen speed win over the OS's
 *           reduce-motion setting (data-motion-override, globals.css)
 *   language  the dashboard's language (i18n/): "system" follows the Mac,
 *           or one of the languages with a translation
 */

export type ThemePref = "system" | "light" | "dark";
export type MotionPref = "fast" | "slow" | "reduced";
export type SoundPref = "off" | "on";
export type VolumePref = "normal" | "soft" | "softer" | "quiet";
export type OverridePref = "off" | "on";
export type NotificationPref =
  | "top-left"
  | "top"
  | "top-right"
  | "bottom-left"
  | "bottom"
  | "bottom-right";

export type Preferences = {
  language: LanguageChoice;
  theme: ThemePref;
  motion: MotionPref;
  sound: SoundPref;
  volume: VolumePref;
  motionOverride: OverridePref;
  notifications: NotificationPref;
};

export const PREFERENCE_KEYS: Record<keyof Preferences, string> = {
  language: "macxenos-language",
  theme: "macxenos-theme",
  motion: "macxenos-motion",
  sound: "macxenos-sound",
  volume: "macxenos-volume",
  motionOverride: "macxenos-motion-override",
  notifications: "macxenos-notifications"
};

const DEFAULTS: Preferences = {
  language: "system",
  theme: "system",
  motion: "fast",
  sound: "on",
  volume: "soft",
  motionOverride: "off",
  notifications: "bottom"
};

const ALLOWED: { [K in keyof Preferences]: readonly Preferences[K][] } = {
  language: ["system", ...LOCALES.map((locale) => locale.code), PSEUDO],
  theme: ["system", "light", "dark"],
  motion: ["fast", "slow", "reduced"],
  sound: ["off", "on"],
  volume: ["normal", "soft", "softer", "quiet"],
  motionOverride: ["off", "on"],
  notifications: ["top-left", "top", "top-right", "bottom-left", "bottom", "bottom-right"]
};

/** The settings offered as a row of options (the override is a checkbox). */
export type ChoicePref = Exclude<keyof Preferences, "motionOverride">;

/** Text on an option: a catalog key, or (a language's own name) as is. */
export type ChoiceText = MessageKey | { raw: string };
export const choiceText = (text: ChoiceText) => (typeof text === "string" ? t(text) : text.raw);

export type Choice<K extends keyof Preferences> = {
  value: Preferences[K];
  /** On the option itself. */
  label: ChoiceText;
  /** At the right of the setting's menu row; the label when not given. */
  short?: ChoiceText;
};

export const PREFERENCE_CHOICES: { [K in ChoicePref]: Choice<K>[] } = {
  // Same as my Mac, then each language with a translation, in its own
  // words; the pseudo-locale only in development.
  language: [
    { value: "system", label: "pref.language.system" },
    ...AVAILABLE.map((code) => ({
      value: code,
      label: { raw: LOCALES.find((locale) => locale.code === code)!.name }
    })),
    ...(import.meta.env.DEV ? [{ value: PSEUDO, label: { raw: "Pseudo" } } as Choice<"language">] : [])
  ],
  theme: [
    { value: "system", label: "pref.theme.system", short: "pref.theme.system.short" },
    { value: "light", label: "legend.day" },
    { value: "dark", label: "legend.night" }
  ],
  motion: [
    { value: "fast", label: "pref.motion.fast" },
    { value: "slow", label: "pref.motion.slow" },
    { value: "reduced", label: "pref.motion.reduced", short: "pref.motion.reduced.short" }
  ],
  sound: [
    { value: "off", label: "common.off" },
    { value: "on", label: "common.on" }
  ],
  volume: [
    { value: "normal", label: "pref.volume.normal" },
    { value: "soft", label: "pref.volume.soft" },
    { value: "softer", label: "pref.volume.softer" },
    { value: "quiet", label: "pref.volume.quiet" }
  ],
  notifications: [
    { value: "top-left", label: "pref.notifications.top-left" },
    { value: "top", label: "pref.notifications.top", short: "pref.notifications.top.short" },
    { value: "top-right", label: "pref.notifications.top-right" },
    { value: "bottom-left", label: "pref.notifications.bottom-left" },
    { value: "bottom", label: "pref.notifications.bottom", short: "pref.notifications.bottom.short" },
    { value: "bottom-right", label: "pref.notifications.bottom-right" }
  ]
};

export const PREFERENCE_NAMES: Record<keyof Preferences, MessageKey> = {
  language: "pref.language",
  theme: "pref.theme",
  motion: "pref.motion",
  sound: "pref.sound",
  volume: "pref.volume",
  motionOverride: "pref.motionOverride",
  notifications: "pref.notifications"
};

/** "Theme: Night", for the Preference saved toast. */
export function describePreference<K extends keyof Preferences>(
  key: K,
  value: Preferences[K]
): string {
  const name = t(PREFERENCE_NAMES[key]);
  if (key === "motionOverride") {
    return t("toast.preferenceValue", { name, value: t(value === "on" ? "common.on" : "common.off") });
  }
  const choice = (PREFERENCE_CHOICES[key as ChoicePref] as Choice<K>[]).find(
    (c) => c.value === value
  );
  return t("toast.preferenceValue", { name, value: choice ? choiceText(choice.label) : String(value) });
}

let current: Preferences = DEFAULTS;
let started = false;
const listeners = new Set<() => void>();

function read(): Preferences {
  const next = { ...DEFAULTS };
  try {
    for (const key of Object.keys(PREFERENCE_KEYS) as (keyof Preferences)[]) {
      const stored = window.localStorage.getItem(PREFERENCE_KEYS[key]);
      if (stored && (ALLOWED[key] as readonly string[]).includes(stored)) {
        (next as Record<string, string>)[key] = stored;
      }
    }
  } catch {
    // Storage blocked: run on the defaults.
  }
  return next;
}

const systemDark = () => window.matchMedia("(prefers-color-scheme: dark)");

function apply(prefs: Preferences) {
  const root = document.documentElement;
  root.dataset.theme =
    prefs.theme === "system" ? (systemDark().matches ? "dark" : "light") : prefs.theme;
  root.dataset.motion = prefs.motion;
  root.dataset.motionOverride = prefs.motionOverride;
  setLanguageChoice(prefs.language);
}

/** Whether the OS asks for reduced motion — the override box only shows then. */
export const systemReducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)");

function emit() {
  for (const listener of listeners) listener();
}

function start() {
  if (started || typeof window === "undefined") return;
  started = true;
  current = read();
  apply(current);
  // "Match my system" keeps matching if the OS flips while the page is open.
  systemDark().addEventListener("change", () => {
    if (current.theme === "system") apply(current);
  });
  // Another tab changed a setting.
  window.addEventListener("storage", (event) => {
    if (event.key && Object.values(PREFERENCE_KEYS).includes(event.key)) {
      current = read();
      apply(current);
      emit();
    }
  });
}

export function getPreferences(): Preferences {
  start();
  return current;
}

export function setPreference<K extends keyof Preferences>(key: K, value: Preferences[K]) {
  start();
  current = { ...current, [key]: value };
  try {
    if (value === DEFAULTS[key]) window.localStorage.removeItem(PREFERENCE_KEYS[key]);
    else window.localStorage.setItem(PREFERENCE_KEYS[key], value);
  } catch {
    // Private browsing: the choice just will not survive a reload.
  }
  apply(current);
  emit();
}

function subscribe(listener: () => void) {
  start();
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function usePreferences(): Preferences {
  return useSyncExternalStore(subscribe, getPreferences, () => DEFAULTS);
}
