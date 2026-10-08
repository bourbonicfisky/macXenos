import { useEffect, useState } from "react";
import { setPreference } from "./preferences";
import { playSound } from "./sound";
import { announcePreference } from "./toast";

export type Theme = "light" | "dark";

/**
 * Flips day and night, the Y button's job on the legend (and the Y key). It
 * fixes the theme to the opposite of what is showing, leaving "Match my
 * system" — Preferences can put that back.
 */
export function toggleTheme() {
  const showing = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  const next = showing === "dark" ? "light" : "dark";
  setPreference("theme", next);
  // Y keeps its own sound; the toast shows without a second one.
  playSound("toggle");
  announcePreference("theme", next, { sound: false });
}

/**
 * Reads the theme the no-flash script in layout.tsx already resolved, then
 * tracks it. The attribute is the single source of truth so the shader and the
 * ring canvas repaint in step with the CSS tokens.
 */
export function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>("light");

  useEffect(() => {
    const read = () =>
      (document.documentElement.dataset.theme as Theme | undefined) ?? "light";

    setTheme(read());

    const observer = new MutationObserver(() => setTheme(read()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"]
    });

    return () => observer.disconnect();
  }, []);

  return theme;
}

/**
 * The active blade's hue, published on the root element so the background
 * field can wash to match the blade you are standing on.
 */
export function useTone(): string {
  const [tone, setTone] = useState("green");

  useEffect(() => {
    const read = () => document.documentElement.dataset.tone ?? "green";
    setTone(read());
    const observer = new MutationObserver(() => setTone(read()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-tone"]
    });
    return () => observer.disconnect();
  }, []);

  return tone;
}
