import { useEffect, useSyncExternalStore } from "react";
import {
  describePreference,
  getPreferences,
  Preferences,
  usePreferences
} from "./preferences";
import { loadSounds, playSound } from "./sound";

/**
 * The 360's achievement-style toast: a dark glossy capsule that slides in with
 * the NotifyPopup sound, holds, and goes. Used for "Preference saved". One at
 * a time; a new one replaces the last. Where it appears is a preference —
 * bottom center by default, as on the 360.
 */

type ToastMessage = { id: number; title: string; detail: string };

const SHOW_MS = 3200;

let message: ToastMessage | null = null;
let nextId = 1;
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

export function showToast(title: string, detail: string, { sound = true } = {}) {
  message = { id: nextId++, title, detail };
  emit();
  // Turning sound on from Preferences: the sounds may still be loading.
  if (sound && getPreferences().sound === "on") {
    void loadSounds().then(() => playSound("notify"));
  }
}

/** "Preference saved — Theme: Night". */
export function announcePreference<K extends keyof Preferences>(
  key: K,
  value: Preferences[K],
  options?: { sound?: boolean }
) {
  showToast("Preference saved", describePreference(key, value), options);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function Toast() {
  const current = useSyncExternalStore(subscribe, () => message, () => null);
  const { notifications } = usePreferences();

  useEffect(() => {
    if (!current) return;
    const timer = window.setTimeout(() => {
      if (message?.id === current.id) {
        message = null;
        emit();
      }
    }, SHOW_MS);
    return () => window.clearTimeout(timer);
  }, [current]);

  return (
    <div
      className="toast-region pointer-events-none fixed inset-x-0 z-50 flex"
      data-position={notifications}
      role="status"
      aria-live="polite"
    >
      {current ? (
        <div key={current.id} className="toast flex items-center gap-[0.75em]">
          <img className="toast-orb" src="/macxenos-icon-dark.webp" alt="" width={48} height={48} />
          <span className="grid leading-[1.25]">
            <span className="toast-title">{current.title}</span>
            <span className="toast-detail">{current.detail}</span>
          </span>
        </div>
      ) : null}
    </div>
  );
}
