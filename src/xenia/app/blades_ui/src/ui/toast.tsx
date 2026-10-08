import { useEffect, useState, useSyncExternalStore } from "react";
import {
  describePreference,
  getPreferences,
  Preferences,
  usePreferences
} from "./preferences";
import { loadSounds, playSound } from "./sound";

/**
 * The 360's achievement-style toast: a dark glossy capsule that pops in with
 * the notify sound (rising, with a slight overshoot), holds, and sinks away as
 * it fades. Used for "Preference saved". One at
 * a time; a new one replaces the last. Where it appears is a preference —
 * bottom center by default, as on the 360.
 */

type ToastMessage = { id: number; title: string; detail: string };

const SHOW_MS = 3200;
// The exit animation's length (content.css toast-out), with a margin; the
// toast is removed then even if no animationend arrives (reduced motion).
const LEAVE_MS = 320;

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

  // The toast on screen holds, then leaves; a new one replaces it at once.
  const [leaving, setLeaving] = useState<number | null>(null);
  const finish = (id: number) => {
    if (message?.id === id) {
      message = null;
      emit();
    }
  };

  useEffect(() => {
    if (!current) return;
    setLeaving(null);
    const hold = window.setTimeout(() => setLeaving(current.id), SHOW_MS);
    const gone = window.setTimeout(() => finish(current.id), SHOW_MS + LEAVE_MS + 120);
    return () => {
      window.clearTimeout(hold);
      window.clearTimeout(gone);
    };
  }, [current]);

  return (
    <div
      className="toast-region pointer-events-none fixed inset-x-0 z-50 flex"
      data-position={notifications}
      role="status"
      aria-live="polite"
    >
      {current ? (
        <div
          key={current.id}
          className="toast flex items-center gap-[0.75em]"
          data-leaving={leaving === current.id || undefined}
          onAnimationEnd={(event) => {
            if (event.animationName.startsWith("toast-out")) finish(current.id);
          }}
        >
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
