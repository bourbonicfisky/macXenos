import { useSyncExternalStore } from "react";
import {
  Choice,
  ChoicePref,
  PREFERENCE_CHOICES,
  PREFERENCE_NAMES,
  Preferences,
  setPreference,
  systemReducedMotion,
  usePreferences
} from "./preferences";
import { announcePreference } from "./toast";

/** The current setting, shown at the right of its menu row. */
export function PreferenceValue({ pref }: { pref: ChoicePref }) {
  const prefs = usePreferences();
  const choice = (PREFERENCE_CHOICES[pref] as Choice<typeof pref>[]).find((c) => c.value === prefs[pref]);
  return <>{choice?.short}</>;
}

/** The Sound row's value: "Off", or the volume when sound is on. */
export function SoundValue() {
  const prefs = usePreferences();
  if (prefs.sound !== "on") return <>Off</>;
  const choice = PREFERENCE_CHOICES.volume.find((c) => c.value === prefs.volume);
  return <>{choice?.short}</>;
}

/** The options for one setting, as a row of selectable sheets. */
export function PreferenceOptions({ pref }: { pref: ChoicePref }) {
  const prefs = usePreferences();

  const choose = (value: Preferences[typeof pref]) => {
    if (value === prefs[pref]) return;
    setPreference(pref, value);
    // The toast's NotifyPopup is the confirmation; turning sound on plays it
    // once the sounds have loaded, at the chosen volume.
    announcePreference(pref, value);
  };

  return (
    <div
      role="radiogroup"
      aria-label={PREFERENCE_NAMES[pref]}
      className="mt-[0.3em] flex flex-wrap gap-[0.5em]"
    >
      {(PREFERENCE_CHOICES[pref] as Choice<typeof pref>[]).map((choice) => (
        <button
          key={choice.value}
          type="button"
          role="radio"
          aria-checked={prefs[pref] === choice.value}
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => choose(choice.value)}
        >
          {choice.label}
        </button>
      ))}
    </div>
  );
}

function subscribeReducedMotion(listener: () => void) {
  const query = systemReducedMotion();
  query.addEventListener("change", listener);
  return () => query.removeEventListener("change", listener);
}

/**
 * Only shown when the OS asks for reduced motion: lets the speed chosen here
 * win instead. Choosing Reduced motion above still applies either way.
 */
export function ReducedMotionOverride() {
  const prefs = usePreferences();
  const systemReduced = useSyncExternalStore(
    subscribeReducedMotion,
    () => systemReducedMotion().matches,
    () => false
  );
  if (!systemReduced) return null;

  return (
    <label className="pref-check mt-[0.2em] flex cursor-pointer items-center gap-[0.55em]">
      <input
        type="checkbox"
        className="pref-check-box"
        checked={prefs.motionOverride === "on"}
        onChange={(event) => {
          const value = event.target.checked ? "on" : "off";
          setPreference("motionOverride", value);
          announcePreference("motionOverride", value);
        }}
      />
      Override reduced motion
    </label>
  );
}
