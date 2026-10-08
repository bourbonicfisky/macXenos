import { getNative, inApp, send } from "../native";
import { getPreferences, VolumePref } from "./preferences";

/**
 * The Blades dashboard's sounds. They are not shipped: the app downloads a
 * community pack on request (dashboard_sounds_mac.h) and plays the files
 * natively, under the names below. Until they are there the dashboard is
 * silent. In a browser (npm run dev) they are read from public/dash-sounds,
 * if a developer copied them there; that folder is never committed.
 *
 * Web Audio rather than <audio> in the browser, so a focus tick lands on the
 * keypress instead of a beat later. Browsers keep the context suspended until
 * the first click or keypress; it resumes on that gesture.
 */

export type SoundName =
  | "switch1"
  | "switch2"
  | "switch3"
  | "switch4"
  | "focus"
  | "select"
  | "bump"
  | "toggle"
  | "open"
  | "back"
  | "notify";

// The installed files (without .wav). The pack has one blade-switch sound
// where the dashboard had four, and no separate "nowhere further" tick: that
// is the focus tick, quieter.
const FILES: Record<SoundName, string> = {
  switch1: "blade",
  switch2: "blade",
  switch3: "blade",
  switch4: "blade",
  focus: "focus", // the cursor moves to another row
  select: "select", // A: a row, button or option is chosen
  bump: "focus", // nowhere further to go
  toggle: "toggle", // day / night
  open: "select", // A opens a row's sub-menu
  back: "back", // B leaves it
  notify: "notify" // the "Preference saved" toast
};

const LEVEL: Partial<Record<SoundName, number>> = { bump: 0.45 };

const SWITCHES: SoundName[] = ["switch1", "switch2", "switch3", "switch4"];

/** The switch sound for the blade at this position. */
export function switchSound(index: number): SoundName {
  return SWITCHES[Math.min(Math.max(index, 0), SWITCHES.length - 1)];
}

const GAIN: Record<VolumePref, number> = {
  normal: 1,
  soft: 0.55,
  softer: 0.3,
  quiet: 0.15
};

let context: AudioContext | null = null;
let output: GainNode | null = null;
let loading: Promise<void> | null = null;
// Keyed by file: several sounds share one (A presses and row moves).
const buffers = new Map<string, AudioBuffer>();

function audio(): AudioContext | null {
  if (context) return context;
  const AudioContextClass =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextClass) return null;
  context = new AudioContextClass();
  output = context.createGain();
  output.connect(context.destination);
  const resume = () => {
    if (context?.state === "suspended") void context.resume();
  };
  window.addEventListener("pointerdown", resume, { capture: true });
  window.addEventListener("keydown", resume, { capture: true });
  return context;
}

export function loadSounds(): Promise<void> {
  if (loading) return loading;
  // In the app the sounds play natively (blades_overlay_mac.mm).
  if (inApp()) return Promise.resolve();
  const ctx = audio();
  if (!ctx) return Promise.resolve();
  loading = Promise.all(
    [...new Set(Object.values(FILES))].map(async (file) => {
      try {
        const response = await fetch(`/dash-sounds/${file}.wav`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        buffers.set(file, await ctx.decodeAudioData(await response.arrayBuffer()));
      } catch {
        // Not there: the dashboard stays silent, as in the app before the
        // download.
      }
    })
  ).then(() => undefined);
  return loading;
}

export function playSound(name: SoundName) {
  const prefs = getPreferences();
  if (prefs.sound !== "on") return;
  const gain = GAIN[prefs.volume] * (LEVEL[name] ?? 1);
  if (inApp()) {
    if (getNative().dashSounds.status !== "ready") return;
    send({ op: "sound", file: FILES[name], gain });
    return;
  }
  const ctx = audio();
  if (!ctx || !output) return;
  if (ctx.state === "suspended") void ctx.resume();
  const buffer = buffers.get(FILES[name]);
  if (!buffer) {
    void loadSounds();
    return;
  }
  output.gain.value = gain;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(output);
  source.start();
}
