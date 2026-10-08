import { inApp, send } from "../native";
import { getPreferences, VolumePref } from "./preferences";

/**
 * The Blades dashboard's own sounds (public/dash-sounds, AAC copies of the
 * pack in assets-source/dash-sounds). Nothing is fetched until sound is turned
 * on, and nothing plays while it is off.
 *
 * Web Audio rather than <audio>, so a focus tick lands on the keypress instead
 * of a beat later. Browsers keep the context suspended until the first click
 * or keypress; it resumes on that gesture.
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

const FILES: Record<SoundName, string> = {
  // A blade slides open: the dashboard gave each blade position its own.
  switch1: "dash_BladeSwitch_1",
  switch2: "dash_BladeSwitch_2",
  switch3: "dash_BladeSwitch_3",
  switch4: "dash_BladeSwitch_4",
  focus: "btn_Focus", // the cursor moves to another row
  select: "btn_Focus", // A: a row, button or option is chosen
  bump: "btn_InactiveFocus", // nowhere further to go
  toggle: "tab_Switch", // day / night
  open: "btn_Focus", // A opens a row's sub-menu
  back: "btn_Back", // B leaves it
  notify: "NotifyPopup" // the "Preference saved" toast
};

const SWITCHES: SoundName[] = ["switch1", "switch2", "switch3", "switch4"];

/** The switch sound for the blade at this position. There are four; blades
    past the fourth reuse the last. */
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
        const response = await fetch(`/dash-sounds/${file}.m4a`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        buffers.set(file, await ctx.decodeAudioData(await response.arrayBuffer()));
      } catch (error) {
        // A missing sound stays silent rather than breaking the page, but
        // says so (the app logs console.error as "Blades page:").
        console.error(`dashboard sound ${file} did not load: ${error}`);
      }
    })
  ).then(() => undefined);
  return loading;
}

export function playSound(name: SoundName) {
  const prefs = getPreferences();
  if (prefs.sound !== "on") return;
  if (inApp()) {
    send({ op: "sound", file: FILES[name], gain: GAIN[prefs.volume] });
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
  output.gain.value = GAIN[prefs.volume];
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(output);
  source.start();
}

// State for the app's debug log (blades_overlay_mac.mm logs it on close).
(window as unknown as { __xeniaSoundDebug: () => unknown }).__xeniaSoundDebug = () => ({
  context: context?.state ?? "none",
  sampleRate: context?.sampleRate,
  loaded: buffers.size,
  loading: loading !== null,
  sound: getPreferences().sound,
  volume: getPreferences().volume
});
