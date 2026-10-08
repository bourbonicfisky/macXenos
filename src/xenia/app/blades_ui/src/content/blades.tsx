import type { BladeDef } from "../ui/blade";
import { send, type NativeState } from "../native";
import { NEXT_LAUNCH } from "../ui/settings-controls";
import { advanced } from "./advanced";
import { audio } from "./audio";
import { controller } from "./controller";
import { cpu } from "./cpu";
import { graphics } from "./graphics";
import { home } from "./home";
import { profiles } from "./profiles";
import { system } from "./system";

/* ==========================================================================
   The blades, left to right. Each blade's words live in its own file; this
   one only orders them, and adds the tray every blade shares.

   Each blade is a dashboard menu: rows down the left, and a pane on the right
   that follows whichever row has focus. A row's pane holds its settings;
   A opens them, B comes back. Written for players, not developers.
   ========================================================================== */

/** Audio and CPU share a blade: the audio settings first, then the CPU's,
    each under its own heading. Keeps the audio blade's id and tone, so links
    to #audio/... still land. */
function audioAndCpu(state: NativeState): BladeDef {
  const sound = audio(state);
  const processor = cpu(state);
  return {
    ...sound,
    label: "audio & cpu",
    title: "Audio & CPU",
    menu: [
      ...sound.menu.map((item, index) => (index === 0 ? { ...item, section: "Audio" } : item)),
      ...processor.menu.map((item, index) => (index === 0 ? { ...item, section: "CPU" } : item))
    ]
  };
}

export function buildBlades(state: NativeState): BladeDef[] {
  const blades = [
    home(state),
    graphics(state),
    controller(state),
    audioAndCpu(state),
    profiles(state),
    system(state),
    advanced(state)
  ];

  // A change that only applies at the next launch, made while a game runs:
  // the tray restarts the game with it, as Apply & Relaunch did.
  const waiting =
    state.titleOpen &&
    Object.values(state.cvars).some((cvar) => cvar.changed && NEXT_LAUNCH.has(cvar.name));
  if (waiting) {
    const tray = {
      label: "Restart the game to apply your changes",
      onSelect: () => send({ op: "action", id: "relaunch" })
    };
    for (const blade of blades) blade.tray = tray;
  }
  return blades;
}
