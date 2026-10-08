import { useSyncExternalStore } from "react";

/* ==========================================================================
   The bridge to the emulator (blades_overlay_mac.mm / blades_model.cc).

   The app pushes its whole state as JSON through window.__xeniaReceive; the
   page posts small messages back through the "xenia" WebKit message handler.
   Everything the blades show comes from this state, and every change goes
   back as a message, so nothing here owns a setting.

   Opened in a plain browser (npm run dev), there is no app: a mock state
   stands in so the blades can be built and screenshotted on their own.
   ========================================================================== */

export type CvarType = "bool" | "int" | "uint" | "float" | "string" | "path";
export type CvarValue = boolean | number | string;

export type Cvar = {
  name: string;
  category: string;
  type: CvarType;
  value: CvarValue;
  defaultValue: CvarValue;
  description: string;
  /** Changed since this process started, for "applies next launch". */
  changed: boolean;
};

export type Renderer = "metal" | "vulkan" | "none";

export type Controller = {
  slot: number;
  present: boolean;
  kind: string;
  name: string;
};

export type Hotkey = {
  id: number;
  /** "A + Guide = Toggle Fullscreen": the buttons, then what they do. */
  label: string;
  enabled: boolean;
  /** Works during a game (needs controller_hotkeys); otherwise home screen. */
  inGame: boolean;
};

export type RebindOutput = { output: string; label: string; binding: string };

export type RecentTitle = {
  name: string;
  path: string;
  lastPlayed: number;
  /** The game's own icon (/title-icons/ID.png), or "" if not seen yet. */
  icon: string;
};

export type MusicPlayer = {
  /** The player is on (enable_xmp) and exists. */
  available: boolean;
  state: "idle" | "playing" | "paused";
  volume: number;
};

export type Profile = {
  /** 16 hex digits; too wide for a JavaScript number. */
  xuid: string;
  gamertag: string;
  /** The player slot it is signed in to, or -1. */
  slot: number;
};

export type ConsoleOption = { value: number; label: string };

export type ConsoleField = {
  id: string;
  label: string;
  type: "choice" | "bool" | "float" | "text";
  value: number | boolean | string;
  options?: ConsoleOption[];
  /** Read-only (shown, never edited), e.g. the console's network identity. */
  readOnly?: boolean;
};

/** The dashboard's sounds: not shipped, downloaded on request from a
    community pack (dashboard_sounds_mac.h), which the page credits. */
export type DashSounds = {
  status: "missing" | "downloading" | "ready" | "failed" | "unavailable";
  /** Why the last download failed. */
  error: string;
  /** The first-run offer has been answered, either way. */
  asked: boolean;
  pack?: string;
  author?: string;
  site?: string;
};

export type NativeState = {
  /** False until the app has answered "ready"; the mock is always true. */
  connected: boolean;
  renderer: Renderer;
  /** The renderer the `gpu` setting picks for the next launch. */
  selectedRenderer: Renderer;
  titleOpen: boolean;
  titleName: string;
  /** Upscalers the running presenter offers (MetalFX is Metal-only). */
  upscalers: string[];
  cvars: Record<string, Cvar>;
  controllers: Controller[];
  hotkeys: Hotkey[];
  rebind: {
    slot: number;
    name: string;
    capturing: string;
    outputs: RebindOutput[];
  };
  console: {
    /** The console settings can't change under a running game. */
    editable: boolean;
    fields: ConsoleField[];
  };
  timeScalar: number;
  recentTitles: RecentTitle[];
  musicPlayer: MusicPlayer;
  profiles: Profile[];
  dashSounds: DashSounds;
  /** The release, "1.0 beta 3", for the Dashboard blade. */
  appVersion: string;
  /** The build: branch@commit (date). */
  appBuild: string;
};

export type NativeMessage =
  | { op: "ready" }
  | { op: "set"; name: string; value: CvarValue }
  | { op: "reset"; name: string }
  | { op: "resetAll" }
  | { op: "console"; id: string; value: number | boolean | string }
  | { op: "consoleReset" }
  | { op: "hotkey"; id: number; enabled: boolean }
  | { op: "assign"; from: number; to: number }
  | { op: "rebind"; output: string }
  | { op: "rebindCancel" }
  | { op: "rebindReset" }
  | { op: "action"; id: string; value?: number }
  | { op: "sound"; file: string; gain: number }
  | { op: "dashSounds"; action: "download" | "decline" }
  | { op: "launch"; path: string }
  | { op: "openGame" }
  | { op: "installContent" }
  | { op: "stopGame" }
  | { op: "music"; id: "pause" | "resume" | "next" | "previous" | "volume"; value?: number }
  | { op: "profileCreate"; gamertag: string }
  | { op: "profileSignIn"; xuid: string; slot: number }
  | { op: "profileSignOut"; xuid: string }
  | { op: "profileRename"; xuid: string; gamertag: string }
  | { op: "profileDelete"; xuid: string }
  | { op: "profileFolder"; xuid: string }
  | { op: "close" };

type WebKitHandler = { postMessage(message: unknown): void };

function handler(): WebKitHandler | null {
  const webkit = (window as unknown as {
    webkit?: { messageHandlers?: { xenia?: WebKitHandler } };
  }).webkit;
  return webkit?.messageHandlers?.xenia ?? null;
}

/** True inside the app; false in a browser on the mock. */
export const inApp = () => handler() !== null;

let state: NativeState = emptyState();
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

/** The app calls this with its whole state, as JSON. */
(window as unknown as { __xeniaReceive: (json: string) => void }).__xeniaReceive = (
  json: string
) => {
  try {
    state = { ...JSON.parse(json), connected: true } as NativeState;
    emit();
  } catch (error) {
    console.error("xenia state", error);
  }
};

/** The app calls this each time it shows the dashboard. */
export const SHOWN_EVENT = "xenia-shown";
(window as unknown as { __xeniaShown: () => void }).__xeniaShown = () => {
  window.dispatchEvent(new CustomEvent(SHOWN_EVENT));
};

export function send(message: NativeMessage) {
  const native = handler();
  if (native) {
    native.postMessage(message);
    return;
  }
  // Outside the app only the dev server has a stand-in; a shipped page with
  // no app behind it has nothing to talk to.
  if (!import.meta.env.DEV) return;
  void import("./mock").then(({ mockApply }) => {
    state = mockApply(state, message);
    emit();
  });
}

let started = false;
function start() {
  if (started) return;
  started = true;
  if (handler()) {
    send({ op: "ready" });
  } else if (import.meta.env.DEV) {
    void import("./mock").then(({ mockState }) => {
      state = mockState();
      emit();
    });
  }
}

function subscribe(listener: () => void) {
  start();
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useNative(): NativeState {
  return useSyncExternalStore(subscribe, () => state, () => state);
}

export function getNative(): NativeState {
  return state;
}

/** Sets a setting right away in the page (so the UI answers at once) and
    tells the app, which applies and saves it and pushes the new state. */
export function setCvar(name: string, value: CvarValue) {
  const cvar = state.cvars[name];
  if (cvar) {
    state = {
      ...state,
      cvars: { ...state.cvars, [name]: { ...cvar, value, changed: true } }
    };
    emit();
  }
  send({ op: "set", name, value });
}

function emptyState(): NativeState {
  return {
    connected: false,
    renderer: "none",
    selectedRenderer: "none",
    titleOpen: false,
    titleName: "",
    upscalers: ["bilinear", "cas", "fsr"],
    cvars: {},
    controllers: [],
    hotkeys: [],
    rebind: { slot: -1, name: "", capturing: "", outputs: [] },
    console: { editable: true, fields: [] },
    timeScalar: 1,
    recentTitles: [],
    musicPlayer: { available: false, state: "idle", volume: 1 },
    profiles: [],
    dashSounds: { status: "unavailable", error: "", asked: true },
    appVersion: "",
    appBuild: ""
  };
}
