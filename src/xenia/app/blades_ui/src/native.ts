import { useSyncExternalStore } from "react";
import { setSystemLanguages } from "./i18n";

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

/** Every config setting the emulator defines, by name: the keys of
    mock-cvars.json, which tools/gen-mock-cvars.py builds from the DEFINE_*
    macros (npm run check fails if it is out of date). A name the emulator
    doesn't have is a type error, not a row that silently shows nothing. */
export type CvarName = keyof typeof import("./mock-cvars.json");
export type CvarValue = boolean | number | string;

export type Cvar = {
  name: CvarName;
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
  /** The first signed-in player's progress in it, when they have played it. */
  achievements?: { unlocked: number; total: number; score: number; maxScore: number };
};

/** The folders the app looks in for games (game_library.h). */
export type GameFolders = {
  folders: string[];
  /** A scan is running; the list fills in when it ends. */
  scanning: boolean;
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
  /** Every setting the app reports; a few may be missing (another
      platform's), so look them up rather than assume them. */
  cvars: { [K in CvarName]?: Cvar };
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
  /** Every game in the game folders and every game played here, once each,
      in no particular order (All Games sorts them). Never played: 0. */
  allGames: RecentTitle[];
  gameFolders: GameFolders;
  musicPlayer: MusicPlayer;
  profiles: Profile[];
  dashSounds: DashSounds;
  /** The Mac's preferred languages, most preferred first (BCP 47); empty
      off macOS. The dashboard's "Same as my Mac" picks from these. */
  systemLanguages: string[];
  /** The release, "1.0 beta 3", for the Dashboard blade. */
  appVersion: string;
  /** The build: branch@commit (date). */
  appBuild: string;
};

export type NativeMessage =
  | { op: "ready" }
  | { op: "set"; name: CvarName; value: CvarValue }
  | { op: "reset"; name: CvarName }
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
  | { op: "gameFolderAdd" }
  | { op: "gameFolderRemove"; path: string }
  | { op: "gameFoldersRescan" }
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

/** The app's state and this page's NativeState are written by hand on both
    sides (blades_model.cc, above). A field one side renamed shows up here,
    once, as a "Blades page:" line in the app's log rather than as a blank
    row nobody notices. */
let contractChecked = false;
function checkContract(received: Record<string, unknown>) {
  if (contractChecked) return;
  contractChecked = true;
  const expected = Object.keys(emptyState()).filter((key) => key !== "connected");
  const missing = expected.filter((key) => !(key in received));
  const unknown = Object.keys(received).filter((key) => !expected.includes(key));
  if (missing.length || unknown.length) {
    console.error(
      `state from the app does not match NativeState: missing ${missing.join(", ") || "none"}; unknown ${unknown.join(", ") || "none"}`
    );
  }
}

/** The app calls this with its whole state, as JSON. */
(window as unknown as { __xeniaReceive: (json: string) => void }).__xeniaReceive = (
  json: string
) => {
  try {
    const received = JSON.parse(json) as Record<string, unknown>;
    checkContract(received);
    state = { ...received, connected: true } as NativeState;
    setSystemLanguages(state.systemLanguages ?? []);
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
export function setCvar(name: CvarName, value: CvarValue) {
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
    allGames: [],
    gameFolders: { folders: [], scanning: false },
    musicPlayer: { available: false, state: "idle", volume: 1 },
    profiles: [],
    dashSounds: { status: "unavailable", error: "", asked: true },
    systemLanguages: [],
    appVersion: "",
    appBuild: ""
  };
}
