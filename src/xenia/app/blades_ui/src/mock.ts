import cvars from "./mock-cvars.json";
import { keptByResetAll } from "./content/advanced";
import type { Cvar, CvarName, NativeMessage, NativeState } from "./native";

/* A stand-in for the app, for `npm run dev` in a browser. mock-cvars.json is
   every config setting with its source default (regenerate it from the
   DEFINE_* macros when settings change); the rest is a plausible console. */

export function mockState(): NativeState {
  const state: NativeState = {
    connected: true,
    renderer: "metal",
    selectedRenderer: "metal",
    titleOpen: false,
    titleName: "",
    upscalers: ["bilinear", "cas", "fsr", "metalfx"],
    cvars: structuredClone(cvars as unknown as Record<string, Cvar>),
    controllers: [
      { slot: 0, present: true, kind: "Xbox", name: "Xbox Wireless Controller" },
      { slot: 1, present: false, kind: "Unknown", name: "" },
      { slot: 2, present: false, kind: "Unknown", name: "" },
      { slot: 3, present: false, kind: "Unknown", name: "" }
    ],
    hotkeys: [
      { id: 1, label: "Y + Guide = Toggle Fullscreen", enabled: true, inGame: true },
      { id: 2, label: "Right Shoulder + Guide = Clear GPU Cache", enabled: true, inGame: true },
      { id: 3, label: "Left Shoulder + Guide = Toggle Controller Vibration", enabled: true, inGame: true },
      { id: 4, label: "D-PAD Down + Guide = Half CPU Scalar", enabled: true, inGame: true },
      { id: 5, label: "A + Guide = Toggle Readback Resolve (Vulkan only)", enabled: true, inGame: true },
      { id: 6, label: "B + Guide = Toggle between loglevel set in config and the 'Disabled' loglevel.", enabled: false, inGame: true },
      { id: 7, label: "Start = Run Selected Title", enabled: true, inGame: false },
      { id: 8, label: "D-PAD Down = Title Selection +1", enabled: true, inGame: false },
      { id: 9, label: "D-PAD Up = Title Selection -1", enabled: true, inGame: false }
    ],
    rebind: {
      slot: 0,
      name: "Xbox Wireless Controller",
      capturing: "",
      outputs: [
        { output: "a", label: "A / Cross", binding: "b0" },
        { output: "b", label: "B / Circle", binding: "b1" },
        { output: "x", label: "X / Square", binding: "b2" },
        { output: "y", label: "Y / Triangle", binding: "b3" },
        { output: "leftshoulder", label: "Left bumper (LB / L1)", binding: "b4" },
        { output: "rightshoulder", label: "Right bumper (RB / R1)", binding: "b5" },
        { output: "lefttrigger", label: "Left trigger (LT / L2)", binding: "a2" },
        { output: "righttrigger", label: "Right trigger (RT / R2)", binding: "a5" },
        { output: "back", label: "Back / Share", binding: "b6" },
        { output: "start", label: "Start / Options", binding: "b7" },
        { output: "guide", label: "Guide / Home", binding: "b8" },
        { output: "leftstick", label: "Left stick click (LS / L3)", binding: "b9" },
        { output: "rightstick", label: "Right stick click (RS / R3)", binding: "b10" },
        { output: "dpup", label: "D-pad up", binding: "h0.1" },
        { output: "dpdown", label: "D-pad down", binding: "h0.4" },
        { output: "dpleft", label: "D-pad left", binding: "h0.8" },
        { output: "dpright", label: "D-pad right", binding: "h0.2" }
      ]
    },
    console: {
      editable: true,
      fields: [
        { id: "language", label: "Language", type: "choice", value: 1, options: [
          { value: 1, label: "English" }, { value: 2, label: "Japanese" }, { value: 3, label: "German" },
          { value: 4, label: "French" }, { value: 5, label: "Spanish" }, { value: 6, label: "Italian" }
        ] },
        { id: "country", label: "Country", type: "choice", value: 103, options: [
          { value: 103, label: "United States" }, { value: 39, label: "Great Britain" }, { value: 36, label: "Japan" }
        ] },
        { id: "time_zone", label: "Time zone", type: "choice", value: 4, options: [
          { value: 3, label: "(GMT-08:00) Pacific Time" }, { value: 4, label: "(GMT-07:00) Mountain Time" },
          { value: 25, label: "(GMT) Greenwich Mean Time" }
        ] },
        { id: "dst_off", label: "Disable daylight saving time", type: "bool", value: false },
        { id: "clock_24h", label: "24-hour clock", type: "bool", value: false },
        { id: "av_region", label: "AV region", type: "choice", value: 0x00400100, options: [
          { value: 0x00400100, label: "NTSC" }, { value: 0x00400200, label: "NTSC-J" },
          { value: 0x00400400, label: "PAL" }, { value: 0x00800300, label: "PAL 50Hz" }
        ] },
        { id: "resolution", label: "Resolution", type: "choice", value: 2, options: [
          { value: 1, label: "640x480" }, { value: 2, label: "1280x720" }, { value: 3, label: "1920x1080" }
        ] },
        { id: "widescreen", label: "Widescreen", type: "bool", value: true },
        { id: "audio_mono", label: "Mono", type: "bool", value: false },
        { id: "dolby_pro_logic", label: "Dolby Pro Logic", type: "bool", value: false },
        { id: "dolby_digital", label: "Dolby Digital", type: "bool", value: true },
        { id: "music_volume", label: "Music player volume", type: "float", value: 1 },
        { id: "parental_control", label: "Parental control", type: "bool", value: false },
        { id: "mac_address", label: "MAC address", type: "text", value: "7C:ED:8D:12:34:56", readOnly: true }
      ]
    },
    timeScalar: 1,
    recentTitles: [
      { name: "Sonic Generations", path: "/Games/Sonic Generations (World).iso", lastPlayed: 1791530000, icon: "/macxenos-icon-dark.webp", achievements: { unlocked: 12, total: 50, score: 175, maxScore: 1000 } },
      { name: "Just Cause 2", path: "/Games/Just Cause 2.iso", lastPlayed: 1791440000, icon: "", achievements: { unlocked: 3, total: 49, score: 45, maxScore: 1000 } },
      { name: "Viva Piñata", path: "/Games/Viva Pinata.iso", lastPlayed: 1790900000, icon: "" },
      { name: "Red Dead Redemption", path: "/Games/Red Dead Redemption.iso", lastPlayed: 1790500000, icon: "", achievements: { unlocked: 0, total: 50, score: 0, maxScore: 1000 } },
      { name: "Gears of War 3", path: "/Games/Gears of War 3.iso", lastPlayed: 1790100000, icon: "", achievements: { unlocked: 7, total: 50, score: 80, maxScore: 1000 } },
      { name: "Halo 3", path: "/Games/Halo 3.iso", lastPlayed: 1789900000, icon: "", achievements: { unlocked: 1, total: 49, score: 5, maxScore: 1000 } },
      { name: "Assassin's Creed", path: "/Games/Assassins Creed.iso", lastPlayed: 1789000000, icon: "" },
      { name: "Ultra Street Fighter IV", path: "/Games/USF4.iso", lastPlayed: 1788000000, icon: "", achievements: { unlocked: 20, total: 50, score: 310, maxScore: 1000 } },
      { name: "LEGO Star Wars II", path: "/Games/LEGO Star Wars II.iso", lastPlayed: 1787000000, icon: "" }
    ],
    gameFolders: { folders: ["/Volumes/Plex/Emulation/Xbox 360 Decompressed"], scanning: false },
    musicPlayer: { available: false, state: "idle", volume: 1 },
    // In the browser the sounds come from public/dash-sounds if a developer
    // put them there (never committed).
    dashSounds: {
      status: "missing",
      error: "",
      // true skips the first-run offer; set false to see it.
      asked: true,
      pack: "Xbox 360 Blades",
      author: "squabbled",
      site: "DeckThemes"
    },
    profiles: [
      {
        xuid: "E030000012345678",
        gamertag: "dmug",
        slot: 0,
        picture: "/gamer-pictures/robot.png",
        stats: { games: 10, gamerscore: 1250, achievements: 56 }
      },
      { xuid: "E030000087654321", gamertag: "Player Two", slot: -1, picture: "" }
    ],
    gamerPictures: {
      status: "missing",
      error: "",
      pictures: 0,
      titles: 0,
      bytes: 0,
      received: 0,
      total: 116904804,
      collection: "Xbox 360 Profile Picture Collection",
      site: "the Internet Archive"
    },
    systemLanguages: [],
    appVersion: "1.0 beta 3",
    appBuild: "codex/metal-foundation-integration@4774abdb6 (dev)",
    updater: { available: true, asked: true },
    allGames: []
  };
  // The folder holds a few games not played yet, besides the played ones.
  state.allGames = [
    ...state.recentTitles,
    ...["Alan Wake", "Bayonetta", "Crackdown", "Dead Rising", "Fable II", "Mass Effect"].map((name) => ({
      name,
      path: `/Volumes/Plex/Emulation/Xbox 360 Decompressed/${name}.iso`,
      lastPlayed: 0,
      icon: ""
    }))
  ];
  return state;
}

export function mockApply(state: NativeState, message: NativeMessage): NativeState {
  switch (message.op) {
    case "profilePicture":
      return {
        ...state,
        profiles: state.profiles.map((p) => (p.xuid === message.xuid ? { ...p, picture: message.picture } : p))
      };
    case "gamerPicturesDelete":
      return { ...state, gamerPictures: { ...state.gamerPictures, status: "missing", pictures: 0, titles: 0, bytes: 0 } };
    case "gamerPicturesReveal":
      return state;
    case "gamerPicturesDownload":
      return { ...state, gamerPictures: { ...state.gamerPictures, status: "downloading", received: 41_000_000 } };
    case "gameFolderAdd":
      return { ...state, gameFolders: { ...state.gameFolders, folders: [...state.gameFolders.folders, "/Volumes/Games"] } };
    case "pickFolder": {
      const cvar = state.cvars[message.name];
      if (!cvar) return state;
      return { ...state, cvars: { ...state.cvars, [message.name]: { ...cvar, value: "/Volumes/Games/Chosen", changed: true } } };
    }
    case "gameFolderRemove":
      return {
        ...state,
        gameFolders: { ...state.gameFolders, folders: state.gameFolders.folders.filter((f) => f !== message.path) }
      };
    case "updateOffer":
      return {
        ...state,
        updater: { ...state.updater, asked: true },
        cvars: state.cvars.check_for_updates
          ? {
              ...state.cvars,
              check_for_updates: { ...state.cvars.check_for_updates, value: message.schedule }
            }
          : state.cvars
      };
    case "dashSounds":
      return {
        ...state,
        dashSounds: {
          ...state.dashSounds,
          asked: true,
          status: message.action === "download" ? "ready" : state.dashSounds.status
        }
      };
    case "set": {
      const cvar = state.cvars[message.name];
      if (!cvar) return state;
      const next = { ...state, cvars: { ...state.cvars, [message.name]: { ...cvar, value: message.value, changed: true } } };
      if (message.name === "gpu") {
        const v = String(message.value);
        next.selectedRenderer = v === "vulkan" ? "vulkan" : "metal";
      }
      return next;
    }
    case "reset": {
      const cvar = state.cvars[message.name];
      if (!cvar) return state;
      return { ...state, cvars: { ...state.cvars, [message.name]: { ...cvar, value: cvar.defaultValue, changed: true } } };
    }
    case "resetAll": {
      const cvars = { ...state.cvars };
      for (const [name, cvar] of Object.entries(cvars)) {
        if (keptByResetAll(cvar)) continue;
        if (cvar.value !== cvar.defaultValue) cvars[name as CvarName] = { ...cvar, value: cvar.defaultValue, changed: true };
      }
      return { ...state, cvars };
    }
    case "console":
      return {
        ...state,
        console: {
          ...state.console,
          fields: state.console.fields.map((f) => (f.id === message.id ? { ...f, value: message.value } : f))
        }
      };
    case "profileSignIn":
      return {
        ...state,
        profiles: state.profiles.map((p) =>
          p.xuid === message.xuid ? { ...p, slot: message.slot } : p.slot === message.slot ? { ...p, slot: -1 } : p
        )
      };
    case "profileSignOut":
      return { ...state, profiles: state.profiles.map((p) => (p.xuid === message.xuid ? { ...p, slot: -1 } : p)) };
    case "profileRename":
      return {
        ...state,
        profiles: state.profiles.map((p) => (p.xuid === message.xuid ? { ...p, gamertag: message.gamertag } : p))
      };
    case "profileDelete":
      return { ...state, profiles: state.profiles.filter((p) => p.xuid !== message.xuid) };
    case "profileCreate":
      return {
        ...state,
        profiles: [
          ...state.profiles,
          { xuid: `E0300000${Date.now().toString(16).slice(-8).toUpperCase()}`, gamertag: message.gamertag, slot: -1, picture: "" }
        ]
      };
    case "hotkey":
      return { ...state, hotkeys: state.hotkeys.map((h) => (h.id === message.id ? { ...h, enabled: message.enabled } : h)) };
    case "rebind":
      return { ...state, rebind: { ...state.rebind, capturing: message.output } };
    case "rebindCancel":
      return { ...state, rebind: { ...state.rebind, capturing: "" } };
    case "action":
      if (message.id === "timeScalar" && typeof message.value === "number") {
        return { ...state, timeScalar: message.value };
      }
      return state;
    default:
      return state;
  }
}

/** A few entries for the Acknowledgements sub-blade on the dev server, which
    has no collected acknowledgements.json. */
export const mockAcknowledgements = [
  {
    name: "Xenia",
    license: "BSD-3-Clause",
    url: "https://github.com/xenia-canary/xenia-canary",
    note: "macXenos is a macOS port of Xenia Canary.",
    text: "Copyright (c) 2015, Ben Vanik.\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:\n\n* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer."
  },
  {
    name: "SDL2",
    license: "Zlib",
    url: "https://www.libsdl.org",
    note: "",
    text: "Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>\n\nThis software is provided 'as-is', without any express or implied warranty."
  },
  {
    name: "Sparkle",
    license: "MIT",
    url: "https://sparkle-project.org",
    note: "",
    text: "Copyright (c) 2006-2013 Andy Matuschak.\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software."
  }
];
