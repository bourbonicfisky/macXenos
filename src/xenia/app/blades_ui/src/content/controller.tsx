import type { BladeDef } from "../ui/blade";
import { send, useNative, type NativeState } from "../native";
import {
  NextLaunchNote,
  Options,
  Slider,
  Toggle,
  Value,
  announceSetting,
  type Option
} from "../ui/settings-controls";
import { Aside, PaneAction } from "./parts";
import { rendererShown } from "./visibility";
import { ControllerMap } from "../ui/controller-map";
import { t, type MessageKey } from "../i18n";

const players = () => [1, 2, 3, 4].map((player) => t("controller.player", { player }));

const KEYBOARD_MODES: Option[] = [
  { value: 0, get label() { return t("graphics.qos.off"); }, get short() { return t("graphics.qos.off"); } },
  { value: 1, get label() { return t("controller.playsAsAController"); }, get short() { return t("controller.title"); } },
  { value: 2, get label() { return t("controller.passthrough"); }, get short() { return t("controller.passthrough"); } }
];

const keyboardPlayers = (): Option[] => players().map((label, value) => ({ value, label, short: label }));

const DASHBOARD_KEYS: Option[] = [
  { value: "F6", label: "F6", short: "F6", recommended: true },
  { value: "F7", label: "F7", short: "F7" },
  { value: "F8", label: "F8", short: "F8" },
  { value: "F10", label: "F10", short: "F10" },
  { value: "none", get label() { return t("home.gameFolders.none"); }, get short() { return t("home.gameFolders.none"); } }
];

const DASHBOARD_BUTTONS: Option[] = [
  { value: "guide", get label() { return t("controller.tapGuide"); }, short: "Guide", recommended: true },
  { value: "back_start", label: "Back + Start", short: "Back + Start" },
  { value: "none", get label() { return t("home.gameFolders.none"); }, get short() { return t("home.gameFolders.none"); } }
];

const INPUT_SYSTEMS: Option[] = [
  { value: "any", get label() { return t("controller.automatic"); }, get short() { return t("controller.automatic"); } },
  { value: "sdl", get label() { return t("controller.gameControllersSDL"); }, short: "SDL" },
  { value: "keyboard", get label() { return t("controller.keyboardOnly"); }, get short() { return t("controller.keyboard"); } },
  { value: "nop", get label() { return t("home.gameFolders.none"); }, get short() { return t("home.gameFolders.none"); } }
];

const percent = (value: number) => `${Math.round(value * 100)}%`;

/** Each pad that is plugged in, and which player it plays as. */
function Controllers() {
  const { controllers } = useNative();
  const present = controllers.filter((c) => c.present);
  if (present.length === 0) {
    return <p className="pad-status m-0">{t("controller.noneConnected")}</p>;
  }
  return (
    <>
      {present.map((controller) => (
        <div key={controller.slot} className="grid gap-[0.2em]">
          <p className="pref-group m-0">
            {controller.name || t("controller.title")}{" "}
            <span className="pad-status font-normal">({controller.kind})</span>
          </p>
          <div
            role="radiogroup"
            aria-label={t("controller.playsAs", { name: controller.name })}
            className="flex flex-wrap gap-[0.5em]"
          >
            {players().map((label, slot) => (
              <button
                key={slot}
                type="button"
                role="radio"
                aria-checked={controller.slot === slot}
                className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
                onClick={() => {
                  if (controller.slot === slot) return;
                  send({ op: "assign", from: controller.slot, to: slot });
                  announceSetting(controller.name || t("controller.title"), label);
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </>
  );
}

/** Button remapping for the first connected pad, on a drawing of it. */
function Remap() {
  return (
    <>
      <ControllerMap />
      <PaneAction
        onSelect={() => {
          send({ op: "rebindReset" });
          announceSetting(t("controller.map.title"), t("controller.map.default"));
        }}
      >
        {t("controller.map.reset")}
      </PaneAction>
    </>
  );
}

/* The controller hotkeys, as a chart: the buttons to press, drawn as the
   pad's own, what they do, and Off / On. */

const BUTTONS: Record<string, { text: string; className: string; name: string }> = {
  Guide: { text: "", className: "pad-chip pad-chip--guide", name: "Guide" },
  A: { text: "A", className: "pad-key pad-key--a", name: "A" },
  B: { text: "B", className: "pad-key pad-key--b", name: "B" },
  X: { text: "X", className: "pad-key pad-key--x", name: "X" },
  Y: { text: "Y", className: "pad-key pad-key--y", name: "Y" },
  "Left Shoulder": { text: "LB", className: "pad-chip", get name() { return t("controller.leftBumper"); } },
  "Right Shoulder": { text: "RB", className: "pad-chip", get name() { return t("controller.rightBumper"); } },
  Start: { text: "Start", className: "pad-chip", name: "Start" },
  Back: { text: "Back", className: "pad-chip", name: "Back" },
  "D-PAD Up": { text: "↑", className: "pad-chip", get name() { return t("controller.dpadUp"); } },
  "D-PAD Down": { text: "↓", className: "pad-chip", get name() { return t("controller.dpadDown"); } },
  "D-PAD Left": { text: "←", className: "pad-chip", get name() { return t("controller.dpadLeft"); } },
  "D-PAD Right": { text: "→", className: "pad-chip", get name() { return t("controller.dpadRight"); } }
};

// The app's own wording, in plain words.
const ACTIONS: Record<string, MessageKey> = {
  "Toggle Fullscreen": "controller.action.fullscreen",
  "Clear GPU Cache": "controller.action.clearCache",
  "Toggle Controller Vibration": "controller.action.vibration",
  "Half CPU Scalar": "controller.action.halfSpeed",
  "Double CPU Scalar": "controller.action.doubleSpeed",
  "Reset CPU Scalar": "controller.action.normalSpeed",
  "Toggle Readback Resolve": "controller.action.readback",
  "Toggle Clear Memory Page State": "controller.action.memoryPages",
  "Run Selected Title": "controller.action.play",
  "Title Selection +1": "controller.action.next",
  "Title Selection -1": "controller.action.previous"
};

function parseHotkey(label: string) {
  const [combo, ...rest] = label.split(" = ");
  let action = rest.join(" = ").trim();
  const vulkanOnly = /\(Vulkan only\)/.test(action);
  action = action.replace(/\s*\(Vulkan only\)/, "").trim();
  if (/loglevel/i.test(action)) action = t("controller.action.logging");
  const buttons = combo
    .split(" + ")
    .map((b) => b.trim())
    // Guide is held first, so it leads.
    .sort((a, b) => (a === "Guide" ? -1 : b === "Guide" ? 1 : 0));
  return { buttons, action: ACTIONS[action] ? t(ACTIONS[action]) : action, vulkanOnly };
}

function HotkeyChart({ inGame }: { inGame: boolean }) {
  const state = useNative();
  const listening = state.cvars.controller_hotkeys?.value === true;
  const hotkeys = state.hotkeys
    .filter((h) => h.inGame === inGame)
    .map((h) => ({ ...h, ...parseHotkey(h.label) }))
    .filter((h) => !h.vulkanOnly || rendererShown(state, "vulkan"));
  return (
    <div className="grid">
      {inGame && !listening ? (
        <p className="menu-note m-0 text-[0.78em]">{t("controller.hotkeys.disabled")}</p>
      ) : null}
      {hotkeys.map((hotkey) => {
        const off = !hotkey.enabled || (inGame && !listening);
        return (
          <div
            key={hotkey.id}
            className="hotkey-row flex items-center gap-[0.7em]"
            data-off={off || undefined}
          >
            <span className="hotkey-combo inline-flex shrink-0 items-center gap-[0.25em]">
              {hotkey.buttons.map((button, i) => {
                const shape = BUTTONS[button] ?? { text: button, className: "pad-chip", name: button };
                return (
                  <span key={button} className="inline-flex items-center gap-[0.25em]">
                    {i > 0 ? <span aria-hidden="true">+</span> : null}
                    <span className={shape.className} title={shape.name} aria-label={shape.name}>
                      {shape.text}
                    </span>
                  </span>
                );
              })}
            </span>
            <span className="hotkey-action min-w-0 flex-1">
              {hotkey.action}
              {hotkey.vulkanOnly ? <span className="pref-tag">{t("controller.vulkanOnly")}</span> : null}
            </span>
            <button
              type="button"
              role="switch"
              aria-checked={hotkey.enabled}
              aria-label={hotkey.action}
              className="pref-switch shrink-0 cursor-pointer"
              onClick={() => {
                send({ op: "hotkey", id: hotkey.id, enabled: !hotkey.enabled });
                announceSetting(hotkey.action, hotkey.enabled ? t("common.off") : t("common.on"));
              }}
            >
              <span className="pref-switch-knob" aria-hidden="true" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

function ConnectedValue() {
  const { controllers } = useNative();
  const count = controllers.filter((c) => c.present).length;
  return <>{count === 0 ? t("home.gameFolders.none") : t("controller.connected", { count })}</>;
}

export function controller(_state: NativeState): BladeDef {
  return {
    id: "controller",
    label: t("controller.tab"),
    title: t("controller.title"),
    tone: "orange",
    menu: [
      {
        id: "controllers",
        label: t("controller.controllers"),
        glyph: "pad",
        value: <ConnectedValue />,
        heading: t("controller.connectedControllers"),
        detail: (
          <>
            <p>{t("controller.connected.detail")}</p>
            <Controllers />
          </>
        )
      },
      {
        id: "remap",
        label: t("controller.map.title"),
        glyph: "keys",
        heading: t("controller.map.title"),
        detail: (
          <>
            <Remap />
          </>
        )
      },
      {
        id: "vibration",
        label: t("controller.vibration"),
        glyph: "bolt",
        value: <Value name="vibration" />,
        heading: t("controller.vibration"),
        detail: (
          <>
            <p>{t("controller.vibration.detail")}</p>
            <Toggle name="vibration" title={t("controller.vibration")} />
          </>
        )
      },
      {
        id: "deadzones",
        label: t("controller.stickDeadZones"),
        glyph: "target",
        heading: t("controller.stickDeadZones"),
        detail: (
          <>
            <p>{t("controller.deadZones.detail")}</p>
            <p className="pref-group">{t("controller.leftStick")}</p>
            <Slider
              name="left_stick_deadzone_percentage"
              title={t("controller.leftStickDeadZone")}
              min={0}
              max={0.5}
              step={0.01}
              format={percent}
            />
            <p className="pref-group">{t("controller.rightStick")}</p>
            <Slider
              name="right_stick_deadzone_percentage"
              title={t("controller.rightStickDeadZone")}
              min={0}
              max={0.5}
              step={0.01}
              format={percent}
            />
          </>
        )
      },
      {
        id: "guide",
        label: t("controller.guideButton"),
        glyph: "play",
        value: <Value name="guide_button" />,
        heading: t("controller.guideButton"),
        detail: (
          <>
            <p>{t("controller.guide.detail")}</p>
            <Toggle name="guide_button" title={t("controller.guideButton")} />
          </>
        )
      },
      {
        id: "hotkeys",
        label: t("controller.hotkeys"),
        glyph: "star",
        value: <Value name="controller_hotkeys" />,
        heading: t("controller.hotkeys"),
        detail: (
          <>
            <p>{t("controller.hotkeys.detail")}</p>
            <p className="pref-group">{t("controller.duringAGame")}</p>
            <Toggle name="controller_hotkeys" title={t("controller.hotkeys.inGame")} />
            <NextLaunchNote names={["controller_hotkeys"]} />
            <HotkeyChart inGame />
            <p className="pref-group">{t("controller.onTheHomeScreen")}</p>
            <HotkeyChart inGame={false} />
          </>
        )
      },
      {
        id: "dashboard-shortcut",
        label: t("controller.dashboardShortcut"),
        glyph: "keys",
        value: <Value name="preferences_button" options={DASHBOARD_BUTTONS} />,
        heading: t("controller.dashboardShortcut"),
        detail: (
          <>
            <p>{t("controller.shortcut.detail")}</p>
            <p className="pref-group">{t("controller.onAController")}</p>
            <Options name="preferences_button" title={t("controller.dashboardButton")} options={DASHBOARD_BUTTONS} />
            <Aside>{t("controller.shortcut.guideNote")}</Aside>
            <p className="pref-group">{t("controller.onTheKeyboard")}</p>
            <Options name="preferences_key" title={t("controller.dashboardKey")} options={DASHBOARD_KEYS} />
            <Aside>{t("controller.shortcut.keyboardNote")}</Aside>
          </>
        )
      },
      {
        id: "keyboard",
        label: t("controller.keyboard"),
        glyph: "keys",
        value: <Value name="keyboard_mode" options={KEYBOARD_MODES} />,
        heading: t("controller.playingOnTheKeyboard"),
        detail: (
          <>
            <p>{t("controller.keyboard.detail")}</p>
            <Options name="keyboard_mode" title={t("controller.keyboard")} options={KEYBOARD_MODES} />
            <p className="pref-group">{t("controller.keyboard.player")}</p>
            <Options name="keyboard_user_index" title={t("controller.keyboardPlayer")} options={keyboardPlayers()} />
            <NextLaunchNote names={["keyboard_mode", "keyboard_user_index"]} />
          </>
        )
      },
      {
        id: "input-system",
        label: t("controller.inputSystem"),
        glyph: "plug",
        value: <Value name="hid" options={INPUT_SYSTEMS} />,
        heading: t("controller.inputSystem"),
        detail: (
          <>
            <p>{t("controller.input.detail")}</p>
            <Options name="hid" title={t("controller.inputSystem")} options={INPUT_SYSTEMS} />
            <Aside>{t("controller.input.nextLaunch")}</Aside>
          </>
        )
      }
    ]
  };
}
