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
import { t } from "../i18n";

const PLAYERS = ["Player 1", "Player 2", "Player 3", "Player 4"];

const KEYBOARD_MODES: Option[] = [
  { value: 0, label: "Off", short: "Off" },
  { value: 1, label: "Plays as a controller", short: "Controller" },
  { value: 2, label: "Passthrough", short: "Passthrough" }
];

const KEYBOARD_PLAYER: Option[] = PLAYERS.map((label, value) => ({ value, label, short: label }));

const DASHBOARD_KEYS: Option[] = [
  { value: "F6", label: "F6", short: "F6", recommended: true },
  { value: "F7", label: "F7", short: "F7" },
  { value: "F8", label: "F8", short: "F8" },
  { value: "F10", label: "F10", short: "F10" },
  { value: "none", label: "None", short: "None" }
];

const DASHBOARD_BUTTONS: Option[] = [
  { value: "guide", label: "Tap Guide", short: "Guide", recommended: true },
  { value: "back_start", label: "Back + Start", short: "Back + Start" },
  { value: "none", label: "None", short: "None" }
];

const INPUT_SYSTEMS: Option[] = [
  { value: "any", label: "Automatic", short: "Automatic" },
  { value: "sdl", label: "Game controllers (SDL)", short: "SDL" },
  { value: "keyboard", label: "Keyboard only", short: "Keyboard" },
  { value: "nop", label: "None", short: "None" }
];

const percent = (value: number) => `${Math.round(value * 100)}%`;

/** Each pad that is plugged in, and which player it plays as. */
function Controllers() {
  const { controllers } = useNative();
  const present = controllers.filter((c) => c.present);
  if (present.length === 0) {
    return <p className="pad-status m-0">No controller is connected. Plug one in or pair it over Bluetooth.</p>;
  }
  return (
    <>
      {present.map((controller) => (
        <div key={controller.slot} className="grid gap-[0.2em]">
          <p className="pref-group m-0">
            {controller.name || "Controller"}{" "}
            <span className="pad-status font-normal">({controller.kind})</span>
          </p>
          <div
            role="radiogroup"
            aria-label={`${controller.name} plays as`}
            className="flex flex-wrap gap-[0.5em]"
          >
            {PLAYERS.map((label, slot) => (
              <button
                key={slot}
                type="button"
                role="radio"
                aria-checked={controller.slot === slot}
                className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
                onClick={() => {
                  if (controller.slot === slot) return;
                  send({ op: "assign", from: controller.slot, to: slot });
                  announceSetting(controller.name || "Controller", label);
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
  "Left Shoulder": { text: "LB", className: "pad-chip", name: "Left bumper" },
  "Right Shoulder": { text: "RB", className: "pad-chip", name: "Right bumper" },
  Start: { text: "Start", className: "pad-chip", name: "Start" },
  Back: { text: "Back", className: "pad-chip", name: "Back" },
  "D-PAD Up": { text: "↑", className: "pad-chip", name: "D-pad up" },
  "D-PAD Down": { text: "↓", className: "pad-chip", name: "D-pad down" },
  "D-PAD Left": { text: "←", className: "pad-chip", name: "D-pad left" },
  "D-PAD Right": { text: "→", className: "pad-chip", name: "D-pad right" }
};

// The app's own wording, in plain words.
const ACTIONS: Record<string, string> = {
  "Toggle Fullscreen": "Full screen on or off",
  "Clear GPU Cache": "Clear the GPU caches",
  "Toggle Controller Vibration": "Vibration on or off",
  "Half CPU Scalar": "Half game speed",
  "Double CPU Scalar": "Double game speed",
  "Reset CPU Scalar": "Normal game speed",
  "Toggle Readback Resolve": "Cycle readback resolve",
  "Toggle Clear Memory Page State": "Clear memory page state on or off",
  "Run Selected Title": "Play the selected game",
  "Title Selection +1": "Next game",
  "Title Selection -1": "Previous game"
};

function parseHotkey(label: string) {
  const [combo, ...rest] = label.split(" = ");
  let action = rest.join(" = ").trim();
  const vulkanOnly = /\(Vulkan only\)/.test(action);
  action = action.replace(/\s*\(Vulkan only\)/, "").trim();
  if (/loglevel/i.test(action)) action = "Logging on or off";
  const buttons = combo
    .split(" + ")
    .map((b) => b.trim())
    // Guide is held first, so it leads.
    .sort((a, b) => (a === "Guide" ? -1 : b === "Guide" ? 1 : 0));
  return { buttons, action: ACTIONS[action] ?? action, vulkanOnly };
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
        <p className="menu-note m-0 text-[0.78em]">
          Turn hotkeys on above to use these during a game.
        </p>
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
              {hotkey.vulkanOnly ? <span className="pref-tag">Vulkan only</span> : null}
            </span>
            <button
              type="button"
              role="switch"
              aria-checked={hotkey.enabled}
              aria-label={hotkey.action}
              className="pref-switch shrink-0 cursor-pointer"
              onClick={() => {
                send({ op: "hotkey", id: hotkey.id, enabled: !hotkey.enabled });
                announceSetting(hotkey.action, hotkey.enabled ? "Off" : "On");
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
  return <>{count === 0 ? "None" : count === 1 ? "1 connected" : `${count} connected`}</>;
}

export function controller(_state: NativeState): BladeDef {
  return {
    id: "controller",
    label: "controller",
    title: "Controller",
    tone: "orange",
    menu: [
      {
        id: "controllers",
        label: "Controllers",
        glyph: "pad",
        value: <ConnectedValue />,
        heading: "Connected controllers",
        detail: (
          <>
            <p>
              Choose which player each controller plays as. Games always see an
              Xbox 360 controller, whatever you hold.
            </p>
            <Controllers />
          </>
        )
      },
      {
        id: "remap",
        label: "Button mapping",
        glyph: "keys",
        heading: "Button mapping",
        detail: (
          <>
            <Remap />
          </>
        )
      },
      {
        id: "vibration",
        label: "Vibration",
        glyph: "bolt",
        value: <Value name="vibration" />,
        heading: "Vibration",
        detail: (
          <>
            <p>Rumble when the game asks for it.</p>
            <Toggle name="vibration" title="Vibration" />
          </>
        )
      },
      {
        id: "deadzones",
        label: "Stick dead zones",
        glyph: "target",
        heading: "Stick dead zones",
        detail: (
          <>
            <p>
              How far a stick moves before the game sees it. Raise one if a
              worn stick drifts on its own.
            </p>
            <p className="pref-group">Left stick</p>
            <Slider
              name="left_stick_deadzone_percentage"
              title="Left stick dead zone"
              min={0}
              max={0.5}
              step={0.01}
              format={percent}
            />
            <p className="pref-group">Right stick</p>
            <Slider
              name="right_stick_deadzone_percentage"
              title="Right stick dead zone"
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
        label: "Guide button",
        glyph: "play",
        value: <Value name="guide_button" />,
        heading: "Guide button",
        detail: (
          <>
            <p>
              Send the Guide button to the game, as the console did. Turn it off
              if you only use Guide for the hotkeys below.
            </p>
            <Toggle name="guide_button" title="Guide button" />
          </>
        )
      },
      {
        id: "hotkeys",
        label: "Controller hotkeys",
        glyph: "star",
        value: <Value name="controller_hotkeys" />,
        heading: "Controller hotkeys",
        detail: (
          <>
            <p>Hold Guide and press a button. Turn off any you press by accident.</p>
            <p className="pref-group">During a game</p>
            <Toggle name="controller_hotkeys" title="Controller hotkeys during a game" />
            <NextLaunchNote names={["controller_hotkeys"]} />
            <HotkeyChart inGame />
            <p className="pref-group">On the home screen</p>
            <HotkeyChart inGame={false} />
          </>
        )
      },
      {
        id: "dashboard-shortcut",
        label: "Dashboard shortcut",
        glyph: "keys",
        value: <Value name="preferences_button" options={DASHBOARD_BUTTONS} />,
        heading: "Dashboard shortcut",
        detail: (
          <>
            <p>
              How to open and close this dashboard, in a game or out of one.
              The game pauses while it is open, as under the 360&rsquo;s guide.
            </p>
            <p className="pref-group">On a controller</p>
            <Options name="preferences_button" title="Dashboard button" options={DASHBOARD_BUTTONS} />
            <Aside>
              A tap of Guide on its own opens and hides the dashboard; Guide
              held with another button is a hotkey instead. While this is
              chosen, macOS&rsquo;s own Games overlay stays off that button.
            </Aside>
            <p className="pref-group">On the keyboard</p>
            <Options name="preferences_key" title="Dashboard key" options={DASHBOARD_KEYS} />
            <Aside>⌘, always opens it too.</Aside>
          </>
        )
      },
      {
        id: "keyboard",
        label: "Keyboard",
        glyph: "keys",
        value: <Value name="keyboard_mode" options={KEYBOARD_MODES} />,
        heading: "Playing on the keyboard",
        detail: (
          <>
            <p>
              The keyboard can play as a controller. Passthrough sends keys to
              games that read a keyboard, and needs a controller connected too.
            </p>
            <Options name="keyboard_mode" title="Keyboard" options={KEYBOARD_MODES} />
            <p className="pref-group">The keyboard plays as</p>
            <Options name="keyboard_user_index" title="Keyboard player" options={KEYBOARD_PLAYER} />
            <NextLaunchNote names={["keyboard_mode", "keyboard_user_index"]} />
          </>
        )
      },
      {
        id: "input-system",
        label: "Input system",
        glyph: "plug",
        value: <Value name="hid" options={INPUT_SYSTEMS} />,
        heading: "Input system",
        detail: (
          <>
            <p>
              Automatic reads controllers and the keyboard together. Pick one
              only to rule the other out while troubleshooting.
            </p>
            <Options name="hid" title="Input system" options={INPUT_SYSTEMS} />
            <Aside>Takes effect the next time macXenos starts.</Aside>
          </>
        )
      }
    ]
  };
}
