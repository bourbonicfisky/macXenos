import { send, useNative, type RebindOutput } from "../native";
import { t } from "../i18n";

/**
 * Button mapping on a drawing of the controller: each button that can be
 * remapped sits where it is on the pad, drawn as the pad draws it (the face
 * buttons in their colours, the bumpers and Back / Start as chips). Choosing
 * one listens for the button to use instead; choosing it again stops.
 * Beneath, what each is mapped to now.
 *
 * The drawing is our own outline of a 360-style pad, not Microsoft's art.
 */

type Spot = { x: number; y: number; text: string; className: string };

// Where each output sits on the 400 x 260 drawing.
const SPOTS: Record<string, Spot> = {
  leftshoulder: { x: 112, y: 40, text: "LB", className: "pad-chip" },
  rightshoulder: { x: 288, y: 40, text: "RB", className: "pad-chip" },
  back: { x: 166, y: 106, text: "Back", className: "pad-chip pad-spot--small" },
  start: { x: 234, y: 106, text: "Start", className: "pad-chip pad-spot--small" },
  y: { x: 304, y: 70, text: "Y", className: "pad-key pad-key--y" },
  x: { x: 276, y: 98, text: "X", className: "pad-key pad-key--x" },
  b: { x: 332, y: 98, text: "B", className: "pad-key pad-key--b" },
  a: { x: 304, y: 126, text: "A", className: "pad-key pad-key--a" }
};

/** "b3" -> "button 3", "a2+" -> "axis 2", as the host names its inputs. */
export function bindingText(binding: string): string {
  if (!binding) return t("controller.map.default");
  const button = /^b(\d+)$/.exec(binding);
  if (button) return t("controller.map.button", { n: Number(button[1]) });
  const axis = /^[+-]?a(\d+)[+-~]?$/.exec(binding);
  if (axis) return t("controller.map.axis", { n: Number(axis[1]) });
  return binding;
}

function PadDrawing() {
  return (
    <svg className="pad-drawing absolute inset-0 h-full w-full" viewBox="0 0 400 260" aria-hidden="true">
      <defs>
        <linearGradient id="pad-body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f4f4f6" />
          <stop offset="0.55" stopColor="#d9dade" />
          <stop offset="1" stopColor="#b7b9bf" />
        </linearGradient>
        <radialGradient id="pad-well" cx="0.5" cy="0.4" r="0.6">
          <stop offset="0" stopColor="#5b5e66" />
          <stop offset="1" stopColor="#2b2d33" />
        </radialGradient>
      </defs>
      {/* Bumpers, behind the body. */}
      <path d="M66 70 C88 40 132 30 168 40 L166 54 C132 46 98 54 80 78 Z" fill="#c9cad0" stroke="#8d9097" strokeWidth="1.5" />
      <path d="M334 70 C312 40 268 30 232 40 L234 54 C268 46 302 54 320 78 Z" fill="#c9cad0" stroke="#8d9097" strokeWidth="1.5" />
      {/* The body, with its two grips. */}
      <path
        d="M62 72 C92 46 150 44 200 54 C250 44 308 46 338 72 C374 106 394 188 380 228 C370 254 336 258 316 234 C296 210 280 188 256 182 L144 182 C120 188 104 210 84 234 C64 258 30 254 20 228 C6 188 26 106 62 72 Z"
        fill="url(#pad-body)"
        stroke="#8d9097"
        strokeWidth="2"
      />
      {/* Sticks and d-pad: not remappable here, drawn for the shape. */}
      <circle cx="120" cy="100" r="25" fill="url(#pad-well)" />
      <circle cx="120" cy="100" r="15" fill="#4a4d55" stroke="#2a2c31" strokeWidth="2" />
      <circle cx="248" cy="150" r="22" fill="url(#pad-well)" />
      <circle cx="248" cy="150" r="13" fill="#4a4d55" stroke="#2a2c31" strokeWidth="2" />
      <path
        d="M146 142 h18 v-14 h14 v14 h18 v14 h-18 v14 h-14 v-14 h-18 z"
        transform="translate(-10 0)"
        fill="#55585f"
        stroke="#2a2c31"
        strokeWidth="1.5"
      />
      {/* The Guide button, with its ring of light. */}
      <circle cx="200" cy="80" r="15" fill="#d0d1d6" stroke="#8d9097" strokeWidth="1.5" />
      <circle cx="200" cy="80" r="10" fill="#3f8f2c" />
      <circle cx="200" cy="80" r="6" fill="#b9ec85" opacity="0.65" />
    </svg>
  );
}

function Spot({ output, capturing }: { output: RebindOutput; capturing: boolean }) {
  const spot = SPOTS[output.output];
  if (!spot) return null;
  return (
    <button
      type="button"
      className={`pad-spot ${spot.className} absolute`}
      style={{ left: `${(spot.x / 400) * 100}%`, top: `${(spot.y / 260) * 100}%` }}
      data-capturing={capturing || undefined}
      aria-label={t(capturing ? "controller.map.listening" : "controller.map.choose", { button: output.label })}
      aria-pressed={capturing}
      onClick={() => send(capturing ? { op: "rebindCancel" } : { op: "rebind", output: output.output })}
    >
      {spot.text}
    </button>
  );
}

export function ControllerMap() {
  const { rebind } = useNative();
  if (rebind.slot < 0) {
    return <p className="pad-status m-0">{t("controller.map.none")}</p>;
  }
  const capturing = rebind.outputs.find((output) => output.output === rebind.capturing);
  return (
    <>
      <p className="pad-status m-0">{t("controller.map.who", { player: rebind.slot + 1, name: rebind.name })}</p>
      <div className="pad-map relative w-full" style={{ aspectRatio: "400 / 260" }}>
        <PadDrawing />
        {rebind.outputs.map((output) => (
          <Spot key={output.output} output={output} capturing={output.output === rebind.capturing} />
        ))}
      </div>
      <p className={`m-0 text-[0.85em] ${capturing ? "pad-capture" : "pad-status"}`}>
        {capturing ? t("controller.map.press", { button: capturing.label }) : t("controller.map.hint")}
      </p>
      <dl className="pad-bindings m-0 grid grid-cols-[auto_minmax(0,1fr)] gap-x-[0.8em] gap-y-[0.15em] text-[0.8em]">
        {rebind.outputs.map((output) => (
          <div key={output.output} className="contents">
            <dt>{output.label}</dt>
            <dd className="pad-status m-0">{bindingText(output.binding)}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}
