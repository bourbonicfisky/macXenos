import { send, useNative, type RebindOutput } from "../native";
import { t } from "../i18n";

/**
 * Button mapping on a drawing of the controller: each button that can be
 * remapped sits where it is on the pad, drawn as the pad draws it (the face
 * buttons in their colours, the bumpers and Back / Start as chips). Choosing
 * one listens for the button to use instead; choosing it again stops.
 * Beneath, what each is mapped to now.
 *
 * The drawing is our own of a 360-style pad (no logo), not Microsoft's art.
 */

type Spot = { x: number; y: number; text: string; className: string };

// The drawing's frame (in the coordinates its shapes are drawn in), and
// where each remappable output sits on it.
const VIEW = { x: 110, y: 52, w: 530, h: 358 };
const SPOTS: Record<string, Spot> = {
  leftshoulder: { x: 226, y: 126, text: "LB", className: "pad-chip" },
  rightshoulder: { x: 524, y: 126, text: "RB", className: "pad-chip" },
  back: { x: 316, y: 222, text: "◀", className: "pad-chip pad-spot--arrow" },
  start: { x: 434, y: 222, text: "▶", className: "pad-chip pad-spot--arrow" },
  y: { x: 528, y: 184, text: "Y", className: "pad-key pad-key--y" },
  x: { x: 486, y: 220, text: "X", className: "pad-key pad-key--x" },
  b: { x: 566, y: 218, text: "B", className: "pad-key pad-key--b" },
  a: { x: 520, y: 254, text: "A", className: "pad-key pad-key--a" },
  lefttrigger: { x: 240, y: 92, text: "LT", className: "pad-chip pad-spot--small" },
  righttrigger: { x: 510, y: 92, text: "RT", className: "pad-chip pad-spot--small" },
  guide: { x: 375, y: 218, text: "", className: "pad-chip pad-chip--guide pad-spot--guide" },
  leftstick: { x: 214, y: 236, text: "LS", className: "pad-chip pad-spot--small" },
  rightstick: { x: 442, y: 300, text: "RS", className: "pad-chip pad-spot--small" },
  dpup: { x: 300, y: 278, text: "▲", className: "pad-chip pad-spot--arrow" },
  dpdown: { x: 300, y: 318, text: "▼", className: "pad-chip pad-spot--arrow" },
  dpleft: { x: 279, y: 298, text: "◀", className: "pad-chip pad-spot--arrow" },
  dpright: { x: 321, y: 298, text: "▶", className: "pad-chip pad-spot--arrow" }
};

/** "b3" -> "button 3", "a2+" -> "axis 2", as the host names its inputs. */
export function bindingText(binding: string): string {
  if (!binding) return t("controller.map.default");
  const button = /^b(\d+)$/.exec(binding);
  if (button) return t("controller.map.button", { n: Number(button[1]) });
  const axis = /^[+-]?a(\d+)[+-~]?$/.exec(binding);
  if (axis) return t("controller.map.axis", { n: Number(axis[1]) });
  const hat = /^h\d+\.(\d+)$/.exec(binding);
  if (hat) {
    const directions: Record<string, string> = { "1": "↑", "2": "→", "4": "↓", "8": "←" };
    return t("controller.map.hat", { direction: directions[hat[1]] ?? hat[1] });
  }
  return binding;
}

// The shell from the front and a little above: shoulders rising to the
// bumpers, the raised middle where the cable plugs in, grips swinging down
// round a flat middle.
const BODY =
  "M162 152 C186 118 236 106 280 116 C300 121 308 132 328 132 L422 132 " +
  "C442 132 450 121 470 116 C514 106 564 118 588 152 " +
  "C606 182 616 240 622 300 C628 352 624 390 600 396 C568 404 538 380 518 354 " +
  "C498 330 482 324 452 324 L298 324 C268 324 252 330 232 354 " +
  "C212 380 182 404 150 396 C126 390 122 352 128 300 C134 240 144 182 162 152 Z";
// The bumpers: the band along each shoulder.
const LEFT_BUMPER = "M166 150 C190 120 236 108 280 117 C292 120 300 126 306 132 C288 128 270 124 252 126 C214 128 186 140 168 158 Z";
const RIGHT_BUMPER = "M584 150 C560 120 514 108 470 117 C458 120 450 126 444 132 C462 128 480 124 498 126 C536 128 564 140 582 158 Z";

function Well({ x, y, r }: { x: number; y: number; r: number }) {
  return <circle cx={x} cy={y} r={r} fill="url(#pad-raised)" stroke="#c9cbd0" strokeWidth="1" />;
}

function Stick({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <Well x={x} y={y} r={46} />
      <circle cx={x} cy={y} r={31} fill="url(#pad-cap)" stroke="#1d1f23" strokeWidth="2" />
      <circle cx={x - 2} cy={y - 3} r={21} fill="none" stroke="#5d6067" strokeWidth="1.5" />
    </g>
  );
}

function PadDrawing() {
  return (
    <svg
      className="pad-drawing absolute inset-0 h-full w-full"
      viewBox={`${VIEW.x} ${VIEW.y} ${VIEW.w} ${VIEW.h}`}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="pad-body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fbfbfc" />
          <stop offset="0.55" stopColor="#ebecef" />
          <stop offset="1" stopColor="#cdcfd4" />
        </linearGradient>
        <radialGradient id="pad-raised" cx="0.45" cy="0.4" r="0.6">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.85" stopColor="#eceef1" />
          <stop offset="1" stopColor="#d6d8dd" />
        </radialGradient>
        <radialGradient id="pad-cap" cx="0.45" cy="0.35" r="0.65">
          <stop offset="0" stopColor="#62656c" />
          <stop offset="1" stopColor="#2a2c31" />
        </radialGradient>
        <radialGradient id="pad-guide" cx="0.42" cy="0.38" r="0.65">
          <stop offset="0" stopColor="#f6f7f8" />
          <stop offset="1" stopColor="#a8abb2" />
        </radialGradient>
      </defs>
      {/* Triggers, standing up behind the bumpers. */}
      <path d="M222 118 L228 84 C230 78 248 78 252 84 L258 114 Z" fill="#d3d5da" stroke="#9a9da4" strokeWidth="1.5" />
      <path d="M528 118 L522 84 C520 78 502 78 498 84 L492 114 Z" fill="#d3d5da" stroke="#9a9da4" strokeWidth="1.5" />
      {/* The raised middle with the plug. */}
      <path d="M314 134 L322 124 L428 124 L436 134 Z" fill="#dfe0e4" stroke="#9a9da4" strokeWidth="1.2" />
      <path d={BODY} fill="url(#pad-body)" stroke="#7d8087" strokeWidth="2.5" />
      <path d={LEFT_BUMPER} fill="#e3e4e8" stroke="#9a9da4" strokeWidth="1.2" />
      <path d={RIGHT_BUMPER} fill="#e3e4e8" stroke="#9a9da4" strokeWidth="1.2" />
      {/* Left stick high; d-pad and right stick low; face buttons high. */}
      <Stick x={214} y={236} />
      <Well x={300} y={298} r={50} />
      <circle cx={300} cy={298} r={36} fill="url(#pad-cap)" stroke="#1d1f23" strokeWidth="2" />
      <path
        d="M292 268 h16 v22 h22 v16 h-22 v22 h-16 v-22 h-22 v-16 h22 z"
        fill="#45484f"
        stroke="#6f727a"
        strokeWidth="1.2"
      />
      <Stick x={442} y={300} />
      {/* The Guide button in its ring of light, Back and Start beside it. */}
      <circle cx="375" cy="218" r="30" fill="#d4d6da" stroke="#9a9da4" strokeWidth="1.2" />
      <circle cx="375" cy="218" r="25" fill="none" stroke="#7be04a" strokeWidth="4" opacity="0.9" />
      <circle cx="375" cy="218" r="25" fill="none" stroke="#d4ffb0" strokeWidth="1.2" opacity="0.8" />
      <circle cx="375" cy="218" r="20" fill="url(#pad-guide)" />
      <circle cx="316" cy="222" r="9" fill="#d9dbdf" stroke="#9a9da4" strokeWidth="1.2" />
      <circle cx="434" cy="222" r="9" fill="#d9dbdf" stroke="#9a9da4" strokeWidth="1.2" />
      <text x="316" y="204" textAnchor="middle" className="pad-drawing-label">
        BACK
      </text>
      <text x="434" y="204" textAnchor="middle" className="pad-drawing-label">
        START
      </text>
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
      style={{ left: `${((spot.x - VIEW.x) / VIEW.w) * 100}%`, top: `${((spot.y - VIEW.y) / VIEW.h) * 100}%` }}
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
      <div className="pad-map relative w-full" style={{ aspectRatio: `${VIEW.w} / ${VIEW.h}` }}>
        <PadDrawing />
        {rebind.outputs.map((output) => (
          <Spot key={output.output} output={output} capturing={output.output === rebind.capturing} />
        ))}
      </div>
      <p className={`m-0 text-[0.85em] ${capturing ? "pad-capture" : "pad-status"}`}>
        {capturing ? t("controller.map.press", { button: capturing.label }) : t("controller.map.hint")}
      </p>
      {/* Every button, as a list: choosing one listens, as on the drawing. */}
      <ul className="pad-bindings m-0 grid list-none grid-cols-1 gap-[0.2em] p-0 text-[0.8em] @min-[520px]:grid-cols-2">
        {rebind.outputs.map((output) => {
          const listening = output.output === rebind.capturing;
          return (
            <li key={output.output}>
              <button
                type="button"
                className="pad-binding rounded-row grid w-full cursor-pointer grid-cols-[minmax(0,1fr)_auto] items-center gap-x-[0.6em] px-[0.5em] py-[0.25em] text-left"
                data-capturing={listening || undefined}
                onClick={() => send(listening ? { op: "rebindCancel" } : { op: "rebind", output: output.output })}
              >
                <span className="truncate font-semibold">{output.label}</span>
                <span className="pad-status">{listening ? t("controller.map.waiting") : bindingText(output.binding)}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </>
  );
}
