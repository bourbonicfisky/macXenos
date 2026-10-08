import { send } from "../native";

/* ==========================================================================
   A controller drives the blades the way the keyboard does: every button
   becomes the key the menus already answer (blade-menu.tsx, blade-deck.tsx).

     d-pad / left stick   arrow keys, repeating while held
     A  B  X  Y           a  b  x  y   (X hides the dashboard)
     LB  RB               change blade (← →), even inside a sub-menu's pane
     Guide                hides the dashboard, from anywhere

   Two sources feed it: the page's own Gamepad API, and the app, which reads
   the emulator's controllers while Preferences is open and calls
   window.__xeniaPad(button, down). Either is enough; a press from both within
   one poll counts once.
   ========================================================================== */

type Button = "up" | "down" | "left" | "right" | "a" | "b" | "x" | "y" | "lb" | "rb" | "guide";

const KEYS: Record<Button, string> = {
  up: "ArrowUp",
  down: "ArrowDown",
  left: "ArrowLeft",
  right: "ArrowRight",
  a: "a",
  b: "b",
  x: "x",
  y: "y",
  lb: "ArrowLeft",
  rb: "ArrowRight",
  guide: ""
};

const REPEAT_DELAY_MS = 380;
const REPEAT_EVERY_MS = 110;
const STICK = 0.55;

function press(button: Button) {
  // Guide hides the dashboard from anywhere, sub-menu or not.
  if (button === "guide") {
    send({ op: "close" });
    return;
  }
  const key = KEYS[button];
  // LB / RB change blade even with a sub-menu open: leave it first.
  if (button === "lb" || button === "rb") {
    if (document.documentElement.dataset.submenu === "on") press("b");
  }
  const target = (document.activeElement as HTMLElement | null) ?? document.body;
  target.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
}

type Held = { since: number; last: number };
const held = new Map<Button, Held>();

function update(down: Set<Button>, now: number) {
  for (const button of down) {
    const state = held.get(button);
    if (!state) {
      held.set(button, { since: now, last: now });
      press(button);
    } else if (
      (button === "up" || button === "down" || button === "left" || button === "right") &&
      now - state.since > REPEAT_DELAY_MS &&
      now - state.last > REPEAT_EVERY_MS
    ) {
      state.last = now;
      press(button);
    }
  }
  for (const button of [...held.keys()]) if (!down.has(button)) held.delete(button);
}

// Buttons the app reports as down, merged into each poll.
const fromApp = new Set<Button>();

function poll() {
  const down = new Set<Button>(fromApp);
  for (const pad of navigator.getGamepads?.() ?? []) {
    if (!pad || pad.mapping !== "standard") continue;
    const b = (i: number) => pad.buttons[i]?.pressed === true;
    const [x = 0, y = 0] = pad.axes;
    if (b(12) || y < -STICK) down.add("up");
    if (b(13) || y > STICK) down.add("down");
    if (b(14) || x < -STICK) down.add("left");
    if (b(15) || x > STICK) down.add("right");
    if (b(0)) down.add("a");
    if (b(1)) down.add("b");
    if (b(2)) down.add("x");
    if (b(3)) down.add("y");
    if (b(4)) down.add("lb");
    if (b(5)) down.add("rb");
    if (b(16)) down.add("guide");
  }
  update(down, performance.now());
  window.requestAnimationFrame(poll);
}

let started = false;

export function startGamepad() {
  if (started) return;
  started = true;
  (window as unknown as { __xeniaPad: (button: Button, isDown: boolean) => void }).__xeniaPad = (
    button,
    isDown
  ) => {
    if (!(button in KEYS)) return;
    if (isDown) fromApp.add(button);
    else fromApp.delete(button);
  };
  window.requestAnimationFrame(poll);
}
