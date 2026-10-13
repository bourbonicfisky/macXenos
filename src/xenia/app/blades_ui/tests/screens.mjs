// Screenshots of the blades in a fixed set of states (a hash, then keys or
// eval:<script> steps), for checking that a
// change meant to be invisible (a refactor of the stylesheets, say) is.
//
//   node tests/screens.mjs record <dir>    write <dir>/<state>.png
//   node tests/screens.mjs compare <dir>   compare against them, byte for
//                                          byte; differing ones are written
//                                          beside them as <state>.actual.png

import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { withBlades } from "./harness.mjs";

const STATES = {
  home: ["#home"],
  graphics: ["#graphics"],
  "graphics-resolution": ["#graphics/resolution"],
  "graphics-resolution-open": ["#graphics/resolution", "Enter", "Enter", "ArrowDown"],
  "graphics-unset-inputs": ["#graphics/unset-inputs"],
  profiles: ["#profiles"],
  "profiles-unsigned": ["#profiles", "ArrowDown"],
  "picker": ["#profiles", "Enter", "Enter"],
  "picker-grid": ["#profiles", "Enter", "Enter", "Enter", "ArrowRight", "ArrowDown"],
  "picker-downloading": ["#profiles", "Enter", "Enter", "ArrowDown", "Enter"],
  "controller-map": ["#controller/remap"],
  "controller-map-listening": ["#controller/remap", "Enter", "Enter"],
  "system-sound": ["#system/sound"],
  "system-language": ["#system/language"],
  "system-about": ["#system/about"],
  "system-animation": ["#system/motion"],
  "home-folders": ["#home/game-folders"],
  "home-more": ["#home", ...Array(12).fill("ArrowDown")],
  // Down the list and back up: the pane must not have scrolled away.
  "home-back-up": ["#home", ...Array(12).fill("ArrowDown"), ...Array(9).fill("ArrowUp")],
  "all-games": ["#all-games"],
  "all-games-alpha": ["#all-games", "x"],
  "game-folders": ["#game-folders"],
  "game-folders-armed": ["#game-folders", "ArrowDown", "a"],
  // The first-run offer: the mock's state with the question unanswered.
  "sound-offer": [
    "#home",
    `eval:import("/src/mock.ts").then(({ mockState }) => {
      const state = mockState();
      state.dashSounds.asked = false;
      window.__xeniaReceive(JSON.stringify(state));
    })`
  ]
};

const [mode, dir] = process.argv.slice(2);
if (!["record", "compare"].includes(mode) || !dir) {
  console.error("usage: node tests/screens.mjs record|compare <dir>");
  process.exit(2);
}
mkdirSync(dir, { recursive: true });

let differing = 0;
for (const scheme of ["dark", "light"]) {
  await withBlades(
    async (page, base) => {
      for (const [name, [hash, ...keys]] of Object.entries(STATES)) {
        await page.open(base, hash);
        for (const step of keys) {
          if (step.startsWith("eval:")) await page.eval(step.slice(5));
          else await page.key(step);
        }
        await page.eval("new Promise((r) => setTimeout(r, 400))");
        const shot = await page.screenshot();
        const file = join(dir, `${name}-${scheme}.png`);
        if (mode === "record") {
          writeFileSync(file, shot);
          console.log(`recorded ${name}-${scheme}`);
        } else if (!existsSync(file)) {
          console.log(`MISSING  ${name}-${scheme}`);
          differing++;
        } else if (!readFileSync(file).equals(shot)) {
          writeFileSync(join(dir, `${name}-${scheme}.actual.png`), shot);
          console.log(`DIFFERS  ${name}-${scheme}`);
          differing++;
        } else {
          console.log(`same     ${name}-${scheme}`);
        }
      }
    },
    { colorScheme: scheme }
  );
}
process.exit(differing ? 1 : 0);
