// Smoke tests for the blades page: npm test. Each test loads the page fresh
// on the mock (src/mock.ts) and drives it with keys, as a keyboard or a pad
// (gamepad.ts turns buttons into the same keys) would.

import { withBlades } from "./harness.mjs";

const BLADES = ["home", "graphics", "controller", "audio", "profiles", "system", "advanced"];
const focusedRow = "document.querySelector('.blade[data-state=open] .menu-row[data-focused=true] span')?.textContent";
const focusedSubRow =
  "document.querySelector('.subblade-row[data-focused=true] :is(.subblade-name, .subblade-label)')?.textContent";
const subLegend = "[...document.querySelectorAll('.subblade .legend-entry')].map((e) => e.textContent).join(' | ')";

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

async function downTo(page, label, limit = 30) {
  for (let i = 0; i < limit; i++) {
    if ((await page.eval(focusedRow)) === label) return;
    await page.key("ArrowDown");
  }
  throw new Error(`no row "${label}"`);
}

const tests = {
  async "every blade opens from its deep link"(page, base) {
    for (const id of BLADES) {
      await page.open(base, `#${id}`);
      const open = await page.eval("document.querySelector('.blade[data-state=open]')?.id");
      expect(open === id, `#${id} opened ${open}`);
      const title = await page.eval("document.querySelector('.blade[data-state=open] h2')?.textContent");
      expect(title, `#${id} has no title`);
    }
  },

  async "B on a blade goes back to Home"(page, base) {
    await page.open(base, "#graphics");
    await page.key("b");
    await page.eval("new Promise((r) => setTimeout(r, 300))");
    expect((await page.eval("document.querySelector('.blade[data-state=open]')?.id")) === "home", "B did not go Home");
  },

  async "Y switches day and night on the blades"(page, base) {
    await page.open(base, "#home");
    const before = await page.eval("document.documentElement.dataset.theme");
    await page.key("y");
    const after = await page.eval("document.documentElement.dataset.theme");
    expect(before && after && before !== after, `theme ${before} -> ${after}`);
  },

  async "legends stack B over A"(page, base) {
    await page.open(base, "#graphics");
    const order = await page.eval(`(() => {
      const entries = [...document.querySelectorAll('.blade[data-state=open] .legend-stack--right .legend-entry')];
      return entries.map((e) => e.textContent + ':' + Math.round(e.getBoundingClientRect().top));
    })()`);
    expect(order.length === 2 && order[0].startsWith("Back") && order[1].startsWith("Select"), `legend ${order}`);
    expect(Number(order[0].split(":")[1]) < Number(order[1].split(":")[1]), "Back is not above Select");
  },

  async "All games opens from Home and goes back to its row"(page, base) {
    await page.open(base, "#home");
    await downTo(page, "All games");
    await page.key("Enter");
    await page.eval("new Promise((r) => setTimeout(r, 300))");
    expect((await page.eval("document.querySelector('.subblade h2')?.textContent")) === "All Games", "All Games not open");
    expect((await page.eval(focusedSubRow)) === "Sonic Generations", "first game is not the most recent");
    await page.key("ArrowDown");
    expect((await page.eval(focusedSubRow)) === "Just Cause 2", "↓ did not move");
    await page.key("ArrowRight");
    expect((await page.eval("location.hash")) === "#home", "→ changed blade under All Games");
    const theme = await page.eval("document.documentElement.dataset.theme");
    await page.key("y");
    expect((await page.eval("document.documentElement.dataset.theme")) === theme, "Y changed the theme in All Games");
    await page.key("b");
    await page.eval("new Promise((r) => setTimeout(r, 700))");
    expect(!(await page.eval("!!document.querySelector('.subblade-layer')")), "B did not close All Games");
    expect((await page.eval("document.activeElement?.textContent")).startsWith("All games"), "focus did not return");
  },

  async "X sorts All Games alphabetically and remembers it"(page, base) {
    await page.open(base, "#all-games");
    expect((await page.eval(subLegend)).includes("Most recently played"), "legend does not show the order");
    await page.key("x");
    const names = await page.eval("[...document.querySelectorAll('.subblade-name')].map((e) => e.textContent)");
    const sorted = [...names].sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));
    expect(JSON.stringify(names) === JSON.stringify(sorted), `not alphabetical: ${names}`);
    expect((await page.eval("localStorage.getItem('macxenos.allGames.sort')")) === "alpha", "order not remembered");
  },

  async "Game Folders adds, and removes only on a second A"(page, base) {
    await page.open(base, "#game-folders");
    expect((await page.eval(focusedSubRow)) === "Add a Folder", "first row is not Add a Folder");
    expect((await page.eval(subLegend)).includes("Add"), "legend does not say Add");
    const folders = () => page.eval("document.querySelectorAll('.subblade-item .subblade-row').length - 1");
    const start = await folders();
    await page.key("ArrowDown");
    expect((await page.eval(subLegend)).includes("Remove"), "legend does not say Remove");
    await page.key("a");
    expect((await page.eval(subLegend)).includes("Confirm"), "first A did not arm the removal");
    expect((await folders()) === start, "first A removed the folder");
    await page.key("a");
    expect((await folders()) === start - 1, "second A did not remove the folder");
    await page.key("ArrowUp");
    await page.key("a");
    expect((await folders()) === start, "A on Add a Folder did not add one");
  },

  async "Game folders pane offers Edit and Refresh side by side"(page, base) {
    await page.open(base, "#home/game-folders");
    const tops = await page.eval(`[...document.querySelectorAll('.blade[data-state=open] .menu-detail:not([hidden]) .pane-button')]
      .map((b) => b.textContent + ':' + Math.round(b.getBoundingClientRect().top))`);
    expect(tops.length === 2 && tops[0].startsWith("Edit folders") && tops[1].startsWith("Refresh folders"), `${tops}`);
    expect(tops[0].split(":")[1] === tops[1].split(":")[1], "the buttons are not on one line");
  },

  async "Dashboard sounds puts the download first when missing"(page, base) {
    await page.open(base, "#system/sound");
    const order = await page.eval(`[...document.querySelectorAll('.blade[data-state=open] .menu-detail:not([hidden]) button')].map((b) => b.textContent)`);
    expect(order[0] === "Download sounds", `first control is ${order[0]}`);
  }
};

const only = process.argv[2];
let failed = 0;
await withBlades(async (page, base) => {
  for (const [name, test] of Object.entries(tests)) {
    if (only && !name.includes(only)) continue;
    await page.eval("localStorage.clear()").catch(() => {});
    try {
      await test(page, base);
      console.log(`ok    ${name}`);
    } catch (error) {
      failed++;
      console.log(`FAIL  ${name}\n      ${error.message}`);
    }
  }
});
process.exit(failed ? 1 : 0);
