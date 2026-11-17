// Smoke tests for the blades page: npm test. Each test loads the page fresh
// on the mock (src/mock.ts) and drives it with keys, as a keyboard or a pad
// (gamepad.ts turns buttons into the same keys) would.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
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
  async "Color vision palettes persist, preserve game settings and support both themes"(page, base) {
    await page.call("Emulation.setFocusEmulationEnabled", { enabled: true });
    await page.open(base, "#system/color-vision");
    const gameSettings = await page.eval("import('/src/native.ts').then(({getNative}) => JSON.stringify(getNative().cvars))");
    expect((await page.eval("document.documentElement.dataset.colorVision")) === "off", "not off by default");
    for (const [mode, label] of [["red-green", "Red–green"], ["blue-yellow", "Blue–yellow"]]) {
      await page.eval(`[...document.querySelectorAll('.blade[data-state=open] .menu-detail:not([hidden]) [role=radio]')].find(b => !b.closest('[inert]') && b.checkVisibility() && b.textContent === ${JSON.stringify(label)}).click()`);
      expect((await page.eval("document.documentElement.dataset.colorVision")) === mode, "palette did not apply");
      expect((await page.eval("localStorage.getItem('macxenos-color-vision')")) === mode, "palette not saved");
      await page.open(base, "#system/color-vision");
      expect((await page.eval("document.documentElement.dataset.colorVision")) === mode, "palette lost on reload");
      for (const theme of ["light", "dark"]) {
        await page.eval(`import('/src/ui/preferences.ts').then(({setPreference}) => setPreference('theme', '${theme}'))`);
        // Wait for Vite's stylesheet and the preference update before reading tokens.
        await page.eval(`new Promise((resolve, reject) => {
          let tries = 0;
          const ready = () => {
            if (getComputedStyle(document.documentElement).getPropertyValue('--cv-hi').trim()) resolve();
            else if (++tries < 120) requestAnimationFrame(ready);
            else reject(new Error('color vision stylesheet did not become ready'));
          };
          ready();
        })`);
        const contrast = await page.eval(`(() => {
          const s = getComputedStyle(document.querySelector('.blade[data-state=open]'));
          const lum = hex => {
            const c = hex.trim().slice(1).match(/../g).map(x => parseInt(x,16)/255).map(x => x <= .04045 ? x/12.92 : ((x+.055)/1.055)**2.4);
            return c[0]*.2126+c[1]*.7152+c[2]*.0722;
          };
          const ratio = (a,b) => (Math.max(a,b)+.05)/(Math.min(a,b)+.05);
          return ['--cv-hi','--cv-base','--cv-lo','--cv-hover'].flatMap(bg => ['--cv-ink','--cv-soft'].map(fg => ratio(lum(s.getPropertyValue(bg)),lum(s.getPropertyValue(fg))))).concat(ratio(lum(s.getPropertyValue('--cv-accent')),lum(s.getPropertyValue('--cv-selected-ink'))));
        })()`);
        const accent = await page.eval("getComputedStyle(document.documentElement).getPropertyValue('--cv-accent').trim()");
        expect(accent === (mode === "red-green" ? (theme === "light" ? "#005a9c" : "#8ecbff") : (theme === "light" ? "#81345d" : "#ffb0cf")), "wrong palette accent");
        expect(contrast.every(r => r >= 4.5), `insufficient text contrast ${mode} ${theme}: ${contrast}`);
        const state = await page.eval(`(() => {
          const b = [...document.querySelectorAll('.blade[data-state=open] .menu-detail:not([hidden]) [role=radio][aria-checked=true]')].find(b => !b.closest('[inert]') && b.checkVisibility());
          b.focus(); const s = getComputedStyle(b);
          return [s.outlineStyle, s.outlineWidth, s.boxShadow, s.backgroundColor, s.color];
        })()`);
        expect(state[3] !== "rgba(0, 0, 0, 0)" && state[3] !== state[4], "selection needs a contrasting fill and label");
        expect(state[0] === "dashed" && state[1] === "3px" && state[2].includes("inset"), `focus/selection cues missing: ${state}`);
        if (process.env.BLADES_COLOR_VISION_SHOTS) {
          mkdirSync(process.env.BLADES_COLOR_VISION_SHOTS, {recursive:true});
          writeFileSync(join(process.env.BLADES_COLOR_VISION_SHOTS, `${mode}-${theme}.png`), await page.screenshot());
        }
      }
    }
    expect((await page.eval("import('/src/native.ts').then(({getNative}) => JSON.stringify(getNative().cvars))")) === gameSettings, "UI preference changed game settings");
    await page.eval("[...document.querySelectorAll('.blade[data-state=open] .menu-detail:not([hidden]) [role=radio]')].find(b => !b.closest('[inert]') && b.checkVisibility() && b.textContent === 'Off').click()");
    expect((await page.eval("document.documentElement.dataset.colorVision")) === "off", "Off did not restore original palette");
    expect((await page.eval("localStorage.getItem('macxenos-color-vision')")) === null, "Off not reset to default");
    await page.eval("localStorage.setItem('macxenos-color-vision', 'invalid')");
    await page.open(base, "#system/color-vision");
    expect((await page.eval("document.documentElement.dataset.colorVision")) === "off", "invalid saved choice did not fall back");
    await page.key("a");
    await page.key("ArrowRight");
    expect((await page.eval("document.documentElement.dataset.colorVision")) === "red-green", "controller-style navigation did not select a palette");
    await page.key("b");
    expect((await page.eval("document.activeElement?.classList.contains('menu-row')")), `B did not return to the setting row: ${await page.eval("document.activeElement?.outerHTML.slice(0,200)")}`);
    await page.call("Emulation.setFocusEmulationEnabled", { enabled: false });
  },

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
    const tab = await page.eval("document.querySelector('.subblade-tab')?.textContent");
    expect(tab.includes("dmug’s game library"), `the tab reads "${tab}"`);
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

  async "X goes on to the two orders backwards, then round again"(page, base) {
    await page.open(base, "#all-games");
    const names = "[...document.querySelectorAll('.subblade-name')].map((e) => e.textContent)";
    // Whatever an earlier test left: step to most recently played first.
    for (let i = 0; i < 4 && !(await page.eval(subLegend)).includes("Most recently played"); i++) await page.key("x");
    const recent = await page.eval(names);
    await page.key("x");
    const alpha = await page.eval(names);
    await page.key("x");
    expect((await page.eval(subLegend)).includes("Least recently played"), "third order is not least recently played");
    expect(JSON.stringify(await page.eval(names)) === JSON.stringify([...recent].reverse()), "not most recent backwards");
    await page.key("x");
    expect((await page.eval(subLegend)).includes("Reverse alphabetical"), "fourth order is not reverse alphabetical");
    expect(JSON.stringify(await page.eval(names)) === JSON.stringify([...alpha].reverse()), "not alphabetical backwards");
    expect((await page.eval("localStorage.getItem('macxenos.allGames.sort')")) === "alphaReverse", "order not remembered");
    await page.key("x");
    expect(JSON.stringify(await page.eval(names)) === JSON.stringify(recent), "a fourth X did not come back round");
  },

  async "Y shows All Games as cards and back, and the arrows move across them"(page, base) {
    await page.open(base, "#all-games");
    const cards = "document.querySelectorAll('.game-card').length";
    const focusedCard = "document.querySelector('.game-card[data-focused=true] .game-card-name')?.textContent";
    if (await page.eval(cards)) await page.key("y"); // An earlier test left cards.
    const rows = await page.eval("document.querySelectorAll('.subblade-row').length");
    expect(rows > 1 && (await page.eval(cards)) === 0, "not a list to begin with");
    expect((await page.eval(subLegend)).includes("Cards"), "legend does not offer cards");
    const second = await page.eval("document.querySelectorAll('.subblade-name')[1]?.textContent");
    await page.key("y");
    expect((await page.eval(cards)) === rows, `cards ${await page.eval(cards)} of ${rows} games`);
    expect((await page.eval(subLegend)).includes("List"), "legend does not offer the list");
    expect((await page.eval("localStorage.getItem('macxenos.allGames.view')")) === "cards", "view not remembered");
    const across = await page.eval(`(() => {
      const tiles = [...document.querySelectorAll('.game-card')];
      return tiles.filter((tile) => tile.offsetTop === tiles[0].offsetTop).length;
    })()`);
    expect(across > 1, `only ${across} card across`);
    await page.key("ArrowRight");
    expect((await page.eval(focusedCard)) === second, "→ did not move to the next card");
    await page.key("ArrowLeft");
    await page.key("ArrowDown");
    const below = await page.eval(`document.querySelectorAll('.game-card-name')[${across}]?.textContent`);
    if (below) expect((await page.eval(focusedCard)) === below, "↓ did not move a row down");
    const card = await page.eval(`(() => {
      const c = document.querySelector('.game-card');
      return !!c.querySelector('.game-card-icon') && !!c.querySelector('.game-card-name') && !!c.querySelector('.game-card-sub');
    })()`);
    expect(card, "a card is missing its icon, name or achievements line");
    await page.key("y");
    expect((await page.eval(cards)) === 0, "a second Y did not go back to the list");
    expect((await page.eval("localStorage.getItem('macxenos.allGames.view')")) === "list", "list not remembered");
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

  async "the language preference reaches translated text"(page, base) {
    await page.open(base, "#home");
    await page.eval("localStorage.setItem('macxenos-language', 'en-XA')");
    await page.open(base, "#home");
    const back = await page.eval(
      "document.querySelector('.blade[data-state=open] .legend-stack--right .legend-entry')?.textContent"
    );
    expect(back === "[Ɓåçķ]B", `legend reads ${back}`);
    const row = await page.eval(
      "[...document.querySelectorAll('.blade[data-state=open] .menu-row span')].map((e) => e.textContent).find((x) => x.includes('ĝåɱéš'))"
    );
    expect(row === "[Åļļ ĝåɱéš]", `All games row reads ${row}`);
    await page.eval("localStorage.removeItem('macxenos-language')");
    await page.open(base, "#home");
    expect((await page.eval(focusedRow)) !== undefined, "page did not come back");
  },

  async "Russian switches every blade and live options, without changing the console language"(page, base) {
    await page.open(base, "#system/language");
    const consoleLanguage = await page.eval("import('/src/native.ts').then(({ getNative }) => getNative().console.fields.find((f) => f.id === 'language')?.value)");
    await page.eval("[...document.querySelectorAll('.blade[data-state=open] [role=radio]')].find((b) => b.textContent === 'Русский').click()");
    expect((await page.eval("document.documentElement.lang")) === "ru", "Russian did not activate");
    expect((await page.eval("document.title")) === "Настройки", "page title stayed English");
    expect((await page.eval("localStorage.getItem('macxenos-language')")) === "ru", "Russian was not saved");
    const titles = { home: "macXenos", graphics: "Графика", controller: "Контроллер", audio: "Звук и ЦП", profiles: "Профили", system: "Система", advanced: "Дополнительно" };
    for (const [id, expected] of Object.entries(titles)) {
      await page.open(base, `#${id}`);
      const title = await page.eval("document.querySelector('.blade[data-state=open] h2')?.textContent");
      expect(title === expected, `#${id}: ${title}`);
    }
    const pane = "document.querySelector('.blade[data-state=open] .menu-detail:not([hidden])')";
    await page.open(base, "#graphics/upscaling");
    expect((await page.eval(`${pane}.textContent`)).includes("Билинейная"), "static renderer options stayed English");
    await page.open(base, "#controller/keyboard");
    expect((await page.eval(`${pane}.textContent`)).includes("Игрок 1"), "keyboard player options stayed English");
    await page.open(base, "#system/clock");
    expect((await page.eval(`${pane}.textContent`)).includes("Отключить переход на летнее время"), "native console labels stayed English");
    expect((await page.eval("import('/src/native.ts').then(({ getNative }) => getNative().console.fields.find((f) => f.id === 'language')?.value)")) === consoleLanguage, "dashboard language changed the console language");
    await page.eval(`import('/src/mock.ts').then(({ mockState }) => {
      const state = mockState();
      state.console.fields.find((f) => f.id === 'language').options.push({ value: 12, label: 'Russian' });
      state.console.fields.find((f) => f.id === 'country').options = [{ value: 88, label: 'RU' }, { value: 103, label: 'US' }];
      window.__xeniaReceive(JSON.stringify(state));
    })`);
    await page.eval("[...document.querySelectorAll('.blade[data-state=open] .menu-row')].find((b) => b.textContent.startsWith('Язык и регион')).click()");
    const consoleText = await page.eval(`${pane}.textContent`);
    expect(consoleText.includes("русский") && consoleText.includes("Россия"), `console choices were not localized: ${consoleText}`);
    await page.open(base, "#system/language");
    await page.eval("[...document.querySelectorAll('.blade[data-state=open] [role=radio]')].find((b) => b.textContent === 'Как на моём Mac').click()");
    await page.eval("import('/src/i18n/index.ts').then((i18n) => i18n.setSystemLanguages(['ru-RU', 'en-US']))");
    expect((await page.eval("document.documentElement.lang")) === "ru", "Russian Mac language was not matched");
    await page.eval("[...document.querySelectorAll('.blade[data-state=open] [role=radio]')].find((b) => b.textContent === 'English').click()");
    await page.open(base, "#graphics/upscaling");
    expect((await page.eval(`${pane}.textContent`)).includes("Bilinear"), "static renderer options stayed Russian");
    expect((await page.eval("document.title")) === "Preferences", "page title stayed Russian");
  },

  async "every language switches the blades, in its own name, and follows the Mac"(page, base) {
    const { readdirSync } = await import("node:fs");
    const codes = readdirSync(new URL("../src/i18n/catalogs/", import.meta.url))
      .filter((file) => file.endsWith(".ts"))
      .map((file) => file.slice(0, -3));
    const names = {
      da: "Dansk", de: "Deutsch", es: "Español", fi: "Suomi", fr: "Français", it: "Italiano", ja: "日本語", ko: "한국어",
      nb: "Norsk bokmål", nl: "Nederlands", pl: "Polski", "pt-BR": "Português (Brasil)", ru: "Русский", sv: "Svenska",
      "zh-Hans": "简体中文", "zh-Hant": "繁體中文"
    };
    for (const code of codes) {
      await page.open(base, "#system/language");
      const option = `[...document.querySelectorAll('.blade[data-state=open] [role=radio]')].find((b) => b.textContent === ${JSON.stringify(names[code])})`;
      expect(await page.eval(`!!${option}`), `no option named ${names[code]} for ${code}`);
      expect((await page.eval(`${option}.querySelector('[lang]')?.lang`)) === code, `${code}'s name is not marked lang=${code}`);
      await page.eval(`${option}.click()`);
      expect((await page.eval("document.documentElement.lang")) === code, `${code} did not activate`);
      const want = await page.eval(`import('/src/i18n/catalogs/${code}.ts').then((m) => m.default["graphics.title"])`);
      await page.open(base, "#graphics");
      const title = await page.eval("document.querySelector('.blade[data-state=open] h2')?.textContent");
      expect(title === want, `${code}: Graphics reads ${title}, not ${want}`);
      await page.eval("localStorage.clear()");
    }
    // Same as my Mac: regional variants find their language; Portugal's
    // Portuguese reads Brazil's, Nynorsk Bokmål, Singapore's Chinese
    // Simplified.
    await page.open(base, "#system/language");
    for (const [tags, want] of [[["hu-HU", "fr-CA"], "fr"], [["es-MX"], "es"], [["zh-HK"], "zh-Hant"], [["pt-PT"], "pt-BR"], [["nn-NO"], "nb"], [["zh-SG"], "zh-Hans"], [["hu-HU"], "en"]]) {
      await page.eval(`import('/src/i18n/index.ts').then((i18n) => i18n.setSystemLanguages(${JSON.stringify(tags)}))`);
      const lang = await page.eval("document.documentElement.lang");
      expect(lang === want, `Mac languages ${tags} gave ${lang}, not ${want}`);
    }
  },

  async "Dashboard language offers Same as my Mac and English"(page, base) {
    await page.open(base, "#system/language");
    const row = await page.eval(
      "[...document.querySelectorAll('.blade[data-state=open] .menu-row')].map((e) => e.textContent).find((x) => x.startsWith('Dashboard language'))"
    );
    expect(row === "Dashboard languageEnglish", `row reads ${row}`);
    const options = await page.eval(
      "[...document.querySelectorAll('.blade[data-state=open] [role=radio]')].map((b) => b.textContent)"
    );
    expect(options[0] === "Same as my Mac" && options.includes("English"), `options ${options}`);
    expect((await page.eval("document.documentElement.lang")) === "en", "lang is not en");
  },

  async "Animation holds speed and background; reduced motion holds the light still"(page, base) {
    await page.open(base, "#system/motion");
    const groups = await page.eval(
      "[...document.querySelectorAll('.blade[data-state=open] .menu-detail:not([hidden]) .pref-group')].map((e) => e.textContent)"
    );
    expect(groups.join() === "Speed,Background", `groups ${groups}`);
    // The harness runs with the OS asking for reduced motion.
    expect((await page.eval("document.documentElement.dataset.background")) === "still", "light drifts under reduced motion");
    expect(
      (await page.eval("document.querySelector('.blade[data-state=open] .menu-detail:not([hidden]) .menu-note')?.textContent")) ===
        "Held still while motion is reduced.",
      "no note that the light is held"
    );
    // Let the chosen speed win over the OS: the light drifts.
    await page.eval("localStorage.setItem('macxenos-motion-override', 'on')");
    await page.open(base, "#system/motion");
    expect((await page.eval("document.documentElement.dataset.background")) === "shifting", "light does not drift");
    const animation = await page.eval(
      "getComputedStyle(document.querySelector('.blade[data-state=open] .blade-skin--panel'), '::before').animationName"
    );
    expect(animation === "blade-light-drift", `panel light animation is ${animation}`);
    // Still holds it.
    await page.eval("localStorage.setItem('macxenos-background', 'still')");
    await page.open(base, "#system/motion");
    expect((await page.eval("document.documentElement.dataset.background")) === "still", "Still does not hold the light");
  },

  async "Profiles shows each profile's gamercard"(page, base) {
    await page.open(base, "#profiles");
    const card = () =>
      page.eval(`(() => {
        const card = document.querySelector('.blade[data-state=open] .gamercard');
        return card && [card.querySelector('.gamercard-head').textContent,
          ...[...card.querySelectorAll('dd')].map((d) => d.textContent),
          card.querySelector('.gamercard-note') ? 'note' : ''].join('|');
      })()`);
    expect((await card()) === "dmug|10|1250|56|", `player one's card: ${await card()}`);
    await page.key("ArrowDown");
    expect((await card()) === "Player Two|–|–|–|note", `signed-out card: ${await card()}`);
  },

  async "Change gamer picture offers the official library and shows progress"(page, base) {
    await page.open(base, "#profiles");
    const rows = await page.eval("[...document.querySelectorAll('.blade[data-state=open] .menu-row')].map((e) => e.textContent).join('|')");
    expect(!rows.includes("Gamer pictures"), `Profiles still has a Gamer pictures row: ${rows}`);
    await page.key("Enter");
    await page.key("Enter");
    await page.eval("new Promise((r) => setTimeout(r, 300))");
    const note = "document.querySelector('.subblade .menu-note')?.textContent ?? ''";
    const text = await page.eval(note);
    expect(text.includes("official gamer picture library") && text.includes("archive.org") && text.includes("117 MB"), `note: ${text}`);
    await page.key("ArrowDown");
    expect((await page.eval("document.querySelector('.subblade-row[data-focused=true] .subblade-name')?.textContent")) === "More pictures", "no More pictures row");
    await page.key("Enter");
    expect(await page.eval("!!document.querySelector('.subblade [role=progressbar]')"), "no progress bar while downloading");
    expect((await page.eval("document.querySelector('.subblade-row[data-focused=true] .subblade-sub')?.textContent")) === "Downloading: 35%", "row does not show progress");
  },

  async "the picture picker sets a profile's gamer picture"(page, base) {
    await page.open(base, "#profiles");
    await page.key("Enter");
    expect((await page.eval("document.activeElement?.textContent")) === "Change gamer picture", "no Change button");
    await page.key("Enter");
    await page.eval("new Promise((r) => setTimeout(r, 300))");
    expect((await page.eval("document.querySelector('.subblade h2')?.textContent")) === "Gamer Picture", "picker not open");
    await page.key("Enter");
    expect((await page.eval("document.querySelectorAll('.subblade-tile').length")) === 175, "macXenos grid is not 175 pictures");
    const index = "[...document.querySelectorAll('.subblade-tile')].findIndex((t) => t.dataset.focused === 'true')";
    const start = await page.eval(index);
    expect(
      (await page.eval("document.querySelector('.subblade-tile[data-focused=true] img')?.getAttribute('src')")) ===
        "/gamer-pictures/robot.png",
      "the grid did not open on the profile's picture"
    );
    const across = await page.eval(
      "(() => { const t = [...document.querySelectorAll('.subblade-tile')]; return t.filter((x) => x.offsetTop === t[0].offsetTop).length; })()"
    );
    await page.key("ArrowRight");
    await page.key("ArrowDown");
    expect((await page.eval(index)) === start + 1 + across, `→ ↓ moved from ${start} to ${await page.eval(index)} (${across} across)`);
    await page.key("b");
    expect((await page.eval("document.querySelector('.subblade h2')?.textContent")) === "Gamer Picture", "B did not go back a level");
    await page.key("Enter");
    await page.key("Home");
    await page.key("ArrowRight");
    await page.key("a");
    await page.eval("new Promise((r) => setTimeout(r, 700))");
    expect(!(await page.eval("!!document.querySelector('.subblade-layer')")), "choosing did not close the picker");
    const card = await page.eval("document.querySelector('.blade[data-state=open] .gamercard-picture')?.getAttribute('src')");
    expect(card === "/gamer-pictures/abstract.png", `gamercard shows ${card}`);
  },

  async "a controller opens a drop-down, moves through it and picks"(page, base) {
    // As the app forwards a pad: window.__xeniaPad(button, down).
    const pad = (b) =>
      page.eval(`new Promise((r) => { window.__xeniaPad("${b}", true);
        setTimeout(() => { window.__xeniaPad("${b}", false); setTimeout(r, 120); }, 120); })`);
    const value = "document.querySelector('.blade[data-state=open] .pref-dropdown span')?.textContent";
    await page.open(base, "#graphics/resolution");
    await pad("a");
    expect(await page.eval("document.activeElement?.classList.contains('pref-dropdown')"), "A did not reach the drop-down");
    await pad("a");
    expect(await page.eval("!!document.querySelector('[role=listbox]')"), "A did not open the list");
    await pad("down");
    await pad("a");
    expect(!(await page.eval("!!document.querySelector('[role=listbox]')")), "A did not close the list");
    expect((await page.eval(value)) === "2x — 1440p", `picked ${await page.eval(value)}`);
  },

  async "Manage pictures shows the folder and deletes the library on a second A"(page, base) {
    await page.open(base, "#profiles");
    await page.eval(`import("/src/mock.ts").then(({ mockState }) => {
      const state = mockState();
      Object.assign(state.gamerPictures, { status: "ready", pictures: 10964, titles: 313, bytes: 87_300_000 });
      window.__xeniaReceive(JSON.stringify(state));
    })`);
    await page.key("Enter");
    await page.key("Enter");
    await page.eval("new Promise((r) => setTimeout(r, 300))");
    const row = "document.querySelector('.subblade-row[data-focused=true] .subblade-name')?.textContent";
    await page.key("End");
    expect((await page.eval(row)) === "Manage pictures", `last row is ${await page.eval(row)}`);
    await page.key("Enter");
    expect((await page.eval("document.querySelector('.subblade h2')?.textContent")) === "Manage pictures", "Manage did not open");
    expect((await page.eval(row)) === "Show in Finder", "first choice is not Show in Finder");
    await page.key("ArrowDown");
    await page.key("a");
    expect((await page.eval("document.querySelector('.subblade-action[data-armed]')?.textContent")) === "Press A again", "first A did not arm");
    await page.key("a");
    await page.eval("new Promise((r) => setTimeout(r, 200))");
    expect((await page.eval("document.querySelector('.subblade h2')?.textContent")) === "Gamer Picture", "did not return to the collections");
    await page.key("End");
    expect((await page.eval(row)) === "More pictures", "library still installed after delete");
  },

  async "Volume steps with left and right on the controller"(page, base) {
    await page.open(base, "#audio/volume");
    await page.key("Enter");
    const slider = "document.activeElement?.getAttribute('aria-valuenow')";
    expect((await page.eval(slider)) === "100", `volume starts at ${await page.eval(slider)}`);
    await page.key("ArrowLeft");
    await page.key("ArrowLeft");
    expect((await page.eval(slider)) === "90", `two steps left reads ${await page.eval(slider)}`);
    await page.key("ArrowRight");
    expect((await page.eval(slider)) === "95", `a step right reads ${await page.eval(slider)}`);
    const row = await page.eval("document.querySelector('.blade[data-state=open] .menu-row[data-focused=true]')?.textContent");
    expect(row === "Volume95%", `row reads ${row}`);
  },

  async "Anti-dead zones offer presets for each stick and a custom value"(page, base) {
    await page.open(base, "#controller/anti-deadzones");
    const pane = "document.querySelector('.blade[data-state=open] .menu-detail:not([hidden])')";
    const text = await page.eval(`${pane}.textContent`);
    for (const label of ["Left stick", "Right stick", "Off", "15%", "24%", "27%", "Custom"])
      expect(text.includes(label), `the pane lacks "${label}"`);
    const push = (edit) => page.eval(`import("/src/mock.ts").then(({ mockState }) => {
      const state = mockState();
      (${edit})(state);
      window.__xeniaReceive(JSON.stringify(state));
    })`);
    const sliders = `${pane}.querySelectorAll('[role=slider], input[type=range]').length`;
    expect((await page.eval(sliders)) === 0, "a slider shows with both sticks on a preset");
    // A value off the presets is Custom, and brings that stick's slider.
    await push("(s) => { s.cvars.right_stick_anti_deadzone_percentage.value = 0.2; }");
    await page.eval("new Promise((r) => setTimeout(r, 100))");
    expect((await page.eval(sliders)) === 1, "no slider for a custom value");
    expect((await page.eval(`${pane}.textContent`)).includes("20%"), "the custom value is not shown");
  },

  async "Vibration offers a test, and says when a controller cannot rumble"(page, base) {
    await page.open(base, "#controller/vibration");
    const pane = "document.querySelector('.blade[data-state=open] .menu-detail:not([hidden])')";
    const test = `[...${pane}.querySelectorAll('button')].find((b) => b.textContent === 'Test vibration')`;
    expect(await page.eval(`!!${test}`), "no Test vibration button");
    expect((await page.eval(`${test}.getAttribute('aria-disabled')`)) === null,
      "the test is disabled with a controller that rumbles");
    const push = (edit) => page.eval(`import("/src/mock.ts").then(({ mockState }) => {
      const state = mockState();
      (${edit})(state);
      window.__xeniaReceive(JSON.stringify(state));
    })`);
    // A controller that reports no motors: named in a note, and nothing to test.
    await push("(s) => { s.controllers[0].rumble = false; s.controllers[0].name = 'Plain Pad'; }");
    await page.eval("new Promise((r) => setTimeout(r, 100))");
    expect((await page.eval(`${pane}.textContent`)).includes("Plain Pad does not report vibration motors"),
      "no note for a controller without motors");
    expect((await page.eval(`${test}.getAttribute('aria-disabled')`)) === "true",
      "the test is offered for a controller without motors");
    // Vibration off: the test is unavailable and says why.
    await push("(s) => { s.cvars.vibration.value = false; }");
    await page.eval("new Promise((r) => setTimeout(r, 100))");
    expect((await page.eval(`${pane}.textContent`)).includes("Turn vibration on to test it."),
      "no hint with vibration off");
    expect((await page.eval(`${test}.getAttribute('aria-disabled')`)) === "true",
      "the test is offered with vibration off");
    // No controller at all.
    await push("(s) => { s.controllers[0].present = false; }");
    await page.eval("new Promise((r) => setTimeout(r, 100))");
    expect((await page.eval(`${pane}.textContent`)).includes("Connect a controller to test its vibration."),
      "no hint without a controller");
  },

  async "Game installs chooses a folder and goes back to the default"(page, base) {
    await page.open(base, "#system/installs");
    const pane = "document.querySelector('.blade[data-state=open] .menu-detail:not([hidden])')";
    const button = (label) =>
      `[...${pane}.querySelectorAll('button')].find((b) => b.textContent === '${label}')`;
    expect((await page.eval(`${pane}.textContent`)).includes("Default (the storage folder)"), "no default shown");
    await page.eval(`${button("Choose folder…")}.click()`);
    await page.eval("new Promise((r) => setTimeout(r, 100))");
    expect((await page.eval(`${pane}.textContent`)).includes("/Volumes/Games/Chosen"), "chosen folder not shown");
    await page.eval(`${button("Use the default")}.click()`);
    await page.eval("new Promise((r) => setTimeout(r, 100))");
    expect((await page.eval(`${pane}.textContent`)).includes("Default (the storage folder)"), "default not restored");
  },

  async "Game saves chooses a folder"(page, base) {
    await page.open(base, "#system/saves");
    const pane = "document.querySelector('.blade[data-state=open] .menu-detail:not([hidden])')";
    expect((await page.eval(`${pane}.textContent`)).includes("Default (Documents › macXenos › Saves)"), "no default shown");
    await page.eval(`[...${pane}.querySelectorAll('button')].find((b) => b.textContent === 'Choose folder…').click()`);
    await page.eval("new Promise((r) => setTimeout(r, 100))");
    expect((await page.eval(`${pane}.textContent`)).includes("/Volumes/Games/Chosen"), "chosen folder not shown");
    const row = await page.eval("document.querySelector('.blade[data-state=open] .menu-row[data-focused=true]')?.textContent");
    expect(row === "Game savesCustom", `row reads ${row}`);
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
