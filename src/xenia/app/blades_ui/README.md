# Preferences blades

The app's Preferences, drawn as the Xbox 360 Blades dashboard (2005–08):
metal tabs stacked down both edges, one saturated blade open in the middle, a
focus list on the left of it and a pane on the right that follows the cursor.
The look and the components come from the macXenos site
(`xenia-mac-site`, see its `RESUME.md` for the design notes: the tab curve,
the faceted highlight, the tone tokens).

It is the app's whole front end: the start screen (Home, with recent games),
every setting, profiles, the music player and the in-game dashboard. Open it
over a game with a tap of a controller's Guide button, F6 (both configurable
under System), ⌘, (the application menu's Settings…), or the View,
Controller and Console menus, which open it on their own blade. While it is
open over a game the game is frozen and muted (`dashboard_freezes_game`). A
hash picks the blade and, optionally, the row: `#controller/hotkeys`.

## All Games and game folders

Home's All games opens a sub-blade (`src/ui/all-games.tsx`) that slides out
over the dashboard, as the 360's game library did: every game in the user's
game folders plus every game played here, with icons, achievements and
gamerscore from the signed-in profile. X sorts it alphabetically or by most
recently played (remembered in the page's local storage); `#all-games` opens
it directly. Home's Game folders pane lists the folders and offers Edit
folders, a second sub-blade from the right (`src/ui/folder-editor.tsx`,
`#game-folders`) that adds them and removes them on a second A, and Refresh
folders. Both sub-blades share `src/ui/sub-blade.tsx`.

The app side is `../game_library.{h,cc}`: folders live in
`<storage_root>/game_folders.toml`; a background scan finds disc images,
unpacked games (a folder with default.xex) and downloaded game packages,
reads each one's title ID (and a package's name and icon) without starting
it, and caches that in `game_index.toml`. `blades_model.cc` merges the scan
with the play history (`recent.toml`, which now keeps every game played; the
`recent_titles_entry_amount` setting only limits the File menu) into
`allGames`.

The legends stack their face buttons as the 360 did (`src/ui/legend.tsx`):
Y over X on the left, B over A on the right, listing only the ones that apply.

## Checks and tests

Run these before committing a change to the page (`dist/` is checked in and
is what the app ships):

| Command | What it does |
|---|---|
| `npm run build` | Type-checks and builds `dist/`. |
| `npm run check` | Fails if `src/mock-cvars.json` no longer matches the emulator's `DEFINE_*` settings (fix: `npm run gen:cvars`), if the page doesn't type-check, if the dashboard's text fails its checks (see Languages), or if `dist/` isn't what the sources build to (fix: `npm run build`). |
| `npm test` | Unit tests (`tests/*.test.mjs`), then smoke tests (`tests/smoke.mjs`): every blade opens, B and Y on the blades, the stacked legend, All Games (open, move, sort, back), Game Folders (add, two-press remove), the Game folders pane and the Dashboard sounds pane. |
| `npm run screens -- record <dir>` then `compare <dir>` | Screenshots of a fixed set of states, byte for byte, for changes meant to look the same (a stylesheet refactor). Differing ones are written beside the originals. |

The tests run the page on the mock in a headless Chromium (Chrome,
Chromium, Helium, Brave or Edge, or `BLADES_BROWSER`) with the clock frozen
and reduced motion on, so screenshots repeat exactly. Node 22 or later.

Setting names are checked at compile time: `CvarName` (`src/native.ts`) is
the keys of `mock-cvars.json`, so a name the emulator doesn't define is a type
error. The state the app sends is checked once at runtime against
`NativeState`; a mismatch is logged as a "Blades page:" line in the app log.
`XE_BLADES_STATE_DUMP=<file>` makes the app write each state it sends to that
file, for comparing two builds or capturing real state.

Styles live in `src/styles/`, all inside Tailwind's cascade layers: `tokens`,
`blade` (the deck, tabs and legend), `content` (rows, panes, controls),
`sound-offer` and `sub-blade`. Only the narrow-window and reduced-motion
overrides in `globals.css` sit outside the layers, on purpose. Tab shapes
are keyed on `data-bend` (`left` or `right`), which deck blades and
sub-blade tabs both set.

## Languages

The dashboard's text goes through `t()` (`src/i18n/`), so it can be
translated into the console's sixteen languages. Only English exists today;
the wiring, the checks and the tests are in place for the rest.

- **The text.** `src/i18n/en.ts` is every message in English, keyed
  `"area.thing"`, and the source for translators. Messages take `{name}`
  placeholders and ICU plurals (`{count, plural, one {# game} other {# games}}`,
  `src/i18n/format.ts`). `MessageKey` types every key, so a key that doesn't
  exist fails the build.
- **A translation** is `src/i18n/catalogs/<code>.ts` (`ja.ts`, `zh-Hant.ts`…,
  the codes in `src/i18n/locales.ts`) exporting the messages it has translated:
  `export default { "legend.back": "…" } satisfies Catalog`. It is picked up
  without other wiring and appears under System > Dashboard language; missing
  keys fall back to English.
- **Which language.** Dashboard language defaults to Same as my Mac: the first
  of the Mac's preferred languages (the app reports them; `NSLocale`) that has
  a translation, else English. Taiwan, Hong Kong and Macau read Traditional
  Chinese, other Chinese Simplified; Norwegian of any kind reads Bokmål.
- **Checks** (`npm run check`, `tools/i18n.mjs`): every English key is used;
  every translation keeps each message's placeholders and uses only English's
  keys; and English written straight into a `.tsx` instead of through `t()`
  never grows. `tools/i18n-baseline.json` holds each file's count of such
  text (Home, the Dashboard rows, the legend, the sub-blades and the sound
  offer are at zero; Graphics, Controller, Audio & CPU, Profiles, the rest of
  System and Advanced are still to convert). Convert a file, then
  `npm run i18n:baseline` to lock the lower count in. `npm run i18n` prints the
  report, with each language's coverage.
- **Tests**: `tests/i18n.test.mjs` (language matching, plurals, placeholders,
  the scanner) and smoke tests that switch to the pseudo-locale (`en-XA`,
  development only: every word accented, so untranslated text stands out) and
  check the converted text follows.

## Sounds are downloaded, not shipped

The dashboard's sounds are Microsoft's, so neither the repository nor the app
carries them. On first launch the page offers to download a fan-made pack
("Xbox 360 Blades" by squabbled, published on DeckThemes); System > Dashboard
sounds offers it again later. The app fetches the zip, checks it against a
pinned SHA-256, and unpacks the six sounds it uses into
`<storage_root>/dashboard_sounds` (`dashboard_sounds_mac.mm`, which also holds
the by-ear mapping from the pack's Steam Deck names). Until then the dashboard
is silent. For `npm run dev`, copy those files into `public/dash-sounds/`,
which is git-ignored.

## How it fits together

| Piece | Where |
|---|---|
| The page (React, Vite, Tailwind) | `src/` here, built into `dist/` |
| Hosting it over the game (WKWebView, URL scheme, controller forwarding) | `../blades_overlay_mac.{h,mm}` |
| The state it shows and the changes it makes | `../blades_model.{h,cc}` |
| Opening, closing, the system-UI pause | `EmulatorWindow::ShowPreferences` and friends |

The app serves `dist/` from its own `xenia-blades://` scheme (copied beside the
binary as `blades/` by the build, and into `Contents/Resources/blades` by
`tools/build/package_mac.sh`), so absolute paths like `/fonts/...` work as on
the web.

The app pushes its whole state as one JSON object (`NativeState` in
`src/native.ts`): every config setting with its type, value and default, the
running and selected renderer, the controllers, their bindings and hotkeys,
the console's own settings (XConfig) and the game speed. The page posts small
messages back (`NativeMessage`): set or reset a setting, change a console
setting, reassign a controller, rebind a button, close. Settings apply and
save as they are chosen.

While Preferences is open over a game, the console's system UI is up: the
game gets the system-UI notification (most pause, as under the guide) and
sees no controller input. Closing lowers it again.

## Blades

`src/content/blades.tsx` orders them; each has its own file. Home is the
dashboard you land on when no game was given at launch (Profiles, if there is
no profile yet): what is playing, recent games, Open a game, Hide menu.
Graphics, Controller, Audio, CPU and Profiles carry the settings most people
change, in player-facing words. System opens with the Dashboard's own look
and sound (kept in the page's local storage), then the console's settings.
Advanced lists every setting in the config file by category, with a search,
a warning, and Reset to defaults at the end.

The dashboard opens with a tap of Guide on a controller (or Back + Start),
F6 (or F7, F8, F10) and ⌘,; the button and key are settings
(`preferences_button`, `preferences_key`, Controller > Dashboard shortcut).

A setting that only one renderer reads is shown only while that renderer runs
or is selected for the next launch (`content/visibility.ts`). Settings that
take effect at the next launch are listed in `NEXT_LAUNCH`
(`src/ui/settings-controls.tsx`); changing one while a game runs offers a
tray that restarts the game.

## Controls

Every control is shaped for the dashboard's sub-menu (`src/ui/blade-menu.tsx`):
A opens a row's settings, ↑ ↓ move between groups, ← → choose, B goes back,
and B at the top closes Preferences.

- Options: a row of sheets, one radiogroup (`Options`, `Toggle`).
- More than eight choices: a drop-down (`src/ui/dropdown.tsx`).
- Values on a scale: a slider that ← → step (`Slider`).
- Text and numbers: a field (`TextField`).
- Lists of switches: checkboxes (`Check`).

A controller drives it too (`src/ui/gamepad.ts`), from the page's Gamepad API
and from the app's own controllers while Preferences is open.

## Building

The built `dist/` is checked in, so building the app needs no Node.

    npm install
    npm run build     # type-checks, then writes dist/

Rebuild and commit `dist/` with any change under `src/`.

`npm run dev` runs the page in a browser against a mock of the app
(`src/mock.ts`, `src/mock-cvars.json`). Regenerate the mock's settings after
adding or changing one:

    python3 tools/gen-mock-cvars.py "$(cd ../.. && pwd)" src/mock-cvars.json

In the app, the page's errors land in the emulator log as `Blades page:`
lines, and Safari's Develop menu can inspect it.
