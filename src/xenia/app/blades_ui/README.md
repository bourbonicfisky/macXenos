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
