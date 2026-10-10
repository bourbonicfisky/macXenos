# macXenos 1.0 beta 3 — release notes

macXenos is an unofficial macOS port of Xenia Canary with a native Metal
rendering backend. This build is tagged `macxenos-1.0-beta3` on the
`codex/metal-foundation-integration` trunk. It follows beta 2 (2026-10-03).

## What changed since beta 2

- **A new front end: the Blades dashboard.** The settings dialogs are gone.
  Settings now live in an Xbox 360-style dashboard that opens over the
  game (a tap of the controller's Guide button, F6, or Cmd-comma). Its blades:
  - Home: recent games with their art, plus an All Games library built
    from your game folders;
  - Graphics, Controller, Audio & CPU, Profiles, System and Advanced.

  Profiles show gamercards and a gamer-picture picker: 175 macXenos
  pictures are built in, and the original Xbox 360 collection can be
  downloaded on request. Controller buttons are remapped on a drawing of
  the pad. The dashboard sounds are an optional download the first time
  you open it; nothing of Microsoft's ships in the app.
- **Faster on Apple Silicon** (M4, 720p, unless a title's cap is noted):
  - Red Dead Redemption gameplay 10 -> 30 fps (the cap), and the opening
    cinematic 5 -> 30.
  - Ultra Street Fighter IV fights 45 -> 60 fps (the cap).
  - LEGO Star Wars II 44 -> 60 fps (the cap).
  - Dead to Rights: Retribution 20 -> 30 fps.
  - Just Cause 2 at 2x2 internal resolution 12.5 -> 18.4 fps.
  - Just Cause 2, Viva Piñata, Sonic Generations, Gears of War 2 and 3,
    Halo 3, Skate 2, Dead or Alive 4 and Call of Duty 2 hold their caps.

  Most of the gain came from moving less data between the Xbox 360's
  render memory and the Mac's GPU, and from a much lighter command
  processor. Every change keeps an exact environment-variable rollback.
- **Rendering fixes.**
  - MSAA now uses the Xbox 360's own sample positions. Gears of War's cell
    scene regains a lighting pass that the standard positions lost.
  - Child of Eden's videos play instead of rendering flat green.
- **Titles that relaunch themselves at boot.** Assassin's Creed III and
  Batman: Arkham Origins now start in a fresh process instead of asking
  you to restart them.
- **Metal is the default on every Mac.** Intel Macs now run the native
  Metal backend by default too (see Known issues). MoltenVK stays
  available with `gpu = vulkan`.
- **Thread suspension works on macOS.** It was a silent no-op before.
  Titles and the debugger that suspend threads now behave correctly.
- **Upstream Xenia Canary ports.** XAM, kernel, PowerPC and GPU fixes from
  Canary since beta 2: resolve copy formats, primitive-conversion cache
  invalidation, FPU instructions, RSA and RC4, and file-path handling.

## What you get

- `macXenos.app`, a Universal 2 bundle (Apple Silicon + Intel), signed with
  a Developer ID, distributed as `macXenos-1.0-beta3.dmg`. Drag it to
  Applications. The bundle is not notarized; on first launch use
  right-click > Open.
- Requires macOS 15 or newer.
- Shaders are translated DXBC -> DXIL -> Metal at runtime through the
  bundled dxilconv and Apple Metal Shader Converter libraries. The first
  run of a title compiles pipelines and is slower than later runs.

## First run

- Settings, saves, caches and the log live in `~/.local/share/Xenia`
  (`XDG_DATA_HOME` overrides the base). Put an empty `portable.txt` next to
  the executable inside the bundle to keep everything beside the app.
- Open a game from the dashboard's Home or All Games, with File > Open, or
  from a terminal: `open -n /Applications/macXenos.app --args "/path/to/game.iso"`.
- Controllers: any SDL-recognised gamepad works out of the box. A Guide tap
  opens the dashboard (see Known issues for macOS 26).
- Keyboard: the keyboard driver is included but off. Set
  `keyboard_mode = 1` in the config to make the keyboard act as controller
  1. Default bindings (all rebindable under `[HID.Key]`):

  | Pad | Keys |
  | --- | --- |
  | Left stick | W A S D |
  | D-pad | Shift + W A S D |
  | Right stick | Arrow keys |
  | A / B / X / Y | `;` / `'` / L / P |
  | Start / Back | X / Z |
  | LB / RB | 1 / 3 |
  | LT / RT | Q or I / E or O |
  | Stick clicks | F (left) / K (right) |

  The game window must be focused for keyboard input to register.

## Validated titles

Driven to gameplay by scripted input on this build: Sonic Generations,
Ultra Street Fighter IV, Just Cause 2 (1x and 2x2), Viva Piñata, LEGO Star
Wars II, Red Dead Redemption (opening cinematic and gameplay), Halo 3
(campaign), Gears of War 2 and 3 (campaign), Dead or Alive 4, Skate 2, Dead
to Rights: Retribution and Call of Duty 2. Every one ran at its frame cap
with zero GPU faults in the final sweep. Earlier boot censuses (54 discs)
reached a menu or gameplay on every disc that was not a bad dump or a
second/install disc.

## Known issues

- **Guide/Home button on macOS 26.** The system Game Overlay takes the
  controller's Home button before the app sees it. Set
  `preferences_button = back_start` to open the dashboard with Back + Start,
  or use F6 / Cmd-comma.
- **Intel Macs with AMD GPUs.** Metal on Intel/AMD is new and lightly
  tested. Red Dead Redemption shows corrupted title-screen geometry and GPU
  timeouts in gameplay on a Mac Pro (AMD). Set `gpu = vulkan` for MoltenVK
  if a title misbehaves.
- **Higher internal resolutions.** 2x2 runs well below 1x on an M4 (Just
  Cause 2 18.4 fps); 3x3 is playable only on lighter titles.
- **Tessellated geometry.** Hardware tessellation runs on a CPU tessellator.
  A small class of quad-patch draws is still skipped (missing fragments in
  the Viva Piñata start-screen logo).
- **Ultra Street Fighter IV** occasionally stalls during the boot intro.
  Restarting the game clears it.
- **Multi-disc titles.** Disc swaps open a file picker; install discs (GTA V,
  Forza 4 content disc) do not run on their own.
- **Memory.** One instance at a time on a 16 GB machine; a session uses
  2 to 3.5 GB and grows with the shader cache.

## Reporting problems

Attach `~/.local/share/Xenia/xenia.log` and say which backend the title bar
showed. For rendering issues, a GPU trace captured with F4 (GPU > Trace
Frame; written under `scratch/gpu/` in the working directory unless
`trace_gpu_prefix` is set) is the most useful artifact.
