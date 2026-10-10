# macXenos 1.0 beta 4 — release notes

macXenos is an unofficial macOS port of Xenia Canary with a native Metal
rendering backend. This build is tagged `macxenos-1.0-beta4` on the
`codex/metal-foundation-integration` trunk. It follows beta 3 (2026-10-09);
see `release_notes_1.0-beta3.md` for everything beta 3 added.

## What changed since beta 3

- **Ultra Street Fighter IV holds a locked 60 fps** on an M4 (beta 3: 59.5,
  with a dropped frame every couple of seconds). The game paces itself with
  a busy-wait that overshot on macOS for two reasons, both fixed:
  - the guest clock lost a fraction of a tick on every update, so the
    console's 60 Hz vblank ran at 58.2 Hz;
  - the `Sleep(0)` backoff slept for 250 µs, so the busy-wait
    overshot every frame.

  `macos_spin0_depress = false` and `XE_CLOCK_TRUNCATE_LEGACY=1` restore
  the old behaviour.
- **Gamma for titles that use the console's piecewise-linear gamma curve.**
  - Seven titles in a 55-disc census program this curve instead of the
    usual 256-entry table: Halo 3, Halo 3: ODST, Forza Horizon and Forza
    Horizon 2, Shadowrun, Brothers in Arms: Hell's Highway and Silent Hill:
    Homecoming.
  - Metal showed them with no gamma curve at all. Halo 3, ODST and Silent
    Hill looked washed out; they now match the reference.
  - The other four change only slightly because their curves are close to
    a straight line.
  - `metal_pwl_gamma = false` restores the old output.

## What you get

- `macXenos.app`, a Universal 2 bundle (Apple Silicon + Intel), signed with
  a Developer ID, distributed as `macXenos-1.0-beta4.dmg`. Drag it to
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
