# macXenos 1.0 rc1 — release notes

macXenos is an unofficial macOS port of Xenia Canary with a native Metal
rendering backend. This is the first release candidate; it is built from git
tag `macxenos-1.0-rc1` on the `codex/metal-foundation-integration` trunk.

## What you get

- `macXenos.app`, a Universal 2 bundle (Apple Silicon + Intel), signed with
  a Developer ID, distributed as `macXenos.dmg`. Drag it to Applications.
- Requires macOS 15 or newer.
- **Apple Silicon** runs the native Metal backend by default. Shaders are
  translated DXBC -> DXIL -> Metal at runtime through the bundled dxilconv
  and Apple Metal Shader Converter libraries; the first run of a title
  compiles pipelines and is slower than later runs.
- **Intel Macs** run the MoltenVK (Vulkan) backend by default. The Metal
  backend is present in the Intel slice but has not been validated on AMD
  or Intel GPUs; treat Intel as best effort.

## First run

- Settings, saves, caches and the log live in `~/.local/share/Xenia`
  (the config file is `xenia-canary.config.toml` there; `XDG_DATA_HOME`
  overrides the base). Put an empty `portable.txt` next to the executable
  inside the bundle to keep everything beside the app instead.
- Open a game with File > Open, or from a terminal:
  `open -n /Applications/macXenos.app --args "/path/to/game.iso"`.
- Controllers: any SDL-recognised gamepad works out of the box (`hid = any`).
- Keyboard: the keyboard driver is included but off. Set `keyboard_mode = 1`
  in the config to make the keyboard act as controller 1. Default bindings
  (Windows-style, all rebindable under `[HID.Key]`):

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
- Backend override: set `gpu = vulkan` (or `gpu = metal`) in the config to
  force a backend. The app title bar shows which one is active.

## Validated titles

Titles driven to gameplay by scripted input on this build's lineage:
Sonic Generations, Ultra Street Fighter IV, Just Cause 2, Viva Piñata, LEGO
Star Wars II, Red Dead Redemption, Halo 3 (campaign), Halo 3: ODST, Dead or
Alive 5, Gears of War 2 and 3 (campaign), Mass Effect, Spider-Man: Web of
Shadows. A 54-disc boot census (2026-09-18 and 2026-09-26) reached a menu
or gameplay on every disc that was not a bad dump or a second/install disc.

## Known issues

- **Performance.** Demanding open-world titles run well below full speed
  (Just Cause 2 around 8 fps, Viva Piñata around 11 fps at 720p on an M4).
  The limit is the render-target coherence model, not the CPU; it is the
  next engineering item after this release. Menu and 2D-heavy titles run at
  full speed.
- **Tessellated geometry.** Hardware tessellation runs on a CPU tessellator.
  A small class of quad-patch draws is still skipped, visible as missing
  fragments in the Viva Piñata start-screen logo.
- **Video playback.** Some full-motion video sequences present through a
  fallback path and can look wrong; gameplay after them is unaffected.
- **Titles that relaunch themselves at boot** (Call of Duty 4, Batman:
  Arkham Origins, Assassin's Creed III) may exit on the first launch and
  come up on the second.
- **Multi-disc titles.** Disc swaps open a file picker; install discs (GTA V,
  Forza 4 content disc) do not run on their own.
- **Memory.** One instance at a time on a 16 GB machine; a session uses
  2 to 3 GB and grows with the shader cache.

## Reporting problems

Attach `~/.local/share/Xenia/xenia.log` (the bundled app logs there; a
bare binary logs next to itself) and say which backend the title bar
showed. For rendering issues, a GPU trace captured with F4 (GPU > Trace
Frame; written under `scratch/gpu/` in the working directory unless
`trace_gpu_prefix` is set) is the most useful artifact.
