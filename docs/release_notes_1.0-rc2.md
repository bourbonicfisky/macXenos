# macXenos 1.0 rc2 — release notes

macXenos is an unofficial macOS port of Xenia Canary with a native Metal
rendering backend. This is the second release candidate; it is built from
git tag `macxenos-1.0-rc2` on the `codex/metal-foundation-integration`
trunk.

## What changed since rc1

- **Render-target coherence redesign, default on.** The Metal backend no
  longer drains the GPU every frame to keep guest RAM in step with resolves.
  Resolve and memory-export output stays GPU-resident and is published to
  RAM only when the guest actually reads it (page read watches); the swap
  waits for the frame two swaps back instead of the current one; mid-frame
  buffer rewrites ride a GPU blit instead of a CPU wait; the present reads
  the GPU frontbuffer snapshot instead of decoding RAM. Measured on the
  counted 24-run matrix against the rc1 behaviour on an M4: Just Cause 2
  +49% (10.2 -> 15.2 fps median), Viva Piñata +120% (11.0 -> 24.5 fps),
  zero faults over 12,000-swap soaks.
- **More textures served on the GPU.** Resolve-produced 8/16/32/64-bit
  formats load from shared memory on the GPU instead of the CPU untile
  path, and signed RGBA8 fetches no longer fall back to the CPU. Sonic
  Generations went from 11.5 to 27 fps on the same scene.
- **Shared-memory fix.** A CPU write next to a GPU-written resolve in the
  same 16 KiB host page could re-upload stale RAM over the resolve output.
  Uploads now copy around GPU-resident spans.
- Everything above has an exact `=0` environment rollback
  (`XE_METAL_LAZY_PUBLISH`, `XE_METAL_SWAP_LIFETIME`,
  `XE_METAL_PRESENT_SNAPSHOT`, `XE_METAL_TEX_ROUTE_SIGNED_ALL`,
  `XE_METAL_MEMEXPORT_ORDERING`, `XE_METAL_GPU_RESOLVE_OVERLAP_MEMEXPORT`,
  `XE_METAL_HAZARD_STAGED`) should a title regress.
- Opt-in: `XE_METAL_TEX_ROUTE_ALL=1` routes every eligible texture through
  the GPU load path (Just Cause 2 reaches 23 fps with it). It stays off by
  default because Ultra Street Fighter IV's flash frames render a small
  luminance target incorrectly through it; see Known issues.

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
The rc2 changes were checked live on Just Cause 2, Viva Piñata, Sonic
Generations, Dead or Alive 5 and Gears of War 3, and byte-for-byte on a
13-frame trace corpus covering those plus Ultra Street Fighter IV, LEGO
Star Wars II and Halo 3.

## Known issues

- **Performance.** Open-world titles are still below full speed (Just Cause 2
  about 15 fps and Viva Piñata about 24 fps at 720p on an M4; 30 fps is the
  target). With the per-frame GPU waits gone the remaining cost is the
  emulator's own per-draw CPU work and GPU time, which is the next
  engineering item. Menu and 2D-heavy titles run at full speed.
- **Ultra Street Fighter IV flashes.** The small luminance targets used by
  the Ultra-combo flash render incorrectly on the Metal backend (found
  while validating this release); the visible effect, if any, is limited to
  those frames.
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
