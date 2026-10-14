# macXenos 1.0 beta 2 — release notes

macXenos is an unofficial macOS port of Xenia Canary with a native Metal
rendering backend. This build is tagged `macxenos-1.0-beta2` on the
`codex/metal-foundation-integration` trunk. It follows the two release
candidates of September (rc1, rc2).

## What changed since rc2

- **Full speed on the open-world titles at 720p (Apple Silicon).** The last
  per-frame GPU waits and the duplicated render-target copies are gone, and
  every eligible texture is now served on the GPU by default. On an M4:
  Just Cause 2 15 -> 30 fps (the title's cap), Viva Piñata 25 -> 30 fps,
  Sonic Generations 27 -> 28 fps, Spider-Man: Web of Shadows 28.6 -> 30 fps
  with its 60 Hz vblank restored. Zero GPU faults across the soaks.
- **Internal resolution scaling, 1x to 3x** (Settings > Display > internal
  resolution; `draw_resolution_scale_x/y` in the config). Renders the guest
  frame at 2x2 or 3x3 and samples it back exactly; textures are left at
  their native size. 2x2 on an M4: Just Cause 2 12.5 fps, Viva Piñata 15,
  Sonic Generations 25, Gears of War 3 15. Checked byte-for-byte on the
  trace corpus at 1x and 2x2.
- **Output upscaling.** The presenter offers bilinear, AMD CAS, AMD FSR 1
  (half precision by default) and MetalFX spatial upscaling for the window
  or display resolution (`postprocess_scaling_and_sharpening`,
  `metal_fsr_half_precision`, `metal_metalfx_spatial`). On a 5K display
  FSR costs about 3.6 ms per frame and MetalFX about 2.9 ms on an M4.
- **Rendering fixes.** Batman: Arkham Asylum's character shadows no longer
  darken whole floor regions (the per-caster shadow maps are now served at
  the size the game samples them). Cube maps with mip chains upload every
  level (Batman, Skate 2, Halo 3 reflections). At 2x2 the Metal backend no
  longer stalls the GPU on resolve reads (+15% on Just Cause 2). Ultra
  Street Fighter IV's flash frames, listed as a known issue in rc2, were a
  replay artifact and render correctly.
- **Configurable performance switches.** The Metal performance paths that
  used to be environment variables are config options with their defaults
  documented in the generated config; every rendering change above keeps an
  exact `=0` environment rollback (`XE_METAL_SNAPSHOT_FIT`,
  `XE_METAL_CUBE_MIPS`, `XE_METAL_SCALED_COVER_ORDERING`,
  `XE_METAL_TEX_ROUTE_ALL`, `XE_VBLANK_FLAT_SLEEP=1`, and the rc2 set).
- **Upstream Xenia Canary ports.** Texture layout and resolve fixes, audio
  lock, keyboard, POSIX thread and kernel fixes, raw-index vertex fetch
  bounds, plus a macOS-only fix for self-queued I/O completions that left
  Spider-Man: Web of Shadows at a black screen.

## What you get

- `macXenos.app`, a Universal 2 bundle (Apple Silicon + Intel), signed with
  a Developer ID, distributed as `macXenos.dmg`. Drag it to Applications.
  The bundle is not notarized; on first launch use right-click > Open.
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
Alive 5, Gears of War 2 and 3 (campaign), Mass Effect, Skate 2, Batman:
Arkham Asylum (played interactively through the opening escort), Spider-Man:
Web of Shadows. A 54-disc boot census (2026-09-18 and 2026-09-26) reached a
menu or gameplay on every disc that was not a bad dump or a second/install
disc, and an 18-title texture census on 2026-10-03 found no unsupported
texture format or dimension in use and zero GPU faults.

## Known issues

- **Performance at higher internal resolutions.** 2x2 runs at roughly half
  of 1x on an M4 (see above); 3x3 is playable only on lighter titles. The
  draw-pass tile traffic and the game's own MSAA passes are the remaining
  cost.
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
  2 to 3.5 GB and grows with the shader cache.
- **Window lifecycle.** Resizing, native full screen and minimizing work;
  moving the window between displays of different pixel density has not
  been exercised.

## Reporting problems

Attach `~/.local/share/Xenia/xenia.log` (the bundled app logs there; a
bare binary logs next to itself) and say which backend the title bar
showed. For rendering issues, a GPU trace captured with F4 (GPU > Trace
Frame; written under `scratch/gpu/` in the working directory unless
`trace_gpu_prefix` is set) is the most useful artifact.
