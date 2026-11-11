# Moonshots

Ambitious features worth doing one day, recorded with what is known so the
investigation doesn't have to be repeated. Not on any current plan; picking
one up starts with re-checking the facts below against the then-current
source and upstream.

## Custom soundtracks (a music library from the Mac)

Recorded 2026-10-09.

**Goal:** the Xbox 360's custom soundtracks. Point macXenos at a music folder
on the Mac; games that let the player replace their soundtrack (custom
soundtrack support was a platform feature) play the player's music, and the
dashboard gains the console's music library screen: Albums, Artists, Saved
Playlists, Songs, Genres.

**State in every Xenia (checked 2026-10-09: this tree, Canary 7ba916c01,
Edge 9cdf7a322, XeniOS 87b176a07, Xendroid 779680ade):**
- The music player (XMP, `src/xenia/apu/audio_media_player.cc`,
  `src/xenia/kernel/xam/apps/xmp_app.cc`) only plays playlists a game builds
  from its own files. It decodes common formats through FFmpeg.
- `XMPCreateUserPlaylistEnumerator` is answered with an empty list in
  Canary, Edge and XeniOS (it exists so titles waiting on it don't freeze);
  this tree logs and returns success. `XMPGetMediaSources` reports none.
- Nothing reads music from the host. No fork has the feature, so it would
  be a first for the Xenia family and a candidate to offer upstream.

**Rough shape:**
1. A music folder setting, indexed by tags (album, artist, title, genre,
   track) into a library kept in the storage root.
2. Fill the user-playlist enumerator, media sources and song queries from
   that library, and let the player stream the files.
3. The dashboard screen, as a sub-blade off Audio's Music player.

**Risk:** step 2. Each game reads playlists its own way; it needs testing
title by title (games named in xmp_app.cc's comments are a start).

**Related:** the Music player setting (`enable_xmp`) is currently off by
default; with it off, games that play their own soundtrack through the
player have no music. That default should be revisited regardless.

## Original Xbox games (through XeFu)

Recorded 2026-10-09.

**Goal:** play original Xbox games the way a 360 does: through XeFu, the
console's own backward-compatibility emulator (`xbox.xex` plus its per-game
data on the HDD compatibility partition), run as a 360 title. Opening an
`.xbe`, an original Xbox disc image or an Xbox Original package would start
it, and the game library would list those games as themselves.

**State (checked 2026-10-09: this tree a563c3b80, Edge 417b37dfe):**
- Edge has it working on Windows. About 56 commits from 2026-10-01 to
  2026-10-08, starting at 615c357ef (run original Xbox games through XeFu),
  04c298a0d (mount the HDD compatibility partition) and 59427fe60 (read
  `.xbe` metadata). XeFu files are user-supplied: the launcher names the
  missing file and the folder it belongs in.
- Nothing of it is in this tree, and none of it cherry-picks cleanly.
- The core is kernel user mode: XeFu enters user mode via `KeCreateUserMode`
  with its own page table. Edge translates user mode addresses through that
  table (b5cc59e85, f55adea3f, 8358e0d53, 4c6992b39), hands user mode page
  faults to the title's trap handler (Win32 7873b1ed9, POSIX **x64 only**
  241633054), keeps a trap frame (6aa3406e8, 1df3d93fc), and adds
  KeSetTimerEx/DPC/APC fixes (2741af843, 54729c29f, 0e7e29aae, fb1c5bf08,
  7598c1779) and the LDI decompression and Ob exports.
- XeFu compiles original Xbox (x86) code at runtime, so Edge also handles
  guest-written code: icache sweeps drop compiled functions (cebb1d1ab,
  0492b38b2, 2317b5547, 723f54212, edf01c1e6). Much of the dispatch work is
  in the x64 backend (the dynamic call cache in 624459ffc, 60b2e5a10,
  1342e40f1, 4773531a9, and the longjmp handling).

**What a macOS (arm64) port needs beyond picking Edge's commits:**
1. POSIX user-mode fault delivery on arm64: Edge's signal-stack diversion
   thunk and `ResumeHostContext` (bdcc36f3b) exist only for x64.
2. A64 equivalents of the x64 dynamic-code dispatch: the dynamic call cache,
   return-site caching, and the longjmp landing rules.
3. Page granularity. Edge can honor only 64 KB and 16 MB user mode pages,
   because a host view starts at the allocation granularity. On Apple
   Silicon that granularity is 16 KB, so check which segments still work.
4. The Blades library and launcher: listing and launching `.xbe` games and
   the XeFu-missing prompt.

**Risk:** items 1 and 2. They are new CPU and kernel work on the arm64
path, not a port, and the whole series is days old upstream and still
changing (its POSIX side landed on 2026-10-08). Revisit once Edge's POSIX
support has settled.

## Frame generation (MetalFX frame interpolation, Metal 4)

Recorded 2026-10-09.

**Goal:** show games that are capped at 30 fps at 60 (or 60 at 120 on
ProMotion displays) by having MetalFX generate a frame between each pair the
game renders. This targets macOS 26 (Tahoe), where Metal 4's MetalFX frame
interpolation first shipped, as an opt-in setting on the Graphics blade, per
title. Older macOS versions keep presenting guest frames as today.

**State (checked 2026-10-09: this tree 1ba0158be, macOS 26.5 SDK, Edge
417b37dfe, Canary 6e2caa5c6, XeniOS 87b176a07, Xendroid 779680ade):**
- The API already exists in the macOS 26 SDK: `MTLFXFrameInterpolator` and
  its Metal 4 twin `MTL4FXFrameInterpolator` (`MetalFX/MTLFXFrameInterpolator.h`,
  `API_AVAILABLE(macos(26.0))`, `+supportsDevice:` / `+supportsMetal4FX:`;
  Metal 4 needs M1 or later).
  - Inputs: the current and previous colour, depth, per-pixel motion
    vectors, the time between frames, the camera's near plane, far plane,
    field of view and aspect ratio, and the jitter offset.
  - Also an optional UI texture, so the HUD is overlaid on the generated
    frame instead of being interpolated.
- macOS 26 is enough: the deployment check is `@available(macOS 26.0, *)`
  plus `+supportsDevice:`. Anything macOS 27 adds on top is unverified
  (2026-10-09 search results covered only the WWDC 2025 release) and is a
  bonus, not a requirement.
- This tree already uses MetalFX: the spatial scaler is a presenter output
  option (`metal_presenter.mm`, `postprocess_scaling_and_sharpening =
  metalfx`). Nothing interpolates frames.
- No Xenia fork has frame generation, so this would be a first.

**Why it is hard for an emulator:** the interpolator wants what a game
engine knows and the Xbox 360 never hands us.
1. **Motion vectors.** 360 games don't produce them. Options:
   - optical flow between consecutive frontbuffers (a GPU block-matching
     pass, or Vision's optical-flow request if fast enough);
   - reprojection from depth plus camera matrices read from vertex-shader
     constants (title-specific);
   - flow at reduced resolution, refined by depth edges.
2. **Depth and camera.** The scene depth is one of several EDRAM depth
   surfaces (some titles resolve depth several times a frame, in predicated
   tiles), and near/far/FOV are unknown. These need a per-title
   configuration or a heuristic (the largest depth surface before the final
   composite).
3. **UI.** The HUD is drawn into the same frame, and interpolating it
   smears text and meters. Separating it needs a per-title cut point (for
   example, draws after the last 3D pass).
4. **Pacing.** Presentation follows the guest's vblank swaps. A generated
   frame has to be presented halfway between two guest frames, so
   presentation must come off the guest swap (a present thread or
   `CAMetalDisplayLink`). That adds about one guest frame of input latency.

**Rough shape:**
1. **Gate 0, offline with a ground truth.** Take titles that already hold
   60 fps (Ultra Street Fighter IV, Dead or Alive 4, Call of Duty 2). Drop
   every other frame, interpolate it back from its neighbours, and compare
   against the real frame. Run this once per motion-vector option above.
   If flow-based vectors don't beat a plain blend clearly, stop here.
2. Present-path decoupling and a pacing census (frame-time variance, added
   latency).
3. A per-title opt-in config (depth surface, camera hints, UI cut) and a
   Graphics blade switch, off by default.

**Risk:** items 1 to 3 of the hard part. A generic solution may never match
engine-integrated frame generation, so it should ship per title where it
looks right. GPU headroom also matters: titles at their 30 cap on an M4
have room, while Red Dead's ~31.5 ms of GPU time per frame does not.

## Exact EDRAM output merging (ROV-style, in the pixel shader)

Recorded 2026-10-12.

**Goal:** draw the way the Xenos EDRAM does, bit for bit. Instead of host
render targets with fixed-function blending and depth/stencil, the
translated pixel shader reads and writes the 10 MiB EDRAM directly, in
primitive order, and does the depth/stencil test, blend, guest format
packing and per-sample MSAA coverage itself. Every guest render-target
format then works exactly, including those with no host equivalent:
7e3 float colour, the −32…32 `16_16` range and float24 depth. Changing a
surface's format or MSAA mode becomes free, as it is on the console.
It would be an opt-in accuracy path beside host render targets, not a
replacement for them.

**State (checked 2026-10-12: this tree b231a03db, Canary 2372a5410,
vendored Metal Shader Converter 3.1.1):**
- Canary D3D12 has this as its ROV path (`render_target_path_d3d12 = rov`,
  output merger in `dxbc_shader_translator_om.cc`); Vulkan has the same idea
  over fragment shader interlock (`render_target_path_vulkan = fsi`,
  `spirv_shader_translator_rb.cc`). Canary's own help text calls it the
  highest-accuracy path, "performance limited primarily by overdraw", and
  its automatic choice is the host render-target path everywhere except
  pre-Arc Intel GPUs (a stencil bug there breaks Direct3D 9 clears).
- This tree's shared DXBC translator still carries the ROV output-merger
  code (`dxbc_shader_translator_om.cc`, 3,749 lines against Canary's 4,158,
  so it is behind upstream), but the Metal back end never enables it and
  nothing in `src/xenia/gpu/metal/` uses raster order groups. Metal always
  draws to host render targets backed by the canonical EDRAM buffer.
- Measured negative: Vulkan FSI through MoltenVK on an M4 (2026-07) was
  slower than host render targets (USF4 fights ~33–40 vs ~44 fps) and showed
  artifacts. That was interlock over a device-memory EDRAM buffer with no
  tile-memory benefit; it does not test native raster order groups or tile
  memory.
- Related study: the 2026-10-09 tile-memory RT-cache study rejected a full
  tile-memory redesign. Metal exposes no depth to tile functions or
  framebuffer fetch, format and MSAA aliasing across passes still needs
  memory, and an Apple-only path cannot serve AMD/Intel Macs (archived change
  `openspec/changes/archive/2026-10-09-design-metal-tile-memory-edram/`).
  It also found a new premise: Metal Shader Converter 3.1.1 has
  `IRCompilerSetFramebufferFetchResourceSpace`, which maps a DXIL texture to
  a framebuffer-fetch input.
- No Xenia fork has an exact output merger on Metal.

**Why it matters:**
1. **Accuracy.** Every remaining host-format approximation (blend precision
   on 7e3 and 16-bit targets, float24 depth compares, reinterpretation
   between formats) disappears by construction. Red Dead Redemption's
   float24 speckle and the rejected global float24 pixel-shader mode
   (it fixed RDR and darkened JC2) are examples of what the host path has
   to handle case by case.
2. **The EDRAM machinery.** Ownership transfers, packs and materializes
   exist because host textures cannot alias. At 2×2 scale they cost JC2
   ~43 ms of a 93 ms GPU frame. An exact output merger would not need
   most of them, though it pays per fragment instead.

**Rough shape:**
1. **Gate 0, mechanism.** Find out whether the Metal Shader Converter
   lowers DXIL rasterizer-ordered resources (`RasterizerOrderedByteAddress
   Buffer`) to Metal raster order groups, and on which GPUs (Apple Silicon;
   AMD/Intel Macs report raster-order-group support separately). If it does
   not, decide between patching the DXIL, an MSL output-merger epilogue, or
   framebuffer fetch.
2. **Gate 1, cost, before writing the path.** A device microbenchmark of
   ordered EDRAM read-modify-write against host blending at real overdraw
   (USF4 fight, RDR, JC2 depth prepass), 1× and 4× MSAA. Stop if it cannot
   come within reach of the host path on an M-series GPU.
3. Enable the shared translator's ROV output merger for Metal behind a
   setting, after bringing `dxbc_shader_translator_om.cc` up to Canary,
   with resolves reading the EDRAM buffer directly.
4. Per-title opt-in where the host path is visibly wrong; the default
   stays host render targets.

**Risk:** cost. ROV is the slow path even on desktop D3D12, and Apple's
tile-based GPUs pay for ordered device-memory writes in a way the FSI
experiment already showed. The accuracy gain is in edge cases, so it
has to be cheap enough to turn on for the titles that need it.

## Dialogue enhance (centre-channel boost)

Recorded 2026-10-14.

**Goal:** an Audio setting that makes speech easier to hear over music and
effects, as many TVs and soundbars offer. Most 5.1 game mixes put dialogue
in the centre channel, so raising that channel's volume, with a simple
presence EQ (a gentle boost around 1 to 4 kHz), lifts voices without
touching the rest of the mix. Off by default, with a small strength choice
(for example Low / Medium / High).

**State (checked 2026-10-14: this tree 429f1c1c4):**
- Games hand the audio system 5.1 frames: 6 channels in the order front
  left, front right, centre, LFE, back left, back right, 256 samples per
  channel at 48 kHz (`src/xenia/apu/audio_driver.h`, `conversion.h`).
- On macOS the SDL driver converts each frame in its callback
  (`src/xenia/apu/sdl/sdl_audio_driver.cc`). For a stereo device,
  `sequential_6_BE_to_interleaved_2_LE` folds half the centre into left and
  right and drops the LFE; for a 6-channel device the channels pass through.
  The centre channel is still separate at that point, so this is where a
  gain and EQ would go.
- The `volume_` path in the same callback only runs when no conversion is
  needed, so it doesn't apply to the usual 6-to-2 case either. Worth fixing
  while there.
- No Xenia fork has this.

**Rough shape:**
1. A centre gain (for example +3, +6, +9 dB) and a fixed presence biquad,
   applied to the centre samples before the downmix or interleave. Clamp or
   soft-limit the output, since boosting can clip.
2. An Audio blade setting, global with a per-title override for games whose
   mix needs more or less.
3. Check which games really put speech in the centre: compare the channel
   levels during a dialogue scene in a few routed titles (Mass Effect's
   first conversation, the Red Dead opening cutscene).

**Risk:** games that mix in stereo only, or put speech in the front pair.
There the centre channel is empty and the setting does nothing. A
mid-channel fallback (boost what is common to left and right) could cover
them, but it also lifts centred music and effects. Also check what speaker
setup the emulated console reports (`XAudioGetSpeakerConfig` reads the
user audio flags from xconfig), because games choose stereo or 5.1 output
from it.

## Disc image trimming (keep only the data a game uses)

Recorded 2026-11-11.

**Goal:** shrink a library of full Xbox 360 disc images by repacking each
one so it holds only what the game reads: the game partition's file system
and files, without the video partition, the unused gaps and the padding to
the end of the disc. A "Trim" action in the game library (or a command-line
tool) writes a trimmed copy beside the original and, once the copy is
verified, offers to replace it.

**State (checked 2026-11-11: this tree 77ac18894):**
- The test library holds 257 full images at 2.08 TB, a median of 7.8 GB
  each (`/Volumes/NAS/Emulation/Xbox 360 Decompressed`). Many games use
  well under half of that.
- The disc reader (`src/xenia/vfs/devices/disc_image_device.cc`) looks for
  the game partition's `MICROSOFT*XBOX*MEDIA` header at a few known offsets,
  including 0, so an image that holds only the game partition already
  mounts. It reads files through the directory entries' sector numbers and
  never touches the video partition.
- Nothing in this tree writes images. Community tools exist (extract-xiso's
  rewrite mode, the Xbox 360 "trim" options in ISO managers), so the format
  of a trimmed image is settled; the work is a safe, built-in way to do it.

**Rough shape:**
1. Walk the game partition's directory tree, as the reader does, and list
   every file's sectors plus the file system's own sectors.
2. Simplest: copy the game partition and cut it after the last used
   sector. Further: rebuild it, packing files together and rewriting the
   directory entries' sector numbers, which also drops gaps between files.
3. Verify by mounting the copy and comparing every file's bytes with the
   original before anything is replaced.
4. The library action, with the saving shown per game and for the whole
   library.

**Risk:** a game that reads raw sectors outside its files (copy protection
checks, data hidden past the file system) would break, so test the routed
titles on trimmed copies before offering this to users. A trimmed image
also no longer matches the Redump checksums people use to verify dumps;
keep trimming opt-in and say so where the action is offered.
