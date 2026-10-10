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
