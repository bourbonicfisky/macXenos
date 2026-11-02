# Built-in game patches

Game-code patches that ship with macXenos, for bugs in a title's own code that
the emulator cannot reasonably fix (for example a race the console's timing
almost never exposes). Each file uses the standard Xenia patch format
(`<title id> - <title name>.patch.toml`, keyed to the executable's module
hash, one or more `[[patch]]` entries with `is_enabled`).

How they load (`src/xenia/patcher/patch_db.cc`):

- The build copies this folder beside the binary as `game_patches/`; the app
  bundle carries it as `Contents/Resources/game_patches/`.
- They load before the storage root's `patches/` folder. A user file with the
  same file name replaces the built-in one, so a user can turn a shipped patch
  off (`is_enabled = false`) or change it.
- `--apply_patches=false` disables all patches.

Rules for adding one:

- Prefer an emulator fix or a per-title config (`docs/game_configs`) when the
  cause is the emulator. A patch here is for a defect in the game's code.
- Start the file with a comment: what fails, the guest addresses involved, why
  the patch is correct, how often it failed before and after, and where the
  evidence is.
- Enable it by default only after cold-boot runs show the failure gone.

| Title | Patch | Why |
|---|---|---|
| NBA Ballers: Chosen One (4D5707DA) | Fix lost I/O wakeup | Boot hang on LOADING (~30% of boots): the I/O worker clears its ready flag without rechecking the queue. 2026-11-01. |
