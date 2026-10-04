# Per-title configs

Files named `<title id>.config.toml` in this folder are built-in defaults:
the build copies them next to the binary (`game_configs/`) and the bundle
carries them in `Contents/Resources/game_configs/`, and `LoadGameConfig`
applies the matching file when a title launches. A user's own
`<storage root>/config/<title id>.config.toml` is loaded afterwards and
overrides them.

`samples/` holds files that are NOT applied automatically: historical
workarounds kept for reference (Halo 3's `mount_cache = false`, superseded
by the real cache-partition fix on 2026-09-17).
