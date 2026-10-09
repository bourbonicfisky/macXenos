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
