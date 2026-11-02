# macXenos documentation

## Using macXenos

- [User guide](user_guide.md): installing, adding games, the dashboard,
  controllers, graphics settings, and where your files are kept.
- [Game compatibility](compatibility.md): the games tested regularly, how far
  each test goes, and how fast they run.
- [Troubleshooting](troubleshooting.md): games that won't start, graphics
  problems, slow games, and how to report a problem.
- [Release notes](releases/): what changed in each build.

## Working on macXenos

- [Building macXenos](building.md): building, running and packaging from
  source on a Mac.
- [Developer documentation](dev/README.md): the Metal renderer, project
  records, and documentation inherited from Xenia.

## Layout

| Path | What's there |
| --- | --- |
| `user_guide.md`, `compatibility.md`, `troubleshooting.md` | Documentation for players |
| `building.md` | Building from source |
| `releases/` | Release notes, one file per build |
| `dev/` | Engineering records and inherited Xenia documentation |
| `game_configs/` | Built-in per-game settings, copied into the app by the build |
| `game_patches/` | Built-in game patches (fixes for bugs in a game's own code), copied into the app by the build; see its README |
| `acknowledgements/` | Third-party license notices shown in the app |
| `AGENTS.md` | Rules for coding agents reading these docs |
