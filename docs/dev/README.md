# Developer documentation

For building from source, see [Building macXenos](../building.md). For the
app itself, see the [user guide](../user_guide.md).

Background reading: [AMD vs Apple and the challenges of Xbox emulation](metal-explainer.md)
explains why the same shader can pick different texture detail levels on
Apple and AMD GPUs.

## App interface

The macXenos front end (start screen, settings, profiles and the in-game
dashboard) is the Blades page in
[`src/xenia/app/blades_ui`](../../src/xenia/app/blades_ui/README.md), hosted
in a WKWebView over the game. The ImGui settings dialogs it replaced are
gone; add new user-facing settings there, not as dialogs.

## Moonshots

Ambitious features recorded for one day, with what is already known:
[moonshots.md](moonshots.md) (custom soundtracks, original Xbox games through
XeFu, frame generation through MetalFX, exact EDRAM output merging, dialogue
enhance).

## Release notes

Published release notes are in [`../releases/`](../releases/).
## Inherited Xenia documentation

[`xenia/`](xenia/) holds `cpu.md`, `cpu_todo.md`, `gpu.md`, `kernel.md`,
`instruction_tracing.md`, `style_guide.md` and the PowerPC manuals in
`ppc/`. They come from the Xenia lineage and are useful background, but they
may describe Canary, Windows, Linux, Vulkan, or older source layouts rather
than this macOS Metal fork.

## Build inputs kept outside this directory

`docs/game_configs/` (built-in per-title configs copied into the app) and
`docs/acknowledgements/` (third-party notices shown in the app) are read by
the build. Don't move them without updating the build.

