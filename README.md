<p align="center">
    <img height="256px" src="assets/mac-icon/macxenos_icon_1024.png" alt="macXenos icon" />
</p>

<h1 align="center">macXenos</h1>

<p align="center">An Xbox 360 emulator for the Mac.</p>

<p align="center"><a href="https://macxenos.com/macxenos-sizzle-reel-h264.mp4">Watch the sizzle reel</a></p>

macXenos is an unofficial macOS fork and port of
[Xenia Canary](https://github.com/xenia-canary/xenia-canary), with a few
tweaks from [Xenia Edge](https://github.com/has207/xenia-edge). Most of the
emulator is Canary. The work here is about making it run well on a Mac.

The foundational research and discovery belong to the Xenia, Xenia Canary
and Xenia Edge contributors. This project covers only the Mac-specific work.

Website and downloads: [macxenos.com](https://macxenos.com)

## About the name

Out of respect for the Xenia projects, the name is related to Xenia. Xenia
is the Greek idea of hospitality to strangers, likely a play on *xenos*,
which means "stranger" or "foreigner". Xenos is also the codename of the
GPU inside the Xbox 360. This port is mostly about the graphics side, hence
macXenos.

## What's different on the Mac

- **A native Metal renderer.** macXenos draws the game directly with Metal,
  Apple's graphics API, ported from Xenia Canary and Edge with a focus on
  stability, compatibility, performance and power use.
- **Upscaling.** AMD FSR, AMD CAS and Apple MetalFX can upscale and sharpen
  the picture to your window or display.
- **Internal resolution scaling.** Games can render at a higher resolution
  than the Xbox 360 did.
- **Frame generation through Metal 4** is planned.

The older Vulkan renderer (through MoltenVK) is kept for developers to compare
against; use Metal.

### Only in macXenos

- **A Blades dashboard.** Every setting lives in a dashboard modeled on the
  Xbox 360's original "Blades" dashboard. Open it over a game with the
  controller's Guide button, F6 or Cmd-, and the game pauses until you close
  it.
- **A game library.** Point macXenos at the folders that hold your games and
  it lists them all with their icons, plus the ones you played recently.
- **Profiles with gamer pictures.** Player profiles with gamercards, a set of
  macXenos gamer pictures, and the original Xbox 360 pictures if you choose
  to download them.
- **Controller setup on screen.** Choose which player each controller is,
  remap buttons on a picture of the pad, and set vibration and stick dead
  zones. Most controllers macOS recognizes work without setup.
- **Performance cores.** On Apple silicon, the game's threads are kept on the
  performance cores instead of being parked on the efficiency cores.
- **A Mac app.** A signed, notarized universal app with automatic updates.
  Saves go to `~/Documents`, F12 screenshots to `~/Pictures`, and everything
  else to Application Support.

## Requirements

- macOS 15 Sequoia or later
- A Mac with Apple silicon, or an Intel Mac with a dedicated AMD GPU with at
  least 4 GB of video memory (Radeon RX 580 class or faster). Intel Macs are
  supported on a best-effort basis, and Intel integrated graphics aren't
  supported.
- About 2 to 3 GB of free memory while a game runs
- A game controller
- Xbox 360 games

## Getting started

1. Download the latest build from [macxenos.com](https://macxenos.com) and
   drag macXenos to your Applications folder.
2. Open it and point it at the folder that holds your games, or use
   File > Open to start one directly.
3. Plug in or pair a controller. Most controllers that macOS recognizes work
   without setup.

The first time you play a game, it will stutter for a while as it builds its
shaders. Later runs are smoother.

Saves are kept in `~/Documents/macXenos/Saves`. Settings, profiles,
downloads, game installs and the log are kept in
`~/Library/Application Support/macXenos`.

The [user guide](docs/user_guide.md) covers the dashboard, controllers,
graphics settings and more.

## Game compatibility

Many games run well, and some do not run at all yet. The
[compatibility list](docs/compatibility.md) shows the games macXenos is
tested with regularly, how far each test goes and the frame rate on an M4
Mac mini. The community
[compatibility list](https://xenia-manager.github.io/compatibility) tracks
Xenia as a whole; results on the Mac can differ.

## Reporting problems

See [Troubleshooting](docs/troubleshooting.md) for common problems and what
to include in a report. Please report problems with macXenos in
[this project's issues](https://github.com/bourbonicfisky/macXenos/issues),
not to the Xenia Canary or Xenia Edge projects.

## Building from source

See [Building macXenos](docs/building.md). macXenos builds on macOS only.
More for developers is in the
[developer documentation](docs/dev/README.md).

## Credits

- [Xenia](https://github.com/xenia-project/xenia) and
  [Xenia Canary](https://github.com/xenia-canary/xenia-canary), which this
  project is built on
- [Xenia Edge](https://github.com/has207/xenia-edge), for several fixes
  and ideas
- The macXenos icon and the Blades dashboard interface (its design, artwork
  and code) are original work by Greg Gant
- Third-party libraries and their licenses are listed in the app under
  System > About > Acknowledgements; the license texts macXenos ships are in
  [docs/acknowledgements](docs/acknowledgements/)

## License

macXenos is released under the same BSD license as Xenia. See
[LICENSE](LICENSE).

The macXenos icon and the Blades dashboard interface are original work by
Greg Gant. If you reuse either one, or anything based on it, credit Greg
Gant as its creator.

## Disclaimer

macXenos is not affiliated with the Xenia projects, Microsoft or Apple.

This project is for research, experimentation and learning about emulation.
**It is not for enabling illegal activity.** Use your own games. All
information comes from reverse engineering legally purchased hardware and
games, and from material that is publicly available.
