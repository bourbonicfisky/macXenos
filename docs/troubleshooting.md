# Troubleshooting

Common problems and what to try. For setup and settings, see the
[user guide](user_guide.md).

## macOS won't open the app

macOS may say it can't check macXenos for malicious software. Open
**System Settings > Privacy & Security**, scroll down, and click
**Open Anyway** next to the message about macXenos.

## A game won't start

- **Check the dump.** A damaged or incomplete disc image is the most common
  reason a game won't boot. If you can, make a fresh dump.
- **Install and content discs don't run by themselves.** Some games ship
  a disc that only holds data to install (for example GTA V's second disc or
  Forza 4's content disc). Start the game from its play disc.
- **Multi-disc games.** When a game asks for the next disc, macXenos opens a
  file picker. Choose the image for that disc.
- **Some games restart themselves while booting.** macXenos starts these
  again in a fresh window. If one closes instead, start it a second time.

## The Guide button opens the macOS Games overlay

On macOS 26, the system's Games overlay can take the controller's Guide
(Home) button. Open the dashboard with **F6** or **Cmd-,** instead, then go
to **Controller > Dashboard shortcut** and choose **Back + Start**.

## A game looks wrong

First put back anything you changed, then try the switches made for this.
They're all on the **Graphics** blade. Some only take effect the next time the
game starts; the pane says so when that's the case.

1. **Internal resolution** back to **1x**. Some games draw shadows,
   reflections or effects that only line up at the console's own resolution.
2. **Upscaling** back to **Bilinear**. This rules out a sharpening filter
   (CAS, FSR or MetalFX) as the cause of halos or shimmering edges.
3. **Unset shader inputs**: *Auto*, *On* or *Off*. Some games draw with a
   value they never set, and the Xbox 360 simply used whatever was left there,
   so no one choice suits every game. *Auto* covers the games known to need
   it. If shadows flicker or lighting looks off, try *On* and *Off*.
4. **Troubleshooting**, under *Metal renderer*:
   - **Memory exports from drawing shaders**: *Auto*, *On* or *Off*. A few
     games write data from shaders that also draw; without it Child of Eden's
     videos stay a flat green. *Auto* turns it on for the games known to need
     it. If something is missing or stuck on one colour, try *On*. It costs
     some speed, so go back to *Auto* if it doesn't help.
   - **Wait for the GPU before signalling the game** removes brief
     whole-screen colour flashes, as seen in Rise of the Tomb Raider. It costs
     frame rate on M1 and M2 Macs, so leave it off unless a game flashes.
   - **The six switches above those** are speed shortcuts in the Metal
     renderer, all on by default. If a glitch appeared after an update, turn
     them off one at a time and restart the game after each. If one makes the
     glitch go away, please report which, then turn it back on. **Reset**
     puts every switch back to its default.

Check the [compatibility list](compatibility.md) for problems already known in
the games we test. If none of this helps, please
[report it](#reporting-a-problem) with a screenshot and say which switches you
tried.

## A game runs slowly

- The first session with a new game stutters while shaders are built. Give
  it a few minutes, or play a second session.
- **Internal resolution** above 1x costs a lot of GPU time. Try 1x.
- Close other heavy apps, and run only one copy of macXenos at a time.
- Intel Macs, especially those with AMD graphics, are slower and less tested
  than Apple silicon.
- **Performance cores** on the Graphics blade should be on **Every thread**
  (the default). With it off, macOS parks most of a game's threads on the
  slower efficiency cores.

## Controller isn't detected

- Make sure macOS sees it: **System Settings > Game Controllers** should list
  it. If it doesn't, pair it first. Apple's guides walk through it:
  - [Connect a game controller to Mac](https://support.apple.com/guide/games/devf8cec167c/mac)
  - [Connect an Xbox wireless controller](https://support.apple.com/111101)
  - [Connect a PlayStation wireless controller](https://support.apple.com/111100)
  - [Connect any other wireless controller](https://support.apple.com/111099),
    including which controllers macOS supports
- If macOS says the controller needs a firmware update, update it with the
  maker's tools (for Xbox controllers, Microsoft's instructions), then pair
  it again.
- Check **Controller > Input system** in the dashboard is set to
  **Automatic** or **Game controllers (SDL)**.
- Unplug and reconnect it, or re-pair it over Bluetooth, while macXenos is
  open.

## Starting over with default settings

Quit macXenos and delete (or rename) `macXenos.config.toml` in
`~/Library/Application Support/macXenos`. macXenos writes a fresh one with
the default settings the next time it starts. Your profiles are kept
elsewhere in the same folder and your saves in `~/Documents/macXenos/Saves`;
neither is touched.

## Reporting a problem

Open an issue on [macXenos on GitHub](https://github.com/bourbonicfisky/macXenos/issues),
or choose **Help > Report a Bug** in the app. Please include:

- the game's name and, if you know it, its title ID,
- your Mac model and macOS version,
- which renderer you were using (it's shown in the window's title bar),
- the log file, `~/Library/Application Support/macXenos/xenia.log`, from
  the run that went wrong,
- a screenshot, if it's a graphics problem.

For graphics problems, a GPU trace is the most useful thing you can send.
Press **F4** (or choose **GPU > Trace Frame**) while the problem is on
screen. The trace is written under `scratch/gpu/` in the folder macXenos was
started from, unless you set `trace_gpu_prefix` in the config.

Please report problems to macXenos, not to the Xenia, Xenia Canary or Xenia
Edge projects. Their developers don't support this port.
