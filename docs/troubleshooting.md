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

- Try the other renderer: **Graphics > Renderer > Vulkan (MoltenVK)**, then
  restart the game. If it looks right on Vulkan, please report it. That
  helps narrow the problem down.
- If you raised **Internal resolution**, set it back to **1x** and see
  whether the problem goes away.
- A game may have known problems in Xenia generally. The community
  [compatibility list](https://xenia-manager.github.io/compatibility) is a
  good place to check.

## A game runs slowly

- The first session with a new game stutters while shaders are built. Give
  it a few minutes, or play a second session.
- **Internal resolution** above 1x costs a lot of GPU time. Try 1x.
- Close other heavy apps, and run only one copy of macXenos at a time.
- Intel Macs, especially those with AMD graphics, are slower and less tested
  than Apple silicon. If a game misbehaves there, try the Vulkan renderer.

## Controller isn't detected

- Make sure macOS sees it: **System Settings > Game Controllers** should list
  it.
- Check **Controller > Input system** in the dashboard is set to
  **Automatic** or **Game controllers (SDL)**.
- Unplug and reconnect it, or re-pair it over Bluetooth, while macXenos is
  open.

## Starting over with default settings

Quit macXenos and delete (or rename) `xenia-canary.config.toml` in
`~/.local/share/Xenia`. macXenos writes a fresh one with the default
settings the next time it starts. Your saves and profiles are kept elsewhere
in the same folder and are not touched.

## Reporting a problem

Please include:

- the game's name and, if you know it, its title ID,
- your Mac model and macOS version,
- which renderer you were using (it's shown in the window's title bar),
- the log file, `~/.local/share/Xenia/xenia.log`, from the run that went
  wrong,
- a screenshot, if it's a graphics problem.

For graphics problems, a GPU trace is the most useful thing you can send.
Press **F4** (or choose **GPU > Trace Frame**) while the problem is on
screen. The trace is written under `scratch/gpu/` in the folder macXenos was
started from, unless you set `trace_gpu_prefix` in the config.

Please report problems to macXenos, not to the Xenia, Xenia Canary or Xenia
Edge projects. Their developers don't support this port.
