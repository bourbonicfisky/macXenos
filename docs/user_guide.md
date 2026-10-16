# Using macXenos

This guide covers installing macXenos, adding your games, and the settings
most people will want to change. If something isn't working, see
[Troubleshooting](troubleshooting.md).

## What you need

- macOS 15 Sequoia or later
- A Mac with Apple silicon (Intel Macs work on a best-effort basis)
- About 2 to 3 GB of free memory while a game runs
- A game controller
- Xbox 360 games

## Installing

1. Download the latest build from [macxenos.com](https://macxenos.com).
2. Open the `.dmg` and drag macXenos into your Applications folder.
3. Open macXenos from Applications.

If macOS says it can't check the app for malicious software, open
**System Settings > Privacy & Security**, scroll down, and click
**Open Anyway** next to the message about macXenos. You only need to do this
once.

### Updates

macXenos checks for new versions by itself and offers to install them. You
can also check by hand with **macXenos > Check for Updates…** in the menu
bar.

## Adding your games

macXenos reads:

- disc images (`.iso`),
- unpacked games (a folder with a `default.xex` file inside),
- downloaded game packages (Xbox Live Arcade games, demos and similar).

The easiest way to keep a library is to tell macXenos which folders hold
your games. On the **Home** blade, choose **Game folders**, then
**Edit folders** to add one. macXenos scans the folders and lists every game
it finds under **All games**, with its icon. Choose **Refresh folders**
after you add new games.

You can also start a game with **File > Open**, or from Terminal:

```sh
open -n /Applications/macXenos.app --args "/path/to/game.iso"
```

The first time you play a game it may stutter for a while. macXenos is
building that game's shaders and saves them for next time, so later runs are
smoother.

## The dashboard

All of macXenos's settings live in a dashboard modeled on the Xbox 360's
original "Blades" dashboard. To open it:

- tap the **Guide** (Xbox) button on your controller,
- press **F6**, or
- press **Cmd-,** (comma).

While the dashboard is open over a game, the game pauses and goes quiet.

The blades are **Home** (recent games and your library), **Graphics**,
**Controller**, **Audio & CPU**, **Profiles**, **System** and
**Advanced**. Some settings take effect only the next time you start a
game; the dashboard tells you when that's the case.

## Controllers

Most controllers that macOS recognizes work without any setup, including
Xbox, PlayStation and Switch Pro controllers. Plug them in or pair them over
Bluetooth.

Under **Controller** you can:

- choose which player each controller is,
- remap buttons on a picture of the pad,
- adjust vibration and stick dead zones,
- choose how to open the dashboard (**Dashboard shortcut**: tap Guide, or
  hold Back + Start).

### Playing on the keyboard

The keyboard is off by default. To use it, go to **Controller > Keyboard**
and choose **Plays as a controller**. The game window has to be in front for
keys to register. The default keys are:

| Controller | Keys |
| --- | --- |
| Left stick | W A S D |
| D-pad | Shift + W A S D |
| Right stick | Arrow keys |
| A / B / X / Y | `;` / `'` / L / P |
| Start / Back | X / Z |
| LB / RB | 1 / 3 |
| LT / RT | Q or I / E or O |
| Stick clicks | F (left) / K (right) |

You can change these in the config file under `[HID.Key]` (see
[Where macXenos keeps your files](#where-macxenos-keeps-your-files)).

## Graphics

Everything here is on the **Graphics** blade.

- **Renderer.** **Metal** is the default and the one to use. **Vulkan
  (MoltenVK)** is an older renderer kept as a fallback. Try it if a game
  looks wrong on Metal.
- **Internal resolution.** How sharp the game is drawn. **1x** is the
  Xbox 360's own 720p. **2x** (1440p) and **3x** (2160p) look sharper but
  cost a lot more GPU time. On many games 2x runs at about half the frame
  rate of 1x, and 3x is only playable in lighter games.
- **Upscaling and sharpening.** How the picture is stretched to fill your
  window or screen: **Bilinear**, **AMD CAS**, **AMD FSR 1.0** or
  **Apple MetalFX**. FSR and MetalFX look best on large or high-resolution
  displays.
- **Refresh rate**, **FPS counter** and **Screen** options are here too.

## Profiles

The **Profiles** blade holds your player profiles, with gamercards and
gamer pictures. macXenos comes with its own set of pictures, and can download
the original Xbox 360 pictures if you ask it to. Saves and achievements
belong to the profile that is signed in.

## Where macXenos keeps your files

Game saves are kept in your Documents folder:

```
~/Documents/macXenos/Saves
```

Everything else (settings, profiles, downloads and title updates, game
installs, the shader cache and the log) is kept in:

```
~/Library/Application Support/macXenos
```

The Library folder is hidden by default. In Finder, choose **Go > Go to
Folder…** and paste the path above. Both folders can be changed on the
**System** blade (**Game saves** and **Game installs**).

Versions before 1.0 kept everything in `~/.local/share/Xenia`. The first
time a newer version starts, it copies that folder to Application Support
and moves the saves in the copy to Documents. The old folder is left as it
was, as a backup; once you're happy everything came across, you can delete
it.

Inside it:

- `xenia-canary.config.toml` is the main settings file. The dashboard writes
  to it for you. You only need to open it for the rare setting that isn't on
  a blade.
- `config/<title ID>.config.toml` holds settings for a single game, if you
  make one. These override the main file for that game.
- `xenia.log` is the log from the last run. Include it when you report a
  problem.

To keep everything next to the app instead (for example on an external
drive), put an empty file named `portable.txt` in
`macXenos.app/Contents/MacOS/`.

## Performance

- Apple silicon Macs run most tested games at the speed the game was built
  for (usually 30 or 60 frames per second) at 1x.
- Intel Macs are supported on a best-effort basis and are slower.
- Run one copy of macXenos at a time. On a Mac with 16 GB of memory, two
  copies at once can run the system out of memory.
- Shader compiling makes the first few minutes of a new game less smooth than
  later sessions.
