# Using macXenos

This guide covers installing macXenos, adding your games, and the settings
most people will want to change. If something isn't working, see
[Troubleshooting](troubleshooting.md).

## What you need

- macOS 15 Sequoia or later
- A Mac with Apple silicon, or an Intel Mac with a dedicated AMD GPU with at
  least 4 GB of video memory (Radeon RX 580 class or faster). Intel Macs are
  supported on a best-effort basis, and Intel integrated graphics aren't
  supported.
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
Bluetooth; Apple explains pairing for
[Xbox](https://support.apple.com/111101),
[PlayStation](https://support.apple.com/111100) and
[other wireless controllers](https://support.apple.com/111099). If one isn't
picked up, see [Controller isn't detected](troubleshooting.md#controller-isnt-detected).

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

- **Renderer.** Leave this on **Metal**, which draws natively on your Mac's
  GPU. **Vulkan** runs through the MoltenVK translation layer, is slower, and
  is kept only for comparison by developers.
- **Internal resolution.** How sharp the game is drawn. **1x** is the
  Xbox 360's own 720p. **2x** (1440p) and **3x** (2160p) look sharper but
  cost a lot more GPU time. On many games 2x runs at about half the frame
  rate of 1x, and 3x is only playable in lighter games.
- **Upscaling and sharpening.** How the picture is stretched to fill your
  window or screen: **Bilinear**, **AMD CAS**, **AMD FSR 1.0** or
  **Apple MetalFX**. FSR and MetalFX look best on large or high-resolution
  displays.
- **Uncapped games.** A few games, like GTA IV and Mafia II, never set a
  frame rate of their own and run as fast as your Mac can. **60 FPS** holds
  them to your display's rate and saves power.
- **Unset shader inputs.** Some games draw with a value they never set. The
  default, **Auto**, fills one in only for games known to need it (Resident
  Evil 5 today).
- **Performance cores.** Keeps the game's threads on your Mac's performance
  cores. Leave it on **Every thread** unless a game runs worse with it.
- **Troubleshooting.** Switches for narrowing down a rendering problem; the
  defaults are right for every game. See
  [A game looks wrong](troubleshooting.md#a-game-looks-wrong) for what each
  one does.
- **Refresh rate**, **FPS counter** and **Screen** options are here too.

## Profiles

The **Profiles** blade holds your player profiles, with gamercards and
gamer pictures. macXenos comes with its own set of pictures, and can download
the original Xbox 360 pictures if you ask it to. Saves and achievements
belong to the profile that is signed in.

## Where macXenos keeps your files

macXenos keeps your things in two places:

```
~/Documents/macXenos/Saves                 your game saves
~/Library/Application Support/macXenos     everything else
```

The Library folder is hidden by default. In Finder, choose **Go > Go to
Folder…** and paste the path.

| What | Where | Change it on the dashboard |
| --- | --- | --- |
| Game saves, one folder per profile | `~/Documents/macXenos/Saves` | **System > Game saves** |
| Your games (disc images and game folders) | Wherever you keep them; macXenos only reads them | **Home > Game folders** |
| Settings | `macXenos.config.toml` in Application Support | Every blade writes to it |
| Settings for one game | `config/<title ID>.config.toml` | |
| Profiles, downloads and title updates | `content/` | **System > Storage > Content folder** |
| Games' hard drive installs (the console's cache drives) | Application Support itself (`cache`, `cache0`, `cache1`) | **System > Game installs** |
| Game library details: the list of your game folders, what was found in them and what you played | `game_folders.toml`, `game_index.toml`, `recent.toml`, `library/` | |
| Shader cache, rebuilt if deleted | `cache_host/` | |
| The log from the last run | `xenia.log` | |
| Game patches you add | `patches/` | |
| Screenshots (**F12**), one folder per game | `~/Pictures/macXenos` | `screenshot_path` in the config file |

Game saves and game installs can go on an external drive. If that drive
isn't connected when a game starts, macXenos uses the default folder for that
session instead.

A new folder takes effect the next time macXenos starts, and nothing already
there is moved for you. To keep playing old saves, move each profile's folder
from the old **Saves** folder to the new one. A game whose installs you leave
behind simply installs again.

Versions before 1.0 kept everything in `~/.local/share/Xenia`. The first
time a newer version starts, it copies that folder to Application Support
and moves the saves in the copy to Documents. The old folder is left as it
was, as a backup; once you're happy everything came across, you can delete
it. Builds before November 2026 named the settings file
`xenia-canary.config.toml`; it is renamed the first time a newer build starts.

A few notes on those files:

- `macXenos.config.toml` is the main settings file. The dashboard writes
  to it for you. You only need to open it for the rare setting that isn't on
  a blade.
- A game's own `config/<title ID>.config.toml` overrides the main file for
  that game.
- Include `xenia.log` when you report a problem.
- `patches/` holds game patches you add (`<title ID> - <name>.patch.toml`,
  the Xenia patch format). Patches change a game's own code to work around a
  bug in the game.

macXenos also ships a few patches of its own for bugs in a game's code, for
example the NBA Ballers: Chosen One loading hang. They are on by default. To
turn one off, copy its file from
`macXenos.app/Contents/Resources/game_patches/` into your `patches/` folder
under the same name and set `is_enabled = false`; your copy replaces the
built-in one. `--apply_patches=false` turns off every patch.

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
