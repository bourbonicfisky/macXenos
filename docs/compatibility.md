# Game compatibility

These are the games macXenos is tested with regularly. Each one is started
automatically, played to the point shown with a scripted controller, and
measured for a minute. Results below are from the run on October 13, 2026,
on a base M4 Mac mini at the original resolution (1x), using the Metal
renderer.

**"Reaches" is how far the test goes, not how far the game works.**

The frame rate is what the game ran at during the test. Xbox 360 games were
built to run at a fixed frame rate, almost always 30 or 60 frames per second,
and the game itself holds to that limit. On the console a game set to 30 never
goes above 30, however fast the hardware is, and macXenos runs each game at the
rate it was designed for. So 30 or 60 in the table means the game is running at
full speed, and a reading a frame or two either side of those numbers is the
same thing. A faster Mac does not raise the limit; it only helps a game that
falls below it. Many games also tie their timing to the frame rate, so running
them faster than intended would speed up the game itself rather than make it
smoother.

| Game | Reaches | Frame rate | Notes |
| --- | --- | ---: | --- |
| Batman: Arkham Asylum | Game options menu | | The test stops at the menu; gameplay not tested yet |
| Burnout Paradise | Driving in Paradise City | 30 | |
| Call of Duty 2 | First mission | 60 | |
| Crysis 2 | First mission | 31 | |
| Dark Souls | Undead Asylum | 30 | |
| Dead or Alive 4 | Fight (watch mode) | 60 | |
| Dead or Alive 5 | Title screen | | Only the title screen is tested |
| Dead to Rights: Retribution | Prologue | 30 | |
| Devil May Cry 4 | Prologue | 30 | |
| Gears of War 2 | Campaign, Act I | 30 | |
| Gears of War 3 | Campaign prologue | 30 | |
| Gears of War: Judgment | Survival, Estate | 30 | |
| Grand Theft Auto IV | Opening drive | 32 | |
| Halo 3 | Campaign, Arrival | 30 | |
| Halo 4 | Main menu | | Only the main menu is tested |
| Halo: Reach | Campaign, Winter Contingency | 29 | |
| Just Cause 2 | Open world | 30 | |
| LEGO Star Wars II | Mos Eisley Cantina | 60 | |
| Mass Effect | First conversation | 30 | Its built-in settings fix the one-frame colour glitches and characters vanishing for a frame (October 14) |
| Portal 2 | Relaxation Vault | 30 | |
| Red Dead Redemption | Opening cutscene | 29 | Disc 1 of the Game of the Year Edition |
| Resident Evil 5 | Chapter 1-1 | 14 | Playable, but well below full speed |
| Skate 2 | Skating | 30 | |
| Sonic Generations | First stage | 30 | |
| Ultra Street Fighter IV | Arcade fight | 60 | |
| Viva Piñata | Garden | 30 | |

None of these games crashed or hit a GPU error during its test.

## Other games

Many more games start and reach their menus or gameplay, but they aren't
tested regularly, so they aren't listed here. The community
[compatibility list](https://xenia-manager.github.io/compatibility) tracks
Xenia as a whole; results on the Mac can differ.

If a game doesn't work, see [Troubleshooting](troubleshooting.md) for what to
include in a report.
