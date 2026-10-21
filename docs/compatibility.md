# Game compatibility

These are the games macXenos is tested with regularly. Each one is started
automatically, played to the point shown with a scripted controller, and
measured for a minute. Results below are from the runs on October 19-20, 2026,
on a base M4 Mac mini and on an M1 Max MacBook Pro, at the original
resolution (1x), using the Metal renderer.

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

| Game | Reaches | M4 mini | M1 Max | Notes |
| --- | --- | ---: | ---: | --- |
| Assassin's Creed II | Story mode opening | 30 | 30 |  |
| Banjo-Kazooie: Nuts & Bolts | Spiral Mountain | 30 | 30 |  |
| Batman: Arkham Asylum | Intensive Treatment, escorting the Joker | 30 | 30 |  |
| Burnout Paradise | Driving in Paradise City | 60 | 58 | Its target is 60 |
| Call of Duty 2 | First mission | 60 | 60 |  |
| Crysis 2 | First mission | 31 | 31 |  |
| Dark Souls | Undead Asylum | 30 | 30 |  |
| Dead or Alive 4 | Fight (watch mode) | 60 | 60 |  |
| Dead or Alive 5 | Training fight | 60 | 60 |  |
| Dead to Rights: Retribution | Prologue | 30 | 28 | 28-30 across repeated runs on the M1 Max |
| Devil May Cry 4 | Prologue | 30 | 30 |  |
| Dishonored | New game opening | 30 | 30 |  |
| F.E.A.R. | Instant Action, Distribution | 30 | 30 |  |
| Fable II | Childhood, Bowerstone at night | 30 | 30 |  |
| Far Cry 3 | Opening (the cage) | 29 | 28 | Below full speed on the M1 Max |
| Gears of War | Campaign, Ashes | 30 | 30 |  |
| Gears of War 2 | Campaign, Act I | 30 | 30 |  |
| Gears of War 3 | Campaign prologue | 30 | 30 |  |
| Gears of War: Judgment | Survival, Estate | 30 | 30 |  |
| Grand Theft Auto IV | Opening drive | 48 | 50 | The game does not cap its own frame rate |
| Halo 3 | Campaign, Arrival | 30 | 30 |  |
| Halo 3: ODST | Campaign, Prepare to Drop | 30 | 30 |  |
| Halo 4 | Main menu | 30 | 30 | Only the main menu is tested |
| Halo: Reach | Campaign, Winter Contingency | 30 | 30 |  |
| Just Cause 2 | Open world | 27 | 30 |  |
| LEGO Star Wars II | Mos Eisley Cantina | 60 | 60 |  |
| Mafia II | Story opening | 82 | 83 | The game does not cap its own frame rate |
| Mass Effect | First conversation | 30 | 30 | Its built-in settings fix the one-frame colour glitches and characters vanishing for a frame (October 14) |
| Mirror's Edge | Story, first chapter | 30 | 30 |  |
| Portal 2 | Relaxation Vault | 30 | 30 |  |
| Prince of Persia: The Forgotten Sands | Opening | 30 | 30 | Sometimes stalls at the loading screen (a known intermittent hang); both test runs loaded |
| Rage | Campaign opening | 60 | 60 |  |
| Rayman Legends | Teensies in Trouble | 60 | 60 |  |
| Red Dead Redemption | Opening cutscene | 30 | 30 | Disc 1 of the Game of the Year Edition |
| Resident Evil 5 | Chapter 1-1 | 29 | 30 |  |
| Rise of the Tomb Raider | Ascension, the mountain | 30 | 30 |  |
| Skate 2 | Skating | 30 | 30 |  |
| Sonic Generations | First stage | 30 | 30 |  |
| Ultra Street Fighter IV | Arcade fight | 60 | 60 |  |
| Viva Piñata | Garden | 30 | 30 |  |

None of the games hit a GPU error on either Mac in these runs.

## Other games

Many more games start and reach their menus or gameplay, but they aren't
tested regularly, so they aren't listed here. The community
[compatibility list](https://xenia-manager.github.io/compatibility) tracks
Xenia as a whole; results on the Mac can differ.

If a game doesn't work, see [Troubleshooting](troubleshooting.md) for what to
include in a report.
