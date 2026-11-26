# 2x and 3x benchmarks

**Every game in the [compatibility list](compatibility.md) should work at 2x and 3x.** The figures below are a barometer, not a compatibility verdict: they show how demanding each game is at a higher internal resolution on each Mac, so you can pick a scale that suits your Mac and the game.

The internal resolution scale (Graphics > Resolution scale) renders the game at two or three times its original resolution (1280×720 becomes 2560×1440 at 2x and 3840×2160 at 3x), for a sharper picture. It costs GPU time, so a game that holds its frame rate at 1x can drop below it at 2x or 3x on a smaller Mac.

## How to read the table

- Each game was started automatically, played to the same point as in the compatibility list with a scripted controller, and measured there. The figure is the median frame rate.
- Most Xbox 360 games are capped at 30 or 60 frames per second. 30 or 60 here means the game runs at full speed; a figure below its cap means that Mac can't keep up at that scale.
- *(uncapped)* games don't limit their own frame rate, so they run as fast as the Mac allows, and the number is a measure of headroom.
- A dash means there is no figure from this run, not that the game fails at that scale. The 2x runs on the M4 Macs and the M1 Max didn't reach every game before the run ended.
- Measured on November 23 and 24, 2026 with a macXenos 1.1 development build that has the same 2x and 3x rendering as the 1.1 release, using the Metal renderer. The Macs are a base M4 Mac mini, an M4 Pro MacBook Pro, an M1 Max MacBook Pro with 32 GB and an M5 Max Mac Studio. The M5 Max ran 3x only.

| Game | M4 mini 1x | M4 mini 2x | M4 Pro 1x | M4 Pro 2x | M1 Max 1x | M1 Max 2x | M5 Max 3x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 Cent: Blood on the Sand *(uncapped)* | 101 | 35 | 179 | 61 | 101 | 52 | 69 |
| Alan Wake | 30 | – | 30 | – | 30 | – | 30 |
| Assassin's Creed | 30 | 28 | 30 | 30 | 30 | 30 | 30 |
| Assassin's Creed II | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Assassin's Creed III | 60 | 27 | 60 | 49 | 60 | 38 | 56 |
| Batman: Arkham Asylum | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Batman: Arkham Origins | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Battlefield 3 | 30 | – | 30 | – | 30 | – | 21 |
| Bayonetta | 60 | 60 | 60 | 60 | 60 | 60 | 60 |
| BioShock | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| BioShock 2 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Blood Bowl *(uncapped)* | 132 | 46 | 132 | 79 | 97 | 67 | 77 |
| Blur | 30 | – | 30 | – | 30 | – | 30 |
| Bully: Scholarship Edition | 60 | – | 60 | – | 60 | – | 21 |
| Burnout Paradise | 60 | 58 | 58 | 56 | 41 | 37 | 58 |
| Call of Duty 2 | 60 | – | 60 | – | 60 | – | 60 |
| Call of Duty 4: Modern Warfare | 60 | 32 | 60 | 59 | 60 | 60 | 60 |
| Call of Duty: Black Ops | 60 | – | 60 | – | 60 | – | 27 |
| Call of Duty: Black Ops II | 60 | – | 60 | – | 60 | – | – |
| Call of Duty: Modern Warfare 2 | 60 | – | 60 | – | 60 | – | 60 |
| Call of Duty: Modern Warfare 3 | 60 | – | 60 | – | 60 | – | 60 |
| Crysis 2 | 31 | 21 | 31 | 31 | 31 | 28 | 31 |
| Crysis 3 | 31 | 26 | 31 | 31 | 31 | 24 | 31 |
| Dante's Inferno | 60 | – | 60 | – | 60 | – | 60 |
| Dark Souls | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Dark Souls II | 60 | 40 | 60 | 60 | 60 | 50 | 60 |
| Darkest of Days *(uncapped)* | 158 | 58 | 241 | 98 | 149 | 102 | 119 |
| Darksiders *(uncapped)* | 133 | 42 | 224 | 63 | 96 | 55 | 77 |
| Dead Rising | 30 | – | 30 | – | 30 | – | 21 |
| Dead Space | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Dead Space 2 | 30 | 22 | 30 | 30 | 30 | 25 | 30 |
| Dead to Rights: Retribution | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Deus Ex: Human Revolution | 30 | – | 30 | – | 30 | 30 | 30 |
| Devil May Cry 4 | 30 | 30 | 60 | 30 | 30 | 30 | 30 |
| Diablo III | 60 | – | 60 | – | 60 | 49 | 60 |
| Dishonored | 30 | – | 30 | – | 30 | – | 30 |
| Doom 3: BFG Edition | 60 | – | 60 | – | 60 | – | 60 |
| The Elder Scrolls IV: Oblivion | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| The Elder Scrolls V: Skyrim | 30 | 26 | 30 | 30 | 30 | 30 | 30 |
| F.E.A.R. | 30 | – | 30 | – | 30 | – | 30 |
| Fable II | 30 | 26 | 30 | – | 29 | 25 | 30 |
| Fable III | 30 | – | 30 | – | 30 | 30 | – |
| Fallout: New Vegas | 30 | 30 | 30 | – | 30 | 30 | 30 |
| Far Cry 2 | 30 | – | 30 | – | 30 | – | 30 |
| Far Cry 3 | 30 | – | 30 | – | 30 | – | 6 |
| Final Fantasy XIII | 30 | – | 30 | – | 30 | 30 | 30 |
| Forza Horizon | 30 | – | 30 | – | 30 | – | 30 |
| Forza Horizon 2 | 30 | – | 30 | – | 30 | – | 30 |
| Forza Motorsport 4 | 60 | – | 60 | – | 60 | – | 60 |
| Gears of War | 30 | 27 | 30 | 30 | 30 | 30 | 30 |
| Gears of War 2 | 30 | 29 | 30 | 30 | 30 | 30 | 30 |
| Gears of War 3 | 30 | 26 | 30 | 30 | 30 | 30 | 30 |
| Gears of War: Judgment | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Grand Theft Auto IV *(uncapped)* | 60 | 25 | 60 | 60 | 60 | 36 | 57 |
| Grand Theft Auto V | 30 | 26 | 30 | 30 | 30 | 29 | 30 |
| Halo 3 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Halo 3: ODST | 30 | 30 | 28 | 28 | 30 | 30 | 30 |
| Halo 4 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Halo: Reach | 30 | 28 | 30 | 30 | 30 | 30 | 30 |
| Juiced 2: Hot Import Nights *(uncapped)* | 108 | 40 | 135 | 68 | 83 | 57 | 79 |
| Just Cause 2 | 28 | 27 | 27 | 27 | 30 | 30 | 26 |
| L.A. Noire | 30 | – | 30 | – | 30 | – | 30 |
| LEGO Star Wars II | 60 | 60 | 60 | 60 | 60 | 59 | 60 |
| Mafia II *(uncapped)* | 143 | 51 | 163 | 85 | 101 | 67 | 109 |
| Mass Effect | 30 | 30 | 30 | – | 30 | 30 | 30 |
| Mass Effect 3 | 30 | 30 | 30 | – | 30 | 30 | 30 |
| Max Payne 3 | 33 | – | 33 | – | 33 | – | 33 |
| Metal Gear Solid V: The Phantom Pain | 60 | – | 58 | – | 60 | – | 52 |
| Mirror's Edge | 30 | – | 30 | – | 30 | – | 30 |
| Need for Speed: Most Wanted | 60 | – | 60 | – | 60 | – | 60 |
| Project Gotham Racing 3 | 30 | – | 30 | – | 30 | – | 30 |
| Rayman Legends | 60 | 60 | 60 | 60 | 60 | 60 | 60 |
| Red Dead Redemption | 30 | 25 | 30 | 30 | 30 | 29 | 30 |
| Resident Evil 5 | 29 | 29 | 27 | 28 | 30 | 30 | 28 |
| Saints Row 2 | 29 | – | 28 | – | 30 | – | 12 |
| Saints Row IV *(uncapped)* | 63 | 20 | 111 | 36 | 40 | 27 | 39 |
| Skate 2 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Sonic Generations | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Tales of Vesperia | 30 | – | 30 | – | 30 | 30 | 30 |
| Ultra Street Fighter IV | 60 | 40 | 60 | 60 | 60 | 60 | 60 |
