# Game compatibility

> **2x and 3x:** every game in this list should work at 2x and 3x resolution. To see how demanding each game is on each Mac at those scales, see the [2x and 3x benchmarks](scaling.md).

These are the games macXenos is tested with regularly: 305 of them, plus the
54 Xbox Live Arcade games further down. Each one is started automatically,
played to the point shown with a scripted controller, and measured. The four
columns are four Macs: a base M4 Mac mini, an M4 Pro MacBook Pro, an M1 Max
MacBook Pro with 32 GB and an M5 Max Mac Studio, all at the original
resolution (1x) with the Metal renderer. A dash means that Mac hasn't measured
the game yet.

### Which build each figure comes from

Games are re-tested build by build and Mac by Mac, so one row can hold
figures from different builds. The small letter after each figure names the
build it was measured with:

| Key | Build | Tested |
| --- | --- | --- |
| **a** | macXenos 1.1 (build 202611252154) | November 25, 2026 |
| **b** | macXenos 1.1-rc2 (build 202611250825) | November 25, 2026 |
| **c** | macXenos 1.1 development builds | November 21–24, 2026 |
| **d** | macXenos 1.0.1-rc2 (build 202611152108), the figures published with 1.0.1 and 1.0.2 | November 15–16, 2026 |
| **e** | earlier 1.0.x builds | November 2–15, 2026 |

A figure keeps its letter until a newer build measures that game on that Mac
again; new builds are added to the top of the key.

**"Test reaches" is only how far the automated test goes, not how far the
game can be played.** Each test plays the opening of the game with a scripted
controller, stops at the point shown, and measures that scene. What happens
later in the game isn't covered: it may play to the end, or it may have
problems the test never gets to.

The frame rate is what the game ran at during the test. Xbox 360 games were
built to run at a fixed frame rate, almost always 30 or 60 frames per second,
and the game itself holds to that limit. On the console a game set to 30 never
goes above 30, however fast the hardware is, and macXenos runs each game at the
rate it was designed for. So 30 or 60 in the table means the game is running at
full speed, and a reading a frame or two either side of those numbers is the
same thing. A faster Mac does not raise the limit; it only helps a game that
falls below it. The M1 Max, an older chip, falls below the limit in a few
demanding games that the M4 Macs run at full speed, such as NBA 2K11. Many games also tie their timing to the frame rate, so running
them faster than intended would speed up the game itself rather than make it
smoother.

| Game | Test reaches | M4 mini | M4 Pro | M1 Max | M5 Max | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 007 Legends | Opening level | 60 <sup>a</sup> | 60 <sup>a</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| 007: Blood Stone | Athens harbour, in the speedboat | 30 <sup>a</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>e</sup> |  |
| 007: Quantum of Solace | First level | 30 <sup>a</sup> | 30 <sup>b</sup> | – | 30 <sup>d</sup> |  |
| 2010 FIFA World Cup South Africa | Quick Match, USA v Mexico, CPU vs CPU | 60 <sup>b</sup> | 60 <sup>a</sup> | – | – |  |
| 2014 FIFA World Cup Brazil | Kick Off, Brazil v Argentina at the Maracanã | 60 <sup>b</sup> | 60 <sup>a</sup> | – | – | Runs 30 when the test ends on a Brazil set piece waiting for the player; open play holds 60 |
| 50 Cent: Blood on the Sand | Mission 1, the opening street | 102 <sup>a</sup> | 179 <sup>b</sup> | 101 <sup>c</sup> | 128 <sup>d</sup> | Doesn't cap its own frame rate (see below) |
| Ace Combat 6: Fires of Liberation | Mission 1, the take-off sequence | 30 <sup>a</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>e</sup> |  |
| Ace Combat: Assault Horizon | Mission 1, over Miami | 30 <sup>b</sup> | 30 <sup>a</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Adventure Time: Explore the Dungeon Because I Don't Know! | New game, the Candy Kingdom hub | 60 <sup>a</sup> | 60 <sup>a</sup> | 60 <sup>b</sup> | – |  |
| Adventure Time: Finn & Jake Investigations | The tutorial prompt in the treehouse | 30 <sup>a</sup> | 30 <sup>a</sup> | 30 <sup>b</sup> | – |  |
| AFL Live | Single match, CPU vs CPU at the MCG | 122 <sup>a</sup> | 136 <sup>a</sup> | – | 134 <sup>b</sup> | Doesn't cap its own frame rate (see below) |
| AFL Live 2 | Single match, CPU vs CPU (random teams and venue) | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Afro Samurai | Village tutorial fight | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>e</sup> |  |
| Alan Wake | Opening nightmare on the mountain road | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Assassin's Creed | The Animus tutorial crowd | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Assassin's Creed II | Story mode opening | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Assassin's Creed III | Opening, Desmond at the cave entrance | 60 <sup>a</sup> | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>d</sup> | This scene runs at 60 |
| Banjo-Kazooie: Nuts & Bolts | Spiral Mountain | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | Occasionally freezes at the boot logo or after the menu loads |
| Batman: Arkham Asylum | Intensive Treatment, escorting the Joker | 30 <sup>a</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Batman: Arkham Origins | Blackgate rooftop | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Battlefield 3 | Semper Fidelis, the pause menu over the train fight | 30 <sup>a</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | – | Tested on Easy, ending on the pause menu because the idle player dies quickly |
| Battlefield: Bad Company | First level | 30 <sup>a</sup> | – | 30 <sup>d</sup> | 30 <sup>b</sup> |  |
| Bayonetta | Prologue, the clock tower fight | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>d</sup> |  |
| Beautiful Katamari | Rolling practice | 60 <sup>a</sup> | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Bionic Commando | First level | 60 <sup>b</sup> | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| BioShock | The plane crash, first control | 30 <sup>a</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| BioShock 2 | Adonis Luxury Resort | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Birds of Steel | Takeoff tutorial, the fighter on the island airstrip | 60 <sup>b</sup> | – | 60 <sup>c</sup> | – |  |
| Black College Football: The Xperience | Play Now, the opening coin toss | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Blades of Time | Chapter 1, the first fight on the stone stairs | 30 <sup>b</sup> | – | – | – |  |
| Bladestorm: The Hundred Years' War | New game, character creation in the tavern | 30 <sup>b</sup> | – | 30 <sup>c</sup> | – |  |
| BlazBlue: Calamity Trigger | Training, Ragna vs the training dummy | 60 <sup>a</sup> | 60 <sup>b</sup> | 60 <sup>c</sup> | – |  |
| BlazBlue: Continuum Shift Extend | Training, Ragna vs the training dummy | 60 <sup>b</sup> | – | 60 <sup>c</sup> | – |  |
| Blazing Angels 2: Secret Missions of WWII | Campaign mission 1, the Spitfire on the airshow runway | 30 <sup>b</sup> | – | 30 <sup>c</sup> | – |  |
| Blazing Angels: Squadrons of WWII | Campaign mission 1, the biplane on the airfield | 60 <sup>a</sup> | – | 60 <sup>b</sup> | – |  |
| Blitz: The League | Quick Play, the pause menu over the line of scrimmage | – | – | 60 <sup>b</sup> | 60 <sup>c</sup> |  |
| Blitz: The League II | Pause menu over the opening kickoff | – | – | 43 <sup>d</sup> | 60 <sup>d</sup> | Can stop during start-up (rare). Not measured on the M4 Macs: their tests don't reach this scene |
| Blood Bowl | One-off match, the kick-off on the pitch | 132 <sup>c</sup> | 132 <sup>c</sup> | 101 <sup>b</sup> | 144 <sup>a</sup> | Doesn't cap its own frame rate (see below) |
| Blood Drive | Tournament, the first event among the zombies | 30 <sup>b</sup> | – | 30 <sup>c</sup> | – |  |
| Bloodbath | Training, a free-for-all against bots | – | – | 30 <sup>b</sup> | – |  |
| Blue Dragon | Shu on the village path after the opening | – | – | 30 <sup>b</sup> | 30 <sup>c</sup> |  |
| Blur | Career, the first race at the Barcelona Oval | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> |  |
| Brothers in Arms: Hell's Highway | First chapter, the hospital | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Bully: Scholarship Edition | Bullworth campus entrance | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Burnout Paradise | Driving in Paradise City | 60 <sup>b</sup> | 58 <sup>c</sup> | 41 <sup>c</sup> | 60 <sup>a</sup> |  |
| Burnout Revenge | First World Tour event | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Cabela's Big Game Hunter 2010 | First hunt, the forest trail | – | – | 30 <sup>b</sup> | 30 <sup>e</sup> |  |
| Call of Duty 2 | First mission | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>d</sup> |  |
| Call of Duty 3 | Saint-Lô, the training yard | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>e</sup> |  |
| Call of Duty 4: Modern Warfare | First mission, the hangar | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>d</sup> |  |
| Call of Duty: Black Ops | Operation 40, the bar in Havana | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>e</sup> |  |
| Call of Duty: Black Ops II | First mission, Pyrrhic Victory | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | The main menu background sometimes renders almost white |
| Call of Duty: Ghosts | Local play, Team Deathmatch on Prison Break | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>e</sup> | Needs a profile that has played the game once (its brightness and margin screens done) |
| Call of Duty: Modern Warfare 2 | First mission, the shooting range | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Call of Duty: Modern Warfare 3 | First mission, Black Tuesday | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>d</sup> |  |
| Call of Juarez: Bound in Blood | Chapter 1, the trenches | 60 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Capcom Digital Collection | 1942: Joint Strike, mission 1, picked from the launcher | 60 <sup>b</sup> | – | 60 <sup>c</sup> | – | Tested as the trial of 1942: Joint Strike |
| Captain America: Super Soldier | The trench at first control | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>e</sup> | Occasionally starts to a black screen (one start in four in testing); starting the game again gets past it |
| Carrier Command: Gaea Mission | Campaign, first control at the crash site | – | – | 60 <sup>b</sup> | – |  |
| Cartoon Network: Punch Time Explosion XL | Story mode, the first level | – | – | 60 <sup>b</sup> | – |  |
| Champion Jockey: G1 Jockey & Gallop Racer | Free Race, Greenfield 3200 m | 30 <sup>b</sup> | 30 <sup>c</sup> | – | – |  |
| Chaotic: Shadow Warriors | Story mode, first control in the Forest of Life | 60 <sup>b</sup> | – | 60 <sup>c</sup> | – |  |
| Chikyuu Boueigun 4 (Earth Defense Force 4, Japan) | Mission 1, the city street as the giant ants advance | 44 <sup>c</sup> | – | 30 <sup>b</sup> | – | Runs below its 60 limit in this scene, which draws far more than most |
| Child of Eden | Stage 1, the Player Life tutorial card over the paused tunnel | 60 <sup>c</sup> | – | – | 60 <sup>b</sup> |  |
| Chromehounds | First training sortie | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>e</sup> |  |
| The Chronicles of Narnia: Prince Caspian | Level 1, Defend Cair Paravel | – | – | 30 <sup>c</sup> | 30 <sup>b</sup> | Tested without saving: the game's save step still loops |
| The Chronicles of Riddick: Assault on Dark Athena | Butcher Bay, the cell block | – | 30 <sup>b</sup> | – | 30 <sup>e</sup> |  |
| Civil War: A Nation Divided | Fredericksburg, first control before the bridge | – | – | 30 <sup>b</sup> | – |  |
| Clash of the Titans: The Videogame | First control on the beach | – | 30 <sup>b</sup> | – | 30 <sup>e</sup> |  |
| Clive Barker's Jericho | Opening scene in the transport | – | – | 60 <sup>b</sup> | 60 <sup>e</sup> |  |
| Command & Conquer: Red Alert 3 | Tutorial, the first training map | – | – | 30 <sup>b</sup> | – |  |
| Conan | Chapter 1, first control in the temple hall | 30 <sup>c</sup> | – | 21 <sup>b</sup> | – |  |
| Condemned: Criminal Origins | First control in the opening alley | – | – | – | 44 <sup>b</sup> | Runs at about 44 everywhere, menus included, instead of 30 or 60 |
| Conflict: Denied Ops | Campaign, first control at the monastery | – | 30 <sup>c</sup> | – | 30 <sup>b</sup> |  |
| The Conveni 200X | Scenario 1, the store placement map | – | – | 60 <sup>b</sup> | – | Japanese release |
| Crackdown | Campaign, first control in the Agency hangar | – | – | – | 30 <sup>b</sup> | Runs a little under its 30 limit |
| Crackdown 2 | The training room after the opening movie | 30 <sup>b</sup> | – | 30 <sup>b</sup> | 30 <sup>c</sup> |  |
| Crash Time 4: The Syndicate | Career, the first patrol in the city | 30 <sup>b</sup> | 30 <sup>c</sup> | – | – |  |
| Crash Time 5: Undercover | Career, the first mission on an Alpine pass | – | 30 <sup>c</sup> | – | 30 <sup>b</sup> |  |
| Crash Time: Autobahn Pursuit | The Cases, the first chase | – | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Crash: Mind over Mutant | First control in Crash's house | – | 30 <sup>c</sup> | – | – |  |
| Create | The first-run tutorial stage | – | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Cross Channel: In Memory of All People | New game, the first story line | 60 <sup>b</sup> | – | 60 <sup>c</sup> | – | Japanese release |
| Crysis 2 | First mission | 31 <sup>c</sup> | 31 <sup>b</sup> | 31 <sup>c</sup> | 31 <sup>a</sup> |  |
| Crysis 3 | First mission, inside the Liberty Dome | 31 <sup>b</sup> | 31 <sup>c</sup> | 31 <sup>c</sup> | 31 <sup>a</sup> |  |
| CSI: Deadly Intent | Case 1, the first conversation at the crime scene | – | 60 <sup>c</sup> | 60 <sup>b</sup> | – |  |
| CSI: Fatal Conspiracy | Episode 1, the first crime scene | 51 <sup>b</sup> | 60 <sup>c</sup> | – | – |  |
| Dante's Inferno | First control in the Citadel of Acre | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>a</sup> |  |
| Dark | First control in the night club | – | 175 <sup>c</sup> | 122 <sup>b</sup> | – | Doesn't cap its own frame rate (see below). The club scene renders almost black |
| Dark Messiah of Might and Magic: Elements | Prologue, first control on the canyon ledge | – | 30 <sup>c</sup> | – | 30 <sup>b</sup> |  |
| Dark Sector | First control on the Lasria rooftop | – | – | 60 <sup>b</sup> | 60 <sup>c</sup> |  |
| Dark Souls | Undead Asylum | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | – |  |
| Dark Souls II | Things Betwixt, the first glade | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> |  |
| Darkest of Days | Little Bighorn, first control on the hillside | 158 <sup>c</sup> | 241 <sup>c</sup> | 146 <sup>b</sup> | – | Doesn't cap its own frame rate (see below) |
| The Darkness | The opening car ride, the tunnel chase | – | – | – | 30 <sup>c</sup> | In 1.1 it can stay on a white screen at the start (an audio change in 1.1; a fix is in progress) |
| The Darkness II | The opening restaurant scene | – | – | – | 32 <sup>b</sup> |  |
| Darksiders | The pause menu over the ruined city at first control | 144 <sup>b</sup> | 224 <sup>c</sup> | 96 <sup>c</sup> | 120 <sup>c</sup> | Doesn't cap its own frame rate (see below) |
| Darksiders II | First control in the frozen Veil | 38 <sup>b</sup> | – | 85 <sup>c</sup> | 38 <sup>c</sup> | Holds about 30-35 with the game's adaptive vsync; the 85 is from before a built-in game patch, when some starts ran uncapped |
| Dead or Alive 4 | Fight (watch mode) | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Dead or Alive 5 | Training fight | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Dead or Alive Xtreme 2 | New Zack Island, the Day 1 map | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Dead Rising | 72 Hour Mode, the mall heliport | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | Once stayed on a loading screen after the roof-landing cutscene was skipped |
| Dead Rising 2: Off the Record | The opening arena match | – | – | – | 30 <sup>b</sup> |  |
| Dead Space | Opening cutscene, the Kellion's cockpit | 30 <sup>a</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Dead Space 2 | The pause menu over the opening hospital scene | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> |  |
| Dead Space 3 | Prologue, first control in the blizzard | 30 <sup>a</sup> | – | – | – |  |
| Dead to Rights: Retribution | Prologue | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Def Jam: Icon | Exhibition fight | – | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Deus Ex: Human Revolution | Sarif Industries, Megan's lab | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Devil May Cry 4 | Prologue | 30 <sup>c</sup> | 60 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Diablo III | First area, the night road | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>a</sup> |  |
| Dishonored | New game opening | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | – |  |
| Doom 3: BFG Edition | Mars City hangar | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Dragon Age II | Prologue battle | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Dragon Age: Origins | Human Noble origin, Castle Highever | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Dragon Age: Origins - Awakening | The road to Vigil's Keep | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Dynasty Warriors 6 | Challenge mode, Rampage | – | 60 <sup>b</sup> | – | 60 <sup>d</sup> |  |
| Dynasty Warriors 7 | Wei story, chapter 1 | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Earth Defense Force 2017 | Mission 1, the city street under the mothership | 60 <sup>b</sup> | – | – | 60 <sup>c</sup> |  |
| Earth Defense Force 2025 | Mission 1, the city crossing | – | 42 <sup>b</sup> | – | 50 <sup>c</sup> | Runs below its 60 limit in this scene, which draws far more than most |
| Earth Defense Force: Insect Armageddon | Chapter 1, the pause menu over the helicopter ride | 30 <sup>b</sup> | – | – | 30 <sup>c</sup> |  |
| Eat Lead: The Return of Matt Hazard | Chapter 1, the warehouse yard | 30 <sup>b</sup> | – | 30 <sup>c</sup> | – |  |
| The Elder Scrolls IV: Oblivion | Imperial Prison cell | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| The Elder Scrolls V: Skyrim | Opening cart ride to Helgen | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> | Sometimes crashes or stops responding in play; being worked on |
| Enchanted Arms | Opening battle | 30 <sup>b</sup> | 30 <sup>d</sup> | – | 30 <sup>d</sup> |  |
| Eragon | First level, the forest clearing | 60 <sup>b</sup> | 60 <sup>c</sup> | – | – |  |
| F.E.A.R. | Instant Action, Distribution | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| F1 2011 | Time Trial, Melbourne | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Fable II | Childhood, Bowerstone at night | 30 <sup>c</sup> | 30 <sup>b</sup> | 29 <sup>c</sup> | 30 <sup>d</sup> |  |
| Fable III | New game, the Prince's bedroom | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | – | Can stop on a black screen loading after character select, most often on M5 Macs (also in 1.0.x) |
| Fallout: New Vegas | Doc Mitchell's house | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Far Cry 2 | Opening taxi ride | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Far Cry 3 | Opening (the cage) | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | At 2x internal resolution a magenta band can smear down from the top of the picture |
| Far Cry 4 | Opening, dinner with Pagan Min | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>b</sup> |  |
| FIFA 14 | Kick Off match, CPU vs CPU | 60 <sup>a</sup> | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| FIFA 16 | Kick Off match, CPU vs CPU | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| FIFA Street 3 | Play Now match | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | Can stall on the EA SPORTS BIG logo at start-up |
| Fight Night Champion | Fight Now, Tyson vs Ali, CPU vs CPU | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | Occasionally the ring ropes show wrong colours for a whole bout |
| Fight Night Round 4 | Fight Now, Ali vs Tyson, round 1 | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>a</sup> | Boxers' skin can show a foil-like sparkle for a whole session |
| Final Fantasy XIII | The Hanging Edge, after the first battle | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | – | After a Game Over and Retry in the first battle, Lightning's skin and the walkway can render wrongly |
| Fist of the North Star: Ken's Rage | Legend Mode, first chapter fight | – | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Forza Horizon | Opening drive | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Forza Horizon 2 | Drive to the Festival | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Forza Motorsport 2 | Exhibition race, on the grid | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | Some menu text is missing letters |
| Forza Motorsport 3 | The opening race, the Audi R8 on the start straight | 60 <sup>c</sup> | – | – | 60 <sup>b</sup> |  |
| Forza Motorsport 4 | First race, cockpit view | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>d</sup> | In cockpit view the windshield and side windows are flat grey |
| Frontlines: Fuel of War | First mission, defending the crash site | – | 46 <sup>b</sup> | – | 49 <sup>d</sup> | The game paces itself at about 46-50, in menus and videos too |
| Fuel | Career, The Leap of Faith | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Gears of War | Campaign, Ashes | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Gears of War 2 | Campaign, Act I | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Gears of War 3 | Campaign prologue | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Gears of War: Judgment | Survival, Estate | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Ghost Recon Advanced Warfighter | Training mission, first street | 60 <sup>b</sup> | – | – | 60 <sup>d</sup> | Not measured on the M4 Pro and M1 Max: their tests don't reach this scene |
| The Godfather II | Havana rooftop party | – | 30 <sup>b</sup> | – | 30 <sup>d</sup> |  |
| GoldenEye 007: Reloaded | First mission | – | 60 <sup>b</sup> | – | 60 <sup>d</sup> |  |
| Grand Theft Auto IV | Opening drive | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>a</sup> | 60 <sup>d</sup> | Now holds the console's 60 refresh rate (54-55 in October) |
| Grand Theft Auto V | Prologue, first control in the bank | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>a</sup> | Runs from Disc 2; Disc 1 is read from the image beside it |
| Guitar Hero II | Practice, a full song | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Half-Life 2 (The Orange Box) | Main menu over its live map | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | Starting the first chapter can stop at the loading screen |
| Halo 3 | Campaign, Arrival | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>a</sup> | 30 <sup>d</sup> |  |
| Halo 3: ODST | Campaign, Prepare to Drop | 30 <sup>c</sup> | 29 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Halo 4 | Main menu | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Halo 4 (campaign) | Mission Dawn, the cryo bay | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | Cortana's face is black in the opening cutscene |
| Halo: Reach | Campaign, Winter Contingency | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Harry Potter and the Order of the Phoenix | Opening tutorial | – | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Hitman: Absolution | Prologue, the grounds of Diana's estate | 30 <sup>b</sup> | 30 <sup>d</sup> | – | 30 <sup>d</sup> |  |
| Hitman: Blood Money | Training mission, the pier | 30 <sup>b</sup> | 30 <sup>d</sup> | – | 30 <sup>d</sup> |  |
| The Incredible Hulk | First story scene | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | The Hulk isn't visible and doesn't respond in this scene; may be by design |
| Infinite Undiscovery | Opening scene in the dungeon | – | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Iron Man | First mission, Escape | – | 60 <sup>d</sup> | – | 60 <sup>b</sup> |  |
| JASF: Jane's Advanced Strike Fighters | Campaign mission 1, the jet on the runway | – | – | 30 <sup>c</sup> | 30 <sup>b</sup> |  |
| Juiced 2: Hot Import Nights | Career start, the driver intro before the first challenge | 108 <sup>c</sup> | 135 <sup>c</sup> | 82 <sup>b</sup> | – | Doesn't cap its own frame rate (see below) |
| Jumper: Griffin's Story | The courtyard fight tutorial | 60 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>e</sup> |  |
| Jurassic Park: The Game | Episode 1 opening | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Jurassic: The Hunted | First level, the jungle path | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | In the opening walk, a vertical seam in the lighting runs down the right of the picture |
| Just Cause 2 | Open world | 28 <sup>c</sup> | 27 <sup>c</sup> | 30 <sup>a</sup> | 30 <sup>d</sup> |  |
| Kameo: Elements of Power | Thorn's Castle | – | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Kane & Lynch 2: Dog Days | Shanghai opening | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>e</sup> |  |
| Kane & Lynch: Dead Men | The breakout, beside the van | – | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>e</sup> |  |
| The King of Fighters XII | Arcade fight | – | – | 60 <sup>b</sup> | 60 <sup>e</sup> |  |
| Kingdoms of Amalur: Reckoning | First control in the Well of Souls | – | – | 30 <sup>b</sup> | 30 <sup>c</sup> |  |
| Knights Contract | New game, the prologue in the plague village | – | – | 30 <sup>b</sup> | – |  |
| L.A. Noire | Upon Reflection, the patrol car | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| The Legend of Spyro: Dawn of the Dragon | First control in the arena, the Basic moves tutorial | – | – | 60 <sup>b</sup> | – |  |
| LEGO Batman: The Videogame | Episode 1-1, Gotham street | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| LEGO Indiana Jones 2: The Adventure Continues | The college hub | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| LEGO Indiana Jones: The Original Adventures | The Lost Temple, first control | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| LEGO Star Wars II | Mos Eisley Cantina | 60 <sup>c</sup> | 60 <sup>c</sup> | 59 <sup>b</sup> | 60 <sup>d</sup> |  |
| Lost Planet: Colonies | Mission 1, the hangar | 43 <sup>b</sup> | 30 <sup>d</sup> | 49 <sup>d</sup> | 30 <sup>d</sup> | A late beta of the game. Runs above 30 on the M4 mini and M1 Max, not yet explained. Rarely crashes in play |
| Madden NFL 10 | The coin toss, then the first play call or the kickoff | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | The test ends on the play call (30) or the kickoff (60), depending on the coin toss |
| Madden NFL 13 | Play Now, CPU vs CPU | 48 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Madden NFL 15 | Play Now, CPU vs CPU | – | 30 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Madden NFL 17 | Play Now, CPU vs CPU | 60 <sup>a</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Madden NFL 25 | Play Now, CPU vs CPU | 60 <sup>a</sup> | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Mafia II | Story opening | 137 <sup>b</sup> | 163 <sup>c</sup> | 101 <sup>c</sup> | 180 <sup>d</sup> | Doesn't cap its own frame rate (see below). Rarely crashes in play |
| Major League Baseball 2K13 | Demo game, CPU vs CPU | 60 <sup>b</sup> | – | 56 <sup>d</sup> | 59 <sup>d</sup> | Dips to about 30 in some camera shots |
| Major League Baseball 2K7 | Demo game, CPU vs CPU | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | Dips to 30 in the busiest camera shots |
| Marvel: Ultimate Alliance | First mission, the Helicarrier deck | 31 <sup>d</sup> | 31 <sup>b</sup> | 31 <sup>d</sup> | 31 <sup>d</sup> |  |
| Marvel: Ultimate Alliance 2 | First mission, the bridge fight in Latveria | – | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Mass Effect | First conversation | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Mass Effect 3 | Prologue, the Reapers reach Earth | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>a</sup> |  |
| Max Payne 3 | Chapter 1, the bullet-time tutorial | 33 <sup>c</sup> | 33 <sup>b</sup> | 33 <sup>c</sup> | – |  |
| Medal of Honor: Warfighter | First mission, the harbour in Karachi | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | Can crash about 25 seconds into the first mission |
| Mercenaries 2: World in Flames | New game, the speedboat off the coast | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Metal Gear Solid HD Collection | Metal Gear Solid 3, Virtuous Mission, the jungle at first control | – | 60 <sup>d</sup> | – | 60 <sup>b</sup> | Disc 1 of the collection |
| Metal Gear Solid V: Ground Zeroes | Camp Omega, the cliff at first control | – | – | 57 <sup>d</sup> | 60 <sup>d</sup> | Some textures show wrong colours or coloured noise, Snake's suit among them. Not yet measured on the M4 Macs: their tests stop short of this scene |
| Metal Gear Solid V: The Phantom Pain | Prologue, the hospital ward at first control | 60 <sup>c</sup> | 58 <sup>c</sup> | 60 <sup>b</sup> | – | Runs from Disc 2; Disc 1 is read from the image beside it |
| Metro 2033 | Prologue, first control | 30 <sup>b</sup> | – | 29 <sup>d</sup> | 30 <sup>d</sup> |  |
| Midnight Club: Los Angeles | Free roam | – | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Mirror's Edge | Story, first chapter | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>a</sup> |  |
| Mortal Kombat vs. DC Universe | Practice mode | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Mortal Kombat: Komplete Edition | Practice mode | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| MotoGP 08 | Quick Race, Losail at night, on the grid | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| MX vs. ATV Reflex | Career, the first race start | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| MX vs. ATV Supercross | Single race, Anaheim | – | 30 <sup>b</sup> | – | 30 <sup>d</sup> |  |
| NASCAR 09 | On the grid | – | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>e</sup> |  |
| NBA 2K11 | The Jordan intro game, Lakers at Bulls | 60 <sup>b</sup> | 60 <sup>d</sup> | 17 <sup>d</sup> | 60 <sup>d</sup> |  |
| NBA 2K14 | Quick Game, CPU vs CPU | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| NBA 2K18 | CPU vs CPU game | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| NBA Ballers: Chosen One | 1-on-1 game | 60 <sup>d</sup> | 60 <sup>b</sup> | 59 <sup>d</sup> | 60 <sup>d</sup> | A built-in game patch fixes the game's own loading hang |
| NBA Live 09 | Practice gym | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | Can crash loading the practice gym if buttons are pressed during the opening movies |
| NBA Street Homecourt | Venice Beach game | 58 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | Can stall on the EA SPORTS BIG logo at start-up |
| NCAA Football 14 | Exhibition, the opening coin toss | 30 <sup>b</sup> | 30 <sup>c</sup> | – | – |  |
| Need for Speed: Carbon | Quick Race, the grid of a circuit race | 60 <sup>b</sup> | – | – | 60 <sup>c</sup> |  |
| Need for Speed: Most Wanted | Quick Race, Ironwood Estates, on the start line | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>a</sup> |  |
| Need for Speed: ProStreet | Race Day, Portland, on the grid | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Need for Speed: The Run | Stage 1, the car-crusher escape | 30 <sup>d</sup> | 30 <sup>d</sup> | 29 <sup>b</sup> | 30 <sup>a</sup> | Sometimes stops in this opening scene |
| NHL 10 | Play Now, CPU vs CPU | – | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| NHL 15 | Play Now, CPU vs CPU | 60 <sup>a</sup> | 60 <sup>d</sup> | 60 <sup>a</sup> | 60 <sup>d</sup> | Replays and goal cut-ins run at 30 |
| Ninja Gaiden II | First fight, the shop balcony in Tokyo | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| The Outfit | Mission 1, the Allied base | 29 <sup>b</sup> | 29 <sup>d</sup> | 29 <sup>d</sup> | 29 <sup>d</sup> |  |
| Overlord | New game, the tower pit | 30 <sup>b</sup> | 30 <sup>d</sup> | – | 30 <sup>d</sup> |  |
| Overlord II | New game, the snowy woods of Nordberg | – | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Painkiller: Hell & Damnation | First level, the cemetery gate | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Perfect Dark Zero | Mission 0, Datacore | – | 29 <sup>b</sup> | – | 30 <sup>d</sup> |  |
| Persona 4 Arena | Training mode | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Pinball Hall of Fame: The Williams Collection | Medieval Madness table, first ball | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| Portal 2 | Relaxation Vault | – | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>a</sup> |  |
| Prince of Persia: The Forgotten Sands | Opening | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Project Gotham Racing 3 | Street race in Tokyo, on the grid | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Project Sylpheed: Arc of Deception | Stage 1, the asteroid field | – | – | 60 <sup>c</sup> | 60 <sup>b</sup> |  |
| Pure | Tutorial ride | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Quantum Theory | First control in the Tower | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Rage | Campaign opening | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Rayman Legends | Teensies in Trouble | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Rayman Origins | Opening clearing | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| Red Dead Redemption | Opening cutscene | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | Disc 1 of the Game of the Year Edition |
| Red Dead Redemption: Undead Nightmare | Opening cinematic | – | 30 <sup>b</sup> | – | 30 <sup>d</sup> | Disc 2 of the Game of the Year Edition |
| Resident Evil 5 | Chapter 1-1 | 29 <sup>c</sup> | 27 <sup>c</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Ridge Racer 6 | Single race | – | 60 <sup>b</sup> | – | 60 <sup>d</sup> |  |
| Ridge Racer Unbounded | Domination race | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Rio | Story mode, the first minigame | 60 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Rise of the Tomb Raider | Ascension, the mountain | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>e</sup> | The white flashes and rings in play are fixed. A single false-colour frame can still flash by now and then on M4 and newer Macs (about twice a minute on an M4 Pro). Only the opening mountain has been checked |
| Rock Band | Main menu | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> | Every mode asks for a microphone when played with a controller, so the test stops at the menu |
| Rockstar Games Presents Table Tennis | Training, the serve drill | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| Rogue Warrior | First level, the night forest | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Saints Row | Campaign opening, outside the church | 29 <sup>d</sup> | 28 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Saints Row 2 | First mission, the prison infirmary | 29 <sup>b</sup> | 28 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Saints Row IV | Opening mission, the bunker tunnel | 63 <sup>c</sup> | 111 <sup>c</sup> | 51 <sup>b</sup> | 119 <sup>d</sup> | Doesn't cap its own frame rate (see below) |
| Shadowrun | Training, chapter 1 | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Sid Meier's Civilization Revolution | Play Now, the first turn | – | 60 <sup>b</sup> | 45 <sup>d</sup> | 60 <sup>d</sup> |  |
| Silent Hill: Homecoming | Opening nightmare, the operating room | 28 <sup>d</sup> | 28 <sup>d</sup> | 29 <sup>b</sup> | 29 <sup>d</sup> | The gurney mattress shows a blocky green and magenta texture |
| Singularity | Katorga-12, the pier | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Skate 2 | Skating | 30 <sup>a</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Skate 3 | Free skate, the first plaza | 60 <sup>a</sup> | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Skate. | Freeskate, Downtown library plaza | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| Snoopy's Grand Adventure | New game, the backyard | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Sonic Generations | First stage | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | 30 <sup>d</sup> |  |
| Sonic the Hedgehog (2006) | Sonic's story, the town of Soleanna | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Sonic Unleashed | Windmill Isle, Act 1 | 30 <sup>b</sup> | – | 30 <sup>d</sup> | – |  |
| Space Chimps | First level | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Spider-Man: Friend or Foe | The Helicarrier hub | – | 60 <sup>b</sup> | – | 60 <sup>d</sup> |  |
| Spider-Man: Web of Shadows | Manhattan, after the opening | 30 <sup>b</sup> | 30 <sup>d</sup> | – | 30 <sup>d</sup> |  |
| SSX | World Tour, the first drop | 30 <sup>a</sup> | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Star Trek | First corridor as Kirk | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Star Trek: Legacy | Skirmish battle | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| Stranglehold | Hong Kong marketplace | 30 <sup>b</sup> | 30 <sup>d</sup> | 32 <sup>d</sup> | 32 <sup>e</sup> | Runs a little above its 30 limit, not yet explained |
| Surf's Up | Championship, the first surf run | 30 <sup>e</sup> | 30 <sup>e</sup> | – | – | The M1 Max test stops short of this scene |
| Tales of Vesperia | Zaphias, the Lower Quarter | 30 <sup>b</sup> | 30 <sup>c</sup> | 30 <sup>c</sup> | – |  |
| Tekken Tag Tournament 2 | Practice mode | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>b</sup> | VS screens show one name over all four portraits. Can stop during a load (rare) |
| Test Drive Unlimited | O'ahu, the first rental car | – | – | 30 <sup>b</sup> | – |  |
| Tetris Evolution | Marathon game | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Tiger Woods PGA Tour 06 | Play Now, the 1st tee | 53 <sup>d</sup> | 52 <sup>d</sup> | 54 <sup>b</sup> | 55 <sup>d</sup> |  |
| Tiger Woods PGA Tour 08 | Play Now, the 1st tee | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Tiger Woods PGA Tour 13 | Augusta, the 1st tee | – | – | 30 <sup>b</sup> | 30 <sup>a</sup> |  |
| Tiger Woods PGA Tour 14 | Augusta, the 1st tee | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>a</sup> | Some menu text is missing letters. Can crash about 10 minutes in |
| TMNT | Mystical Jungle, first level | 60 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Tomb Raider: Anniversary | Peru, the mountain caves | 30 <sup>b</sup> | 30 <sup>d</sup> | – | 30 <sup>d</sup> |  |
| Tomb Raider: Legend | Bolivia, the jungle path | – | 30 <sup>b</sup> | – | 30 <sup>d</sup> |  |
| Tony Hawk's American Wasteland | Free Skate, Hollywood | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Tony Hawk's Project 8 | Free Skate, first level | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Ultra Street Fighter IV | Arcade fight | 60 <sup>b</sup> | 60 <sup>c</sup> | 60 <sup>c</sup> | 60 <sup>d</sup> | Rarely crashes while starting |
| Vanquish | Chapter 1, the opening tutorial aboard the ship | 22 <sup>b</sup> | – | – | – | Runs at about 21, below its 30 limit |
| Virtua Tennis 4 | Exhibition match, first serve | – | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Viva Piñata | Garden | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| WALL-E | First level | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Wanted: Weapons of Fate | Combat training, target practice | – | 60 <sup>b</sup> | – | 60 <sup>d</sup> |  |
| Warhammer 40,000: Space Marine | First fight with the orks | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Warriors Orochi 3 | Free Mode, the camp | – | – | 30 <sup>b</sup> | 30 <sup>c</sup> |  |
| WET | First level, the tea house | 30 <sup>b</sup> | 30 <sup>d</sup> | – | 30 <sup>d</sup> |  |
| Wolfenstein | Train station, first control | – | 30 <sup>b</sup> | 30 <sup>d</sup> | – |  |
| Worms Collection | Worms, main menu | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| WWE SmackDown vs. Raw 2007 | Exhibition match, CPU vs CPU | 60 <sup>b</sup> | 60 <sup>d</sup> | – | 60 <sup>d</sup> |  |
| WWE SmackDown vs. Raw 2011 | One on One match, CPU vs CPU | – | 60 <sup>b</sup> | – | 60 <sup>d</sup> |  |
| X-Men: Destiny | First control in the wrecked street | – | 30 <sup>d</sup> | 30 <sup>b</sup> | – |  |

## Xbox Live Arcade

54 Xbox Live Arcade games are tested too, in the same way, with the same
columns and the same build letters.

| Game | Test reaches | M4 mini | M4 Pro | M1 Max | M5 Max | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Awesomenauts | First-run tutorial, Lonestar at his base | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Blade Kitten | First control, after the opening cutscene | 30 <sup>d</sup> | 30 <sup>d</sup> | 30 <sup>b</sup> | 30 <sup>d</sup> | Half the screen can flash white in the opening cutscene |
| Bliss Island | Adventure game, the Fruit Smoothie mini-game | 60 <sup>b</sup> | – | – | – | Tested as the trial |
| Brain Challenge | Brain Test, the Balance rule demo | 60 <sup>b</sup> | – | 60 <sup>c</sup> | – | Tested as the trial |
| Brothers: A Tale of Two Sons | The village lane at dusk | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Call of Duty Classic | Camp Toccoa, the training ground | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Call of Juarez: Gunslinger | Episode 1, the road by the ruined barn | 30 <sup>e</sup> | 30 <sup>e</sup> | – | 30 <sup>b</sup> |  |
| Castle Crashers | The castle bar, after the opening cutscene | 55 <sup>b</sup> | 56 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | Runs a little under its 60 limit, not yet explained |
| DeathSpank: Thongs of Virtue | The prison camp kitchen | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Earthworm Jim HD | New Junk City, the start of the level | 57 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Ecco the Dolphin | Home Bay, the first level | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Flashback | The Titan jungle, by the waterfall | 30 <sup>b</sup> | 30 <sup>e</sup> | – | 30 <sup>d</sup> |  |
| Frogger | The arcade game, the frog waiting on the bank | 60 <sup>b</sup> | – | – | – | Tested as the trial |
| Frogger 2 | Story mode, the first jungle island | 60 <sup>c</sup> | – | – | 60 <sup>b</sup> | Tested as the trial |
| Frogger: Hyper Arcade Edition | Tile Capture, a party game against three CPU frogs | 214 <sup>c</sup> | – | 167 <sup>b</sup> | – | Tested as the trial. Doesn't cap its own frame rate (see below) |
| From Dust | The first scene, the tribe on the night shore | – | 32 <sup>b</sup> | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Guardian Heroes | Story mode, the tavern at the start of stage 1 | 60 <sup>c</sup> | – | 60 <sup>b</sup> | – | Tested as the trial |
| Guardians of Middle-Earth | Training match, first lesson card | 30 <sup>c</sup> | – | 30 <sup>b</sup> | – |  |
| Halo: Spartan Assault | Mission A-1, the outpost on an icy moon | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | Tested as the trial |
| Haunted House | The first room, Jacob on Spooky difficulty | – | 60 <sup>b</sup> | – | 60 <sup>c</sup> |  |
| Heavy Weapon: Atomic Tank | Mission 1, Tundra Flats | 60 <sup>b</sup> | – | – | – | Tested as the trial |
| Hydrophobia | Kate's apartment, first control | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| I Am Alive | First control on the ruined highway | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Ion Assault | Echelon Territory, stage 1 | 60 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| The King of Fighters '98 Ultimate Match | Arcade Play, the first fight | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| The King of Fighters 2002 Unlimited Match | Team Play, a CPU vs CPU team match | 60 <sup>b</sup> | – | – | – | Tested as the trial |
| KOF SKY STAGE | Pause menu over stage 1 | 60 <sup>b</sup> | – | – | – | Tested as the trial |
| Lara Croft and the Guardian of Light | The Temple of Light entrance, the first bomb prompt | 30 <sup>b</sup> | – | 30 <sup>c</sup> | – |  |
| Marvel Puzzle Quest: Dark Reign | The first match board | 30 <sup>b</sup> | – | 30 <sup>d</sup> | 30 <sup>d</sup> |  |
| Matt Hazard: Blood Bath and Beyond | The Hate Boat, the pool deck | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Mega Man 10 | Solar Man's stage, the first room | 60 <sup>c</sup> | – | – | 60 <sup>b</sup> | Tested as the trial |
| Mortal Kombat Arcade Kollection | Mortal Kombat, the attract loop | – | 54 <sup>b</sup> | – | 54 <sup>d</sup> | Runs at 54, the original arcade board's rate |
| NBA Jam | Play Now, Celtics vs Lakers in a live game | 60 <sup>c</sup> | – | 60 <sup>b</sup> | – | Tested as the trial |
| NBA Jam: On Fire Edition | Jam Now, a game in the Heat arena | 60 <sup>b</sup> | – | – | 60 <sup>d</sup> |  |
| Orcs Must Die! | The Hallway, the first level | 30 <sup>b</sup> | – | – | 30 <sup>d</sup> |  |
| PAC-MAN Championship Edition | Championship Mode, the replay after the run | 60 <sup>b</sup> | – | – | – | Tested as the trial |
| Prison Architect: Xbox 360 Edition | Prison Stories chapter 1, the canteen | 30 <sup>c</sup> | – | 30 <sup>b</sup> | – | Tested as the trial |
| R-Type Dimensions | R-Type stage 1, in 3D | – | 60 <sup>b</sup> | – | 60 <sup>d</sup> |  |
| Rez HD | Pause menu over Area 1 | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| RISK | World Domination, a game between three computer players | 60 <sup>c</sup> | – | – | – | Tested as the trial |
| RoboBlitz | Karl's workshop, first control | – | – | 30 <sup>b</sup> | 30 <sup>d</sup> |  |
| Sam & Max: Beyond Time and Space | Episode 1, first control in the office | 60 <sup>b</sup> | – | – | – | Tested as the trial |
| Scarygirl | Scarygirl's tree house, first control | 30 <sup>c</sup> | – | 30 <sup>b</sup> | – | Tested as the trial |
| SEGA Bass Fishing | Arcade Mode, the Lodge | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Serious Sam 3: BFE | Broken Wings demo, the Cairo park | 16 <sup>b</sup> | 14 <sup>d</sup> | 6 <sup>d</sup> | 11 <sup>d</sup> | Tested as the trial. The buildings behind the park can render solid black, and the level runs at about 15 on the M4 Macs and 6 on the M1 Max |
| Snoopy Flying Ace | Mission 1, the Frozen Fjord ring race | 31 <sup>c</sup> | – | 31 <sup>b</sup> | – | Tested as the trial |
| Sonic & Knuckles | Mushroom Hill Zone, act 1 | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| SoulCalibur II HD Online | Arcade fight, Mitsurugi vs Cassandra | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Super Street Fighter II Turbo HD Remix | Training mode, Ryu vs Ken | – | – | 60 <sup>b</sup> | 60 <sup>d</sup> |  |
| Teenage Mutant Ninja Turtles 1989 Arcade | Pause menu over Scene 1 | 60 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Trials Evolution | First career event, on the start ramp | 60 <sup>b</sup> | – | 60 <sup>d</sup> | 60 <sup>d</sup> |  |
| Trials Fusion | Training Program 1, at the start line | 60 <sup>d</sup> | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> | A flat brown patch covers part of the track, and white specks show along some edges |
| Trials HD | Tutorial track, at the start gate | – | – | 60 <sup>b</sup> | – |  |
| Virtua Fighter 2 | Pause menu over the first Arcade fight | – | 60 <sup>b</sup> | 60 <sup>d</sup> | 60 <sup>d</sup> |  |

Every figure in the tables comes from a test where the game was still running
at the end and hit no GPU error. Notes list the problems still open in these games;
problems fixed before these runs are not listed.

Mafia II, Saints Row IV, 50 Cent: Blood on the Sand, Juiced 2: Hot Import
Nights, Blood Bowl, AFL Live, Dark, Darkest of Days, Darksiders and Darksiders
II don't limit their own frame rate, so they run as fast as the Mac allows, and the table measures them that way. To hold them at 60, set **Uncapped games** to **60 FPS** in the
Graphics settings. Frogger: Hyper Arcade Edition doesn't limit its frame rate
either. The game-select screen of Worms Collection is uncapped in
the same way, but the Worms menu the test measures holds 60.

A game not listed here may still work; these are only the ones tested on every
build. If one of them, or another game, misbehaves, see
[Troubleshooting](troubleshooting.md).
