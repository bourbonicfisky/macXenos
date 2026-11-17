# Game compatibility

These are the games macXenos is tested with regularly: 228 of them. Each one
is started automatically, played to the point shown with a scripted
controller, and measured for a minute. The M4 mini, M4 Pro and M1 Max columns
are from macXenos 1.0.1-rc2 (build 202611152108), run on November 15 and 16,
2026, at the original resolution (1x), using the Metal renderer. The Macs are
a base M4 Mac mini, an M4 Pro Mac mini and an M1 Max MacBook Pro with 32 GB.
The games were split between them, so many ran on only one; a dash means that
Mac didn't run the game in this round. The M4 mini column is partial for now:
its run stopped part-way and will be completed. A few rows no Mac measured in
this round keep figures from earlier builds and are marked "(earlier build)".
The M5 Max column is from an M5 Max Mac Studio, in runs between November 14
and 16 (most of them with the 1.0.1 release candidate) that are still filling
the column in.

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

| Game | Test reaches | M4 mini · rc2 | M4 Pro · rc2 | M1 Max · rc2 | M5 Max | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 007 Legends | Opening level | – | 60 | – | 60 |  |
| 007: Blood Stone | Athens harbour, in the speedboat | – | – | 30 | 30 |  |
| 007: Quantum of Solace | First level | – | 30 | – | 30 |  |
| 50 Cent: Blood on the Sand | Mission 1, the opening street | – | – | 118 | 128 | Doesn't cap its own frame rate (see below) |
| Ace Combat 6: Fires of Liberation | Mission 1, the take-off sequence | – | – | 30 | 30 |  |
| Ace Combat: Assault Horizon | Mission 1, over Miami | – | – | 30 | 30 |  |
| Afro Samurai | Village tutorial fight | – | – | 30 | 30 |  |
| Alan Wake | Opening nightmare on the mountain road | – | – | 30 | 30 |  |
| Assassin's Creed | The Animus tutorial crowd | – | 30 | 30 | – |  |
| Assassin's Creed II | Story mode opening | – | 30 | 30 | 30 |  |
| Assassin's Creed III | Opening, Desmond at the cave entrance | – | 60 | 60 | 60 | This scene runs at 60 |
| Banjo-Kazooie: Nuts & Bolts | Spiral Mountain | 30 | 30 | 30 | 30 | Occasionally freezes at the boot logo or after the menu loads |
| Batman: Arkham Asylum | Intensive Treatment, escorting the Joker | 30 | 30 | 30 | 30 |  |
| Batman: Arkham Origins | Blackgate rooftop | – | 30 | 30 | 30 |  |
| Battlefield: Bad Company | First level | – | – | 30 | – |  |
| Bayonetta | Prologue, the clock tower fight | – | – | 60 | 60 |  |
| Beautiful Katamari | Rolling practice | – | 60 | – | 60 |  |
| Bionic Commando | First level | – | – | 60 | 60 |  |
| BioShock | The plane crash, first control | 30 | 30 | 30 | 30 |  |
| BioShock 2 | Adonis Luxury Resort | – | 30 | – | 30 |  |
| Blitz: The League II | Pause menu over the opening kickoff | – | – | 43 | 60 | Can stop during start-up (rare). Not measured on the M4 Macs in this round: their tests didn't reach this scene |
| Brothers in Arms: Hell's Highway | First chapter, the hospital | – | – | 30 | 30 |  |
| Bully: Scholarship Edition | Bullworth campus entrance | 60 | 60 | 60 | 60 |  |
| Burnout Paradise | Driving in Paradise City | – | 59 | 60 | 60 |  |
| Burnout Revenge | First World Tour event | – | 60 | 60 | 60 |  |
| Cabela's Big Game Hunter 2010 | First hunt, the forest trail | – | – | 30 | 30 |  |
| Call of Duty 2 | First mission | 60 | 60 | 60 | 60 |  |
| Call of Duty 3 | Saint-Lô, the training yard | – | 60 | – | 60 |  |
| Call of Duty 4: Modern Warfare | First mission, the hangar | – | 60 | 60 | 60 |  |
| Call of Duty: Black Ops | Operation 40, the bar in Havana | – | 60 | – | 60 |  |
| Call of Duty: Black Ops II | First mission, Pyrrhic Victory | 60 | 60 | 60 | 60 | The main menu background sometimes renders almost white |
| Call of Duty: Ghosts | Local play, Team Deathmatch on Prison Break | 60 | 60 | 60 | 60 | Needs a profile that has played the game once (its brightness and margin screens done) |
| Call of Duty: Modern Warfare 2 | First mission, the shooting range | – | – | 60 | 60 |  |
| Call of Duty: Modern Warfare 3 | First mission, Black Tuesday | – | 60 | – | 60 |  |
| Call of Juarez: Bound in Blood | Chapter 1, the trenches | – | – | 60 | 60 |  |
| Captain America: Super Soldier | The trench at first control | 60 | 60 | 60 | 60 | Occasionally starts to a black screen (one start in four in testing); starting the game again gets past it |
| Chromehounds | First training sortie | – | 60 | – | 60 |  |
| The Chronicles of Riddick: Assault on Dark Athena | Butcher Bay, the cell block | – | 30 | – | 30 |  |
| Clash of the Titans: The Videogame | First control on the beach | – | 30 | – | 30 |  |
| Clive Barker's Jericho | Opening scene in the transport | – | – | 60 | 60 |  |
| Crysis 2 | First mission | 31 | 31 | 31 | 31 |  |
| Crysis 3 | First mission, inside the Liberty Dome | – | 31 | 31 | – |  |
| Dark Souls | Undead Asylum | – | 30 | 30 | – |  |
| Dark Souls II | Things Betwixt, the first glade | – | 60 | 60 | 60 |  |
| Dead or Alive 4 | Fight (watch mode) | 60 | 60 | 60 | 60 |  |
| Dead or Alive 5 | Training fight | 60 | 60 | 60 | 60 |  |
| Dead Space | Opening cutscene, the Kellion's cockpit | – | 30 | 30 | 30 |  |
| Dead to Rights: Retribution | Prologue | 30 | 30 | 30 | – |  |
| Def Jam: Icon | Exhibition fight | – | – | 30 | 30 |  |
| Deus Ex: Human Revolution | Sarif Industries, Megan's lab | – | – | 30 | 30 |  |
| Devil May Cry 4 | Prologue | 30 | 30 | 30 | 30 |  |
| Diablo III | First area, the night road | – | – | 60 | 60 |  |
| Dishonored | New game opening | – | 30 | 30 | – |  |
| Doom 3: BFG Edition | Mars City hangar | – | 60 | 60 | 60 |  |
| Dragon Age II | Prologue battle | 30 | 30 | 30 | 30 |  |
| Dragon Age: Origins | Human Noble origin, Castle Highever | 30 | 30 | 30 | 30 |  |
| Dragon Age: Origins - Awakening | The road to Vigil's Keep | 30 | 30 | 30 | 30 |  |
| Dynasty Warriors 6 | Challenge mode, Rampage | – | 60 | – | 60 |  |
| Dynasty Warriors 7 | Wei story, chapter 1 | – | – | 60 | 60 |  |
| The Elder Scrolls IV: Oblivion | Imperial Prison cell | – | – | 30 | 30 |  |
| The Elder Scrolls V: Skyrim | Opening cart ride to Helgen | 30 | 30 | 30 | 30 | Sometimes crashes or stops responding in play; being worked on |
| Enchanted Arms | Opening battle | – | 30 | – | 30 |  |
| F.E.A.R. | Instant Action, Distribution | – | 30 | 30 | 30 |  |
| F1 2011 | Time Trial, Melbourne | – | – | 30 | 30 |  |
| Fable II | Childhood, Bowerstone at night | – | 30 | 29 | 30 |  |
| Fable III | New game, the Prince's bedroom | 30 | 30 | 30 | – |  |
| Fallout: New Vegas | Doc Mitchell's house | – | 30 | – | 30 |  |
| Far Cry 2 | Opening taxi ride | – | 30 | 30 | 30 |  |
| Far Cry 3 | Opening (the cage) | 30 | 30 | 30 | 30 | At 2x internal resolution a magenta band can smear down from the top of the picture |
| Far Cry 4 | Opening, dinner with Pagan Min | 30 | 30 | 30 | – |  |
| FIFA 14 | Kick Off match, CPU vs CPU | – | – | 60 | 60 |  |
| FIFA 16 | Kick Off match, CPU vs CPU | 60 | 60 | 60 | 60 |  |
| FIFA Street 3 | Play Now match | – | 60 | 60 | 60 | Can stall on the EA SPORTS BIG logo at start-up |
| Fight Night Champion | Fight Now, Tyson vs Ali, CPU vs CPU | 30 | 30 | 30 | 30 | Occasionally the ring ropes show wrong colours for a whole bout |
| Fight Night Round 4 | Fight Now, Ali vs Tyson, round 1 | 60 | 60 | 60 | 60 | Boxers' skin can show a foil-like sparkle for a whole session |
| Final Fantasy XIII | The Hanging Edge, after the first battle | 30 | 30 | 30 | – | After a Game Over and Retry in the first battle, Lightning's skin and the walkway can render wrongly |
| Fist of the North Star: Ken's Rage | Legend Mode, first chapter fight | – | 30 | – | 30 |  |
| Forza Horizon | Opening drive | – | 30 | 30 | 30 |  |
| Forza Horizon 2 | Drive to the Festival | 30 | 30 | 30 | 30 |  |
| Forza Motorsport 2 | Exhibition race, on the grid | 60 | 60 | 60 | 60 | Some menu text is missing letters |
| Forza Motorsport 4 | First race, cockpit view | 60 | 60 | 60 | 60 | In cockpit view the windshield and side windows are flat grey |
| Frontlines: Fuel of War | First mission, defending the crash site | – | 47 | – | 49 | The game paces itself at about 46-50, in menus and videos too |
| Fuel | Career, The Leap of Faith | – | – | 30 | 30 |  |
| Gears of War | Campaign, Ashes | 30 | 30 | 30 | 30 |  |
| Gears of War 2 | Campaign, Act I | 30 | 30 | 30 | 30 |  |
| Gears of War 3 | Campaign prologue | 30 | 30 | 30 | 30 |  |
| Gears of War: Judgment | Survival, Estate | – | 30 | 30 | 30 |  |
| Ghost Recon Advanced Warfighter | Training mission, first street | 60 | – | – | 60 | Not measured on the M4 Pro and M1 Max in this round: their tests didn't reach this scene |
| The Godfather II | Havana rooftop party | – | 30 | – | 30 |  |
| GoldenEye 007: Reloaded | First mission | – | 60 | – | 60 |  |
| Grand Theft Auto IV | Opening drive | – | 60 | 60 | 60 | Now holds the console's 60 refresh rate (54-55 in October) |
| Guitar Hero II | Practice, a full song | – | – | 60 | 60 |  |
| Half-Life 2 (The Orange Box) | Main menu over its live map | 30 | 30 | 30 | 30 | Starting the first chapter can stop at the loading screen |
| Halo 3 | Campaign, Arrival | 30 | 30 | 30 | 30 |  |
| Halo 3: ODST | Campaign, Prepare to Drop | 29 | 28 | 30 | 30 |  |
| Halo 4 | Main menu | 30 | 30 | 30 | 30 |  |
| Halo 4 (campaign) | Mission Dawn, the cryo bay | 30 | 30 | 30 | 30 | Cortana's face is black in the opening cutscene |
| Halo: Reach | Campaign, Winter Contingency | – | 30 | 30 | 30 |  |
| Harry Potter and the Order of the Phoenix | Opening tutorial | – | – | 30 | 30 |  |
| Hitman: Absolution | Prologue, the grounds of Diana's estate | – | 30 | – | 30 |  |
| Hitman: Blood Money | Training mission, the pier | – | 30 | – | 30 |  |
| The Incredible Hulk | First story scene | 30 | 30 | 30 | 30 | The Hulk isn't visible and doesn't respond in this scene; may be by design |
| Infinite Undiscovery | Opening scene in the dungeon | – | – | 30 | 30 |  |
| Iron Man | First mission, Escape | – | 60 | – | 60 |  |
| Jumper: Griffin's Story | The courtyard fight tutorial | – | – | 60 | 60 |  |
| Jurassic Park: The Game | Episode 1 opening | – | – | 60 | 60 |  |
| Jurassic: The Hunted | First level, the jungle path | 30 | 30 | 30 | 30 | In the opening walk, a vertical seam in the lighting runs down the right of the picture |
| Just Cause 2 | Open world | 27 | 28 | 30 | 30 |  |
| Kameo: Elements of Power | Thorn's Castle | – | – | 30 | 30 |  |
| Kane & Lynch 2: Dog Days | Shanghai opening | – | 60 | – | 60 |  |
| Kane & Lynch: Dead Men | The breakout, beside the van | – | – | 30 | 30 |  |
| The King of Fighters XII | Arcade fight | – | – | 60 | 60 |  |
| L.A. Noire | Upon Reflection, the patrol car | – | 30 | 30 | 30 |  |
| LEGO Batman: The Videogame | Episode 1-1, Gotham street | – | 60 | – | 60 |  |
| LEGO Indiana Jones 2: The Adventure Continues | The college hub | 60 | 60 | 60 | 60 |  |
| LEGO Indiana Jones: The Original Adventures | The Lost Temple, first control | 60 | 60 | 60 | 60 |  |
| LEGO Star Wars II | Mos Eisley Cantina | 60 | 60 | 60 | 60 |  |
| Lost Planet: Colonies | Mission 1, the hangar | 47 | 30 | 49 | 30 | A late beta of the game. Runs above 30 on the M4 mini and M1 Max, not yet explained. Rarely crashes in play |
| Madden NFL 10 | The coin toss, then the first play call or the kickoff | 30 | 60 | 60 | 60 | The test ends on the play call (30) or the kickoff (60), depending on the coin toss |
| Madden NFL 13 | Play Now, CPU vs CPU | 48 | 60 | 60 | 60 |  |
| Madden NFL 15 | Play Now, CPU vs CPU | – | – | 60 | 60 |  |
| Madden NFL 17 | Play Now, CPU vs CPU | 60 | 60 | 60 | 60 |  |
| Madden NFL 25 | Play Now, CPU vs CPU | 60 | 60 | 60 | 60 |  |
| Mafia II | Story opening | 120 | 164 | 118 | 180 | Doesn't cap its own frame rate (see below). Rarely crashes in play |
| Major League Baseball 2K13 | Demo game, CPU vs CPU | – | – | 56 | 59 | Dips to about 30 in some camera shots |
| Major League Baseball 2K7 | Demo game, CPU vs CPU | – | 60 | – | 60 | Dips to 30 in the busiest camera shots |
| Marvel: Ultimate Alliance | First mission, the Helicarrier deck | 31 | 31 | 31 | 31 |  |
| Marvel: Ultimate Alliance 2 | First mission, the bridge fight in Latveria | – | – | 30 | 30 |  |
| Mass Effect | First conversation | 30 | 30 | 30 | 30 |  |
| Mass Effect 3 | Prologue, the Reapers reach Earth | – | – | 30 | – |  |
| Max Payne 3 | Chapter 1, the bullet-time tutorial | – | – | 33 | – |  |
| Medal of Honor: Warfighter | First mission, the harbour in Karachi | 30 | 30 | 30 | 30 | Can crash about 25 seconds into the first mission |
| Mercenaries 2: World in Flames | New game, the speedboat off the coast | – | 60 | – | 60 |  |
| Metal Gear Solid HD Collection | Metal Gear Solid 3, Virtuous Mission, the jungle at first control | – | 60 | – | 60 | Disc 1 of the collection |
| Metal Gear Solid V: Ground Zeroes | Camp Omega, the cliff at first control | – | – | 57 | 60 | Some textures show wrong colours or coloured noise, Snake's suit among them. Not yet measured on the M4 Macs: their tests stop short of this scene |
| Metro 2033 | Prologue, first control | – | – | 29 | 30 |  |
| Midnight Club: Los Angeles | Free roam | – | – | 30 | 30 |  |
| Mirror's Edge | Story, first chapter | – | 30 | 30 | 30 |  |
| Mortal Kombat vs. DC Universe | Practice mode | – | 60 | – | 60 |  |
| Mortal Kombat: Komplete Edition | Practice mode | – | 60 | – | 60 |  |
| MotoGP 08 | Quick Race, Losail at night, on the grid | – | – | 30 | 30 |  |
| MX vs. ATV Reflex | Career, the first race start | 30 | 30 | 30 | 30 |  |
| MX vs. ATV Supercross | Single race, Anaheim | – | 30 | – | 30 |  |
| NASCAR 09 | On the grid | – | 30 | – | 30 |  |
| NBA 2K11 | The Jordan intro game, Lakers at Bulls | 60 | 60 | 17 | 60 |  |
| NBA 2K14 | Quick Game, CPU vs CPU | – | – | 60 | 60 |  |
| NBA 2K18 | CPU vs CPU game | – | 60 | – | 60 |  |
| NBA Ballers: Chosen One | 1-on-1 game | 60 | 60 | 59 | 60 | A built-in game patch fixes the game's own loading hang |
| NBA Live 09 | Practice gym | – | 60 | 60 | 60 | Can crash loading the practice gym if buttons are pressed during the opening movies |
| NBA Street Homecourt | Venice Beach game | 60 | 60 | 60 | 60 | Can stall on the EA SPORTS BIG logo at start-up |
| Need for Speed: Most Wanted | Quick Race, Ironwood Estates, on the start line | – | – | 60 | 60 |  |
| Need for Speed: ProStreet | Race Day, Portland, on the grid | – | – | 60 | 60 |  |
| Need for Speed: The Run | Stage 1, the car-crusher escape | 30 | 30 | 30 | 30 | Sometimes stops in this opening scene |
| NHL 10 | Play Now, CPU vs CPU | – | – | 60 | 60 |  |
| NHL 15 | Play Now, CPU vs CPU | – | 60 | – | 60 | Replays and goal cut-ins run at 30 |
| Ninja Gaiden II | First fight, the shop balcony in Tokyo | 60 | 60 | 60 | 60 |  |
| The Outfit | Mission 1, the Allied base | 29 | 29 | 29 | 29 |  |
| Overlord | New game, the tower pit | – | 30 | – | 30 |  |
| Overlord II | New game, the snowy woods of Nordberg | – | – | 30 | 30 |  |
| Painkiller: Hell & Damnation | First level, the cemetery gate | – | – | 30 | 30 |  |
| Perfect Dark Zero | Mission 0, Datacore | – | 29 | – | 30 |  |
| Persona 4 Arena | Training mode | – | – | 60 | 60 |  |
| Pinball Hall of Fame: The Williams Collection | Medieval Madness table, first ball | – | 60 | – | 60 |  |
| Portal 2 | Relaxation Vault | – | 30 | 30 | 30 |  |
| Prince of Persia: The Forgotten Sands | Opening | 30 | 30 | 30 | 30 |  |
| Project Gotham Racing 3 | Street race in Tokyo, on the grid | – | 30 | 30 | 30 |  |
| Pure | Tutorial ride | – | – | 30 | 30 |  |
| Quantum Theory | First control in the Tower | 30 | 30 | 30 | 30 |  |
| Rage | Campaign opening | – | 60 | 60 | 60 |  |
| Rayman Legends | Teensies in Trouble | – | 60 | 60 | 60 |  |
| Rayman Origins | Opening clearing | – | 60 | – | 60 |  |
| Red Dead Redemption | Opening cutscene | 30 | 30 | 30 | 30 | Disc 1 of the Game of the Year Edition |
| Red Dead Redemption: Undead Nightmare | Opening cinematic | – | 30 | – | 30 | Disc 2 of the Game of the Year Edition |
| Resident Evil 5 | Chapter 1-1 | 29 | 27 | 30 | 30 |  |
| Ridge Racer 6 | Single race | – | 60 | – | 60 |  |
| Ridge Racer Unbounded | Domination race | – | 30 | 30 | 30 |  |
| Rio | Story mode, the first minigame | – | – | 60 | 60 |  |
| Rise of the Tomb Raider | Ascension, the mountain | 30 | 30 | 30 | 30 | The white flashes and rings in play are fixed. A single false-colour frame can still flash by now and then on M4 and newer Macs (about twice a minute on an M4 Pro). Only the opening mountain has been checked |
| Rock Band | Main menu | – | – | 60 | 60 | Every mode asks for a microphone when played with a controller, so the test stops at the menu |
| Rockstar Games Presents Table Tennis | Training, the serve drill | – | 60 | – | 60 |  |
| Rogue Warrior | First level, the night forest | 30 | 30 | 30 | 30 |  |
| Saints Row | Campaign opening, outside the church | 29 | 28 | 30 | 30 |  |
| Saints Row 2 | First mission, the prison infirmary | – | 29 | – | 30 |  |
| Saints Row IV | Opening mission, the bunker tunnel | 57 | 95 | 48 | 119 | Doesn't cap its own frame rate (see below) |
| Shadowrun | Training, chapter 1 | 30 | 30 | 30 | 30 |  |
| Sid Meier's Civilization Revolution | Play Now, the first turn | – | – | 45 | 60 |  |
| Silent Hill: Homecoming | Opening nightmare, the operating room | 28 | 28 | 29 | 29 | The gurney mattress shows a blocky green and magenta texture |
| Singularity | Katorga-12, the pier | – | – | 30 | 30 |  |
| Skate 2 | Skating | 30 | 30 | 30 | 30 |  |
| Skate 3 | Free skate, the first plaza | – | – | 60 | 60 |  |
| Skate. | Freeskate, Downtown library plaza | – | 60 | – | 60 |  |
| Snoopy's Grand Adventure | New game, the backyard | – | – | 30 | 30 |  |
| Sonic Generations | First stage | 30 | 30 | 30 | 30 |  |
| Sonic the Hedgehog (2006) | Sonic's story, the town of Soleanna | – | – | 60 | 60 |  |
| Sonic Unleashed | Windmill Isle, Act 1 | – | – | 30 | – |  |
| Space Chimps | First level | – | – | 60 | 60 |  |
| Spider-Man: Friend or Foe | The Helicarrier hub | – | 60 | – | 60 |  |
| Spider-Man: Web of Shadows | Manhattan, after the opening | – | 30 | – | 30 |  |
| SSX | World Tour, the first drop | – | – | 30 | 30 |  |
| Star Trek | First corridor as Kirk | 30 | 30 | 30 | 30 |  |
| Star Trek: Legacy | Skirmish battle | – | 60 | – | 60 |  |
| Stranglehold | Hong Kong marketplace | 30 | 30 | 32 | 32 | Runs a little above its 30 limit, not yet explained |
| Surf's Up | Championship, the first surf run | 30 | 30 | – | – | (earlier build). The M1 Max test stops short of this scene |
| Tales of Vesperia | Zaphias, the Lower Quarter | – | 30 | – | – |  |
| Tekken Tag Tournament 2 | Practice mode | 60 | 60 | 60 | 60 | VS screens show one name over all four portraits. Can stop during a load (rare) |
| Test Drive Unlimited | O'ahu, the first rental car | – | – | 30 | – |  |
| Tetris Evolution | Marathon game | – | 60 | – | 60 |  |
| Tiger Woods PGA Tour 06 | Play Now, the 1st tee | 53 | 52 | 55 | 55 |  |
| Tiger Woods PGA Tour 08 | Play Now, the 1st tee | – | – | 30 | 30 |  |
| Tiger Woods PGA Tour 13 | Augusta, the 1st tee | – | – | 30 | 30 |  |
| Tiger Woods PGA Tour 14 | Augusta, the 1st tee | 30 | 30 | 30 | 30 | Some menu text is missing letters. Can crash about 10 minutes in |
| TMNT | Mystical Jungle, first level | – | – | 60 | 60 |  |
| Tomb Raider: Anniversary | Peru, the mountain caves | – | 30 | – | 30 |  |
| Tomb Raider: Legend | Bolivia, the jungle path | – | 30 | – | 30 |  |
| Tony Hawk's American Wasteland | Free Skate, Hollywood | – | – | 60 | 60 |  |
| Tony Hawk's Project 8 | Free Skate, first level | 60 | 60 | 60 | 60 |  |
| Ultra Street Fighter IV | Arcade fight | – | 60 | 60 | 60 | Rarely crashes while starting |
| Virtua Tennis 4 | Exhibition match, first serve | – | 60 | – | 60 |  |
| Viva Piñata | Garden | 30 | 30 | 30 | 30 |  |
| WALL-E | First level | – | – | 60 | 60 |  |
| Wanted: Weapons of Fate | Combat training, target practice | – | 60 | – | 60 |  |
| Warhammer 40,000: Space Marine | First fight with the orks | – | – | 30 | 30 |  |
| WET | First level, the tea house | – | 30 | – | 30 |  |
| Wolfenstein | Train station, first control | – | – | 30 | – |  |
| Worms Collection | Worms, main menu | 60 | 60 | 60 | 60 |  |
| WWE SmackDown vs. Raw 2007 | Exhibition match, CPU vs CPU | – | 60 | – | 60 |  |
| WWE SmackDown vs. Raw 2011 | One on One match, CPU vs CPU | – | 60 | – | 60 |  |
| X-Men: Destiny | First control in the wrecked street | – | 30 | – | – |  |

## Xbox Live Arcade

33 Xbox Live Arcade games are tested too, in the same way and in the same
runs. Two that no Mac measured in this round keep figures from earlier builds
(November 2 to 15).

| Game | Test reaches | M4 mini · rc2 | M4 Pro · rc2 | M1 Max · rc2 | M5 Max | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Awesomenauts | First-run tutorial, Lonestar at his base | – | – | 60 | 60 |  |
| Blade Kitten | First control, after the opening cutscene | 30 | 30 | 30 | 30 | Half the screen can flash white in the opening cutscene |
| Brothers: A Tale of Two Sons | The village lane at dusk | – | – | 30 | 30 |  |
| Call of Duty Classic | Camp Toccoa, the training ground | – | – | 60 | 60 |  |
| Call of Juarez: Gunslinger | Episode 1, the road by the ruined barn | 30 | 30 | – | 30 | (earlier build) |
| Castle Crashers | The castle bar, after the opening cutscene | 56 | 56 | 60 | 60 | Runs a little under its 60 limit, not yet explained |
| DeathSpank: Thongs of Virtue | The prison camp kitchen | – | – | 30 | 30 |  |
| Earthworm Jim HD | New Junk City, the start of the level | – | – | 60 | 60 |  |
| Ecco the Dolphin | Home Bay, the first level | – | – | 60 | 60 |  |
| Flashback | The Titan jungle, by the waterfall | – | 30 | – | 30 | (earlier build) |
| From Dust | The first scene, the tribe on the night shore | – | – | 30 | 30 |  |
| Halo: Spartan Assault | Mission A-1, the outpost on an icy moon | 60 | 60 | 60 | 60 | Tested as the trial |
| Hydrophobia | Kate's apartment, first control | – | – | 30 | 30 |  |
| I Am Alive | First control on the ruined highway | – | – | 30 | 30 |  |
| Ion Assault | Echelon Territory, stage 1 | – | – | 60 | 60 |  |
| The King of Fighters '98 Ultimate Match | Arcade Play, the first fight | – | – | 60 | 60 |  |
| Marvel Puzzle Quest: Dark Reign | The first match board | – | – | 30 | 30 |  |
| Matt Hazard: Blood Bath and Beyond | The Hate Boat, the pool deck | – | – | 60 | 60 |  |
| Mortal Kombat Arcade Kollection | Mortal Kombat, the attract loop | – | – | – | 54 | Runs at 54, the original arcade board's rate |
| NBA Jam: On Fire Edition | Jam Now, a game in the Heat arena | – | – | – | 60 |  |
| Orcs Must Die! | The Hallway, the first level | – | – | – | 30 |  |
| R-Type Dimensions | R-Type stage 1, in 3D | – | – | – | 60 |  |
| Rez HD | Pause menu over Area 1 | – | – | 60 | 60 |  |
| RoboBlitz | Karl's workshop, first control | – | – | – | 30 |  |
| SEGA Bass Fishing | Arcade Mode, the Lodge | – | – | 60 | 60 |  |
| Serious Sam 3: BFE | Broken Wings demo, the Cairo park | 15 | 14 | 6 | 11 | Tested as the trial. The buildings behind the park can render solid black, and the level runs at about 15 on the M4 Macs and 6 on the M1 Max |
| Sonic & Knuckles | Mushroom Hill Zone, act 1 | – | – | 60 | 60 |  |
| SoulCalibur II HD Online | Arcade fight, Mitsurugi vs Cassandra | 60 | 60 | 60 | 60 |  |
| Super Street Fighter II Turbo HD Remix | Training mode, Ryu vs Ken | – | – | 60 | 60 |  |
| Teenage Mutant Ninja Turtles 1989 Arcade | Pause menu over Scene 1 | – | – | 60 | 60 |  |
| Trials Evolution | First career event, on the start ramp | – | – | 60 | 60 |  |
| Trials Fusion | Training Program 1, at the start line | 60 | 60 | 60 | 60 | A flat brown patch covers part of the track, and white specks show along some edges |
| Virtua Fighter 2 | Pause menu over the first Arcade fight | – | – | 60 | 60 |  |

Every figure in the tables comes from a test where the game was still running
at the end and hit no GPU error. Notes list the problems still open in these games;
problems fixed before these runs are not listed.

Mafia II, Saints Row IV and 50 Cent: Blood on the Sand don't limit their own
frame rate, so they run as fast as the Mac allows, and the table measures them
that way. To hold them at 60, set **Uncapped games** to **60 FPS** in the
Graphics settings. The game-select screen of Worms Collection is uncapped in
the same way, but the Worms menu the test measures holds 60.

A game not listed here may still work; these are only the ones tested on every
build. If one of them, or another game, misbehaves, see
[Troubleshooting](troubleshooting.md).
