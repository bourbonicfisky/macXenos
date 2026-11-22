/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "third_party/catch/include/catch.hpp"
#include "xenia/disc_set_names.h"

namespace xe {
namespace test {

TEST_CASE("the disc number is read from a dump's name", "[disc_set]") {
  REQUIRE(DiscNumberInName("Halo 4 (USA) (Disc 1).iso") == 1);
  REQUIRE(DiscNumberInName("Game (Disc 12) (Install).iso") == 12);
  REQUIRE(DiscNumberInName("Sonic Generations (World).iso") == 0);
  REQUIRE(DiscNumberInName("Game (Disc A).iso") == 0);
  REQUIRE(DiscNumberInName("Game (Disc 2.iso") == 0);
  // A disc's role stands in for a missing number; a number wins.
  REQUIRE(DiscNumberInName("Game (USA) (Game Disc).iso") == 1);
  REQUIRE(DiscNumberInName("Game (USA) (Install Disc).iso") == 2);
  REQUIRE(DiscNumberInName("Game (Disc 3) (Install Disc).iso") == 3);
}

TEST_CASE("unnumbered game and install discs are a set", "[disc_set]") {
  // Call of Duty: Advanced Warfare as Redump names it (GitHub #24).
  const char* game =
      "Call of Duty - Advanced Warfare (USA, Europe) (En,Fr,De,Es,It,Pl,Ru) "
      "(Game Disc).iso";
  const char* install =
      "Call of Duty - Advanced Warfare (USA, Europe) (En,Fr,De,Es,It,Pl,Ru) "
      "(Install Disc).iso";
  REQUIRE(IsOtherDiscOfSet(game, install));
  REQUIRE(IsOtherDiscOfSet(install, game));
  REQUIRE_FALSE(IsOtherDiscOfSet(
      game, "Call of Duty - Ghosts (USA, Europe) (Disc 2) (Install Disc).iso"));
}

TEST_CASE("discs of a set match when their tags differ", "[disc_set]") {
  // Call of Duty: Ghosts as Redump names it: the language tag is on the
  // install disc only.
  const char* game = "Call of Duty - Ghosts (USA, Europe) (Disc 1) (Game Disc).iso";
  const char* install =
      "Call of Duty - Ghosts (USA, Europe) (En,Fr,De,Es,It,Pt,Pl,Ru) (Disc 2) "
      "(Install Disc).iso";
  REQUIRE(IsOtherDiscOfSet(game, install));
  REQUIRE(IsOtherDiscOfSet(install, game));
  // The plain case, tags alike.
  REQUIRE(IsOtherDiscOfSet("Grand Theft Auto V (USA) (Disc 1) (Install).iso",
                           "Grand Theft Auto V (USA) (Disc 2).iso"));
}

TEST_CASE("other files are not discs of the set", "[disc_set]") {
  const char* game = "Call of Duty - Ghosts (USA, Europe) (Disc 1) (Game Disc).iso";
  // The same disc again (a second copy).
  REQUIRE_FALSE(IsOtherDiscOfSet(
      game, "Call of Duty - Ghosts (USA, Europe) (Disc 1) (Game Disc)-1.iso"));
  // Another title, also on two discs.
  REQUIRE_FALSE(IsOtherDiscOfSet(
      game, "Call of Duty - Advanced Warfare (USA, Europe) (Disc 2).iso"));
  // A title that starts the same.
  REQUIRE_FALSE(IsOtherDiscOfSet("Halo 4 (USA) (Disc 1).iso",
                                 "Halo 4 Demo (USA) (Disc 2).iso"));
  // A single-disc game, either way round.
  REQUIRE_FALSE(IsOtherDiscOfSet(game, "Call of Duty - Ghosts (USA).iso"));
  REQUIRE_FALSE(IsOtherDiscOfSet("Call of Duty - Ghosts (USA).iso", game));
}

}  // namespace test
}  // namespace xe
