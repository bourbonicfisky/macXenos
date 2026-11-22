/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_DISC_SET_NAMES_H_
#define XENIA_DISC_SET_NAMES_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace xe {

// Which images in a folder are the discs of one multi-disc title, from their
// file names. Dumps are named "<title> <tags> (Disc N) <tags>.iso", and the
// tags before "(Disc N)" are not always the same on every disc: Call of
// Duty: Ghosts is "Call of Duty - Ghosts (USA, Europe) (Disc 1) (Game
// Disc)" beside "Call of Duty - Ghosts (USA, Europe) (En,Fr,De,Es,It,Pt,Pl,
// Ru) (Disc 2) (Install Disc)". So the discs are matched on the title, the
// text before the first " (".

// Some two-disc sets carry no number at all, only the disc's role: Call of
// Duty: Advanced Warfare is "... (Game Disc).iso" beside "... (Install
// Disc).iso". Those count as discs 1 and 2.

// The N of "(Disc N)" in a file name (or 1 for "(Game Disc)" and 2 for
// "(Install Disc)" when there is no number), or 0 when it has none.
inline uint32_t DiscNumberInName(std::string_view name) {
  const size_t at = name.find("(Disc ");
  if (at == std::string_view::npos) {
    if (name.find("(Game Disc)") != std::string_view::npos) {
      return 1;
    }
    if (name.find("(Install Disc)") != std::string_view::npos) {
      return 2;
    }
    return 0;
  }
  const std::string digits(name.substr(at + 6, 4));
  char* end = nullptr;
  const unsigned long number = std::strtoul(digits.c_str(), &end, 10);
  return end != digits.c_str() && *end == ')' ? uint32_t(number) : 0;
}

// The title of a dump's file name: what comes before its first " (".
inline std::string_view DiscTitleInName(std::string_view name) {
  const size_t at = name.find(" (");
  return at == std::string_view::npos ? name : name.substr(0, at);
}

// Whether `candidate` is another disc of the title `launched` is a disc of:
// both name a disc, the numbers differ and the titles are the same.
inline bool IsOtherDiscOfSet(std::string_view launched,
                             std::string_view candidate) {
  const uint32_t launched_disc = DiscNumberInName(launched);
  const uint32_t candidate_disc = DiscNumberInName(candidate);
  return launched_disc && candidate_disc && launched_disc != candidate_disc &&
         !DiscTitleInName(launched).empty() &&
         DiscTitleInName(launched) == DiscTitleInName(candidate);
}

}  // namespace xe

#endif  // XENIA_DISC_SET_NAMES_H_
