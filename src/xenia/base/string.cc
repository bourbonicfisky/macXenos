/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/string.h"

#include <string.h>

#include "xenia/base/platform.h"
#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif

#if !XE_PLATFORM_WIN32
#include <strings.h>
#endif  // !XE_PLATFORM_WIN32

// https://github.com/nemtrif/utfcpp/issues/85
#if defined(_MSVC_LANG) && _MSVC_LANG > __cplusplus
#define UTF_CPP_CPLUSPLUS _MSVC_LANG
#endif
#include "third_party/utfcpp/source/utf8.h"

namespace utfcpp = utf8;

namespace xe {

int xe_strcasecmp(const char* string1, const char* string2) {
#if XE_PLATFORM_WIN32
  return _stricmp(string1, string2);
#else
  return strcasecmp(string1, string2);
#endif  // XE_PLATFORM_WIN32
}

int xe_strncasecmp(const char* string1, const char* string2, size_t count) {
#if XE_PLATFORM_WIN32
  return _strnicmp(string1, string2, count);
#else
  return strncasecmp(string1, string2, count);
#endif  // XE_PLATFORM_WIN32
}

char* xe_strdup(const char* source) {
#if XE_PLATFORM_WIN32
  return _strdup(source);
#else
  return strdup(source);
#endif  // XE_PLATFORM_WIN32
}

// Guest strings are not trusted text: a title can hand a kernel export a
// buffer with an unpaired surrogate (Mass Effect's character-name keyboard
// on the M4, 2026-10-18), and the checked utfcpp conversion throws
// utf8::invalid_utf16, which nothing above catches (the exception terminated
// the process from the export trampoline's parameter logging). Convert
// leniently instead: invalid units become U+FFFD, exactly as a lossy decoder
// would, so logging and lookups degrade instead of aborting.
std::string to_utf8(const std::u16string_view source) {
  try {
    return utfcpp::utf16to8(source);
  } catch (const utfcpp::exception&) {
  }
  std::string result;
  result.reserve(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    uint32_t code_point = source[i];
    if (code_point >= 0xD800u && code_point <= 0xDBFFu) {
      if (i + 1 < source.size() && source[i + 1] >= 0xDC00u &&
          source[i + 1] <= 0xDFFFu) {
        code_point = 0x10000u + ((code_point - 0xD800u) << 10) +
                     (source[i + 1] - 0xDC00u);
        ++i;
      } else {
        code_point = 0xFFFDu;
      }
    } else if (code_point >= 0xDC00u && code_point <= 0xDFFFu) {
      code_point = 0xFFFDu;
    }
    utfcpp::unchecked::append(code_point, std::back_inserter(result));
  }
  return result;
}

std::u16string to_utf16(const std::string_view source) {
  try {
    return utfcpp::utf8to16(source);
  } catch (const utfcpp::exception&) {
  }
  // Same leniency for malformed UTF-8 (a guest path or a config value).
  std::string repaired;
  utfcpp::replace_invalid(source.begin(), source.end(),
                        std::back_inserter(repaired), 0xFFFDu);
  return utfcpp::utf8to16(repaired);
}

namespace {
// Windows-1252 bytes 0x80-0x9F. The five it leaves undefined map to the C1
// control of the same value, as Windows maps them.
constexpr char32_t kWin1252C1[32] = {
    0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
    0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F,
    0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
    0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
};
}  // namespace

std::string utf8_to_win1252(const std::string_view source) {
  const std::string valid = utfcpp::replace_invalid(source);
  std::string result;
  result.reserve(valid.size());
  for (auto it = valid.cbegin(); it != valid.cend();) {
    const char32_t cp = utfcpp::unchecked::next(it);
    char byte = '_';
    if (cp < 0x80 || (cp >= 0xA0 && cp <= 0xFF)) {
      byte = char(cp);
    } else {
      for (size_t i = 0; i < std::size(kWin1252C1); ++i) {
        if (kWin1252C1[i] == cp) {
          byte = char(0x80 + i);
          break;
        }
      }
    }
    result.push_back(byte);
  }
  return result;
}

std::string win1252_to_utf8(const std::string_view source) {
  std::string result;
  result.reserve(source.size());
  for (const char c : source) {
    const uint8_t byte = uint8_t(c);
    const char32_t cp =
        byte >= 0x80 && byte < 0xA0 ? kWin1252C1[byte - 0x80] : byte;
    utfcpp::append(cp, result);
  }
  return result;
}

}  // namespace xe
