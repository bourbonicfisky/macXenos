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

std::string utf8_to_win1252(const std::string_view source) {
#if XE_PLATFORM_WIN32
  std::string input_str(source);
  int srclen = static_cast<int>(input_str.size());

  int wlen =
      MultiByteToWideChar(CP_UTF8, 0, input_str.c_str(), srclen, NULL, 0);
  if (!wlen) {
    return "";
  }
  std::vector<WCHAR> wbuf(wlen);
  int result = MultiByteToWideChar(CP_UTF8, 0, input_str.c_str(), srclen,
                                   wbuf.data(), wlen);
  if (!result) {
    return "";
  }

  int len = WideCharToMultiByte(1252, 0, wbuf.data(), wlen, NULL, 0, "_", NULL);
  if (!len) {
    return "";
  }
  std::vector<CHAR> buf(len);
  result = WideCharToMultiByte(1252, 0, wbuf.data(), wlen, buf.data(), len, "_",
                               NULL);
  if (!result) {
    return "";
  }

  std::string output_str(buf.begin(), buf.end());
  return output_str;
#else
  // TODO: Use iconv on POSIX.
  std::string output_str(source);
  return output_str;
#endif
}

std::string win1252_to_utf8(const std::string_view source) {
#if XE_PLATFORM_WIN32
  std::string input_str(source);
  int srclen = static_cast<int>(input_str.size());

  int wlen = MultiByteToWideChar(1252, 0, input_str.c_str(), srclen, NULL, 0);
  if (!wlen) {
    return "";
  }
  std::vector<WCHAR> wbuf(wlen);
  int result = MultiByteToWideChar(1252, 0, input_str.c_str(), srclen,
                                   wbuf.data(), wlen);
  if (!result) {
    return "";
  }

  int len =
      WideCharToMultiByte(CP_UTF8, 0, wbuf.data(), wlen, NULL, 0, "_", NULL);
  if (!len) {
    return "";
  }
  std::vector<CHAR> buf(len);
  result = WideCharToMultiByte(CP_UTF8, 0, wbuf.data(), wlen, buf.data(), len,
                               "_", NULL);
  if (!result) {
    return "";
  }

  std::string output_str(buf.begin(), buf.end());
  return output_str;
#else
  // TODO: Use iconv on POSIX.
  std::string output_str(source);
  return output_str;
#endif
}

}  // namespace xe
