/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"

#include <cstdio>
#include <cstdlib>

#include "xenia/base/logging.h"

// dxilconv (Microsoft DirectXShaderCompiler component). On non-Windows this
// pulls in the WinAdapter COM shim (__EMULATE_UUID, CoTaskMemFree=free, etc.).
#include "DxbcConverter.h"

// On the emulated-UUID path the linker needs the IUnknown uuid definition in
// exactly one translation unit of the consumer.
#if !defined(_WIN32) && defined(__EMULATE_UUID)
DEFINE_CROSS_PLATFORM_UUIDOF(IUnknown)
#endif

namespace xe {
namespace gpu {
namespace metal {

namespace {

// Default converter flags. -skip-container-parts drops the DXBC container parts
// that don't survive the DXIL round-trip and that the Metal Shader Converter
// does not consume, keeping the output to the DXIL module we care about.
constexpr wchar_t kDefaultOptions[] = L"-skip-container-parts";

std::wstring WidenAscii(const char* value) {
  std::wstring out;
  for (const char* p = value; p && *p; ++p) {
    out.push_back(static_cast<wchar_t>(*p));
  }
  return out;
}

std::string NarrowAscii(const wchar_t* value) {
  std::string out;
  for (const wchar_t* p = value; p && *p; ++p) {
    out.push_back(static_cast<char>(*p));
  }
  return out;
}

bool HasDxbcMagic(const std::vector<uint8_t>& data) {
  return data.size() >= 4 && data[0] == 'D' && data[1] == 'X' &&
         data[2] == 'B' && data[3] == 'C';
}

// Owns a per-thread converter and releases it on thread exit. dxilconv's
// converter object is single-threaded, so one instance per thread avoids
// cross-thread contention without a shared lock.
struct ThreadConverter {
  IDxbcConverter* converter = nullptr;
  ~ThreadConverter() {
    if (converter) {
      converter->Release();
    }
  }
};

}  // namespace

bool DxbcToDxilConverter::Initialize(std::string* error_out) {
  if (const char* env = std::getenv("XENIA_DXBC2DXIL_FLAGS")) {
    options_ = WidenAscii(env);
  } else {
    options_ = kDefaultOptions;
  }

  // A create/release round-trip proves the library is present and the CLSID is
  // registered before any shader depends on it.
  IDxbcConverter* probe = nullptr;
  HRESULT hr = DxcCreateInstance(CLSID_DxbcConverter, __uuidof(IDxbcConverter),
                                 reinterpret_cast<void**>(&probe));
  if (hr != S_OK || !probe) {
    is_available_ = false;
    if (error_out) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "hr=0x%08X", static_cast<unsigned>(hr));
      *error_out = std::string("dxilconv unavailable (") + buf + ")";
    }
    XELOGE("DxbcToDxilConverter: failed to create IDxbcConverter (hr=0x{:08X})",
           static_cast<unsigned>(hr));
    return false;
  }
  probe->Release();

  is_available_ = true;
  XELOGI("DxbcToDxilConverter: ready (options='{}')", NarrowAscii(options_.c_str()));
  return true;
}

IDxbcConverter* DxbcToDxilConverter::GetThreadConverter(std::string* error_out) {
  static thread_local ThreadConverter tls;
  if (tls.converter) {
    return tls.converter;
  }
  HRESULT hr = DxcCreateInstance(CLSID_DxbcConverter, __uuidof(IDxbcConverter),
                                 reinterpret_cast<void**>(&tls.converter));
  if (hr != S_OK || !tls.converter) {
    tls.converter = nullptr;
    if (error_out) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "hr=0x%08X", static_cast<unsigned>(hr));
      *error_out = std::string("failed to create thread converter (") + buf + ")";
    }
    return nullptr;
  }
  return tls.converter;
}

bool DxbcToDxilConverter::Convert(const std::vector<uint8_t>& dxbc,
                                  std::vector<uint8_t>& dxil_out,
                                  std::string* error_out) {
  if (!is_available_) {
    if (error_out) {
      *error_out = "DxbcToDxilConverter not initialized";
    }
    return false;
  }
  if (!HasDxbcMagic(dxbc)) {
    if (error_out) {
      *error_out = "input is not a DXBC container (missing 'DXBC' magic)";
    }
    return false;
  }

  IDxbcConverter* converter = GetThreadConverter(error_out);
  if (!converter) {
    return false;
  }

  void* dxil_ptr = nullptr;
  UINT32 dxil_size = 0;
  wchar_t* diagnostics = nullptr;
  HRESULT hr = converter->Convert(
      dxbc.data(), static_cast<UINT32>(dxbc.size()),
      options_.empty() ? nullptr : options_.c_str(), &dxil_ptr, &dxil_size,
      &diagnostics);

  if (hr != S_OK || !dxil_ptr || dxil_size == 0) {
    if (error_out) {
      if (diagnostics) {
        *error_out = "dxilconv Convert failed: " + NarrowAscii(diagnostics);
      } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "hr=0x%08X", static_cast<unsigned>(hr));
        *error_out = std::string("dxilconv Convert failed (") + buf + ")";
      }
    }
    CoTaskMemFree(diagnostics);
    CoTaskMemFree(dxil_ptr);
    return false;
  }

  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(dxil_ptr);
  dxil_out.assign(bytes, bytes + dxil_size);
  CoTaskMemFree(diagnostics);
  CoTaskMemFree(dxil_ptr);
  return true;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
