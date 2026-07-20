/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_analysis_decode.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "third_party/fmt/include/fmt/format.h"
// STB_IMAGE_WRITE_IMPLEMENTATION is defined in trace_dump.cc (same static lib);
// here we only declare + call.
#include "third_party/stb/stb_image_write.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"

namespace xe {
namespace gpu {

static void WriteRgba(std::vector<uint8_t>& rgba, size_t px, uint8_t r,
                      uint8_t g, uint8_t b, uint8_t a) {
  rgba[px * 4 + 0] = r;
  rgba[px * 4 + 1] = g;
  rgba[px * 4 + 2] = b;
  rgba[px * 4 + 3] = a;
}

bool DecodeEdramColor32bpp(const uint8_t* edram, size_t edram_size,
                           uint32_t base_tiles, uint32_t pitch_tiles,
                           uint32_t msaa_samples,
                           xenos::ColorRenderTargetFormat format,
                           uint32_t width, uint32_t height,
                           uint32_t sample_index, DecodedImage* out) {
  if (!edram || !out || !width || !height) {
    return false;
  }
  out->width = width;
  out->height = height;
  out->rgba.assign(size_t(width) * height * 4, 0);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t off = XeEdramSnapshotOffsetBytes(
          x, y, base_tiles, pitch_tiles, msaa_samples, /*is_depth=*/false,
          /*format_ints_log2=*/0u, sample_index);
      size_t px = size_t(y) * width + x;
      if (off + 4 > edram_size) {
        // Out of the 10 MB window (surface extends past EDRAM / bad params) --
        // leave transparent rather than read OOB.
        continue;
      }
      uint32_t v;
      std::memcpy(&v, edram + off, sizeof(v));
      switch (format) {
        case xenos::ColorRenderTargetFormat::k_8_8_8_8:
        case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: {
          // 32bpp RGBA8 as stored; low byte = R.
          WriteRgba(out->rgba, px,
                    uint8_t(GetEdramColor8888Channel(v, 0u)),
                    uint8_t(GetEdramColor8888Channel(v, 1u)),
                    uint8_t(GetEdramColor8888Channel(v, 2u)),
                    uint8_t(GetEdramColor8888Channel(v, 3u)));
        } break;
        case xenos::ColorRenderTargetFormat::k_2_10_10_10:
        case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10: {
          // 10/10/10/2 -> 8/8/8/8 (drop low 2 bits of RGB, expand alpha).
          uint32_t r = (v >> 0) & 0x3FF, g = (v >> 10) & 0x3FF,
                   b = (v >> 20) & 0x3FF, a = (v >> 30) & 0x3;
          WriteRgba(out->rgba, px, uint8_t(r >> 2), uint8_t(g >> 2),
                    uint8_t(b >> 2), uint8_t(a * 85));
        } break;
        default: {
          // Best-effort raw bytes for formats not yet specifically handled
          // (float/64bpp decode is a follow-up); still useful as a structural
          // divergence signal.
          WriteRgba(out->rgba, px, uint8_t(v & 0xFF), uint8_t((v >> 8) & 0xFF),
                    uint8_t((v >> 16) & 0xFF), 0xFF);
        } break;
      }
    }
  }
  return true;
}

bool DecodeEdramDepth(const uint8_t* edram, size_t edram_size,
                      uint32_t base_tiles, uint32_t pitch_tiles,
                      uint32_t msaa_samples,
                      xenos::DepthRenderTargetFormat format, uint32_t width,
                      uint32_t height, uint32_t sample_index, DecodedImage* out,
                      std::vector<uint32_t>* out_depth24) {
  if (!edram || !out || !width || !height) {
    return false;
  }
  out->width = width;
  out->height = height;
  out->rgba.assign(size_t(width) * height * 4, 0);
  if (out_depth24) {
    out_depth24->assign(size_t(width) * height, 0);
  }
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t off = XeEdramSnapshotOffsetBytes(
          x, y, base_tiles, pitch_tiles, msaa_samples, /*is_depth=*/true,
          /*format_ints_log2=*/0u, sample_index);
      size_t px = size_t(y) * width + x;
      if (off + 4 > edram_size) {
        continue;
      }
      uint32_t v;
      std::memcpy(&v, edram + off, sizeof(v));
      // Xenos EDRAM depth/stencil word: depth in the high 24 bits, stencil in
      // the low 8.
      uint32_t depth24 = GetEdramDepth24(v);
      if (out_depth24) {
        (*out_depth24)[px] = depth24;
      }
      uint8_t gray;
      if (format == xenos::DepthRenderTargetFormat::kD24FS8) {
        float d = xenos::Float20e4To32(depth24);        // [0, 2)
        float n = d * 0.5f;                             // -> [0, 1)
        n = n < 0.0f ? 0.0f : (n > 1.0f ? 1.0f : n);
        gray = uint8_t(n * 255.0f + 0.5f);
      } else {
        // unorm24 -> take the top 8 bits.
        gray = uint8_t(depth24 >> 16);
      }
      WriteRgba(out->rgba, px, gray, gray, gray, 0xFF);
    }
  }
  return true;
}

bool RunEdramDecodeSelfTest() {
  // The offset formula must be a bijection over the samples it covers, or the
  // decode (and every downstream pixel compare) is silently wrong. Fill a
  // synthetic EDRAM by writing a unique marker at each sample's computed
  // offset, then read it back at each (x,y,sample) and confirm no collisions
  // and exact recovery. Covers a representative surface for a few configs.
  struct Case {
    uint32_t msaa;
    bool is_depth;
    uint32_t format_ints_log2;
    uint32_t width;
    uint32_t height;
    uint32_t samples;
  };
  const Case cases[] = {
      {uint32_t(xenos::MsaaSamples::k1X), false, 0, 80, 16, 1},
      {uint32_t(xenos::MsaaSamples::k2X), false, 0, 80, 16, 2},
      {uint32_t(xenos::MsaaSamples::k4X), false, 0, 80, 16, 4},
      {uint32_t(xenos::MsaaSamples::k1X), true, 0, 80, 16, 1},
      {uint32_t(xenos::MsaaSamples::k1X), false, 1, 40, 16, 1},  // 64bpp
  };
  const uint32_t base_tiles = 3;  // non-zero base to exercise tile origin.
  for (const Case& c : cases) {
    uint32_t pitch_tiles = XeEdramPitchTiles(c.width, c.msaa);
    // Map offset -> encoded (x,y,sample) marker; detect collisions.
    std::vector<uint32_t> seen;  // offset/4 -> marker+1 (0 = empty)
    seen.assign(size_t(xenos::kEdramSizeBytes) / 4, 0);
    for (uint32_t s = 0; s < c.samples; ++s) {
      for (uint32_t y = 0; y < c.height; ++y) {
        for (uint32_t x = 0; x < c.width; ++x) {
          uint32_t off = XeEdramSnapshotOffsetBytes(
              x, y, base_tiles, pitch_tiles, c.msaa, c.is_depth,
              c.format_ints_log2, s);
          if ((off & 3u) != 0 || (off >> 2) >= seen.size()) {
            XELOGE(
                "EDRAM decode self-test FAIL: offset {} out of range "
                "(msaa={} depth={} fmtlog2={} x={} y={} s={})",
                off, c.msaa, c.is_depth, c.format_ints_log2, x, y, s);
            return false;
          }
          uint32_t marker = ((s & 0xF) << 28) | ((y & 0x3FFF) << 14) |
                            (x & 0x3FFF);
          uint32_t& slot = seen[off >> 2];
          if (slot != 0 && slot != marker + 1) {
            XELOGE(
                "EDRAM decode self-test FAIL: offset collision at {} "
                "(msaa={} depth={} fmtlog2={}) between two samples",
                off, c.msaa, c.is_depth, c.format_ints_log2);
            return false;
          }
          slot = marker + 1;
        }
      }
    }
  }
  // Full color decode round-trip: write a unique value per pixel at its
  // offset, decode, require exact recovery (proves re-tile == input, not just
  // offset uniqueness).
  {
    const uint32_t w = 96, h = 40, base = 5;
    uint32_t pitch = XeEdramPitchTiles(w, uint32_t(xenos::MsaaSamples::k1X));
    std::vector<uint8_t> edram(xenos::kEdramSizeBytes, 0);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t off = XeEdramSnapshotOffsetBytes(
            x, y, base, pitch, uint32_t(xenos::MsaaSamples::k1X), false, 0, 0);
        uint32_t v = uint32_t(x & 0xFF) | (uint32_t(y & 0xFF) << 8) |
                     (uint32_t((x ^ y) & 0xFF) << 16) | (0xFFu << 24);
        std::memcpy(edram.data() + off, &v, sizeof(v));
      }
    }
    DecodedImage img;
    if (!DecodeEdramColor32bpp(edram.data(), edram.size(), base, pitch,
                               uint32_t(xenos::MsaaSamples::k1X),
                               xenos::ColorRenderTargetFormat::k_8_8_8_8, w, h,
                               0, &img)) {
      XELOGE("EDRAM decode self-test FAIL: color decode returned false");
      return false;
    }
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        size_t p = (size_t(y) * w + x) * 4;
        if (img.rgba[p + 0] != uint8_t(x) || img.rgba[p + 1] != uint8_t(y) ||
            img.rgba[p + 2] != uint8_t(x ^ y) || img.rgba[p + 3] != 0xFF) {
          XELOGE("EDRAM decode self-test FAIL: color round-trip mismatch at "
                 "({},{})", x, y);
          return false;
        }
      }
    }
  }

  // Depth24 round-trip: the compare oracle diffs the raw depth24 channel
  // byte-exact, so prove DecodeEdramDepth recovers the exact 24-bit depth the
  // offset formula placed (independent of the 8-bit grayscale PNG).
  {
    const uint32_t w = 96, h = 40, base = 3;
    uint32_t pitch = XeEdramPitchTiles(w, uint32_t(xenos::MsaaSamples::k1X));
    std::vector<uint8_t> edram(xenos::kEdramSizeBytes, 0);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t off = XeEdramSnapshotOffsetBytes(
            x, y, base, pitch, uint32_t(xenos::MsaaSamples::k1X),
            /*is_depth=*/true, 0, 0);
        uint32_t word = (((x << 12) | y) << 8) | 0xAB;  // depth24<<8 | stencil
        std::memcpy(edram.data() + off, &word, sizeof(word));
      }
    }
    DecodedImage dimg;
    std::vector<uint32_t> depth24;
    if (!DecodeEdramDepth(edram.data(), edram.size(), base, pitch,
                          uint32_t(xenos::MsaaSamples::k1X),
                          xenos::DepthRenderTargetFormat::kD24S8, w, h, 0, &dimg,
                          &depth24)) {
      XELOGE("EDRAM decode self-test FAIL: depth decode returned false");
      return false;
    }
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        if (depth24[size_t(y) * w + x] != ((x << 12) | y)) {
          XELOGE("EDRAM decode self-test FAIL: depth24 round-trip mismatch at "
                 "({},{})", x, y);
          return false;
        }
      }
    }
  }

  XELOGI(
      "EDRAM decode self-test PASS ({} offset configs + color & depth24 "
      "round-trip)",
      int(sizeof(cases) / sizeof(cases[0])));
  return true;
}

// --- capture-directory decode ---------------------------------------------

// Minimal extractor for the byte-stable manifest we emit (avoids a JSON dep):
// pulls an unsigned integer value for `key` from within `obj`.
static bool ExtractUint(const std::string& obj, const char* key,
                        uint32_t* out) {
  std::string needle = std::string("\"") + key + "\":";
  size_t p = obj.find(needle);
  if (p == std::string::npos) {
    return false;
  }
  p += needle.size();
  // Skip an optional opening quote (some values are quoted hex; not used here).
  bool hex = false;
  if (p < obj.size() && obj[p] == '"') {
    hex = true;
    ++p;
  }
  uint32_t value = 0;
  bool any = false;
  for (; p < obj.size(); ++p) {
    char ch = obj[p];
    if (ch >= '0' && ch <= '9') {
      value = value * (hex ? 16u : 10u) + uint32_t(ch - '0');
      any = true;
    } else if (hex && ((ch >= 'A' && ch <= 'F') || (ch >= 'a' && ch <= 'f'))) {
      value = value * 16u + uint32_t((ch | 0x20) - 'a' + 10);
      any = true;
    } else {
      break;
    }
  }
  if (!any) {
    return false;
  }
  *out = value;
  return true;
}

// Returns the substring of the `"name":{ ... }` object inside `line`.
static std::string ExtractObject(const std::string& line, const char* name) {
  std::string needle = std::string("\"") + name + "\":{";
  size_t p = line.find(needle);
  if (p == std::string::npos) {
    return std::string();
  }
  size_t start = p + needle.size() - 1;  // at '{'
  int depth = 0;
  for (size_t i = start; i < line.size(); ++i) {
    if (line[i] == '{') {
      ++depth;
    } else if (line[i] == '}') {
      if (--depth == 0) {
        return line.substr(start, i - start + 1);
      }
    }
  }
  return std::string();
}

int DecodeCaptureDirectory(const std::filesystem::path& capture_dir,
                           uint32_t width, uint32_t height,
                           uint32_t sample_index) {
  if (!RunEdramDecodeSelfTest()) {
    XELOGE("--decode aborted: EDRAM offset self-test failed (tiling drift?)");
    return -1;
  }
  // Read manifest.jsonl into a draw_index -> line map (by first field "draw").
  std::filesystem::path manifest_path = capture_dir / "manifest.jsonl";
  FILE* mf = xe::filesystem::OpenFile(manifest_path, "rb");
  if (!mf) {
    XELOGE("--decode: no manifest.jsonl in {}", capture_dir.string());
    return -1;
  }
  std::vector<std::pair<uint32_t, std::string>> rows;
  {
    std::string content;
    fseek(mf, 0, SEEK_END);
    long sz = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    if (sz > 0) {
      content.resize(size_t(sz));
      size_t read = fread(&content[0], 1, size_t(sz), mf);
      content.resize(read);
    }
    fclose(mf);
    size_t pos = 0;
    while (pos < content.size()) {
      size_t nl = content.find('\n', pos);
      std::string line = content.substr(
          pos, nl == std::string::npos ? std::string::npos : nl - pos);
      if (!line.empty()) {
        uint32_t draw = 0;
        if (ExtractUint(line, "draw", &draw)) {
          rows.emplace_back(draw, line);
        }
      }
      if (nl == std::string::npos) {
        break;
      }
      pos = nl + 1;
    }
  }

  int decoded = 0;
  std::error_code ec;
  for (const auto& entry :
       std::filesystem::directory_iterator(capture_dir, ec)) {
    std::string fname = entry.path().filename().string();
    // Match draw_NNNNNN.edram.bin
    if (fname.rfind("draw_", 0) != 0 ||
        fname.find(".edram.bin") == std::string::npos) {
      continue;
    }
    uint32_t draw_index = uint32_t(std::atoi(fname.c_str() + 5));
    // Find the manifest row for this draw.
    const std::string* row = nullptr;
    for (const auto& r : rows) {
      if (r.first == draw_index) {
        row = &r.second;
        break;
      }
    }
    if (!row) {
      XELOGW("--decode: no manifest row for {}, skipping", fname);
      continue;
    }
    std::string color0 = ExtractObject(*row, "color0");
    std::string surface = ExtractObject(*row, "surface");
    std::string depth = ExtractObject(*row, "depth");
    uint32_t base = 0, fmt = 0, msaa = 0, pitch = 0;
    uint32_t depth_base = 0, depth_fmt = 0, depth_en = 0;
    ExtractUint(color0, "base", &base);
    ExtractUint(color0, "fmt", &fmt);
    ExtractUint(surface, "msaa", &msaa);
    ExtractUint(surface, "pitch", &pitch);
    ExtractUint(depth, "base", &depth_base);
    ExtractUint(depth, "fmt", &depth_fmt);
    ExtractUint(depth, "z_en", &depth_en);

    // Load the snapshot.
    FILE* sf = xe::filesystem::OpenFile(entry.path(), "rb");
    if (!sf) {
      continue;
    }
    std::vector<uint8_t> edram(xenos::kEdramSizeBytes);
    size_t got = fread(edram.data(), 1, edram.size(), sf);
    fclose(sf);
    if (got != edram.size()) {
      XELOGW("--decode: {} is {} bytes (expected {}), skipping", fname, got,
             edram.size());
      continue;
    }

    uint32_t dec_w = width ? width : (pitch ? pitch : 1280u);
    uint32_t dec_h = height ? height : 720u;
    uint32_t pitch_tiles = XeEdramPitchTiles(pitch ? pitch : dec_w, msaa);
    DecodedImage img;
    if (!DecodeEdramColor32bpp(
            edram.data(), edram.size(), base, pitch_tiles, msaa,
            xenos::ColorRenderTargetFormat(fmt), dec_w, dec_h, sample_index,
            &img)) {
      XELOGW("--decode: failed to decode {}", fname);
      continue;
    }
    // Typed raw RGBA8 buffer (what the Python compare layer diffs -- no Xenos
    // knowledge needed downstream).
    std::filesystem::path bin_path =
        capture_dir / fmt::format("color_draw{:06d}.rgba8.bin", draw_index);
    if (FILE* bf = xe::filesystem::OpenFile(bin_path, "wb")) {
      fwrite(img.rgba.data(), 1, img.rgba.size(), bf);
      fclose(bf);
    }
    std::filesystem::path png_path =
        capture_dir /
        fmt::format("color_draw{:06d}.png", draw_index);
    FILE* pf = xe::filesystem::OpenFile(png_path, "wb");
    if (pf) {
      auto cb = [](void* ctx, void* data, int size) {
        fwrite(data, 1, size_t(size), reinterpret_cast<FILE*>(ctx));
      };
      stbi_write_png_to_func(cb, pf, int(img.width), int(img.height), 4,
                             img.rgba.data(), int(img.width * 4));
      fclose(pf);
      ++decoded;
      XELOGI(
          "--decode: {} -> {} ({}x{} base_tiles={} pitch_tiles={} msaa={} "
          "fmt={})",
          fname, png_path.filename().string(), img.width, img.height, base,
          pitch_tiles, msaa, fmt);
    }

    // Depth, when the draw had depth testing enabled (same EDRAM snapshot).
    if (depth_en) {
      DecodedImage dimg;
      std::vector<uint32_t> depth24;
      if (DecodeEdramDepth(edram.data(), edram.size(), depth_base, pitch_tiles,
                           msaa, xenos::DepthRenderTargetFormat(depth_fmt),
                           dec_w, dec_h, sample_index, &dimg, &depth24)) {
        std::filesystem::path dpng =
            capture_dir / fmt::format("depth_draw{:06d}.png", draw_index);
        if (FILE* df = xe::filesystem::OpenFile(dpng, "wb")) {
          auto cb = [](void* ctx, void* data, int size) {
            fwrite(data, 1, size_t(size), reinterpret_cast<FILE*>(ctx));
          };
          stbi_write_png_to_func(cb, df, int(dimg.width), int(dimg.height), 4,
                                 dimg.rgba.data(), int(dimg.width * 4));
          fclose(df);
        }
        // Byte-exact raw depth24 channel (little-endian uint32 per pixel,
        // row-major). Correct for unorm24 (kD24S8); for float24 (kD24FS8) the
        // low mantissa bits are host-nondeterministic, so the compare oracle
        // uses the depthf32 channel below with a tolerance instead. Kept for
        // exact integer work / provenance.
        std::filesystem::path dbin =
            capture_dir / fmt::format("depth_draw{:06d}.depth24.bin", draw_index);
        if (FILE* db = xe::filesystem::OpenFile(dbin, "wb")) {
          fwrite(depth24.data(), sizeof(uint32_t), depth24.size(), db);
          fclose(db);
        }
        // Full-precision decoded depth value (float32 per pixel). This is what
        // the compare oracle diffs with a tolerance: float24 z-interpolation is
        // not bit-exact across GPU vendors, so a byte-exact depth diff drowns
        // real coverage divergence in mantissa noise. Ranges differ by format
        // (unorm24 -> [0,1], float24 -> [0,2)) but each side uses the same
        // format, so a relative/absolute epsilon separates noise from coverage.
        std::vector<float> depthf(depth24.size());
        bool is_f24 =
            xenos::DepthRenderTargetFormat(depth_fmt) ==
            xenos::DepthRenderTargetFormat::kD24FS8;
        for (size_t i = 0; i < depth24.size(); ++i) {
          depthf[i] = is_f24 ? xenos::Float20e4To32(depth24[i])
                             : float(depth24[i]) / float(0xFFFFFF);
        }
        std::filesystem::path fbin =
            capture_dir /
            fmt::format("depth_draw{:06d}.depthf32.bin", draw_index);
        if (FILE* fb = xe::filesystem::OpenFile(fbin, "wb")) {
          fwrite(depthf.data(), sizeof(float), depthf.size(), fb);
          fclose(fb);
        }
      }
    }
  }
  XELOGI("--decode: decoded {} snapshot(s) in {}", decoded,
         capture_dir.string());
  return decoded;
}

}  // namespace gpu
}  // namespace xe
