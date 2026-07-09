/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Phase 2a spike: drive the DXBC -> DXIL -> metallib back half of the native
// Metal shader path end-to-end, no renderer. Reads DXBC produced by
// xenia-gpu-shader-compiler (--shader_output_type=dxbc
// --shader_output_bindless_resources=true) and writes a .metallib. Validates the
// whole DirectX->Metal chain inside our build before the Metal backend exists.
//
//   metal-shader-translate-spike <shader.dxbc> vs|ps <out.metallib>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_shader_converter.h"

namespace xe {
namespace gpu {
namespace metal {

static bool ReadFile(const std::string& path, std::vector<uint8_t>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    size_t got = std::fread(out.data(), 1, out.size(), f);
    out.resize(got);
  }
  std::fclose(f);
  return !out.empty();
}

static bool WriteFile(const std::string& path, const std::vector<uint8_t>& d) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = std::fwrite(d.data(), 1, d.size(), f) == d.size();
  std::fclose(f);
  return ok;
}

int SpikeMain(const std::vector<std::string>& args) {
  if (args.size() < 4) {
    std::printf("usage: %s <shader.dxbc> vs|ps|vsgeo <out.metallib>\n",
                args[0].c_str());
    return 2;
  }
  const std::string& dxbc_path = args[1];
  const std::string& mode = args[2];
  const bool is_vertex = mode == "vs";
  // vsgeo: run the vertex stage through the geometry-emulation path (object stage
  // + synthesized stage-in) and assert a non-empty stage-in metallib. This is the
  // decisive off-device check for the rect-list geometry-emulation pipeline: the
  // MSC runtime dereferences the stage-in library unconditionally, so if
  // synthesis (IRMetalLibSynthesizeStageInFunction, numElements=0) came back empty
  // the pipeline would crash on device. Validates without the emulator/Processor.
  const bool is_vertex_geometry = mode == "vsgeo";
  const std::string& out_path = args[3];

  std::vector<uint8_t> dxbc;
  if (!ReadFile(dxbc_path, dxbc)) {
    std::printf("FAIL: cannot read %s\n", dxbc_path.c_str());
    return 1;
  }

  std::string error;
  DxbcToDxilConverter dxil_converter;
  if (!dxil_converter.Initialize(&error)) {
    std::printf("FAIL: dxil converter init: %s\n", error.c_str());
    return 1;
  }
  std::vector<uint8_t> dxil;
  if (!dxil_converter.Convert(dxbc, dxil, &error)) {
    std::printf("FAIL: DXBC->DXIL: %s\n", error.c_str());
    return 1;
  }
  std::printf("DXBC %zu -> DXIL %zu bytes\n", dxbc.size(), dxil.size());

  MetalShaderConverter metal_converter;
  if (!metal_converter.Initialize(/*bindless_resources_used=*/true, &error)) {
    std::printf("FAIL: metal converter init: %s\n", error.c_str());
    return 1;
  }
  if (is_vertex_geometry) {
    MetalShaderConverter::GeometryEmulationResult geo;
    if (!metal_converter.ConvertForGeometryEmulation(
            MetalShaderConverter::Stage::kVertex, dxil,
            MetalShaderConverter::InputTopology::kTriangle, geo)) {
      std::printf("FAIL: geometry-emulation VS convert: %s\n", geo.error.c_str());
      return 1;
    }
    if (geo.metallib.size() < 4 ||
        std::memcmp(geo.metallib.data(), "MTLB", 4) != 0) {
      std::printf("FAIL: object-stage output is not a metallib (bad magic)\n");
      return 1;
    }
    if (geo.stage_in_metallib.empty()) {
      std::printf("FAIL: stage-in synthesis produced no metallib\n");
      return 1;
    }
    if (geo.stage_in_metallib.size() < 4 ||
        std::memcmp(geo.stage_in_metallib.data(), "MTLB", 4) != 0) {
      std::printf("FAIL: stage-in output is not a metallib (bad magic)\n");
      return 1;
    }
    if (geo.vertex_output_size_in_bytes == 0) {
      std::printf("FAIL: VS reflection returned zero output size\n");
      return 1;
    }
    if (!WriteFile(out_path, geo.metallib)) {
      std::printf("FAIL: cannot write %s\n", out_path.c_str());
      return 1;
    }
    std::printf(
        "OK: geometry-emulation VS: object metallib %zu bytes, stage-in %zu "
        "bytes (MTLB), vs_output_size=%u -> %s\n",
        geo.metallib.size(), geo.stage_in_metallib.size(),
        geo.vertex_output_size_in_bytes, out_path.c_str());
    return 0;
  }

  std::vector<uint8_t> metallib;
  auto stage = is_vertex ? MetalShaderConverter::Stage::kVertex
                         : MetalShaderConverter::Stage::kFragment;
  if (!metal_converter.Convert(stage, dxil, metallib, &error)) {
    std::printf("FAIL: DXIL->metallib: %s\n", error.c_str());
    return 1;
  }
  if (metallib.size() < 4 || std::memcmp(metallib.data(), "MTLB", 4) != 0) {
    std::printf("FAIL: output is not a metallib (bad magic)\n");
    return 1;
  }
  if (!WriteFile(out_path, metallib)) {
    std::printf("FAIL: cannot write %s\n", out_path.c_str());
    return 1;
  }
  std::printf("OK: DXIL %zu -> metallib %zu bytes (MTLB) -> %s\n", dxil.size(),
              metallib.size(), out_path.c_str());
  return 0;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

int main(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
  return xe::gpu::metal::SpikeMain(args);
}
