/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// The DXIL-cache KILL-GATE for adopt-xenios-shader-artifact-caching (task 2.1).
//
// Caching dxilconv output is only sound if dxilconv is a pure function of
// (DXBC bytes, option string). If it is not, a content-derived key names an
// artifact that the same inputs would not reproduce, and the cache serves
// something this build does not produce. Stage 2 does not land unless these
// cases pass; that is what "kill-gate" means here.
//
// The four questions, in the order they must be answered:
//
//   1. CALL ORDER — does converting B first change the artifact for A?
//      This is not hypothetical. A reused IDxbcConverter carried call history
//      into later conversions, and an adaptive-QUAD domain artifact converted
//      after an adaptive-TRIANGLE one was CORRUPT at runtime (root cause of the
//      tessellation 5.2 residual, fixed 2026-08-02 as fresh-per-Convert in
//      dxbc_to_dxil_converter.cc). Order independence is the property that bug
//      violated, so it is the property the gate measures.
//
//   2. CROSS PROCESS — is the artifact the same in a different process image?
//      An in-process check cannot see content derived from an address, a pid,
//      a clock, or anything else that happens to be stable within one run. A
//      durable cache is read by a LATER process, so that is the comparison
//      that matters. Two independent children plus this process.
//
//   3. OPTION COVERAGE — the option string is an input to the conversion, so
//      it must reach the key. Measured here rather than assumed: the same DXBC
//      under the default "-skip-container-parts" and under an empty string
//      produces DIFFERENT DXIL. Before this change that string reached no cache
//      key at all (metal_converter_options.h now feeds both the converter and
//      ComputeMetalPipelineStorageIdentity).
//
//   4. HIT VS FRESH — the load-bearing invariant: a hit must equal a fresh
//      conversion byte for byte, or it must miss
//      (metal_artifact_hit_verify.h). Key discipline is fallible by
//      construction — a key cannot encode a behavior nobody knew had changed —
//      so the behavioral check is what catches the next bug of this shape.
//      Exercised here over REAL dxilconv artifacts, including the case that
//      matters most: a well-formed artifact from a DIFFERENT payload, which is
//      what a key collision or a stale entry actually looks like.
//
// RISK THIS CHANGE ACCEPTS, STATED PLAINLY: every artifact kind added to the
// cache widens the surface on which a future correctness bug is preserved and
// re-served rather than merely committed. That is what happened with the
// converter-reuse bug. These gates are the price of the amortization.
//
// Every identity assertion below is gated by a distinctness control, because
// "all the artifacts matched" is also what a converter that emitted a constant
// would report.

#import <Foundation/Foundation.h>

#include <mach-o/dyld.h>
#include <limits.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_artifact_hit_verify.h"
#include "xenia/gpu/metal/metal_converter_options.h"
#include "xenia/gpu/metal/metal_pipeline_storage.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

using metal::ComputeMetalPipelineStorageIdentity;
using metal::DxbcToDxilConverter;
using metal::kMetalDxbcToDxilDefaultOptions;
using metal::kMetalDxbcToDxilOptionsEnvVar;
using metal::MetalArtifactHitIsServable;
using metal::MetalArtifactHitVerdict;
using metal::MetalArtifactHitVerdictName;
using metal::MetalDxbcToDxilEffectiveOptions;
using metal::MetalShader;
using metal::MetalTranslatorConfig;
using metal::VerifyMetalArtifactHit;

// Minimal fetch-free guest vertex program: receives the translated vertex index
// in r0.x and exports (r0.x, 0, 0, 1) as position. Shape taken from an
// authoritative trace shader (shared with
// metal_geometry_clip_plane_discriminator_test.mm) — real microcode keeps
// DxbcShaderTranslator in the loop instead of pinning a generated artifact.
alignas(uint32_t) constexpr uint8_t kVertexUcode[] = {
    0x03, 0x10, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC4, 0x04, 0x10, 0x00, 0x00,
    0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC2, 0x05, 0x10, 0x00, 0x00, 0x00, 0x12, 0x06, 0x10,
    0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2,
    0x00, 0xC0, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2, 0x3E, 0xC0, 0x89, 0xC8,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xE2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Minimal guest PIXEL program. Assembled rather than reused from the vertex
// fixture: analyzing vertex microcode as a pixel shader traps in the ucode
// analyzer (its exports are positions/interpolators, which a pixel shader
// cannot carry).
std::vector<uint32_t> MakePixelUcode() {
  uint32_t alloc_d0 = 0;
  uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kPsColors) << 9) |
                      (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);
  uint32_t exec_d0 = (1u & 0xFFF) | ((1u & 0x7) << 12);
  uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;

  std::vector<uint32_t> u(6, 0);
  u[0] = alloc_d0;
  u[1] = (alloc_d1 & 0xFFFF) | ((exec_d0 & 0xFFFF) << 16);
  u[2] = (exec_d0 >> 16) | ((exec_d1 & 0xFFFF) << 16);

  uint32_t alu0 = (0u & 0x3F) | (1u << 15) | (0xFu << 16) |
                  (uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26);
  uint32_t alu1 = 0;
  uint32_t alu2 = (uint32_t(ucode::AluVectorOpcode::kMax) << 24);
  u[3] = alu0;
  u[4] = alu1;
  u[5] = alu2;
  return u;
}

// One DXBC payload plus the name it is reported under.
struct Payload {
  std::string name;
  std::vector<uint8_t> dxbc;
};

// Builds the corpus from real translator output. Three payloads, chosen so the
// corpus is not one shader converted three times: two shader TYPES (vertex and
// pixel) and, within the vertex type, two MODIFICATIONS that emit different
// code for the same microcode. A single-payload corpus cannot express the
// order-dependence question at all.
std::vector<Payload> BuildCorpus() {
  std::vector<Payload> corpus;

  auto translate = [&corpus](const char* name, MetalShader& shader,
                             DxbcShaderTranslator::Modification modification) {
    DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                    MetalTranslatorConfig::kBindlessResourcesUsed,
                                    MetalTranslatorConfig::kEdramRovUsed,
                                    MetalTranslatorConfig::GammaRenderTargetAsUnorm8());
    Shader::Translation* translation =
        shader.GetOrCreateTranslation(modification.value);
    if (!translation) {
      return;
    }
    if (translation->translated_binary().empty() &&
        !translator.TranslateAnalyzedShader(*translation)) {
      return;
    }
    corpus.push_back(Payload{name, translation->translated_binary()});
  };

  static MetalShader vertex_shader(
      xenos::ShaderType::kVertex, 0x7101710171017101ull,
      reinterpret_cast<const uint32_t*>(kVertexUcode),
      sizeof(kVertexUcode) / sizeof(uint32_t), std::endian::little);
  static std::vector<uint32_t> pixel_ucode = MakePixelUcode();
  static MetalShader pixel_shader(xenos::ShaderType::kPixel,
                                  0x7202720272027202ull, pixel_ucode.data(),
                                  pixel_ucode.size(), std::endian::little);
  static bool analyzed = false;
  if (!analyzed) {
    StringBuffer disasm;
    vertex_shader.AnalyzeUcode(disasm);
    pixel_shader.AnalyzeUcode(disasm);
    analyzed = true;
  }

  DxbcShaderTranslator probe(ui::GraphicsProvider::GpuVendorID::kApple,
                             MetalTranslatorConfig::kBindlessResourcesUsed,
                             MetalTranslatorConfig::kEdramRovUsed,
                             MetalTranslatorConfig::GammaRenderTargetAsUnorm8());

  DxbcShaderTranslator::Modification vm(
      probe.GetDefaultVertexShaderModification(
          vertex_shader.GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters),
          Shader::HostVertexShaderType::kVertex));
  vm.vertex.interpolator_mask = 0;
  vm.vertex.user_clip_plane_count = 0;
  translate("vs.clip0", vertex_shader, vm);

  DxbcShaderTranslator::Modification vm_clip = vm;
  vm_clip.vertex.user_clip_plane_count = 2;
  translate("vs.clip2", vertex_shader, vm_clip);

  DxbcShaderTranslator::Modification pm(
      probe.GetDefaultPixelShaderModification(
          pixel_shader.GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters)));
  pm.pixel.interpolator_mask = 0;
  translate("ps.color0", pixel_shader, pm);

  return corpus;
}

const std::vector<Payload>& Corpus() {
  static const std::vector<Payload> corpus = BuildCorpus();
  return corpus;
}

// Converts one payload through a caller-supplied converter, so a case can
// choose between "a converter that has never been used" and "a converter that
// has already converted something else".
std::vector<uint8_t> ConvertWith(DxbcToDxilConverter& converter,
                                 const std::vector<uint8_t>& dxbc) {
  std::vector<uint8_t> dxil;
  std::string error;
  if (!converter.Convert(dxbc, dxil, &error)) {
    INFO("dxilconv error: " << error);
    return {};
  }
  return dxil;
}

// Converts through a converter that has done nothing else. This is the
// reference artifact every other production is compared against.
std::vector<uint8_t> ConvertStandalone(const std::vector<uint8_t>& dxbc) {
  DxbcToDxilConverter converter;
  std::string error;
  if (!converter.Initialize(&error)) {
    return {};
  }
  return ConvertWith(converter, dxbc);
}

bool WriteBlob(const std::filesystem::path& path,
               const std::vector<uint8_t>& bytes) {
  FILE* file = fopen(path.string().c_str(), "wb");
  if (!file) {
    return false;
  }
  const bool ok =
      bytes.empty() || fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
  fclose(file);
  return ok;
}

std::vector<uint8_t> ReadBlob(const std::filesystem::path& path) {
  FILE* file = fopen(path.string().c_str(), "rb");
  if (!file) {
    return {};
  }
  std::vector<uint8_t> out;
  uint8_t buffer[8192];
  size_t read = 0;
  while ((read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
    out.insert(out.end(), buffer, buffer + read);
  }
  fclose(file);
  return out;
}

// --- cross-process plumbing -------------------------------------------------

constexpr char kChildDirEnvVar[] = "XE_DXIL_DETERMINISM_CHILD_DIR";
// Makes the child emit a deliberately divergent artifact. It exists so the
// cross-process comparison can be shown to FAIL on a real divergence: without
// it, "the artifacts matched" and "the comparison never ran" are the same
// observation, and the gate could only ever pass.
constexpr char kChildPoisonEnvVar[] = "XE_DXIL_DETERMINISM_CHILD_POISON";
constexpr char kChildTag[] = "[dxilchild]";

std::string ExecutablePath() {
  char raw[PATH_MAX * 2];
  uint32_t size = sizeof(raw);
  if (_NSGetExecutablePath(raw, &size) != 0) {
    return {};
  }
  char resolved[PATH_MAX];
  if (realpath(raw, resolved)) {
    return std::string(resolved);
  }
  return std::string(raw);
}

std::filesystem::path ScratchRoot(const char* name) {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() / "xe_dxil_determinism" / name;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  return root;
}

// Runs this same binary as a child process, asking it to convert the corpus
// into `dir`. Returns false if the child did not produce a completed set —
// which FAILS the case rather than skipping it, because a silently absent
// child would turn the cross-process gate into a test that can only pass.
bool RunChildConversion(const std::filesystem::path& dir, bool poison = false) {
  const std::string exe = ExecutablePath();
  if (exe.empty()) {
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  setenv(kChildDirEnvVar, dir.string().c_str(), 1);
  if (poison) {
    setenv(kChildPoisonEnvVar, "1", 1);
  }
  const std::string command =
      "\"" + exe + "\" \"" + kChildTag + "\" >/dev/null 2>&1";
  const int rc = system(command.c_str());
  unsetenv(kChildDirEnvVar);
  unsetenv(kChildPoisonEnvVar);
  if (rc != 0) {
    std::printf("DXILGATE|CHILD|nonzero exit rc=%d\n", rc);
    return false;
  }
  return std::filesystem::exists(dir / "done");
}

std::filesystem::path ChildDxbcPath(const std::filesystem::path& dir,
                                    size_t index) {
  return dir / (std::string("p") + std::to_string(index) + ".dxbc");
}
std::filesystem::path ChildDxilPath(const std::filesystem::path& dir,
                                    size_t index) {
  return dir / (std::string("p") + std::to_string(index) + ".dxil");
}

std::pair<size_t, size_t> DiffStats(const std::vector<uint8_t>& a,
                                    const std::vector<uint8_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t first_diff = n;
  size_t diff_bytes = 0;
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      if (first_diff == n) {
        first_diff = i;
      }
      ++diff_bytes;
    }
  }
  return {first_diff, diff_bytes};
}

}  // namespace

// Hidden worker: only runs when the parent asks for it by tag AND names an
// output directory. Harmless if invoked otherwise.
TEST_CASE("dxil determinism child worker", "[.][dxilchild]") {
  const char* dir_env = std::getenv(kChildDirEnvVar);
  if (!dir_env) {
    SUCCEED("not a child invocation");
    return;
  }
  const std::filesystem::path dir(dir_env);
  const char* poison = std::getenv(kChildPoisonEnvVar);
  const auto& corpus = Corpus();
  REQUIRE_FALSE(corpus.empty());
  for (size_t i = 0; i < corpus.size(); ++i) {
    REQUIRE(WriteBlob(ChildDxbcPath(dir, i), corpus[i].dxbc));
    std::vector<uint8_t> dxil = ConvertStandalone(corpus[i].dxbc);
    REQUIRE_FALSE(dxil.empty());
    if (poison && poison[0]) {
      // One byte, same length: the shape a real conversion divergence takes.
      dxil[dxil.size() / 2] ^= 0x01;
    }
    REQUIRE(WriteBlob(ChildDxilPath(dir, i), dxil));
  }
  // Written last: the parent treats its absence as "the child did not run",
  // which must fail the gate rather than silently skip it.
  REQUIRE(WriteBlob(dir / "done", {0x01}));
}

TEST_CASE("The DXIL determinism corpus is distinct and non-empty",
          "[dxil_cache][determinism][premise]") {
  // Everything below asserts that artifacts MATCH. A converter that returned a
  // constant, or a corpus of three copies of one payload, would satisfy all of
  // it. This case is what makes the matches mean something.
  const auto& corpus = Corpus();
  REQUIRE(corpus.size() >= 3);

  std::vector<std::vector<uint8_t>> dxil;
  for (const Payload& payload : corpus) {
    REQUIRE_FALSE(payload.dxbc.empty());
    dxil.push_back(ConvertStandalone(payload.dxbc));
    INFO("payload " << payload.name);
    REQUIRE_FALSE(dxil.back().empty());
  }

  for (size_t i = 0; i < corpus.size(); ++i) {
    for (size_t j = i + 1; j < corpus.size(); ++j) {
      INFO(corpus[i].name << " vs " << corpus[j].name);
      CHECK(corpus[i].dxbc != corpus[j].dxbc);
      CHECK(dxil[i] != dxil[j]);
    }
  }
  for (size_t i = 0; i < corpus.size(); ++i) {
    std::printf("DXILGATE|CORPUS|%s dxbc=%zuB dxil=%zuB\n",
                corpus[i].name.c_str(), corpus[i].dxbc.size(), dxil[i].size());
  }
  std::fflush(stdout);
}

TEST_CASE("dxilconv output is stable across repeated conversions",
          "[dxil_cache][determinism]") {
  // The weakest form of the property, and the one a content-derived key most
  // directly depends on: the same input converted again yields the same bytes.
  const auto& corpus = Corpus();
  REQUIRE_FALSE(corpus.empty());
  constexpr int kRepeats = 4;

  for (const Payload& payload : corpus) {
    INFO("payload " << payload.name);
    const std::vector<uint8_t> reference = ConvertStandalone(payload.dxbc);
    REQUIRE_FALSE(reference.empty());
    for (int repeat = 1; repeat < kRepeats; ++repeat) {
      INFO("repeat " << repeat);
      const std::vector<uint8_t> again = ConvertStandalone(payload.dxbc);
      REQUIRE_FALSE(again.empty());
      const auto stats = DiffStats(reference, again);
      INFO("first_diff@" << stats.first << " diff_bytes=" << stats.second);
      CHECK(again == reference);
    }
  }
}

TEST_CASE("dxilconv output is independent of call order",
          "[dxil_cache][determinism][callorder]") {
  // THE case the converter-reuse bug violated. Every permutation of the corpus
  // is converted through ONE converter object, and every artifact must equal
  // the artifact that same payload produces on a converter that has done
  // nothing else.
  const auto& corpus = Corpus();
  REQUIRE(corpus.size() >= 3);

  std::vector<std::vector<uint8_t>> standalone;
  for (const Payload& payload : corpus) {
    standalone.push_back(ConvertStandalone(payload.dxbc));
    REQUIRE_FALSE(standalone.back().empty());
  }

  std::vector<size_t> order(corpus.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end());
  size_t permutations = 0;
  do {
    ++permutations;
    std::string order_desc;
    for (size_t index : order) {
      order_desc += corpus[index].name + " ";
    }
    INFO("order: " << order_desc);

    DxbcToDxilConverter converter;
    std::string error;
    REQUIRE(converter.Initialize(&error));
    for (size_t position = 0; position < order.size(); ++position) {
      const size_t index = order[position];
      INFO("position " << position << " payload " << corpus[index].name);
      const std::vector<uint8_t> produced =
          ConvertWith(converter, corpus[index].dxbc);
      REQUIRE_FALSE(produced.empty());
      const auto stats = DiffStats(standalone[index], produced);
      INFO("first_diff@" << stats.first << " diff_bytes=" << stats.second);
      CHECK(produced == standalone[index]);
    }
  } while (std::next_permutation(order.begin(), order.end()));

  // 3 payloads -> 6 permutations. A permutation count of 1 would mean the loop
  // degenerated and the case proved nothing about ordering.
  CHECK(permutations == 6);
  std::printf("DXILGATE|CALLORDER|permutations=%zu payloads=%zu\n",
              permutations, corpus.size());
  std::fflush(stdout);
}

TEST_CASE("dxilconv output is identical across processes",
          "[dxil_cache][determinism][crossprocess]") {
  // A durable cache is written by one process and read by another, so this is
  // the comparison the cache actually depends on. Two independent children,
  // because parent-vs-child alone cannot distinguish "deterministic" from
  // "both inherited the same accident of this process".
  const auto& corpus = Corpus();
  REQUIRE_FALSE(corpus.empty());

  const std::filesystem::path root = ScratchRoot("crossprocess");
  const std::filesystem::path child_a = root / "a";
  const std::filesystem::path child_b = root / "b";
  REQUIRE(RunChildConversion(child_a));
  REQUIRE(RunChildConversion(child_b));

  for (size_t i = 0; i < corpus.size(); ++i) {
    INFO("payload " << corpus[i].name);
    const std::vector<uint8_t> local = ConvertStandalone(corpus[i].dxbc);
    REQUIRE_FALSE(local.empty());

    // Ordered so the input is cleared before the output is judged: if the
    // translator were the nondeterministic party, the DXIL comparison would
    // fail for a reason that has nothing to do with dxilconv.
    const std::vector<uint8_t> dxbc_a = ReadBlob(ChildDxbcPath(child_a, i));
    const std::vector<uint8_t> dxbc_b = ReadBlob(ChildDxbcPath(child_b, i));
    REQUIRE_FALSE(dxbc_a.empty());
    CHECK(dxbc_a == corpus[i].dxbc);
    CHECK(dxbc_b == corpus[i].dxbc);

    const std::vector<uint8_t> dxil_a = ReadBlob(ChildDxilPath(child_a, i));
    const std::vector<uint8_t> dxil_b = ReadBlob(ChildDxilPath(child_b, i));
    REQUIRE_FALSE(dxil_a.empty());
    REQUIRE_FALSE(dxil_b.empty());
    const auto stats_a = DiffStats(local, dxil_a);
    const auto stats_ab = DiffStats(dxil_a, dxil_b);
    std::printf(
        "DXILGATE|CROSSPROC|%s local=%zuB childA=%zuB childB=%zuB "
        "local==A=%d A==B=%d first_diff@%zu diff_bytes=%zu\n",
        corpus[i].name.c_str(), local.size(), dxil_a.size(), dxil_b.size(),
        int(local == dxil_a), int(dxil_a == dxil_b), stats_a.first,
        stats_a.second);
    INFO("A vs B first_diff@" << stats_ab.first
                              << " diff_bytes=" << stats_ab.second);
    CHECK(dxil_a == local);
    CHECK(dxil_b == local);
  }
  std::fflush(stdout);

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("The cross-process comparison detects a divergent child",
          "[dxil_cache][determinism][crossprocess][negative]") {
  // The negative control for the case above. "The artifacts matched" and "the
  // comparison never ran" are the same observation from the outside, so the
  // harness is run once against a child instructed to diverge by exactly one
  // byte — the smallest signal a real conversion divergence could carry, and
  // the shape the converter-reuse corruption actually had.
  //
  // If this case ever passes trivially (child absent, files unread), the
  // REQUIREs below fail rather than the CHECKs, which is the distinction that
  // matters.
  const auto& corpus = Corpus();
  REQUIRE_FALSE(corpus.empty());

  const std::filesystem::path root = ScratchRoot("crossprocess_poison");
  REQUIRE(RunChildConversion(root, /*poison=*/true));

  size_t detected = 0;
  for (size_t i = 0; i < corpus.size(); ++i) {
    INFO("payload " << corpus[i].name);
    const std::vector<uint8_t> local = ConvertStandalone(corpus[i].dxbc);
    REQUIRE_FALSE(local.empty());
    const std::vector<uint8_t> poisoned = ReadBlob(ChildDxilPath(root, i));
    REQUIRE_FALSE(poisoned.empty());

    // Same length, one differing byte: nothing but a byte comparison sees it.
    REQUIRE(poisoned.size() == local.size());
    const auto stats = DiffStats(local, poisoned);
    CHECK(stats.second == 1);
    CHECK(local != poisoned);
    const MetalArtifactHitVerdict verdict =
        VerifyMetalArtifactHit(poisoned, local);
    CHECK(verdict == MetalArtifactHitVerdict::kMissByteMismatch);
    if (verdict == MetalArtifactHitVerdict::kMissByteMismatch) {
      ++detected;
    }
    std::printf(
        "DXILGATE|CROSSPROC-NEG|%s first_diff@%zu diff_bytes=%zu verdict=%s\n",
        corpus[i].name.c_str(), stats.first, stats.second,
        MetalArtifactHitVerdictName(verdict));
  }
  CHECK(detected == corpus.size());
  std::fflush(stdout);

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("The dxilconv option string changes the artifact and reaches the key",
          "[dxil_cache][determinism][options]") {
  // The premise behind putting converter options in the storage identity,
  // measured rather than assumed. If the two option strings produced identical
  // DXIL, the identity field would be dead weight and this case would say so.
  const auto& corpus = Corpus();
  REQUIRE_FALSE(corpus.empty());
  const std::vector<uint8_t>& dxbc = corpus.front().dxbc;

  const char* saved = std::getenv(kMetalDxbcToDxilOptionsEnvVar);
  const std::string saved_value = saved ? std::string(saved) : std::string();
  const bool had_override = saved != nullptr;
  auto restore = [&]() {
    if (had_override) {
      setenv(kMetalDxbcToDxilOptionsEnvVar, saved_value.c_str(), 1);
    } else {
      unsetenv(kMetalDxbcToDxilOptionsEnvVar);
    }
  };

  unsetenv(kMetalDxbcToDxilOptionsEnvVar);
  REQUIRE(MetalDxbcToDxilEffectiveOptions() ==
          std::string(kMetalDxbcToDxilDefaultOptions));
  const std::vector<uint8_t> default_dxil = ConvertStandalone(dxbc);
  const uint64_t default_key =
      ComputeMetalPipelineStorageIdentity(nullptr).converter_options_hash;

  setenv(kMetalDxbcToDxilOptionsEnvVar, " ", 1);
  REQUIRE(MetalDxbcToDxilEffectiveOptions() == std::string(" "));
  const std::vector<uint8_t> alternate_dxil = ConvertStandalone(dxbc);
  const uint64_t alternate_key =
      ComputeMetalPipelineStorageIdentity(nullptr).converter_options_hash;

  restore();

  REQUIRE_FALSE(default_dxil.empty());
  REQUIRE_FALSE(alternate_dxil.empty());
  std::printf(
      "DXILGATE|OPTIONS|default=%zuB alternate=%zuB differ=%d "
      "key_default=%016llX key_alternate=%016llX\n",
      default_dxil.size(), alternate_dxil.size(),
      int(default_dxil != alternate_dxil),
      (unsigned long long)default_key, (unsigned long long)alternate_key);
  std::fflush(stdout);

  // Output-affecting...
  CHECK(default_dxil != alternate_dxil);
  // ...therefore the key must move with it, or a warm cache written under one
  // option string is served under the other.
  CHECK(default_key != alternate_key);
  // And neither may collide with the value a manifest written before the field
  // existed carries in that slot.
  CHECK(default_key != 0ull);
  CHECK(alternate_key != 0ull);
}

TEST_CASE("A DXIL cache hit must equal a fresh conversion or miss",
          "[dxil_cache][determinism][hit_verify]") {
  // Item 4, over real artifacts. The positive leg comes first: if a correct
  // artifact did not verify as servable, the rejections below would prove
  // nothing beyond "the check rejects everything".
  const auto& corpus = Corpus();
  REQUIRE(corpus.size() >= 2);

  const std::vector<uint8_t> cached = ConvertStandalone(corpus[0].dxbc);
  const std::vector<uint8_t> other = ConvertStandalone(corpus[1].dxbc);
  REQUIRE_FALSE(cached.empty());
  REQUIRE_FALSE(other.empty());

  SECTION("an artifact produced by this build serves") {
    const std::vector<uint8_t> fresh = ConvertStandalone(corpus[0].dxbc);
    REQUIRE_FALSE(fresh.empty());
    CHECK(VerifyMetalArtifactHit(cached, fresh) ==
          MetalArtifactHitVerdict::kServe);
  }

  SECTION("a single corrupted byte misses") {
    // The dxilconv-corruption shape: a well-formed container of the right
    // length whose bitcode is wrong. Nothing short of a byte comparison sees
    // it.
    std::vector<uint8_t> poisoned = cached;
    poisoned[poisoned.size() / 2] ^= 0x01;
    const std::vector<uint8_t> fresh = ConvertStandalone(corpus[0].dxbc);
    REQUIRE_FALSE(fresh.empty());
    REQUIRE(poisoned.size() == fresh.size());
    CHECK(VerifyMetalArtifactHit(poisoned, fresh) ==
          MetalArtifactHitVerdict::kMissByteMismatch);
  }

  SECTION("a truncated entry misses") {
    std::vector<uint8_t> truncated(cached.begin(), cached.end() - 1);
    const std::vector<uint8_t> fresh = ConvertStandalone(corpus[0].dxbc);
    REQUIRE_FALSE(fresh.empty());
    CHECK_FALSE(MetalArtifactHitIsServable(
        VerifyMetalArtifactHit(truncated, fresh)));
  }

  SECTION("a valid artifact from a different payload misses") {
    // What a key collision or a stale entry actually looks like: not garbage,
    // but a perfectly well-formed artifact of the wrong shader. This is the
    // exact failure the converter-reuse bug produced and caches preserved.
    const std::vector<uint8_t> fresh = ConvertStandalone(corpus[0].dxbc);
    REQUIRE_FALSE(fresh.empty());
    const MetalArtifactHitVerdict verdict =
        VerifyMetalArtifactHit(other, fresh);
    INFO("verdict " << MetalArtifactHitVerdictName(verdict));
    CHECK_FALSE(MetalArtifactHitIsServable(verdict));
  }

  SECTION("an unverifiable hit misses") {
    // Fresh production failed. The cached bytes may well be right, but nothing
    // proved it, and "could not check" must not resolve to "checked out".
    const std::vector<uint8_t> no_fresh;
    CHECK(VerifyMetalArtifactHit(cached, no_fresh) ==
          MetalArtifactHitVerdict::kMissUnverifiable);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
