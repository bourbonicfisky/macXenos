/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Device-free conformance for the draw-pass attachment-use predicate
// (metal_pass_attachment_use.h), the hysteresis binding decision, and the
// pass census accumulator (metal_pass_census.h) of
// reduce-metal-draw-pass-traffic. Every read and write path the design D2
// enumerates has a case: colour write mask after MRT routing, blend reads,
// depth test and write, pixel-shader depth export, stencil test and write.

#include <cstring>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_pass_attachment_use.h"
#include "xenia/gpu/metal/metal_pass_census.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::DecideMetalPassBinding;
using metal::GetMetalPassAttachmentUse;
using metal::kMetalPassAttachmentDepth;
using metal::kMetalPassAttachmentStencil;
using metal::MetalPassAttachmentUse;
using metal::MetalPassCensus;
using metal::MetalPassColorAttachmentBit;
using metal::MetalPassEndReason;

constexpr uint32_t kC0 = MetalPassColorAttachmentBit(0);
constexpr uint32_t kC1 = MetalPassColorAttachmentBit(1);
constexpr uint32_t kC3 = MetalPassColorAttachmentBit(3);
constexpr uint32_t kD = kMetalPassAttachmentDepth;
constexpr uint32_t kS = kMetalPassAttachmentStencil;

uint32_t ReplaceBlend() {
  reg::RB_BLENDCONTROL blend{};
  blend.color_srcblend = xenos::BlendFactor::kOne;
  blend.color_destblend = xenos::BlendFactor::kZero;
  blend.color_comb_fcn = xenos::BlendOp::kAdd;
  blend.alpha_srcblend = xenos::BlendFactor::kOne;
  blend.alpha_destblend = xenos::BlendFactor::kZero;
  blend.alpha_comb_fcn = xenos::BlendOp::kAdd;
  return blend.value;
}

uint32_t AlphaBlend() {
  reg::RB_BLENDCONTROL blend{};
  blend.color_srcblend = xenos::BlendFactor::kSrcAlpha;
  blend.color_destblend = xenos::BlendFactor::kOneMinusSrcAlpha;
  blend.color_comb_fcn = xenos::BlendOp::kAdd;
  blend.alpha_srcblend = xenos::BlendFactor::kOne;
  blend.alpha_destblend = xenos::BlendFactor::kZero;
  blend.alpha_comb_fcn = xenos::BlendOp::kAdd;
  return blend.value;
}

struct DrawState {
  reg::RB_DEPTHCONTROL depth{};
  uint32_t color_mask = 0;
  uint32_t blends[4] = {ReplaceBlend(), ReplaceBlend(), ReplaceBlend(),
                        ReplaceBlend()};
  uint32_t stencil_ref_mask = 0;
  uint32_t stencil_ref_mask_bf = 0;
  bool ps_depth = false;

  MetalPassAttachmentUse Use() const {
    return GetMetalPassAttachmentUse(depth, color_mask, blends,
                                     stencil_ref_mask, stencil_ref_mask_bf,
                                     ps_depth);
  }
};

uint32_t StencilMask(uint32_t read, uint32_t write) {
  reg::RB_STENCILREFMASK m{};
  m.stencilmask = read;
  m.stencilwritemask = write;
  return m.value;
}

}  // namespace

TEST_CASE("Pass attachment use: colour write mask after MRT routing",
          "[metal][pass-traffic]") {
  DrawState s;
  s.color_mask = 0xF;  // RT0 only
  MetalPassAttachmentUse u = s.Use();
  REQUIRE(u.writes == kC0);
  REQUIRE(u.reads == 0);

  s.color_mask = 0xF00F;  // RT0 + RT3
  u = s.Use();
  REQUIRE(u.writes == (kC0 | kC3));
  REQUIRE(u.used() == (kC0 | kC3));

  // A partial mask still writes (and the pass keeps Load for the rest).
  s.color_mask = 0x0070;
  u = s.Use();
  REQUIRE(u.writes == kC1);

  // A zero mask (RB_COLOR_MASK off, or the shader never writes RT0) binds
  // nothing.
  s.color_mask = 0;
  u = s.Use();
  REQUIRE(u.used() == 0);
}

TEST_CASE("Pass attachment use: blending reads the colour target",
          "[metal][pass-traffic]") {
  DrawState s;
  s.color_mask = 0xF;
  s.blends[0] = AlphaBlend();
  MetalPassAttachmentUse u = s.Use();
  REQUIRE(u.reads == kC0);
  REQUIRE(u.writes == kC0);

  // Blending on a target the mask does not write is not a read either.
  s.color_mask = 0xF0;
  s.blends[1] = ReplaceBlend();
  u = s.Use();
  REQUIRE(u.reads == 0);
  REQUIRE(u.writes == kC1);
}

TEST_CASE("Pass attachment use: depth test, depth write, depth export",
          "[metal][pass-traffic]") {
  DrawState s;
  s.color_mask = 0xF;
  MetalPassAttachmentUse u = s.Use();
  REQUIRE((u.used() & (kD | kS)) == 0);

  s.depth.z_enable = 1;
  s.depth.zfunc = xenos::CompareFunction::kLessEqual;
  u = s.Use();
  REQUIRE(u.reads == kD);
  REQUIRE((u.writes & kD) == 0);

  s.depth.z_write_enable = 1;
  u = s.Use();
  REQUIRE((u.writes & kD) == kD);
  REQUIRE((u.used() & kS) == 0);

  // The pixel shader exporting depth binds depth even with the test off.
  DrawState e;
  e.color_mask = 0xF;
  e.ps_depth = true;
  u = e.Use();
  REQUIRE((u.writes & kD) == kD);
  REQUIRE((u.used() & kS) == 0);
}

TEST_CASE("Pass attachment use: stencil test and write", "[metal][pass-traffic]") {
  DrawState s;
  s.color_mask = 0xF;
  s.depth.stencil_enable = 1;
  s.depth.stencilfunc = xenos::CompareFunction::kEqual;
  s.depth.stencilfail = xenos::StencilOp::kKeep;
  s.depth.stencilzfail = xenos::StencilOp::kKeep;
  s.depth.stencilzpass = xenos::StencilOp::kKeep;
  s.stencil_ref_mask = StencilMask(0xFF, 0xFF);
  MetalPassAttachmentUse u = s.Use();
  // Test only: a read of the stencil plane, no depth (replace blending does
  // not read the colour target).
  REQUIRE(u.reads == kS);
  REQUIRE((u.writes & kS) == 0);
  REQUIRE((u.used() & kD) == 0);

  // KEEP everywhere with a write mask is still no write; a REPLACE with a
  // zero write mask is no write; both together write.
  s.depth.stencilzpass = xenos::StencilOp::kReplace;
  s.stencil_ref_mask = StencilMask(0xFF, 0);
  u = s.Use();
  REQUIRE((u.writes & kS) == 0);
  s.stencil_ref_mask = StencilMask(0xFF, 0xFF);
  u = s.Use();
  REQUIRE((u.writes & kS) == kS);

  // Back-face ops only count with backface_enable.
  DrawState b;
  b.color_mask = 0xF;
  b.depth.stencil_enable = 1;
  b.depth.stencilzpass_bf = xenos::StencilOp::kIncrementClamp;
  b.stencil_ref_mask = StencilMask(0xFF, 0xFF);
  b.stencil_ref_mask_bf = StencilMask(0xFF, 0xFF);
  u = b.Use();
  REQUIRE((u.writes & kS) == 0);
  b.depth.backface_enable = 1;
  u = b.Use();
  REQUIRE((u.writes & kS) == kS);
}

TEST_CASE("Pass binding decision: hysteresis and the attachment-needed split",
          "[metal][pass-traffic]") {
  const uint32_t present = kD | kS | kC0;
  // No pass open: bind the draw's own use, restricted to what is present.
  auto d = DecideMetalPassBinding(present, kC0 | kC1, false, 0);
  REQUIRE(d.bound == kC0);
  REQUIRE(!d.reuse);
  REQUIRE(!d.split);
  // A covered draw reuses the open pass as bound.
  d = DecideMetalPassBinding(present, kC0, true, kD | kC0);
  REQUIRE(d.reuse);
  REQUIRE(d.bound == (kD | kC0));
  // A draw needing more splits and the next segment binds the union.
  d = DecideMetalPassBinding(present, kD | kC0, true, kC0);
  REQUIRE(d.split);
  REQUIRE(d.bound == (kD | kC0));
  // After the union, an alternating no-depth draw no longer splits.
  d = DecideMetalPassBinding(present, kC0, true, kD | kC0);
  REQUIRE(d.reuse);
}

TEST_CASE("Pass census: avoidable traffic and the D2 simulation",
          "[metal][pass-traffic]") {
  MetalPassCensus census;
  // 1280x720 BGRA8 colour (3,686,400 B), D32 depth (3,686,400 B), S8 stencil
  // (921,600 B).
  uint64_t bytes[metal::kMetalPassAttachmentCount] = {3686400, 921600, 3686400,
                                                      0, 0, 0};
  const uint32_t legacy_bound = kD | kS | kC0;

  // Pass 1: two HUD draws, colour only. Depth and stencil are avoidable.
  census.BeginPass(/*key=*/1, legacy_bound, bytes);
  MetalPassAttachmentUse hud;
  hud.writes = kC0;
  census.NoteDraw(hud);
  census.NoteDraw(hud);
  census.EndPass(MetalPassEndReason::kAttachmentKey, 0, false);
  const MetalPassCensus::Totals& w = census.window();
  REQUIRE(w.passes == 1);
  REQUIRE(w.draws == 2);
  REQUIRE(w.traffic_bytes == 2 * (3686400 + 921600 + 3686400));
  REQUIRE(w.avoidable_bytes == 2 * (3686400 + 921600));
  REQUIRE(w.sim_bytes == 2 * 3686400);
  REQUIRE(w.sim_splits == 0);

  // Pass 2 (different key): colour-only draw, then a depth-tested draw, then
  // colour-only again. D2 would split once and bind D|C0 for the rest.
  census.BeginPass(/*key=*/2, legacy_bound, bytes);
  census.NoteDraw(hud);
  MetalPassAttachmentUse scene;
  scene.reads = kD;
  scene.writes = kD | kC0;
  census.NoteDraw(scene);
  census.NoteDraw(hud);
  census.EndPass(MetalPassEndReason::kColorResolve, 0, true);
  REQUIRE(w.passes == 2);
  REQUIRE(w.sim_splits == 1);
  // Only the stencil plane was avoidable in pass 2.
  REQUIRE(w.avoidable_bytes == 2 * (3686400 + 921600) + 2 * 921600);
  REQUIRE(w.sim_bytes == 2 * 3686400 + 2 * 3686400 + 2 * (3686400 + 3686400));
  REQUIRE(w.resolve_own[0] == 1);
  REQUIRE(w.resolve_other[0] == 0);

  // Pass 3 resumes key 2 after the resolve: a same-key resume for cres.
  census.BeginPass(/*key=*/2, legacy_bound, bytes);
  census.NoteDraw(scene);
  census.EndPass(MetalPassEndReason::kSubmission, /*seam=*/3, false);
  REQUIRE(w.resumes[uint32_t(MetalPassEndReason::kColorResolve)] == 1);
  REQUIRE(w.seam_ends[3] == 1);
  REQUIRE(w.bound_set_passes[legacy_bound] == 3);
  REQUIRE(MetalPassCensus::BoundSetName(legacy_bound) == "DSC0");

  census.NoteFrame();
  std::string line = census.Format("test", /*window=*/true);
  REQUIRE(line.find("passes/frame=3.0") != std::string::npos);
  REQUIRE(line.find("cres=1.0(resume 1.0)[own 1.0 other 0.0]") !=
          std::string::npos);
  // The window reset; the lifetime record kept everything.
  REQUIRE(census.window().passes == 0);
  REQUIRE(census.lifetime().passes == 3);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
