/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <array>
#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_binding_layout.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

struct BindingStub {
  uint32_t bindless_descriptor_index;
};

TEST_CASE("Metal binding layout owns root and argument slot order",
          "[l0][metal][binding-parity]") {
  constexpr std::array<MetalBindingKind, kMetalBindingSlotCount> kKinds = {
      MetalBindingKind::kSrvTable,
      MetalBindingKind::kSrvTable,
      MetalBindingKind::kSrvTable,
      MetalBindingKind::kSrvTable,
      MetalBindingKind::kSrvTable,
      MetalBindingKind::kUavTable,
      MetalBindingKind::kUavTable,
      MetalBindingKind::kUavTable,
      MetalBindingKind::kUavTable,
      MetalBindingKind::kSamplerTable,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kConstantBuffer,
      MetalBindingKind::kFunctionConstantTable,
  };
  constexpr std::array<uint32_t, kMetalBindingSlotCount> kRegisters = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
      3, 4, 1, 3, 4, 1, 3, 4, 1, 3, 4, 0};
  constexpr std::array<uint32_t, kMetalBindingSlotCount> kSpaces = {
      0, 1, 2, 3, 10,
      0, 1, 2, 3, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, kMetalFunctionConstantRegisterSpace};
  constexpr std::array<MetalBindingVisibility, kMetalBindingSlotCount>
      kVisibilities = {
          MetalBindingVisibility::kAll,    MetalBindingVisibility::kAll,
          MetalBindingVisibility::kAll,    MetalBindingVisibility::kAll,
          MetalBindingVisibility::kAll,    MetalBindingVisibility::kAll,
          MetalBindingVisibility::kAll,    MetalBindingVisibility::kAll,
          MetalBindingVisibility::kAll,    MetalBindingVisibility::kAll,
          MetalBindingVisibility::kAll,    MetalBindingVisibility::kVertex,
          MetalBindingVisibility::kAll,    MetalBindingVisibility::kVertex,
          MetalBindingVisibility::kVertex, MetalBindingVisibility::kHull,
          MetalBindingVisibility::kHull,   MetalBindingVisibility::kHull,
          MetalBindingVisibility::kDomain, MetalBindingVisibility::kDomain,
          MetalBindingVisibility::kDomain, MetalBindingVisibility::kPixel,
          MetalBindingVisibility::kPixel,  MetalBindingVisibility::kPixel,
          MetalBindingVisibility::kAll,
      };
  REQUIRE(kMetalBindingLayout.size() == kMetalBindingSlotCount);
  CHECK(kMetalBindingSlotCount == 25);
  CHECK(IsMetalBindingLayoutValid());
  for (uint32_t i = 0; i < kMetalBindingSlotCount; ++i) {
    CHECK(MetalBindingSlotIndex(kMetalBindingLayout[i].slot) == i);
    CHECK(kMetalBindingLayout[i].kind == kKinds[i]);
    CHECK(kMetalBindingLayout[i].shader_register == kRegisters[i]);
    CHECK(kMetalBindingLayout[i].register_space == kSpaces[i]);
    CHECK(kMetalBindingLayout[i].visibility == kVisibilities[i]);
  }
}

TEST_CASE("Metal binding layout preserves descriptor count policy",
          "[l0][metal][binding-parity]") {
  const MetalBindingParameter& srv =
      kMetalBindingLayout[MetalBindingSlotIndex(MetalBindingSlot::kSrvSpace1)];
  const MetalBindingParameter& sampler =
      kMetalBindingLayout[MetalBindingSlotIndex(
          MetalBindingSlot::kSamplerSpace0)];
  const MetalBindingParameter& function_constants =
      kMetalBindingLayout[MetalBindingSlotIndex(
          MetalBindingSlot::kFunctionConstants)];
  const MetalBindingParameter& cbv =
      kMetalBindingLayout[MetalBindingSlotIndex(MetalBindingSlot::kCbvSystem)];
  CHECK(GetMetalBindingDescriptorCount(srv, true) == UINT32_MAX);
  CHECK(GetMetalBindingDescriptorCount(sampler, true) == UINT32_MAX);
  CHECK(GetMetalBindingDescriptorCount(srv, false) ==
        kMetalBindfulResourceDescriptorCount);
  CHECK(GetMetalBindingDescriptorCount(sampler, false) ==
        kMetalBindfulSamplerDescriptorCount);
  CHECK(GetMetalBindingDescriptorCount(function_constants, true) == 1);
  CHECK(GetMetalBindingDescriptorCount(cbv, true) == 0);
}

TEST_CASE("Metal float bindings pack sparse registers in ascending order",
          "[l0][metal][binding-parity]") {
  RegisterFile regs;
  Shader::ConstantRegisterMap map = {};
  map.float_bitmap[0] = (uint64_t(1) << 0) | (uint64_t(1) << 2);
  map.float_bitmap[1] = uint64_t(1) << 1;
  map.float_count = 3;
  constexpr uint32_t kBase = XE_GPU_REG_SHADER_CONSTANT_000_X;
  const uint32_t constant_indices[] = {0, 2, 65};
  for (uint32_t constant_index : constant_indices) {
    for (uint32_t component = 0; component < 4; ++component) {
      regs[kBase + constant_index * 4 + component] =
          0xA0000000u | (constant_index << 4) | component;
    }
  }

  std::array<uint32_t, 12> packed = {};
  REQUIRE(GetMetalPackedFloatConstantCount(map) == 3);
  REQUIRE(
      PackMetalFloatConstants(regs, map, kBase, packed.data(), sizeof(packed)));
  for (uint32_t packed_index = 0; packed_index < 3; ++packed_index) {
    for (uint32_t component = 0; component < 4; ++component) {
      CHECK(packed[packed_index * 4 + component] ==
            (0xA0000000u | (constant_indices[packed_index] << 4) | component));
    }
  }
}

TEST_CASE("Metal float bindings provide a zero null payload",
          "[l0][metal][binding-parity]") {
  RegisterFile regs;
  Shader::ConstantRegisterMap map = {};
  std::array<uint32_t, 4> packed = {1, 2, 3, 4};
  CHECK(PackMetalFloatConstants(regs, map, XE_GPU_REG_SHADER_CONSTANT_000_X,
                                packed.data(), sizeof(packed)));
  CHECK(packed == std::array<uint32_t, 4>{});
  CHECK_FALSE(PackMetalFloatConstants(
      regs, map, XE_GPU_REG_SHADER_CONSTANT_000_X, packed.data(), 4));
}

TEST_CASE("Metal validates Xenos constant bank encodings",
          "[l0][metal][binding-parity]") {
  CHECK(AreMetalConstantBankBasesSupported(0x000FF000u, 0x000FF100u));
  CHECK(AreMetalConstantBankBasesSupported(0, 0));
  CHECK_FALSE(AreMetalConstantBankBasesSupported(0x000FF000u, 0xDEADBEEFu));
  CHECK_FALSE(AreMetalConstantBankBasesSupported(0x000FE000u, 0x000FF100u));
  CHECK_FALSE(AreMetalConstantBankBasesSupported(0x000FF000u, 0x000FE100u));
}

TEST_CASE("Metal descriptor plans cover sparse flat indices",
          "[l0][metal][binding-parity]") {
  std::vector<BindingStub> textures = {{1}, {6}};
  std::vector<BindingStub> samplers = {{3}};
  MetalDescriptorIndexPlan plan =
      GetMetalDescriptorIndexPlan(textures, samplers);
  CHECK(plan.valid);
  CHECK(plan.count == 7);
  std::array<uint32_t, 7> indices;
  indices.fill(UINT32_MAX);
  REQUIRE(
      InitializeMetalDescriptorIndices(plan, indices.data(), sizeof(indices)));
  CHECK(indices == std::array<uint32_t, 7>{});
  CHECK_FALSE(InitializeMetalDescriptorIndices(plan, indices.data(),
                                               sizeof(uint32_t) * 6));

  textures.clear();
  samplers.clear();
  plan = GetMetalDescriptorIndexPlan(textures, samplers);
  CHECK(plan.valid);
  CHECK(plan.count == 1);
}

TEST_CASE("Metal descriptor plans reject collisions and overflow",
          "[l0][metal][binding-parity]") {
  std::vector<BindingStub> textures = {{4}};
  std::vector<BindingStub> samplers = {{4}};
  CHECK_FALSE(GetMetalDescriptorIndexPlan(textures, samplers).valid);

  samplers = {{kMetalMaxDescriptorIndexCount}};
  MetalDescriptorIndexPlan invalid_plan =
      GetMetalDescriptorIndexPlan(textures, samplers);
  CHECK_FALSE(invalid_plan.valid);
  std::array<uint32_t, 1> indices = {UINT32_MAX};
  CHECK_FALSE(InitializeMetalDescriptorIndices(invalid_plan, indices.data(),
                                               sizeof(indices)));
}

TEST_CASE("Metal top-level bindings keep inactive stages null",
          "[l0][metal][binding-parity]") {
  MetalBindingAddresses addresses;
  addresses.null_buffer = 10;
  addresses.view_heap = 20;
  addresses.sampler_heap = 30;
  addresses.shared_memory_srv = 40;
  addresses.null_uav = 50;
  addresses.shared_memory_uav = 60;
  addresses.system_constants = 70;
  addresses.float_vertex = 80;
  addresses.bool_loop = 90;
  addresses.fetch = 100;
  addresses.descriptor_indices_vertex = 110;
  addresses.float_pixel = 120;
  addresses.descriptor_indices_pixel = 130;

  auto normal = BuildMetalTopLevelArgumentBuffer(addresses, false, false);
  const std::array<uint64_t, kMetalBindingSlotCount> expected_normal = {
      40,  20,  20, 20, 20, 50, 20, 20, 20, 30, 70, 80, 90,
      100, 110, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
  CHECK(normal == expected_normal);

  auto exporting = BuildMetalTopLevelArgumentBuffer(addresses, true, true);
  const std::array<uint64_t, kMetalBindingSlotCount> expected_exporting = {
      40,  20,  20, 20, 20, 60, 20, 20, 20,  30,  70,  80, 90,
      100, 110, 10, 10, 10, 10, 10, 10, 120, 100, 130, 10};
  CHECK(exporting == expected_exporting);
}

TEST_CASE(
    "Metal top-level bindings populate hull/domain slots only when "
    "tessellating",
    "[l0][metal][binding-parity][tessellation]") {
  MetalBindingAddresses addresses;
  addresses.null_buffer = 10;
  addresses.view_heap = 20;
  addresses.sampler_heap = 30;
  addresses.shared_memory_srv = 40;
  addresses.null_uav = 50;
  addresses.shared_memory_uav = 60;
  addresses.system_constants = 70;
  addresses.float_vertex = 80;
  addresses.bool_loop = 90;
  addresses.fetch = 100;
  addresses.descriptor_indices_vertex = 110;
  addresses.float_pixel = 120;
  addresses.descriptor_indices_pixel = 130;

  // Tessellated draw with a guest PS: the six hull/domain CBV slots alias the
  // VERTEX-stage allocations (the domain stage IS the guest VS translation;
  // the HS consumes the same guest constant state) — XeniOS
  // BuildGraphicsRootArgumentEntries :4247-4264 ported into the pure builder.
  auto tessellated =
      BuildMetalTopLevelArgumentBuffer(addresses, true, false, true);
  const std::array<uint64_t, kMetalBindingSlotCount> expected_tessellated = {
      40,  20,  20, 20,  20,  50,  20, 20,  20,  30,  70,  80, 90,
      100, 110, 80, 100, 110, 80, 100, 110, 120, 100, 130, 10};
  CHECK(tessellated == expected_tessellated);

  // Depth-only tessellated draw (no guest PS -> the pipeline's depth-only
  // fallback): hull/domain populated, pixel slots stay null.
  auto tessellated_no_ps =
      BuildMetalTopLevelArgumentBuffer(addresses, false, false, true);
  CHECK(tessellated_no_ps[MetalBindingSlotIndex(
            MetalBindingSlot::kCbvFloatHull)] == 80);
  CHECK(tessellated_no_ps[MetalBindingSlotIndex(
            MetalBindingSlot::kCbvFetchDomain)] == 100);
  CHECK(tessellated_no_ps[MetalBindingSlotIndex(
            MetalBindingSlot::kCbvFloatPixel)] == 10);

  // Regression guard for every non-tessellated draw: the explicit-false form
  // and the defaulted form are identical, hull/domain stay at the null
  // buffer (a silent null-slot regression is what the oracle exists for).
  auto normal_default = BuildMetalTopLevelArgumentBuffer(addresses, true,
                                                         false);
  auto normal_explicit =
      BuildMetalTopLevelArgumentBuffer(addresses, true, false, false);
  CHECK(normal_default == normal_explicit);
  for (MetalBindingSlot slot :
       {MetalBindingSlot::kCbvFloatHull, MetalBindingSlot::kCbvFetchHull,
        MetalBindingSlot::kCbvDescriptorIndicesHull,
        MetalBindingSlot::kCbvFloatDomain, MetalBindingSlot::kCbvFetchDomain,
        MetalBindingSlot::kCbvDescriptorIndicesDomain}) {
    CHECK(normal_default[MetalBindingSlotIndex(slot)] ==
          addresses.null_buffer);
  }
}

TEST_CASE("Metal binding stage plans cover geometry emulation",
          "[l0][metal][binding-parity]") {
  CHECK(GetMetalBindingStageMask(false) ==
        (kMetalBindingStageVertex | kMetalBindingStageFragment));
  CHECK(GetMetalBindingStageMask(true) ==
        (kMetalBindingStageObject | kMetalBindingStageMesh |
         kMetalBindingStageFragment));
}




}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
