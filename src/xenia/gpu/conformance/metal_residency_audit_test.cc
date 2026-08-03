/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0: the encoded-vs-resident decision matrix for port-xenios-tessellation's
// final attribution slice. Every classification is pinned INDEPENDENTLY, so a
// change that collapses two of them fails a specific case rather than blurring
// one aggregate assertion -- the staging rule applied to the apparatus itself,
// after three consecutive defects were found in the episode tracker exactly
// because its tests encoded a model of the mechanism rather than its real
// event ordering.

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_residency_audit.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kVertex = 1u << 0;
constexpr uint32_t kObject = 1u << 1;
constexpr uint32_t kMesh = 1u << 2;
constexpr uint32_t kFragment = 1u << 3;

MetalEncoderResidencyRecord ConsistentRecord() {
  MetalEncoderResidencyRecord r;
  r.submission = 100;
  r.rt_hash = 0x5AB4CBBEA5BE4F56ull;
  r.tessellation = true;
  r.encoded = {{/*slot=*/7, /*resource_id=*/42, /*generation=*/3,
                /*required=*/kObject | kMesh | kFragment}};
  r.declared = {{/*resource_id=*/42, /*generation=*/3,
                 /*declared=*/kObject | kMesh | kFragment}};
  return r;
}

TEST_CASE("Residency audit: a matched encoder is consistent",
          "[l0][metal][residency-audit]") {
  const MetalEncoderResidencyRecord r = ConsistentRecord();
  CHECK(ClassifyMetalEncoderResidency(r) == 0);
  CHECK(MetalResidencyRecordIsConsistent(r));
}

TEST_CASE("Residency audit: encoded resource omitted from declarations",
          "[l0][metal][residency-audit]") {
  // The plainest form of the hypothesis: the shader can dereference a resource
  // Metal was never told about.
  MetalEncoderResidencyRecord r = ConsistentRecord();
  r.declared.clear();
  CHECK((ClassifyMetalEncoderResidency(r) &
         uint32_t(MetalResidencyVerdict::kResourceNotDeclared)) != 0);
  CHECK_FALSE(MetalResidencyRecordIsConsistent(r));

  // A declaration for a DIFFERENT resource must not satisfy it.
  r.declared = {{/*resource_id=*/43, /*generation=*/3, kObject | kMesh}};
  CHECK((ClassifyMetalEncoderResidency(r) &
         uint32_t(MetalResidencyVerdict::kResourceNotDeclared)) != 0);
}

TEST_CASE("Residency audit: slot rewritten between encode and sweep",
          "[l0][metal][residency-audit]") {
  // Same resource id, different slot generation: the descriptor the draw
  // encoded is not the descriptor that was made resident.
  MetalEncoderResidencyRecord r = ConsistentRecord();
  r.declared[0].generation = 4;
  const uint32_t v = ClassifyMetalEncoderResidency(r);
  CHECK((v & uint32_t(MetalResidencyVerdict::kGenerationMismatch)) != 0);
  // Must NOT be misreported as "never declared" -- the two are different
  // defects with different fixes, so they are pinned apart.
  CHECK((v & uint32_t(MetalResidencyVerdict::kResourceNotDeclared)) == 0);
}

TEST_CASE("Residency audit: correct resource, missing required stage",
          "[l0][metal][residency-audit]") {
  // The all-stage-mask arm's hypothesis, expressed as an invariant: a wrongly
  // NARROW declaration is a fault even though the resource is resident.
  MetalEncoderResidencyRecord r = ConsistentRecord();
  r.declared[0].declared_stage_mask = kObject | kFragment;  // mesh missing
  const uint32_t v = ClassifyMetalEncoderResidency(r);
  CHECK((v & uint32_t(MetalResidencyVerdict::kMissingRequiredStage)) != 0);
  CHECK((v & uint32_t(MetalResidencyVerdict::kResourceNotDeclared)) == 0);
  CHECK((v & uint32_t(MetalResidencyVerdict::kGenerationMismatch)) == 0);

  // A WIDER declaration than required is fine -- widening can only add.
  r.declared[0].declared_stage_mask =
      kVertex | kObject | kMesh | kFragment;
  CHECK(ClassifyMetalEncoderResidency(r) == 0);
}

TEST_CASE("Residency audit: a slot naming nothing is not a residency question",
          "[l0][metal][residency-audit]") {
  MetalEncoderResidencyRecord r = ConsistentRecord();
  r.encoded[0].resource_id = 0;
  r.declared.clear();
  CHECK(ClassifyMetalEncoderResidency(r) == 0);
}

TEST_CASE("Residency audit: classifications compose without masking",
          "[l0][metal][residency-audit]") {
  // Two encoded slots failing DIFFERENT ways must both be reported: an
  // aggregate that stopped at the first failure would hide a second defect.
  MetalEncoderResidencyRecord r = ConsistentRecord();
  r.encoded.push_back({/*slot=*/9, /*resource_id=*/77, /*generation=*/1,
                       /*required=*/kObject});
  r.declared[0].declared_stage_mask = kObject | kFragment;  // mesh missing
  const uint32_t v = ClassifyMetalEncoderResidency(r);
  CHECK((v & uint32_t(MetalResidencyVerdict::kMissingRequiredStage)) != 0);
  CHECK((v & uint32_t(MetalResidencyVerdict::kResourceNotDeclared)) != 0);
}

TEST_CASE("Residency capture store never evicts an unfinished record",
          "[l0][metal][residency-audit]") {
  // A fault storm is exactly when eviction would destroy the evidence, so
  // exhaustion must FAIL LOUDLY rather than silently drop -- a dropped record
  // would read as "no mismatch found".
  MetalResidencyCaptureStore store(2);
  MetalEncoderResidencyRecord a = ConsistentRecord();
  a.submission = 10;
  MetalEncoderResidencyRecord b = ConsistentRecord();
  b.submission = 11;
  MetalEncoderResidencyRecord c = ConsistentRecord();
  c.submission = 12;
  CHECK(store.Add(a));
  CHECK(store.Add(b));
  CHECK(store.healthy());
  // Full: refused, counted, and the existing records are UNTOUCHED.
  CHECK_FALSE(store.Add(c));
  CHECK(store.size() == 2);
  CHECK(store.dropped() == 1);
  CHECK_FALSE(store.healthy());

  // Draining only retires completed submissions; unfinished ones stay.
  auto drained = store.Drain(10);
  REQUIRE(drained.size() == 1);
  CHECK(drained[0].submission == 10);
  CHECK(store.size() == 1);
  auto rest = store.Drain(11);
  REQUIRE(rest.size() == 1);
  CHECK(rest[0].submission == 11);
  CHECK(store.size() == 0);
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
