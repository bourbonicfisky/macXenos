/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// When a depth snapshot may stand in for a fetch larger than itself, decided
// by the real MetalTextureCache from its resolved-target registry.
//
// The shapes are Red Dead Redemption: Undead Nightmare's: a 128x128 scratch
// depth map resolved to the base of a 1600x1600 shadow cascade atlas whose
// cascades are separate depth resolves further into the same guest extent.
// Serving the atlas fetch from the scratch snapshot zero-fills every cascade.

#import <Metal/Metal.h>

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kAtlas = 0x040FB000;
constexpr uint32_t kAtlasSide = 1600;
constexpr uint32_t kScratchSide = 128;
constexpr uint32_t kScratchBytes = kScratchSide * kScratchSide * 4;
// The second cascade's resolve destination, inside the atlas extent.
constexpr uint32_t kCascade = 0x04114000;
constexpr uint32_t kCascadeBytes = 800 * 800 * 4;
// Past the end of the atlas (1600 * 1600 * 4 bytes from its base).
constexpr uint32_t kUnrelated = 0x04D00000;

id<MTLTexture> MakeSnapshot(id<MTLDevice> device, uint32_t side) {
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:side
                                                        height:side
                                                     mipmapped:NO];
  return [device newTextureWithDescriptor:desc];
}

}  // namespace

TEST_CASE("A depth snapshot does not bridge a fetch other resolves fill",
          "[l1][metal][reader][service]") {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  Memory* memory = conformance::TestPhysicalMemory();
  if (!device || !memory) {
    SUCCEED("no Metal device");
    return;
  }
  id<MTLBuffer> heap = [device newBufferWithLength:16 * 1024
                                           options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  MetalTextureCache cache((__bridge void*)device, memory, nullptr, (__bridge void*)heap, 64);
  cache.SetSubmissionProvider([]() { return uint64_t(1); });

  id<MTLTexture> scratch = MakeSnapshot(device, kScratchSide);
  id<MTLTexture> cascade = MakeSnapshot(device, 800);
  REQUIRE(scratch != nil);
  REQUIRE(cascade != nil);

  cache.RegisterResolvedTarget(kAtlas, (__bridge void*)scratch, kScratchSide, kScratchSide,
                               /*is_depth=*/true, /*requires_fetch_swizzle=*/false, kAtlas,
                               kScratchBytes);

  // Alone, the snapshot is the only content in the larger fetch (Batman:
  // Arkham Asylum's per-caster shadow maps): the fitted view serves it.
  CHECK(cache.BridgeServes(kAtlas, kAtlasSide, kAtlasSide, kAtlasSide));

  // A resolve past the fetch's extent changes nothing.
  cache.RegisterResolvedTarget(kUnrelated, (__bridge void*)cascade, 800, 800, true, false,
                               kUnrelated, kCascadeBytes);
  CHECK(cache.BridgeServes(kAtlas, kAtlasSide, kAtlasSide, kAtlasSide));

  // A strip or column joined into the same snapshot object is part of it.
  cache.RegisterResolvedTarget(kCascade, (__bridge void*)scratch, kScratchSide, kScratchSide,
                               true, false, kCascade, kScratchBytes);
  CHECK(cache.BridgeServes(kAtlas, kAtlasSide, kAtlasSide, kAtlasSide));

  // Another resolve's snapshot inside the extent: the fetch is a composition
  // only guest RAM holds.
  cache.RegisterResolvedTarget(kCascade, (__bridge void*)cascade, 800, 800, true, false,
                               kCascade, kCascadeBytes);
  CHECK_FALSE(cache.BridgeServes(kAtlas, kAtlasSide, kAtlasSide, kAtlasSide));
  // The pitch, not the width, sets the extent: a narrow fetch on the atlas
  // pitch still reaches the cascade's rows.
  CHECK_FALSE(cache.BridgeServes(kAtlas, 256, 256, kAtlasSide));
  // A fetch the snapshot covers is unaffected.
  CHECK(cache.BridgeServes(kAtlas, kScratchSide, kScratchSide, kScratchSide));
  // So is one whose extent ends before the cascade.
  CHECK(cache.BridgeServes(kAtlas, 160, 160, 160));

  // The answer follows the registry: retiring the cascades restores the bridge.
  cache.RetireResolvedTargetsNaming((__bridge void*)cascade);
  CHECK(cache.BridgeServes(kAtlas, kAtlasSide, kAtlasSide, kAtlasSide));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
