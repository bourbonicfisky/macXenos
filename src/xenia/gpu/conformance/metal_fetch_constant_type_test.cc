/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0: the texture fetch constant type contract every Metal serve site
// shares. The reference backends and the shared texture cache serve a
// constant typed "invalid texture" when --gpu_allow_invalid_fetch_constants
// is set (the upstream default); Gears of War 3's occlusion queries depend on
// it. Vertex-typed constants never serve as textures.

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_fetch_constant_type.h"

namespace xe {
namespace gpu {
namespace metal {

TEST_CASE("Texture-typed fetch constants always serve",
          "[metal][fetch-constant]") {
  CHECK(IsTextureFetchConstantServable(xenos::FetchConstantType::kTexture,
                                       false));
  CHECK(IsTextureFetchConstantServable(xenos::FetchConstantType::kTexture,
                                       true));
}

TEST_CASE("Invalid-texture-typed constants serve only when allowed",
          "[metal][fetch-constant]") {
  CHECK(IsTextureFetchConstantServable(
      xenos::FetchConstantType::kInvalidTexture, true));
  CHECK_FALSE(IsTextureFetchConstantServable(
      xenos::FetchConstantType::kInvalidTexture, false));
}

TEST_CASE("Vertex-typed constants never serve as textures",
          "[metal][fetch-constant]") {
  for (bool allow : {false, true}) {
    CHECK_FALSE(IsTextureFetchConstantServable(
        xenos::FetchConstantType::kVertex, allow));
    CHECK_FALSE(IsTextureFetchConstantServable(
        xenos::FetchConstantType::kInvalidVertex, allow));
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
