/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/metal/metal_immediate_drawer.h"

namespace xe {
namespace ui {
namespace metal {

namespace {
// Trivial texture so ImGui's font-atlas upload has something to hold onto even
// though the no-op drawer never samples it.
class MetalImmediateTexture : public ImmediateTexture {
 public:
  MetalImmediateTexture(uint32_t width, uint32_t height)
      : ImmediateTexture(width, height) {}
};
}  // namespace

std::unique_ptr<ImmediateTexture> MetalImmediateDrawer::CreateTexture(
    uint32_t width, uint32_t height, ImmediateTextureFilter filter,
    bool is_repeated, const uint8_t* data) {
  return std::make_unique<MetalImmediateTexture>(width, height);
}

void MetalImmediateDrawer::BeginDrawBatch(const ImmediateDrawBatch& batch) {}
void MetalImmediateDrawer::Draw(const ImmediateDraw& draw) {}
void MetalImmediateDrawer::EndDrawBatch() {}

}  // namespace metal
}  // namespace ui
}  // namespace xe
