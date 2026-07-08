/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_METAL_METAL_IMMEDIATE_DRAWER_H_
#define XENIA_UI_METAL_METAL_IMMEDIATE_DRAWER_H_

#include <cstdint>
#include <memory>

#include "xenia/ui/immediate_drawer.h"

namespace xe {
namespace ui {
namespace metal {

// Phase 2b.3 placeholder immediate drawer. The Metal present path can clear and
// present a frame before the UI (ImGui) overlay is ported, so this drawer is a
// no-op: it accepts batches but draws nothing. Replaced by a real Metal
// immediate drawer when the UI overlay is brought up.
class MetalImmediateDrawer : public ImmediateDrawer {
 public:
  MetalImmediateDrawer() = default;
  ~MetalImmediateDrawer() override = default;

  std::unique_ptr<ImmediateTexture> CreateTexture(uint32_t width,
                                                  uint32_t height,
                                                  ImmediateTextureFilter filter,
                                                  bool is_repeated,
                                                  const uint8_t* data) override;

  void BeginDrawBatch(const ImmediateDrawBatch& batch) override;
  void Draw(const ImmediateDraw& draw) override;
  void EndDrawBatch() override;
};

}  // namespace metal
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_METAL_METAL_IMMEDIATE_DRAWER_H_
