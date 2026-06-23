/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_SURFACE_MAC_H_
#define XENIA_UI_SURFACE_MAC_H_

#include <cstdint>

#include "xenia/ui/surface.h"

namespace xe {
namespace ui {

// A surface backed by a CAMetalLayer, presented to via VK_EXT_metal_surface
// (MoltenVK). The layer is stored as an opaque pointer so this header can be
// included from plain C++ translation units (e.g. the Vulkan presenter); the
// concrete CAMetalLayer access lives in surface_mac.mm.
class MetalLayerSurface : public Surface {
 public:
  explicit MetalLayerSurface(void* metal_layer) : metal_layer_(metal_layer) {}

  TypeIndex GetType() const override { return kTypeIndex_MetalLayer; }

  // CAMetalLayer*.
  void* metal_layer() const { return metal_layer_; }

 protected:
  bool GetSizeImpl(uint32_t& width_out, uint32_t& height_out) const override;

 private:
  void* metal_layer_;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_SURFACE_MAC_H_
