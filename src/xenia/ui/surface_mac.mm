/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/surface_mac.h"

#import <QuartzCore/CAMetalLayer.h>

namespace xe {
namespace ui {

bool MetalLayerSurface::GetSizeImpl(uint32_t& width_out,
                                    uint32_t& height_out) const {
  CAMetalLayer* layer = (__bridge CAMetalLayer*)metal_layer_;
  if (!layer) {
    width_out = 0;
    height_out = 0;
    return false;
  }
  // drawableSize is in physical pixels, which is what the surface contract
  // wants (1:1 with the display).
  CGSize size = layer.drawableSize;
  width_out = uint32_t(size.width);
  height_out = uint32_t(size.height);
  return true;
}

}  // namespace ui
}  // namespace xe
