/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_texture_cache.h"

#import <Metal/Metal.h>

#include <vector>

#include "third_party/xxhash/xxhash.h"
#include "xenia/base/byte_order.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/memory.h"

#include "metal_irconverter_runtime.h"

namespace xe {
namespace gpu {
namespace metal {

namespace {
// Guest physical memory is 512 MiB; guard reads to it.
constexpr uint32_t kPhysicalSize = 512u * 1024 * 1024;
}  // namespace

MetalTextureCache::MetalTextureCache(void* device, Memory* memory,
                                     void* view_heap, uint32_t view_heap_slots)
    : device_(device),
      memory_(memory),
      view_heap_(view_heap),
      view_heap_slots_(view_heap_slots) {
  id<MTLBuffer> heap = (__bridge id<MTLBuffer>)view_heap_;
  view_heap_contents_ = [heap contents];
}

MetalTextureCache::~MetalTextureCache() {
  for (auto& pair : cache_) {
    if (pair.second.texture) {
      id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)pair.second.texture;
      (void)released;
    }
  }
  cache_.clear();
}

uint32_t MetalTextureCache::GetTextureSlot(
    const xenos::xe_gpu_texture_fetch_t& fetch) {
  if (fetch.type != xenos::FetchConstantType::kTexture) {
    return 0;
  }
  // Bring-up: 2D k_8_8_8_8 only.
  if (fetch.dimension != xenos::DataDimension::k2DOrStacked ||
      fetch.format != xenos::TextureFormat::k_8_8_8_8) {
    return 0;
  }
  uint32_t width = fetch.size_2d.width + 1;
  uint32_t height = fetch.size_2d.height + 1;
  uint32_t base = fetch.base_address << 12;
  uint32_t pitch_pixels = fetch.pitch << 5;  // already 32-aligned
  if (!base || !width || !height || width > 8192 || height > 8192 ||
      base >= kPhysicalSize) {
    return 0;
  }
  if (!pitch_pixels) {
    pitch_pixels = (width + 31) & ~31u;
  }

  struct KeyBytes {
    uint32_t base, width, height, pitch;
    uint32_t tiled_endian;
  } kb{base, width, height, pitch_pixels,
       (uint32_t(fetch.tiled) << 8) | uint32_t(fetch.endianness)};
  uint64_t key = XXH3_64bits(&kb, sizeof(kb));
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    active_textures_.push_back(it->second.texture);
    return it->second.slot;
  }
  if (next_slot_ >= view_heap_slots_) {
    return 0;
  }

  const uint8_t* guest = static_cast<const uint8_t*>(
      memory_->TranslatePhysical(base));
  if (!guest) {
    return 0;
  }
  // Untile / copy into a linear RGBA8 staging buffer.
  std::vector<uint8_t> linear(size_t(width) * height * 4);
  bool tiled = fetch.tiled != 0;
  xenos::Endian endian = fetch.endianness;
  uint32_t max_src_bytes = kPhysicalSize - base;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t src_off;
      if (tiled) {
        src_off = uint32_t(texture_address::Tiled2D(
                      int32_t(x), int32_t(y), pitch_pixels,
                      /*bytes_per_block_log2=*/2));
      } else {
        src_off = (y * pitch_pixels + x) * 4;
      }
      uint32_t texel = 0;
      if (src_off + 4 <= max_src_bytes) {
        std::memcpy(&texel, guest + src_off, 4);
      }
      // Guest 32bpp is big-endian (k8in32); swap to host little-endian.
      if (endian == xenos::Endian::k8in32) {
        texel = xe::byte_swap(texel);
      }
      std::memcpy(&linear[(size_t(y) * width + x) * 4], &texel, 4);
    }
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    return 0;
  }
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
               withBytes:linear.data()
             bytesPerRow:width * 4];

  uint32_t slot = next_slot_++;
  auto* entries = static_cast<IRDescriptorTableEntry*>(view_heap_contents_);
  IRDescriptorTableSetTexture(&entries[slot], texture, 0.0f, 0);
  void* texture_ptr = (__bridge_retained void*)texture;
  cache_.emplace(key, Entry{texture_ptr, slot});
  active_textures_.push_back(texture_ptr);
  return slot;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
