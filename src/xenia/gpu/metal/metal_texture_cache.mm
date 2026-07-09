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

#include <algorithm>
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

// Describes how a guest texture format maps to a host upload. `bytes_per_block`
// / `block_log2` / `block_texels` are the compressed-block geometry (1x1 for
// uncompressed 8_8_8_8). `host_format` is the Metal pixel format we upload to;
// for compressed formats we hand the raw (endian-corrected) blocks to the
// hardware BC decoder, for 8_8_8_8 we build linear RGBA8.
struct FormatPlan {
  bool supported;
  bool compressed;
  uint32_t bytes_per_block;
  uint32_t block_log2;   // log2(bytes_per_block), for the tiled addresser.
  uint32_t block_texels;  // 1 (uncompressed) or 4 (DXT).
  MTLPixelFormat host_format;
};

FormatPlan PlanForFormat(xenos::TextureFormat format, bool bc_supported) {
  switch (format) {
    case xenos::TextureFormat::k_8_8_8_8:
      return {true, false, 4, 2, 1, MTLPixelFormatRGBA8Unorm};
    case xenos::TextureFormat::k_DXT1:
    case xenos::TextureFormat::k_DXT1_AS_16_16_16_16:
      return {bc_supported, true, 8, 3, 4, MTLPixelFormatBC1_RGBA};
    case xenos::TextureFormat::k_DXT2_3:
    case xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16:
      return {bc_supported, true, 16, 4, 4, MTLPixelFormatBC2_RGBA};
    case xenos::TextureFormat::k_DXT4_5:
    case xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16:
      return {bc_supported, true, 16, 4, 4, MTLPixelFormatBC3_RGBA};
    default:
      return {false, false, 0, 0, 0, MTLPixelFormatInvalid};
  }
}

// Convert a block (or texel) of `bytes` from the guest endianness to host
// little-endian in place. Guest 32bpp is k8in32; guest DXT is normally k8in16
// (each 16-bit word byte-swapped), but honor whatever the fetch declares.
void SwapEndian(uint8_t* data, uint32_t bytes, xenos::Endian endian) {
  switch (endian) {
    case xenos::Endian::k8in16:
      for (uint32_t i = 0; i + 1 < bytes; i += 2) {
        std::swap(data[i], data[i + 1]);
      }
      break;
    case xenos::Endian::k8in32:
      for (uint32_t i = 0; i + 3 < bytes; i += 4) {
        std::swap(data[i], data[i + 3]);
        std::swap(data[i + 1], data[i + 2]);
      }
      break;
    case xenos::Endian::k16in32:
      for (uint32_t i = 0; i + 3 < bytes; i += 4) {
        std::swap(data[i], data[i + 2]);
        std::swap(data[i + 1], data[i + 3]);
      }
      break;
    default:
      break;
  }
}
}  // namespace

MetalTextureCache::MetalTextureCache(void* device, Memory* memory,
                                     void* view_heap, uint32_t view_heap_slots)
    : device_(device),
      memory_(memory),
      view_heap_(view_heap),
      view_heap_slots_(view_heap_slots) {
  id<MTLBuffer> heap = (__bridge id<MTLBuffer>)view_heap_;
  view_heap_contents_ = [heap contents];
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_;
  bc_supported_ = [mtl_device supportsBCTextureCompression];
  XELOGGPU("MetalTextureCache: BC (DXT) texture compression {}",
           bc_supported_ ? "supported" : "UNSUPPORTED (DXT stays gray)");
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
  // Bring-up: 2D only.
  if (fetch.dimension != xenos::DataDimension::k2DOrStacked) {
    return 0;
  }
  FormatPlan plan = PlanForFormat(fetch.format, bc_supported_);
  if (!plan.supported) {
    return 0;
  }

  uint32_t width = fetch.size_2d.width + 1;
  uint32_t height = fetch.size_2d.height + 1;
  uint32_t base = fetch.base_address << 12;
  uint32_t pitch_texels = fetch.pitch << 5;  // guest pitch is texels/32.
  if (!base || !width || !height || width > 8192 || height > 8192 ||
      base >= kPhysicalSize) {
    return 0;
  }
  if (!pitch_texels) {
    pitch_texels = (width + 31) & ~31u;
  }

  struct KeyBytes {
    uint32_t base, width, height, pitch;
    uint32_t format_tiled_endian;
  } kb{base, width, height, pitch_texels,
       (uint32_t(fetch.format) << 16) | (uint32_t(fetch.tiled) << 8) |
           uint32_t(fetch.endianness)};
  uint64_t key = XXH3_64bits(&kb, sizeof(kb));
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    active_textures_.push_back(it->second.texture);
    return it->second.slot;
  }
  if (next_slot_ >= view_heap_slots_) {
    return 0;
  }

  const uint8_t* guest =
      static_cast<const uint8_t*>(memory_->TranslatePhysical(base));
  if (!guest) {
    return 0;
  }

  bool tiled = fetch.tiled != 0;
  xenos::Endian endian = fetch.endianness;
  uint32_t max_src_bytes = kPhysicalSize - base;

  // Block grid (1x1 for uncompressed). The tiled addresser works in block
  // coordinates; the pitch it wants is in blocks, 32-block aligned.
  uint32_t block_texels = plan.block_texels;
  uint32_t bpb = plan.bytes_per_block;
  uint32_t blocks_w = (width + block_texels - 1) / block_texels;
  uint32_t blocks_h = (height + block_texels - 1) / block_texels;
  uint32_t pitch_blocks = pitch_texels / block_texels;
  uint32_t pitch_blocks_aligned = (pitch_blocks + 31u) & ~31u;

  // Untile / copy into a linear block-major staging buffer (host byte order).
  std::vector<uint8_t> linear(size_t(blocks_w) * blocks_h * bpb);
  for (uint32_t by = 0; by < blocks_h; ++by) {
    for (uint32_t bx = 0; bx < blocks_w; ++bx) {
      uint32_t src_off;
      if (tiled) {
        src_off = uint32_t(texture_address::Tiled2D(
            int32_t(bx), int32_t(by), pitch_blocks_aligned, plan.block_log2));
      } else {
        src_off = (by * pitch_blocks + bx) * bpb;
      }
      uint8_t* dst = &linear[(size_t(by) * blocks_w + bx) * bpb];
      if (src_off + bpb <= max_src_bytes) {
        std::memcpy(dst, guest + src_off, bpb);
        SwapEndian(dst, bpb, endian);
      }
    }
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:plan.host_format
                                   width:width
                                  height:height
                               mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    return 0;
  }
  // For BC formats bytesPerRow is a row of blocks; MTLRegion is still in texels
  // and Metal handles partial edge blocks.
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
               withBytes:linear.data()
             bytesPerRow:blocks_w * bpb];

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
