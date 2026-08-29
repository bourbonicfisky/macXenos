/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L1 (Metal device required): the PRODUCTION lifecycle of the OpenSpec 6.1
// mipped route, driven through the REAL MetalSharedTextureCache object.
//
// This file exists because the lifecycle half of the 6.1 ladder had no harness
// that constructed the subclass, so provenance, fault refusal, full-level
// sampling, clear/repopulate and the atomic post-precreation fallback were
// inspection-only. Everything below runs the real object -- real
// MetalSharedMemory over the real 512 MiB buffer, real legacy MetalTextureCache
// as the slot/residency authority, real base-class RequestTextures/
// LoadTexturesData flow, real direct-load pipeline -- and uses only the
// production entry points (PrepareEligible, GetServedSlot, AppendAllTextures,
// OnSubmissionsFaulted, ClearCache) plus the production callback seams. There
// is NO model of the route, NO test-only accessor or telemetry on the
// production class, and NO environment configuration.
//
// EVERY claim below is read from a durable production observation:
//   - sampled/base level count and per-level byte readback of the real texture;
//   - the submission ids the production fault query asks about, which ARE the
//     per-region load provenance (a region never loaded is never queried);
//   - how many times the production prepare-transfer seam is asked for a
//     command buffer, which is exactly once per load that got past planning,
//     consumed-range registration and level-view pre-creation;
//   - GetServedSlot refusing or serving;
//   - legacy retirement counts, slot accounting and weak/retain lifetime;
//   - clear then repopulate;
//   - watch-driven recovery.
//
// Failure injection uses a production-owned seam: SetPrepareTransferCallback
// returning nullptr lands exactly after the level views are pre-created and
// before any encoding, which is the atomicity boundary the design claims.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include "metal_irconverter_runtime.h"
#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_mip_level_plan.h"
#include "xenia/gpu/metal/metal_resolved_fetch_service.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_shared_texture_cache.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/metal/metal_texture_load_eligibility.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/trace_protocol.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kPage = 4096;
constexpr uint32_t kFetchIndex = 3;
// Distinct guest regions so base and mip invalidation never alias.
constexpr uint32_t kMippedBase = 0x0D500000;
constexpr uint32_t kMippedMips = 0x0D600000;
constexpr uint32_t kFlatBase = 0x0D700000;
// The content-guard pair: a valid mipped shape whose MIP region can be zeroed
// without disturbing any other case's memory.
constexpr uint32_t kGuardBase = 0x0D800000;
constexpr uint32_t kGuardMips = 0x0D900000;
constexpr uint32_t kRegionBytes = 0x00100000;
constexpr uint32_t kTexW = 256;
constexpr uint32_t kTexH = 256;
constexpr uint32_t kMipMaxLevel = 8;
constexpr xenos::Endian kEndian = xenos::Endian::k8in32;

void EnsureAllocated(Memory& memory, uint32_t base, uint32_t length) {
  uint32_t first = base & ~(kPage - 1);
  uint32_t last = (base + length + kPage - 1) & ~(kPage - 1);
  memory.GetPhysicalHeap()->AllocFixed(
      first, last - first, kPage,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      kMemoryProtectRead | kMemoryProtectWrite);
}

// Content = the guest address of each word. Every level then has a distinct,
// address-derived expectation, so a level loaded from the wrong plan entry
// (wrong pointer, wrong offset, wrong sub-tile origin) cannot accidentally
// match.
void FillAddressPattern(Memory& memory, uint32_t base, uint32_t length) {
  EnsureAllocated(memory, base, length);
  auto* words = memory.TranslatePhysical<uint32_t*>(base);
  REQUIRE(words != nullptr);
  for (uint32_t i = 0; i < length / 4; ++i) {
    words[i] = base + i * 4u;
  }
}

// The all-zero guest region a trace replay restores for mip memory captures
// never recorded -- the exact condition the production content guard exists for.
void FillZero(Memory& memory, uint32_t base, uint32_t length) {
  EnsureAllocated(memory, base, length);
  auto* words = memory.TranslatePhysical<uint32_t*>(base);
  REQUIRE(words != nullptr);
  std::memset(words, 0, length);
}

texture_util::TextureGuestLayout LayoutFor(uint32_t w, uint32_t h,
                                           uint32_t max_level) {
  uint32_t pitch_div32 = ((w + 31u) & ~31u) >> 5;
  return texture_util::GetGuestTextureLayout(
      xenos::DataDimension::k2DOrStacked, pitch_div32, w, h, 1, /*tiled=*/true,
      xenos::TextureFormat::k_8_8_8_8, /*has_packed_levels=*/true,
      /*has_base=*/true, max_level);
}

xenos::xe_gpu_texture_fetch_t MakeFetch(uint32_t base, uint32_t mip,
                                        uint32_t w, uint32_t h,
                                        uint32_t mip_max_level) {
  xenos::xe_gpu_texture_fetch_t f = {};
  f.type = xenos::FetchConstantType::kTexture;
  f.sign_x = f.sign_y = f.sign_z = f.sign_w = xenos::TextureSign::kUnsigned;
  f.format = xenos::TextureFormat::k_8_8_8_8;
  f.endianness = kEndian;
  f.tiled = 1;
  f.dimension = xenos::DataDimension::k2DOrStacked;
  f.base_address = base >> 12;
  f.mip_address = mip >> 12;
  f.pitch = ((w + 31u) & ~31u) >> 5;
  f.size_2d.width = w - 1;
  f.size_2d.height = h - 1;
  f.swizzle = 0x688u;  // RGBA
  f.mip_min_level = 0;
  f.mip_max_level = mip_max_level;
  // The guest bit that puts the small levels in a packed tail. Setting it is
  // what makes this fixture exercise the packed-tail plan at all: the key
  // carries it straight through to GetGuestLayout, and without it the layout
  // reports packed_level == UINT32_MAX.
  f.packed_mips = mip_max_level ? 1 : 0;
  return f;
}

// The resolve registration that makes a fetch kRamContent -- the ONLY class
// the route may serve. Done through the production registration calls, not by
// asserting the classifier's answer into existence.
void RegisterRamBackedTarget(MetalTextureCache& legacy, id<MTLTexture> snapshot,
                             uint32_t base, uint32_t w, uint32_t h,
                             uint32_t extent) {
  legacy.RegisterResolvedTarget(base, (__bridge void*)snapshot, w, h,
                                /*is_depth=*/false,
                                /*requires_fetch_swizzle=*/false, base, extent);
  legacy.MarkResolvedTargetRamBacked(base, w, h, /*full_rect_geometry=*/true,
                                     uint32_t(kEndian));
}

// ---------------------------------------------------------------------------
// Fixture: the real object graph, wired exactly as the command processor wires
// it (prepare-transfer at a pass boundary, monotonic submission ids, fault
// query), with the harness standing in only for the frame loop.
// ---------------------------------------------------------------------------

struct DeviceEnv {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  Memory* memory = nullptr;
  std::unique_ptr<MetalSharedMemory> shared_memory;
  bool ok = false;

  DeviceEnv() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    shared_memory =
        std::make_unique<MetalSharedMemory>(*memory, (__bridge void*)device);
    if (!shared_memory->Initialize()) {
      shared_memory.reset();
      return;
    }
    queue = [device newCommandQueue];
    ok = queue != nil;
  }
};

DeviceEnv& Env() {
  static DeviceEnv env;
  return env;
}

#define REQUIRE_DEVICE()                                  \
  if (!Env().ok) {                                        \
    FAIL("Metal device required for [miproute-lifecycle]"); \
    return;                                               \
  }

struct Fixture {
  RegisterFile regs;
  id<MTLBuffer> view_heap = nil;
  id<MTLTexture> null_texture = nil;
  id<MTLTexture> snapshot = nil;
  std::unique_ptr<MetalTextureCache> legacy;
  std::unique_ptr<MetalSharedTextureCache> cache;

  // Frame-loop stand-ins.
  id<MTLCommandBuffer> frame_cb = nil;
  bool fail_prepare_transfer = false;
  // How many times production asked for a command buffer. The load impl calls
  // the seam exactly ONCE per load, and only after planning, consumed-range
  // registration and level-view pre-creation have all succeeded -- so this is
  // both the load count and the proof of how far a failing load got.
  uint32_t prepare_transfer_calls = 0;
  uint64_t submission = 1;
  std::set<uint64_t> faulted;
  std::vector<uint64_t> fault_queries;
  bool init_ok = false;

  Fixture() {
    DeviceEnv& env = Env();
    view_heap = [env.device newBufferWithLength:4096 * sizeof(IRDescriptorTableEntry)
                                        options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* nd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:1
                                    height:1
                                 mipmapped:NO];
    null_texture = [env.device newTextureWithDescriptor:nd];
    MTLTextureDescriptor* sd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:kTexW
                                    height:kTexH
                                 mipmapped:NO];
    snapshot = [env.device newTextureWithDescriptor:sd];
    if (!view_heap || !null_texture || !snapshot) {
      return;
    }
    legacy = std::make_unique<MetalTextureCache>(
        (__bridge void*)env.device, env.memory, env.shared_memory.get(),
        (__bridge void*)view_heap, 4096);
    legacy->SetNullTexture((__bridge void*)null_texture);
    legacy->SetSubmissionProvider([this]() { return submission; });

    cache = std::make_unique<MetalSharedTextureCache>(
        regs, *env.shared_memory, (__bridge void*)env.device, legacy.get());
    cache->SetPrepareTransferCallback([this]() -> void* {
      ++prepare_transfer_calls;
      if (fail_prepare_transfer) {
        return nullptr;
      }
      if (!frame_cb) {
        frame_cb = [Env().queue commandBuffer];
      }
      return (__bridge void*)frame_cb;
    });
    cache->SetSubmissionProvider([this]() { return submission; });
    cache->SetSubmissionFaultQuery([this](uint64_t s) {
      fault_queries.push_back(s);
      return faulted.count(s) != 0;
    });
    init_ok = cache->Initialize();
  }

  ~Fixture() {
    if (frame_cb) {
      [frame_cb commit];
      [frame_cb waitUntilCompleted];
      frame_cb = nil;
    }
    // Destruction order mirrors ShutdownDrawResources: the subclass retires
    // into the legacy pools, so it goes first.
    cache.reset();
    if (legacy) {
      legacy->ReclaimStaleSlots(UINT64_MAX);
    }
    legacy.reset();
  }

  void SetFetch(uint32_t index, const xenos::xe_gpu_texture_fetch_t& fetch) {
    std::memcpy(&regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * index],
                &fetch, sizeof(fetch));
  }

  // One "draw": the pre-encoder prepass over the eligible mask.
  void Prepare(uint32_t index) { cache->PrepareEligible(1u << index); }

  // Close the frame: commit, wait, then run the completion actions the command
  // processor runs (LRU advance, slot reclaim, consumed-range release).
  void EndFrame() {
    if (frame_cb) {
      [frame_cb commit];
      [frame_cb waitUntilCompleted];
      REQUIRE(frame_cb.status == MTLCommandBufferStatusCompleted);
      frame_cb = nil;
    }
    cache->CompletedSubmissionUpdated(submission);
    legacy->ReclaimStaleSlots(submission);
    Env().shared_memory->ReleaseConsumedRanges(submission);
    ++submission;
  }

  // One serve, with the per-region load provenance it consulted. The
  // production fault query is asked once per region that carries a load, so
  // the set of ids IS the observation of which submission each region's
  // content came from -- and a region that never loaded is never queried.
  struct Serve {
    uint32_t slot = 0;
    std::vector<uint64_t> queried;
    std::set<uint64_t> provenance;
  };
  Serve ServeSlot(uint32_t index) {
    fault_queries.clear();
    Serve s;
    s.slot = cache->GetServedSlot(index);
    s.queried = fault_queries;
    s.provenance.insert(fault_queries.begin(), fault_queries.end());
    return s;
  }

  // The one sampled view the subclass exposes for residency; production reads
  // it through exactly this call.
  id<MTLTexture> SoleSampledView() {
    std::vector<void*> views;
    cache->AppendAllTextures(views);
    if (views.size() != 1) {
      return nil;
    }
    return (__bridge id<MTLTexture>)views[0];
  }
};

// Blit one mip level of a private texture into a shared buffer and return its
// tightly packed RGBA8 bytes.
std::vector<uint8_t> ReadLevel(id<MTLTexture> texture, uint32_t level,
                               uint32_t w, uint32_t h) {
  DeviceEnv& env = Env();
  size_t row = size_t(w) * 4u;
  id<MTLBuffer> out = [env.device newBufferWithLength:row * h
                                              options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [env.queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromTexture:texture
             sourceSlice:0
             sourceLevel:level
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(w, h, 1)
                toBuffer:out
       destinationOffset:0
  destinationBytesPerRow:row
destinationBytesPerImage:row * h];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  std::vector<uint8_t> bytes(row * h);
  std::memcpy(bytes.data(), out.contents, bytes.size());
  return bytes;
}

// CPU oracle for one planned level, reading the SAME guest RAM the GPU load
// read, through the production tiled addresser and endian swap.
//
// Deliberately NOT one Catch assertion per texel: the oracle is exact over
// every texel, but a per-texel CHECK would emit ~150k assertions for one mip
// chain and drown the suite's assertion count without adding a single new
// claim. Per-texel validity is folded into one flag, and the comparison below
// reports the first differing texel so a failure still localizes exactly.
std::vector<uint8_t> OracleLevel(Memory& memory, const MipLevelPlan& p,
                                 bool* all_offsets_valid) {
  std::vector<uint8_t> exp(size_t(p.width) * p.height * 4u);
  bool ok = true;
  for (uint32_t y = 0; y < p.height; ++y) {
    for (uint32_t x = 0; x < p.width; ++x) {
      int32_t off = texture_address::Tiled2D(int32_t(p.px + x),
                                             int32_t(p.py + y),
                                             p.pitch_blocks, 2);
      if (off < 0) {
        ok = false;
        continue;
      }
      uint32_t word = *memory.TranslatePhysical<const uint32_t*>(
          p.guest_offset + uint32_t(off));
      word = xenos::GpuSwap(word, kEndian);
      std::memcpy(&exp[(size_t(y) * p.width + x) * 4u], &word, 4);
    }
  }
  *all_offsets_valid = ok;
  return exp;
}

// Full-level byte identity as TWO assertions, with the first differing texel
// named. Strength is unchanged -- every byte of the level is compared -- only
// the assertion accounting is compact.
void CheckLevelBytes(Memory& memory, id<MTLTexture> texture,
                     const MipLevelPlan& p) {
  bool offsets_valid = false;
  std::vector<uint8_t> exp = OracleLevel(memory, p, &offsets_valid);
  std::vector<uint8_t> got = ReadLevel(texture, p.host_level, p.width, p.height);
  size_t first_diff = exp.size();
  if (got.size() == exp.size()) {
    for (size_t i = 0; i < exp.size(); ++i) {
      if (got[i] != exp[i]) {
        first_diff = i;
        break;
      }
    }
  }
  INFO("level " << p.host_level << " " << p.width << "x" << p.height
                << " origin " << p.px << "," << p.py << " guest_offset 0x"
                << std::hex << p.guest_offset << std::dec);
  if (first_diff < exp.size()) {
    size_t texel = first_diff / 4;
    INFO("first differing byte " << first_diff << " (texel " << (texel % p.width)
                                 << "," << (texel / p.width) << "): got "
                                 << int(got[first_diff]) << " expected "
                                 << int(exp[first_diff]));
  }
  REQUIRE(offsets_valid);
  REQUIRE(got.size() == exp.size());
  CHECK(first_diff == exp.size());
}

std::vector<MipLevelPlan> ExpectedPlan(uint32_t w, uint32_t h,
                                       uint32_t max_level, bool load_base,
                                       bool load_mips) {
  texture_util::TextureGuestLayout layout = LayoutFor(w, h, max_level);
  MipLevelPlanInputs in;
  in.base_address = kMippedBase;
  in.mip_address = kMippedMips;
  in.width = w;
  in.height = h;
  in.mip_max_level = layout.max_level;
  in.guest_buffer_size = SharedMemory::kBufferSize;
  in.load_base = load_base;
  in.load_mips = load_mips;
  std::vector<MipLevelPlan> plan;
  REQUIRE(BuildMipLevelPlan(in, layout, &plan));
  return plan;
}

void PrimeGuestMemory() {
  Memory& memory = *Env().memory;
  FillAddressPattern(memory, kMippedBase, kRegionBytes);
  FillAddressPattern(memory, kMippedMips, kRegionBytes);
  FillAddressPattern(memory, kFlatBase, kRegionBytes);
  // Every writer reaches the shared-memory authority the same way; do it once
  // so the buffer revalidates from this content rather than a stale page.
  Env().shared_memory->MemoryInvalidationCallback(kMippedBase, kRegionBytes,
                                                  true);
  Env().shared_memory->MemoryInvalidationCallback(kMippedMips, kRegionBytes,
                                                  true);
  Env().shared_memory->MemoryInvalidationCallback(kFlatBase, kRegionBytes,
                                                  true);
}

// ---------------------------------------------------------------------------
// Trace sweep: run the production WriteAllCachedTexturesToTrace against a REAL
// TraceWriter over a temporary file, then read back ONLY the stable
// trace_protocol command fields (type, base_ptr, encoded/decoded_length). No
// payload byte is interpreted, no other command type is handled, and nothing
// about the trace format is broadened -- an unexpected command fails loudly
// instead of being skipped.
// ---------------------------------------------------------------------------

struct TraceMemoryRead {
  uint32_t base_ptr = 0;
  uint32_t decoded_length = 0;
  bool operator==(const TraceMemoryRead& o) const {
    return base_ptr == o.base_ptr && decoded_length == o.decoded_length;
  }
};

// Removes the temp trace even when a Catch assertion throws out of the sweep.
struct TempTracePath {
  std::filesystem::path path;
  explicit TempTracePath(int tag) {
    path = std::filesystem::temp_directory_path() /
           ("xe-miproute-lifecycle-" + std::to_string(tag) + ".xtr");
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  ~TempTracePath() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  TempTracePath(const TempTracePath&) = delete;
  TempTracePath& operator=(const TempTracePath&) = delete;
};

std::vector<TraceMemoryRead> SweepTraceMemoryReads(
    MetalSharedTextureCache& cache, Memory& memory, int tag) {
  TempTracePath temp(tag);
  {
    TraceWriter writer(memory.physical_membase());
    REQUIRE(writer.Open(temp.path, /*title_id=*/0));
    cache.WriteAllCachedTexturesToTrace(writer);
    writer.Close();
  }
  std::ifstream file(temp.path, std::ios::binary);
  REQUIRE(file.good());
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  file.close();
  REQUIRE(bytes.size() >= sizeof(TraceHeader));
  std::vector<TraceMemoryRead> reads;
  size_t at = sizeof(TraceHeader);
  while (at + sizeof(MemoryCommand) <= bytes.size()) {
    TraceCommandType type;
    std::memcpy(&type, bytes.data() + at, sizeof(type));
    // The sweep emits memory reads and nothing else.
    REQUIRE(type == TraceCommandType::kMemoryRead);
    MemoryCommand cmd;
    std::memcpy(&cmd, bytes.data() + at, sizeof(cmd));
    reads.push_back({cmd.base_ptr, cmd.decoded_length});
    at += sizeof(MemoryCommand) + cmd.encoded_length;
  }
  // The whole stream was consumed as whole commands.
  CHECK(at == bytes.size());
  return reads;
}

// ---------------------------------------------------------------------------
// 1. A mipped strict RGBA8 kRamContent texture is created and loaded through
//    the production planner, and its sampled view covers the FULL mip range.
// ---------------------------------------------------------------------------

TEST_CASE("A mipped route load covers every level and samples the full range",
          "[miproute-lifecycle]") {
  REQUIRE_DEVICE();
  @autoreleasepool {
    PrimeGuestMemory();
    Fixture f;
    REQUIRE(f.init_ok);

    auto fetch = MakeFetch(kMippedBase, kMippedMips, kTexW, kTexH, kMipMaxLevel);

    // The route's own admission contract, evaluated exactly as the command
    // processor evaluates it: the mipped sibling predicate admits it, the
    // promoted single-mip predicate does not, and the ONE classifier says
    // kRamContent (which is what makes it routable at all).
    uint32_t pitch_texels = uint32_t(fetch.pitch) << 5;
    CHECK(IsStrictRgba8MipRouteEligible(fetch, false, kTexW, kTexH,
                                        pitch_texels));
    CHECK_FALSE(IsStrictRgba8RouteEligible(fetch, false, kTexW, kTexH,
                                           pitch_texels));
    // Before any resolve registration the fetch is not resolve-backed at all,
    // so the route would never see it.
    CHECK(f.legacy->ClassifyFetchService(fetch, false) ==
          ResolvedFetchService::kNotResolveBacked);
    RegisterRamBackedTarget(*f.legacy, f.snapshot, kMippedBase, kTexW, kTexH,
                            0x40000);
    CHECK(f.legacy->ClassifyFetchService(fetch, false) ==
          ResolvedFetchService::kRamContent);

    f.SetFetch(kFetchIndex, fetch);
    const uint64_t s1 = f.submission;
    f.Prepare(kFetchIndex);

    // ONE command-buffer request for the WHOLE chain: every level is batched as
    // a dispatch on the single frame command buffer, never one per level.
    CHECK(f.prepare_transfer_calls == 1u);

    std::vector<MipLevelPlan> plan =
        ExpectedPlan(kTexW, kTexH, kMipMaxLevel, true, true);
    // The fixture is chosen so the packed tail is genuinely exercised: the
    // shared planner -- the same authority production uses -- yields at least
    // one level at a non-zero sub-tile origin.
    CHECK(std::any_of(plan.begin(), plan.end(),
                      [](const MipLevelPlan& p) { return p.px || p.py; }));

    // Served before the command buffer is even committed -- the load rides the
    // frame command buffer, exactly like production. Both regions were loaded
    // by this one submission, so both report it as their provenance.
    Fixture::Serve served = f.ServeSlot(kFetchIndex);
    CHECK(served.slot != 0u);
    CHECK(served.provenance == std::set<uint64_t>{s1});

    id<MTLTexture> view = f.SoleSampledView();
    REQUIRE(view != nil);
    id<MTLTexture> base_texture = view.parentTexture;
    REQUIRE(base_texture != nil);
    // Guest shaders sample texture2d_array; a single-level view here would
    // silently pin LOD selection to mip 0.
    CHECK(view.textureType == MTLTextureType2DArray);
    CHECK(view.mipmapLevelCount == plan.size());
    CHECK(base_texture.mipmapLevelCount == plan.size());

    f.EndFrame();

    // Every planned level actually holds the bytes its own plan entry names.
    for (const MipLevelPlan& p : plan) {
      CheckLevelBytes(*Env().memory, base_texture, p);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Base and mip provenance update independently, and GetServedSlot refuses
//    a fault in EITHER region.
// ---------------------------------------------------------------------------

TEST_CASE("Base and mip provenance are independent and either fault refuses",
          "[miproute-lifecycle]") {
  REQUIRE_DEVICE();
  @autoreleasepool {
    PrimeGuestMemory();
    Fixture f;
    REQUIRE(f.init_ok);
    f.SetFetch(kFetchIndex,
               MakeFetch(kMippedBase, kMippedMips, kTexW, kTexH, kMipMaxLevel));

    // Frame 1: whole texture at submission S1.
    const uint64_t s1 = f.submission;
    f.Prepare(kFetchIndex);
    REQUIRE(f.prepare_transfer_calls == 1u);
    f.EndFrame();

    // Frame 2: invalidate ONLY the base region. The base class must ask for a
    // base-only reload -- one more load, and only the base provenance advances.
    const uint64_t s2 = f.submission;
    REQUIRE(s2 != s1);
    Env().shared_memory->MemoryInvalidationCallback(kMippedBase, 0x40000, true);
    f.Prepare(kFetchIndex);
    CHECK(f.prepare_transfer_calls == 2u);
    f.EndFrame();

    // The fault query is the production observation of which submission each
    // region carries: base must now be S2 while mips are still S1. Two distinct
    // ids is exactly the independent-provenance claim.
    Fixture::Serve after_base_reload = f.ServeSlot(kFetchIndex);
    CHECK(after_base_reload.slot != 0u);
    CHECK(after_base_reload.provenance == std::set<uint64_t>{s1, s2});

    // Faulting the MIP submission alone must refuse the serve. Checking only
    // the base -- the defect this gate exists for -- would serve a texture
    // whose mip chain rode a lost command buffer.
    f.faulted = {s1};
    CHECK(f.cache->GetServedSlot(kFetchIndex) == 0u);

    // Faulting the BASE submission alone must refuse it too.
    f.faulted = {s2};
    CHECK(f.cache->GetServedSlot(kFetchIndex) == 0u);

    f.faulted.clear();
    CHECK(f.cache->GetServedSlot(kFetchIndex) != 0u);

    // Mip-only reload is the mirror case: invalidate only the mip region and
    // the mip provenance advances alone.
    const uint64_t s3 = f.submission;
    Env().shared_memory->MemoryInvalidationCallback(kMippedMips, 0x20000, true);
    f.Prepare(kFetchIndex);
    CHECK(f.prepare_transfer_calls == 3u);
    f.EndFrame();

    Fixture::Serve after_mip_reload = f.ServeSlot(kFetchIndex);
    CHECK(after_mip_reload.slot != 0u);
    CHECK(after_mip_reload.provenance == std::set<uint64_t>{s2, s3});

    // And the eager eviction path treats the two regions independently as well:
    // faulting the base submission invalidates the base only, so the next
    // prepare reloads the base -- new base provenance, mip provenance still S3.
    const uint64_t s4 = f.submission;
    f.cache->OnSubmissionsFaulted({s2});
    f.Prepare(kFetchIndex);
    CHECK(f.prepare_transfer_calls == 4u);
    Fixture::Serve after_fault_reload = f.ServeSlot(kFetchIndex);
    CHECK(after_fault_reload.slot != 0u);
    CHECK(after_fault_reload.provenance == std::set<uint64_t>{s3, s4});
    f.EndFrame();
  }
}

// ---------------------------------------------------------------------------
// 3. ClearCache releases the mipped texture and both view classes, and the
//    cache repopulates without slot or resource growth.
// ---------------------------------------------------------------------------

TEST_CASE("Clear releases the mipped texture and repopulation does not grow",
          "[miproute-lifecycle]") {
  REQUIRE_DEVICE();
  // Outside the pool: the release claim must be observed after every
  // autoreleased Metal object from the run has drained.
  __weak id<MTLTexture> weak_first_base = nil;
  @autoreleasepool {
    PrimeGuestMemory();
    Fixture f;
    REQUIRE(f.init_ok);
    f.SetFetch(kFetchIndex,
               MakeFetch(kMippedBase, kMippedMips, kTexW, kTexH, kMipMaxLevel));

    uint32_t slots_after_first = 0;

    for (int cycle = 0; cycle < 3; ++cycle) {
      INFO("cycle " << cycle);
      @autoreleasepool {
        const uint32_t loads_before = f.prepare_transfer_calls;
        f.Prepare(kFetchIndex);
        // The previous cycle's clear really dropped the TEXTURE, not just its
        // descriptor state: repopulation has to run a fresh load.
        CHECK(f.prepare_transfer_calls == loads_before + 1u);
        uint32_t slot = f.cache->GetServedSlot(kFetchIndex);
        CHECK(slot != 0u);
        std::vector<void*> views;
        f.cache->AppendAllTextures(views);
        CHECK(views.size() == 1u);
        if (cycle == 0) {
          weak_first_base =
              ((__bridge id<MTLTexture>)views[0]).parentTexture;
          REQUIRE(weak_first_base != nil);
        }
        f.EndFrame();

        // Production clear order: the subclass retires into the legacy pools,
        // then the legacy clear drains them.
        f.cache->ClearCache();
        f.legacy->ClearCache();

        // Nothing of the subclass survives the clear.
        std::vector<void*> after;
        f.cache->AppendAllTextures(after);
        CHECK(after.empty());
        CHECK(f.legacy->retired_texture_count() == 0u);
        CHECK(f.legacy->cached_texture_count() == 0u);
        CHECK(f.legacy->slots_stale() == 0u);

        uint32_t slots_now = f.legacy->slots_used();
        if (cycle == 0) {
          slots_after_first = slots_now;
        } else {
          // Repopulation reuses the freed slot rather than extending the heap.
          CHECK(slots_now == slots_after_first);
        }
      }
    }

  }
  // The mipped texture, its sampled view and every ephemeral level view are
  // actually gone -- a retained level view would keep the parent alive.
  CHECK(weak_first_base == nil);
}

// ---------------------------------------------------------------------------
// 4. An injected failure after level-view pre-creation and before encoding
//    falls back atomically and leaks no retained views.
// ---------------------------------------------------------------------------

TEST_CASE("A failure after view precreation falls back atomically and leaks nothing",
          "[miproute-lifecycle]") {
  REQUIRE_DEVICE();
  // Declared outside the pool so the weak check below observes the state AFTER
  // every autoreleased Metal object from the run has drained; otherwise the
  // probe reads pool timing rather than ownership.
  __weak id<MTLTexture> weak_base = nil;
  @autoreleasepool {
    PrimeGuestMemory();
    {
      Fixture f;
      REQUIRE(f.init_ok);
      f.SetFetch(kFetchIndex, MakeFetch(kMippedBase, kMippedMips, kTexW, kTexH,
                                        kMipMaxLevel));

      // A good load first, so a texture (and therefore level views on reload)
      // exists to be created and then abandoned.
      f.Prepare(kFetchIndex);
      REQUIRE(f.prepare_transfer_calls == 1u);
      id<MTLTexture> view = f.SoleSampledView();
      REQUIRE(view != nil);
      id<MTLTexture> base_texture = view.parentTexture;
      REQUIRE(base_texture != nil);
      weak_base = base_texture;
      f.EndFrame();

      const uint32_t loads_before = f.prepare_transfer_calls;
      const uint32_t retired_before = f.legacy->retired_texture_count();
      NSUInteger parent_retains_before =
          CFGetRetainCount((__bridge CFTypeRef)base_texture);

      // Observability precondition: a level view retains its parent, so a
      // leaked view is visible as a retain-count increase. If this ever stops
      // holding, the probe below is meaningless and must FAIL loudly.
      @autoreleasepool {
        id<MTLTexture> probe =
            [base_texture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                            textureType:MTLTextureType2D
                                                 levels:NSMakeRange(0, 1)
                                                 slices:NSMakeRange(0, 1)];
        REQUIRE(probe != nil);
        REQUIRE(CFGetRetainCount((__bridge CFTypeRef)base_texture) >
                parent_retains_before);
        probe = nil;
      }
      REQUIRE(CFGetRetainCount((__bridge CFTypeRef)base_texture) ==
              parent_retains_before);

      // Inject the failure: prepare-transfer returns nullptr, which lands
      // AFTER every level view is created and BEFORE any encoding.
      f.fail_prepare_transfer = true;
      // Invalidate the MIP region only, so the abandoned load pre-creates a
      // whole chain of level views (levels 1..max) rather than one, and the
      // BASE watch survives to drive the recovery below.
      Env().shared_memory->MemoryInvalidationCallback(kMippedMips, 0x20000,
                                                      true);
      f.Prepare(kFetchIndex);

      // The failure landed where it was aimed: production reaches the
      // prepare-transfer seam ONLY after planning, consumed-range registration
      // and every level view have succeeded. So a seam call is what proves the
      // views really were created and then abandoned, rather than the load
      // bailing out earlier and making the leak probe vacuous.
      CHECK(f.prepare_transfer_calls == loads_before + 1u);
      // Nothing was encoded: the seam refused before a command buffer existed.
      CHECK(f.frame_cb == nil);
      // The fallback is atomic: the texture stays outdated, so the serve
      // refuses and the legacy CPU path covers the whole texture ...
      CHECK(f.cache->GetServedSlot(kFetchIndex) == 0u);
      // ... the views were RELEASED, not retired into the submission pool ...
      CHECK(f.legacy->retired_texture_count() == retired_before);
      // ... and no level view outlived the failed load.
      CHECK(CFGetRetainCount((__bridge CFTypeRef)base_texture) ==
            parent_retains_before);

      // ---- Recovery semantics, stated exactly as the SHARED base provides
      // them (audited, not worked around; nothing here is a 6.1 behavior).
      //
      // A failed load skips MakeUpToDateAndWatch entirely, so the failed
      // region stays outdated AND unwatched. PrepareEligible clears the
      // in-sync bit every draw, so the binding is re-evaluated -- but
      // RequestTextures only enqueues a load when the KEY changes, and for a
      // stable fetch the only thing that changes it is ResetTextureBindings,
      // which runs solely when texture_became_outdated_ is set, whose sole
      // setter is TextureCache::WatchCallback. So: recovery is driven by a
      // WATCH FIRING, not by the route, and re-attempting eagerly per draw
      // would be a change to shared base behavior. Meanwhile the state is
      // safe, not merely tolerable: GetServedSlot refuses an outdated texture,
      // so the legacy CPU path serves it for as long as it takes.

      // (a) Restoring the seam alone changes nothing -- no eager retry exists,
      // so production never even asks for a command buffer again.
      const uint32_t loads_after_failure = f.prepare_transfer_calls;
      f.fail_prepare_transfer = false;
      f.Prepare(kFetchIndex);
      CHECK(f.prepare_transfer_calls == loads_after_failure);
      CHECK(f.cache->GetServedSlot(kFetchIndex) == 0u);

      // (b) Nor does invalidating the FAILED region: its watch already fired
      // and was never reinstalled, so there is nothing left there to fire.
      // This is what makes (c) a proof of the mechanism rather than a
      // coincidence of ordering.
      Env().shared_memory->MemoryInvalidationCallback(kMippedMips, 0x20000,
                                                      true);
      f.Prepare(kFetchIndex);
      CHECK(f.prepare_transfer_calls == loads_after_failure);
      CHECK(f.cache->GetServedSlot(kFetchIndex) == 0u);

      // (c) A watch that IS still installed -- here the base region, which
      // recurring resolve destinations invalidate every frame -- resets the
      // bindings and re-routes the texture. Both regions were outdated, so the
      // recovery reloads the WHOLE chain, not just the newly invalidated half:
      // ONE load, and afterwards BOTH regions report the SAME submission.
      std::vector<MipLevelPlan> full_plan =
          ExpectedPlan(kTexW, kTexH, kMipMaxLevel, true, true);
      const uint64_t s_recover = f.submission;
      Env().shared_memory->MemoryInvalidationCallback(kMippedBase, 0x40000,
                                                      true);
      f.Prepare(kFetchIndex);
      CHECK(f.prepare_transfer_calls == loads_after_failure + 1u);
      Fixture::Serve recovered = f.ServeSlot(kFetchIndex);
      CHECK(recovered.slot != 0u);
      CHECK(recovered.provenance == std::set<uint64_t>{s_recover});
      f.EndFrame();

      // And the recovered chain holds the bytes every level's plan entry names.
      for (const MipLevelPlan& p : full_plan) {
        CheckLevelBytes(*Env().memory, base_texture, p);
      }

      f.cache->ClearCache();
      f.legacy->ClearCache();
    }
  }
  CHECK(weak_base == nil);
}

// ---------------------------------------------------------------------------
// 5. The promoted single-mip path is unchanged by the sibling.
// ---------------------------------------------------------------------------

TEST_CASE("The promoted single-mip path is exact with the sibling present",
          "[miproute-lifecycle]") {
  REQUIRE_DEVICE();
  @autoreleasepool {
    PrimeGuestMemory();
    Fixture f;
    REQUIRE(f.init_ok);

    auto fetch = MakeFetch(kFlatBase, 0, kTexW, kTexH, /*mip_max_level=*/0);
    uint32_t pitch_texels = uint32_t(fetch.pitch) << 5;
    // Its own predicate still decides it, and the mipped sibling refuses it.
    CHECK(IsStrictRgba8RouteEligible(fetch, false, kTexW, kTexH, pitch_texels));
    CHECK_FALSE(IsStrictRgba8MipRouteEligible(fetch, false, kTexW, kTexH,
                                              pitch_texels));

    f.SetFetch(kFetchIndex, fetch);
    const uint64_t s1 = f.submission;
    f.Prepare(kFetchIndex);

    // Exactly one load ...
    CHECK(f.prepare_transfer_calls == 1u);
    // ... and it touched the BASE REGION ONLY: the mip region carries no
    // provenance at all, so production queries exactly one submission.
    Fixture::Serve served = f.ServeSlot(kFetchIndex);
    CHECK(served.slot != 0u);
    CHECK(served.queried.size() == 1u);
    CHECK(served.provenance == std::set<uint64_t>{s1});

    id<MTLTexture> view = f.SoleSampledView();
    REQUIRE(view != nil);
    id<MTLTexture> base_texture = view.parentTexture;
    REQUIRE(base_texture != nil);
    // A flat texture stays a flat, non-view-typed 2D texture with one level.
    CHECK(base_texture.mipmapLevelCount == 1u);
    CHECK(view.mipmapLevelCount == 1u);
    CHECK(view.textureType == MTLTextureType2DArray);
    // The signed serve casts this texture to RGBA8Snorm via a format view,
    // which Metal only defines when the base declares PixelFormatView usage.
    CHECK((base_texture.usage & MTLTextureUsagePixelFormatView) != 0u);

    f.EndFrame();

    // And the bytes are the ones the single-mip contract names: the planner's
    // level 0 at origin (0,0), from the base pointer.
    MipLevelPlan p;
    p.host_level = 0;
    p.guest_offset = kFlatBase;
    p.pitch_blocks = pitch_texels;
    p.width = kTexW;
    p.height = kTexH;
    CheckLevelBytes(*Env().memory, base_texture, p);
  }
}

// ---------------------------------------------------------------------------
// 6. An all-zero mip region trips the production content guard BEFORE the
//    prepare-transfer seam, leaving the shared slot unserved and the legacy
//    fallback whole. Non-vacuity is proved by a positive control: the SAME
//    fetch, on the SAME shape, with nonzero mip content loads and serves.
// ---------------------------------------------------------------------------

TEST_CASE("An all-zero mip region declines before PrepareTransfer",
          "[miproute-lifecycle]") {
  REQUIRE_DEVICE();
  @autoreleasepool {
    PrimeGuestMemory();
    Memory& memory = *Env().memory;
    auto fetch = MakeFetch(kGuardBase, kGuardMips, kTexW, kTexH, kMipMaxLevel);
    const uint32_t pitch_texels = uint32_t(fetch.pitch) << 5;

    // The fetch is a genuine strict mipped candidate by the production
    // predicates -- so a decline can only come from content, never eligibility.
    REQUIRE(IsStrictRgba8MipRouteEligible(fetch, false, kTexW, kTexH,
                                          pitch_texels));
    REQUIRE_FALSE(IsStrictRgba8RouteEligible(fetch, false, kTexW, kTexH,
                                             pitch_texels));

    // ---- Positive control: nonzero base AND nonzero mip region. ----
    FillAddressPattern(memory, kGuardBase, kRegionBytes);
    FillAddressPattern(memory, kGuardMips, kRegionBytes);
    Env().shared_memory->MemoryInvalidationCallback(kGuardBase, kRegionBytes,
                                                    true);
    Env().shared_memory->MemoryInvalidationCallback(kGuardMips, kRegionBytes,
                                                    true);
    {
      Fixture f;
      REQUIRE(f.init_ok);
      RegisterRamBackedTarget(*f.legacy, f.snapshot, kGuardBase, kTexW, kTexH,
                              0x40000);
      REQUIRE(f.legacy->ClassifyFetchService(fetch, false) ==
              ResolvedFetchService::kRamContent);
      f.SetFetch(kFetchIndex, fetch);
      const uint64_t s1 = f.submission;
      f.Prepare(kFetchIndex);
      CHECK(f.prepare_transfer_calls == 1u);
      Fixture::Serve served = f.ServeSlot(kFetchIndex);
      CHECK(served.slot != 0u);
      CHECK(served.provenance == std::set<uint64_t>{s1});
      f.EndFrame();
      f.cache->ClearCache();
      f.legacy->ClearCache();
    }

    // ---- Guard: identical fetch and base content, mip region ALL ZERO. ----
    FillAddressPattern(memory, kGuardBase, kRegionBytes);
    FillZero(memory, kGuardMips, kRegionBytes);
    Env().shared_memory->MemoryInvalidationCallback(kGuardBase, kRegionBytes,
                                                    true);
    Env().shared_memory->MemoryInvalidationCallback(kGuardMips, kRegionBytes,
                                                    true);
    // The two content facts the guard discriminates on.
    REQUIRE(*memory.TranslatePhysical<const uint32_t*>(kGuardBase) != 0u);
    REQUIRE(*memory.TranslatePhysical<const uint32_t*>(kGuardMips) == 0u);

    Fixture f;
    REQUIRE(f.init_ok);
    RegisterRamBackedTarget(*f.legacy, f.snapshot, kGuardBase, kTexW, kTexH,
                            0x40000);
    REQUIRE(f.legacy->ClassifyFetchService(fetch, false) ==
            ResolvedFetchService::kRamContent);
    f.SetFetch(kFetchIndex, fetch);
    f.Prepare(kFetchIndex);

    // The guard runs BEFORE the seam: production never asked for a command
    // buffer, so no encoder and no command buffer ever existed for this load.
    CHECK(f.prepare_transfer_calls == 0u);
    CHECK(f.frame_cb == nil);
    // Atomic legacy fallback: the shared slot is UNSERVED, so the legacy CPU
    // path covers the whole texture -- base level included, which is the
    // "decline the whole load" half of the contract that prepare_transfer_calls
    // == 0 already witnesses.
    CHECK(f.cache->GetServedSlot(kFetchIndex) == 0u);
    // No level view was ever created, so nothing entered the submission-gated
    // retirement pool: the guard precedes planning AND view pre-creation.
    CHECK(f.legacy->retired_texture_count() == 0u);
    // The decline is stable, not one-shot: re-preparing still never reaches the
    // seam and still serves nothing.
    f.Prepare(kFetchIndex);
    CHECK(f.prepare_transfer_calls == 0u);
    CHECK(f.cache->GetServedSlot(kFetchIndex) == 0u);
    // NOTE deliberately not asserted: a binding slot may be pre-bound for this
    // texture (the same post-failure state case 4 observes) and the legacy
    // cache holds its own slots for the registered resolved target. Neither is
    // part of the content-guard contract; refusing to SERVE is.
  }
}

// ---------------------------------------------------------------------------
// 7. The warm trace sweep records BOTH up-to-date guest regions of a loaded
//    mipped texture, and omits a region that has gone outdated.
// ---------------------------------------------------------------------------

TEST_CASE("The warm trace sweep records both regions and omits an outdated one",
          "[miproute-lifecycle]") {
  REQUIRE_DEVICE();
  @autoreleasepool {
    PrimeGuestMemory();
    Fixture f;
    REQUIRE(f.init_ok);
    auto fetch =
        MakeFetch(kMippedBase, kMippedMips, kTexW, kTexH, kMipMaxLevel);
    f.SetFetch(kFetchIndex, fetch);
    f.Prepare(kFetchIndex);
    REQUIRE(f.prepare_transfer_calls == 1u);
    REQUIRE(f.cache->GetServedSlot(kFetchIndex) != 0u);
    f.EndFrame();

    // The two extents pinned to the authorities production actually uses. NOTE
    // the base is the LEGACY warm-entry extent expression (ComputeWatchedBaseBytes),
    // NOT Texture::GetGuestBaseSize; the mip chain is the shared layout's
    // mips_total_extent_bytes, which IS exactly Texture::GetGuestMipsSize().
    const uint32_t pitch_texels = uint32_t(fetch.pitch) << 5;
    const uint32_t expected_base_bytes = ComputeWatchedBaseBytes(
        pitch_texels, kTexH, 4, kMippedBase, SharedMemory::kBufferSize);
    const uint32_t expected_mip_bytes =
        LayoutFor(kTexW, kTexH, kMipMaxLevel).mips_total_extent_bytes;
    REQUIRE(expected_base_bytes != 0u);
    REQUIRE(expected_mip_bytes != 0u);

    // Warm and fully up to date: BOTH guest ranges are recorded, base first.
    CHECK(SweepTraceMemoryReads(*f.cache, *Env().memory, 0) ==
          std::vector<TraceMemoryRead>{{kMippedBase, expected_base_bytes},
                                       {kMippedMips, expected_mip_bytes}});

    // Outdate the MIP region only: a replay would restore bytes the capture
    // never proved current, so only the still-warm base is recorded.
    Env().shared_memory->MemoryInvalidationCallback(kMippedMips, 0x20000, true);
    CHECK(SweepTraceMemoryReads(*f.cache, *Env().memory, 1) ==
          std::vector<TraceMemoryRead>{{kMippedBase, expected_base_bytes}});

    // Outdate the base too and the sweep records nothing at all.
    Env().shared_memory->MemoryInvalidationCallback(kMippedBase, 0x40000, true);
    CHECK(SweepTraceMemoryReads(*f.cache, *Env().memory, 2).empty());
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
