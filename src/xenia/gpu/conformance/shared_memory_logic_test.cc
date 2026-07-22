/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free): the REAL xe::gpu::SharedMemory page-validity/range logic —
// the exact base class MetalSharedMemory runs — exercised through a minimal
// test subclass whose UploadRanges transport memcpys into a sparse host
// mirror instead of an MTLBuffer. The validity decisions (what is valid, what
// re-uploads, page granularity, coalescing, clamping) are entirely the
// production base class; only the byte transport differs, and the transport
// records exactly what the base class asked it to upload.

#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/shared_memory.h"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace {


class MirrorSharedMemory : public SharedMemory {
 public:
  explicit MirrorSharedMemory(Memory& memory) : SharedMemory(memory) {}
  ~MirrorSharedMemory() override { ShutdownCommon(); }
  bool Init() { return InitializeCommon(); }
  // The base class's own page size (system page size — 16KiB on Apple
  // silicon, NOT 4096); all upload range indices are in these units.
  uint32_t PageSize() const { return 1u << page_size_log2(); }

  // Sparse host mirror keyed by page index + a log of uploaded page ranges.
  std::map<uint32_t, std::vector<uint8_t>> mirror;
  std::vector<std::pair<uint32_t, uint32_t>> upload_log;

  bool UploadRanges(const std::pair<uint32_t, uint32_t>* upload_page_ranges,
                    uint32_t num_upload_ranges) override {
    for (uint32_t i = 0; i < num_upload_ranges; ++i) {
      upload_log.push_back(upload_page_ranges[i]);
      uint32_t first_page = upload_page_ranges[i].first;
      uint32_t page_count = upload_page_ranges[i].second;
      // Base-class contract (same as the production backends): the transport
      // marks the pages valid — RequestRange itself never does. Validate
      // BEFORE the copy so a racing CPU write re-invalidates, not gets lost.
      MakeRangeValid(first_page << page_size_log2(),
                     page_count << page_size_log2(), false);
      for (uint32_t p = first_page; p < first_page + page_count; ++p) {
        auto& page = mirror[p];
        page.resize(PageSize());
        std::memcpy(page.data(),
                    memory().TranslatePhysical(p * PageSize()), PageSize());
      }
    }
    return true;
  }

  std::vector<uint8_t> MirrorRead(uint32_t addr, uint32_t len) {
    std::vector<uint8_t> out(len, 0xCD);
    for (uint32_t i = 0; i < len;) {
      uint32_t p = (addr + i) / PageSize();
      uint32_t off = (addr + i) % PageSize();
      uint32_t n = std::min(len - i, PageSize() - off);
      auto it = mirror.find(p);
      if (it != mirror.end()) {
        std::memcpy(out.data() + i, it->second.data() + off, n);
      }
      i += n;
    }
    return out;
  }
};

struct LogicHarness {
  Memory* memory = nullptr;
  std::unique_ptr<MirrorSharedMemory> sm;
  bool ok = false;
  LogicHarness() {
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    sm = std::make_unique<MirrorSharedMemory>(*memory);
    ok = sm->Init();
  }
  // SM page size shorthand for the cases.
  uint32_t P() const { return sm->PageSize(); }
  // Commit whole SM pages: UploadRanges reads full SM pages through
  // TranslatePhysical, so partially committed pages would fault.
  void Alloc(uint32_t addr, uint32_t len) {
    uint32_t page = P();
    uint32_t base = addr & ~(page - 1);
    uint32_t end = (addr + len + page - 1) & ~(page - 1);
    memory->GetPhysicalHeap()->AllocFixed(
        base, end - base, page,
        kMemoryAllocationReserve | kMemoryAllocationCommit,
        kMemoryProtectRead | kMemoryProtectWrite);
  }
  void Write(uint32_t addr, const std::vector<uint8_t>& b) {
    Alloc(addr, uint32_t(b.size()));
    std::memcpy(memory->TranslatePhysical(addr), b.data(), b.size());
  }
  std::vector<uint8_t> Pat(uint32_t n, uint8_t s) {
    std::vector<uint8_t> v(n);
    for (uint32_t i = 0; i < n; ++i) v[i] = uint8_t(s + i * 3);
    return v;
  }
};

LogicHarness& L() {
  static LogicHarness h;
  return h;
}

TEST_CASE("SM logic: initially invalid; request uploads; second request "
          "does not re-upload",
          "[l0][sharedmem]") {
  REQUIRE(L().ok);
  auto& h = L();
  const uint32_t kPage = h.P();
  const uint32_t addr = 0x01000000;
  auto b = h.Pat(kPage, 1);
  h.Write(addr, b);
  size_t uploads_before = h.sm->upload_log.size();
  REQUIRE(h.sm->RequestRange(addr, kPage));
  REQUIRE(h.sm->upload_log.size() == uploads_before + 1);
  REQUIRE(h.sm->MirrorRead(addr, kPage) == b);
  // Valid page: no new upload on a second request.
  REQUIRE(h.sm->RequestRange(addr, kPage));
  REQUIRE(h.sm->upload_log.size() == uploads_before + 1);
}

TEST_CASE("SM logic: invalidation forces re-upload; page granularity for "
          "partial-page writes",
          "[l0][sharedmem]") {
  REQUIRE(L().ok);
  auto& h = L();
  const uint32_t kPage = h.P();
  const uint32_t addr = 0x01100000;
  auto b1 = h.Pat(kPage * 2, 2);
  h.Write(addr, b1);
  REQUIRE(h.sm->RequestRange(addr, kPage * 2));
  REQUIRE(h.sm->MirrorRead(addr, kPage * 2) == b1);
  // Partial-page write: invalidating 16 bytes inside page 1 must invalidate
  // (at least) that whole page and re-upload it; page 0 stays valid.
  auto b2 = h.Pat(kPage * 2, 3);
  h.Write(addr, b2);
  size_t uploads_before = h.sm->upload_log.size();
  h.sm->MemoryInvalidationCallback(addr + kPage + 100, 16, true);
  REQUIRE(h.sm->RequestRange(addr, kPage * 2));
  REQUIRE(h.sm->upload_log.size() > uploads_before);
  auto got = h.sm->MirrorRead(addr, kPage * 2);
  // Page 1 has the fresh bytes.
  REQUIRE(std::memcmp(got.data() + kPage, b2.data() + kPage, kPage) == 0);
  // Page 0 was NOT invalidated: mirror still holds the earlier upload.
  REQUIRE(std::memcmp(got.data(), b1.data(), kPage) == 0);
}

TEST_CASE("SM logic: overlapping and cross-page ranges coalesce correctly",
          "[l0][sharedmem]") {
  REQUIRE(L().ok);
  auto& h = L();
  const uint32_t kPage = h.P();
  const uint32_t addr = 0x01200000;
  auto b = h.Pat(kPage * 4, 4);
  h.Write(addr, b);
  // Cross-page, unaligned range: base logic must validate every touched page.
  REQUIRE(h.sm->RequestRange(addr + 100, kPage * 2));
  auto got = h.sm->MirrorRead(addr, kPage * 3);
  REQUIRE(std::memcmp(got.data(), b.data(), kPage * 3) == 0);
  // Overlapping request extends validity without redundant re-upload of the
  // already-valid middle pages.
  size_t uploads_before = h.sm->upload_log.size();
  REQUIRE(h.sm->RequestRange(addr, kPage * 4));
  size_t new_pages = 0;
  for (size_t i = uploads_before; i < h.sm->upload_log.size(); ++i) {
    new_pages += h.sm->upload_log[i].second;
  }
  // Pages 0..2 were validated by the first request (100-offset start still
  // touches page 0); only page 3 may upload now.
  REQUIRE(new_pages <= 2);
  REQUIRE(h.sm->MirrorRead(addr, kPage * 4) == b);
}

TEST_CASE("SM logic: repeated invalidation/revalidation cycles converge",
          "[l0][sharedmem]") {
  REQUIRE(L().ok);
  auto& h = L();
  const uint32_t kPage = h.P();
  const uint32_t addr = 0x01300000;
  for (uint8_t i = 0; i < 6; ++i) {
    auto b = h.Pat(kPage, uint8_t(0x10 + i));
    h.Write(addr, b);
    h.sm->MemoryInvalidationCallback(addr, kPage, true);
    REQUIRE(h.sm->RequestRange(addr, kPage));
    REQUIRE(h.sm->MirrorRead(addr, kPage) == b);
  }
}

TEST_CASE("SM logic: backend invalidation can suppress only global watches",
          "[l0][sharedmem][authority]") {
  REQUIRE(L().ok);
  auto& h = L();
  const uint32_t addr = 0x01380000;
  struct WatchCounts {
    uint32_t global = 0;
    uint32_t range = 0;
  } counts;
  auto global_callback = [](const global_unique_lock_type&, void* context,
                            uint32_t, uint32_t, bool) {
    ++static_cast<WatchCounts*>(context)->global;
  };
  auto range_callback = [](const global_unique_lock_type&, void* context,
                           void*, uint64_t, bool) {
    ++static_cast<WatchCounts*>(context)->range;
  };
  auto global_watch =
      h.sm->RegisterGlobalWatch(global_callback, static_cast<void*>(&counts));
  REQUIRE(h.sm->WatchMemoryRange(addr, 0x100u, range_callback,
                                 static_cast<void*>(&counts), nullptr, 0) !=
          nullptr);

  h.sm->MemoryInvalidationCallback(addr, 0x100u, true,
                                   /*fire_global_watches=*/false);
  REQUIRE(counts.global == 0u);
  REQUIRE(counts.range == 1u);
  h.sm->UnregisterGlobalWatch(global_watch);
}

TEST_CASE("SM logic: physical boundary requests and invalidation clamp",
          "[l0][sharedmem]") {
  REQUIRE(L().ok);
  auto& h = L();
  const uint32_t kPage = h.P();
  const uint32_t end = 512u * 1024 * 1024;
  // Overshooting invalidation at the top of physical space is clamped by the
  // base logic (no crash, no wrap).
  h.sm->MemoryInvalidationCallback(end - kPage, kPage * 8, true);
  // Out-of-range request is rejected, in-range succeeds.
  REQUIRE_FALSE(h.sm->RequestRange(end - kPage, kPage * 2));
  const uint32_t addr = 0x1E000000;
  auto b = h.Pat(kPage, 0x77);
  h.Write(addr, b);
  REQUIRE(h.sm->RequestRange(addr, kPage));
  REQUIRE(h.sm->MirrorRead(addr, kPage) == b);
}

TEST_CASE("SM logic: RangeWrittenByGpu protects GPU-produced pages from "
          "stale CPU re-upload",
          "[l0][sharedmem]") {
  REQUIRE(L().ok);
  auto& h = L();
  const uint32_t kPage = h.P();
  const uint32_t addr = 0x01400000;
  auto ram = h.Pat(kPage, 0x40);
  h.Write(addr, ram);
  REQUIRE(h.sm->RequestRange(addr, kPage));
  // GPU writes the buffer (mirror): newest content exists GPU-side only.
  auto gpu = h.Pat(kPage, 0x50);
  auto it = h.sm->mirror.find(addr / kPage);
  REQUIRE(it != h.sm->mirror.end());
  REQUIRE(it->second.size() == kPage);
  std::memcpy(it->second.data(), gpu.data(), kPage);
  h.sm->RangeWrittenByGpu(addr, kPage);
  // A later consumer request must NOT re-upload stale RAM over it.
  size_t uploads_before = h.sm->upload_log.size();
  REQUIRE(h.sm->RequestRange(addr, kPage));
  REQUIRE(h.sm->upload_log.size() == uploads_before);
  REQUIRE(h.sm->MirrorRead(addr, kPage) == gpu);
  // Memexport-drain convergence: RAM is made to match the buffer, with no
  // invalidation; the buffer still serves the GPU-produced bytes.
  h.Write(addr, gpu);
  REQUIRE(h.sm->RequestRange(addr, kPage));
  REQUIRE(h.sm->MirrorRead(addr, kPage) == gpu);
}

}  // namespace
}  // namespace gpu
}  // namespace xe
