/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 1 (Metal device required; passes-as-skipped without one): alias-aware
// uploads and memexport convergence -- pre-gate-metal-zero-copy-shared-memory
// tasks 4.1-4.3.
//
// Everything here runs the REAL MetalSharedMemory over the REAL Memory, through
// the REAL typed convergence seam (ConvergeMemexportRam / DecideConsumer /
// DrainSelectedProducers). Nothing about copying is asserted from the backing
// MODE: the production shared memory reports the bytes it physically moved, and
// these cases assert that number. That distinction is the whole point of the
// group -- "alias mode is on" is a configuration, "zero bytes moved and guest
// RAM still holds the exported values" is a result.
//
// The two arms are the same code path with different backings, so every
// byte-level claim is made TWICE and compared: whatever the alias arm saves, it
// must publish the identical final guest bytes and the identical trace.
//
// Normal launch defaults are untouched: the gate is exact-off and every arm
// sets it explicitly around one Initialize, restoring the previous environment
// afterwards.

#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/memory.h"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_zero_copy_contract.h"
#include "xenia/gpu/trace_protocol.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// Own pages, 1 MiB apart, in a region no other suite in this binary touches, so
// a neighbouring fixture's write-watch state cannot bleed into an upload count.
constexpr uint32_t kConvergeBase = 0x1A000000u;
constexpr uint32_t kUploadBase = 0x1A100000u;
constexpr uint32_t kAwaitBase = 0x1A200000u;
constexpr uint32_t kFailBase = 0x1A300000u;
constexpr uint32_t kUntaggedBase = 0x1A400000u;
// SharedMemory's validity granularity is the HOST page (16 KiB on Apple
// Silicon), so an upload range is rounded up to it. Sizing the upload stimulus
// in host pages keeps "bytes uploaded" comparable to "bytes requested" without
// re-deriving the base class's rounding here.
constexpr uint32_t kPage = 4096u;  // guest allocation granularity.
uint32_t HostPage() { return uint32_t(xe::memory::page_size()); }

// Sets an environment variable for the duration of one arm and restores the
// exact previous state (including "was unset").
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* previous = std::getenv(name);
    had_previous_ = previous != nullptr;
    if (had_previous_) {
      previous_ = previous;
    }
    if (value) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (had_previous_) {
      setenv(name_, previous_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_previous_ = false;
  std::string previous_;
};

id<MTLDevice> Device() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

std::vector<uint8_t> SeededPattern(uint32_t bytes, uint8_t seed) {
  std::vector<uint8_t> v(bytes);
  for (uint32_t i = 0; i < bytes; ++i) {
    v[i] = uint8_t(seed + i * 7 + (i >> 8));
  }
  return v;
}

// One initialized MetalSharedMemory with the gate set for this arm. Alias state
// is read back from the shared memory rather than assumed from the request:
// a host that rejects the wrap runs the mirror in BOTH arms, and these cases
// then compare the mirror against itself instead of silently claiming an alias
// result they never had.
struct BackingArm {
  std::unique_ptr<MetalSharedMemory> shared;
  bool ok = false;

  explicit BackingArm(bool request_alias) {
    if (!Device()) {
      return;
    }
    Memory* memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    ScopedEnv gate(kZeroCopyGateEnvVar, request_alias ? "1" : nullptr);
    ScopedEnv force(kZeroCopyForceRejectEnvVar, nullptr);
    shared =
        std::make_unique<MetalSharedMemory>(*memory, (__bridge void*)Device());
    ok = shared->Initialize();
    if (!ok) {
      shared.reset();
    }
  }

  bool aliases() const { return shared && shared->is_zero_copy(); }
};

Memory& Guest() { return *conformance::TestPhysicalMemory(); }

uint8_t* RamAt(uint32_t address) {
  return Guest().TranslatePhysical<uint8_t*>(address);
}

// What a shader reading the shared buffer would see: the buffer's own bytes.
// In alias mode this pointer IS guest RAM, which is exactly the property under
// test -- so it is read through the buffer, never through TranslatePhysical.
const uint8_t* BufferAt(const MetalSharedMemory& shared, uint32_t address) {
  const uint8_t* contents = static_cast<const uint8_t*>(
      [(__bridge id<MTLBuffer>)shared.GetBuffer() contents]);
  return contents ? contents + address : nullptr;
}

// Commits the physical pages behind a range so it looks like a real guest
// allocation. UploadRanges validates heap access before copying, so a range no
// guest ever allocated is REJECTED -- correctly, and identically in both
// backings. These cases are about copying, not about that gate, so they present
// allocated memory the way the coherence suite does.
void EnsureAllocated(uint32_t address, uint32_t length) {
  uint32_t base = address & ~(kPage - 1);
  uint32_t end = (address + length + kPage - 1) & ~(kPage - 1);
  Guest().GetPhysicalHeap()->AllocFixed(
      base, end - base, kPage,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      kMemoryProtectRead | kMemoryProtectWrite);
}

void WriteRam(uint32_t address, const std::vector<uint8_t>& bytes) {
  EnsureAllocated(address, uint32_t(bytes.size()));
  std::memcpy(RamAt(address), bytes.data(), bytes.size());
}

std::vector<uint8_t> ReadRam(uint32_t address, uint32_t length) {
  std::vector<uint8_t> out(length);
  std::memcpy(out.data(), RamAt(address), length);
  return out;
}

std::vector<uint8_t> ReadBuffer(const MetalSharedMemory& shared,
                                uint32_t address, uint32_t length) {
  std::vector<uint8_t> out(length);
  std::memcpy(out.data(), BufferAt(shared, address), length);
  return out;
}

// Stands in for the memexport UAV store: deposits the exported bytes into the
// SHARED BUFFER, which is what a completed export leaves behind for convergence
// to publish. In alias mode that destination is guest RAM itself; not
// special-casing this here is deliberate, because the production writer does
// not special-case it either.
void StoreIntoSharedBuffer(MetalSharedMemory& shared, uint32_t address,
                           const std::vector<uint8_t>& bytes) {
  EnsureAllocated(address, uint32_t(bytes.size()));
  uint8_t* contents = const_cast<uint8_t*>(BufferAt(shared, address));
  REQUIRE(contents != nullptr);
  std::memcpy(contents, bytes.data(), bytes.size());
}

// The production convergence port, wired exactly as MetalCommandProcessor's
// MetalMemexportConvergenceTarget wires it: the shared memory owns the
// copy-or-alias decision and reports the bytes it moved; the target only
// forwards that fact (this harness constructs no texture cache, so the
// invalidation leg reports zero and is covered by its own suite).
class SharedMemoryConvergenceTarget final : public MemexportConvergenceTarget {
 public:
  explicit SharedMemoryConvergenceTarget(MetalSharedMemory* shared)
      : shared_(shared) {}

  bool CopyBufferToRam(uint32_t base, uint32_t length,
                       uint32_t* invalidated_entries,
                       uint64_t* bytes_copied) override {
    ++calls;
    *invalidated_entries = 0;
    SharedBufferConvergenceOutcome published =
        shared_->ConvergeRangeToRam(base, length);
    *bytes_copied = published.bytes_copied;
    alias_skipped = published.alias_skipped;
    return published.converged;
  }

  uint32_t calls = 0;
  bool alias_skipped = false;

 private:
  MetalSharedMemory* shared_;
};

PendingProducer MemexportRecord(uint32_t base, uint32_t length,
                                uint64_t submission, uint64_t sequence) {
  std::optional<GuestRange> range =
      GuestRange::FromExactBaseLength(base, length);
  REQUIRE(range.has_value());
  return PendingProducer{*range, TransitState::kPendingMemexport, submission,
                         sequence};
}

// One arm's upload evidence: what the shared memory reports it copied and
// aliased across a stimulus, plus what the GPU-visible bytes became.
struct UploadEvidence {
  uint64_t copied_delta = 0;
  uint64_t aliased_delta = 0;
  uint64_t repeat_copied_delta = 0;
  uint64_t repeat_aliased_delta = 0;
  std::vector<uint8_t> gpu_visible;
  std::vector<uint8_t> trace_bytes;
  bool aliased_backing = false;
  bool request_ok = false;
};

std::vector<uint8_t> ReadWholeFile(const std::filesystem::path& path) {
  std::vector<uint8_t> bytes;
  FILE* file = std::fopen(path.string().c_str(), "rb");
  if (!file) {
    return bytes;
  }
  uint8_t chunk[4096];
  size_t read = 0;
  while ((read = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
    bytes.insert(bytes.end(), chunk, chunk + read);
  }
  std::fclose(file);
  return bytes;
}

// True when the trace file contains a kMemoryRead command header naming EXACTLY
// this base and decoded length. Stronger than "the file is non-empty": it is the
// record a replay reads back to restore the range, so its presence is what makes
// the alias arm replayable at all. Scanned at 4-byte alignment because the
// command headers are written straight into the stream; a false positive would
// need the payload to contain this exact triple, and the trace is then also
// compared byte-for-byte against the mirror arm.
bool TraceRecordsMemoryRead(const std::vector<uint8_t>& trace, uint32_t base,
                            uint32_t length) {
  if (trace.size() < sizeof(MemoryCommand)) {
    return false;
  }
  for (size_t offset = 0; offset + sizeof(MemoryCommand) <= trace.size();
       offset += 4) {
    MemoryCommand cmd;
    std::memcpy(&cmd, trace.data() + offset, sizeof(cmd));
    if (cmd.type == TraceCommandType::kMemoryRead && cmd.base_ptr == base &&
        cmd.decoded_length == length) {
      return true;
    }
  }
  return false;
}

// Runs the identical upload stimulus on one backing: guest CPU writes the
// range, the physical-invalidation callback publishes that write, then a GPU
// consumer requests the range (the production entry that reaches UploadRanges).
// A second request immediately after answers a separate question -- whether the
// validity bookkeeping ran at all -- because pages that were never validated
// would be uploaded again.
UploadEvidence RunUploadArm(bool request_alias, uint32_t address,
                            uint32_t length, uint8_t seed,
                            const std::filesystem::path& trace_path) {
  UploadEvidence evidence;
  BackingArm arm(request_alias);
  REQUIRE(arm.ok);
  evidence.aliased_backing = arm.aliases();

  TraceWriter trace_writer(static_cast<uint8_t*>(Guest().TranslatePhysical(0)));
  REQUIRE(trace_writer.Open(trace_path, /*title_id=*/0x5A5A5A5Au));
  arm.shared->SetTraceWriter(&trace_writer);

  WriteRam(address, SeededPattern(length, seed));
  arm.shared->MemoryInvalidationCallback(address, length, /*exact_range=*/true);

  const uint64_t copied_before = arm.shared->upload_bytes_copied();
  const uint64_t aliased_before = arm.shared->upload_bytes_aliased();
  evidence.request_ok = arm.shared->RequestRange(address, length);
  evidence.copied_delta = arm.shared->upload_bytes_copied() - copied_before;
  evidence.aliased_delta = arm.shared->upload_bytes_aliased() - aliased_before;
  evidence.gpu_visible = ReadBuffer(*arm.shared, address, length);

  const uint64_t copied_mid = arm.shared->upload_bytes_copied();
  const uint64_t aliased_mid = arm.shared->upload_bytes_aliased();
  REQUIRE(arm.shared->RequestRange(address, length));
  evidence.repeat_copied_delta = arm.shared->upload_bytes_copied() - copied_mid;
  evidence.repeat_aliased_delta =
      arm.shared->upload_bytes_aliased() - aliased_mid;

  arm.shared->SetTraceWriter(nullptr);
  trace_writer.Close();
  evidence.trace_bytes = ReadWholeFile(trace_path);
  return evidence;
}

// One arm's convergence evidence over the same stimulus.
struct ConvergenceEvidence {
  bool converged = false;
  uint64_t bytes_copied = 0;
  bool alias_skipped = false;
  uint32_t target_calls = 0;
  uint64_t drained_bytes = 0;
  uint32_t drained_records = 0;
  std::vector<uint8_t> final_ram;
  bool aliased_backing = false;
};

// Prime RAM with a filler the export must overwrite, deposit the exported bytes
// in the shared buffer, then converge one COMPLETED record through the real
// seam and retire it. Whatever the backing, guest RAM must end holding the
// exported payload.
ConvergenceEvidence RunConvergenceArm(bool request_alias, uint32_t address,
                                      uint32_t length, uint8_t filler_seed,
                                      uint8_t export_seed) {
  ConvergenceEvidence evidence;
  BackingArm arm(request_alias);
  REQUIRE(arm.ok);
  evidence.aliased_backing = arm.aliases();

  WriteRam(address, SeededPattern(length, filler_seed));
  StoreIntoSharedBuffer(*arm.shared, address, SeededPattern(length,
                                                            export_seed));

  std::vector<PendingProducer> pending = {
      MemexportRecord(address, length, /*submission=*/4u, /*sequence=*/1u)};
  GuestRangeSet requested;
  requested.Add(GuestRange::FromExactBaseLength(address, length));
  // The export's submission has completed, so the real decision is a drain.
  ConsumerDecision decision = DecideConsumer(ConsumerKind::kVertex, requested,
                                             pending, /*completed=*/4u);
  REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE_FALSE(decision.await);

  SharedMemoryConvergenceTarget target(arm.shared.get());
  ProducerDrainOutcome drain = DrainSelectedProducers(
      decision, [&](const PendingProducer& selected) {
        ProducerDrainOutcome one;
        // Through the shared production gate, with the same completed
        // submission the decision was built from.
        GatedMemexportConvergence gated =
            ConvergeCompletedMemexportRam(selected, /*completed=*/4u, target);
        REQUIRE(gated.attempted());
        evidence.converged = gated.result.converged;
        evidence.bytes_copied = gated.result.bytes_copied;
        if (!gated.retires()) {
          return one;  // retained: drained_records stays 0.
        }
        one.drained_records = 1;
        one.memexport_records = 1;
        one.memexport_bytes_copied = gated.result.bytes_copied;
        pending.clear();  // exactly-once retirement of the converged record.
        return one;
      });
  evidence.alias_skipped = target.alias_skipped;
  evidence.target_calls = target.calls;
  evidence.drained_bytes = drain.memexport_bytes_copied;
  evidence.drained_records = drain.drained_records;
  REQUIRE(pending.empty());
  evidence.final_ram = ReadRam(address, length);
  return evidence;
}

}  // namespace

TEST_CASE("Zero-copy group 4: the device harness is live (not skipped)",
          "[l1][metal][zerocopy][convergence]") {
  if (!Device()) {
    WARN("no Metal device; zero-copy group-4 device arms are skipped");
  }
  SUCCEED();
}

TEST_CASE("Zero-copy convergence: mirror copies the record's bytes, alias "
          "copies none, and both publish the same guest bytes",
          "[l1][metal][zerocopy][convergence]") {
  if (!Device()) {
    return;
  }
  const uint32_t length = kPage;
  const std::vector<uint8_t> expected = SeededPattern(length, 0x31);

  ConvergenceEvidence mirror =
      RunConvergenceArm(/*request_alias=*/false, kConvergeBase, length,
                        /*filler_seed=*/0x11, /*export_seed=*/0x31);
  ConvergenceEvidence alias =
      RunConvergenceArm(/*request_alias=*/true, kConvergeBase, length,
                        /*filler_seed=*/0x11, /*export_seed=*/0x31);

  // Mirror: two allocations, so publication IS a copy of exactly the record's
  // length, counted at the memcpy that moved them.
  CHECK_FALSE(mirror.aliased_backing);
  CHECK(mirror.converged);
  CHECK_FALSE(mirror.alias_skipped);
  CHECK(mirror.bytes_copied == uint64_t(length));
  CHECK(mirror.drained_bytes == uint64_t(length));
  CHECK(mirror.drained_records == 1u);
  CHECK(mirror.final_ram == expected);

  if (!alias.aliased_backing) {
    // This host rejected the wrap (recorded by the initialization suite). There
    // is no alias arm to compare, and claiming one would be false.
    WARN("host does not alias guest RAM; the alias convergence arm is the "
         "mirror path and is not asserted as zero-copy");
    CHECK(alias.bytes_copied == uint64_t(length));
    return;
  }

  // Alias: the same record retires, the same invalidation/accounting steps run,
  // and ZERO bytes move -- the copy was suppressed because the destination was
  // PROVEN to be the source, not because a flag said so.
  CHECK(alias.converged);
  CHECK(alias.alias_skipped);
  CHECK(alias.bytes_copied == 0u);
  CHECK(alias.drained_bytes == 0u);
  CHECK(alias.drained_records == 1u);
  CHECK(alias.target_calls == 1u);

  // The claim that matters: saving the copy did not change what the guest sees.
  CHECK(alias.final_ram == expected);
  CHECK(alias.final_ram == mirror.final_ram);
}

TEST_CASE("Zero-copy convergence: a claimed alias whose addresses disagree "
          "must not suppress the copy",
          "[l0][metal][zerocopy][convergence]") {
  // The device cannot produce this state: once the wrap is accepted, the
  // buffer's contents ARE the guest base, so contents + base equals
  // TranslatePhysical(base) for every in-domain base. The rule is therefore
  // pinned where it is reachable -- as a value -- and production calls this
  // exact function (metal_shared_memory.mm, ConvergeRangeToRam). The exhaustive
  // combination sweep lives in metal_zero_copy_contract_test.cc; what this case
  // adds is the consequence: a disagreement is NOT a licensed skip.
  uint8_t elsewhere = 0;
  uint8_t here = 0;
  CHECK_FALSE(ZeroCopySelfCopyIsProven(true, &elsewhere, &here));
  CHECK(ZeroCopySelfCopyIsProven(true, &here, &here));
  // ...and the mirror backing never takes the skip even when handed the same
  // address twice, so the mode is a necessary condition too.
  CHECK_FALSE(ZeroCopySelfCopyIsProven(false, &here, &here));
}

TEST_CASE("Zero-copy convergence: degenerate and out-of-domain ranges publish "
          "nothing and move nothing",
          "[l1][metal][zerocopy][convergence]") {
  if (!Device()) {
    return;
  }
  const uint32_t kBufferSize = SharedMemory::kBufferSize;
  auto check_rejected = [](MetalSharedMemory& shared, uint32_t base,
                           uint32_t length) {
    SharedBufferConvergenceOutcome outcome =
        shared.ConvergeRangeToRam(base, length);
    // Not converged: the caller retains its record instead of retiring bytes
    // nothing published. And nothing is reported as moved or as an alias skip,
    // so a rejected range can never look like a zero-copy success.
    CHECK_FALSE(outcome.converged);
    CHECK(outcome.bytes_copied == 0u);
    CHECK_FALSE(outcome.alias_skipped);
  };

  for (bool request_alias : {false, true}) {
    BackingArm arm(request_alias);
    REQUIRE(arm.ok);
    // Zero length: an empty publication is not a publication.
    check_rejected(*arm.shared, kConvergeBase, 0u);
    // Base at and past the end of the guest physical domain.
    check_rejected(*arm.shared, kBufferSize, kPage);
    check_rejected(*arm.shared, kBufferSize + kPage, kPage);
    // Length running off the end -- rejected without ever forming base+length.
    check_rejected(*arm.shared, kBufferSize - kPage, kPage + 1u);
    check_rejected(*arm.shared, kBufferSize - 1u, 2u);
    // Maximal length from a valid base: base + length overflows uint32 entirely.
    check_rejected(*arm.shared, kConvergeBase, UINT32_MAX);
    check_rejected(*arm.shared, 0u, UINT32_MAX);
    // An exact-fit range is NOT rejected: hardening must not shrink the
    // legitimate domain. Kept off the very last page (whose guest mapping is
    // not this suite's to commit) but still an exactly-bounded interval.
    const uint32_t fit_base = kConvergeBase;
    const uint32_t fit_length = kPage;
    EnsureAllocated(fit_base, fit_length);
    SharedBufferConvergenceOutcome fit =
        arm.shared->ConvergeRangeToRam(fit_base, fit_length);
    CHECK(fit.converged);
  }
}

TEST_CASE("Zero-copy convergence: a failed publication retains the record",
          "[l1][metal][zerocopy][convergence]") {
  if (!Device()) {
    return;
  }
  const uint32_t length = kPage;
  BackingArm arm(/*request_alias=*/true);
  REQUIRE(arm.ok);

  SharedMemoryConvergenceTarget target(arm.shared.get());
  std::vector<PendingProducer> pending = {
      MemexportRecord(kFailBase, length, /*submission=*/2u, /*sequence=*/1u)};

  // A real production failure, not a stubbed one: after Shutdown the backing is
  // gone, so there is nothing to publish from.
  arm.shared->Shutdown();

  GuestRangeSet requested;
  requested.Add(GuestRange::FromExactBaseLength(kFailBase, length));
  ConsumerDecision decision = DecideConsumer(ConsumerKind::kVertex, requested,
                                             pending, /*completed=*/2u);
  REQUIRE(decision.selected.size() == 1u);

  ProducerDrainOutcome drain = DrainSelectedProducers(
      decision, [&](const PendingProducer& selected) {
        ProducerDrainOutcome one;
        GatedMemexportConvergence gated =
            ConvergeCompletedMemexportRam(selected, /*completed=*/2u, target);
        // The gate ATTEMPTED (the submission is complete); the target failed.
        CHECK(gated.attempted());
        CHECK_FALSE(gated.result.converged);
        CHECK(gated.result.action == AuthorityAction::kNoDependency);
        CHECK(gated.result.bytes_copied == 0u);
        if (gated.retires()) {
          one.drained_records = 1;
          one.memexport_records = 1;
          pending.clear();
        }
        return one;
      });

  CHECK(target.calls == 1u);
  CHECK(drain.drained_records == 0u);
  CHECK(drain.memexport_bytes_copied == 0u);
  // The record survives for retry, and the publication is therefore incomplete
  // by the same predicate production presents on.
  CHECK(pending.size() == 1u);
  CHECK_FALSE(ConsumerPublicationComplete(decision, drain));
}

TEST_CASE("Zero-copy convergence: an incomplete submission neither converges "
          "nor retires",
          "[l1][metal][zerocopy][convergence]") {
  if (!Device()) {
    return;
  }
  const uint32_t length = kPage;
  BackingArm arm(/*request_alias=*/true);
  REQUIRE(arm.ok);
  EnsureAllocated(kAwaitBase, length);

  // The export wrote the buffer under submission 9; the GPU has only finished
  // 8. Aliasing does not make an unfinished GPU write guest-visible, so this
  // must behave exactly as the mirror does.
  std::vector<PendingProducer> pending = {
      MemexportRecord(kAwaitBase, length, /*submission=*/9u, /*sequence=*/1u)};
  REQUIRE_FALSE(ProducerComplete(pending.front(), 8u));
  REQUIRE(ProducerComplete(pending.front(), 9u));

  GuestRangeSet requested;
  requested.Add(GuestRange::FromExactBaseLength(kAwaitBase, length));
  ConsumerDecision awaiting = DecideConsumer(ConsumerKind::kVertex, requested,
                                             pending, /*completed=*/8u);
  CHECK(awaiting.action == AuthorityAction::kAwaitProducer);
  CHECK(awaiting.await);
  CHECK(awaiting.max_wait_submission == 9u);

  // The drain runs through the SAME shared gate production uses. Even driven
  // past the await -- i.e. with the caller ignoring the decision -- an
  // incomplete record must not reach the convergence target at all.
  SharedMemoryConvergenceTarget target(arm.shared.get());
  ProducerDrainOutcome premature = DrainSelectedProducers(
      awaiting, [&](const PendingProducer& selected) {
        ProducerDrainOutcome one;
        GatedMemexportConvergence gated =
            ConvergeCompletedMemexportRam(selected, /*completed=*/8u, target);
        CHECK_FALSE(gated.attempted());
        CHECK_FALSE(gated.retires());
        CHECK(gated.gate ==
              MemexportConvergenceGate::kNotAttemptedIncompleteSubmission);
        return one;  // retained.
      });
  CHECK(target.calls == 0u);
  CHECK(premature.drained_records == 0u);
  CHECK(premature.memexport_bytes_copied == 0u);
  CHECK(pending.size() == 1u);
  CHECK_FALSE(ConsumerPublicationComplete(awaiting, premature));

  // Once the submission completes, the same range drains through the same seam.
  ConsumerDecision completed = DecideConsumer(ConsumerKind::kVertex, requested,
                                              pending, /*completed=*/9u);
  CHECK(completed.action == AuthorityAction::kDrainMatchingRecords);
  ProducerDrainOutcome drain = DrainSelectedProducers(
      completed, [&](const PendingProducer& selected) {
        ProducerDrainOutcome one;
        GatedMemexportConvergence gated =
            ConvergeCompletedMemexportRam(selected, /*completed=*/9u, target);
        REQUIRE(gated.attempted());
        REQUIRE(gated.retires());
        one.drained_records = 1;
        one.memexport_records = 1;
        one.memexport_bytes_copied = gated.result.bytes_copied;
        pending.clear();
        return one;
      });
  CHECK(target.calls == 1u);
  CHECK(drain.drained_records == 1u);
  CHECK(pending.empty());
  CHECK(ConsumerPublicationComplete(completed, drain));
}

TEST_CASE("Zero-copy convergence: an untagged record publishes only under "
          "synchronous-boundary evidence",
          "[l1][metal][zerocopy][convergence]") {
  if (!Device()) {
    return;
  }
  // Submission 0 is the DEFAULT record shape on this backend, not a diagnostic:
  // S7TagsOff() is true unless XE_METAL_ASYNC is set, so the publication seam
  // registers `S7TagsOff() ? 0 : CurrentEncodingSubmission()`. Nothing about the
  // record decides its finality; the consumer path's commit-and-wait does. This
  // runs BOTH backings, because aliasing must not change the lifecycle.
  const uint32_t length = kPage;
  const std::vector<uint8_t> exported = SeededPattern(length, 0x5D);
  const std::vector<uint8_t> filler = SeededPattern(length, 0x0E);

  for (bool request_alias : {false, true}) {
    BackingArm arm(request_alias);
    REQUIRE(arm.ok);
    INFO("alias requested: " << request_alias
                             << " aliased: " << arm.aliases());

    WriteRam(kUntaggedBase, filler);
    StoreIntoSharedBuffer(*arm.shared, kUntaggedBase, exported);

    std::vector<PendingProducer> pending = {
        MemexportRecord(kUntaggedBase, length, /*submission=*/0u,
                        /*sequence=*/1u)};
    GuestRangeSet requested;
    requested.Add(GuestRange::FromExactBaseLength(kUntaggedBase, length));
    // DecideConsumer still awaits an untagged record -- the tag proves nothing,
    // so the consumer path must reach its boundary before draining.
    ConsumerDecision decision = DecideConsumer(ConsumerKind::kVertex, requested,
                                               pending, /*completed=*/7u);
    REQUIRE(decision.selected.size() == 1u);
    CHECK(decision.await);

    // What guest RAM holds BEFORE any convergence. In the mirror backing that
    // is still the filler; in the alias backing the store above landed in guest
    // RAM itself, because the buffer IS guest RAM. That difference is the
    // backing, not the publication -- so this is captured rather than assumed,
    // and the no-publication claim below is "RAM did not change", which is the
    // claim that means the same thing in both arms.
    const std::vector<uint8_t> ram_before_convergence =
        ReadRam(kUntaggedBase, length);
    CHECK(ram_before_convergence == (arm.aliases() ? exported : filler));

    // Without the caller's boundary proof the record is NOT publishable:
    // nothing was read, nothing moved, nothing retired, and guest RAM is
    // unchanged.
    SharedMemoryConvergenceTarget unproven_target(arm.shared.get());
    uint32_t unproven_observations = 0;
    ProducerDrainOutcome unproven = DrainSelectedProducers(
        decision, [&](const PendingProducer& selected) {
          ProducerDrainOutcome one;
          GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
              selected, /*completed=*/7u, MemexportBoundaryEvidence::kNone,
              unproven_target, [&]() { ++unproven_observations; });
          CHECK(gated.gate ==
                MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary);
          CHECK_FALSE(gated.attempted());
          CHECK_FALSE(gated.retires());
          return one;
        });
    CHECK(unproven_target.calls == 0u);
    CHECK(unproven_observations == 0u);
    CHECK(unproven.drained_records == 0u);
    CHECK(pending.size() == 1u);
    CHECK_FALSE(ConsumerPublicationComplete(decision, unproven));
    CHECK(ReadRam(kUntaggedBase, length) == ram_before_convergence);

    // With it -- the evidence the production path mints only after
    // EndFrameEncoderAndWait and a proven high-water mark -- the same record
    // publishes exactly once and retires, in both backings.
    SharedMemoryConvergenceTarget proven_target(arm.shared.get());
    uint32_t proven_observations = 0;
    ProducerDrainOutcome proven = DrainSelectedProducers(
        decision, [&](const PendingProducer& selected) {
          ProducerDrainOutcome one;
          GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
              selected, /*completed=*/7u,
              MemexportBoundaryEvidence::kSynchronousBoundaryCompleted,
              proven_target, [&]() {
                // Runs before the publication, so guest RAM here still holds
                // whatever the BACKING left there -- the filler in the mirror
                // arm, the exported bytes in the alias arm, exactly as
                // ram_before_convergence captured above. The claim is "the
                // publication has not run yet", which is why the observation
                // is inside the gate rather than around it.
                ++proven_observations;
                CHECK(ReadRam(kUntaggedBase, length) ==
                      ram_before_convergence);
              });
          CHECK(gated.gate ==
                MemexportConvergenceGate::kAttemptedSynchronousBoundary);
          REQUIRE(gated.retires());
          one.drained_records = 1;
          one.memexport_records = 1;
          one.memexport_bytes_copied = gated.result.bytes_copied;
          pending.clear();
          return one;
        });
    CHECK(proven_target.calls == 1u);
    CHECK(proven_observations == 1u);
    CHECK(proven.drained_records == 1u);
    CHECK(pending.empty());
    CHECK(ConsumerPublicationComplete(decision, proven));
    // The published bytes, whichever backing moved them (or moved none).
    CHECK(ReadRam(kUntaggedBase, length) == exported);
    CHECK(proven.memexport_bytes_copied ==
          (arm.aliases() ? 0u : uint64_t(length)));
  }
}

TEST_CASE("Zero-copy uploads: the mirror copies, the alias makes the same "
          "bytes GPU-visible without copying",
          "[l1][metal][zerocopy][convergence]") {
  if (!Device()) {
    return;
  }
  const uint32_t length = 2 * HostPage();
  const std::vector<uint8_t> expected = SeededPattern(length, 0x7C);
  const std::filesystem::path mirror_trace =
      std::filesystem::temp_directory_path() / "xe_zero_copy_mirror.xtr";
  const std::filesystem::path alias_trace =
      std::filesystem::temp_directory_path() / "xe_zero_copy_alias.xtr";

  UploadEvidence mirror = RunUploadArm(/*request_alias=*/false, kUploadBase,
                                       length, 0x7C, mirror_trace);
  UploadEvidence alias = RunUploadArm(/*request_alias=*/true, kUploadBase,
                                      length, 0x7C, alias_trace);

  CHECK(mirror.request_ok);
  CHECK_FALSE(mirror.aliased_backing);
  CHECK(mirror.copied_delta == uint64_t(length));
  CHECK(mirror.aliased_delta == 0u);
  CHECK(mirror.gpu_visible == expected);
  // Validity bookkeeping: the pages are valid now, so the immediate repeat
  // uploads nothing.
  CHECK(mirror.repeat_copied_delta == 0u);
  // Trace bookkeeping: the uploaded source bytes were recorded, as a memory-read
  // record naming this exact base and length.
  CHECK(mirror.trace_bytes.size() > 0u);
  CHECK(TraceRecordsMemoryRead(mirror.trace_bytes, kUploadBase, length));

  CHECK(alias.request_ok);
  if (!alias.aliased_backing) {
    WARN("host does not alias guest RAM; the alias upload arm is the mirror "
         "path and is not asserted as zero-copy");
    return;
  }

  // No CPU memcpy and no Metal copy -- and the GPU-visible bytes are the
  // written bytes anyway, which is the only reason suppressing the copy is
  // legitimate.
  CHECK(alias.copied_delta == 0u);
  CHECK(alias.aliased_delta == uint64_t(length));
  CHECK(alias.gpu_visible == expected);
  CHECK(alias.gpu_visible == mirror.gpu_visible);

  // Validity bookkeeping is NOT inside the copy branch: had MakeRangeValid been
  // skipped with the copy, the same pages would be presented for upload again.
  CHECK(alias.repeat_copied_delta == 0u);
  CHECK(alias.repeat_aliased_delta == 0u);

  // Neither is trace recording: a replay of the alias arm must restore exactly
  // the bytes the mirror arm's replay would, so the two traces are compared
  // byte for byte rather than merely both being non-empty.
  CHECK(TraceRecordsMemoryRead(alias.trace_bytes, kUploadBase, length));
  CHECK(alias.trace_bytes.size() == mirror.trace_bytes.size());
  CHECK(alias.trace_bytes == mirror.trace_bytes);

  std::error_code ignored;
  std::filesystem::remove(mirror_trace, ignored);
  std::filesystem::remove(alias_trace, ignored);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
