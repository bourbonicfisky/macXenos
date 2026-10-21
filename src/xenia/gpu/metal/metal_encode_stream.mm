/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_encode_stream.h"

#include <pthread.h>
#include <pthread/qos.h>
#include <sys/qos.h>

#include <os/os_sync_wait_on_address.h>

#include <chrono>
#include <cstdlib>

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"

namespace xe {
namespace gpu {
namespace metal {

enum class MetalEncodeStream::Op : uint16_t {
  kWrap,
  kStop,
  kSetEncoder,
  kViewport,
  kDepthClipMode,
  kCullMode,
  kWinding,
  kFillMode,
  kDepthBias,
  kScissor,
  kPipeline,
  kBlendColor,
  kDepthStencil,
  kStencilRef,
  kVisibility,
  kBuffer,
  kBufferOffset,
  kBytes,
  kUseResource,
  kDraw,
  kDrawIndexed,
  kIRDraw,
  kIRDrawIndexed,
  kIRDrawIndexedGeometry,
  kIRDrawGeometry,
};

namespace {

inline void SpinPause() { __asm__ __volatile__("isb sy" ::: "memory"); }

// Spins before parking: the producer records a draw every ~2 us, so a short
// spin keeps the consumer hot inside a pass and parks it between frames.
// XE_METAL_ENCODE_SPINS: polls before the consumer parks (default 4096).
uint32_t ConsumerSpins() {
  static const uint32_t s_spins = [] {
    const char* v = std::getenv("XE_METAL_ENCODE_SPINS");
    return v ? uint32_t(strtoul(v, nullptr, 10)) : 4096u;
  }();
  return s_spins;
}
// XE_METAL_ENCODE_LAZY_WAKE_US=<n> (default 0 = off): a batched publish never
// wakes a parked consumer; it parks for at most n us and picks the batch up
// then. Drains and a full ring still wake it at once. Trades replay lag
// inside a pass (absorbed by the next drain) for not spinning a core between
// batches, with no wake syscall on the command processor.
uint32_t LazyWakeUs() {
  static const uint32_t s_us = [] {
    const char* v = std::getenv("XE_METAL_ENCODE_LAZY_WAKE_US");
    return v ? uint32_t(strtoul(v, nullptr, 10)) : 0u;
  }();
  return s_us;
}
constexpr uint32_t kProducerSpins = 1 << 14;
// Publication batch: each publish is a store the spinning consumer's core
// owns the line of, so per-draw publication (~140 B) paid a cross-core miss
// every draw. ~2 KB is ~15 draws, which keeps a Drain's replay tail short.
// XE_METAL_ENCODE_PUBLISH_BYTES overrides it (0 = every draw).
uint64_t PublishBytes() {
  static const uint64_t s_bytes = [] {
    const char* v = std::getenv("XE_METAL_ENCODE_PUBLISH_BYTES");
    return v ? uint64_t(strtoull(v, nullptr, 10)) : uint64_t(2048);
  }();
  return s_bytes;
}
// Park timeout: bounds the delay of a wake lost to the batched publish's
// relaxed waiting check.
constexpr uint32_t kParkTimeoutUs = 1000;

struct PtrOp {
  void* object;
};
struct U32Op {
  uint32_t value;
  uint32_t pad;
};
struct ViewportOp {
  MTLViewport viewport;
};
struct DepthBiasOp {
  float constant, slope, clamp, pad;
};
struct ScissorOp {
  MTLScissorRect rect;
};
struct BlendColorOp {
  float rgba[4];
};
struct StencilRefOp {
  uint32_t front, back;
};
struct VisibilityOp {
  uint64_t mode;
  uint64_t offset;
};
struct BufferOp {
  void* buffer;
  uint64_t offset;
  uint32_t index;
  uint32_t stage;
};
struct BufferOffsetOp {
  uint64_t offset;
  uint32_t index;
  uint32_t stage;
};
struct BytesOp {
  uint32_t length;
  uint32_t index;
  uint32_t stage;
  uint32_t pad;
  // Followed by `length` bytes.
};
struct UseResourceOp {
  void* resource;
  uint64_t usage;
  uint64_t stages;
  uint64_t retained;
};
struct DrawOp {
  uint64_t type;
  uint64_t start;
  uint64_t count;
};
struct DrawIndexedOp {
  uint64_t type;
  uint64_t count;
  uint64_t index_type;
  void* buffer;
  uint64_t offset;
  uint64_t instance_count;
  int64_t base_vertex;
  uint64_t base_instance;
};
struct IRDrawIndexedGeometryOp {
  IRRuntimeGeometryPipelineConfig config;
  void* buffer;
  uint32_t type;
  uint32_t index_type;
  uint32_t instance_count;
  uint32_t index_count;
  uint32_t start_index;
  int32_t base_vertex;
  uint32_t base_instance;
  uint32_t pad;
};
struct IRDrawGeometryOp {
  IRRuntimeGeometryPipelineConfig config;
  uint32_t type;
  uint32_t instance_count;
  uint32_t vertex_count;
  uint32_t base_vertex;
  uint32_t base_instance;
  uint32_t pad;
};

constexpr uint32_t RoundUp8(uint32_t v) { return (v + 7u) & ~7u; }

}  // namespace

MetalEncodeStream::MetalEncodeStream() = default;

MetalEncodeStream::~MetalEncodeStream() { Shutdown(); }

MetalEncodeStream::Mode MetalEncodeStream::ModeFromEnvironment() {
  const char* v = std::getenv("XE_METAL_ENCODE_THREAD");
  if (!v || !v[0]) {
    return Mode::kThreaded;
  }
  switch (v[0]) {
    case '0':
      return Mode::kDirect;
    case '1':
      return Mode::kInline;
    default:
      return Mode::kThreaded;
  }
}

void MetalEncodeStream::Start(Mode mode) {
  assert_true(!ring_);
  mode_ = mode;
  if (mode_ == Mode::kDirect) {
    return;
  }
  ring_ = static_cast<uint8_t*>(std::aligned_alloc(64, size_t(kCapacity)));
  if (mode_ == Mode::kThreaded) {
    thread_ = std::thread([this]() { ConsumerMain(); });
  }
  XELOGI("MetalEncodeStream: mode {} ({} MiB ring)", uint32_t(mode_),
         kCapacity >> 20);
}

void MetalEncodeStream::Shutdown() {
  if (!ring_) {
    return;
  }
  if (mode_ == Mode::kThreaded && thread_.joinable()) {
    Emit<U32Op>(Op::kStop);
    Publish(true);
    thread_.join();
  } else {
    Commit();
  }
  std::free(ring_);
  ring_ = nullptr;
  mode_ = Mode::kDirect;
}

void* MetalEncodeStream::Reserve(Op op, uint32_t payload_size) {
  const uint32_t total = RoundUp8(uint32_t(sizeof(OpHeader)) + payload_size);
  uint64_t index = write_pos_ & (kCapacity - 1);
  const uint64_t padding =
      index + total > kCapacity ? kCapacity - index : uint64_t(0);
  if (write_pos_ + padding + total - consumed_cached_ > kCapacity) {
    consumed_cached_ = consumed_.load(std::memory_order_acquire);
    if (write_pos_ + padding + total - consumed_cached_ > kCapacity) {
      // Full (threaded only: inline replays every Commit and a draw is far
      // below the capacity). Let the consumer see what is recorded, then wait.
      ++stats_.full_waits;
      Publish(true);
      uint32_t spins = 0;
      while (write_pos_ + padding + total - consumed_cached_ > kCapacity) {
        if (++spins < kProducerSpins) {
          SpinPause();
        } else {
          std::this_thread::yield();
        }
        consumed_cached_ = consumed_.load(std::memory_order_acquire);
      }
    }
  }
  if (padding) {
    auto* wrap = reinterpret_cast<OpHeader*>(ring_ + index);
    wrap->op = Op::kWrap;
    wrap->size = uint32_t(padding);
    write_pos_ += padding;
    index = 0;
  }
  auto* header = reinterpret_cast<OpHeader*>(ring_ + index);
  header->op = op;
  header->size = total;
  write_pos_ += total;
  stats_.bytes += total;
  return header + 1;
}

void MetalEncodeStream::Wake() {
  wake_word_.fetch_add(1, std::memory_order_release);
  os_sync_wake_by_address_any(&wake_word_, sizeof(uint32_t),
                              OS_SYNC_WAKE_BY_ADDRESS_NONE);
}

void MetalEncodeStream::Publish(bool force) {
  if (write_pos_ == published_local_) {
    if (force &&
        consumed_.load(std::memory_order_acquire) != published_local_ &&
        consumer_waiting_.load(std::memory_order_seq_cst)) {
      Wake();
    }
    return;
  }
  if (!force && write_pos_ - published_local_ < PublishBytes()) {
    return;
  }
  published_local_ = write_pos_;
  if (force) {
    // Sequentially consistent store + load pair against the consumer's
    // waiting flag store + published load, so a parking consumer can't miss
    // it.
    published_.store(write_pos_, std::memory_order_seq_cst);
    if (consumer_waiting_.load(std::memory_order_seq_cst)) {
      Wake();
    }
    return;
  }
  published_.store(write_pos_, std::memory_order_release);
  if (!LazyWakeUs() && consumer_waiting_.load(std::memory_order_relaxed)) {
    Wake();
  }
}

void MetalEncodeStream::Commit() {
  if (mode_ == Mode::kDirect) {
    return;
  }
  ++stats_.commits;
  if (mode_ == Mode::kInline) {
    const uint64_t begin = consumed_.load(std::memory_order_relaxed);
    if (begin != write_pos_) {
      @autoreleasepool {
        ReplayRange(begin, write_pos_);
      }
      published_local_ = write_pos_;
      consumed_.store(write_pos_, std::memory_order_relaxed);
    }
    return;
  }
  Publish(false);
}

void MetalEncodeStream::Drain() {
  if (mode_ != Mode::kThreaded) {
    Commit();
    return;
  }
  Publish(true);
  ++stats_.drains;
  if (consumed_.load(std::memory_order_acquire) == write_pos_) {
    return;
  }
  ++stats_.drain_waits;
  const auto start = std::chrono::steady_clock::now();
  uint32_t spins = 0;
  for (;;) {
    uint64_t consumed = consumed_.load(std::memory_order_acquire);
    if (consumed == write_pos_) {
      break;
    }
    if (++spins < kProducerSpins) {
      SpinPause();
      continue;
    }
    producer_waiting_.store(1, std::memory_order_seq_cst);
    consumed = consumed_.load(std::memory_order_seq_cst);
    if (consumed != write_pos_) {
      consumed_.wait(consumed, std::memory_order_acquire);
    }
    producer_waiting_.store(0, std::memory_order_relaxed);
  }
  stats_.drain_wait_ns += uint64_t(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

MetalEncodeStream::Stats MetalEncodeStream::TakeStats() {
  Stats s = stats_;
  s.consumer_sleeps = consumer_sleeps_.exchange(0, std::memory_order_relaxed);
  stats_ = Stats();
  return s;
}

void MetalEncodeStream::ConsumerMain() {
  pthread_setname_np("Metal Encoder");
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  uint64_t pos = 0;
  for (;;) {
    uint64_t end = published_.load(std::memory_order_acquire);
    if (end == pos) {
      uint32_t spins = 0;
      const uint32_t max_spins = ConsumerSpins();
      while (end == pos && ++spins < max_spins) {
        SpinPause();
        end = published_.load(std::memory_order_acquire);
      }
      if (end == pos) {
        const uint32_t wake = wake_word_.load(std::memory_order_acquire);
        consumer_waiting_.store(1, std::memory_order_seq_cst);
        end = published_.load(std::memory_order_seq_cst);
        if (end == pos) {
          consumer_sleeps_.fetch_add(1, std::memory_order_relaxed);
          os_sync_wait_on_address_with_timeout(
              &wake_word_, uint64_t(wake), sizeof(uint32_t),
              OS_SYNC_WAIT_ON_ADDRESS_NONE, OS_CLOCK_MACH_ABSOLUTE_TIME,
              uint64_t(LazyWakeUs() ? LazyWakeUs() : kParkTimeoutUs) * 1000);
          end = published_.load(std::memory_order_acquire);
        }
        consumer_waiting_.store(0, std::memory_order_relaxed);
        if (end == pos) {
          continue;
        }
      }
    }
    @autoreleasepool {
      ReplayRange(pos, end);
    }
    pos = end;
    // Only once caught up with what is published: an intermediate store would
    // be one more line the producer's Drain/space check has to pull over.
    if (published_.load(std::memory_order_acquire) != pos) {
      continue;
    }
    consumed_.store(pos, std::memory_order_seq_cst);
    if (producer_waiting_.load(std::memory_order_seq_cst)) {
      consumed_.notify_one();
    }
    if (stop_.load(std::memory_order_relaxed)) {
      return;
    }
  }
}

void MetalEncodeStream::ReplayRange(uint64_t begin, uint64_t end) {
  __unsafe_unretained id<MTLRenderCommandEncoder> enc = replay_encoder_;
  uint64_t pos = begin;
  while (pos != end) {
    const auto* header =
        reinterpret_cast<const OpHeader*>(ring_ + (pos & (kCapacity - 1)));
    const void* payload = header + 1;
    switch (header->op) {
      case Op::kWrap:
        break;
      case Op::kStop:
        stop_.store(true, std::memory_order_relaxed);
        break;
      case Op::kSetEncoder:
        enc = (__bridge id<MTLRenderCommandEncoder>)(
            static_cast<const PtrOp*>(payload)->object);
        break;
      case Op::kViewport:
        [enc setViewport:static_cast<const ViewportOp*>(payload)->viewport];
        break;
      case Op::kDepthClipMode:
        [enc setDepthClipMode:MTLDepthClipMode(
                                  static_cast<const U32Op*>(payload)->value)];
        break;
      case Op::kCullMode:
        [enc setCullMode:MTLCullMode(static_cast<const U32Op*>(payload)->value)];
        break;
      case Op::kWinding:
        [enc setFrontFacingWinding:MTLWinding(
                                       static_cast<const U32Op*>(payload)->value)];
        break;
      case Op::kFillMode:
        [enc setTriangleFillMode:MTLTriangleFillMode(
                                     static_cast<const U32Op*>(payload)->value)];
        break;
      case Op::kDepthBias: {
        const auto* p = static_cast<const DepthBiasOp*>(payload);
        [enc setDepthBias:p->constant slopeScale:p->slope clamp:p->clamp];
      } break;
      case Op::kScissor:
        [enc setScissorRect:static_cast<const ScissorOp*>(payload)->rect];
        break;
      case Op::kPipeline:
        [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)(
                                        static_cast<const PtrOp*>(payload)
                                            ->object)];
        break;
      case Op::kBlendColor: {
        const float* c = static_cast<const BlendColorOp*>(payload)->rgba;
        [enc setBlendColorRed:c[0] green:c[1] blue:c[2] alpha:c[3]];
      } break;
      case Op::kDepthStencil:
        [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)(
                                      static_cast<const PtrOp*>(payload)
                                          ->object)];
        break;
      case Op::kStencilRef: {
        const auto* p = static_cast<const StencilRefOp*>(payload);
        [enc setStencilFrontReferenceValue:p->front backReferenceValue:p->back];
      } break;
      case Op::kVisibility: {
        const auto* p = static_cast<const VisibilityOp*>(payload);
        [enc setVisibilityResultMode:MTLVisibilityResultMode(p->mode)
                              offset:NSUInteger(p->offset)];
      } break;
      case Op::kBuffer: {
        const auto* p = static_cast<const BufferOp*>(payload);
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)p->buffer;
        switch (p->stage) {
          case kStageVertex:
            [enc setVertexBuffer:b offset:p->offset atIndex:p->index];
            break;
          case kStageFragment:
            [enc setFragmentBuffer:b offset:p->offset atIndex:p->index];
            break;
          case kStageObject:
            [enc setObjectBuffer:b offset:p->offset atIndex:p->index];
            break;
          default:
            [enc setMeshBuffer:b offset:p->offset atIndex:p->index];
            break;
        }
      } break;
      case Op::kBufferOffset: {
        const auto* p = static_cast<const BufferOffsetOp*>(payload);
        switch (p->stage) {
          case kStageVertex:
            [enc setVertexBufferOffset:p->offset atIndex:p->index];
            break;
          case kStageFragment:
            [enc setFragmentBufferOffset:p->offset atIndex:p->index];
            break;
          case kStageObject:
            [enc setObjectBufferOffset:p->offset atIndex:p->index];
            break;
          default:
            [enc setMeshBufferOffset:p->offset atIndex:p->index];
            break;
        }
      } break;
      case Op::kBytes: {
        const auto* p = static_cast<const BytesOp*>(payload);
        const void* bytes = p + 1;
        switch (p->stage) {
          case kStageVertex:
            [enc setVertexBytes:bytes length:p->length atIndex:p->index];
            break;
          case kStageFragment:
            [enc setFragmentBytes:bytes length:p->length atIndex:p->index];
            break;
          case kStageObject:
            [enc setObjectBytes:bytes length:p->length atIndex:p->index];
            break;
          default:
            [enc setMeshBytes:bytes length:p->length atIndex:p->index];
            break;
        }
      } break;
      case Op::kUseResource: {
        const auto* p = static_cast<const UseResourceOp*>(payload);
        [enc useResource:(__bridge id<MTLResource>)p->resource
                   usage:MTLResourceUsage(p->usage)
                  stages:MTLRenderStages(p->stages)];
        if (p->retained) {
          CFRelease(p->resource);
        }
      } break;
      case Op::kDraw: {
        const auto* p = static_cast<const DrawOp*>(payload);
        [enc drawPrimitives:MTLPrimitiveType(p->type)
                vertexStart:p->start
                vertexCount:p->count];
      } break;
      case Op::kDrawIndexed: {
        const auto* p = static_cast<const DrawIndexedOp*>(payload);
        [enc drawIndexedPrimitives:MTLPrimitiveType(p->type)
                        indexCount:p->count
                         indexType:MTLIndexType(p->index_type)
                       indexBuffer:(__bridge id<MTLBuffer>)p->buffer
                 indexBufferOffset:p->offset
                     instanceCount:p->instance_count
                        baseVertex:p->base_vertex
                      baseInstance:p->base_instance];
      } break;
      case Op::kIRDraw: {
        const auto* p = static_cast<const DrawOp*>(payload);
        IRRuntimeDrawPrimitives(enc, MTLPrimitiveType(p->type), p->start,
                                p->count);
      } break;
      case Op::kIRDrawIndexed: {
        const auto* p = static_cast<const DrawIndexedOp*>(payload);
        IRRuntimeDrawIndexedPrimitives(
            enc, MTLPrimitiveType(p->type), p->count,
            MTLIndexType(p->index_type), (__bridge id<MTLBuffer>)p->buffer,
            p->offset, p->instance_count, p->base_vertex, p->base_instance);
      } break;
      case Op::kIRDrawIndexedGeometry: {
        const auto* p = static_cast<const IRDrawIndexedGeometryOp*>(payload);
        IRRuntimeDrawIndexedPrimitivesGeometryEmulation(
            enc, IRRuntimePrimitiveType(p->type), MTLIndexType(p->index_type),
            (__bridge id<MTLBuffer>)p->buffer, p->config, p->instance_count,
            p->index_count, p->start_index, p->base_vertex, p->base_instance);
      } break;
      case Op::kIRDrawGeometry: {
        const auto* p = static_cast<const IRDrawGeometryOp*>(payload);
        IRRuntimeDrawPrimitivesGeometryEmulation(
            enc, IRRuntimePrimitiveType(p->type), p->config, p->instance_count,
            p->vertex_count, p->base_vertex, p->base_instance);
      } break;
    }
    pos += header->size;
  }
  replay_encoder_ = enc;
}

// --- Recording. ---

void MetalEncodeStream::SetEncoder(id<MTLRenderCommandEncoder> encoder) {
  if (mode_ == Mode::kDirect) {
    direct_encoder_ = encoder;
    return;
  }
  Emit<PtrOp>(Op::kSetEncoder)->object = (__bridge void*)encoder;
}

void MetalEncodeStream::SetViewport(const MTLViewport& viewport) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setViewport:viewport];
    return;
  }
  Emit<ViewportOp>(Op::kViewport)->viewport = viewport;
}

void MetalEncodeStream::SetDepthClipMode(MTLDepthClipMode mode) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setDepthClipMode:mode];
    return;
  }
  Emit<U32Op>(Op::kDepthClipMode)->value = uint32_t(mode);
}

void MetalEncodeStream::SetCullMode(MTLCullMode mode) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setCullMode:mode];
    return;
  }
  Emit<U32Op>(Op::kCullMode)->value = uint32_t(mode);
}

void MetalEncodeStream::SetFrontFacingWinding(MTLWinding winding) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setFrontFacingWinding:winding];
    return;
  }
  Emit<U32Op>(Op::kWinding)->value = uint32_t(winding);
}

void MetalEncodeStream::SetTriangleFillMode(MTLTriangleFillMode mode) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setTriangleFillMode:mode];
    return;
  }
  Emit<U32Op>(Op::kFillMode)->value = uint32_t(mode);
}

void MetalEncodeStream::SetDepthBias(float constant, float slope, float clamp) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setDepthBias:constant slopeScale:slope clamp:clamp];
    return;
  }
  auto* p = Emit<DepthBiasOp>(Op::kDepthBias);
  p->constant = constant;
  p->slope = slope;
  p->clamp = clamp;
}

void MetalEncodeStream::SetScissorRect(const MTLScissorRect& rect) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setScissorRect:rect];
    return;
  }
  Emit<ScissorOp>(Op::kScissor)->rect = rect;
}

void MetalEncodeStream::SetRenderPipelineState(
    id<MTLRenderPipelineState> state) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setRenderPipelineState:state];
    return;
  }
  Emit<PtrOp>(Op::kPipeline)->object = (__bridge void*)state;
}

void MetalEncodeStream::SetBlendColor(float r, float g, float b, float a) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setBlendColorRed:r green:g blue:b alpha:a];
    return;
  }
  float* c = Emit<BlendColorOp>(Op::kBlendColor)->rgba;
  c[0] = r;
  c[1] = g;
  c[2] = b;
  c[3] = a;
}

void MetalEncodeStream::SetDepthStencilState(id<MTLDepthStencilState> state) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setDepthStencilState:state];
    return;
  }
  Emit<PtrOp>(Op::kDepthStencil)->object = (__bridge void*)state;
}

void MetalEncodeStream::SetStencilReferenceValues(uint32_t front,
                                                  uint32_t back) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setStencilFrontReferenceValue:front
                                backReferenceValue:back];
    return;
  }
  auto* p = Emit<StencilRefOp>(Op::kStencilRef);
  p->front = front;
  p->back = back;
}

void MetalEncodeStream::SetVisibilityResultMode(MTLVisibilityResultMode mode,
                                                NSUInteger offset) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ setVisibilityResultMode:mode offset:offset];
    return;
  }
  auto* p = Emit<VisibilityOp>(Op::kVisibility);
  p->mode = uint64_t(mode);
  p->offset = uint64_t(offset);
}

void MetalEncodeStream::SetBuffer(uint32_t stage, id<MTLBuffer> buffer,
                                  NSUInteger offset, NSUInteger index) {
  if (mode_ == Mode::kDirect) {
    switch (stage) {
      case kStageVertex:
        [direct_encoder_ setVertexBuffer:buffer offset:offset atIndex:index];
        break;
      case kStageFragment:
        [direct_encoder_ setFragmentBuffer:buffer offset:offset atIndex:index];
        break;
      case kStageObject:
        [direct_encoder_ setObjectBuffer:buffer offset:offset atIndex:index];
        break;
      default:
        [direct_encoder_ setMeshBuffer:buffer offset:offset atIndex:index];
        break;
    }
    return;
  }
  auto* p = Emit<BufferOp>(Op::kBuffer);
  p->buffer = (__bridge void*)buffer;
  p->offset = uint64_t(offset);
  p->index = uint32_t(index);
  p->stage = stage;
}

void MetalEncodeStream::SetBufferOffset(uint32_t stage, NSUInteger offset,
                                        NSUInteger index) {
  if (mode_ == Mode::kDirect) {
    switch (stage) {
      case kStageVertex:
        [direct_encoder_ setVertexBufferOffset:offset atIndex:index];
        break;
      case kStageFragment:
        [direct_encoder_ setFragmentBufferOffset:offset atIndex:index];
        break;
      case kStageObject:
        [direct_encoder_ setObjectBufferOffset:offset atIndex:index];
        break;
      default:
        [direct_encoder_ setMeshBufferOffset:offset atIndex:index];
        break;
    }
    return;
  }
  auto* p = Emit<BufferOffsetOp>(Op::kBufferOffset);
  p->offset = uint64_t(offset);
  p->index = uint32_t(index);
  p->stage = stage;
}

void MetalEncodeStream::SetBytes(uint32_t stage, const void* bytes,
                                 NSUInteger length, NSUInteger index) {
  if (mode_ == Mode::kDirect) {
    switch (stage) {
      case kStageVertex:
        [direct_encoder_ setVertexBytes:bytes length:length atIndex:index];
        break;
      case kStageFragment:
        [direct_encoder_ setFragmentBytes:bytes length:length atIndex:index];
        break;
      case kStageObject:
        [direct_encoder_ setObjectBytes:bytes length:length atIndex:index];
        break;
      default:
        [direct_encoder_ setMeshBytes:bytes length:length atIndex:index];
        break;
    }
    return;
  }
  auto* p = static_cast<BytesOp*>(
      Reserve(Op::kBytes, uint32_t(sizeof(BytesOp) + length)));
  p->length = uint32_t(length);
  p->index = uint32_t(index);
  p->stage = stage;
  std::memcpy(p + 1, bytes, length);
}

void MetalEncodeStream::UseResource(id<MTLResource> resource,
                                    MTLResourceUsage usage,
                                    MTLRenderStages stages, bool retain) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ useResource:resource usage:usage stages:stages];
    return;
  }
  auto* p = Emit<UseResourceOp>(Op::kUseResource);
  p->resource = (__bridge void*)resource;
  p->usage = uint64_t(usage);
  p->stages = uint64_t(stages);
  p->retained = retain ? 1 : 0;
  if (retain) {
    CFRetain((__bridge CFTypeRef)resource);
  }
}

void MetalEncodeStream::DrawPrimitives(MTLPrimitiveType type,
                                       uint64_t vertex_start,
                                       uint64_t vertex_count) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ drawPrimitives:type
                        vertexStart:vertex_start
                        vertexCount:vertex_count];
    return;
  }
  auto* p = Emit<DrawOp>(Op::kDraw);
  p->type = uint64_t(type);
  p->start = vertex_start;
  p->count = vertex_count;
}

void MetalEncodeStream::DrawIndexedPrimitives(
    MTLPrimitiveType type, uint64_t index_count, MTLIndexType index_type,
    id<MTLBuffer> index_buffer, uint64_t index_offset, uint64_t instance_count,
    int64_t base_vertex, uint64_t base_instance) {
  if (mode_ == Mode::kDirect) {
    [direct_encoder_ drawIndexedPrimitives:type
                                indexCount:index_count
                                 indexType:index_type
                               indexBuffer:index_buffer
                         indexBufferOffset:index_offset
                             instanceCount:instance_count
                                baseVertex:base_vertex
                              baseInstance:base_instance];
    return;
  }
  auto* p = Emit<DrawIndexedOp>(Op::kDrawIndexed);
  p->type = uint64_t(type);
  p->count = index_count;
  p->index_type = uint64_t(index_type);
  p->buffer = (__bridge void*)index_buffer;
  p->offset = index_offset;
  p->instance_count = instance_count;
  p->base_vertex = base_vertex;
  p->base_instance = base_instance;
}

void MetalEncodeStream::IRDrawPrimitives(MTLPrimitiveType type,
                                         uint64_t vertex_start,
                                         uint64_t vertex_count) {
  if (mode_ == Mode::kDirect) {
    IRRuntimeDrawPrimitives(direct_encoder_, type, vertex_start, vertex_count);
    return;
  }
  auto* p = Emit<DrawOp>(Op::kIRDraw);
  p->type = uint64_t(type);
  p->start = vertex_start;
  p->count = vertex_count;
}

void MetalEncodeStream::IRDrawIndexedPrimitives(
    MTLPrimitiveType type, uint64_t index_count, MTLIndexType index_type,
    id<MTLBuffer> index_buffer, uint64_t index_offset, uint64_t instance_count,
    int64_t base_vertex, uint64_t base_instance) {
  if (mode_ == Mode::kDirect) {
    IRRuntimeDrawIndexedPrimitives(direct_encoder_, type, index_count,
                                   index_type, index_buffer, index_offset,
                                   instance_count, base_vertex, base_instance);
    return;
  }
  auto* p = Emit<DrawIndexedOp>(Op::kIRDrawIndexed);
  p->type = uint64_t(type);
  p->count = index_count;
  p->index_type = uint64_t(index_type);
  p->buffer = (__bridge void*)index_buffer;
  p->offset = index_offset;
  p->instance_count = instance_count;
  p->base_vertex = base_vertex;
  p->base_instance = base_instance;
}

void MetalEncodeStream::IRDrawIndexedGeometryEmulation(
    IRRuntimePrimitiveType type, MTLIndexType index_type,
    id<MTLBuffer> index_buffer, const IRRuntimeGeometryPipelineConfig& config,
    uint32_t instance_count, uint32_t index_count, uint32_t start_index,
    int base_vertex, uint32_t base_instance) {
  if (mode_ == Mode::kDirect) {
    IRRuntimeDrawIndexedPrimitivesGeometryEmulation(
        direct_encoder_, type, index_type, index_buffer, config,
        instance_count, index_count, start_index, base_vertex, base_instance);
    return;
  }
  auto* p = Emit<IRDrawIndexedGeometryOp>(Op::kIRDrawIndexedGeometry);
  p->config = config;
  p->buffer = (__bridge void*)index_buffer;
  p->type = uint32_t(type);
  p->index_type = uint32_t(index_type);
  p->instance_count = instance_count;
  p->index_count = index_count;
  p->start_index = start_index;
  p->base_vertex = base_vertex;
  p->base_instance = base_instance;
}

void MetalEncodeStream::IRDrawGeometryEmulation(
    IRRuntimePrimitiveType type, const IRRuntimeGeometryPipelineConfig& config,
    uint32_t instance_count, uint32_t vertex_count, uint32_t base_vertex,
    uint32_t base_instance) {
  if (mode_ == Mode::kDirect) {
    IRRuntimeDrawPrimitivesGeometryEmulation(direct_encoder_, type, config,
                                             instance_count, vertex_count,
                                             base_vertex, base_instance);
    return;
  }
  auto* p = Emit<IRDrawGeometryOp>(Op::kIRDrawGeometry);
  p->config = config;
  p->type = uint32_t(type);
  p->instance_count = instance_count;
  p->vertex_count = vertex_count;
  p->base_vertex = base_vertex;
  p->base_instance = base_instance;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
