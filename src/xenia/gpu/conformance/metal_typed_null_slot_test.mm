/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 1 (Metal device): the typed texture-handle / non-selection mechanism
// behind OpenSpec `align-metal-texture-binding-signedness` task 3.2 -- the
// per-dimension null-slot safety proof.
//
// WHAT THE PRODUCTION QUESTION IS
// --------------------------------
// The signedness gate would write the EXISTING null view-heap slot for a sign
// interpretation the translated shader cannot select. Slot 0 (and every
// unwritten slot) holds one 1x1 `MTLTextureType2DArray` `RGBA8Unorm` texture
// filled mid-gray (128,128,128,255) -- `metal_command_processor.mm:919-946`,
// heap prefill `951-976` (fill loop `965-969`), handed to the cache by
// `SetNullTexture` at `1006`, made resident by `useResource` at `4116`.
//
// The declared MSL texture type a binding is read through is NOT uniform. The
// DXBC translator is used in bindless mode, so each declared binding dimension
// lands in a different bindless SRV register space
// (`dxbc_shader_translator_fetch.cc:1871-1884` and `1379-1392`), and each space
// is declared with its own resource dimension
// (`dxbc_shader_translator.cc:3512-3525`):
//
//   declared dimension   DXBC SRV space               DXBC resource dim
//   k2D (and k1D)        kBindlessTextures2DArray(1)  Texture2DArray
//   k3DOrStacked         kBindlessTextures3D(2)       Texture3D
//   kCube                kBindlessTexturesCube(3)     TextureCube
//
// MSC is compiled with `IRCompatibilityFlagForceTextureArray`
// (`metal_shader_converter.cc:244-248`), which arrays the 2D and cube shapes
// but not the 3D one, so the MSL exposure shapes are:
//
//   k2D           -> texture2d_array<float>
//   k3DOrStacked  -> texture3d<float>
//   kCube         -> texturecube_array<float>
//
// and all three spaces are bound to the SAME view heap buffer
// (`metal_binding_layout.h:384-392`, `metal_command_processor.mm:3971`), so
// slot 0 -- one 2D-array texture -- is reachable through all three declared
// types. Only the k2D shape is type-matched to it.
//
// WHAT THIS TEST PROVES, AND WHAT IT DOES NOT
// -------------------------------------------
// Each case declares the real MSL texture type in a typed argument-buffer
// field, encodes a texture into that field with `MTLArgumentEncoder`, and
// dispatches with the access guarded by a RUNTIME control buffer so the branch
// and the declaration cannot be constant-folded away.
//
//   * type-MATCHED (2D-array null in a texture2d_array<float> field): the null
//     is both bound AND sampled, so the read itself is exercised and must
//     return the null's exact mid-gray texel.
//   * type-MISMATCHED (2D-array null in a texture3d<float> or
//     texturecube_array<float> field): the null is bound and the access branch
//     is live but NOT taken -- the production non-selection property. This test
//     deliberately never samples a mismatched handle; doing so is the undefined
//     access the spec asks about, not a demonstration of its absence.
//
// Positive and negative controls run the SAME pipeline and the SAME argument
// buffer field with correctly typed resources of two DIFFERENT known colours,
// so a branch that had been optimised away, or an output that were not
// sample-derived, fails observably instead of passing vacuously.
//
// EVIDENCE BOUNDARY (must be carried into any verdict drawn from this test):
// production writes texture handles into an `IRDescriptorTableEntry` heap via
// `IRDescriptorTableSetTexture`, which this test does NOT use. This test proves
// Metal typed-handle behaviour and the non-selection property for the same
// DECLARED texture types and the same null resource type; it does not prove
// byte-for-byte MSC descriptor-table plumbing.

#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// The sentinel a non-accessing dispatch must write. Outside [0, 1], so no
// UNORM sample can forge it.
constexpr float kNotAccessed = -7.0f;

// The production null texture's texel (`metal_command_processor.mm:941`).
constexpr uint8_t kNullGray[4] = {128, 128, 128, 255};

// Two distinct control colours, so "sample-derived" is observable rather than
// asserted.
constexpr uint8_t kControlA[4] = {64, 128, 192, 255};
constexpr uint8_t kControlB[4] = {255, 0, 32, 255};

constexpr float kUnormTolerance = 1.0f / 512.0f;

static const char* kTypedNullProbeMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ProbeControl {
  uint access;
};

// One typed texture field per declared exposure shape. The [[id(0)]] field is
// what MTLArgumentEncoder writes into.
struct Args2DArray {
  texture2d_array<float> tex [[id(0)]];
};
struct Args3D {
  texture3d<float> tex [[id(0)]];
};
struct ArgsCubeArray {
  texturecube_array<float> tex [[id(0)]];
};

constant float4 kNotAccessed = float4(-7.0f, -7.0f, -7.0f, -7.0f);

kernel void probe_texture2d_array(
    device const Args2DArray& args [[buffer(0)]],
    device const ProbeControl& control [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  float4 v = kNotAccessed;
  if (control.access != 0u) {
    constexpr sampler s(coord::normalized, filter::nearest, mip_filter::none,
                        address::clamp_to_edge);
    v = args.tex.sample(s, float2(0.5f, 0.5f), 0);
  }
  out[tid] = v;
}

kernel void probe_texture3d(
    device const Args3D& args [[buffer(0)]],
    device const ProbeControl& control [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  float4 v = kNotAccessed;
  if (control.access != 0u) {
    constexpr sampler s(coord::normalized, filter::nearest, mip_filter::none,
                        address::clamp_to_edge);
    v = args.tex.sample(s, float3(0.5f, 0.5f, 0.5f));
  }
  out[tid] = v;
}

kernel void probe_texturecube_array(
    device const ArgsCubeArray& args [[buffer(0)]],
    device const ProbeControl& control [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  float4 v = kNotAccessed;
  if (control.access != 0u) {
    constexpr sampler s(coord::normalized, filter::nearest, mip_filter::none,
                        address::clamp_to_edge);
    v = args.tex.sample(s, float3(0.0f, 0.0f, 1.0f), 0);
  }
  out[tid] = v;
}
)MSL";

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

// EXACTLY the production null texture shape: 1x1 MTLTextureType2DArray
// RGBA8Unorm, one array slice, one mip, shader-read, shared storage, mid-gray.
// Mirrors `metal_command_processor.mm:926-945`.
id<MTLTexture> MakeProductionShapedNull(id<MTLDevice> device) {
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType2DArray;
  desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  desc.width = 1;
  desc.height = 1;
  desc.arrayLength = 1;
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    return nil;
  }
  [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
             mipmapLevel:0
                   slice:0
               withBytes:kNullGray
             bytesPerRow:4
           bytesPerImage:0];
  return texture;
}

id<MTLTexture> Make2DArray(id<MTLDevice> device, const uint8_t rgba[4]) {
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType2DArray;
  desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  desc.width = 1;
  desc.height = 1;
  desc.arrayLength = 1;
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (texture) {
    [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
               mipmapLevel:0
                     slice:0
                 withBytes:rgba
               bytesPerRow:4
             bytesPerImage:0];
  }
  return texture;
}

id<MTLTexture> Make3D(id<MTLDevice> device, const uint8_t rgba[4]) {
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType3D;
  desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  desc.width = 1;
  desc.height = 1;
  desc.depth = 1;
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (texture) {
    [texture replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
               mipmapLevel:0
                     slice:0
                 withBytes:rgba
               bytesPerRow:4
             bytesPerImage:4];
  }
  return texture;
}

id<MTLTexture> MakeCubeArray(id<MTLDevice> device, const uint8_t rgba[4]) {
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureTypeCubeArray;
  desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  desc.width = 1;
  desc.height = 1;
  desc.arrayLength = 1;  // cube count; 6 slices.
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (texture) {
    for (uint32_t face = 0; face < 6; ++face) {
      [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                 mipmapLevel:0
                       slice:face
                   withBytes:rgba
                 bytesPerRow:4
               bytesPerImage:0];
    }
  }
  return texture;
}

// ---------------------------------------------------------------------------
// Probe execution
// ---------------------------------------------------------------------------

struct ProbeShape {
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLArgumentEncoder> argument_encoder = nil;
  id<MTLBuffer> argument_buffer = nil;
};

struct RunOutcome {
  // True only when nothing raised, the command buffer reached Completed, and
  // `commandBuffer.error` was nil.
  bool clean = false;
  // Populated with evidence whenever `clean` is false.
  std::string failure;
  float value[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

// Wraps every validation-prone Objective-C call so a Metal API-validation
// exception surfaces as test evidence instead of terminating the suite.
RunOutcome RunProbe(id<MTLCommandQueue> queue, const ProbeShape& shape,
                    id<MTLTexture> texture, uint32_t access) {
  RunOutcome outcome;
  @autoreleasepool {
    @try {
      id<MTLDevice> device = queue.device;
      [shape.argument_encoder setArgumentBuffer:shape.argument_buffer offset:0];
      [shape.argument_encoder setTexture:texture atIndex:0];

      id<MTLBuffer> control_buffer =
          [device newBufferWithBytes:&access
                              length:sizeof(access)
                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> out_buffer =
          [device newBufferWithLength:sizeof(float) * 4
                              options:MTLResourceStorageModeShared];
      if (!control_buffer || !out_buffer) {
        outcome.failure = "buffer allocation failed";
        return outcome;
      }
      std::memset([out_buffer contents], 0, sizeof(float) * 4);

      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:shape.pipeline];
      [encoder setBuffer:shape.argument_buffer offset:0 atIndex:0];
      [encoder setBuffer:control_buffer offset:0 atIndex:1];
      [encoder setBuffer:out_buffer offset:0 atIndex:2];
      // Argument-buffer residency: the handle is reachable from the shader
      // whether or not the access branch is taken, exactly as production makes
      // the null texture resident (`metal_command_processor.mm:4116`).
      [encoder useResource:texture usage:MTLResourceUsageRead];
      [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];

      if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        outcome.failure = "command buffer status " +
                          std::to_string(int(command_buffer.status));
        return outcome;
      }
      if (command_buffer.error) {
        outcome.failure =
            std::string("command buffer error: ") +
            [[command_buffer.error localizedDescription] UTF8String];
        return outcome;
      }
      std::memcpy(outcome.value, [out_buffer contents], sizeof(float) * 4);
      outcome.clean = true;
    } @catch (NSException* exception) {
      outcome.failure = std::string("ObjC exception ") +
                        [[exception name] UTF8String] + ": " +
                        [[exception reason] UTF8String];
    }
  }
  return outcome;
}

bool IsNotAccessedSentinel(const RunOutcome& outcome) {
  for (uint32_t i = 0; i < 4; ++i) {
    if (outcome.value[i] != kNotAccessed) {
      return false;
    }
  }
  return true;
}

bool MatchesUnorm(const RunOutcome& outcome, const uint8_t rgba[4]) {
  for (uint32_t i = 0; i < 4; ++i) {
    float expected = float(rgba[i]) / 255.0f;
    if (std::abs(outcome.value[i] - expected) > kUnormTolerance) {
      return false;
    }
  }
  return true;
}

std::string Describe(const RunOutcome& outcome) {
  if (!outcome.clean) {
    return "NOT CLEAN: " + outcome.failure;
  }
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "clean, value (%.5f, %.5f, %.5f, %.5f)",
                outcome.value[0], outcome.value[1], outcome.value[2],
                outcome.value[3]);
  return buffer;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct TypedNullFixture {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::string init_failure;

  bool Init() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return false;
    }
    queue = [device newCommandQueue];
    if (!queue) {
      init_failure = "no command queue";
      return false;
    }
    NSError* error = nil;
    library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:kTypedNullProbeMsl]
                     options:nil
                       error:&error];
    if (!library) {
      init_failure = std::string("MSL compile failed: ") +
                     (error ? [[error localizedDescription] UTF8String]
                            : "unknown");
      return false;
    }
    return true;
  }

  // Builds the pipeline plus the TYPED argument buffer for one exposure shape.
  bool MakeShape(const char* function_name, ProbeShape* out_shape,
                 std::string* failure) {
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:
                                                   function_name]];
    if (!function) {
      *failure = std::string("missing function ") + function_name;
      return false;
    }
    NSError* error = nil;
    out_shape->pipeline = [device newComputePipelineStateWithFunction:function
                                                                error:&error];
    if (!out_shape->pipeline) {
      *failure = std::string("pipeline failed for ") + function_name + ": " +
                 (error ? [[error localizedDescription] UTF8String] : "unknown");
      return false;
    }
    out_shape->argument_encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!out_shape->argument_encoder) {
      *failure = std::string("no argument encoder for ") + function_name;
      return false;
    }
    out_shape->argument_buffer = [device
        newBufferWithLength:out_shape->argument_encoder.encodedLength
                    options:MTLResourceStorageModeShared];
    if (!out_shape->argument_buffer) {
      *failure = std::string("no argument buffer for ") + function_name;
      return false;
    }
    return true;
  }
};

// Runs the shared control ladder for one exposure shape:
//   * access=0 with `bound` -> the sentinel, cleanly (the non-selection case);
//   * access=1 with `control_a` -> colour A (declaration and branch are live);
//   * access=1 with `control_b` -> colour B, different from A (the output is
//     sample-derived; a constant-folded or hoisted access fails here).
void CheckControlLadder(TypedNullFixture& fixture, const ProbeShape& shape,
                        id<MTLTexture> bound, id<MTLTexture> control_a,
                        id<MTLTexture> control_b) {
  RunOutcome not_selected = RunProbe(fixture.queue, shape, bound, 0);
  INFO("access=0, bound handle: " << Describe(not_selected));
  REQUIRE(not_selected.clean);
  REQUIRE(IsNotAccessedSentinel(not_selected));

  RunOutcome positive = RunProbe(fixture.queue, shape, control_a, 1);
  INFO("access=1, correctly typed control A: " << Describe(positive));
  REQUIRE(positive.clean);
  REQUIRE_FALSE(IsNotAccessedSentinel(positive));
  REQUIRE(MatchesUnorm(positive, kControlA));

  RunOutcome negative = RunProbe(fixture.queue, shape, control_b, 1);
  INFO("access=1, correctly typed control B: " << Describe(negative));
  REQUIRE(negative.clean);
  REQUIRE(MatchesUnorm(negative, kControlB));
  // Deliberately observable: B must not equal A, so an output that were not
  // genuinely sample-derived cannot satisfy both.
  REQUIRE(negative.value[0] != positive.value[0]);
}

}  // namespace

// ---------------------------------------------------------------------------
// k2D / stacked -- texture2d_array<float>, TYPE-MATCHED to the null
// ---------------------------------------------------------------------------

TEST_CASE(
    "Metal typed null slot is well-defined for the 2D-array exposure shape",
    "[l1][metal][typed-null][texture-signedness]") {
  TypedNullFixture fixture;
  if (!fixture.Init()) {
    if (!fixture.device) {
      SUCCEED("no device");
      return;
    }
    FAIL(fixture.init_failure);
  }

  ProbeShape shape;
  std::string failure;
  if (!fixture.MakeShape("probe_texture2d_array", &shape, &failure)) {
    FAIL(failure);
  }
  // Non-vacuity: the argument buffer really carries a texture field.
  REQUIRE(shape.argument_encoder.encodedLength >= sizeof(uint64_t));

  id<MTLTexture> null_texture = MakeProductionShapedNull(fixture.device);
  REQUIRE(null_texture != nil);
  id<MTLTexture> control_a = Make2DArray(fixture.device, kControlA);
  id<MTLTexture> control_b = Make2DArray(fixture.device, kControlB);
  REQUIRE(control_a != nil);
  REQUIRE(control_b != nil);

  CheckControlLadder(fixture, shape, null_texture, control_a, control_b);

  // The k2D exposure shape is type-matched to the null, so the READ itself is
  // exercised, not merely the non-selection property: sampling slot 0's
  // resource through texture2d_array<float> must return its exact mid-gray
  // texel with no fault and no validation error.
  RunOutcome read_null = RunProbe(fixture.queue, shape, null_texture, 1);
  INFO("access=1, production-shaped null read through texture2d_array: "
       << Describe(read_null));
  REQUIRE(read_null.clean);
  REQUIRE_FALSE(IsNotAccessedSentinel(read_null));
  REQUIRE(MatchesUnorm(read_null, kNullGray));
}

// ---------------------------------------------------------------------------
// k3DOrStacked -- texture3d<float>, MISMATCHED against the 2D-array null
// ---------------------------------------------------------------------------

TEST_CASE(
    "Metal typed null slot is bindable but unread for the 3D exposure shape",
    "[l1][metal][typed-null][texture-signedness]") {
  TypedNullFixture fixture;
  if (!fixture.Init()) {
    if (!fixture.device) {
      SUCCEED("no device");
      return;
    }
    FAIL(fixture.init_failure);
  }

  ProbeShape shape;
  std::string failure;
  if (!fixture.MakeShape("probe_texture3d", &shape, &failure)) {
    FAIL(failure);
  }
  REQUIRE(shape.argument_encoder.encodedLength >= sizeof(uint64_t));

  id<MTLTexture> null_texture = MakeProductionShapedNull(fixture.device);
  REQUIRE(null_texture != nil);
  id<MTLTexture> control_a = Make3D(fixture.device, kControlA);
  id<MTLTexture> control_b = Make3D(fixture.device, kControlB);
  REQUIRE(control_a != nil);
  REQUIRE(control_b != nil);

  // The 2D-array null is encoded into a texture3d<float> field and made
  // resident with the access branch live but untaken. This is the
  // NON-SELECTION property only; the mismatched handle is never sampled, so
  // read well-definedness for this shape remains UNPROVEN by design.
  CheckControlLadder(fixture, shape, null_texture, control_a, control_b);
}

// ---------------------------------------------------------------------------
// kCube -- texturecube_array<float>, MISMATCHED against the 2D-array null
// ---------------------------------------------------------------------------

TEST_CASE(
    "Metal typed null slot outcome is recorded for the cube exposure shape",
    "[l1][metal][typed-null][texture-signedness]") {
  TypedNullFixture fixture;
  if (!fixture.Init()) {
    if (!fixture.device) {
      SUCCEED("no device");
      return;
    }
    FAIL(fixture.init_failure);
  }

  ProbeShape shape;
  std::string failure;
  if (!fixture.MakeShape("probe_texturecube_array", &shape, &failure)) {
    FAIL(failure);
  }
  REQUIRE(shape.argument_encoder.encodedLength >= sizeof(uint64_t));

  id<MTLTexture> null_texture = MakeProductionShapedNull(fixture.device);
  REQUIRE(null_texture != nil);
  id<MTLTexture> control_a = MakeCubeArray(fixture.device, kControlA);
  id<MTLTexture> control_b = MakeCubeArray(fixture.device, kControlB);
  REQUIRE(control_a != nil);
  REQUIRE(control_b != nil);

  // The declaration and access branch must be live for this shape too --
  // asserted unconditionally, because a dead branch would make any conclusion
  // about the mismatched handle vacuous.
  RunOutcome positive = RunProbe(fixture.queue, shape, control_a, 1);
  INFO("access=1, correctly typed cube-array control A: " << Describe(positive));
  REQUIRE(positive.clean);
  REQUIRE(MatchesUnorm(positive, kControlA));
  RunOutcome negative = RunProbe(fixture.queue, shape, control_b, 1);
  INFO("access=1, correctly typed cube-array control B: " << Describe(negative));
  REQUIRE(negative.clean);
  REQUIRE(MatchesUnorm(negative, kControlB));
  REQUIRE(negative.value[0] != positive.value[0]);

  // Encoding a 2D-array handle into a texturecube_array<float> field may be
  // rejected outright under Metal API validation. That is a legitimate
  // OUTCOME of this proof (cube excluded from the admitted dimension set), not
  // a suite failure -- so it is recorded, not asserted away. Either way the
  // verdict for cube is UNPROVEN: the mismatched handle is never sampled here.
  RunOutcome not_selected = RunProbe(fixture.queue, shape, null_texture, 0);
  if (!not_selected.clean) {
    INFO(
        "cube exposure shape: the 2D-array null could not be bound/dispatched "
        "under validation -- "
        << not_selected.failure
        << ". Cube is EXCLUDED from the admitted dimension set.");
    SUCCEED("cube mismatched-null binding rejected; recorded as UNPROVEN");
    return;
  }
  INFO("access=0, 2D-array null bound to texturecube_array field: "
       << Describe(not_selected));
  REQUIRE(IsNotAccessedSentinel(not_selected));
  INFO(
      "cube exposure shape: binding + untaken access of the 2D-array null was "
      "clean, but the mismatched handle was never sampled. Cube remains "
      "UNPROVEN for read well-definedness and is EXCLUDED from the admitted "
      "dimension set.");
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
