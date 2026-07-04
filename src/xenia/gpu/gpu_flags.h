/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GPU_FLAGS_H_
#define XENIA_GPU_GPU_FLAGS_H_
#include "xenia/base/cvar.h"

DECLARE_path(trace_gpu_prefix);
DECLARE_bool(trace_gpu_stream);

DECLARE_path(dump_shaders);

DECLARE_bool(vsync);

DECLARE_uint64(framerate_limit);

DECLARE_bool(gpu_allow_invalid_fetch_constants);

DECLARE_bool(non_seamless_cube_map);

DECLARE_bool(half_pixel_offset);

DECLARE_string(occlusion_query);

DECLARE_int32(occlusion_query_fake_lower_threshold);

DECLARE_int32(occlusion_query_fake_upper_threshold);

DECLARE_int32(occlusion_query_querybatch_range);

DECLARE_double(occlusion_query_saturation);

DECLARE_int32(anisotropic_override);

DECLARE_bool(disassemble_pm4);

DECLARE_bool(no_discard_stencil_in_transfer_pipelines);

DECLARE_bool(async_shader_compilation);

DECLARE_bool(gpu_3d_to_2d_texture);

DECLARE_bool(ac6_ground_fix);

DECLARE_bool(depth_float24_sample_z_varying);

DECLARE_bool(depth_float24_sample_z_from_derivatives);

DECLARE_bool(texture_alpha_test_lod_snap);

DECLARE_bool(texture_decompress_bc);

DECLARE_bool(texture_trilinear_ramp_match_amd);

DECLARE_double(texture_mip_lod_bias_adjust);

DECLARE_bool(interpolators_manual_perspective);

DECLARE_bool(texture_computed_lod_in_shader);

DECLARE_bool(texture_lod_covered_lane_inputs);

DECLARE_int32(texture_lod_force_level);

DECLARE_bool(texture_max_lod_clamp_probe);

DECLARE_bool(force_per_sample_shading_probe);

DECLARE_int32(vertex_position_quantize_bits);

#define XE_GPU_FINE_GRAINED_DRAW_SCOPES 1

#endif  // XENIA_GPU_GPU_FLAGS_H_
