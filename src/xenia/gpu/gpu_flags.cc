/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/gpu_flags.h"

DEFINE_path(trace_gpu_prefix, "scratch/gpu/",
            "Prefix path for GPU trace files.", "GPU");
DEFINE_bool(trace_gpu_stream, false, "Trace all GPU packets.", "GPU");

DEFINE_path(
    dump_shaders, "",
    "For shader debugging, path to dump GPU shaders to as they are compiled.",
    "GPU");

DEFINE_bool(vsync, true, "Enable VSYNC.", "GPU");

DEFINE_uint64(framerate_limit, 0,
              "Maximum frames per second. 0 = Unlimited frames.\n"
              "Defaults to 60, when set to 0, and VSYNC is enabled.",
              "GPU");
UPDATE_from_uint64(framerate_limit, 2024, 8, 31, 20, 60);

DEFINE_bool(
    gpu_allow_invalid_fetch_constants, true,
    "Allow texture and vertex fetch constants with invalid type - generally "
    "unsafe because the constant may contain completely invalid values, but "
    "may be used to bypass fetch constant type errors in certain games until "
    "the real reason why they're invalid is found.",
    "GPU");
DEFINE_bool(
    gpu_allow_invalid_upload_range, false,
    "Allows games to read data from pages that are marked as no access.",
    "GPU");

DEFINE_bool(
    non_seamless_cube_map, true,
    "Disable filtering between cube map faces near edges where possible "
    "(Vulkan with VK_EXT_non_seamless_cube_map) to reproduce the Direct3D 9 "
    "behavior.",
    "GPU");

// Extremely bright screen borders in 4D5307E6.
// Reading between texels with half-pixel offset in 58410954.
DEFINE_bool(
    half_pixel_offset, true,
    "Enable support of vertex half-pixel offset (D3D9 PA_SU_VTX_CNTL "
    "PIX_CENTER). Generally games are aware of the half-pixel offset, and "
    "having this enabled is the correct behavior (disabling this may "
    "significantly break post-processing in some games), but in certain games "
    "it might have been ignored, resulting in slight blurriness of UI "
    "textures, for instance, when they are read between texels rather than "
    "at texel centers, or the leftmost/topmost pixels may not be fully covered "
    "when MSAA is used with fullscreen passes.",
    "GPU");

DEFINE_int32(occlusion_query_fake_lower_threshold, 80,
             "Lower end of the fake sample count value written on "
             "EVENT_WRITE_ZPD when real occlusion queries are disabled.\n"
             "-1 writes nothing, resulting in some games that sit and hang.\n"
             "0 means the fake result stays fully occluded.",
             "GPU");
DEFINE_int32(occlusion_query_fake_upper_threshold, 100,
             "Upper end of the fake sample count value written on "
             "EVENT_WRITE_ZPD when real occlusion queries are disabled.\n"
             "Keep this higher than occlusion_query_fake_lower_threshold.\n"
             "Ignored if occlusion_query_fake_lower_threshold is -1.",
             "GPU");
DEFINE_int32(occlusion_query_querybatch_range, 0,
             "Range of fake sample count values to walk for titles using the "
             "D3D QueryBatch standard before wrapping back to "
             "occlusion_query_fake_lower_threshold.\n"
             "This shouldn't be changed from the default value of 0 (disabled) "
             "unless necessary for a specific title.",
             "GPU");
DEFINE_double(
    occlusion_query_saturation, 1.0,
    "Compress higher occlusion query sample counts before guest writeback.\n"
    "This can be useful if effects such as lens flares appear too strong.\n"
    "1.0 = default behavior\n"
    "0.0 = collapse all nonzero sample counts to 1\n"
    "Values around 0.90 are a good starting point for subtle tuning.",
    "GPU");

DEFINE_int32(anisotropic_override, -1,
             "Forces anisotropic filtering (AF) for eligible textures.\n"
             "Higher values keep textures sharper at oblique angles at the "
             "cost of GPU bandwidth, though most GPUs handle up to 16x fine.\n"
             "In rare cases, forcing AF can introduce visual artifacts.\n"
             " -1 = No override\n"
             "  0 = Disable anisotropic filtering\n"
             "  1 = Force 1x anisotropic filtering\n"
             "  2 = Force 2x anisotropic filtering\n"
             "  3 = Force 4x anisotropic filtering\n"
             "  4 = Force 8x anisotropic filtering\n"
             "  5 = Force 16x anisotropic filtering",
             "GPU");

DEFINE_bool(no_discard_stencil_in_transfer_pipelines, false,
            "Skip stencil bit discard in render target transfer pipelines. "
            "May improve performance on some GPUs.",
            "GPU");

DEFINE_bool(gpu_3d_to_2d_texture, true,
            "Handle shaders that sample 3D textures as 2D by creating a 2D "
            "texture from slice 0 of the guest memory.",
            "GPU");

DEFINE_bool(
    async_shader_compilation, true,
    "Compile shaders and create pipelines asynchronously in background "
    "threads. "
    "Eliminates shader compilation stutter but may cause brief rendering "
    "artifacts while pipelines are being created. When disabled, pipelines are "
    "created synchronously which causes stutter but no visual artifacts.",
    "GPU");

DEFINE_bool(
    ac6_ground_fix, false,
    "This fixes(hide) issues with black ground in AC6. Use only in AC6. "
    "Might cause issues in other titles.",
    "HACKS");

DEFINE_bool(
    depth_float24_sample_z_varying, false,
    "With depth_float24_convert_in_pixel_shader: pass the NDC depth (z/w) "
    "from "
    "the vertex shader as a sample-interpolated noperspective varying and "
    "convert that instead of gl_FragCoord.z - the exact per-sample depth "
    "even on hosts where FragCoord.z is not per-sample under sample-rate "
    "shading (Apple GPUs via MoltenVK), without the precision loss of the "
    "derivative-based reconstruction at steep triangles. Takes precedence "
    "over depth_float24_sample_z_from_derivatives.",
    "GPU");

DEFINE_bool(
    depth_float24_sample_z_from_derivatives, false,
    "With depth_float24_convert_in_pixel_shader: reconstruct the per-sample "
    "depth as z + dFdx(z)*dx + dFdy(z)*dy from the sample position instead "
    "of trusting gl_FragCoord.z, for hosts where FragCoord.z is not "
    "evaluated at the sample position under forced sample-rate shading "
    "(Apple GPUs via MoltenVK: sample pairs receive identical depth, which "
    "neutralizes Z-prepasses and flattens MSAA depth).",
    "GPU");

DEFINE_bool(
    texture_alpha_test_lod_snap, false,
    "For draws with the fixed-function alpha test enabled, snap the "
    "shader-computed/register texture LOD to the nearest integer mip. "
    "Alpha-tested fragment survival flips on the filtered alpha crossing "
    "the reference, so sampler-internal cross-vendor differences (blend "
    "weight precision, ramp shape) directly flip fragment ownership at "
    "triangle edges (the Apple-vs-AMD/Xenos mesh); single-level sampling "
    "makes the verdict host-invariant. Requires "
    "texture_computed_lod_in_shader for computed-LOD fetch coverage.",
    "GPU");
// The Apple texture-determinism stack was defaulted on (a45a63eb0); it is now
// opt-in. Migrate configs that still hold the old default off, preserving any
// explicit user override.
UPDATE_from_bool(texture_alpha_test_lod_snap, 2026, 7, 7, 12, true);

DEFINE_bool(
    texture_decompress_bc, false,
    "Decompress BC1/BC2/BC3 (DXT) textures to RGBA8 at upload with Xenia's "
    "own compute shaders instead of sampling the compressed blocks - the "
    "hardware BC decoders' interpolation precision is vendor-specific "
    "(notably the BC3/DXT5 3-bit interpolated alpha), which flips "
    "alpha-test verdicts at threshold-adjacent pixels across GPUs.",
    "GPU");

DEFINE_bool(
    texture_trilinear_ramp_match_amd, false,
    "Remap the explicit-LOD trilinear blend fraction to the AMD/Xenos ramp "
    "(weight = saturate((frac - 0.1) / 0.8), measured; Apple GPUs use a "
    "plain linear ramp) - reproduces AMD's mip pair and blend weight "
    "including the pure-level dead zones within 0.1 of integer LODs. "
    "Applies to register-LOD and shader-computed-LOD fetches (combine with "
    "texture_computed_lod_in_shader for coverage of computed-LOD fetches).",
    "GPU");

DEFINE_double(
    texture_mip_lod_bias_adjust, 0.0,
    "Constant sampler mipLodBias applied to all guest texture samplers, "
    "calibrating the host GPU's lambda->mip-pair mapping to the Xenos/AMD "
    "convention (Apple GPUs transition ~0.099 LOD later than AMD at "
    "identical explicit LOD - the cross-GPU triangle-edge mesh). "
    "Default 0.0 (neutral); the 0.099 Apple calibration is opt-in and applied "
    "only on the texture-determinism host (Apple, or force-enabled), inert "
    "elsewhere.",
    "GPU");
UPDATE_from_double(texture_mip_lod_bias_adjust, 2026, 7, 7, 12, 0.099);

DEFINE_bool(
    interpolators_manual_perspective, false,
    "Interpolate guest interpolators as noperspective A = o/w (vertex "
    "shader pre-divides by the final position W) and reconstruct "
    "o = A / gl_FragCoord.w in the fragment shader prologue. Quad "
    "derivatives of perspective-interpolated varyings at partially covered "
    "quads are implementation-defined and differ between Apple and AMD "
    "GPUs (up to ~1000x at edge pixels - the cross-GPU wireframe mesh via "
    "texture LOD); noperspective varyings and gl_FragCoord are "
    "helper-lane-exact on both vendors, making values AND derivatives "
    "host-invariant.",
    "GPU");
UPDATE_from_bool(interpolators_manual_perspective, 2026, 7, 7, 12, true);

DEFINE_bool(
    texture_computed_lod_in_shader, false,
    "Convert texture gradients to a scalar LOD with a host-invariant formula "
    "in the shader (1D/2D fetches with computed LOD) instead of passing Grad "
    "operands - the hardware gradients-to-LOD function is "
    "implementation-defined and flat-triangle-constant, so cross-GPU formula "
    "differences flip whole triangles to adjacent mips (the Apple-vs-AMD "
    "wireframe mesh on high-contrast art). Disables anisotropic filtering "
    "for the converted fetches.",
    "GPU");
UPDATE_from_bool(texture_computed_lod_in_shader, 2026, 7, 7, 12, true);

DEFINE_bool(
    texture_lod_snap_all, false,
    "Probe: snap the shader-computed LOD to the nearest integer mip for "
    "ALL fetches, not only alpha-tested draws. Both Apple and AMD evaluate "
    "helper-lane perspective ratios with vendor wobble at partially "
    "covered quads (measured: ~2e-2 p90 deviation from the exact plane "
    "ratio on both, mutually divergent) - snapping collapses the wobble "
    "to a deterministic level choice except within rounding distance of "
    "mip boundaries.",
    "GPU");

DEFINE_bool(
    texture_lod_plane_inputs, false,
    "With texture_computed_lod_in_shader + interpolators_manual_perspective: "
    "difference the raw per-lane fetch coordinates directly for the "
    "analytic LOD, without covered-lane substitution. Under manual "
    "perspective the per-lane UVs derive from noperspective varyings, "
    "which are helper-lane plane-exact on all vendors - so these "
    "differences equal the AMD/Xenos hardware's plane-true gradients "
    "at partially covered quads, where covered-lane substitution "
    "diverges from AMD by construction. Takes precedence over "
    "texture_lod_covered_lane_inputs.",
    "GPU");

DEFINE_bool(
    texture_lod_hw_derivatives, false,
    "Probe / candidate lever: compute the shader analytic-LOD quad "
    "derivatives with hardware OpDPdxCoarse/OpDPdyCoarse instead of the "
    "manual OpGroupNonUniformQuadSwap + gl_FragCoord-parity difference, even "
    "under interpolators_manual_perspective. Tests whether the Apple "
    "analytic-LOD gradient blow-up at covered quads is a subgroup-quad "
    "layout / QuadSwap-neighbor problem.",
    "GPU");

DEFINE_bool(
    texture_lod_quotient_inputs, false,
    "Probe: with texture_computed_lod_in_shader + "
    "interpolators_manual_perspective, compute the analytic LOD via the "
    "quotient rule on the linear A = uv*q and q = 1/w varyings with RAW "
    "per-lane values (the historical no-HelperInvocation fallback), "
    "bypassing covered-lane substitution and direct differencing. Takes "
    "precedence over texture_lod_plane_inputs and "
    "texture_lod_covered_lane_inputs.",
    "GPU");

DEFINE_bool(
    texture_lod_vertex_analytic, false,
    "Apple analytic-per-triangle texture LOD (Route 1, "
    "VK_KHR_fragment_shader_barycentric): with "
    "texture_computed_lod_in_shader + interpolators_manual_perspective, "
    "compute the screen-space UV gradient analytically from the triangle's "
    "three vertices' per-vertex (A = uv/w, W = 1/w, NDC position) read via "
    "PerVertexKHR, instead of the 2x2-quad derivative that scatters on Apple "
    "(numerator A not plane-exact under noperspective interpolation). The "
    "per-triangle-constant gradient has zero per-quad scatter by "
    "construction, collapsing the seam/lattice. SPIKE: sources the gradient "
    "from interpolator texture_lod_vertex_analytic_interp components .xy and "
    "applies it to the fetch whose constant index is "
    "texture_lod_vertex_analytic_tf.",
    "GPU");
DEFINE_int32(texture_lod_vertex_analytic_tf, 0,
             "See texture_lod_vertex_analytic: fetch-constant index to apply "
             "the analytic gradient to (-1 = all computed-LOD fetches).",
             "GPU");
DEFINE_int32(texture_lod_vertex_analytic_interp, 0,
             "See texture_lod_vertex_analytic: source interpolator index whose "
             ".xy carries the fetch's uv (specimen ground PS: 0).",
             "GPU");

DEFINE_bool(
    texture_lod_primitive_id_analytic, false,
    "EXPERIMENTAL / research only - not a shipping rendering path. Enabling it "
    "allocates a 256 MiB device ring and adds a per-vertex SSBO write plus "
    "per-fragment index/SSBO loads; it fixes only the narrow desert/character- "
    "seam LOD-scatter and is not calibrated for general content. "
    "Apple analytic per-triangle texture-LOD gradient sourced via "
    "gl_PrimitiveID + the per-vertex SSBO (no PerVertexKHR / MoltenVK fork). "
    "The VS writes {A=uv/w, W=1/w, ndc.xy} per vertex; the PS reads its "
    "triangle's three vertices by primitive-id and computes a "
    "per-triangle-constant (plane-exact) LOD, removing the per-quad LOD "
    "scatter that dashes the ground on non-plane-exact Apple interpolation. "
    "fix-apple-lod-primitive-id-analytic-gradient.",
    "GPU");
// One-time reset of an experimental rendering path. Unlike the stack cvars
// above (migrated only from their old default, preserving explicit overrides),
// this UNCONDITIONALLY resets texture_lod_primitive_id_analytic to its default
// (off) for any config predating this date - the heavy 256 MiB analytic path
// was only ever meant for investigation and should not persist from a debug
// session as a silent perf cost. Configs written after this date (i.e. anyone
// who deliberately re-enables it later) are unaffected. This is a deliberate
// exception for an experimental path, NOT a general policy for opt-in cvars.
UPDATE_from_any(texture_lod_primitive_id_analytic, 2026, 7, 7, 12);
DEFINE_int32(texture_lod_primitive_id_analytic_tf, -1,
             "See texture_lod_primitive_id_analytic: fetch-constant index to "
             "apply the primitive-id analytic LOD to (-1 = all).",
             "GPU");

DEFINE_bool(
    texture_determinism_force_non_apple, false,
    "Diagnostic: treat non-Apple GPUs as a texture-determinism host so the "
    "individually-enabled stack cvars (texture_computed_lod_in_shader, "
    "interpolators_manual_perspective, texture_alpha_test_lod_snap, "
    "texture_mip_lod_bias_adjust) also take effect off Apple, for symmetric "
    "cross-vendor A/B experiments. The stack defaults OFF everywhere (opt-in); "
    "this only widens the host gate, it does not enable any layer on its own.",
    "GPU");

DEFINE_bool(
    texture_lod_covered_lane_inputs, false,
    "With texture_computed_lod_in_shader + interpolators_manual_perspective: "
    "before quad-differencing the analytic-LOD inputs, replace each helper "
    "(uncovered) lane's value with the first covered lane's value in the "
    "2x2 quad. Helper-lane values are only cross-vendor equal to ~ulp, and "
    "the quotient rule catastrophically cancels at edge-on triangles, "
    "amplifying that noise into multi-mip LOD differences at partially "
    "covered quads (Apple-vs-AMD alpha-test ownership flips - the mesh). "
    "Quad coverage itself is cross-vendor identical, so the corrected "
    "lambda is host-invariant; fully covered quads are unaffected.",
    "GPU");
UPDATE_from_bool(texture_lod_covered_lane_inputs, 2026, 7, 7, 12, true);

DEFINE_int32(
    texture_lod_force_level, -1,
    "Diagnostic: pin all guest texture samplers to this exact mip level "
    "(minLod = maxLod = N). Comparing single-level renders across GPUs "
    "separates mip content / BC-decode divergence from LOD selection "
    "divergence. -1 = off.",
    "GPU");

DEFINE_bool(
    apple_texture_lod_force_base, true,
    "On Apple GPUs, pin texture sampling to the base mip (level 0) by default. "
    "Apple's per-triangle quad-derivative LOD diverges across triangle edges, "
    "producing a mip-seam \"crosshatch\" net on character skin in many titles "
    "(USF4/RDR/TMNT). Pinning the base mip removes it at near-zero cost - much "
    "cheaper than the in-shader analytic LOD - trading mip minification "
    "(some distant-texture shimmer) for clean surfaces, which suits close-camera "
    "content. Applied only on Apple hosts. An explicit texture_lod_force_level "
    ">= 0 overrides this; set this false to restore normal mip selection on "
    "Apple.",
    "GPU");

DEFINE_bool(
    texture_max_lod_clamp_probe, false,
    "Diagnostic: clamp all guest texture samplers to the base mip "
    "(minLod=0, maxLod=0.25), removing implicit-LOD (quad derivative) "
    "influence from sampling - for attributing cross-GPU divergence along "
    "triangle edges to LOD selection differences.",
    "GPU");

DEFINE_bool(
    force_per_sample_shading_probe, false,
    "Diagnostic: add a SampleId input to all translated pixel shaders, "
    "forcing per-sample shading without any depth conversion - isolates "
    "the shading-rate side effect of depth_float24_convert_in_pixel_shader "
    "from its depth quantization math.",
    "GPU");

DEFINE_int32(
    vertex_position_quantize_bits, 0,
    "Round the final vertex position outputs (clip XYZ and W) to a "
    "2^-(23-N) relative grid by dropping N low mantissa bits "
    "(round-to-nearest) in all translated vertex shaders. Collapses "
    "sub-ulp position differences between logically identical vertex "
    "shaders compiled into different pipelines, emulating the tolerance "
    "that guest f24 (and host D24) depth quantization provides - for "
    "hosts whose GPU compilers produce per-function rounding differences "
    "(Apple/Metal multi-pass depth conflicts). 0 = off; 3 aligns with one "
    "f24 depth ulp.",
    "GPU");

DEFINE_int32(
    ps_debug_export_instr, -1,
    "Diagnostic, with ps_debug_export_ps_hash: ucode instruction slot index "
    "(the number shown in ucode disassembly) after which the register "
    "selected by ps_debug_export_reg is captured in the matching pixel "
    "shader; the captured value replaces guest color output 0 at the end of "
    "the shader, so intermediate ALU dataflow can be read back through the "
    "render target. -1 = off.",
    "GPU");

DEFINE_int32(ps_debug_export_reg, 0,
             "See ps_debug_export_instr: index of the temporary register "
             "(rN) to capture.",
             "GPU");

DEFINE_int32(
    ps_debug_export_lod, -1,
    "Diagnostic, with ps_debug_export_ps_hash: capture the shader-computed "
    "analytic texture LOD of the matching pixel shader's computed-LOD "
    "fetches into guest color output 0 (readback through the render "
    "target). 1 = R:max texel-gradient magnitude/32, G:lambda/16+0.5, "
    "B:helper-neighbor fraction of the 2x2 quad; 2 = R:axis-X gradient/32, "
    "G:axis-Y gradient/32, B:lambda/16+0.5; 3 = analytic-vs-hardware de-risk: "
    "R:analytic lambda/16+0.5, G:hardware OpImageQueryLod lambda/16+0.5, "
    "B:(analytic-hardware)/16+0.5 (same coords/GPU -> isolates whether the "
    "analytic FORMULA reproduces hardware LOD); 4 = plane-exactness walk (uses "
    "ps_debug_export_scale as gain): R:q(=inv_w)*scale, G:0.5+scale*d(q)/dx, "
    "B:0.5+scale*d(o)/dx where o=A/q is the reconstructed perspective UV -> "
    "shows whether the ballooning UV finite difference is inherited from the "
    "denominator q (G scatters) or appears only in o (numerator A not "
    "plane-exact). Isolates whether the Apple "
    "analytic-LOD collapse is a gradient magnitude loss and which axis "
    "loses it; 5 = exact-bits export: pack the full IEEE-754 word of the "
    "quantity selected by ps_debug_export_bits into RGBA8 (byte 0->R .. byte "
    "3->A, each value/255) so the interpolated value can be compared bit-exact "
    "cross-machine / used offline (untile, reassemble uint, bitcast to float). "
    "-1 = off.",
    "GPU");
DEFINE_int32(ps_debug_export_bits, 0,
             "See ps_debug_export_lod=5: which quantity to bit-pack. "
             "0=A.x (o.x*W), 1=A.y, 2=W(=1/w), 3=o.x (reconstructed UV), "
             "4=o.y.",
             "GPU");

DEFINE_int32(
    ps_debug_export_lod_tf, -1,
    "See ps_debug_export_lod: restrict the LOD capture to the fetch whose "
    "texture fetch-constant index equals this value (a pixel shader samples "
    "several textures with different gradients; without this the last fetch "
    "wins). -1 = capture every computed-LOD fetch (last stored wins).",
    "GPU");

DEFINE_double(
    ps_debug_export_lod_range, 8.0,
    "See ps_debug_export_lod: half-range (in mips) of the lambda encoding. "
    "Encoded value E = lambda/(2*range) + 0.5, so lambda in [-range,+range] "
    "maps to [0,1]. Default 8 matches the historical /16 scale; widen (e.g. "
    "16) to see whether horizon LODs genuinely agree or are merely clamped at "
    "the +8 ceiling.",
    "GPU");

DEFINE_string(ps_debug_export_ps_hash, "",
              "See ps_debug_export_instr: hex ucode hash of the pixel shader "
              "to instrument (required - empty disables the export).",
              "GPU");

DEFINE_double(ps_debug_export_scale, 1.0,
              "See ps_debug_export_instr: multiplier applied to the captured "
              "register before it is written to color output 0, for reading "
              "values much smaller than the render target's LSB.",
              "GPU");

DEFINE_int32(ps_debug_export_component, -1,
             "See ps_debug_export_instr: -1 = export the register as-is; "
             "0-3 = broadcast that component (x/y/z/w) to RGB, making the "
             "w component readable through the color output.",
             "GPU");
