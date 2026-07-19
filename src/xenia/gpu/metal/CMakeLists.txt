# Experimental native-Metal shader path (Apple only, opt-in via XENIA_GPU_METAL).
# Phase 2a: the DXBC -> DXIL -> metallib translation wrappers + an offline spike
# that validates the whole chain without a renderer.

# Shared translation library: DxbcToDxilConverter (dxilconv) + MetalShaderConverter
# (Metal Shader Converter). Kept separate from xenia-gpu so the heavy third-party
# deps only build when the Metal path is requested.
add_library(xenia-gpu-metal STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/dxbc_to_dxil_converter.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_shader_converter.cc
)
target_include_directories(xenia-gpu-metal PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(xenia-gpu-metal PUBLIC
  xenia-base
  xenia-third-party-dxilconv
  xenia-third-party-metal-shader-converter
)
xe_target_defaults(xenia-gpu-metal)

# Offline spike driver: one DXBC fixture -> metallib. Own main(), no renderer.
add_executable(xenia-gpu-metal-shader-translate-spike
  ${CMAKE_CURRENT_SOURCE_DIR}/testing/metal_shader_translate_spike.cc
)
target_link_libraries(xenia-gpu-metal-shader-translate-spike PRIVATE
  xenia-gpu-metal
  "-framework Foundation"
  "-framework Metal"
)
xe_target_defaults(xenia-gpu-metal-shader-translate-spike)

# Phase 2b backend skeleton: the GraphicsSystem/CommandProcessor that register the
# "metal" backend. Separate from the shader-translation lib so the spike stays
# lean; links xenia-gpu (CommandProcessor base) and xenia-ui.
add_library(xenia-gpu-metal-backend STATIC
  # NOTE: the shared DXBC geometry-shader generator (gpu/dxbc_geometry_shader.cc,
  # point/rect/quad expansion for the MSC geometry-emulation path) is compiled into
  # xenia-gpu via xe_platform_sources' directory glob; do NOT add it here or the
  # symbol duplicates. The Metal backend gets it by linking xenia-gpu.
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_graphics_system.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_command_processor.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_pipeline_cache.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_primitive_processor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_render_target_cache.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_shared_memory.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_shared_texture_cache.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_texture_cache.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_upload_arena.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/ir_runtime_impl.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_shader.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_load_shader_pipelines.mm
)
# Generated texture-load metallib headers (bytecode/metal/*.h) consumed by
# metal_load_shader_pipelines.mm.
xe_shader_rules_metal(xenia-gpu-metal-backend ${CMAKE_CURRENT_SOURCE_DIR}/../shaders)
target_include_directories(xenia-gpu-metal-backend PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(xenia-gpu-metal-backend PUBLIC
  xenia-base
  xenia-gpu
  xenia-gpu-metal
  xenia-ui
  xenia-ui-metal
  xxhash
  "-framework Foundation"
  "-framework Metal"
)
xe_target_defaults(xenia-gpu-metal-backend)

# Metal trace dump: headless trace replay through the Metal backend, so captured
# guest frames can be re-rendered + dumped off-device for fast render-path
# iteration. Mirrors xenia-gpu-vulkan-trace-dump.
add_executable(xenia-gpu-metal-trace-dump
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_trace_dump_main.cc
)
if(WIN32)
  target_sources(xenia-gpu-metal-trace-dump PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_win.cc)
else()
  target_sources(xenia-gpu-metal-trace-dump PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_posix.cc)
endif()
target_include_directories(xenia-gpu-metal-trace-dump PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(xenia-gpu-metal-trace-dump PRIVATE
  xenia-apu xenia-apu-nop xenia-base xenia-core xenia-cpu
  xenia-gpu xenia-gpu-metal xenia-gpu-metal-backend xenia-hid xenia-hid-nop
  xenia-kernel xenia-patcher xenia-ui xenia-ui-metal xenia-vfs
  aes_128 capstone fmt imgui libavcodec libavutil mspack snappy xxhash
)
if(XE_TARGET_X86_64)
  target_link_libraries(xenia-gpu-metal-trace-dump PRIVATE xenia-cpu-backend-x64)
  if(APPLE)
    target_link_options(xenia-gpu-metal-trace-dump PRIVATE
      -Wl,-pagezero_size,0x1000)
  endif()
elseif(XE_TARGET_AARCH64)
  target_link_libraries(xenia-gpu-metal-trace-dump PRIVATE xenia-cpu-backend-a64)
endif()
xe_target_defaults(xenia-gpu-metal-trace-dump)
