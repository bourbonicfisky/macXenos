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
