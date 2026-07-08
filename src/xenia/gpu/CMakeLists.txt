add_library(xenia-gpu STATIC)
xe_platform_sources(xenia-gpu ${CMAKE_CURRENT_SOURCE_DIR})
# The Vulkan/SPIR-V backend is now built on macOS (MoltenVK), so the spirv_*
# sources and glslang/Vulkan-Headers includes are needed on all platforms.
target_include_directories(xenia-gpu PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/Vulkan-Headers/include
  ${PROJECT_SOURCE_DIR}/third_party/glslang
)
target_link_libraries(xenia-gpu PUBLIC dxbc fmt glslang-spirv snappy xxhash xenia-base xenia-ui)
xe_target_defaults(xenia-gpu)

if(XENIA_BUILD_TESTS)
  set(CMAKE_FOLDER "tests")
  add_subdirectory(testing)
endif()

# Experimental native-Metal shader path (Apple only, opt-in).
if(APPLE AND XENIA_GPU_METAL)
  add_subdirectory(metal)
endif()

if(XENIA_BUILD_MISC)
  # Shader compiler tool
  add_executable(xenia-gpu-shader-compiler
    ${CMAKE_CURRENT_SOURCE_DIR}/shader_compiler_main.cc
  )
  if(WIN32)
    target_sources(xenia-gpu-shader-compiler PRIVATE
      ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_win.cc)
  else()
    target_sources(xenia-gpu-shader-compiler PRIVATE
      ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_posix.cc)
  endif()
  target_include_directories(xenia-gpu-shader-compiler PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/Vulkan-Headers/include
    ${PROJECT_SOURCE_DIR}/third_party/glslang
  )
  target_link_libraries(xenia-gpu-shader-compiler PRIVATE
    dxbc fmt glslang-spirv snappy xenia-base xenia-gpu xenia-ui xenia-ui-vulkan
  )
  xe_target_defaults(xenia-gpu-shader-compiler)
endif()
