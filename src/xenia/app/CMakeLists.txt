# src/xenia/app/CMakeLists.txt - Main xenia_canary executable.

if(WIN32)
  add_executable(xenia-app WIN32)
else()
  add_executable(xenia-app)
endif()

set_target_properties(xenia-app PROPERTIES OUTPUT_NAME "xenia_canary")

# Main source
target_sources(xenia-app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/xenia_main.cc
)

# Other app sources (emulator_window, profile_dialogs, etc.)
xe_platform_sources(xenia-app ${CMAKE_CURRENT_SOURCE_DIR})

# Platform-specific entry points
if(WIN32)
  target_sources(xenia-app PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/main_init_win.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/ui/windowed_app_main_win.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/main_resources.rc
    ${PROJECT_SOURCE_DIR}/src/xenia/base/app_win32.manifest
  )
  if(XE_TARGET_X86_64)
    # main_init_win.cc needs SSE2 only (not AVX)
    set_source_files_properties(
      ${PROJECT_SOURCE_DIR}/src/xenia/base/main_init_win.cc
      PROPERTIES COMPILE_OPTIONS "/arch:SSE2"
    )
  endif()
elseif(APPLE)
  # macOS: Cocoa entry (windowed_app_main_mac.cc) + a non-GTK AVX check, instead
  # of the GTK-coupled posix files.
  target_sources(xenia-app PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/main_init_mac.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/ui/windowed_app_main_mac.cc
  )
  # The default x86_64 __PAGEZERO is 4 GB, which makes the entire sub-4 GB
  # address space unmappable - including the x64 code cache's fixed ranges
  # (indirection table 0x80000000, generated code 0xA0000000/0xB0000000).
  # Shrink __PAGEZERO so those ranges can be mapped (also relocates __TEXT).
  # arm64 must NOT shrink it: macOS 26 SIGKILLs any sub-default __PAGEZERO, so
  # the a64 code cache is relocated above 4 GB instead (64-bit indirection
  # table; see code_cache_base.h / a64_backend.cc).
  if(NOT XE_TARGET_AARCH64)
    target_link_options(xenia-app PRIVATE -Wl,-pagezero_size,0x1000)
  endif()
  if(XE_TARGET_X86_64)
    # The AVX check must run on non-AVX CPUs, so don't build it with AVX.
    set_source_files_properties(
      ${PROJECT_SOURCE_DIR}/src/xenia/base/main_init_mac.cc
      PROPERTIES COMPILE_OPTIONS "-msse2;-mno-avx"
    )
  endif()
else()
  target_sources(xenia-app PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/main_init_posix.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/ui/windowed_app_main_posix.cc
  )
  if(NOT MSVC AND XE_TARGET_X86_64)
    set_source_files_properties(
      ${PROJECT_SOURCE_DIR}/src/xenia/base/main_init_posix.cc
      PROPERTIES COMPILE_OPTIONS "-msse2;-mno-avx"
    )
  endif()
endif()

target_compile_definitions(xenia-app PRIVATE
  XBYAK_NO_OP_NAMES
  XBYAK_ENABLE_OMITTED_OPERAND
)

# Core libraries (all platforms)
target_link_libraries(xenia-app PRIVATE
  xenia-apu
  xenia-apu-nop
  xenia-base
  xenia-core
  xenia-cpu
  xenia-gpu
  xenia-gpu-null
  xenia-hid
  xenia-hid-nop
  xenia-hid-scripted
  xenia-kernel
  xenia-patcher
  xenia-ui
  xenia-vfs
  # Third-party
  aes_128
  capstone
  fmt
  dxbc
  discord-rpc
  imgui
  libavcodec
  libavutil
  mspack
  snappy
  xxhash
)

# Vulkan/SPIR-V backends (macOS via MoltenVK).
target_link_libraries(xenia-app PRIVATE
  xenia-gpu-vulkan
  xenia-ui-vulkan
  glslang-spirv
)

# Experimental native-Metal backend (Apple only, opt-in via XENIA_GPU_METAL).
if(APPLE AND XENIA_GPU_METAL)
  target_link_libraries(xenia-app PRIVATE xenia-gpu-metal-backend)
endif()

# Non-Android specific (always true for Windows/Linux)
target_link_libraries(xenia-app PRIVATE
  xenia-app-discord
  xenia-debug-ui
)
# SDL audio/input. On macOS SDL2 is built from the vendored submodule, so the
# app links the SDL modules on all desktop platforms (audio is registered;
# SDL HID remains unregistered on macOS until input is wired separately).
target_link_libraries(xenia-app PRIVATE
  xenia-apu-sdl
  xenia-helper-sdl
  xenia-hid-sdl
)

# Architecture-specific backend
if(XE_TARGET_X86_64)
  target_link_libraries(xenia-app PRIVATE xenia-cpu-backend-x64)
elseif(XE_TARGET_AARCH64)
  target_link_libraries(xenia-app PRIVATE xenia-cpu-backend-a64)
endif()

# Platform-specific libraries
if(WIN32)
  target_link_libraries(xenia-app PRIVATE
    xenia-apu-xaudio2
    xenia-gpu-d3d12
    xenia-hid-winkey
    xenia-hid-xinput
    xenia-ui-d3d12
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(xenia-app PRIVATE
    xenia-apu-alsa
    X11
    xcb
    X11-xcb
    SDL2
  )
endif()

xe_target_defaults(xenia-app)

if(APPLE)
  # The x64 JIT executes generated code from a private MAP_JIT region (see
  # CodeCacheBase::Initialize); macOS only permits executing such memory when
  # the binary is codesigned with the JIT entitlements. Ad-hoc sign every build
  # so the code cache is executable (without this the guest faults with
  # EXC_BAD_ACCESS code=2 on its first generated instruction).
  add_custom_command(TARGET xenia-app POST_BUILD
    COMMAND codesign -s - --entitlements
            ${CMAKE_CURRENT_SOURCE_DIR}/xenia.entitlements --force
            $<TARGET_FILE:xenia-app>
    COMMENT "Ad-hoc signing xenia_canary with JIT entitlements"
  )
endif()
