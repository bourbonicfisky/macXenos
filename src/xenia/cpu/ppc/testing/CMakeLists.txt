if(NOT XENIA_BUILD_TESTS)
  return()
endif()

add_executable(xenia-cpu-ppc-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/ppc_testing_main.cc
)

# Add platform-specific console app main
if(WIN32)
  target_sources(xenia-cpu-ppc-tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_win.cc)
else()
  target_sources(xenia-cpu-ppc-tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_posix.cc)
endif()

# Add .s files but exclude from build (they're test data)
file(GLOB _ppc_asm  "${CMAKE_CURRENT_SOURCE_DIR}/*.s")
target_sources(xenia-cpu-ppc-tests PRIVATE ${_ppc_asm})
set_source_files_properties(${_ppc_asm} PROPERTIES HEADER_FILE_ONLY TRUE)

target_link_libraries(xenia-cpu-ppc-tests PRIVATE
  xenia-apu xenia-apu-nop
  xenia-base xenia-core xenia-cpu
  xenia-gpu xenia-gpu-null
  xenia-hid xenia-hid-nop
  xenia-kernel xenia-patcher xenia-ui xenia-vfs
  capstone fmt imgui mspack
)

# Architecture-specific backend
if(XE_TARGET_X86_64)
  target_link_libraries(xenia-cpu-ppc-tests PRIVATE xenia-cpu-backend-x64)
elseif(XE_TARGET_AARCH64)
  target_link_libraries(xenia-cpu-ppc-tests PRIVATE xenia-cpu-backend-a64)
endif()

# The x64 JIT code cache reserves fixed sub-4 GB addresses; the default 64-bit
# macOS __PAGEZERO (4 GB) makes them unmappable, so shrink it like xenia-app
# does. arm64 relocates the code cache above 4 GB instead (macOS 26 forbids any
# sub-default __PAGEZERO), so it keeps the default.
if(APPLE AND NOT XE_TARGET_AARCH64)
  target_link_options(xenia-cpu-ppc-tests PRIVATE -Wl,-pagezero_size,0x1000)
endif()

xe_target_defaults(xenia-cpu-ppc-tests)
add_test(NAME xenia-cpu-ppc-tests COMMAND xenia-cpu-ppc-tests)
