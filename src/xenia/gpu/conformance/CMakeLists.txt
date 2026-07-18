# Metal GPU conformance ladder — Level 0 (CPU format/address oracles).
#
# Deliberately a LEAN executable: it compiles only the pure-math oracle sources
# (xenos.cc + header-only addressers) plus the Catch2 harness, and does NOT link
# the monolithic xenia-gpu lib. Linking xenia-gpu drags in the whole emulator
# (command_processor -> kernel -> xam UI -> ImGui), which both bloats the binary
# and bit-rots (missing links). Keeping this target minimal is what lets a
# format/address contract violation fail in milliseconds, isolated — the point
# of the ladder. Higher levels (Metal kernels, synthetic PM4) get their own
# targets that may add the Metal frameworks, but Level 0 stays device-free.
#
# On arm64 there is no shrunk __PAGEZERO, so this binary runs in restricted
# environments (agent/CI) unlike the full app.

file(GLOB _conformance_sources "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cc")

if(NOT _conformance_sources)
  return()
endif()

add_executable(xenia-gpu-conformance
  ${_conformance_sources}
  ${PROJECT_SOURCE_DIR}/src/xenia/gpu/xenos.cc
  # EDRAM tile/sample addressing oracle (mirror of edram.xesli). Pure math +
  # fmt/base/stb — no command_processor, so it stays in the lean target.
  ${PROJECT_SOURCE_DIR}/src/xenia/gpu/trace_analysis_decode.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/stb_image_write_impl.cc
  ${PROJECT_SOURCE_DIR}/tools/build/src/test_suite_main.cc
  ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_posix.cc
)

target_compile_definitions(xenia-gpu-conformance PRIVATE
  "XE_TEST_SUITE_NAME=\"xenia-gpu-conformance\""
)
target_include_directories(xenia-gpu-conformance PRIVATE
  ${PROJECT_SOURCE_DIR}/tools/build
  ${PROJECT_SOURCE_DIR}/tools/build/src
  ${PROJECT_SOURCE_DIR}/tools/build/third_party/catch/include
)
target_link_libraries(xenia-gpu-conformance PRIVATE fmt xenia-base)

# Level 1: Metal compute kernels checked against the L0 CPU oracles. Compiled
# only on Apple (needs a Metal device); on other platforms the executable is
# L0-only. Kept in the SAME binary so one run covers both rungs.
if(APPLE)
  target_sources(xenia-gpu-conformance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_kernel_conformance_test.mm
    # Shared-memory coherence surface (L1): links the REAL Memory +
    # SharedMemory + MetalSharedMemory (no replicas). This pulls a small,
    # deliberate closure — memory + mmio + shared-memory validity + the gpu
    # cvar definitions + trace writer (referenced by MetalSharedMemory) —
    # kept minimal so the target stays a fast isolated contract check.
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_shared_memory_coherence_test.mm
    ${PROJECT_SOURCE_DIR}/src/xenia/memory.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/cpu/mmio_handler.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/gpu/shared_memory.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/gpu/gpu_flags.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/gpu/trace_writer.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/gpu/metal/metal_shared_memory.mm)
  target_include_directories(xenia-gpu-conformance PRIVATE
    ${CMAKE_BINARY_DIR})  # version.h (trace_writer)
  target_link_libraries(xenia-gpu-conformance PRIVATE
    snappy "-framework Metal" "-framework Foundation")
endif()

xe_target_defaults(xenia-gpu-conformance)
set_target_properties(xenia-gpu-conformance PROPERTIES FOLDER "tests")
