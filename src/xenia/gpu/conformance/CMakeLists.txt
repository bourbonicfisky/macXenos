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
xe_target_defaults(xenia-gpu-conformance)
set_target_properties(xenia-gpu-conformance PROPERTIES FOLDER "tests")
