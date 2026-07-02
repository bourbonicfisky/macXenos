xe_test_suite(xenia-cpu-tests ${CMAKE_CURRENT_SOURCE_DIR}
  LINKS
    capstone fmt imgui
    xenia-base xenia-core xenia-cpu xenia-gpu
    xenia-kernel xenia-ui xenia-patcher
    $<$<BOOL:${XE_TARGET_X86_64}>:xenia-cpu-backend-x64>
    $<$<BOOL:${XE_TARGET_AARCH64}>:xenia-cpu-backend-a64>
)

# The float-divergence harness is not named *_test.cc (it is a diff-able
# diagnostic suite, hidden behind the [divergence] tag, not a pass/fail
# test), so xe_test_suite's glob misses it; add it explicitly.
if(TARGET xenia-cpu-tests)
  target_sources(xenia-cpu-tests PRIVATE float_divergence_suite.cc)
endif()
