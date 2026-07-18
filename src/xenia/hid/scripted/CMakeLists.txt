add_library(xenia-hid-scripted STATIC)
xe_platform_sources(xenia-hid-scripted ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xenia-hid-scripted PUBLIC
  tomlplusplus xxhash xenia-base xenia-hid)
xe_target_defaults(xenia-hid-scripted)

if(XENIA_BUILD_TESTS)
  set(CMAKE_FOLDER "tests")
  add_subdirectory(testing)
endif()
