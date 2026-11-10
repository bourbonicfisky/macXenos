add_library(xenia-hid-portal STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/portal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/portal.cc
)

xe_target_defaults(xenia-hid-portal)
