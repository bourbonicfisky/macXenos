# Native-Metal UI provider/presenter (Apple only, opt-in via XENIA_GPU_METAL).
# Parallels ui/vulkan: vends the GraphicsProvider + Presenter + ImmediateDrawer
# the Metal graphics backend presents through.
add_library(xenia-ui-metal STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_immediate_drawer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_presenter.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_provider.mm
)
target_include_directories(xenia-ui-metal PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(xenia-ui-metal PUBLIC
  xenia-base
  xenia-ui
  "-framework Metal"
  "-framework QuartzCore"
  "-framework Foundation"
)
xe_target_defaults(xenia-ui-metal)
