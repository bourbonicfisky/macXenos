# Native-Metal UI provider/presenter (Apple only, opt-in via XENIA_GPU_METAL).
# Parallels ui/vulkan: vends the GraphicsProvider + Presenter + ImmediateDrawer
# the Metal graphics backend presents through.
add_library(xenia-ui-metal STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_immediate_drawer.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_presenter.mm
  ${CMAKE_CURRENT_SOURCE_DIR}/metal_provider.mm
)
target_include_directories(xenia-ui-metal PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(xenia-ui-metal PUBLIC
  xenia-base
  xenia-ui
  "-framework Metal"
  "-framework MetalFX"
  "-framework QuartzCore"
  "-framework Foundation"
)
# The immediate drawer's overlay pipeline and the presenter's guest output
# effects (bilinear, AMD FidelityFX CAS and FSR) consume the shared polyglot
# XeSL sources compiled to metallibs. Same rule and same compiler the GPU
# texture-load shaders use; only the source glob differs, since
# compile_shader_metal.py is stage-agnostic.
xe_shader_rules_metal(xenia-ui-metal
  ${PROJECT_SOURCE_DIR}/src/xenia/ui/shaders
  "immediate.vs.xesl" "immediate.ps.xesl" "guest_output_*.xesl"
)
xe_target_defaults(xenia-ui-metal)
