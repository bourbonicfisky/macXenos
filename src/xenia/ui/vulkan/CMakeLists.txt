add_library(xenia-ui-vulkan STATIC)
xe_platform_sources(xenia-ui-vulkan ${CMAKE_CURRENT_SOURCE_DIR})
xe_platform_sources(xenia-ui-vulkan ${CMAKE_CURRENT_SOURCE_DIR}/functions)
xe_shader_rules_spirv(xenia-ui-vulkan ${CMAKE_CURRENT_SOURCE_DIR}/../shaders)
target_include_directories(xenia-ui-vulkan PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/Vulkan-Headers/include
  ${PROJECT_SOURCE_DIR}/third_party/glslang
)
target_link_libraries(xenia-ui-vulkan PUBLIC xenia-base xenia-ui)
xe_target_defaults(xenia-ui-vulkan)

if(XENIA_BUILD_MISC)
  # Vulkan window demo
  add_executable(xenia-ui-window-vulkan-demo)
  target_sources(xenia-ui-window-vulkan-demo PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/ui/windowed_app_main_posix.cc)
  target_sources(xenia-ui-window-vulkan-demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../window_demo.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_window_demo.cc
  )
  target_include_directories(xenia-ui-window-vulkan-demo PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/Vulkan-Headers/include
  )
  target_link_libraries(xenia-ui-window-vulkan-demo PRIVATE
    fmt imgui xenia-base xenia-ui xenia-ui-vulkan
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(xenia-ui-window-vulkan-demo PRIVATE X11 xcb X11-xcb)
  endif()
  xe_target_defaults(xenia-ui-window-vulkan-demo)
endif()
