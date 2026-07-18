xe_test_suite(xenia-hid-scripted-tests ${CMAKE_CURRENT_SOURCE_DIR}
  LINKS xenia-hid-scripted
)

target_sources(xenia-hid-scripted-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/input_notification_stub.cc)

target_compile_definitions(xenia-hid-scripted-tests PRIVATE
  XENIA_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
