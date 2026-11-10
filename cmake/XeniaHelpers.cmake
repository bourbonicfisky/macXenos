# XeniaHelpers.cmake - Helper functions for the Xenia build system.
# Replaces tools/build/scripts/*.lua

include(CMakeParseArguments)

# Platform suffix lists for file filtering
set(XE_PLATFORM_SUFFIXES
  _win _linux _posix _gnulinux _x11 _gtk _android _mac _amd64 _x64 _arm64
)

# xe_platform_sources(target base_path [RECURSIVE])
#
# Adds platform-filtered source files to a target.
# Globs *.h, *.cc, *.cpp, *.c, *.inc from base_path.
# Excludes all platform suffixes, then adds back the ones matching the current platform.
# Always excludes *_main.cc and *_test.cc.
function(xe_platform_sources target base_path)
  set(options RECURSIVE)
  cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})

  if(ARG_RECURSIVE)
    set(glob_mode GLOB_RECURSE)
    set(pattern "**")
  else()
    set(glob_mode GLOB)
    set(pattern "*")
  endif()

  # Glob all source files. .mm (Objective-C++) is macOS-only; include it so the
  # *_mac.mm windowing/platform sources can be added back on Apple below.
  file(${glob_mode} _all_sources    "${base_path}/*.h"
    "${base_path}/*.cc"
    "${base_path}/*.cpp"
    "${base_path}/*.c"
    "${base_path}/*.inc"
    "${base_path}/*.mm"
  )

  # Build exclude patterns
  set(_excluded)
  foreach(src ${_all_sources})
    get_filename_component(_name_we ${src} NAME_WE)
    # Exclude *_main.cc, *_test.cc, and *_demo.cc
    if(_name_we MATCHES "_main$" OR _name_we MATCHES "_test$" OR _name_we MATCHES "_demo$")
      list(APPEND _excluded ${src})
      continue()
    endif()
    # Exclude all platform suffixes
    foreach(suffix ${XE_PLATFORM_SUFFIXES})
      if(_name_we MATCHES "${suffix}$")
        list(APPEND _excluded ${src})
        break()
      endif()
    endforeach()
  endforeach()

  # Remove excluded files
  set(_sources ${_all_sources})
  if(_excluded)
    list(REMOVE_ITEM _sources ${_excluded})
  endif()

  # Add back platform-specific files
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(${glob_mode} _plat_sources      "${base_path}/*_posix.h"
      "${base_path}/*_posix.cc"
      "${base_path}/*_linux.h"
      "${base_path}/*_linux.cc"
      "${base_path}/*_gnulinux.h"
      "${base_path}/*_gnulinux.cc"
      "${base_path}/*_x11.h"
      "${base_path}/*_x11.cc"
      "${base_path}/*_gtk.h"
      "${base_path}/*_gtk.cc"
    )
  elseif(APPLE)
    # macOS is a POSIX/Unix host: it shares the _posix files (with macOS
    # specifics gated by XE_PLATFORM_MAC inside them) plus its own _mac files.
    # Objective-C++ (_mac.mm, Cocoa/Metal) is macOS-only. Mirrors mainline's
    # platform_files.lua Mac filter; deliberately excludes _linux/_gtk/_x11.
    file(${glob_mode} _plat_sources      "${base_path}/*_posix.h"
      "${base_path}/*_posix.cc"
      "${base_path}/*_mac.h"
      "${base_path}/*_mac.cc"
      "${base_path}/*_mac.mm"
    )
  endif()

  list(APPEND _sources ${_plat_sources})

  # Add back architecture-specific files
  if(XE_TARGET_X86_64)
    file(${glob_mode} _arch_sources "${base_path}/*_amd64.h" "${base_path}/*_amd64.cc"
      "${base_path}/*_x64.h" "${base_path}/*_x64.cc")
  elseif(XE_TARGET_AARCH64)
    file(${glob_mode} _arch_sources "${base_path}/*_arm64.h" "${base_path}/*_arm64.cc")
  endif()
  if(_arch_sources)
    list(APPEND _sources ${_arch_sources})
  endif()

  target_sources(${target} PRIVATE ${_sources})
endfunction()

# xe_target_defaults(target)
#
# Applies xenia-specific compile settings to a target:
# - Fatal warnings (-Werror)
# - Common include directories
function(xe_target_defaults target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
  )
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    # GCC is too noisy for -Werror; only apply to Clang.
    # macOS: Xenia and its bundled third-party libraries were written against an
    # older warning baseline than the current mac toolchain (libc++; e.g. size_t
    # vs uint64_t format mismatches that don't warn on Linux's LP64 typedefs).
    # Keep warnings non-fatal here rather than patching dozens of upstream files;
    # Linux retains FatalWarnings. (Mirrors mainline's macos-moltenvk.)
    target_compile_options(${target} PRIVATE -Werror)
  endif()
endfunction()

# xe_shader_rules_spirv(target shader_dir)
#
# Ensures SPIR-V shaders are compiled before the target builds.
# Uses a stamp file; rebuilds when any shader source changes.
function(xe_shader_rules_spirv target shader_dir)
  get_filename_component(shader_dir "${shader_dir}" ABSOLUTE)
  file(GLOB _sources
    "${shader_dir}/*.xesl" "${shader_dir}/*.glsl"
    "${shader_dir}/*.xesli" "${shader_dir}/*.glsli")
  set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/${target}_spirv.stamp")
  set(_script "${PROJECT_SOURCE_DIR}/tools/build/compile_shader_spirv.py")
  set(_valid_stages vs hs ds gs ps cs)
  set(_commands)
  set(_bytecode_dir "${shader_dir}/bytecode/vulkan_spirv")
  list(APPEND _commands COMMAND ${CMAKE_COMMAND} -E make_directory "${_bytecode_dir}")
  foreach(src ${_sources})
    get_filename_component(_name ${src} NAME)
    string(REGEX REPLACE "\\.[^.]+$" "" _basename "${_name}")
    string(REPLACE "." "_" _id "${_basename}")
    string(LENGTH "${_id}" _len)
    if(_len LESS 3)
      continue()
    endif()
    math(EXPR _s "${_len} - 2")
    string(SUBSTRING "${_id}" ${_s} 2 _stage)
    if(NOT _stage IN_LIST _valid_stages)
      continue()
    endif()
    list(APPEND _commands COMMAND ${Python3_EXECUTABLE} "${_script}" "${src}" "${_bytecode_dir}/${_id}.h")
  endforeach()
  add_custom_command(
    OUTPUT "${_stamp}"
    ${_commands}
    COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
    DEPENDS ${_sources} "${_script}"
    COMMENT "Compiling SPIR-V shaders for ${target}..."
    VERBATIM
  )
  add_custom_target(${target}-spirv-shaders DEPENDS "${_stamp}")
  add_dependencies(${target} ${target}-spirv-shaders)
  # Add sources for IDE visibility but prevent VS from compiling them itself
  set_source_files_properties(${_sources} PROPERTIES HEADER_FILE_ONLY TRUE)
  target_sources(${target} PRIVATE ${_sources})
endfunction()

# xe_shader_rules_metal(target shader_dir [source_glob...])
#
# Ensures Metal shaders are compiled before the target builds (Apple hosts
# only; no-op elsewhere). Compiles the MSL branch of the shared polyglot XeSL
# sources via tools/build/compile_shader_metal.py into
# ${shader_dir}/bytecode/metal/<id>.h (`const uint8_t <id>_metallib[]`),
# mirroring the SPIR-V rule's generated-not-committed convention.
#
# Optional trailing arguments are glob patterns, relative to shader_dir,
# selecting which sources to compile. They default to the texture-load compute
# family this rule originally served, so existing callers are unaffected. The
# compiler itself is stage-agnostic -- it invokes `xcrun metal` on the source
# and derives the identifier from the filename -- so vertex/fragment sources
# (e.g. the ui/shaders immediate.vs/.ps pair the UI overlay drawer consumes)
# compile through the same path with no tool change.
#
# Unlike the SPIR-V rule, the compile work is owned by ONE global
# custom target per shader directory, and consumers only add a dependency on
# it -- two consumers must never race writing the same generated headers.
# Ownership is therefore per-directory: the first registration for a given
# directory defines that directory's source set, so callers must not register
# the same directory with differing globs.
# The stamp depends on the selected sources AND every .xesli in both the shader
# dir and ui/shaders (cross-directory includes like xesl.xesli).
function(xe_shader_rules_metal target shader_dir)
  if(NOT APPLE)
    return()
  endif()
  get_filename_component(shader_dir "${shader_dir}" ABSOLUTE)
  set(_globs ${ARGN})
  if(NOT _globs)
    # BOTH shared compute families the Metal backend consumes. This is the
    # DEFAULT rather than a per-caller glob on purpose: `xenia-gpu-metal-backend`
    # and `xenia-gpu-conformance` both register the same shader directory, and
    # this rule's ownership model says the first registration defines that
    # directory's source set -- so a glob added at one call site only would
    # silently depend on configure order.
    set(_globs "texture_load_*.cs.xesl" "resolve_*.cs.xesl")
  endif()
  string(SHA1 _dir_hash "${shader_dir}")
  string(SUBSTRING "${_dir_hash}" 0 8 _dir_hash)
  set(_owner_target xe-metal-shaders-${_dir_hash})
  if(NOT TARGET ${_owner_target})
    set(_source_patterns)
    foreach(_glob ${_globs})
      list(APPEND _source_patterns "${shader_dir}/${_glob}")
    endforeach()
    file(GLOB _sources ${_source_patterns})
    file(GLOB _includes
      "${shader_dir}/*.xesli"
      "${PROJECT_SOURCE_DIR}/src/xenia/ui/shaders/*.xesli")
    set(_stamp "${PROJECT_BINARY_DIR}/${_owner_target}.stamp")
    set(_script "${PROJECT_SOURCE_DIR}/tools/build/compile_shader_metal.py")
    set(_bytecode_dir "${shader_dir}/bytecode/metal")
    set(_commands)
    list(APPEND _commands
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_bytecode_dir}")
    foreach(src ${_sources})
      get_filename_component(_name ${src} NAME)
      string(REGEX REPLACE "\\.[^.]+$" "" _basename "${_name}")
      string(REPLACE "." "_" _id "${_basename}")
      list(APPEND _commands
        COMMAND ${Python3_EXECUTABLE} "${_script}" "${src}"
                "${_bytecode_dir}/${_id}.h")
    endforeach()
    add_custom_command(
      OUTPUT "${_stamp}"
      ${_commands}
      COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
      DEPENDS ${_sources} ${_includes} "${_script}"
      COMMENT "Compiling Metal shaders (${shader_dir})..."
      VERBATIM
    )
    add_custom_target(${_owner_target} DEPENDS "${_stamp}")
  endif()
  add_dependencies(${target} ${_owner_target})
endfunction()

# xe_force_c(files...)
#
# Forces the given source files to compile as C.
function(xe_force_c)
  set_source_files_properties(${ARGN} PROPERTIES LANGUAGE C)
endfunction()

# xe_force_cxx(files...)
#
# Forces the given source files to compile as C++.
function(xe_force_cxx)
  set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CXX)
endfunction()

# xe_test_suite(name base_path LINKS lib1 lib2 ...)
#
# Creates a Catch2 test executable from *_test.cc files in base_path.
function(xe_test_suite name base_path)
  cmake_parse_arguments(ARG "" "" "LINKS" ${ARGN})

  file(GLOB _test_sources "${base_path}/*_test.cc")

  if(NOT _test_sources)
    return()
  endif()

  add_executable(${name}
    ${_test_sources}
    ${PROJECT_SOURCE_DIR}/tools/build/src/test_suite_main.cc
  )

  # Add platform-specific console app main
  target_sources(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_posix.cc)

  target_compile_definitions(${name} PRIVATE
    "XE_TEST_SUITE_NAME=\"${name}\""
  )
  target_include_directories(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/build
    ${PROJECT_SOURCE_DIR}/tools/build/src
    ${PROJECT_SOURCE_DIR}/tools/build/third_party/catch/include
  )
  if(ARG_LINKS)
    target_link_libraries(${name} PRIVATE ${ARG_LINKS})
  endif()
  # The default 64-bit macOS __PAGEZERO is 4 GB, which makes the entire sub-4 GB
  # range unmappable. The x64 JIT code cache reserves fixed low addresses (e.g.
  # 0xA0000000), so x86_64 test suites that initialize a backend must shrink
  # __PAGEZERO exactly like xenia-app does, or code_cache Initialize() fails.
  # arm64 relocates the code cache above 4 GB instead (macOS 26 forbids any
  # sub-default __PAGEZERO), so it must keep the default.
  if(APPLE AND NOT XE_TARGET_AARCH64)
    target_link_options(${name} PRIVATE -Wl,-pagezero_size,0x1000)
  endif()
  xe_target_defaults(${name})
  set_target_properties(${name} PROPERTIES FOLDER "tests")

  catch_discover_tests(${name})
endfunction()
