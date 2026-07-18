#!/usr/bin/env python3

"""Compiles a single XeSL shader to a Metal library and generates a C header.

Usage: compile_shader_metal.py <input_path> <output_path>

Pipeline (Apple hosts only):
  1. xcrun metal -x metal -D SHADING_LANGUAGE_MSL_XE=1 -> .air
  2. xcrun metallib -> .metallib
  3. Generate .h with a `const uint8_t <identifier>_metallib[]` array.

The XeSL sources are polyglot (GLSL/HLSL/MSL selected by the
SHADING_LANGUAGE_*_XE define); this compiles the MSL branch directly with the
Apple toolchain -- no source forking, no DXC/Metal Shader Converter
involvement (those serve the runtime guest-shader path, not static portable
shaders). Output bytes are deterministic for identical inputs and independent
of the source checkout path (verified: identical metallibs from different
worktree locations), but they are still generated per-build like the SPIR-V
headers, not committed.

Intermediates are written next to the output header (not the system temp
dir) with a pid suffix so parallel builds never collide.
"""

import os
import subprocess
import sys

# Matches the CMake default CMAKE_OSX_DEPLOYMENT_TARGET for this project.
MACOS_MIN_VERSION = "15.0"
METAL_STD = "macos-metal2.4"


def compile_metal(input_path, output_path):
    src_name = os.path.basename(input_path)
    # foo.cs.xesl -> foo_cs
    identifier = src_name[:src_name.rfind(".")].replace(".", "_")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tag = "%s_%d" % (identifier, os.getpid())
    air_path = os.path.join(output_dir, tag + ".air.tmp")
    lib_path = os.path.join(output_dir, tag + ".metallib.tmp")

    src_dir = os.path.dirname(os.path.abspath(input_path))

    try:
        result = subprocess.call([
            "xcrun", "-sdk", "macosx", "metal",
            "-x", "metal",
            "-std=" + METAL_STD,
            "-mmacosx-version-min=" + MACOS_MIN_VERSION,
            "-D", "SHADING_LANGUAGE_MSL_XE=1",
            "-w",
            "-I", src_dir,
            "-c", input_path,
            "-o", air_path,
        ])
        if result != 0:
            print("ERROR: metal frontend failed for %s" % input_path,
                  file=sys.stderr)
            return result

        result = subprocess.call([
            "xcrun", "-sdk", "macosx", "metallib", air_path, "-o", lib_path,
        ])
        if result != 0:
            print("ERROR: metallib link failed for %s" % input_path,
                  file=sys.stderr)
            return result

        with open(lib_path, "rb") as f:
            data = f.read()
    finally:
        for tmp in (air_path, lib_path):
            try:
                os.remove(tmp)
            except OSError:
                pass

    with open(output_path, "w") as f:
        f.write("// Generated with `xb buildshaders`.\n")
        f.write("const uint8_t %s_metallib[] = {" % identifier)
        for i, byte in enumerate(data):
            f.write("\n    " if i % 16 == 0 else " ")
            f.write("0x%02X," % byte)
        f.write("\n};\n")
    return 0


def main():
    if len(sys.argv) != 3:
        print("Usage: compile_shader_metal.py <input> <output>",
              file=sys.stderr)
        return 1
    if sys.platform != "darwin":
        print("ERROR: Metal shaders can be compiled only on an Apple host.",
              file=sys.stderr)
        return 1
    return compile_metal(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
