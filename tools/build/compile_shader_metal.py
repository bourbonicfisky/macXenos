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
import re
import subprocess
import sys

# Matches the CMake default CMAKE_OSX_DEPLOYMENT_TARGET for this project.
MACOS_MIN_VERSION = "15.0"
METAL_STD = "macos-metal2.4"

# HLSL parameter qualifiers (`in T x`, `out T x`, `inout T x`), as used by the
# AMD FidelityFX headers compiled through their HLSL dialect
# (amd_language.xesli). MSL has no such qualifiers: `out` / `inout` become
# thread references and `in` is dropped. The lookahead keeps this to parameter
# declarations, which cannot otherwise contain these words in valid MSL.
HLSL_PARAM_QUALIFIER = re.compile(
    r"\b(inout|out|in)\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)(?=\s*[,)])")


# ffx_a.h's unconditional AZolZeroPassF2/3/4 use an HLSL component-wise `?:`
# on vectors, which MSL rejects (a bool vector condition can't select uint
# vectors). Neither FSR nor CAS calls them; they're redefined with select().
FFX_ZOL_ZERO_PASS_VECTOR = re.compile(
    r"^(\s*)(float([234])) AZolZeroPassF\3\(.*$", re.MULTILINE)


def rewrite_hlsl_param_qualifiers(source):
    def replace(match):
        qualifier, type_name, name = match.groups()
        if qualifier == "in":
            return "%s %s" % (type_name, name)
        return "thread %s& %s" % (type_name, name)
    source, count = HLSL_PARAM_QUALIFIER.subn(replace, source)

    def replace_zol(match):
        indent, float_type, n = match.groups()
        return ("%s%s AZolZeroPassF%s(%s x,%s y){return as_type<%s>(select("
                "as_type<uint%s>(y),uint%s(0u),as_type<uint%s>(x)!=0u));}" %
                (indent, float_type, n, float_type, float_type, float_type,
                 n, n, n))
    source, zol_count = FFX_ZOL_ZERO_PASS_VECTOR.subn(replace_zol, source)
    return source, count + zol_count


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

    metal_command = [
        "xcrun", "-sdk", "macosx", "metal",
        "-x", "metal",
        "-std=" + METAL_STD,
        "-mmacosx-version-min=" + MACOS_MIN_VERSION,
        "-D", "SHADING_LANGUAGE_MSL_XE=1",
        "-w",
        "-I", src_dir,
    ]
    rewritten_path = os.path.join(output_dir, tag + ".metal.tmp")

    try:
        # Sources using HLSL parameter qualifiers are compiled from their
        # rewritten preprocessed text (without metal_stdlib, which xesl.xesli
        # skips under XE_MSL_PREPROCESS_ONLY and is prepended back). Every
        # other source compiles from the original file as before.
        compile_input = input_path
        preprocessed = subprocess.run(
            metal_command + ["-D", "XE_MSL_PREPROCESS_ONLY=1", "-E", "-P",
                             input_path],
            stdout=subprocess.PIPE, universal_newlines=True)
        if preprocessed.returncode != 0:
            print("ERROR: metal preprocessor failed for %s" % input_path,
                  file=sys.stderr)
            return preprocessed.returncode
        rewritten, rewrite_count = rewrite_hlsl_param_qualifiers(
            preprocessed.stdout)
        if rewrite_count:
            with open(rewritten_path, "w") as f:
                f.write("#include <metal_stdlib>\nusing namespace metal;\n")
                f.write(rewritten)
            compile_input = rewritten_path

        result = subprocess.call(
            metal_command + ["-c", compile_input, "-o", air_path])
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
        for tmp in (air_path, lib_path, rewritten_path):
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
