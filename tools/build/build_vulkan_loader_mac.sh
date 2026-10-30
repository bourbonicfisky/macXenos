#!/usr/bin/env bash
# Builds the Khronos Vulkan loader (libvulkan.1.dylib) for the macXenos bundle:
# Universal 2 (arm64 + x86_64) at the app's minimum macOS, so the bundled
# Vulkan fallback loads on every macOS the app supports. Homebrew's bottle is
# built for the macOS it was poured on (macOS 27 on the build machine), which
# is why the bundle can't simply copy it.
#
#   tools/build/build_vulkan_loader_mac.sh [--version v1.4.363] [--min-macos 15.0]
#                                          [--out build/universal-deps/vulkan-loader]
#
# Prints the built dylib's path. package_mac.sh uses it when present.
set -euo pipefail

VERSION="v1.4.363"
MIN_MACOS="15.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT="${ROOT}/build/universal-deps/vulkan-loader"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --min-macos) MIN_MACOS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SRC="${OUT}/src/Vulkan-Loader-${VERSION}"
BUILD="${OUT}/build-${VERSION}"
LIB="${OUT}/lib/libvulkan.1.dylib"

if [[ ! -d "${SRC}" ]]; then
  mkdir -p "${OUT}/src"
  git clone --quiet --depth 1 --branch "${VERSION}" \
    https://github.com/KhronosGroup/Vulkan-Loader.git "${SRC}" >&2
fi

# UPDATE_DEPS fetches the Vulkan-Headers the loader version pins.
cmake -S "${SRC}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MIN_MACOS}" \
  -DUPDATE_DEPS=ON \
  -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="${OUT}" >&2
cmake --build "${BUILD}" >&2
cmake --install "${BUILD}" >&2

# The bundle loads it as @rpath/libvulkan.1.dylib; resolve the versioned name.
REAL="$(readlink -f "${LIB}")"
for arch in arm64 x86_64; do
  minos="$(vtool -arch "${arch}" -show-build "${REAL}" | awk '/minos/ {print $2; exit}')"
  [[ "${minos}" == "${MIN_MACOS}" ]] || {
    echo "${REAL} ${arch} minimum macOS is ${minos}, not ${MIN_MACOS}" >&2
    exit 1
  }
done
echo "${REAL}"
