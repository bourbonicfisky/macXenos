#!/usr/bin/env bash
#
# fetch_sparkle.sh - Download the pinned Sparkle release (app updates) into
# build/sparkle/Sparkle-<version>/ and verify it. package_mac.sh bundles its
# Sparkle.framework; its bin/ holds sign_update, generate_appcast and
# generate_keys (see sparkle_appcast.sh).
#
# Usage: tools/build/fetch_sparkle.sh        prints the extracted folder
#
# To move to a newer Sparkle, change SPARKLE_VERSION and SPARKLE_SHA256 (the
# sha256 of Sparkle-<version>.tar.xz from the GitHub release).

set -euo pipefail

SPARKLE_VERSION="2.10.0"
SPARKLE_SHA256="c2bf58aa8387266ac179357b1415d6f2635f044da8be41042af32425dae6da0c"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${REPO_ROOT}/build/sparkle/Sparkle-${SPARKLE_VERSION}"
ARCHIVE="${DEST}/Sparkle-${SPARKLE_VERSION}.tar.xz"
OUT="${DEST}/x"

if [[ -d "${OUT}/Sparkle.framework" ]]; then
  echo "${OUT}"
  exit 0
fi

mkdir -p "${DEST}"
if [[ ! -f "${ARCHIVE}" ]]; then
  curl -fsSL -o "${ARCHIVE}.part" \
    "https://github.com/sparkle-project/Sparkle/releases/download/${SPARKLE_VERSION}/Sparkle-${SPARKLE_VERSION}.tar.xz"
  mv "${ARCHIVE}.part" "${ARCHIVE}"
fi
actual="$(shasum -a 256 "${ARCHIVE}" | cut -d' ' -f1)"
if [[ "${actual}" != "${SPARKLE_SHA256}" ]]; then
  echo "error: ${ARCHIVE} sha256 ${actual}, expected ${SPARKLE_SHA256}" >&2
  rm -f "${ARCHIVE}"
  exit 1
fi
rm -rf "${OUT}.part"
mkdir -p "${OUT}.part"
tar -xJf "${ARCHIVE}" -C "${OUT}.part"
mv "${OUT}.part" "${OUT}"
echo "${OUT}"
