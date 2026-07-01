#!/usr/bin/env bash
#
# package_mac.sh - Assemble a distributable, Developer-ID-signed macOS app bundle
# (macXenos.app) and a .dmg from an already-built bare xenia_canary binary.
#
# This is a release-time step; it does NOT change the dev build. The bare binary
# produced by `cmake --build` keeps working as-is. See
# openspec/changes/add-macos-app-bundle-distribution for the rationale.
#
# Usage:
#   tools/build/package_mac.sh [--config Release|Checked|Debug] [--notarize] [--adhoc]
#                              [--x86_64 <bin>] [--x86_64-libs <prefix>]
#
#   --config       Build config to package (default: Release).
#   --notarize     Submit to Apple notarytool + staple (default: OFF). Uses
#                  APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID from .env.
#   --adhoc        Sign ad-hoc instead of Developer ID (no .env identity required).
#   --x86_64 <bin> Build a Universal 2 bundle: lipo the natively-built slice in
#                  build/bin/... together with this prebuilt x86_64 xenia_canary.
#                  (The x86_64 slice must be built on/for an Intel toolchain — it
#                  cannot be produced from an arm64-only Homebrew install.)
#   --x86_64-libs <prefix>
#                  Homebrew prefix holding the x86_64 libMoltenVK/libvulkan
#                  (default: /usr/local). Only used with --x86_64; the bundled
#                  dylibs are lipo'd so the fat executable has matching fat deps.
#
# Without --x86_64 the bundle is single-arch (whatever was built natively) — the
# original behavior. codesign signs every slice of a fat binary in one pass.
#
# Signing config is read from a gitignored .env at the repo root (see .env.example).

set -euo pipefail

# --- Locate repo root (this script lives in tools/build/) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="Release"
DO_NOTARIZE=0
ADHOC=0
X86_BIN=""
X86_LIB_PREFIX="/usr/local"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --notarize) DO_NOTARIZE=1; shift ;;
    --adhoc) ADHOC=1; shift ;;
    --x86_64) X86_BIN="$2"; shift 2 ;;
    --x86_64-libs) X86_LIB_PREFIX="$2"; shift 2 ;;
    *) echo "error: unknown arg '$1'" >&2; exit 2 ;;
  esac
done
UNIVERSAL=0
[[ -n "${X86_BIN}" ]] && UNIVERSAL=1

APP_NAME="macXenos"
BUNDLE_EXEC="xenia_canary"
BIN_DIR="${REPO_ROOT}/build/bin/macOS/${CONFIG}"
BIN_PATH="${BIN_DIR}/${BUNDLE_EXEC}"
VERSION_H="${REPO_ROOT}/build/version.h"
DIST_DIR="${REPO_ROOT}/build/dist"
APP_DIR="${DIST_DIR}/${APP_NAME}.app"
CONTENTS="${APP_DIR}/Contents"
PLIST_IN="${REPO_ROOT}/src/xenia/app/Info.plist.in"
ENTITLEMENTS="${REPO_ROOT}/src/xenia/app/xenia.dist.entitlements"
ICON_SRC="${REPO_ROOT}/src/xenia/app/xenia.icns"
CREDITS_SRC="${REPO_ROOT}/src/xenia/app/Credits.rtf"

log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Preconditions ----------------------------------------------------------
[[ -x "${BIN_PATH}" ]] || die "built binary not found: ${BIN_PATH}
  build it first, e.g.: ninja -f build-${CONFIG}.ninja xenia-app"
[[ -f "${PLIST_IN}" ]] || die "missing Info.plist template: ${PLIST_IN}"
[[ -f "${ENTITLEMENTS}" ]] || die "missing entitlements: ${ENTITLEMENTS}"
if [[ "${UNIVERSAL}" -eq 1 ]]; then
  [[ -f "${X86_BIN}" ]] || die "x86_64 binary not found: ${X86_BIN} (--x86_64)"
  # Sanity-check the two inputs really are the two arches we expect.
  lipo -archs "${BIN_PATH}"  | grep -qw x86_64 && die "native build at ${BIN_PATH} already contains x86_64 — expected the arm64 slice"
  lipo -archs "${X86_BIN}"   | grep -qw x86_64 || die "--x86_64 binary ${X86_BIN} is not x86_64 (got: $(lipo -archs "${X86_BIN}"))"
fi

# --- Version strings from build/version.h -----------------------------------
get_define() { sed -n "s/^#define $1 \"\\(.*\\)\"/\\1/p" "${VERSION_H}" 2>/dev/null | head -1; }
COMMIT_SHORT="$(get_define XE_BUILD_COMMIT_SHORT)"; COMMIT_SHORT="${COMMIT_SHORT:-unknown}"
BRANCH="$(get_define XE_BUILD_BRANCH)"; BRANCH="${BRANCH:-unknown}"
SHORT_VERSION="1.0-canary"
BUILD_VERSION="${COMMIT_SHORT}"

# ===========================================================================
# 1. Assemble the bundle skeleton
# ===========================================================================
log "Assembling ${APP_NAME}.app (config=${CONFIG}, commit=${COMMIT_SHORT})"
rm -rf "${APP_DIR}"
mkdir -p "${CONTENTS}/MacOS" "${CONTENTS}/Frameworks" \
         "${CONTENTS}/Resources/vulkan/icd.d"

if [[ "${UNIVERSAL}" -eq 1 ]]; then
  log "Universal 2: lipo arm64 ($(lipo -archs "${BIN_PATH}")) + x86_64 executable"
  lipo -create "${BIN_PATH}" "${X86_BIN}" -output "${CONTENTS}/MacOS/${BUNDLE_EXEC}"
else
  cp "${BIN_PATH}" "${CONTENTS}/MacOS/${BUNDLE_EXEC}"
fi
[[ -f "${ICON_SRC}" ]] && cp "${ICON_SRC}" "${CONTENTS}/Resources/xenia.icns" \
  || log "warning: no icon at ${ICON_SRC}; bundle will use the generic icon"
# Credits.rtf is auto-loaded by the standard About panel (Patreon/repo links,
# "Ported by dmg"). RTF (vs HTML) leaves body text uncolored so it stays
# readable in both light and dark mode.
[[ -f "${CREDITS_SRC}" ]] && cp "${CREDITS_SRC}" "${CONTENTS}/Resources/Credits.rtf" \
  || log "warning: no Credits.rtf at ${CREDITS_SRC}; About panel will show no credits"

# Info.plist from template with version substitution.
sed -e "s/@SHORT_VERSION@/${SHORT_VERSION}/g" \
    -e "s/@BUILD_VERSION@/${BUILD_VERSION}/g" \
    "${PLIST_IN}" > "${CONTENTS}/Info.plist"

# ===========================================================================
# 2. Bundle the runtime Vulkan stack (the only non-system runtime dep)
# ===========================================================================
log "Bundling Vulkan loader + MoltenVK"
# Resolve the real (symlink-dereferenced) dylibs the loader dlopens.
resolve() { local p; for p in "$@"; do [[ -e "$p" ]] && { readlink -f "$p"; return 0; }; done; return 1; }
VK_LOADER_SRC="$(resolve /usr/local/lib/libvulkan.1.dylib /opt/homebrew/lib/libvulkan.1.dylib)" \
  || die "libvulkan.1.dylib not found (install vulkan-loader via Homebrew)"
MVK_SRC="$(resolve /usr/local/lib/libMoltenVK.dylib /opt/homebrew/lib/libMoltenVK.dylib)" \
  || die "libMoltenVK.dylib not found (install molten-vk via Homebrew)"

cp "${VK_LOADER_SRC}" "${CONTENTS}/Frameworks/libvulkan.1.dylib"
cp "${MVK_SRC}"       "${CONTENTS}/Frameworks/libMoltenVK.dylib"
chmod u+w "${CONTENTS}/Frameworks/"*.dylib

# For a Universal 2 bundle the bundled dylibs must also carry both slices, or
# the x86_64 executable slice will fail to load them. lipo in the x86_64 build
# of each dylib from --x86_64-libs (skip any that is already fat).
if [[ "${UNIVERSAL}" -eq 1 ]]; then
  universalize_dylib() {
    local name="$1" dst="${CONTENTS}/Frameworks/$1"
    if lipo -archs "${dst}" | grep -qw x86_64; then
      log "  ${name}: already universal ($(lipo -archs "${dst}"))"; return 0
    fi
    local x86src
    x86src="$(resolve "${X86_LIB_PREFIX}/lib/${name}")" \
      || die "x86_64 ${name} not found under ${X86_LIB_PREFIX}/lib (need it to build a fat bundle; set --x86_64-libs)"
    lipo -archs "${x86src}" | grep -qw x86_64 \
      || die "${x86src} is not x86_64 (got: $(lipo -archs "${x86src}"))"
    lipo -create "${dst}" "${x86src}" -output "${dst}"
    log "  ${name}: lipo'd → $(lipo -archs "${dst}")"
  }
  log "Universal 2: making bundled dylibs fat (x86_64 libs from ${X86_LIB_PREFIX})"
  universalize_dylib libvulkan.1.dylib
  universalize_dylib libMoltenVK.dylib
fi

# Self-referential install names; loader resolves them via the bundle layout
# (OpenVulkanLoaderMac looks in ../Frameworks and ../Resources/vulkan/icd.d).
install_name_tool -id "@rpath/libvulkan.1.dylib"  "${CONTENTS}/Frameworks/libvulkan.1.dylib"
install_name_tool -id "@rpath/libMoltenVK.dylib"  "${CONTENTS}/Frameworks/libMoltenVK.dylib"
# Ensure the executable can resolve @rpath against its bundled Frameworks.
install_name_tool -add_rpath "@executable_path/../Frameworks" \
  "${CONTENTS}/MacOS/${BUNDLE_EXEC}" 2>/dev/null || true

# ICD manifest pointing at the bundled MoltenVK (relative: icd.d -> Frameworks).
cat > "${CONTENTS}/Resources/vulkan/icd.d/MoltenVK_icd.json" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "../../../Frameworks/libMoltenVK.dylib",
        "api_version" : "1.4.0",
        "is_portability_driver" : true
    }
}
JSON

# Provenance: record what we bundled.
MVK_VER="$(basename "$(dirname "$(dirname "${MVK_SRC}")")")"
VKL_VER="$(basename "$(dirname "$(dirname "${VK_LOADER_SRC}")")")"
cat > "${CONTENTS}/Resources/VERSIONS.txt" <<EOF
macXenos (unofficial macOS port) — Xenia Canary ${SHORT_VERSION} (${BRANCH} ${COMMIT_SHORT})
Bundled MoltenVK : ${MVK_SRC} (${MVK_VER})
Bundled loader   : ${VK_LOADER_SRC} (${VKL_VER})
EOF

# Confirm nothing in the bundle points back at an absolute Cellar/host path.
log "Verifying in-bundle dylib references"
if otool -L "${CONTENTS}/MacOS/${BUNDLE_EXEC}" "${CONTENTS}/Frameworks/"*.dylib \
     | grep -E "/usr/local/|/opt/homebrew/|/Cellar/" ; then
  die "bundle still references absolute host paths (see above)"
fi

# Report (and, for --x86_64, enforce) the slices in the shipping binaries.
log "Bundle architectures:"
for f in "${CONTENTS}/MacOS/${BUNDLE_EXEC}" "${CONTENTS}/Frameworks/"*.dylib; do
  printf '    %-22s %s\n' "$(basename "$f")" "$(lipo -archs "$f")"
done
if [[ "${UNIVERSAL}" -eq 1 ]]; then
  for f in "${CONTENTS}/MacOS/${BUNDLE_EXEC}" "${CONTENTS}/Frameworks/"*.dylib; do
    archs="$(lipo -archs "$f")"
    grep -qw arm64 <<<"${archs}" && grep -qw x86_64 <<<"${archs}" \
      || die "Universal 2 requested but $(basename "$f") is not fat (arm64+x86_64): ${archs}"
  done
  log "Universal 2 verified: all shipping binaries are arm64 + x86_64"
fi

# ===========================================================================
# 3. Sign (frameworks first, then deep-sign the app) + DMG
# ===========================================================================
SIGN_ARGS=(--force --timestamp --options runtime --entitlements "${ENTITLEMENTS}")
if [[ "${ADHOC}" -eq 1 ]]; then
  IDENTITY="-"
  log "Signing AD-HOC (--adhoc): no Developer ID"
else
  # Load .env (gitignored) for the Developer ID identity.
  [[ -f "${REPO_ROOT}/.env" ]] && set -a && . "${REPO_ROOT}/.env" && set +a
  IDENTITY="${APPLE_SIGNING_IDENTITY:-}"
  [[ -n "${IDENTITY}" ]] || die "APPLE_SIGNING_IDENTITY unset.
  Add it to ${REPO_ROOT}/.env (see .env.example), or pass --adhoc to sign ad-hoc."
  log "Signing with: ${IDENTITY}"
fi

# Sign nested dylibs first.
codesign "${SIGN_ARGS[@]}" -s "${IDENTITY}" "${CONTENTS}/Frameworks/libMoltenVK.dylib"
codesign "${SIGN_ARGS[@]}" -s "${IDENTITY}" "${CONTENTS}/Frameworks/libvulkan.1.dylib"
# Then deep-sign the app.
codesign "${SIGN_ARGS[@]}" --deep -s "${IDENTITY}" "${APP_DIR}"

log "Verifying signature"
codesign --verify --deep --strict --verbose=2 "${APP_DIR}"
codesign -dvv "${APP_DIR}" 2>&1 | grep -E "Authority|TeamIdentifier|flags|Identifier" || true

# Optional notarization (default OFF).
if [[ "${DO_NOTARIZE}" -eq 1 ]]; then
  [[ "${ADHOC}" -eq 1 ]] && die "--notarize requires Developer ID signing (not --adhoc)"
  : "${APPLE_ID:?APPLE_ID required for --notarize (set in .env)}"
  : "${APPLE_PASSWORD:?APPLE_PASSWORD required for --notarize (set in .env)}"
  : "${APPLE_TEAM_ID:?APPLE_TEAM_ID required for --notarize (set in .env)}"
  log "Notarizing (this can take minutes)"
  NOTARIZE_ZIP="${DIST_DIR}/${APP_NAME}.notarize.zip"
  ditto -c -k --keepParent "${APP_DIR}" "${NOTARIZE_ZIP}"
  xcrun notarytool submit "${NOTARIZE_ZIP}" \
    --apple-id "${APPLE_ID}" --password "${APPLE_PASSWORD}" \
    --team-id "${APPLE_TEAM_ID}" --wait
  xcrun stapler staple "${APP_DIR}"
  rm -f "${NOTARIZE_ZIP}"
else
  log "Skipping notarization (default). Other Macs will need: xattr -dr com.apple.quarantine"
fi

# ---------------------------------------------------------------------------
# 4. DMG with an /Applications symlink for drag-install.
# ---------------------------------------------------------------------------
log "Building DMG"
DMG_PATH="${DIST_DIR}/${APP_NAME}.dmg"
STAGE="${DIST_DIR}/dmg_stage"
rm -rf "${STAGE}" "${DMG_PATH}"
mkdir -p "${STAGE}"
cp -R "${APP_DIR}" "${STAGE}/"
ln -s /Applications "${STAGE}/Applications"
hdiutil create -volname "${APP_NAME}" -srcfolder "${STAGE}" \
  -ov -format UDZO "${DMG_PATH}" >/dev/null
rm -rf "${STAGE}"

log "Done:"
echo "  app: ${APP_DIR}"
echo "  dmg: ${DMG_PATH}"
