#!/usr/bin/env bash
#
# package_mac.sh - Assemble a distributable, Developer-ID-signed macOS app bundle
# (Xenia Canary.app) and a .dmg from an already-built bare xenia_canary binary.
#
# This is a release-time step; it does NOT change the dev build. The bare binary
# produced by `cmake --build` keeps working as-is. See
# openspec/changes/add-macos-app-bundle-distribution for the rationale.
#
# Usage:
#   tools/build/package_mac.sh [--config Release|Checked|Debug] [--notarize] [--adhoc]
#
#   --config   Build config to package (default: Release).
#   --notarize Submit to Apple notarytool + staple (default: OFF). Uses
#              APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID from .env.
#   --adhoc    Sign ad-hoc instead of Developer ID (no .env identity required).
#
# Signing config is read from a gitignored .env at the repo root (see .env.example).

set -euo pipefail

# --- Locate repo root (this script lives in tools/build/) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="Release"
DO_NOTARIZE=0
ADHOC=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --notarize) DO_NOTARIZE=1; shift ;;
    --adhoc) ADHOC=1; shift ;;
    *) echo "error: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

APP_NAME="Xenia Canary"
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

log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Preconditions ----------------------------------------------------------
[[ -x "${BIN_PATH}" ]] || die "built binary not found: ${BIN_PATH}
  build it first, e.g.: ninja -f build-${CONFIG}.ninja xenia-app"
[[ -f "${PLIST_IN}" ]] || die "missing Info.plist template: ${PLIST_IN}"
[[ -f "${ENTITLEMENTS}" ]] || die "missing entitlements: ${ENTITLEMENTS}"

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

cp "${BIN_PATH}" "${CONTENTS}/MacOS/${BUNDLE_EXEC}"
[[ -f "${ICON_SRC}" ]] && cp "${ICON_SRC}" "${CONTENTS}/Resources/xenia.icns" \
  || log "warning: no icon at ${ICON_SRC}; bundle will use the generic icon"

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
Xenia Canary ${SHORT_VERSION} (${BRANCH} ${COMMIT_SHORT})
Bundled MoltenVK : ${MVK_SRC} (${MVK_VER})
Bundled loader   : ${VK_LOADER_SRC} (${VKL_VER})
EOF

# Confirm nothing in the bundle points back at an absolute Cellar/host path.
log "Verifying in-bundle dylib references"
if otool -L "${CONTENTS}/MacOS/${BUNDLE_EXEC}" "${CONTENTS}/Frameworks/"*.dylib \
     | grep -E "/usr/local/|/opt/homebrew/|/Cellar/" ; then
  die "bundle still references absolute host paths (see above)"
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
