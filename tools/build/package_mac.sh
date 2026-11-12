#!/usr/bin/env bash
#
# package_mac.sh - Assemble a distributable, Developer-ID-signed macOS app bundle
# (macXenos.app) and a .dmg from an already-built bare macXenos binary.
#
# This is a release-time step; it does NOT change the dev build. The bare binary
# produced by `cmake --build` keeps working as-is. See
# openspec/changes/add-macos-app-bundle-distribution for the rationale.
#
# Usage:
#   tools/build/package_mac.sh [--config Release|Checked|Debug] [--notarize] [--adhoc]
#                              [--version <str>] [--x86_64 <bin>] [--x86_64-libs <prefix>]
#                              [--feed-url <url>] [--no-sparkle]
#
#   --config       Build config to package (default: Release).
#   --version      CFBundleShortVersionString. Default: from the release tags
#                  of the built commit (generate_version_header.py): "1.0" on
#                  the macxenos-1.0 tag, "1.0+3" three commits after it.
#                  Give it only for a test build, e.g. 1.0-beta9-test6.
#                  CFBundleVersion is the built commit's count (git rev-list
#                  --count): Sparkle orders releases by it, so it only grows.
#   --feed-url     The Sparkle appcast the app checks for updates (default:
#                  https://macxenos.com/appcast.xml). See sparkle_appcast.sh.
#   --no-sparkle   Leave the updater out (no Sparkle.framework, no feed): the
#                  app then has no Check for Updates and never looks.
#   --notarize     Submit to Apple notarytool + staple (default: OFF). Uses
#                  APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID from .env.
#   --adhoc        Sign ad-hoc instead of Developer ID (no .env identity required).
#   --x86_64 <bin> Build a Universal 2 bundle: lipo the natively-built slice in
#                  build/bin/... together with this prebuilt x86_64 macXenos.
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
FEED_URL="https://macxenos.com/appcast.xml"
WITH_SPARKLE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --notarize) DO_NOTARIZE=1; shift ;;
    --adhoc) ADHOC=1; shift ;;
    --version) SHORT_VERSION_OVERRIDE="$2"; shift 2 ;;
    --x86_64) X86_BIN="$2"; shift 2 ;;
    --x86_64-libs) X86_LIB_PREFIX="$2"; shift 2 ;;
    --feed-url) FEED_URL="$2"; shift 2 ;;
    --no-sparkle) WITH_SPARKLE=0; shift ;;
    *) echo "error: unknown arg '$1'" >&2; exit 2 ;;
  esac
done
UNIVERSAL=0
[[ -n "${X86_BIN}" ]] && UNIVERSAL=1

APP_NAME="macXenos"
BUNDLE_EXEC="macXenos"
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
COMMIT_FULL="$(get_define XE_BUILD_COMMIT)"
# The version the built commit's release tags give it (macxenos-1.0 -> 1.0,
# later commits 1.0+N), the same rule the app's version.h follows.
TAG_VERSION="$(MACXENOS_VERSION= python3 "${REPO_ROOT}/tools/build/generate_version_header.py" \
  --source-root "${REPO_ROOT}" --print-version "${COMMIT_FULL:-HEAD}" 2>/dev/null | cut -d' ' -f1)"
SHORT_VERSION="${SHORT_VERSION_OVERRIDE:-${TAG_VERSION:-dev}}"
# Sparkle compares CFBundleVersion to decide what is newer, so it must grow
# with every release: the built commit's position in history. (It was the
# short hash, which has no order.)
BUILD_VERSION="$(git -C "${REPO_ROOT}" rev-list --count "${COMMIT_FULL:-HEAD}" 2>/dev/null)" \
  || die "cannot count commits up to ${COMMIT_FULL:-HEAD} for CFBundleVersion"

# ===========================================================================
# 1. Assemble the bundle skeleton
# ===========================================================================
log "Assembling ${APP_NAME}.app (config=${CONFIG}, commit=${COMMIT_SHORT})"
rm -rf "${APP_DIR}"
mkdir -p "${CONTENTS}/MacOS" "${CONTENTS}/Frameworks" \
         "${CONTENTS}/Resources/vulkan/icd.d"

# Drop every absolute LC_RPATH the dev build baked in (they point into this
# checkout); the bundle resolves @rpath through ../Frameworks only. Done per
# thin slice before lipo: install_name_tool refuses to delete an rpath that
# is not present in every architecture of a fat file.
strip_abs_rpaths() {
  local bin="$1" rp
  while read -r rp; do
    [[ "${rp}" == /* ]] || continue
    install_name_tool -delete_rpath "${rp}" "${bin}"
  done < <(otool -l "${bin}" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}')
}
SLICE_TMP="${DIST_DIR}/slices"
rm -rf "${SLICE_TMP}"; mkdir -p "${SLICE_TMP}"
cp "${BIN_PATH}" "${SLICE_TMP}/arm64"; chmod u+w "${SLICE_TMP}/arm64"
strip_abs_rpaths "${SLICE_TMP}/arm64"
if [[ "${UNIVERSAL}" -eq 1 ]]; then
  log "Universal 2: lipo arm64 ($(lipo -archs "${BIN_PATH}")) + x86_64 executable"
  cp "${X86_BIN}" "${SLICE_TMP}/x86_64"; chmod u+w "${SLICE_TMP}/x86_64"
  strip_abs_rpaths "${SLICE_TMP}/x86_64"
  lipo -create "${SLICE_TMP}/arm64" "${SLICE_TMP}/x86_64" -output "${CONTENTS}/MacOS/${BUNDLE_EXEC}"
else
  cp "${SLICE_TMP}/arm64" "${CONTENTS}/MacOS/${BUNDLE_EXEC}"
fi
rm -rf "${SLICE_TMP}"
[[ -f "${ICON_SRC}" ]] && cp "${ICON_SRC}" "${CONTENTS}/Resources/xenia.icns" \
  || log "warning: no icon at ${ICON_SRC}; bundle will use the generic icon"
# Credits.rtf is auto-loaded by the standard About panel (Patreon/repo links,
# "Ported by dmg"). RTF (vs HTML) leaves body text uncolored so it stays
# readable in both light and dark mode.
[[ -f "${CREDITS_SRC}" ]] && cp "${CREDITS_SRC}" "${CONTENTS}/Resources/Credits.rtf" \
  || log "warning: no Credits.rtf at ${CREDITS_SRC}; About panel will show no credits"
# The Preferences blades (src/xenia/app/blades_ui, built into its dist/) and
# the built-in per-title configs, both read from Contents/Resources by the
# binary (BladesOverlay::FindPageRoot, config::LoadGameConfig).
BLADES_SRC="${REPO_ROOT}/src/xenia/app/blades_ui/dist"
if [[ -f "${BLADES_SRC}/index.html" ]]; then
  cp -R "${BLADES_SRC}" "${CONTENTS}/Resources/blades"
else
  log "warning: no built Preferences page at ${BLADES_SRC}; run npm run build in src/xenia/app/blades_ui"
fi
cp -R "${REPO_ROOT}/docs/game_configs" "${CONTENTS}/Resources/game_configs"
cp -R "${REPO_ROOT}/docs/game_patches" "${CONTENTS}/Resources/game_patches"

# Info.plist from template with version substitution.
sed -e "s/@SHORT_VERSION@/${SHORT_VERSION}/g" \
    -e "s/@BUILD_VERSION@/${BUILD_VERSION}/g" \
    "${PLIST_IN}" > "${CONTENTS}/Info.plist"

# ===========================================================================
# 1b. App updates: Sparkle.framework and its feed. The app loads the framework
#     at run time (src/xenia/ui/updater_mac.mm) only when the bundle carries
#     it and names a feed. Sparkle's own first-check prompt is off: the
#     dashboard asks on first launch instead (check_for_updates_asked).
# ===========================================================================
if [[ "${WITH_SPARKLE}" -eq 1 ]]; then
  SPARKLE_DIR="$("${SCRIPT_DIR}/fetch_sparkle.sh")" || die "could not fetch Sparkle"
  ED_KEY_FILE="${SCRIPT_DIR}/sparkle_public_ed_key.txt"
  ED_KEY="$(tr -d '[:space:]' < "${ED_KEY_FILE}" 2>/dev/null)"
  [[ -n "${ED_KEY}" ]] || die "no Sparkle public key in ${ED_KEY_FILE}"
  log "Bundling Sparkle ($(basename "$(dirname "${SPARKLE_DIR}")")), feed ${FEED_URL}"
  ditto "${SPARKLE_DIR}/Sparkle.framework" "${CONTENTS}/Frameworks/Sparkle.framework"
  # The XPC services exist for sandboxed apps; this one is not sandboxed.
  rm -rf "${CONTENTS}/Frameworks/Sparkle.framework/Versions/B/XPCServices" \
         "${CONTENTS}/Frameworks/Sparkle.framework/XPCServices"
  PB=/usr/libexec/PlistBuddy
  "${PB}" -c "Add :SUFeedURL string ${FEED_URL}" "${CONTENTS}/Info.plist"
  "${PB}" -c "Add :SUPublicEDKey string ${ED_KEY}" "${CONTENTS}/Info.plist"
  "${PB}" -c "Add :SUEnableAutomaticChecks bool false" "${CONTENTS}/Info.plist"
  "${PB}" -c "Add :SUVerifyUpdateBeforeExtraction bool true" "${CONTENTS}/Info.plist"
else
  log "Leaving out Sparkle (--no-sparkle): no app updates"
fi

# ===========================================================================
# 2. Bundle the runtime Vulkan stack (the only non-system runtime dep)
# ===========================================================================
log "Bundling Vulkan loader + MoltenVK"
# Resolve the real (symlink-dereferenced) dylibs the loader dlopens.
resolve() { local p; for p in "$@"; do [[ -e "$p" ]] && { readlink -f "$p"; return 0; }; done; return 1; }
# Apple Silicon Homebrew first: a Mac migrated from Intel can still hold a
# stale x86_64-only Homebrew in /usr/local (the M1 Max had MoltenVK 1.2.0
# from 2022 there, which beat 1.4.2 under /opt/homebrew).
# The loader built by build_vulkan_loader_mac.sh comes first: Universal 2 at
# the app's minimum macOS. Homebrew's bottle targets the macOS it was poured
# on, which the deployment-target check below refuses when that is newer.
VK_LOADER_SRC="$(resolve "${REPO_ROOT}/build/universal-deps/vulkan-loader/lib/libvulkan.1.dylib" \
                         /opt/homebrew/lib/libvulkan.1.dylib /usr/local/lib/libvulkan.1.dylib)" \
  || die "libvulkan.1.dylib not found (run tools/build/build_vulkan_loader_mac.sh)"
MVK_SRC="$(resolve /opt/homebrew/lib/libMoltenVK.dylib /usr/local/lib/libMoltenVK.dylib)" \
  || die "libMoltenVK.dylib not found (install molten-vk via Homebrew)"
for src in "${VK_LOADER_SRC}" "${MVK_SRC}"; do
  lipo -archs "${src}" | grep -qw arm64 \
    || die "${src} has no arm64 slice (got: $(lipo -archs "${src}")); install the Apple Silicon Homebrew package"
done

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

# ===========================================================================
# 2b. Bundle the native-Metal shader path's helpers when the binary links them
#     (XENIA_GPU_METAL builds): dxilconv (DXBC -> DXIL, built from the DXC fork)
#     and Apple's Metal Shader Converter. Both are @rpath references, resolved
#     in the dev tree through absolute LC_RPATH entries that must not ship.
# ===========================================================================
METAL_LIBS=()
if otool -L "${CONTENTS}/MacOS/${BUNDLE_EXEC}" | grep -q "@rpath/libdxilconv.dylib"; then
  log "Bundling native-Metal helpers (dxilconv + Metal Shader Converter)"
  DXILCONV_ARM64="${REPO_ROOT}/third_party/DirectXShaderCompiler/build_dxilconv_macos/lib/libdxilconv.dylib"
  DXILCONV_X86="${REPO_ROOT}/third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib/libdxilconv.dylib"
  MSC_SRC="${REPO_ROOT}/third_party/metal-shader-converter/lib/libmetalirconverter.dylib"
  [[ -f "${DXILCONV_ARM64}" ]] || die "libdxilconv.dylib not built at ${DXILCONV_ARM64}"
  [[ -f "${MSC_SRC}" ]] || die "libmetalirconverter.dylib not found at ${MSC_SRC}"
  if [[ "${UNIVERSAL}" -eq 1 ]]; then
    [[ -f "${DXILCONV_X86}" ]] || die "Universal 2 needs the x86_64 dxilconv at ${DXILCONV_X86} (configure build-x86_64 with -DXENIA_GPU_METAL=ON)"
    lipo -create "${DXILCONV_ARM64}" "${DXILCONV_X86}" -output "${CONTENTS}/Frameworks/libdxilconv.dylib"
  else
    cp "${DXILCONV_ARM64}" "${CONTENTS}/Frameworks/libdxilconv.dylib"
  fi
  cp "${MSC_SRC}" "${CONTENTS}/Frameworks/libmetalirconverter.dylib"
  chmod u+w "${CONTENTS}/Frameworks/"*.dylib
  install_name_tool -id "@rpath/libdxilconv.dylib" "${CONTENTS}/Frameworks/libdxilconv.dylib"
  install_name_tool -id "@rpath/libmetalirconverter.dylib" "${CONTENTS}/Frameworks/libmetalirconverter.dylib"
  METAL_LIBS=(libdxilconv.dylib libmetalirconverter.dylib)
fi

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
VKL_VER="$(basename "${VK_LOADER_SRC}" .dylib)"; VKL_VER="${VKL_VER#libvulkan.}"
cat > "${CONTENTS}/Resources/VERSIONS.txt" <<EOF
macXenos ${SHORT_VERSION} (${BRANCH} ${COMMIT_SHORT})
Bundled MoltenVK : ${MVK_SRC} (${MVK_VER})
Bundled loader   : ${VK_LOADER_SRC} (${VKL_VER})
EOF
if [[ "${WITH_SPARKLE}" -eq 1 ]]; then
  echo "Bundled Sparkle  : ${SPARKLE_DIR} (feed ${FEED_URL})" >> "${CONTENTS}/Resources/VERSIONS.txt"
fi
if [[ ${#METAL_LIBS[@]} -gt 0 ]]; then
  cat >> "${CONTENTS}/Resources/VERSIONS.txt" <<EOF
Bundled dxilconv : ${DXILCONV_ARM64}$([[ "${UNIVERSAL}" -eq 1 ]] && echo " + ${DXILCONV_X86}")
Bundled MSC      : ${MSC_SRC}
EOF
fi

# Third-party notices for System > About > Acknowledgements, beside the page.
# Strict: every license must be readable, and every third-party library on
# the executable's link line and every bundled binary must have an entry in
# docs/acknowledgements/manifest.json.
log "Collecting third-party notices"
BUNDLED_NAMES=()
for f in "${CONTENTS}/Frameworks/"*; do BUNDLED_NAMES+=("$(basename "$f")"); done
# The Homebrew whose MoltenVK and loader were bundled (their license files).
HB_PREFIX=""
for p in /opt/homebrew /usr/local; do
  [[ -d "${p}/opt/molten-vk" ]] && { HB_PREFIX="${p}"; break; }
done
[[ -n "${HB_PREFIX}" ]] || die "no Homebrew molten-vk found for its license file"
ACK_ARGS=(--out "${CONTENTS}/Resources/blades/acknowledgements.json" --strict
          --link-ninja "${REPO_ROOT}/build/CMakeFiles/impl-${CONFIG}.ninja"
          --homebrew-prefix "${HB_PREFIX}"
          --bundled "${BUNDLED_NAMES[@]}")
[[ "${WITH_SPARKLE}" -eq 1 ]] && ACK_ARGS+=(--sparkle-dir "${SPARKLE_DIR}")
python3 "${SCRIPT_DIR}/collect_licenses.py" "${ACK_ARGS[@]}" \
  || die "third-party notices incomplete (see above)"

# Confirm nothing in the bundle points back at an absolute Cellar/host path.
log "Verifying in-bundle dylib references"
if otool -L "${CONTENTS}/MacOS/${BUNDLE_EXEC}" "${CONTENTS}/Frameworks/"*.dylib \
     | grep -E "^[[:space:]]" | grep -E "/usr/local/|/opt/homebrew/|/Cellar/|${REPO_ROOT}" ; then
  die "bundle still references absolute host paths (see above)"
fi
if otool -l "${CONTENTS}/MacOS/${BUNDLE_EXEC}" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}' | grep -E "^/" ; then
  die "executable still carries absolute LC_RPATH entries (see above)"
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

# Every shipping Mach-O (every slice) must load on the app's minimum macOS: a
# dependency built for a newer one fails to load there (the beta 5 arm64
# Vulkan loader was Homebrew's macOS 27 bottle under a macOS 15 app).
MIN_MACOS="$(/usr/libexec/PlistBuddy -c "Print :LSMinimumSystemVersion" "${CONTENTS}/Info.plist")" \
  || die "Info.plist has no LSMinimumSystemVersion"
version_gt() {  # $1 > $2, dotted versions
  awk -v a="$1" -v b="$2" 'BEGIN { na = split(a, x, "."); nb = split(b, y, ".");
    for (i = 1; i <= (na > nb ? na : nb); ++i) { if (x[i] + 0 != y[i] + 0) exit !(x[i] + 0 > y[i] + 0) }
    exit 1 }'
}
log "Verifying deployment targets (LSMinimumSystemVersion ${MIN_MACOS})"
too_new=0
while IFS= read -r -d '' f; do
  archs="$(lipo -archs "$f" 2>/dev/null)" || continue
  for arch in ${archs}; do
    minos="$(vtool -arch "${arch}" -show-build "$f" 2>/dev/null \
             | awk '/minos|^ *version/ {print $2; exit}')"
    [[ -n "${minos}" ]] || die "no deployment target in ${f#"${CONTENTS}/"} (${arch})"
    if version_gt "${minos}" "${MIN_MACOS}"; then
      echo "  ${f#"${CONTENTS}/"} (${arch}) needs macOS ${minos}" >&2
      too_new=1
    fi
  done
done < <(find "${CONTENTS}" -type f -print0)
[[ "${too_new}" -eq 0 ]] || die "bundled code needs a newer macOS than ${MIN_MACOS} (see above; for the Vulkan loader run tools/build/build_vulkan_loader_mac.sh)"

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

# Sign nested code inside out, then the app. Not --deep: that would give
# Sparkle's helpers the app's JIT entitlements.
for f in "${CONTENTS}/Frameworks/"*.dylib; do
  codesign "${SIGN_ARGS[@]}" -s "${IDENTITY}" "${f}"
done
if [[ "${WITH_SPARKLE}" -eq 1 ]]; then
  # Sparkle's documented order, with hardened runtime and no entitlements.
  SPARKLE_FW="${CONTENTS}/Frameworks/Sparkle.framework"
  SPARKLE_SIGN=(--force --timestamp --options runtime -s "${IDENTITY}")
  codesign "${SPARKLE_SIGN[@]}" "${SPARKLE_FW}/Versions/B/Autoupdate"
  codesign "${SPARKLE_SIGN[@]}" "${SPARKLE_FW}/Versions/B/Updater.app"
  codesign "${SPARKLE_SIGN[@]}" "${SPARKLE_FW}"
fi
codesign "${SIGN_ARGS[@]}" -s "${IDENTITY}" "${APP_DIR}"

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

# The DMG is what people download, so Gatekeeper checks it first: sign it, and
# with --notarize submit and staple it too (the app inside is already stapled).
if [[ "${ADHOC}" -eq 0 ]]; then
  log "Signing DMG"
  codesign --force --timestamp -s "${IDENTITY}" "${DMG_PATH}"
fi
if [[ "${DO_NOTARIZE}" -eq 1 ]]; then
  log "Notarizing DMG"
  xcrun notarytool submit "${DMG_PATH}" \
    --apple-id "${APPLE_ID}" --password "${APPLE_PASSWORD}" \
    --team-id "${APPLE_TEAM_ID}" --wait
  xcrun stapler staple "${DMG_PATH}"
  spctl -a -t open --context context:primary-signature -v "${DMG_PATH}"
fi

log "Done:"
echo "  app: ${APP_DIR}"
echo "  dmg: ${DMG_PATH}"
