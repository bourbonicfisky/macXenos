#!/usr/bin/env bash
#
# sparkle_appcast.sh - Write the Sparkle appcast (the update feed the app
# checks, SUFeedURL) for a folder of release DMGs.
#
# Usage:
#   tools/build/sparkle_appcast.sh <releases dir> <download url prefix>
#
#   <releases dir>         Holds one DMG per release, named by version
#                          (macXenos-1.0-beta5.dmg ...), optionally a release
#                          notes file of the same name (.md or .html), and the
#                          appcast.xml this updates (created on first use).
#   <download url prefix>  Where the DMGs are downloaded from, e.g.
#                          https://macxenos.com/releases/ - each item's URL is
#                          this plus the DMG's file name.
#
# Every DMG is signed with the EdDSA key in the login keychain under the
# account "macxenos" (made with: generate_keys --account macxenos; its public
# half is tools/build/sparkle_public_ed_key.txt, baked into Info.plist by
# package_mac.sh). An update whose signature does not match is refused.
#
# Release flow: tag the commit macxenos-<version> (macxenos-1.0) -> build
# -> package_mac.sh (it takes the version from the tag) -> copy build/dist/
# macXenos.dmg to <releases dir>/macXenos-<version>.dmg -> this script ->
# upload the DMG and appcast.xml so the feed URL serves the new appcast.

set -euo pipefail

[[ $# -eq 2 ]] || { sed -n '3,23p' "$0" >&2; exit 2; }
RELEASES="$1"
PREFIX="$2"
[[ -d "${RELEASES}" ]] || { echo "error: no folder ${RELEASES}" >&2; exit 1; }
[[ "${PREFIX}" == */ ]] || PREFIX="${PREFIX}/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARKLE_DIR="$("${SCRIPT_DIR}/fetch_sparkle.sh")"

# No deltas: each update is the whole DMG.
"${SPARKLE_DIR}/bin/generate_appcast" \
  --account macxenos \
  --download-url-prefix "${PREFIX}" \
  --link "https://macxenos.com" \
  --embed-release-notes \
  --maximum-deltas 0 \
  "${RELEASES}"

echo "appcast: ${RELEASES}/appcast.xml"
