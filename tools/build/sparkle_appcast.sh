#!/usr/bin/env bash
#
# sparkle_appcast.sh - Write the Sparkle appcast (the update feed the app
# checks, SUFeedURL) for a folder of release DMGs.
#
# Usage:
#   tools/build/sparkle_appcast.sh <releases dir> [<download url prefix>]
#
#   <releases dir>         Holds one DMG per release, named by version
#                          (macXenos-1.0.dmg ...), optionally a release notes
#                          file of the same name (.md or .html), and the
#                          appcast.xml this updates (created on first use).
#   <download url prefix>  Where the DMGs are downloaded from. Default: the
#                          public repository's GitHub releases,
#                          https://github.com/bourbonicfisky/macXenos/releases/download/
#                          where each DMG sits under its release tag
#                          (.../download/macxenos-1.0/macXenos-1.0.dmg). Any
#                          other prefix is used as is (prefix + file name).
#
# Every DMG is signed with the EdDSA key in the login keychain under the
# account "macxenos" (made with: generate_keys --account macxenos; its public
# half is tools/build/sparkle_public_ed_key.txt, baked into Info.plist by
# package_mac.sh). An update whose signature does not match is refused.
#
# The feed itself is served from https://macxenos.com/appcast.xml (SUFeedURL,
# baked into every shipped copy, so it stays there): the website repository's
# public/appcast.xml.
#
# Release flow: tag the commit macxenos-<version> (macxenos-1.0) -> build
# -> package_mac.sh (it takes the version from the tag) -> copy build/dist/
# macXenos.dmg to <releases dir>/macXenos-<version>.dmg -> upload it to the
# GitHub release for the tag -> this script -> copy appcast.xml to the
# website's public/ and deploy it.

set -euo pipefail

GITHUB_RELEASES="https://github.com/bourbonicfisky/macXenos/releases/download/"
[[ $# -eq 1 || $# -eq 2 ]] || { sed -n '3,33p' "$0" >&2; exit 2; }
RELEASES="$1"
PREFIX="${2:-${GITHUB_RELEASES}}"
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

# GitHub serves each release's files under its tag: point every DMG's
# enclosure at .../download/macxenos-<version>/macXenos-<version>.dmg.
# generate_appcast writes prefix + file name, so this runs after every pass
# (it leaves URLs it already rewrote alone).
if [[ "${PREFIX}" == */releases/download/ ]]; then
  python3 - "${RELEASES}/appcast.xml" "${PREFIX}" <<'PY'
import re, sys
path, prefix = sys.argv[1], sys.argv[2]
text = open(path).read()
pattern = re.compile(re.escape(prefix) + r'(macXenos-([^"/]+)\.dmg)"')
text = pattern.sub(lambda m: f'{prefix}macxenos-{m.group(2)}/{m.group(1)}"', text)
open(path, "w").write(text)
PY
fi

echo "appcast: ${RELEASES}/appcast.xml"
grep -o 'url="[^"]*\.dmg"' "${RELEASES}/appcast.xml" | sed 's/^/  /'
