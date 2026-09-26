#!/bin/bash
# Check out upstream AReaL at the pinned commit and apply the CIS patch.
#
#   bash areal_patch/apply_patch.sh [target_dir]      # default: third_party/AReaL
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=${1:-$HERE/../third_party/AReaL}
COMMIT=e7868690d76a9ef39ed283bcd2e39203f6abe77d

[ -d "$TARGET/.git" ] || git clone https://github.com/areal-project/AReaL.git "$TARGET"
cd "$TARGET"
git checkout --quiet "$COMMIT"
git apply --check "$HERE/cis_areal.patch"
git apply "$HERE/cis_areal.patch"
echo "Patched AReaL ($COMMIT) at $(pwd)"
