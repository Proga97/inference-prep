#!/usr/bin/env bash
# Stage, commit and push the prep kit.  Run:  bash commit.sh ["message"]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MSG="${1:-Add prep kit, daily tracker and reference docs}"

# a stale lock can be left behind if a tool touched the repo without cleaning up
[ -f .git/index.lock ] && { echo "→ clearing stale .git/index.lock"; rm -f .git/index.lock; }

# index.html is the copy GitHub Pages serves — keep it identical to the tracker
[ -f inference-tracker.html ] && cp -f inference-tracker.html index.html

echo "→ staging"
git add -A
git status --short

if git diff --cached --quiet; then
  echo "→ nothing to commit"; exit 0
fi

git commit -m "$MSG"
echo "→ pushing"
git push -u origin "$(git branch --show-current)"

echo
echo "✓ pushed. If Pages is on, the update is live in a minute at:"
echo "   https://$(git remote get-url origin | sed -E 's#.*github.com[:/]([^/]+)/([^/.]+).*#\1.github.io/\2#')/"
