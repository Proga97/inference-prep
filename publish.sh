#!/usr/bin/env bash
# Publishes the tracker to GitHub Pages. Run from this folder:  bash publish.sh
set -euo pipefail

REPO="${1:-inference-prep}"
VISIBILITY="${2:---private}"   # pass --public as the 2nd arg if you want a public repo
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "→ folder: $(pwd)"
[ -f inference-tracker.html ] || { echo "✗ inference-tracker.html not found here"; exit 1; }
cp -f inference-tracker.html index.html

if ! command -v gh >/dev/null 2>&1; then
  cat <<'EOF'
✗ GitHub CLI (gh) is not installed.
  Install it:      brew install gh
  Then re-run:     bash publish.sh
  Or publish without it: create an empty repo on github.com, then
      git init && git add index.html && git commit -m "prep tracker"
      git branch -M main
      git remote add origin https://github.com/<you>/<repo>.git
      git push -u origin main
  …and turn on Pages in the repo's Settings → Pages (branch: main, folder: /).
EOF
  exit 1
fi

gh auth status >/dev/null 2>&1 || { echo "→ signing in to GitHub…"; gh auth login; }

[ -d .git ] || git init -q
git add index.html
git commit -qm "Inference engineer prep tracker" || echo "→ nothing new to commit"
git branch -M main

if gh repo view "$REPO" >/dev/null 2>&1; then
  echo "→ repo $REPO already exists, pushing"
  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin "https://github.com/$(gh api user -q .login)/$REPO.git"
  git push -u origin main
else
  gh repo create "$REPO" "$VISIBILITY" --source=. --push
fi

USER=$(gh api user -q .login)
echo "→ enabling GitHub Pages…"
gh api "repos/$USER/$REPO/pages" -X POST -f "source[branch]=main" -f "source[path]=/" >/dev/null 2>&1 \
  || gh api "repos/$USER/$REPO/pages" -X PUT -f "source[branch]=main" -f "source[path]=/" >/dev/null 2>&1 \
  || echo "  (couldn't enable Pages via API — turn it on in Settings → Pages, branch main, folder /)"

echo
echo "✓ Done. Your tracker will be live in a minute at:"
echo "     https://$USER.github.io/$REPO/"
echo
echo "  Open that on your phone → Share → Add to Home Screen."
echo "  Note: Pages on a PRIVATE repo needs GitHub Pro. If the URL 404s,"
echo "  either upgrade, or re-run as:  bash publish.sh $REPO --public"
