#!/bin/bash
# One-time publication of AXON to the owner's public GitHub repository.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v gh >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo 'Installing GitHub CLI...'
    brew install gh
  else
    echo 'Install GitHub CLI from https://cli.github.com/ and rerun this script.'
    exit 1
  fi
fi

if ! gh auth status >/dev/null 2>&1; then
  echo 'Authorize GitHub in the browser opened by GitHub CLI.'
  gh auth login --hostname github.com --git-protocol https --web
fi

account="$(gh api user --jq '.login')"
if [ "$account" != 'siddhantkrishna' ]; then
  echo "STOP: GitHub CLI is logged in as '$account', not 'siddhantkrishna'."
  echo 'Run gh auth switch and try again.'
  exit 1
fi

if [ ! -d .git ]; then
  git init -b main
fi
if [ "$(git branch --show-current)" != 'main' ]; then
  echo 'STOP: currently on a branch other than main. Check git status first.'
  exit 1
fi

if [ -z "$(git config --local user.name || true)" ]; then
  git config --local user.name 'Siddhant Krishna'
fi
if [ -z "$(git config --local user.email || true)" ]; then
  id="$(gh api user --jq '.id')"
  git config --local user.email "${id}+${account}@users.noreply.github.com"
fi

# Stage only known public source assets, never downloaded data or corrections.
git add README.md LICENSE .gitignore requirements.txt requirements-dev.txt \
  start.command publish_github.command axon/ tests/ .github/ docs/ models/digits.npz
if ! git diff --cached --quiet; then
  git commit -m 'feat: add live neural-network explorer to AXON'
else
  echo 'No additional source changes to commit.'
fi

repo='siddhantkrishna/axon'
if gh repo view "$repo" >/dev/null 2>&1; then
  if git remote get-url origin >/dev/null 2>&1; then
    actual="$(git remote get-url origin)"
    case "$actual" in
      *github.com/siddhantkrishna/axon|*github.com/siddhantkrishna/axon.git) ;;
      *) echo "STOP: existing origin is $actual; not overwriting it."; exit 1 ;;
    esac
  else
    git remote add origin 'https://github.com/siddhantkrishna/axon.git'
  fi
  git push -u origin main
else
  gh repo create "$repo" --public --source=. --remote=origin --push \
    --description 'AXON: neural networks from scratch in NumPy, with a live interactive neuron and connection explorer'
fi

echo
echo 'Publication result:'
gh repo view "$repo" --json url,visibility --jq '.url + " (" + .visibility + ")"'
git status --short
