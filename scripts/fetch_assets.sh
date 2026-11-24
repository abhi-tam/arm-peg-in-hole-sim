#!/usr/bin/env bash
set -euo pipefail

COMMIT=a758567
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone --filter=blob:none --no-checkout https://github.com/TheRobotStudio/SO-ARM100.git "$TMP"
git -C "$TMP" sparse-checkout set Simulation/SO101
git -C "$TMP" checkout "$COMMIT"

FULL_SHA=$(git -C "$TMP" rev-parse HEAD)
echo "Full commit SHA: $FULL_SHA"

mkdir -p assets/SO101
cp -r "$TMP"/Simulation/SO101/. assets/SO101/
if [ -f "$TMP"/LICENSE ]; then
  cp "$TMP"/LICENSE assets/SO101/UPSTREAM_LICENSE
else
  echo "WARNING: no LICENSE at repo root"
fi

ls -la assets/SO101
