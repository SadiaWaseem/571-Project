#!/usr/bin/env bash
# Download the MVRDC benchmark (instances, published solutions and results)
# of Masmoudi et al. (2026) into data/MVRDC. The data stay in the authors'
# repository; we do not redistribute them.
set -euo pipefail
DEST="${1:-data/MVRDC}"
if [ -d "$DEST/Instances" ]; then
  echo "data already present in $DEST"
else
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 https://github.com/Masmoudi-Amine/MVRDC-.git "$DEST"
fi
