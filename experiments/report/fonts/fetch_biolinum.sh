#!/bin/sh
# Copy Linux Biolinum O (regular, bold, italic; GPL/OFL) from a TeX Live tree into this directory. style.py converts
# them to TrueType on first use. Uses a local kpsewhich if there is one, else the texlive/texlive image the paper
# Makefile uses.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
files="LinBiolinum_R.otf LinBiolinum_RB.otf LinBiolinum_RI.otf"
if command -v kpsewhich >/dev/null 2>&1; then
    for f in $files; do cp "$(kpsewhich "$f")" "$here/"; done
else
    docker run --rm -u "$(id -u):$(id -g)" -v "$here":/out texlive/texlive:latest \
        sh -c "for f in $files; do cp \"\$(kpsewhich \$f)\" /out/; done"
fi
ls -l "$here"/LinBiolinum_*.otf
