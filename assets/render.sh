#!/bin/bash
# Render assets/results-table.tex to results-table-{light,dark}.png (3200 px wide).
#   TECTONIC=<tectonic> PDFPY=<python with pymupdf> bash assets/render.sh
set -euo pipefail
cd "$(dirname "$0")"
T="${TECTONIC:-tectonic}"; PY="${PDFPY:-python}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp results-table.tex "$tmp/light.tex"
{ echo '\def\darkmode{}'; cat results-table.tex; } > "$tmp/dark.tex"
for m in light dark; do
    "$T" --chatter minimal --outdir "$tmp" "$tmp/$m.tex"
    "$PY" - "$tmp/$m.pdf" "results-table-$m.png" <<'PY'
import sys, pymupdf
page = pymupdf.open(sys.argv[1])[0]
page.get_pixmap(matrix=pymupdf.Matrix(3200 / page.rect.width, 3200 / page.rect.width)).save(sys.argv[2])
PY
done
