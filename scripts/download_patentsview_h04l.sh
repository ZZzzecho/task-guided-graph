#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/laijizheng/datasets/patentsview_h04l_raw}"
ZENODO_RECORD="15783125"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

files=(
  g_patent.tsv.zip
  g_patent_abstract.tsv.zip
  g_cpc_current.tsv.zip
  g_cpc_title.tsv.zip
  g_us_patent_citation.tsv.zip
)

echo "Downloading USPTO PatentsView final metadata release (2024-12-31)"
echo "Zenodo record: $ZENODO_RECORD"
echo "Destination: $OUT_DIR"
echo "Downloads are resume-safe with wget -c."
echo

for file in "${files[@]}"; do
  echo "==> $file"
  wget -c --retry-connrefused --waitretry=3 --timeout=60     "${BASE_URL}/${file}?download=1"     -O "$file"
done

echo
echo "Done. Expected files:"
printf '  %s\n' "${files[@]}"
echo
echo "Snapshot coverage is fixed at 2024-12-31, matching the current 2005-2024 experiment window."
