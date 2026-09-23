#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/laijizheng/datasets/patentsview_h04l_raw}"
BASE_URL="https://s3.amazonaws.com/data.patentsview.org/download"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

files=(
  g_patent.tsv.zip
  g_patent_abstract.tsv.zip
  g_cpc_current.tsv.zip
  g_cpc_title.tsv.zip
  g_us_patent_citation.tsv.zip
)

echo "Downloading PatentsView core tables to: $OUT_DIR"
echo "Downloads are resume-safe with wget -c."

for file in "${files[@]}"; do
  echo "==> $file"
  wget -c "$BASE_URL/$file"
done

echo
echo "Done. Expected files:"
printf '  %s\n' "${files[@]}"
