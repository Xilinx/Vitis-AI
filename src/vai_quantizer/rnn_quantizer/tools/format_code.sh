#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

yapf_format() {
  yapf -i --style='{based_on_style: yapf, indent_width: 2, blank_lines_around_top_level_definition: 1}' $1
}

strip_trailing_ctrlm() {
  sed -i "s/^M//" $1
}

for file in $(find "${SCRIPT_DIR}/.." -name *.py); do
  echo "${file}"
  yapf_format "${file}"
  strip_trailing_ctrlm "${file}"
  dos2unix "${file}"
done
