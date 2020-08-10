#!/usr/bin/env bash

add_notice_to_file() {
  echo "adding notice to file $1\n"
  FINENAME=$1
  TMP="${FINENAME}.tmp"
  BAK="${FINENAME}.bak"

cat << EOF > ${TMP}
/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

EOF

  cat "${FINENAME}" >> "${TMP}"

  cp "${FINENAME}" "{BAK}"
  mv "${TMP}" "${FINENAME}"
  rm -f "${BAK}"
}

append_hash_to_file() {
  FINENAME=$1

cat << EOF >> "${FINENAME}"

// 2ff8d57c0d5afa55f55c53fea2bba1a8a6bf5eb216ac887dc353ca12e8ead345
EOF
}

add_notice_to_py_file() {
  echo "adding notice to file $1\n"
  FINENAME=$1
  TMP="${FINENAME}.tmp"
  BAK="${FINENAME}.bak"

cat << EOF > ${TMP}
# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

EOF

  cat "${FINENAME}" >> "${TMP}"

  cp "${FINENAME}" "{BAK}"
  mv "${TMP}" "${FINENAME}"
  rm -f "${BAK}"
}

append_hash_to_py_file() {
  FINENAME=$1

cat << EOF >> "${FINENAME}"

# 2ff8d57c0d5afa55f55c53fea2bba1a8a6bf5eb216ac887dc353ca12e8ead345
EOF
}

CANDIDATE_FILE_EXTS=(".h" ".hpp" ".c" ".cc" ".cpp" ".py")

main() {
  echo "File extentions to be processed: ${CANDIDATE_FILE_EXTS[@]}"
  for dir in "$@"; do
    echo "Processing directory ${dir} ..."
    names=""
    for ext in "${CANDIDATE_FILE_EXTS[@]}"; do
      echo "Processing ext: ${ext}"
      files=`find "${dir}" -name "*${ext}"`
      files_arr=($files)
      names="${names} $(echo "${files}" | tr '\n' ' ')"
      for file in "${files_arr[@]}"; do
        if [[ ${ext} == ".py" && ${file} ]]; then
            echo "Processing python: ${file}"
            add_notice_to_py_file ${file}
            append_hash_to_py_file ${file}
        elif [[ ${file} ]]; then
            echo "Processing: ${file}"
            add_notice_to_file ${file}
            append_hash_to_file ${file}
        fi
      done
    done
    # echo -e "${names}\n"
  done
}

if [[ "$#" -lt 1 ]]; then
  echo "Add a xilinx copyright notice and a hash tag to the files in specified dirs."
  echo "Usage: $0 DIRS..."
  echo "Example: $0 include src"
  exit 1
fi

main $@

