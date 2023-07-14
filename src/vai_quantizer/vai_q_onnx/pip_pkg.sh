#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
set -e
set -x

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "pip_pkg /path/to/destination/directory"
  echo "all additional arguments (e.g. --flag1=v1 --flag2=v2) are passed to setup.py"
  exit 1
fi

# Create the destination directory, then do dirname on a non-existent file
# inside it to give us a path with tilde characters resolved (readlink -f is
# another way of doing this but is not available on a fresh macOS install).
# Finally, use cd and pwd to get an absolute path, in case a relative one was
# given.
mkdir -p "$1"
DEST=$(dirname "${1}/does_not_exist")
DEST=$(cd "$DEST" && pwd)

# Pass through remaining arguments (following the first argument, which
# specifies the output dir) to setup.py, e.g.,
#  ./pip_pkg /tmp/vai_q_onnx_pkg --release
# passes `--release` to setup.py.
python3 setup.py bdist_wheel --universal ${@:2} --dist-dir="$DEST" # >/dev/null

set +x
echo -e "\nBuild complete. Wheel files are in $DEST"
