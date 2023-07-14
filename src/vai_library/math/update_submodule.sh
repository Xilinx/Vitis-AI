#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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
#
#!/bin/bash
exit
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR;

function git_submodule_update {
    REPO=$1
    WORK_DIR=$2
    SHA1=$3

    if test -d $WORK_DIR; then
        rm -fr $WORK_DIR
    fi
    git clone $REPO $WORK_DIR
    (cd $WORK_DIR && git fetch --all --tags --prune && git checkout --quiet $SHA1)
}

while read REPO WORK_DIR SHA1; do
    echo "update $REPO  $WORK_DIR $SHA1"
    git_submodule_update  $REPO $WORK_DIR $SHA1
done < submodule
