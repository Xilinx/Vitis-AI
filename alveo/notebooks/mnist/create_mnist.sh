## Copyright 2019 Xilinx Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

#!/bin/bash
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
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf ./mnist_train_${BACKEND}
rm -rf ./mnist_test_${BACKEND}

export LD_LIBRARY_PATH=/opt/caffe/build/lib:$LD_LIBRARY_PATH

./convert_mnist_data.bin ./train-images-idx3-ubyte ./train-labels-idx1-ubyte ./mnist_train_${BACKEND} --backend=${BACKEND}
./convert_mnist_data.bin ./t10k-images-idx3-ubyte ./t10k-labels-idx1-ubyte ./mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."

