#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

vai_add_library(
  SRCS include/vitis/ai/benchmark.hpp
       include/vitis/ai/demo.hpp
       include/vitis/ai/demo2.hpp
       include/vitis/ai/demo3.hpp
       include/vitis/ai/demo_b1.hpp
       include/vitis/ai/demo_bx.hpp
       include/vitis/ai/dpdrm.hpp
       include/vitis/ai/image_list.hpp
       include/vitis/ai/stat_samples.hpp
       include/vitis/ai/demo_accuracy.hpp
       include/vitis/ai/onnx_task.hpp
       include/vitis/ai/getbatch.hpp
       # src/accuracy_facedetection.hpp src/main.cpp src/stat_samples.cpp
)
