/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <thread>
#include <vector>

/* header file for Vitis AI unified API */
#include <vart/dpu/dpu_runner.hpp>

using namespace vitis;
using namespace ai;
struct TensorShape {
    unsigned int height;
    unsigned int width;
    unsigned int channel;
    unsigned int size;
};

struct GraphInfo{
    struct TensorShape *inTensorList;
    struct TensorShape *outTensorList;
    std::vector<int> output_mapping;
};

int getTensorShape(DpuRunner* runner, GraphInfo *shapes, int cntin, const std::vector<std::string> output_names);
int getTensorShape(DpuRunner* runner, GraphInfo *shapes, int cntin, int cnout);


inline std::vector<std::unique_ptr<vitis::ai::Tensor>>
cloneTensorBuffer(const std::vector<vitis::ai::Tensor*> & tensors) {
  auto ret = std::vector<std::unique_ptr<vitis::ai::Tensor>>{};
  auto type = vitis::ai::Tensor::DataType::FLOAT;
  ret.reserve(tensors.size());
  for(const auto &tensor: tensors){
    ret.push_back(std::unique_ptr<vitis::ai::Tensor>(
        new vitis::ai::Tensor(tensor->get_name(), tensor->get_dims(), type)));
  }
  return ret;
}

#endif
