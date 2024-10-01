/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <memory>
#include <iostream>
#include <cstring>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./preprocess.hpp"
#include "./middle_process.hpp"

#include "./centerpoint_imp.hpp"

using namespace std;
namespace vitis {
namespace ai {


CenterPointImp::CenterPointImp(const std::string &model_name_0, const std::string &model_name_1)
      : points_dim_(4), 
        model_0_{ConfigurableDpuTask::create(model_name_0, false)}, 
        model_1_{ConfigurableDpuTask::create(model_name_1, false)} 
       {
  
}


CenterPointImp::~CenterPointImp() {
}

int CenterPointImp::getInputWidth() const {
  return model_0_->getInputWidth();
}

int CenterPointImp::getInputHeight() const {
  return model_0_->getInputHeight();
}

size_t CenterPointImp::get_input_batch() const {
  return model_0_->get_input_batch();
}

std::vector<CenterPointResult>
CenterPointImp::run(const std::vector<float> &input) {
  __TIC__(CENTERPOINT_E2E)
  __TIC__(CENTERPOINT_PREPROCESS)
  std::vector<int> coors;
  auto batch_size = get_input_batch();
  auto model_0_input_size = model_0_->getInputTensor()[0][0].size / batch_size;
  auto input_ptr = (int8_t *)model_0_->getInputTensor()[0][0].get_data(0);
  std::memset(input_ptr, 0, model_0_input_size); 
  auto input_tensor_scale = vitis::ai::library::tensor_scale(model_0_->getInputTensor()[0][0]);
  // auto input_scale = std::vector<float>{0.025, 0.025, 0.25, 0.0078}; // read from config
  //auto input_mean = std::vector<float>{40, 0, -1.5, 127.5}; // read from config
  auto input_scale = std::vector<float>{0.025, 0.025, 0.1429, 0.0351}; // read from config
  auto input_mean = std::vector<float>{40, 0, 1, 73.5}; // read from config
  for (auto i = 0u;  i<input_scale.size(); ++i) {
    input_scale[i] *= input_tensor_scale;
  }
  coors = vitis::ai::centerpoint::preprocess3(input, 4, input_mean, input_scale, input_ptr);
  for (auto i = 0u; i < coors.size();) {
    // cout << i/4 << "\t ";
    for(auto j = 0u; j < 4u; j++) {
      // cout << coors[i] << "\t  ";
      i++;
    }
    //cout << endl;
  }
  coors.resize(3600*4, 0);
  //cout << coors.size() << endl;
  __TOC__(CENTERPOINT_PREPROCESS)
  __TIC__(CENTERPOINT_SET_INPUT)
  __TOC__(CENTERPOINT_SET_INPUT)
  __TIC__(CENTERPOINT_DPU_0)
  model_0_->run(0);
  __TOC__(CENTERPOINT_DPU_0)

  __TIC__(CENTERPOINT_MIDDLE_PROCESS)
  auto model_1_input_tensor_size = model_1_->getInputTensor()[0][0].size;
  auto model_1_input_size = model_1_input_tensor_size / batch_size;
  std::memset(model_1_->getInputTensor()[0][0].get_data(0), 0, model_1_input_size); 
  auto in_channels = 64; 
  vitis::ai::centerpoint::middle_process(coors, 
         (int8_t *)model_0_->getOutputTensor()[0][0].get_data(0),
         vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]),
         (int8_t *)model_1_->getInputTensor()[0][0].get_data(0),
         vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]),
         in_channels);
  __TOC__(CENTERPOINT_MIDDLE_PROCESS)
  __TIC__(CENTERPOINT_DPU_1)
  model_1_->run(0);
  __TOC__(CENTERPOINT_DPU_1)
  __TIC__(CENTERPOINT_POSTPROCESS)
  auto bbox_final = post_process(model_1_->getOutputTensor(), 0);
  __TOC__(CENTERPOINT_POSTPROCESS)
  __TOC__(CENTERPOINT_E2E)
  return bbox_final;
}

std::vector<std::vector<CenterPointResult>> 
CenterPointImp::run(const std::vector<std::vector<float>> &inputs) {
  __TIC__(CENTERPOINT_E2E)
  // auto input_scale = std::vector<float>{0.025, 0.025, 0.25, 0.0078}; // read from config
  // auto input_mean = std::vector<float>{40, 0, -1.5, 127.5}; // read from config
  auto input_scale = std::vector<float>{0.025, 0.025, 0.1429, 0.0351}; // read from config
  auto input_mean = std::vector<float>{40, 0, 1, 73.5}; // read from config
  auto batch_size = get_input_batch();
  auto num = std::min(batch_size, inputs.size());
  std::vector<std::vector<int>> coors_vec(num);
  auto model_0_input_size = model_0_->getInputTensor()[0][0].size / batch_size;
  for (auto i = 0u; i < num; ++i) {
    std::memset(model_0_->getInputTensor()[0][0].get_data(i), 0, model_0_input_size); 
  }
  __TIC__(CENTERPOINT_PREPROCESS)
  for (auto batch_ind = 0u; batch_ind < num; batch_ind++) {
    auto input_ptr = (int8_t *)model_0_->getInputTensor()[0][0].get_data(batch_ind);
    auto input_tensor_scale = vitis::ai::library::tensor_scale(model_0_->getInputTensor()[0][0]);
    for (auto i = 0u;  i<input_scale.size(); ++i) {
      input_scale[i] *= input_tensor_scale;
    }
    coors_vec[batch_ind] = vitis::ai::centerpoint::preprocess3(inputs[batch_ind], 4, input_mean, input_scale, input_ptr);
    /*
    for (auto i = 0u; i < coors.size();) {
      cout << i/4 << "\t ";
      for(auto j = 0u; j < 4u; j++) {
        cout << coors[i] << "\t  ";
        i++;
      }
      cout << endl;
    }
    */
    coors_vec.resize(2560*4);
  }
  __TOC__(CENTERPOINT_PREPROCESS)

  __TIC__(CENTERPOINT_DPU_0)
  model_0_->run(0);
  __TOC__(CENTERPOINT_DPU_0)

  __TIC__(CENTERPOINT_MIDDLE_PROCESS)
  auto model_1_input_tensor_size = model_1_->getInputTensor()[0][0].size;
  auto model_1_input_size = model_1_input_tensor_size / batch_size;
  for (auto i = 0u; i < num; ++i) {
    std::memset(model_1_->getInputTensor()[0][0].get_data(i), 0, model_1_input_size); 
  }
  auto in_channels = 64; // read from config
  for (auto i = 0u; i < num; ++i) {
    vitis::ai::centerpoint::middle_process(coors_vec[i], 
         (int8_t *)model_0_->getOutputTensor()[0][0].get_data(i),
         vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][i]),
         (int8_t *)model_1_->getInputTensor()[0][0].get_data(i),
         vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]),
         in_channels);
  }
  __TOC__(CENTERPOINT_MIDDLE_PROCESS)
  __TIC__(CENTERPOINT_DPU_1)
  model_1_->run(0);
  __TOC__(CENTERPOINT_DPU_1)

  __TIC__(CENTERPOINT_POSTPROCESS)
  __TOC__(CENTERPOINT_POSTPROCESS)

  __TOC__(CENTERPOINT_E2E)
  std::vector<std::vector<CenterPointResult>> bbox_finals(batch_size);
  for (auto batch_ind = 0u; batch_ind < num; batch_ind++) {
    bbox_finals[batch_ind] = post_process(model_1_->getOutputTensor(), batch_ind);
  }
  return bbox_finals;
}

}}

