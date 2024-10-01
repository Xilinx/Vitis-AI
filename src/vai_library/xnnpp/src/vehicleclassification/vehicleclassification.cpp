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

#include "vitis/ai/nnpp/vehicleclassification.hpp"

#include <glog/logging.h>

#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>

#include "vitis/ai/math.hpp"
using namespace std;

extern int GLOBAL_ENABLE_C_SOFTMAX;

namespace vitis {
namespace ai {

const char* VehicleClassificationResult::lookup(int index) {
  static const char* vehicle_make[] = {
#include "vehicle_make.inc"
  };
  static const char* vehicle_type[] = {
#include "vehicle_type.inc"
  };

  if (index < 0) {
    return "";
  }
  if (this->type == 1) {
    return vehicle_make[index];
  } else if (this->type == 2) {
    return vehicle_type[index];
  }
  return vehicle_make[index];
}  // namespace ai

static void softmax_m(int8_t* input, float scale, unsigned int cls,
                      unsigned int group, float* output) {
  float sum = 0.f;
  int8_t maxv = *max_element(input, input+cls);
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp((input[i]-maxv) * scale);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) output[i] /= sum;
}

static VehicleClassificationResult topk(const float* softres, int channel,
                                        int k, int width, int height) {
  auto topkres = VehicleClassificationResult{width, height};
  topkres.scores.reserve(k);
  priority_queue<pair<float, int>> q;
  for (int i = 0; i < channel; ++i) {
    q.push(pair<float, int>(softres[i], i));
  }

  for (int i = 0; i < k; ++i) {
    pair<float, int> maxprob = q.top();
    topkres.scores.emplace_back(VehicleClassificationResult::Score{
        maxprob.second, softres[maxprob.second]});
    q.pop();
  }
  //  DLOG(INFO) << "topkres.size = " << topkres.scores.size();
  return topkres;
}

VehicleClassificationResult vehicleclassification_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx) {
  // softmax use NEON in postprocess reduce accuracy about 7%
  // in vehicle_type xmodel, so disable NEON and use softmax_c here.

  //GLOBAL_ENABLE_C_SOFTMAX = 2;
  auto top_k = config.vehicleclassification_param().top_k();
  std::vector<vitis::ai::library::OutputTensor> virtual_output;
  if (!config.vehicleclassification_param().layer_name().empty()) {
    auto layer_names = config.vehicleclassification_param().layer_name();
    for (auto i = 0u; i < output_tensors.size(); i++) {
      if (output_tensors[i].name.find(layer_names) != std::string::npos) {
        virtual_output.push_back(output_tensors[i]);
      }
    }
  } else {
    virtual_output.push_back(output_tensors[0]);
  }
  auto ret = VehicleClassificationResult{};
  std::vector<float> softres(virtual_output[0].channel);
  softmax_m((int8_t*)virtual_output[0].get_data(batch_idx),
                     vitis::ai::library::tensor_scale(virtual_output[0]),
                     virtual_output[0].channel, 1, &softres[0]);
  //vitis::ai::softmax((int8_t*)virtual_output[0].get_data(batch_idx),
  //                   vitis::ai::library::tensor_scale(virtual_output[0]),
  //                   virtual_output[0].channel, 1, &softres[0]);
  ret = topk(&softres[0], virtual_output[0].channel, top_k,
             input_tensors[0].width, input_tensors[0].height);
  return ret;
}

VehicleClassificationResult vehicleclassification_top1(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx) {
  auto ret = VehicleClassificationResult{(int)input_tensors[0].width,
      (int)input_tensors[0].height};
  ret.scores.reserve(1);
  auto input_data = (int8_t*)output_tensors[0].get_data(batch_idx);
  int max_val = std::distance(input_data, std::max_element(input_data, input_data+output_tensors[0].channel));
  ret.scores.emplace_back(VehicleClassificationResult::Score{
      max_val, 0.99});
  return ret;
}

std::vector<VehicleClassificationResult> vehicleclassification_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  auto batch_size = input_tensors[0].batch;
  auto ret = std::vector<VehicleClassificationResult>{};
  ret.reserve(batch_size);
  auto top_k = config.vehicleclassification_param().top_k();
  for (auto i = 0u; i < batch_size; i++) {
    if (top_k == 1) 
      ret.emplace_back(vehicleclassification_top1(
        input_tensors, output_tensors, config, i));
    else
      ret.emplace_back(vehicleclassification_post_process(
        input_tensors, output_tensors, config, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
