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

#include "vitis/ai/nnpp/classification.hpp"

#include <glog/logging.h>

#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>

#include "vitis/ai/math.hpp"
using namespace std;

namespace vitis {
namespace ai {

const char* ClassificationResult::lookup(int index) {
  static const char* table_1000[] = {
#include "word_list.inc"
  };
  static const char* table_cifar10[] = {
#include "cifar10_label.inc"
  };
  static const char* table_fmnist[] = {
#include "fmnist_label.inc"
  };
  static const char* table_orien[] = {
#include "orien_label.inc"
  };
  static const char* table_131[] = {
#include "word_131.inc"
  };
  static const char* table_dogcat[] = {"dog", "cat"};
  static const char* car_color_chen[] = {
#include "car_color_chen.inc"
  };

  if (index < 0) {
    return "";
  }
  if (this->type == 1) {
    return table_cifar10[index];
  } else if (this->type == 2) {
    return table_fmnist[index];
  } else if (this->type == 3) {
    return table_orien[index];
  } else if (this->type == 4) {
    return table_131[index];
  } else if (this->type == 5) {
    return table_dogcat[index];
  } else if (this->type == 6) {
    return car_color_chen[index];
  }
  return table_1000[index];
}  // namespace ai

static ClassificationResult topk(const float* softres, int channel, int k,
                                 int width, int height) {
  auto topkres = ClassificationResult{width, height};
  topkres.scores.reserve(k);
  priority_queue<pair<float, int>> q;
  for (int i = 0; i < channel; ++i) {
    q.push(pair<float, int>(softres[i], i));
  }

  for (int i = 0; i < k; ++i) {
    pair<float, int> maxprob = q.top();
    topkres.scores.emplace_back(
        ClassificationResult::Score{maxprob.second, softres[maxprob.second]});
    q.pop();
  }
  //  DLOG(INFO) << "topkres.size = " << topkres.scores.size();
  return topkres;
}

ClassificationResult classification_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx) {
  auto top_k = config.classification_param().top_k();
  std::vector<vitis::ai::library::OutputTensor> virtual_output;
  if (!config.classification_param().layer_name().empty()) {
    auto layer_names = config.classification_param().layer_name();
    for (auto i = 0u; i < output_tensors.size(); i++) {
      if (output_tensors[i].name.find(layer_names) != std::string::npos) {
        virtual_output.push_back(output_tensors[i]);
      }
    }
  } else {
    virtual_output.push_back(output_tensors[0]);
  }
  auto use_graph_runner = config.use_graph_runner();
  auto ret = ClassificationResult{};
  std::vector<float> softres(virtual_output[0].channel);
  if (use_graph_runner) {
    // There is generally no softmax on the model diagram
    vitis::ai::softmax((float*)virtual_output[0].get_data(batch_idx), 1.0f,
                       virtual_output[0].channel, 1, &softres[0]);
    ret = topk(&softres[0], virtual_output[0].channel, top_k,
               input_tensors[0].width, input_tensors[0].height);

  } else {
#ifdef ENABLE_DPUCADX8G_RUNNER
    vitis::ai::softmax((float*)virtual_output[0].get_data(batch_idx),
                       vitis::ai::library::tensor_scale(virtual_output[0]),
                       virtual_output[0].channel, 1, &softres[0]);
#else
    vitis::ai::softmax((int8_t*)virtual_output[0].get_data(batch_idx),
                       vitis::ai::library::tensor_scale(virtual_output[0]),
                       virtual_output[0].channel, 1, &softres[0]);
#endif
    // std::cout << std::endl;
    ret = topk(&softres[0], virtual_output[0].channel, top_k,
               input_tensors[0].width, input_tensors[0].height);
  }
  return ret;
}

std::vector<ClassificationResult> classification_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  auto batch_size = input_tensors[0].batch;
  auto ret = std::vector<ClassificationResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(
        classification_post_process(input_tensors, output_tensors, config, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
