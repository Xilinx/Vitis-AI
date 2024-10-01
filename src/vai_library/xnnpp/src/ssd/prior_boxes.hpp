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
#pragma once

#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <memory>
#include <utility>
#include <vector>

namespace vitis {
namespace ai {
namespace dpssd {

class PriorBoxes {
 public:
  PriorBoxes(int image_width, int image_height, int layer_width,
             int layer_height, const std::vector<float>& variances,
             const std::vector<float>& min_sizes,
             const std::vector<float>& max_sizes,
             const std::vector<float>& aspect_ratios, float offset,
             bool if_tfmodel, bool is_mlperf, float step_width = 0.f,
             float step_height = 0.f, bool flip = true, bool clip = false);

  const std::vector<std::shared_ptr<std::vector<float>>>& priors() const {
    return priors_;
  }

 protected:
  void create_priors();
  void create_priors_mlperf();

  std::vector<std::shared_ptr<std::vector<float>>> priors_;

  std::pair<int, int> image_dims_;
  std::pair<int, int> layer_dims_;
  std::pair<float, float> step_dims_;

  std::vector<std::pair<float, float>> boxes_dims_;

  float offset_;
  bool clip_;

  std::vector<float> variances_;
};

std::vector<std::shared_ptr<std::vector<float>>> CreatePriors(
    int image_width, int image_height, bool is_tf, bool is_mlperf,
    const google::protobuf::RepeatedPtrField<vitis::ai::proto::PriorBoxParam>&
        boxes);

}  // namespace dpssd
}  // namespace ai
}  // namespace vitis
