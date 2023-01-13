/*
 * Copyright 2019 xilinx Inc.
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
#include <vart/runner_ext.hpp>

#include "vitis/ai/bevdet.hpp"

using std::shared_ptr;
using std::vector;

namespace vitis {
namespace ai {
class BEVdetImp : public BEVdet {
 public:
  BEVdetImp(const std::string& model_name, bool use_aie);
  virtual ~BEVdetImp();

 private:
  virtual std::vector<CenterPointResult> run(
      const std::vector<cv::Mat>& images,
      const std::vector<std::vector<char>>& input_bins) override;

 private:
  bool use_aie_;
  std::string model;
  std::unique_ptr<vart::RunnerExt> runner;
  std::vector<float> mean;
  std::vector<float> scale;
  vart::Runner* aie_runner;
  std::vector<std::unique_ptr<vart::TensorBuffer>> aie_runner_inputs;
  std::vector<std::unique_ptr<vart::TensorBuffer>> aie_runner_outputs;
};
}  // namespace ai
}  // namespace vitis
