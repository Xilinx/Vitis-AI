/*
 * Copyright 2019 Xilinx Inc.
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
#ifndef DEEPHI_Cifar10Classification_HPP_
#define DEEPHI_Cifar10Classification_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/cifar10classification.hpp>

namespace vitis {

namespace ai {

class Cifar10ClassificationImp
    : public vitis::ai::TConfigurableDpuTask<Cifar10Classification> {
 public:
  Cifar10ClassificationImp(const std::string &model_name,
                         bool need_preprocess = true);
  virtual ~Cifar10ClassificationImp();

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  float scale_conf_;
  Cifar10ClassificationResult post_process(int idx);
  virtual Cifar10ClassificationResult run(const cv::Mat &img) override;
  virtual std::vector<Cifar10ClassificationResult> run(
      const std::vector<cv::Mat> &img) override;
};
}  // namespace ai
}  // namespace vitis

#endif
