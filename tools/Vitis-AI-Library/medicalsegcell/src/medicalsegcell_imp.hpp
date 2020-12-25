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
#pragma once

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/medicalsegcell.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class MedicalSegcellImp
    : public vitis::ai::TConfigurableDpuTask<MedicalSegcell> {
 public:
  MedicalSegcellImp(const std::string &model_name,
                         bool need_preprocess = true);
  virtual ~MedicalSegcellImp();

 private:
  virtual MedicalSegcellResult run(const cv::Mat &img) override;
  virtual std::vector<MedicalSegcellResult> run(
      const std::vector<cv::Mat> &img) override;

  virtual MedicalSegcellResult post_process( unsigned int idx);
  virtual std::vector<MedicalSegcellResult> post_process();
  
 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
};

}  // namespace ai
}  // namespace vitis

