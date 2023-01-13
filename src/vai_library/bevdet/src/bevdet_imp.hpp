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
#include <vitis/ai/configurable_dpu_task.hpp>

#include "vitis/ai/bevdet.hpp"

using std::shared_ptr;
using std::vector;

namespace vitis {
namespace ai {
class BEVdetImp : public BEVdet {
 public:
  BEVdetImp(const std::string& model_name0, const std::string& model_name1,
            const std::string& model_name2);
  virtual ~BEVdetImp();

 private:
  virtual std::vector<CenterPointResult> run(
      const std::vector<cv::Mat>& images,
      const std::vector<std::vector<char>>& input_bins) override;
  //  virtual std::vector<CenterPointResult> run(
  //      const std::vector<std::vector<char>>& input_bins) override;
  void middle_process(const std::vector<std::vector<char>>& input1,
                      const vitis::ai::library::InputTensor& input2);
  void run_aie(const std::vector<std::vector<char>>& input_bins,
               int8_t* output);
  void run_model_0(const std::vector<cv::Mat>& images, int idx);

 private:
  // std::mutex model_0_mtx_;
  bool use_aie_;
  std::unique_ptr<xir::Attrs> dpu_attrs_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_0_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_2_;
  vart::Runner* aie_runner;
  std::vector<std::unique_ptr<vart::TensorBuffer>> aie_runner_inputs;
  std::vector<std::unique_ptr<vart::TensorBuffer>> aie_runner_outputs;
  char output0_80[337920];
  char output0_64[270336];
  char* output0_64_ptr;
  char* output0_80_ptr;
  float output1[658944];
  float voxel_output[1310720];
};
}  // namespace ai
}  // namespace vitis
