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
#include "vitis/ai/nnpp/medicaldetection.hpp"
#include "medicaldetection_post.hpp"
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;

namespace vitis{
namespace ai {


MedicalDetectionPostProcess::MedicalDetectionPostProcess(){};
MedicalDetectionPostProcess::~MedicalDetectionPostProcess(){};

std::unique_ptr<MedicalDetectionPostProcess> MedicalDetectionPostProcess::create(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config ) {
  return std::unique_ptr<MedicalDetectionPostProcess>(
      new medicaldetection::MedicalDetectionPost(input_tensors, output_tensors, config ));
}

}  // namespace ai
}  // namespace 
