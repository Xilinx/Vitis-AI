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

#include "vitis/ai/nnpp/ultrafast.hpp"
#include "ultrafast_postimp.hpp"

using namespace std;
namespace vitis {
namespace ai {

UltraFastPost::UltraFastPost(){}
UltraFastPost::~UltraFastPost(){}

std::unique_ptr<UltraFastPost> UltraFastPost::create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config,
      int batch_size,
      int& real_batch_size,
      std::vector<cv::Size>& pic_size
     )
{
  return std::unique_ptr<UltraFastPost>(new UltraFastPostImp( 
             input_tensors, output_tensors, config, batch_size, real_batch_size, pic_size
           ));
}

}  // namespace ai
}  // namespace vitis
