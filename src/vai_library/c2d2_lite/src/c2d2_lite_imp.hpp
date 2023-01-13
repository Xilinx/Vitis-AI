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
#include <vitis/ai/configurable_dpu_task.hpp>

#include "vitis/ai/c2d2_lite.hpp"

using std::shared_ptr;
using std::vector;

namespace vitis {
namespace ai {
class C2D2_liteImp0;
class C2D2_liteImp1;
class C2D2_liteImp : public vitis::ai::TConfigurableDpuTask<C2D2_lite> {
 public:
  C2D2_liteImp(const std::string& model_name0, const std::string& model_name1,
               bool need_preprocess = true);
  virtual ~C2D2_liteImp();

 private:
  virtual size_t get_input_batch() const override;
  virtual float run(const std::vector<cv::Mat>& image) override;
  virtual std::vector<float> run(
      const std::vector<std::vector<cv::Mat>>& images) override;

  C2D2_liteImp1* imp1;
};
}  // namespace ai
}  // namespace vitis
