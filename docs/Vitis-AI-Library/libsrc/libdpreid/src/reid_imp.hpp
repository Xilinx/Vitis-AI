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

#include <xilinx/ai/configurable_dpu_task.hpp>
#include <xilinx/ai/reid.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

namespace xilinx {
namespace ai {
class ReidImp : public xilinx::ai::TConfigurableDpuTask<Reid> {
public:
  ReidImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~ReidImp();

private:
  virtual ReidResult run(const cv::Mat &image) override;
};

} // namespace ai
} // namespace xilinx
