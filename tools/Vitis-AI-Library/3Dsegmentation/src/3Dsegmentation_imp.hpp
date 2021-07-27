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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/3Dsegmentation.hpp>

namespace vitis {
namespace ai {

class Segmentation3DImp : public vitis::ai::TConfigurableDpuTask<Segmentation3D> {
 public:
  explicit Segmentation3DImp(const std::string &model_name, bool need_preprocess);
  virtual ~Segmentation3DImp();

 private:
  virtual Segmentation3DResult run(std::vector<std::vector<float>> &array) override;
  virtual std::vector<Segmentation3DResult> run(std::vector<std::vector<std::vector<float>>>& arrays) override;
  const std::vector<float> sensor_means_{12.12, 10.88, 0.23, -1.04, 0.21};
  const std::vector<float> sensor_stds_{12.32, 11.47, 6.91, 0.86, 0.16};
  std::vector<int>  map_inv_{0, 10, 11, 15, 18,
                         20, 30, 31, 32, 40,
                         44, 48, 49, 50, 51,
                         70, 71, 72, 80, 81};
};
}  // namespace ai
}  // namespace vitis
