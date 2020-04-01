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
#ifndef DEEPHI_TFSSD_HPP_
#define DEEPHI_TFSSD_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/tfssd.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class TFSSDImp : public vitis::ai::TConfigurableDpuTask<TFSSD> {
public:
  TFSSDImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~TFSSDImp();

private:
  virtual TFSSDResult run(const cv::Mat &img) override;
  virtual std::vector<TFSSDResult> run(const std::vector<cv::Mat> &img) override;
  std::unique_ptr<TFSSDPostProcess> processor_;
};
} // namespace ai
} // namespace vitis

#endif
