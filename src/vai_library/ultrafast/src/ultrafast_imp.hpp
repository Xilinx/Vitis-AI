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
#ifndef DEEPHI_UltraFastIMP_HPP_
#define DEEPHI_UltraFastIMP_HPP_

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/nnpp/ultrafast.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class UltraFastImp : public vitis::ai::TConfigurableDpuTask<UltraFast> {
public:
  UltraFastImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~UltraFastImp();

private:
  virtual UltraFastResult run(const cv::Mat &img) override;
  virtual std::vector<UltraFastResult> run(const std::vector<cv::Mat> &img) override;

  int batch_size = 1;
  std::unique_ptr<UltraFastPost> processor_;
  int real_batch_size = 1;
  std::vector<cv::Size> pic_size;

};
} // namespace ai
} // namespace vitis

#endif
