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
#ifndef DEEPHI_SSD_HPP_
#define DEEPHI_SSD_HPP_
#include <vitis/ai/ssd.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class SSDImp : public SSD {
 public:
  SSDImp(const std::string& model_name, bool need_preprocess = true);
  SSDImp(const std::string& model_name, xir::Attrs* attrs,
         bool need_preprocess = true);
  virtual ~SSDImp();

 private:
  virtual SSDResult run(const cv::Mat& img) override;
  virtual std::vector<SSDResult> run(const std::vector<cv::Mat>& img) override;
  virtual std::vector<SSDResult> run(
      const std::vector<vart::xrt_bo_t>& input_bos) override;
  bool is_tf;
  std::unique_ptr<SSDPostProcess> processor_;
};
}  // namespace ai
}  // namespace vitis

#endif
