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
#pragma once

#include "vitis/ai/RGBDsegmentation.hpp"

namespace vitis {
namespace ai {

class RGBDsegmentationImp : public RGBDsegmentation {
 public:
  RGBDsegmentationImp(const std::string& model_name,
                      bool need_preprocess = true);
  virtual ~RGBDsegmentationImp();

 private:
  virtual SegmentationResult run(const cv::Mat& image_bgr,
                                 const cv::Mat& image_hha) override;
};

}  // namespace ai
}  // namespace vitis
