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
#include <vitis/ai/carplaterecog.hpp>

namespace vitis {
namespace ai {
class CarPlateRecogImp : public CarPlateRecog {
 public:
  explicit CarPlateRecogImp(
                         const std::string &cardetect_model,
                         const std::string &platedetect_model,
                         const std::string &carplaterecog_model,
                         bool need_preprocess);
  explicit CarPlateRecogImp(
                         const std::string &cardetect_model,
                         const std::string &platedetect_model,
                         const std::string &carplaterecog_model,
                         xir::Attrs *attrs,
                         bool need_preprocess);
  virtual ~CarPlateRecogImp();

 private:
  virtual CarPlateRecogResult run(const cv::Mat &image) override;
  virtual std::vector<CarPlateRecogResult> run(const std::vector<cv::Mat> &image) override;
  /// Input width(image cols)
  virtual int getInputWidth() const override;
  /// Input height(image rows)
  virtual int getInputHeight() const override;

  virtual size_t get_input_batch() const override;

  std::unique_ptr<vitis::ai::SSD> ssd_;
  std::unique_ptr<vitis::ai::PlateRecog> plate_recog_;
};
}  // namespace ai
}  // namespace vitis
