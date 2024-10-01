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
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/fusion_cnn.hpp>
#include <vitis/ai/yolovx.hpp>
#include "./clocs_pointpillars.hpp"
#include "vitis/ai/clocs.hpp"

namespace vitis {
namespace ai {

class ClocsImp : public Clocs {
 public:
  explicit ClocsImp(const std::string& yolo, const std::string& pointpilars_0,
                    const std::string& pointpilars_1,
                    const std::string& fusionnet, bool need_preprocess);
  virtual ~ClocsImp();

  virtual ClocsResult run(const clocs::ClocsInfo& input) override;
  virtual ClocsResult run(const std::vector<float>& detect2d_result,
                          const clocs::ClocsInfo& input) override;

  virtual std::vector<ClocsResult> run(
      const std::vector<clocs::ClocsInfo>& batch_inputs) override;
  virtual std::vector<ClocsResult> run(
      const std::vector<std::vector<float>>& batch_detect2d_result,
      const std::vector<clocs::ClocsInfo>& batch_inputs) override;
  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual int getPointsDim() const override;
  virtual size_t get_input_batch() const override;

 private:
  static void run_yolo(ClocsImp* instance,
                       const std::vector<cv::Mat>& batch_input, int batch_size);
  static void run_pointpillars(
      ClocsImp* instance, const std::vector<std::vector<float>>& batch_input,
      int batch_size);

  static void run_transform(
      ClocsImp* instance, const std::vector<clocs::ClocsInfo>& batch_inputs,
      std::vector<fusion_cnn::DetectResult>& batch_detect_2d_result,
      std::vector<fusion_cnn::DetectResult>& batch_detect_3d_result,
      std::vector<fusion_cnn::FusionParam>& batch_fusion_params, int batch_idx);
  static void run_transform_with2d(
      ClocsImp* instance,
      const std::vector<std::vector<float>>& batch_detect2d_ori_result,
      const std::vector<clocs::ClocsInfo>& batch_inputs,
      std::vector<fusion_cnn::DetectResult>& batch_detect_2d_result,
      std::vector<fusion_cnn::DetectResult>& batch_detect_3d_result,
      std::vector<fusion_cnn::FusionParam>& batch_fusion_params, int batch_idx);

  std::vector<ClocsResult> run_clocs(
      const std::vector<std::vector<float>>& batch_detect2d_result,
      const std::vector<clocs::ClocsInfo>& batch_inputs, size_t num);
  ClocsResult postprocess_kernel(fusion_cnn::DetectResult& fusion_result,
                                 ClocsPointPillarsResult& pp_results,
                                 size_t batch_idx);
  std::vector<ClocsResult> postprocess(
      std::vector<fusion_cnn::DetectResult>& batch_fusion_results,
      std::vector<ClocsPointPillarsResult>& batch_pp_results,
      size_t batch_size);
  std::vector<ClocsResult> run_internal(
      const std::vector<std::vector<float>>& batch_detect2d_result,
      const std::vector<clocs::ClocsInfo>& batch_inputs);
  std::vector<ClocsResult> run_internal(
      const std::vector<clocs::ClocsInfo>& batch_inputs);

 private:
  std::vector<YOLOvXResult> batch_yolo_results_;
  std::vector<ClocsPointPillarsResult> batch_pp_results_;
  std::unique_ptr<vitis::ai::YOLOvX> yolo_;
  std::unique_ptr<vitis::ai::ClocsPointPillars> pointpillars_;
  std::unique_ptr<vitis::ai::FusionCNN> fusionnet_;
};

}  // namespace ai
}  // namespace vitis

