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

#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vitis/ai/configurable_dpu_task.hpp>
#include "./anchor.hpp"
#include "./clocs_pointpillars.hpp"
#include "./voxelizer.hpp"

namespace xir {
class Attrs;
};

namespace vitis {
namespace ai {

class ClocsPointPillarsImp : public ClocsPointPillars {
 public:
  explicit ClocsPointPillarsImp(const std::string& model_name_0,
                                const std::string& model_name_1,
                                bool need_preprocess);

  explicit ClocsPointPillarsImp(const std::string& model_name_0,
                                const std::string& model_name_1,
                                xir::Attrs* attrs, bool need_preprocess);
  virtual ~ClocsPointPillarsImp();

  virtual ClocsPointPillarsResult run(
      const std::vector<float>& points) override;

  virtual std::vector<ClocsPointPillarsResult> run(
      const std::vector<std::vector<float>>& batch_points) override;

  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual int getPointsDim() const override;
  virtual size_t get_input_batch() const override;

  virtual void setMultiThread(bool val) override;

 private:
  std::vector<ClocsPointPillarsResult> run_internal(
      const std::vector<std::vector<float>>& batch_points);

  std::vector<ClocsPointPillarsResult> postprocess(
      size_t batch_size, const vector<vector<size_t>>& batch_anchor_indices);

  ClocsPointPillarsResult process_internal(
      size_t batch_index, const vector<size_t>& anchor_indices);
  // ClocsPointPillarsMiddleResult run_internal(const std::vector<float>&
  // points);
 private:
  static void run_preprocess_t(ClocsPointPillarsImp* instance,
                               const vector<vector<float>>& batch_points,
                               int batch_idx);
  static void run_postprocess_t(ClocsPointPillarsImp* instance,
                                vector<ClocsPointPillarsResult>& batch_results,
                                int batch_idx);

 private:
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_0_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_1_;
  // std::unique_ptr<ClocsPointPillarsPostProcess> postprocessor_;
  // std::unique_ptr<char*> postprocessor_;
  vitis::ai::clocs::VoxelConfig voxel_config_;

  // uint32_t points_dim_;
  // std::vector<float> points_range_;
  // uint32_t model_in_channels_;
  // std::vector<float> input_mean_;
  // std::vector<float> input_scale_;
  // int max_points_num_;
  // int max_voxels_num_;
  std::vector<std::unique_ptr<vitis::ai::clocs::Voxelizer>> voxelizers_;
  vitis::ai::clocs::Anchors anchors_;
  vitis::ai::clocs::Anchors anchors_bv_;

  std::vector<vitis::ai::library::InputTensor> input_tensors_;
  std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  std::map<std::string, vitis::ai::library::OutputTensor> output_tensor_map_;
  std::vector<std::vector<int>> batch_coors_;
  bool multi_thread_;
  // std::vector<ClocsPointPillarsResult> batch_results_;
};

}  // namespace ai
}  // namespace vitis

