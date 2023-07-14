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
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include "vitis/ai/pointpillars_nuscenes.hpp"
#include "./voxelize.hpp"
#include <xir/attrs/attrs.hpp>

namespace vitis { namespace ai{ 

class PointPillarsNuscenesImp:  public PointPillarsNuscenes {
public:
  explicit PointPillarsNuscenesImp(const std::string &model_name_0, 
                                   const std::string &model_name_1,
                                   bool need_preprocess);
  explicit PointPillarsNuscenesImp(const std::string &model_name_0, 
                                   const std::string &model_name_1,
                                   xir::Attrs *attrs,
                                   bool need_preprocess);
  virtual ~PointPillarsNuscenesImp();

  virtual std::vector<float> sweepsFusionFilter(const vitis::ai::pointpillars_nus::PointsInfo &input) override;
  virtual std::vector<std::vector<float>> sweepsFusionFilter(
            const std::vector<vitis::ai::pointpillars_nus::PointsInfo> &batch_input) override;

  virtual PointPillarsNuscenesResult run(const std::vector<float>& points) override;
  virtual std::vector<PointPillarsNuscenesResult> run(const std::vector<std::vector<float>>& batch_points) override;
  virtual PointPillarsNuscenesResult run(const vitis::ai::pointpillars_nus::PointsInfo& input) override; 
  virtual std::vector<PointPillarsNuscenesResult> run(const std::vector<vitis::ai::pointpillars_nus::PointsInfo>& batch_input) override; 

  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual int getPointsDim() const override;
  virtual size_t get_input_batch() const override;
private:
  std::vector<std::vector<float>> sweeps_fusion_filter_internal(
          const std::vector<vitis::ai::pointpillars_nus::PointsInfo> &batch_input);
  
  std::vector<PointPillarsNuscenesResult> run_internal(const std::vector<std::vector<float>>& batch_points);
  //PointPillarsNuscenesResult run_internal(const std::vector<float>& points);
  PointPillarsNuscenesResult run_internal(const vitis::ai::pointpillars_nus::PointsInfo& input);
  std::vector<PointPillarsNuscenesResult> run_internal(const std::vector<vitis::ai::pointpillars_nus::PointsInfo>& batch_input);
  //PointPillarsNuscenesResult run_internal(const vitis::ai::pointpillars_nus::PointsInfo& input, 
  //                                       const std::vector<vitis::ai::pointpillars_nus::SweepInfo> sweeps);
  
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_0_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_1_;
  std::unique_ptr<PointPillarsNuscenesPostProcess> postprocessor_;
  
  uint32_t points_dim_;
  std::vector<float> points_range_;
  uint32_t model_in_channels_;
  std::vector<float> input_mean_;
  std::vector<float> input_scale_;
  int max_points_num_;
  int max_voxels_num_;
  std::unique_ptr<vitis::ai::pointpillars_nus::Voxelization> voxelizer_;
};

}}

