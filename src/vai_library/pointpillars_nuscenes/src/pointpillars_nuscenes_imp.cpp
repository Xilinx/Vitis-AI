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
#include <memory>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "./pointpillars_nuscenes_imp.hpp"
#include "./multi_frame_fusion.hpp"
#include "./scatter.hpp"

using namespace std;
using namespace vitis::ai::pointpillars_nus;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_POINTPILLARS_NUS, "0");
DEF_ENV_PARAM(DEBUG_POINTPILLARS_NUS_DUMP, "0");
DEF_ENV_PARAM(USE_PREPROCESS3, "0");
DEF_ENV_PARAM(USE_POINTPAINTING_SCALE, "0");

//static std::vector<float> pointpillars_scale = {0.02, 0.02, 0.25, 0.0078, 3.6364}; // read from config
//static std::vector<float> pointpainting_scale = {0.02, 0.02, 0.25, 0.0078, 4, 
//                                                 2, 2, 2, 2, 2, 
//                                                 2, 2, 2, 2, 2, 
//                                                 2}; // read from config
//static std::vector<float> pointpillars_mean = {0, 0, -1, 127.5, 0.275}; // read from config
//static std::vector<float> pointpainting_mean = {0, 0, -1, 127.5, 0.25, 
//                                                0.5, 0.5, 0.5, 0.5, 0.5,
//                                                0.5, 0.5, 0.5, 0.5, 0.5,
//                                                0.5}; // read from config
//
PointPillarsNuscenesImp::PointPillarsNuscenesImp(
        const std::string &model_name_0, const std::string &model_name_1,
        bool need_preprocess) {
  auto attrs = xir::Attrs::create();
  model_0_ = ConfigurableDpuTask::create(model_name_0, attrs.get(), need_preprocess); 
  model_1_ = ConfigurableDpuTask::create(model_name_1, attrs.get(), need_preprocess); 
        //preprocessor_{},
  postprocessor_ = (vitis::ai::PointPillarsNuscenesPostProcess::create(
          model_1_->getInputTensor()[0],
          model_1_->getOutputTensor()[0],
          model_0_->getConfig()));
  points_dim_ = model_0_->getInputTensor()[0][0].channel;
  //points_range_(model_0_->getConfig().pointpillars_nus_param().point_cloud_range().begin(),
  //                    model_0_->getConfig().pointpillars_nus_param().point_cloud_range().end()),
  std::copy(model_0_->getConfig().pointpillars_nus_param().point_cloud_range().begin(),
            model_0_->getConfig().pointpillars_nus_param().point_cloud_range().end(),
            std::back_inserter(points_range_));

  model_in_channels_ = model_0_->getConfig().pointpillars_nus_param().in_channels();
  //input_mean_(model_0_->getConfig().kernel(0).mean().begin(),
  //            model_0_->getConfig().kernel(0).mean().end()), 
  //input_scale_(model_0_->getConfig().kernel(0).scale().begin(),
  //             model_0_->getConfig().kernel(0).scale().end()),
  std::copy(model_0_->getConfig().kernel(0).mean().begin(),
            model_0_->getConfig().kernel(0).mean().end(),
            std::back_inserter(input_mean_));
  std::copy(model_0_->getConfig().kernel(0).scale().begin(),
            model_0_->getConfig().kernel(0).scale().end(),
            std::back_inserter(input_scale_));
  max_points_num_ = model_0_->getConfig().pointpillars_nus_param().max_points_num();
  max_voxels_num_ = model_0_->getConfig().pointpillars_nus_param().max_voxels_num();
  if (ENV_PARAM(DEBUG_POINTPILLARS_NUS)) {
    std::cout << "input scale:";
    for (auto i = 0u; i < input_scale_.size(); ++i) {
      std::cout << input_scale_[i] << " ";
    }
    std::cout << std::endl;
  
    std::cout << "input mean:";
    for (auto i = 0u; i < input_mean_.size(); ++i) {
      std::cout << input_mean_[i] << " ";
    }
    std::cout << std::endl;
  }
  auto input_tensor_scale = vitis::ai::library::tensor_scale(model_0_->getInputTensor()[0][0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "input tensor scale:" << input_tensor_scale;
  for (auto i = 0u; i < input_scale_.size(); ++i) {
    input_scale_[i] *= input_tensor_scale;
  }

  voxelizer_ = vitis::ai::pointpillars_nus::Voxelization::create(input_mean_, input_scale_,
                   max_points_num_, max_voxels_num_);

  if (ENV_PARAM(DEBUG_POINTPILLARS_NUS)) {
    auto model_0_output_tensor_scale = vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "model 0 output tensor scale:" << model_0_output_tensor_scale;
    auto model_1_input_tensor_scale = vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "model 1 input tensor scale:" << model_1_input_tensor_scale;

    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "max_points_num_:" << max_points_num_;
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "max_voxels_num_:" << max_voxels_num_;
  }
}

PointPillarsNuscenesImp::PointPillarsNuscenesImp(
        const std::string &model_name_0, const std::string &model_name_1,
        xir::Attrs *attrs, bool need_preprocess) {
  model_0_ = ConfigurableDpuTask::create(model_name_0, attrs, need_preprocess); 
  model_1_ = ConfigurableDpuTask::create(model_name_1, attrs, need_preprocess); 
        //preprocessor_{},
  postprocessor_ = (vitis::ai::PointPillarsNuscenesPostProcess::create(
          model_1_->getInputTensor()[0],
          model_1_->getOutputTensor()[0],
          model_0_->getConfig()));
  points_dim_ = model_0_->getInputTensor()[0][0].channel;
  //points_range_(model_0_->getConfig().pointpillars_nus_param().point_cloud_range().begin(),
  //                    model_0_->getConfig().pointpillars_nus_param().point_cloud_range().end()),
  std::copy(model_0_->getConfig().pointpillars_nus_param().point_cloud_range().begin(),
            model_0_->getConfig().pointpillars_nus_param().point_cloud_range().end(),
            std::back_inserter(points_range_));

  model_in_channels_ = model_0_->getConfig().pointpillars_nus_param().in_channels();
  //input_mean_(model_0_->getConfig().kernel(0).mean().begin(),
  //            model_0_->getConfig().kernel(0).mean().end()), 
  //input_scale_(model_0_->getConfig().kernel(0).scale().begin(),
  //             model_0_->getConfig().kernel(0).scale().end()),
  std::copy(model_0_->getConfig().kernel(0).mean().begin(),
            model_0_->getConfig().kernel(0).mean().end(),
            std::back_inserter(input_mean_));
  std::copy(model_0_->getConfig().kernel(0).scale().begin(),
            model_0_->getConfig().kernel(0).scale().end(),
            std::back_inserter(input_scale_));
  max_points_num_ = model_0_->getConfig().pointpillars_nus_param().max_points_num();
  max_voxels_num_ = model_0_->getConfig().pointpillars_nus_param().max_voxels_num();
  if (ENV_PARAM(DEBUG_POINTPILLARS_NUS)) {
    std::cout << "input scale:";
    for (auto i = 0u; i < input_scale_.size(); ++i) {
      std::cout << input_scale_[i] << " ";
    }
    std::cout << std::endl;
  
    std::cout << "input mean:";
    for (auto i = 0u; i < input_mean_.size(); ++i) {
      std::cout << input_mean_[i] << " ";
    }
    std::cout << std::endl;
  }
  auto input_tensor_scale = vitis::ai::library::tensor_scale(model_0_->getInputTensor()[0][0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "input tensor scale:" << input_tensor_scale;
  for (auto i = 0u; i < input_scale_.size(); ++i) {
    input_scale_[i] *= input_tensor_scale;
  }

  voxelizer_ = vitis::ai::pointpillars_nus::Voxelization::create(input_mean_, input_scale_,
                   max_points_num_, max_voxels_num_);

  if (ENV_PARAM(DEBUG_POINTPILLARS_NUS)) {
    auto model_0_output_tensor_scale = vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "model 0 output tensor scale:" << model_0_output_tensor_scale;
    auto model_1_input_tensor_scale = vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "model 1 input tensor scale:" << model_1_input_tensor_scale;

    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "max_points_num_:" << max_points_num_;
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "max_voxels_num_:" << max_voxels_num_;
  }
}

PointPillarsNuscenesImp::~PointPillarsNuscenesImp() {}

int PointPillarsNuscenesImp::getInputWidth() const {
  return model_0_->getInputWidth();
}

int PointPillarsNuscenesImp::getInputHeight() const {
  return model_0_->getInputHeight();
}

size_t PointPillarsNuscenesImp::get_input_batch() const {
  return model_0_->get_input_batch();
}

int PointPillarsNuscenesImp::getPointsDim() const {
  return points_dim_; 
}

std::vector<std::vector<float>> PointPillarsNuscenesImp::sweeps_fusion_filter_internal(const std::vector<PointsInfo> &batch_input) {
  size_t batch = get_input_batch();
  auto num = std::min(batch, batch_input.size());
  std::vector<std::vector<float>> batch_result(num);
  for (auto i = 0u; i < num; ++i) {
    auto fusion_points = vitis::ai::pointpillars_nus::multi_frame_fusion(batch_input[i], batch_input[i].sweep_infos);
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
          << "fusion points[" << i << "] size:" 
          << fusion_points.size();
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
          << "fusion points[" << i << "] shape:" 
          << fusion_points.size() / batch_input[i].points.dim
          << " * " << batch_input[i].points.dim;
    batch_result[i] = vitis::ai::pointpillars_nus::points_filter(fusion_points, batch_input[i].points.dim, points_range_); 
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
          << "filtered points[" << i << "] size:" 
          << batch_result[i].size();
  }
  return batch_result; 
}

std::vector<std::vector<float>> 
PointPillarsNuscenesImp::sweepsFusionFilter(const std::vector<PointsInfo> &batch_input) {
  return sweeps_fusion_filter_internal(batch_input); 
}

std::vector<float> 
PointPillarsNuscenesImp::sweepsFusionFilter(const PointsInfo &input) {
  std::vector<PointsInfo> batch_input(1, input);
  auto batch_filtered_points = sweeps_fusion_filter_internal(batch_input);
  return batch_filtered_points[0]; 
}

//PointPillarsNuscenesResult 
//PointPillarsNuscenesImp::run_internal(const std::vector<float>& points) {
//  __TIC__(POINTPILLARS_NUS_PREPROCESS)
//  auto batch = get_input_batch();
//  std::vector<int> coors;
//  auto input_ptr = (int8_t *)model_0_->getInputTensor()[0][0].get_data(0);
//  auto input_tensor_dim = points_dim_;
//  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "input tensor dim:" << input_tensor_dim;
//  auto model_0_input_size = model_0_->getInputTensor()[0][0].size / batch;
//  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
//        << "model_0 input size:" << model_0_input_size;
//  //coors = vitis::ai::pointpillars_nus::preprocess(points, input_tensor_dim, input_mean_, input_scale_, input_ptr);
//  coors = voxelizer_->voxelize(points, input_tensor_dim, input_ptr, model_0_input_size);
//  __TOC__(POINTPILLARS_NUS_PREPROCESS)
//  __TIC__(POINTPILLARS_NUS_DPU_0)
//  model_0_->run(0);
//  __TOC__(POINTPILLARS_NUS_DPU_0)
//  __TIC__(POINTPILLARS_NUS_MIDDLE_PROCESS)
//  auto model_1_input_size = model_1_->getInputTensor()[0][0].size / batch;
//  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
//        << "model_1 input size:" << model_1_input_size;
//  std::memset(model_1_->getInputTensor()[0][0].get_data(0), 0, model_1_input_size); 
//  auto coors_dim = 4;
//  //auto nx = 400;
//  //auto ny = 400;
//  auto nx = model_1_->getInputTensor()[0][0].width;
//  auto ny = model_1_->getInputTensor()[0][0].height;
//  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
//        << "nx: " << nx << ", ny: " << ny;
//  vitis::ai::pointpillars_nus::scatter(coors, coors_dim, 
//         (int8_t *)model_0_->getOutputTensor()[0][0].get_data(0),
//         vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]),
//         (int8_t *)model_1_->getInputTensor()[0][0].get_data(0),
//         vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]),
//         //in_channels);
//         model_in_channels_, nx, ny);
//  __TOC__(POINTPILLARS_NUS_MIDDLE_PROCESS)
//  __TIC__(POINTPILLARS_NUS_DPU_1)
//  model_1_->run(0);
//  __TOC__(POINTPILLARS_NUS_DPU_1)
//  __TIC__(POINTPILLARS_NUS_POSTPROCESS)
//  auto ret = postprocessor_->postprocess(1u);
//  __TOC__(POINTPILLARS_NUS_POSTPROCESS)
//  return ret[0];
//   
//}

std::vector<PointPillarsNuscenesResult> 
PointPillarsNuscenesImp::run_internal(const std::vector<std::vector<float>>& batch_points) {
  __TIC__(POINTPILLARS_NUS_PREPROCESS)
  size_t batch = get_input_batch();
  auto num = std::min(batch, batch_points.size());
  std::vector<std::vector<int>> batch_coors(num);
  auto input_tensor_dim = points_dim_;
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "input tensor dim:" << input_tensor_dim;
  auto model_0_input_size = model_0_->getInputTensor()[0][0].size / batch;

  for (auto i = 0u; i < num; ++i) {
    std::memset(model_0_->getInputTensor()[0][0].get_data(i), 0, model_0_input_size); 
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
        << "model_0 input size:" << model_0_input_size;
  for (auto i = 0u; i < num; ++i) {
    auto input_ptr = (int8_t *)model_0_->getInputTensor()[0][0].get_data(i);
    batch_coors[i] = voxelizer_->voxelize(batch_points[i], input_tensor_dim, input_ptr, model_0_input_size);
  } 
  __TOC__(POINTPILLARS_NUS_PREPROCESS)
  __TIC__(POINTPILLARS_NUS_DPU_0)
  model_0_->run(0);

  __TOC__(POINTPILLARS_NUS_DPU_0)
  __TIC__(POINTPILLARS_NUS_MIDDLE_PROCESS)
  auto model_1_input_tensor_size = model_1_->getInputTensor()[0][0].size;
  auto model_1_input_size = model_1_input_tensor_size / batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
        << "model_1 input tensor size:" << model_1_input_tensor_size
        << " model_1 input size:" << model_1_input_size;
  for (auto i = 0u; i < num; ++i) {
    std::memset(model_1_->getInputTensor()[0][0].get_data(i), 0, model_1_input_size); 
  }
  auto coors_dim = 4;
  auto nx = model_1_->getInputTensor()[0][0].width;
  auto ny = model_1_->getInputTensor()[0][0].height;
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
        << "nx: " << nx << ", ny: " << ny;
  for (auto i = 0u; i < num; ++i) {
    vitis::ai::pointpillars_nus::scatter(batch_coors[i], coors_dim, 
         (int8_t *)model_0_->getOutputTensor()[0][0].get_data(i),
         vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]),
         (int8_t *)model_1_->getInputTensor()[0][0].get_data(i),
         vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]),
         model_in_channels_, nx, ny);
  }
  __TOC__(POINTPILLARS_NUS_MIDDLE_PROCESS)

  __TIC__(POINTPILLARS_NUS_DPU_1)
  model_1_->run(0);
  __TOC__(POINTPILLARS_NUS_DPU_1)
  if (ENV_PARAM(DEBUG_POINTPILLARS_NUS_DUMP)) {
    auto output_tensor_size = model_1_->getOutputTensor()[0].size();
    LOG(INFO) << "model 1 output tensor size: " << output_tensor_size;
    for (auto i = 0u; i < output_tensor_size; ++i) {
      auto output_ptr = (int8_t *)model_1_->getOutputTensor()[0][i].get_data(0);
      auto size = model_1_->getOutputTensor()[0][i].size;
      std::string name = "pointpillars_nus_model_1_output";
      name = name + std::to_string(i) + ".bin";
      CHECK(std::ofstream(name).write(reinterpret_cast<const char *>(output_ptr), size).good()); 
    }
  }

  __TIC__(POINTPILLARS_NUS_POSTPROCESS)
  auto results = postprocessor_->postprocess(num);
  __TOC__(POINTPILLARS_NUS_POSTPROCESS)
  return results;
}

PointPillarsNuscenesResult 
PointPillarsNuscenesImp::run_internal(const PointsInfo &input) {
  __TIC__(POINTPILLARS_NUS_E2E)
  __TIC__(POINTPILLARS_NUS_SWEEPS_FUSION)
  std::vector<PointsInfo> batch_input(1, input);
  auto batch_points = sweeps_fusion_filter_internal(batch_input); 
  __TOC__(POINTPILLARS_NUS_SWEEPS_FUSION)
  auto results = run_internal(batch_points);
  __TOC__(POINTPILLARS_NUS_E2E)
  return results[0];
}

std::vector<PointPillarsNuscenesResult> 
PointPillarsNuscenesImp:: run_internal(const std::vector<PointsInfo>& batch_input) {
  __TIC__(POINTPILLARS_NUS_E2E)
  __TIC__(POINTPILLARS_NUS_SWEEPS_FUSION)
  auto batch_points = sweeps_fusion_filter_internal(batch_input); 
  __TOC__(POINTPILLARS_NUS_SWEEPS_FUSION)
  auto results = run_internal(batch_points);
  __TOC__(POINTPILLARS_NUS_E2E)
  return results;
}

//PointPillarsNuscenesResult 
//PointPillarsNuscenesImp::run_internal(const PointsInfo &input,
//                                      const std::vector<SweepInfo> sweep_infos) {
//  __TIC__(POINTPILLARS_NUS_E2E)
//  __TIC__(POINTPILLARS_NUS_SWEEPS_FUSION)
//  //std::vector<float> range{-50.0, -50.0, -5.0, 50.0, 50.0, 3.0};
//  auto input_points = vitis::ai::pointpillars_nus::multi_frame_fusion(input, sweep_infos);  
//  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
//        << "input points size:" << input_points.size();
//  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) 
//        << "input points shape:" << input_points.size() / input.points.dim
//        << " * " << input.points.dim;
//  __TOC__(POINTPILLARS_NUS_SWEEPS_FUSION)
//  __TIC__(POINTPILLARS_NUS_FILTER)
//  //auto points = vitis::ai::pointpillars_nuscenes::points_filter(input_points, input.dim, range); 
//  auto points = vitis::ai::pointpillars_nus::points_filter(input_points, input.points.dim, points_range_); 
//  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPILLARS_NUS)) << "filtered points size:" << points.size();
//  __TOC__(POINTPILLARS_NUS_FILTER)
//
//  auto ret = run_internal(points);
//  __TOC__(POINTPILLARS_NUS_E2E)
//
//  return ret; 
//}

PointPillarsNuscenesResult PointPillarsNuscenesImp::run(const std::vector<float>& points) {
  std::vector<std::vector<float>> batch_points(1, points);
  return (this->run_internal(batch_points))[0]; 
} 

std::vector<PointPillarsNuscenesResult> 
PointPillarsNuscenesImp::run(const std::vector<std::vector<float>> &batch_points) {
  return this->run_internal(batch_points); 
}

//PointPillarsNuscenesResult PointPillarsNuscenesImp::run(
//        const vitis::ai::pointpillars_nus::PointsInfo& input, 
//        const std::vector<vitis::ai::pointpillars_nus::SweepInfo> sweeps) {
//  return this->run_internal(input, sweeps);
//}

PointPillarsNuscenesResult 
PointPillarsNuscenesImp::run(const PointsInfo &input) {
  return this->run_internal(input);
}

std::vector<PointPillarsNuscenesResult> 
PointPillarsNuscenesImp::run(const std::vector<PointsInfo> &batch_input) {
  return this->run_internal(batch_input);
}

}}

