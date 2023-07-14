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
#include "./clocs_pointpillars_imp.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "./anchor.hpp"
#include "./scatter.hpp"
#include "./utils.hpp"

using namespace std;
using namespace vitis::ai::clocs;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_CLOCS_POINTPILLARS, "0");
DEF_ENV_PARAM(DEBUG_CLOCS_POINTPILLARS_DUMP, "0");
DEF_ENV_PARAM(DEBUG_CLOCS_POINTPILLARS_LOAD_FLOAT, "0");
DEF_ENV_PARAM(DEBUG_CLOCS_MT, "0");
DEF_ENV_PARAM(DEBUG_CLOCS_POINTPILLARS_MT, "0");

static void build_anchor_info_debug(
    AnchorInfo& anchor_info, const vitis::ai::proto::DpuModelParam& config) {
  anchor_info.featmap_size = vector<float>{1, 248, 216};
  anchor_info.sizes = vector<float>{1.6, 3.9, 1.56};
  anchor_info.strides = vector<float>{0.32, 0.32, 0};
  anchor_info.offsets = vector<float>{0.16, -39.52, -1.78};
  anchor_info.rotations = vector<float>{0., 1.57};
  anchor_info.matched_threshold = 0.6;
  anchor_info.unmatched_threshold = 0.45;
}

// static void build_anchor_info(AnchorInfo &anchor_info, const
// vitis::ai::proto::DpuModelParam& config) {
//  auto &featmap_size = config.pointpillars_nus_param().featmap_size();
//  auto &anchor_config = config.pointpillars_nus_param().anchor_info();
//  std::copy(featmap_size.begin(),
//            featmap_size.end(),
//            std::back_inserter(anchor_info.featmap_size));
//  auto anchor_ranges_size = anchor_config.ranges_size();
//  anchor_info.ranges.resize(anchor_ranges_size);
//  for (auto i = 0; i < anchor_ranges_size; ++i) {
//    std::copy(anchor_config.ranges(i).single_range().begin(),
//              anchor_config.ranges(i).single_range().end(),
//              std::back_inserter(anchor_info.ranges[i]));
//  }
//  auto sizes_ranges_size = anchor_config.sizes_size();
//  anchor_info.sizes.resize(sizes_ranges_size);
//  for (auto i = 0; i < sizes_ranges_size; ++i) {
//    std::copy(anchor_config.sizes(i).single_size().begin(),
//              anchor_config.sizes(i).single_size().end(),
//              std::back_inserter(anchor_info.sizes[i]));
//  }
//  std::copy(anchor_config.rotations().begin(),
//            anchor_config.rotations().end(),
//            std::back_inserter(anchor_info.rotations));
//  std::copy(anchor_config.custom_value().begin(),
//            anchor_config.custom_value().end(),
//            std::back_inserter(anchor_info.custom_values));
//  anchor_info.align_corner = anchor_config.align_corner();
//  anchor_info.scale = anchor_config.scale();
//}

static clocs::VoxelConfig build_voxel_config(
    const vitis::ai::proto::DpuModelParam& model_config,
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors) {
  clocs::VoxelConfig voxel_config = clocs::VoxelConfig();

  // set points dim
  voxel_config.feature_dim = input_tensors[0].channel;
  // set coors range
  std::copy(model_config.pointpillars_kitti_param().base().voxel_size().begin(),
            model_config.pointpillars_kitti_param().base().voxel_size().end(),
            std::back_inserter(voxel_config.voxels_size));

  // set voxel size
  std::copy(
      model_config.pointpillars_kitti_param()
          .base()
          .point_cloud_range()
          .begin(),
      model_config.pointpillars_kitti_param().base().point_cloud_range().end(),
      std::back_inserter(voxel_config.coors_range));

  // set in channels
  // voxel_config.in_channels =
  //    model_config.pointpillars_kitti_param().base().in_channels();
  voxel_config.in_channels = output_tensors[0].channel;

  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
      << "in_channels:" << voxel_config.in_channels;

  // set max points num
  voxel_config.max_points_num =
      model_config.pointpillars_kitti_param().base().max_points_num();
  // set max voxels num
  voxel_config.max_voxels_num =
      model_config.pointpillars_kitti_param().base().max_voxels_num();

  // set coors dim
  voxel_config.coors_dim = 4;

  // set input means and scales
  std::copy(model_config.kernel(0).mean().begin(),
            model_config.kernel(0).mean().end(),
            std::back_inserter(voxel_config.input_means));
  std::copy(model_config.kernel(0).scale().begin(),
            model_config.kernel(0).scale().end(),
            std::back_inserter(voxel_config.input_scales));
  auto input_tensor_scale = vitis::ai::library::tensor_scale(input_tensors[0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
      << "input tensor scale:" << input_tensor_scale;
  for (auto i = 0u; i < voxel_config.input_scales.size(); ++i) {
    voxel_config.input_scales[i] *= input_tensor_scale;
  }

  return voxel_config;
}

static std::map<std::string, vitis::ai::library::OutputTensor> binding_output(
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors) {
  std::map<std::string, vitis::ai::library::OutputTensor> output_tensor_map;
  std::vector<std::string> names{
      "conv_box",
      "conv_cls",
      "conv_dir",
  };
  for (auto i = 0u; i < names.size(); ++i) {
    for (auto j = 0u; j < output_tensors.size(); j++) {
      if (output_tensors[j].name.find(names[i]) != std::string::npos) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
            << "output tensor " << names[i] << " :" << output_tensors[j].name
            << " h: " << output_tensors[j].height
            << " w: " << output_tensors[j].width
            << " c: " << output_tensors[j].channel
            << " scale:" << vitis::ai::library::tensor_scale(output_tensors[j]);
        output_tensor_map[names[i]] = output_tensors[j];
        break;
      }
    }
  }
  return output_tensor_map;
}

ClocsPointPillarsImp::ClocsPointPillarsImp(const std::string& model_name_0,
                                           const std::string& model_name_1,
                                           bool need_preprocess)
    : multi_thread_(false) {
  auto attrs = xir::Attrs::create();
  model_0_ =
      ConfigurableDpuTask::create(model_name_0, attrs.get(), need_preprocess);
  model_1_ =
      ConfigurableDpuTask::create(model_name_1, attrs.get(), need_preprocess);
  voxel_config_ =
      build_voxel_config(model_0_->getConfig(), model_0_->getInputTensor()[0],
                         model_0_->getOutputTensor()[0]);

  size_t batch = get_input_batch();
  for (auto i = 0u; i < batch; ++i) {
    voxelizers_.emplace_back(
        vitis::ai::clocs::Voxelizer::create(voxel_config_));
  }
  batch_coors_.resize(batch);

  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS)) {
    auto model_0_output_tensor_scale =
        vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "model 0 output tensor scale:" << model_0_output_tensor_scale;
    auto model_1_input_tensor_scale =
        vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "model 1 input tensor scale:" << model_1_input_tensor_scale;

    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "max_points_num_:" << voxel_config_.max_points_num;
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "max_voxels_num_:" << voxel_config_.max_voxels_num;
  }

  __TIC__(GENERATE_ANCHORS)
  AnchorInfo anchor_info;
  build_anchor_info_debug(anchor_info, model_0_->getConfig());
  anchors_ = generate_anchors_stride(anchor_info);
  anchors_bv_ = get_anchors_bv(anchors_);
  __TOC__(GENERATE_ANCHORS)
  output_tensor_map_ = binding_output(model_1_->getOutputTensor()[0]);
}

ClocsPointPillarsImp::ClocsPointPillarsImp(const std::string& model_name_0,
                                           const std::string& model_name_1,
                                           xir::Attrs* attrs,
                                           bool need_preprocess)
    : multi_thread_(false) {
  model_0_ = ConfigurableDpuTask::create(model_name_0, attrs, need_preprocess);
  model_1_ = ConfigurableDpuTask::create(model_name_1, attrs, need_preprocess);

  voxel_config_ =
      build_voxel_config(model_0_->getConfig(), model_0_->getInputTensor()[0],
                         model_0_->getOutputTensor()[0]);

  size_t batch = get_input_batch();
  for (auto i = 0u; i < batch; ++i) {
    voxelizers_.emplace_back(
        vitis::ai::clocs::Voxelizer::create(voxel_config_));
  }
  batch_coors_.resize(batch);

  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS)) {
    auto model_0_output_tensor_scale =
        vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "model 0 output tensor scale:" << model_0_output_tensor_scale;
    auto model_1_input_tensor_scale =
        vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "model 1 input tensor scale:" << model_1_input_tensor_scale;

    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "max_points_num_:" << voxel_config_.max_points_num;
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
        << "max_voxels_num_:" << voxel_config_.max_voxels_num;
  }

  __TIC__(GENERATE_ANCHORS)
  AnchorInfo anchor_info;
  build_anchor_info_debug(anchor_info, model_0_->getConfig());
  anchors_ = generate_anchors_stride(anchor_info);
  anchors_bv_ = get_anchors_bv(anchors_);
  __TOC__(GENERATE_ANCHORS)
  output_tensor_map_ = binding_output(model_1_->getOutputTensor()[0]);
}

ClocsPointPillarsImp::~ClocsPointPillarsImp() {}

int ClocsPointPillarsImp::getInputWidth() const {
  return model_0_->getInputWidth();
}

int ClocsPointPillarsImp::getInputHeight() const {
  return model_0_->getInputHeight();
}

size_t ClocsPointPillarsImp::get_input_batch() const {
  return model_0_->get_input_batch();
}

int ClocsPointPillarsImp::getPointsDim() const {
  return voxel_config_.feature_dim;
}

void ClocsPointPillarsImp::setMultiThread(bool val) { multi_thread_ = val; }

void ClocsPointPillarsImp::run_preprocess_t(
    ClocsPointPillarsImp* instance, const vector<vector<float>>& batch_points,
    int batch_idx) {
  auto input_tensor_dim = instance->voxel_config_.feature_dim;
  auto batch = instance->get_input_batch();
  auto model_0_input_size =
      instance->model_0_->getInputTensor()[0][0].size / batch;

  std::memset(instance->model_0_->getInputTensor()[0][0].get_data(batch_idx), 0,
              model_0_input_size);
  auto input_ptr =
      (int8_t*)instance->model_0_->getInputTensor()[0][0].get_data(batch_idx);
  instance->batch_coors_[batch_idx] =
      instance->voxelizers_[batch_idx]->voxelize(batch_points[batch_idx],
                                                 input_tensor_dim, input_ptr,
                                                 model_0_input_size);
}

void ClocsPointPillarsImp::run_postprocess_t(
    ClocsPointPillarsImp* instance,
    vector<ClocsPointPillarsResult>& batch_results, int batch_idx) {
  auto nx = instance->model_1_->getInputTensor()[0][0].width;
  auto ny = instance->model_1_->getInputTensor()[0][0].height;

  float anchor_area_thresh = 1.0;
  auto grid_size = vector<int>{(int)nx, (int)ny, 1};

  __TIC__(CLOCS_POINTPILLARS_GET_VALID_ANCHOR_INDEX)
  auto anchor_indices = get_valid_anchor_index(
      instance->batch_coors_[batch_idx], nx, ny, instance->anchors_bv_,
      anchor_area_thresh, instance->voxel_config_.voxels_size,
      instance->voxel_config_.coors_range, grid_size);

  __TOC__(CLOCS_POINTPILLARS_GET_VALID_ANCHOR_INDEX)
  __TIC__(CLOCS_POINTPILLARS_POSTPROCESS)
  batch_results[batch_idx] =
      instance->process_internal(batch_idx, anchor_indices);
  __TOC__(CLOCS_POINTPILLARS_POSTPROCESS)
}

std::vector<ClocsPointPillarsResult> ClocsPointPillarsImp::run_internal(
    const std::vector<std::vector<float>>& batch_points) {
  __TIC__(CLOCS_POINTPILLARS_E2E)
  __TIC__(CLOCS_POINTPILLARS_PREPROCESS)
  size_t batch = get_input_batch();
  auto num = std::min(batch, batch_points.size());

  // std::vector<std::vector<int>> batch_coors(num);
  auto input_tensor_dim = voxel_config_.feature_dim;
  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
      << "input tensor dim:" << input_tensor_dim;
  auto model_0_input_size = model_0_->getInputTensor()[0][0].size / batch;

  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
      << "model_0 input size:" << model_0_input_size;
  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS_MT) || multi_thread_) {
    std::vector<std::thread> th_pre;
    for (auto i = 0u; i < num; ++i) {
      th_pre.push_back(std::thread(&run_preprocess_t, this, batch_points, i));
    }
    for (auto i = 0u; i < num; ++i) {
      th_pre[i].join();
    }
  } else {
    for (auto i = 0u; i < num; ++i) {
      std::memset(model_0_->getInputTensor()[0][0].get_data(i), 0,
                  model_0_input_size);
    }
    for (auto i = 0u; i < num; ++i) {
      auto input_ptr = (int8_t*)model_0_->getInputTensor()[0][0].get_data(i);
      batch_coors_[i] = voxelizers_[i]->voxelize(
          batch_points[i], input_tensor_dim, input_ptr, model_0_input_size);
    }
  }
  __TOC__(CLOCS_POINTPILLARS_PREPROCESS)
  __TIC__(CLOCS_POINTPILLARS_DPU_0)
  model_0_->run(0);

  __TOC__(CLOCS_POINTPILLARS_DPU_0)
  __TIC__(CLOCS_POINTPILLARS_MIDDLE_PROCESS)
  auto model_1_input_tensor_size = model_1_->getInputTensor()[0][0].size;
  auto model_1_input_size = model_1_input_tensor_size / batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
      << "model_1 input tensor size:" << model_1_input_tensor_size
      << " model_1 input size:" << model_1_input_size;
  for (auto i = 0u; i < num; ++i) {
    std::memset(model_1_->getInputTensor()[0][0].get_data(i), 0,
                model_1_input_size);
  }
  auto coors_dim = voxel_config_.coors_dim;
  auto nx = model_1_->getInputTensor()[0][0].width;
  auto ny = model_1_->getInputTensor()[0][0].height;
  auto in_channels = model_1_->getInputTensor()[0][0].channel;
  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_POINTPILLARS))
      << "nx: " << nx << ", ny: " << ny << ", in_channels:" << in_channels;
  for (auto i = 0u; i < num; ++i) {
    vitis::ai::clocs::scatter(
        batch_coors_[i], coors_dim,
        (int8_t*)model_0_->getOutputTensor()[0][0].get_data(i),
        vitis::ai::library::tensor_scale(model_0_->getOutputTensor()[0][0]),
        (int8_t*)model_1_->getInputTensor()[0][0].get_data(i),
        vitis::ai::library::tensor_scale(model_1_->getInputTensor()[0][0]),
        in_channels, nx, ny);
  }
  __TOC__(CLOCS_POINTPILLARS_MIDDLE_PROCESS)

  __TIC__(CLOCS_POINTPILLARS_DPU_1)
  model_1_->run(0);
  __TOC__(CLOCS_POINTPILLARS_DPU_1)
  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS_DUMP)) {
    auto output_tensor_size = model_1_->getOutputTensor()[0].size();
    LOG(INFO) << "model 1 output tensor size: " << output_tensor_size;
    for (auto i = 0u; i < output_tensor_size; ++i) {
      auto output_ptr = (int8_t*)model_1_->getOutputTensor()[0][i].get_data(0);
      auto size = model_1_->getOutputTensor()[0][i].size;
      std::string name = "clocs_pointpillars_model_1_output";
      name = name + std::to_string(i) + ".bin";
      CHECK(std::ofstream(name)
                .write(reinterpret_cast<const char*>(output_ptr), size)
                .good());
    }
  }

  __TIC__(CLOCS_POINTPILLARS_POSTPROCESS)
  // for (auto i = 0u; i < num; ++i) {
  //  batch_anchor_masks[i] = get_anchor_mask(
  //      batch_coors[i], nx, ny, anchors_bv_, anchor_area_thresh,
  //      voxel_config_.voxels_size, voxel_config_.coors_range, grid_size);
  //}
  std::vector<ClocsPointPillarsResult> results(num);
  // batch_results_.resize(num);
  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS_MT) || multi_thread_) {
    std::vector<std::thread> th_post;
    for (auto i = 0u; i < num; ++i) {
      th_post.push_back(
          std::thread(&run_postprocess_t, this, std::ref(results), i));
    }
    for (auto i = 0u; i < num; ++i) {
      th_post[i].join();
    }
  } else {
    float anchor_area_thresh = 1.0;
    auto grid_size = vector<int>{(int)nx, (int)ny, 1};
    // vector<vector<bool>> batch_anchor_masks(num);
    vector<vector<size_t>> batch_anchor_indices(num);
    __TIC__(CLOCS_POINTPILLARS_GET_VALID_ANCHOR_INDEX)
    for (auto i = 0u; i < num; ++i) {
      batch_anchor_indices[i] = get_valid_anchor_index(
          batch_coors_[i], nx, ny, anchors_bv_, anchor_area_thresh,
          voxel_config_.voxels_size, voxel_config_.coors_range, grid_size);
    }
    __TOC__(CLOCS_POINTPILLARS_GET_VALID_ANCHOR_INDEX)
    __TIC__(CLOCS_POINTPILLARS_POSTPROCESS)
    // auto results = postprocess(num, batch_anchor_indices);
    for (auto i = 0u; i < num; ++i) {
      // batch_results_[i] = process_internal(i, batch_anchor_indices[i]);
      results[i] = process_internal(i, batch_anchor_indices[i]);
    }

    __TOC__(CLOCS_POINTPILLARS_POSTPROCESS)
  }
  // results = batch_results_;
  __TOC__(CLOCS_POINTPILLARS_E2E)
  return results;
}

std::vector<ClocsPointPillarsResult> ClocsPointPillarsImp::postprocess(
    size_t batch_size, const vector<vector<size_t>>& batch_anchor_indices) {
  auto ret = std::vector<ClocsPointPillarsResult>{};

  for (auto i = 0u; i < batch_size; ++i) {
    ret.emplace_back(process_internal(i, batch_anchor_indices[i]));
  }
  return ret;
}

void debug_load_float(int8_t* bbox_ptr, int8_t* cls_ptr, int8_t* dir_ptr,
                      int featmap_size) {
  vector<float> bbox(featmap_size * 7);
  vector<float> cls(featmap_size);
  vector<float> dir(featmap_size * 2);
  std::ifstream("./box_1.bin")
      .read((char*)bbox.data(), bbox.size() * sizeof(float));
  std::ifstream("./cls_1.bin")
      .read((char*)cls.data(), cls.size() * sizeof(float));
  std::ifstream("./dir_1.bin")
      .read((char*)dir.data(), dir.size() * sizeof(float));
  for (auto i = 0u; i < bbox.size(); ++i) {
    bbox_ptr[i] = (int)std::round((bbox[i] * 128));
  }

  for (auto i = 0u; i < cls.size(); ++i) {
    cls_ptr[i] = (int)std::round((cls[i] * 16));
  }

  for (auto i = 0u; i < dir.size(); ++i) {
    dir_ptr[i] = (int)std::round((dir[i] * 32));
  }
}

ClocsPointPillarsResult ClocsPointPillarsImp::process_internal(
    size_t batch_index, const vector<size_t>& anchor_indices) {
  __TIC__(POST_INTERNAL)
  auto& bbox_layer = output_tensor_map_["conv_box"];
  auto& cls_layer = output_tensor_map_["conv_cls"];
  auto& dir_layer = output_tensor_map_["conv_dir"];

  int8_t* bbox_tensor_ptr = (int8_t*)bbox_layer.get_data(batch_index);
  float bbox_tensor_scale = vitis::ai::library::tensor_scale(bbox_layer);

  int8_t* cls_tensor_ptr = (int8_t*)cls_layer.get_data(batch_index);
  float cls_tensor_scale = vitis::ai::library::tensor_scale(cls_layer);

  int8_t* dir_tensor_ptr = (int8_t*)dir_layer.get_data(batch_index);
  float dir_tensor_scale = vitis::ai::library::tensor_scale(dir_layer);

  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS)) {
    vector<float> bbox_float(bbox_layer.size);
    vector<float> cls_float(cls_layer.size);
    for (auto i = 0u; i < bbox_float.size(); ++i) {
      bbox_float[i] = bbox_tensor_ptr[i] * bbox_tensor_scale;
    }

    for (auto i = 0u; i < cls_float.size(); ++i) {
      cls_float[i] = cls_tensor_ptr[i] * cls_tensor_scale;
    }
    for (auto i = 0; i < 10; ++i) {
      std::cout << "output bbox layer:";
      for (auto j = 0; j < 14; ++j) {
        std::cout << bbox_float[i * 14 + j] << " ";
      }
      std::cout << ", score:" << cls_float[i * 2] << " "
                << cls_float[i * 2 + 1];
      std::cout << endl;
    }
    for (auto i = 0; i < 10; ++i) {
      auto index = anchor_indices[i];
      std::cout << "output index:" << index;
      std::cout << ", bbox:";
      for (auto j = 0; j < 7; ++j) {
        std::cout << bbox_float[index * 7 + j] << " ";
      }
      std::cout << ", score:" << cls_float[index];
      std::cout << endl;
    }
  }
  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS_LOAD_FLOAT)) {
    auto featmap_size = cls_layer.size;
    debug_load_float(bbox_tensor_ptr, cls_tensor_ptr, dir_tensor_ptr,
                     featmap_size);
  }
  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS)) {
    LOG(INFO) << "bbox layer size:" << bbox_layer.size
              << ", scale:" << bbox_tensor_scale;
    LOG(INFO) << "cls layer size:" << cls_layer.size
              << ", scale:" << cls_tensor_scale;
    LOG(INFO) << "dir layer size:" << dir_layer.size
              << ", scale:" << dir_tensor_scale;
  }
  auto valid_size = anchor_indices.size();
  // ClocsPointPillarsMiddleResult ret;
  // ret.bboxes.resize(valid_size);
  // ret.scores.resize(valid_size);
  // ret.labels.resize(valid_size);
  ClocsPointPillarsResult ret;
  ret.bboxes.resize(valid_size);

  if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS)) {
    LOG(INFO) << "valid_size:" << valid_size;
  }
  auto box_size = 7u;
  for (auto i = 0u; i < valid_size; ++i) {
    auto index = anchor_indices[i];
    vector<float> bbox(box_size);
    for (auto j = 0u; j < box_size; ++j) {
      bbox[j] = bbox_tensor_ptr[index * box_size + j] * bbox_tensor_scale;
    }
    auto score_ori = cls_tensor_ptr[index] * cls_tensor_scale;
    // ret.scores[i] = (1. / (1. + exp(-score_ori)));
    float score = (1. / (1. + exp(-score_ori)));
    if (ENV_PARAM(DEBUG_CLOCS_POINTPILLARS)) {
      if (i < 10) {
        std::cout << "index:" << index << " ";
        std::cout << "bbox ori: ";
        for (auto j = 0u; j < box_size; ++j) {
          std::cout << bbox[j] << " ";
        }

        std::cout << "score ori:" << score_ori << " ";
        float dir_0 = (*(dir_tensor_ptr + index * 2)) * dir_tensor_scale;
        float dir_1 = (*(dir_tensor_ptr + index * 2 + 1)) * dir_tensor_scale;
        std::cout << "dir ori:" << dir_0 << " " << dir_1;
        std::cout << std::endl;
      }
    }
    decode_bbox(bbox, anchors_[index]);
    // ret.bboxes[i] = bbox;
    auto label =
        std::distance(dir_tensor_ptr + index * 2,
                      std::max_element(dir_tensor_ptr + index * 2,
                                       dir_tensor_ptr + (index + 1) * 2));
    // ret.labels[i] = label;
    ret.bboxes[i] =
        ClocsPointPillarsResult::PPBbox{score, bbox, (uint32_t)label};
  }
  __TOC__(POST_INTERNAL)
  return ret;
}

ClocsPointPillarsResult ClocsPointPillarsImp::run(
    const std::vector<float>& points) {
  std::vector<std::vector<float>> batch_points(1, points);
  return this->run_internal(batch_points)[0];
}

std::vector<ClocsPointPillarsResult> ClocsPointPillarsImp::run(
    const std::vector<std::vector<float>>& batch_points) {
  return run_internal(batch_points);
}

}  // namespace ai
}  // namespace vitis
