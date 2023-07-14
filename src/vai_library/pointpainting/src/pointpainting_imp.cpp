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
#include <algorithm>
#include <utility> 
#include <cstring>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./pointpainting_imp.hpp"

using namespace std;
using namespace vitis::ai::pointpillars_nus;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_POINTPAINTING, "0");
DEF_ENV_PARAM(DEBUG_POINTPAINTING_SEG, "0");

PointPaintingImp::PointPaintingImp(const std::string &seg_name, 
                                   const std::string &pp_model_name_0, 
                                   const std::string &pp_model_name_1,
                                   bool need_preprocess)
      : seg_{vitis::ai::Segmentation::create(seg_name, true)}, 
        pp_nus_{vitis::ai::PointPillarsNuscenes::create(pp_model_name_0, pp_model_name_1)}, 
        //preprocessor_{},
        camera_num_(6), // Todo: read from config
        num_classes_(11), // Todo: read from config
        image_resize_width_(1600), // Todo: read from config
        image_resize_height_(900) { // Todo: read from config
}

PointPaintingImp::~PointPaintingImp() {
}

int PointPaintingImp::getInputWidth() const {
  return seg_->getInputWidth();
}

int PointPaintingImp::getInputHeight() const {
  return seg_->getInputHeight();
}

size_t PointPaintingImp::get_pointpillars_batch() const {
  return pp_nus_->get_input_batch();
}

size_t PointPaintingImp::get_segmentation_batch() const {
  return seg_->get_input_batch();
}

std::vector<cv::Mat> PointPaintingImp::runSegmentation(const std::vector<cv::Mat> batch_images) {
  size_t batch = seg_->get_input_batch(); // segmentation batch

  auto num = std::min(batch, batch_images.size());
  std::vector<cv::Mat> batch_result(num);
  const std::vector<cv::Mat> *input_images_ptr = &batch_images;
  std::vector<cv::Mat> input_images;
  if (batch != num) {
    std::copy(batch_images.begin(), batch_images.begin() + num, std::back_inserter(input_images));
    input_images_ptr = &input_images;
  }

  auto seg_batch_result = seg_->run_8UC1(*input_images_ptr); 
  for (auto i = 0u; i < num; ++i) {
    batch_result[i] = seg_batch_result[i].segmentation;
    if (ENV_PARAM(DEBUG_POINTPAINTING_SEG)) {
      std::string name = "image_";
      name += std::to_string(i);
      name += ".png";
      LOG(INFO) << "save image:" << name;
      cv::imwrite(name, batch_result[i]);
    }
  }

  return batch_result;
}

std::vector<float> PointPaintingImp::fusion(
        const std::vector<cv::Mat> &seg_images, const PointsInfo &points_info) {
  return this->fusion_internal(seg_images, points_info);
}

std::vector<float> PointPaintingImp::fusion_internal(
        const std::vector<cv::Mat> &seg_images, const PointsInfo &points_info) {
  std::vector<cv::Mat> fusion_images(seg_images.size());
  for (auto i = 0u; i < seg_images.size(); ++i) {
    if (seg_images[i].cols == image_resize_width_ && seg_images[i].rows == image_resize_height_) {
      fusion_images[i] = seg_images[i];
    } else {
      cv::resize(seg_images[i], fusion_images[i],
                 cv::Size(image_resize_width_, image_resize_height_), 0, 0, cv::INTER_NEAREST);
    }
  }
  return vitis::ai::pointpainting::fusion(
               points_info.cam_info, *(points_info.points.points),
               points_info.points.dim, fusion_images, num_classes_);
}


PointsInfo PointPaintingImp::runSegmentationFusion(const std::vector<cv::Mat> &input_images,
                                   const PointsInfo &points_info) {
  return run_segmentation_fusion_internal(input_images, points_info);
}

std::vector<PointsInfo> PointPaintingImp::run_segmentation_fusion_internal(
          const std::vector<std::vector<cv::Mat>> &batch_input_images,
          const std::vector<PointsInfo> &batch_points_info) {
__TIC__(SEG_FUSION)
  // 1. segmentation
  auto batch = get_pointpillars_batch();
  auto valid_batch = std::min(batch, batch_input_images.size()); 
  valid_batch = std::min(valid_batch, batch_points_info.size()); 
  auto seg_batch = get_segmentation_batch();
  if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    LOG(INFO) << "valid batch: " << valid_batch;
    LOG(INFO) << "seg batch: " << seg_batch;
  }

  std::vector<PointsInfo> batch_result(batch_points_info.begin(), batch_points_info.begin() + valid_batch);
  for (auto i = 0u; i < batch_result.size(); ++i) {
    batch_result[i].points.dim = pp_nus_->getPointsDim();
    batch_result[i].points.points.reset(new std::vector<float>);
  }

  auto valid_image_num = valid_batch * camera_num_;
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPAINTING)) << "valid image num: " << valid_image_num;
  std::vector<cv::Mat> seg_input;
  for (auto i = 0u; i < valid_batch; ++i) {
    assert(batch_input_images[i].size() >= (uint32_t)camera_num_);
    std::copy(batch_input_images[i].begin(), batch_input_images[i].begin() + camera_num_, std::back_inserter(seg_input));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPAINTING)) << "seg_input size: " << seg_input.size();

__TIC__(SEG)
  std::vector<cv::Mat> all_seg_result(valid_image_num);
  for (auto i = 0u; i < valid_image_num; i+=seg_batch) {
    auto begin = i;
    auto end = std::min(valid_image_num, i + seg_batch);
    LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPAINTING)) 
          << "seg_input begin: " << begin
          << " end: " << end; 
    std::vector<cv::Mat> seg_batch_images(seg_input.begin() + begin, seg_input.begin() + end);
    //auto seg_batch_result = seg_->run_8UC1(seg_batch_images);
    auto seg_batch_result = this->runSegmentation(seg_batch_images);
    for (auto j = 0u; j < seg_batch_result.size(); ++j) {
      //all_seg_result[begin + j] = seg_batch_result[j].segmentation;
      all_seg_result[begin + j] = seg_batch_result[j];
      LOG_IF(INFO, ENV_PARAM(DEBUG_POINTPAINTING)) 
            << "save seg batch : " << i << " reuslt : " << j
            << " to all_seg_result: " << begin + j;
    }
  }
__TOC__(SEG)
  
  // 2. segmentaion result mat resize and points fusion
__TIC__(RESIZE_AND_FUSION)
  for (auto i = 0u; i < valid_batch; ++i) {
    std::vector<cv::Mat> seg_results(all_seg_result.begin() + i * camera_num_, 
                                     all_seg_result.begin() + (i +1) * camera_num_);
    *(batch_result[i].points.points) = this->fusion_internal(seg_results, batch_points_info[i]);
  }
__TOC__(RESIZE_AND_FUSION)
__TOC__(SEG_FUSION)
  return batch_result;
}

PointsInfo PointPaintingImp::run_segmentation_fusion_internal(const std::vector<cv::Mat> &input_images,
                                            const PointsInfo &points_info) {
__TIC__(SEG_FUSION)
  auto batch = get_segmentation_batch();
  //PointsInfo result{points_info.cam_info, points_info.dim + 11, std::vector<float>(), points_info.timestamp, points_info.sweep_infos}; 
  PointsInfo result{points_info};
  result.points.dim = pp_nus_->getPointsDim();
  result.points.points.reset(new std::vector<float>);

__TIC__(SEG)
  // 1. run segmentation 
  auto num = std::min((int)input_images.size(), camera_num_);
  int group = 0;
  if (num % batch == 0) {
    group = num / batch;
  } else {
    group = num / batch + 1;
  }

  std::vector<cv::Mat> seg_results(num);
  for (auto i = 0; i < group; ++i) {
    int begin = i * batch;
    int end = std::min(num, (int)((i + 1) * batch));
    std::vector<cv::Mat> batch_images(input_images.begin() + begin,
                                      input_images.begin() + end); 
    //auto seg_result = seg_->run_8UC1(batch_images);
    auto batch_seg_result = this->runSegmentation(batch_images);
    for (auto j = 0; j < end - begin; ++j) {
      seg_results[i * batch + j] = batch_seg_result[j];
    }
  }
__TOC__(SEG)

  // 2. segmentaion mat resize and points fusion
__TIC__(RESIZE_AND_FUSION)
  *(result.points.points) = this->fusion_internal(seg_results, points_info);
__TOC__(RESIZE_AND_FUSION)
__TOC__(SEG_FUSION)
  return result;
}

PointPaintingResult 
PointPaintingImp::runPointPillars(const PointsInfo &points_info) {
  return pp_nus_->run(points_info); 
}

std::vector<PointPaintingResult> 
PointPaintingImp::runPointPillars(const std::vector<PointsInfo> &batch_points_info) {
  return pp_nus_->run(batch_points_info);
}

std::vector<PointPaintingResult> 
PointPaintingImp::run(const std::vector<std::vector<cv::Mat>> &batch_input_images,
                      const std::vector<PointsInfo> &batch_points_info) {
__TIC__(POINTPAINTING_E2E)
__TIC__(POINTPAINTING_SEG_FUSION)
  auto batch_seg_fusion_result = run_segmentation_fusion_internal(batch_input_images, batch_points_info);
  
__TOC__(POINTPAINTING_SEG_FUSION)

  if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    for (auto n = 0u; n < batch_points_info.size(); ++n) {
      auto &sweeps = batch_points_info[n].sweep_infos;
      auto &seg_fusion_result = batch_seg_fusion_result[n];
      LOG(INFO) << "seg and fusion result points size:" << seg_fusion_result.points.points->size();
      LOG(INFO) << "seg and fusion result points shape:" 
                << seg_fusion_result.points.points->size() / seg_fusion_result.points.dim
                << " * " << seg_fusion_result.points.dim;

      for (auto i = 0u; i < sweeps.size(); ++i) {
        LOG(INFO) << "sweeps[" << i << "] points size:"
                  << sweeps[i].points.points->size(); 
        LOG(INFO) << "sweeps[" << i << "] points shape:" 
                  << sweeps[i].points.points->size() / sweeps[i].points.dim
                  << " * " << sweeps[i].points.dim;
      }
    }
  }

__TIC__(POINTPAINTING_3D)
  auto batch_result = this->runPointPillars(batch_seg_fusion_result);
__TOC__(POINTPAINTING_3D)

__TOC__(POINTPAINTING_E2E)
  return batch_result;
}

PointPaintingResult 
PointPaintingImp::run(const std::vector<cv::Mat> &input_images,
                      const PointsInfo &points_info) {
__TIC__(POINTPAINTING_E2E)
__TIC__(POINTPAINTING_SEG_FUSION)
  auto seg_fusion_result = run_segmentation_fusion_internal(input_images, points_info);
__TOC__(POINTPAINTING_SEG_FUSION)
  if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    auto &sweeps = points_info.sweep_infos;
    LOG(INFO) << "seg and fusion result points size:" << seg_fusion_result.points.points->size();
    LOG(INFO) << "seg and fusion result points shape:" 
              << seg_fusion_result.points.points->size() / seg_fusion_result.points.dim
              << " * " << seg_fusion_result.points.dim;

    for (auto i = 0u; i < sweeps.size(); ++i) {
      LOG(INFO) << "sweeps[" << i << "] points size:"
                << sweeps[i].points.points->size(); 
      LOG(INFO) << "sweeps[" << i << "] points shape:" 
                << sweeps[i].points.points->size() / sweeps[i].points.dim
                << " * " << sweeps[i].points.dim;
    }
  }
__TIC__(POINTPAINTING_3D)
  //auto result = pp_nus_->run(seg_fusion_result); 
  auto result = this->runPointPillars(seg_fusion_result);
__TOC__(POINTPAINTING_3D)
__TOC__(POINTPAINTING_E2E)

  return result;
}

}}

