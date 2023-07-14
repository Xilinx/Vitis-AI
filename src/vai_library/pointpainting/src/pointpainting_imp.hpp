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
#include <vitis/ai/pointpainting.hpp>
#include <vitis/ai/segmentation.hpp>
#include <vitis/ai/pointpillars_nuscenes.hpp>
#include "./fusion.hpp"

using namespace vitis::ai::pointpillars_nus;

namespace vitis { namespace ai{ 

class PointPaintingImp:  public PointPainting {
public:
  PointPaintingImp(const std::string &seg_model_name, 
                   const std::string &pp_model_name_0, 
                   const std::string &pp_model_name_1,
                   bool need_preprocess = true);
  virtual ~PointPaintingImp();

  virtual PointPaintingResult run(const std::vector<cv::Mat> &input_images,
                                  const PointsInfo &points_info) override;
  virtual std::vector<PointPaintingResult> run(
            const std::vector<std::vector<cv::Mat>> &batch_input_images,
            const std::vector<PointsInfo> &batch_points_info) override;

  virtual std::vector<cv::Mat> runSegmentation(std::vector<cv::Mat> batch_images) override;
  
  virtual std::vector<float> fusion(const std::vector<cv::Mat> &seg_images, 
                                    const PointsInfo &points_info) override;
  //virtual std::vector<std::vector<float>> fusion(
  //          const std::vector<cv::Mat> &batch_seg_images, 
  //          const PointsInfo &batch_points_info) override;

  virtual PointsInfo runSegmentationFusion(const std::vector<cv::Mat> &input_images,
                                           const PointsInfo &points_info) override;
  //virtual std::vector<PointsInfo> runSegmentationFusion(
  //           const std::vector<std::vector<cv::Mat>> &batch_input_images, 
  //           const std::vector<PointsInfo> &batch_points_info) override;

  virtual PointPaintingResult runPointPillars(const PointsInfo &points_info) override;
  virtual std::vector<PointPaintingResult> runPointPillars(
            const std::vector<PointsInfo> &batch_points_info) override;

  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual size_t get_pointpillars_batch() const override;
  virtual size_t get_segmentation_batch() const override;
  //virtual int getSegImageNum() const override;
private:
   
  std::vector<float> fusion_internal(const std::vector<cv::Mat> &seg_images, 
                                     const PointsInfo &points_info);
  PointsInfo run_segmentation_fusion_internal(const std::vector<cv::Mat> &input_images,
                                 const PointsInfo &points);
  std::vector<PointsInfo> run_segmentation_fusion_internal(
          const std::vector<std::vector<cv::Mat>> &batch_input_images,
          const std::vector<PointsInfo> &batch_points_info);
  std::unique_ptr<vitis::ai::Segmentation> seg_;
  std::unique_ptr<vitis::ai::PointPillarsNuscenes> pp_nus_;
  int camera_num_;
  int num_classes_;
  int image_resize_width_; 
  int image_resize_height_; 
};

}}

