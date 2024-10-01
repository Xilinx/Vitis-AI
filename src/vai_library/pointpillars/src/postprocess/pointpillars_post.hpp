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
#include <vitis/ai/pointpillars.hpp>
#include <vitis/ai/library/tensor.hpp>
#include "second/protos/pipeline.pb.h"
#include "./helper.hpp"

namespace vitis {
namespace ai {

class PointPillarsPost {
 public:
  /**
   * @brief Create an PointPillarsPostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @return An unique printer of PointPillarsPostProcess.
   */
  static std::unique_ptr<PointPillarsPost> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      std::vector<int>* g_grid_size,
      G_ANCHOR* g_anchor,
      ::second::protos::TrainEvalPipelineConfig* cfg ,
      int batchnum,
      int& realbatchnum
  );

  PointPillarsPost(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      std::vector<int>* g_grid_size,
      G_ANCHOR* g_anchor,
      ::second::protos::TrainEvalPipelineConfig* cfg ,
      int batchnum,
      int& realbatchnum
  );
  virtual ~PointPillarsPost();

  virtual PointPillarsResult post_process(int) ;
  virtual std::vector<PointPillarsResult> post_process() ;
  virtual void do_pointpillar_display(PointPillarsResult & res, int flag, DISPLAY_PARAM& g_test, cv::Mat& rgb_map, cv::Mat& bev_map, int, int, ANNORET& annoret) ;
  void get_anchors_mask( const std::vector<std::shared_ptr<preout_dict>>&) ;
  // void get_anchors_mask( std::shared_ptr<preout_dict>) ;

 private:
  void get_dpu_data(int init=0);
  void fused_get_anchors_area(int);
  void fused_get_anchors_area_thread(int, int, V1I&);
  V1F get_decode_box( int batchidx, int );
  V2I unravel_index_2d(const V1I& index, const V1I& dims );
  V3F corners_nd_2d(const V2F& dims);
  V2F center_to_corner_box2d_to_standup_nd(const V2F& box);

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  std::vector<int>* g_grid_size_;
  G_ANCHOR* g_anchor_;
  ::second::protos::TrainEvalPipelineConfig* cfg_;
  int batchnum;
  int& realbatchnum;

 private:
  DPU_DATA dpu_data;
  int cfg_nms_pre_max_size ;  
  int cfg_nms_post_max_size ; 
  float cfg_nms_iou_threshold ;
  int cfg_num_class;
  float nms_confidence_ ;

  V2I corners_norm;

  V2I anchors_maskx;  
  std::vector<MyV1I> anchors_mask;

  V1F top_scores;
  V2F box_preds; 
  V1I dir_labels;
  V1I top_labels;
  V2F dense_voxel_map;
  int anchors_mask_thread_num = 1;
 private:
  // for test part   
  void predict_kitti_to_anno(V2F&, V2F&, V2F&, V1I&, V1F& ,  ANNORET& , int, int );
  V1F cfg_post_center_limit_range;
  std::vector<std::string> cfg_class_names;
  float cfg_nms_score_threshold;
};

}  // namespace ai
}  // namespace vitis
