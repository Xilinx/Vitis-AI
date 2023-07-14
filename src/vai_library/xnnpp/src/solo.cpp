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
#ifndef OPENPOSE_UTIL_HPP
#define OPENPOSE_UTIL_HPP

#include "vitis/ai/nnpp/solo.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/profiling.hpp>
#include <vector>
#include <fstream>
DEF_ENV_PARAM(DEBUG_SOLO, "0");
DEF_ENV_PARAM(SOLO_2STAGE_INTERPOLATE, "0");
int enable_2stage_interpolate = 0;

using namespace std;
using namespace cv;

namespace vitis {
namespace ai {


std::vector<size_t> trans_size{1600, 2000, 2400, 2500, 2600};
// std::vector<size_t> seg_size{40, 76, 100, 116, 128};
std::vector<size_t> seg_size{40,  60,  80,  90, 100};
//std::vector<size_t> seg_num_grids{40, 36, 24, 16, 12};
std::vector<size_t> seg_num_grids{40, 20, 20, 10, 10};
//std::vector<size_t> stride{8, 8, 16, 32, 32};
std::vector<size_t> stride{8, 8, 16, 32, 32 };
size_t n_stage = 5;
//auto score_thr = 0.1f;
//auto mask_thr = 0.5f;
//auto nms_pre = 500u;
//auto update_thr = 0.05f;
//auto max_per_img = 100u;

SoloResult solo_post_process(
    const std::vector<vitis::ai::library::InputTensor>&
        input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, 
        std::vector<int> size, std::vector<int> ind, size_t batch_ind) {
  CHECK_EQ(output_tensors.size(), ind.size());
  std::vector<Ndarray<float>> dpu_output(ind.size());
  __TIC__(dpu_ouputdata_baseprocess1)
  for(auto i = 0u; i < ind.size(); i++) {
    auto H = output_tensors[ind[i]].height;
    auto W = output_tensors[ind[i]].width;
    auto C = output_tensors[ind[i]].channel;
    auto name = output_tensors[ind[i]].name;
    //auto t_size = height * width * channel;
    auto outscale = vitis::ai::library::tensor_scale(output_tensors[ind[i]]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO))  << name << " " << H << " " << W << " " << C << " " << outscale << " " << batch_ind;
    dpu_output[i].zeros({C,H,W});
    //dpu_output[i].print();
    //auto a = 0;
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        for (auto c = 0u; c < C; c++) {
          dpu_output[i][c*H*W + h*W + w] = ((int8_t*)output_tensors[ind[i]].get_data(batch_ind))[h*W*C +w*C + c] * outscale;
        }
      }  
    }
  }
  __TOC__(dpu_ouputdata_baseprocess1)
  __TIC__(dpu_ouputdata_baseprocess2)
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO))  << "permute over";
  Ndarray<float> seg_preds_x;
  Ndarray<float> seg_preds_y;
  Ndarray<float> cate_preds;
  seg_preds_x.zeros({0, 160, 160});
  seg_preds_y.zeros({0, 160, 160});
  cate_preds.zeros({0, 80});
  for(auto i = 0u; i < 5u; i++) {
    seg_preds_x.extend(dpu_output[i]);
  }
  //seg_preds_x.print();
  //pr("spx ", seg_preds_x, 0,20);
  for(auto i = 5u; i < 10u; i++) {
    seg_preds_y.extend(dpu_output[i]);
  //pr("spy ", dpu_output[i], 0,20);
  }
  //seg_preds_y.print();
  for(auto i = 10u; i < 15u; i++) {
    auto hmax = dpu_output[i];
    //hmax.print();
    //pr("nonzeor ", hmax.nonzero(),0, 10);
    auto keep = dpu_output[i].slice({{0, dpu_output[i].shape[0]}, {0,dpu_output[i].shape[1] - 1}, {0, dpu_output[i].shape[2] - 1}}) 
                                == dpu_output[i + 5];
    //keep.print("keep");
    auto cate_preds_temp0 = dpu_output[i + 5] * keep._float();
    //cate_preds_temp0.print("cpt0");
    Ndarray<float> cate_preds_temp1;
    cate_preds_temp1.zeros(cate_preds_temp0.shape);
    auto _C = cate_preds_temp0.shape[0];
    auto _H = cate_preds_temp0.shape[1];
    auto _W = cate_preds_temp0.shape[2];
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO))  << _C << " " << _H << " " << _W;
    for (auto c = 0u; c < _C; c++) {
      for (auto h = 0u; h < _H; h++) {
        for (auto w = 0u; w < _W; w++) {
          cate_preds_temp1[h*_W*_C + w*_C + c] = cate_preds_temp0[c*_H*_W + h*_W + w];
        }
      }  
    }
    //cate_preds_temp1.print("cpt1");
    cate_preds_temp1.reshape({_H*_W,_C});
    //cate_preds_temp1.print("cpt1");
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO))  << "extend";
    cate_preds.extend(cate_preds_temp1);
  //cate_preds.print("cp");
  //pr("usuk", cate_preds_temp1, 0, 100);
  }
  

  __TOC__(dpu_ouputdata_baseprocess2)
  __TIC__(prepare_trans_data)
  Ndarray<int> trans_diff;
  trans_diff.ones({trans_size[4]});
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << "trans_diff ";
  //trans_diff.print();
  Ndarray<int> num_grids;
  num_grids.ones({trans_size[4]});
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << "num_grids ";
  //num_grids.print();
  Ndarray<int> seg_diff;
  seg_diff.ones({trans_size[4]});
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << "seg_diff ";
  //seg_diff.print();
  Ndarray<int> strides;
  strides.ones({trans_size[4]});
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << "strides ";
  //strides.print();
  trans_diff.mul({0,trans_size[0]}, 0);
  seg_diff.mul({0,trans_size[0]}, 0);
  num_grids.mul({0,trans_size[0]}, seg_num_grids[0]);
  strides.mul({0,trans_size[0]}, stride[0]);
  // std::cout << strides[1599] << " " << strides[1600] << std::endl;
  for(auto ind_ = 1u; ind_ < n_stage; ind_++) {
    // std::cout << trans_size[ind_ -1] << " " << trans_size[ind_] << std::endl;
    trans_diff.mul({trans_size[ind_ -1], trans_size[ind_]}, trans_size[ind_ - 1]);
    seg_diff.mul({trans_size[ind_ -1], trans_size[ind_]}, seg_size[ind_ - 1]);
    num_grids.mul({trans_size[ind_ -1], trans_size[ind_]}, seg_num_grids[ind_]);
    strides.mul({trans_size[ind_ -1], trans_size[ind_]}, stride[ind_]);
  }
  __TOC__(prepare_trans_data)
  __TIC__(get_inds)
  auto inds = cate_preds.filter(config.solo_param().score_thr());
  __TOC__(get_inds)
  __TIC__(bool_select)
  auto cate_scores = cate_preds.bool_select(inds);
  __TOC__(bool_select)
  __TIC__(nonzero)
  auto inds1 = inds.nonzero();
  __TOC__(nonzero)
  __TIC__(get_inds0)
  auto inds2 = inds1.slice({{0, inds1.shape[0]}, {0, 1}});
  __TOC__(get_inds0)
  __TIC__(index_select)
  trans_diff = trans_diff.index_select(0, inds2);
  seg_diff = seg_diff.index_select(0, inds2);
  num_grids = num_grids.index_select(0, inds2);
  strides = strides.index_select(0, inds2);
  inds2.reshape({inds1.shape[0]});
  __TOC__(index_select)
  __TIC__(get_xy_inds)
  auto y_inds = (inds2 - trans_diff) / num_grids + seg_diff;
  auto x_inds = (inds2 - trans_diff) % num_grids + seg_diff;
  __TOC__(get_xy_inds)
  __TIC__(get_cate_labels)
  auto cate_labels = inds1.slice({{0, inds1.shape[0]}, {1, 2}});
  __TOC__(get_cate_labels)
  __TIC__(get_seg_masks_soft)
  auto seg_masks_soft = seg_preds_y.slice_dim0(y_inds) * seg_preds_x.slice_dim0(x_inds); 
  __TOC__(get_seg_masks_soft)
  __TIC__(get_seg_masks)
  auto seg_masks = seg_masks_soft.filter(config.solo_param().mask_thr());
  __TOC__(get_seg_masks)
  __TIC__(get_sum_masks)
  auto sum_masks = seg_masks.reduce_todim1(size_t(0));
  __TOC__(get_sum_masks)
  __TIC__(get_keep)
  auto keep = sum_masks > strides;
  __TOC__(get_keep)
  __TIC__(sms_by_keep)
  seg_masks_soft = seg_masks_soft.slice_dim0(keep.nonzero()); //nocheck
  __TOC__(sms_by_keep)
  __TIC__(sm_by_keep)
  seg_masks = seg_masks.slice_dim0(keep.nonzero()); //nocheck
  __TOC__(sm_by_keep)
  __TIC__(cssmcl_by_keep)
  cate_scores = cate_scores.bool_select(keep);
  sum_masks = sum_masks.bool_select(keep);
  cate_labels = cate_labels.bool_select(keep);
  __TOC__(cssmcl_by_keep)
  __TIC__(get_seg_score)
  auto seg_score = (seg_masks_soft * seg_masks._float()).reduce_todim1(float(0)) / sum_masks._float();
  __TOC__(get_seg_score)
  __TIC__(cate_scoresXseg_score)
  cate_scores = cate_scores * seg_score;
  __TOC__(cate_scoresXseg_score)
  __TIC__(get_sort_inds)
  auto sort_inds = cate_scores.argsort(1);
  if (sort_inds.size() > config.solo_param().nms_pre()) {
    sort_inds.resize(config.solo_param().nms_pre());
    sort_inds.reshape({config.solo_param().nms_pre()});
  }
  __TOC__(get_sort_inds)
  __TIC__(smssm_by_si)
  seg_masks_soft = seg_masks_soft.slice_dim0(sort_inds);
  seg_masks = seg_masks.slice_dim0(sort_inds);
  __TOC__(smssm_by_si)
  __TIC__(cssmcl_by_si)
  cate_scores = cate_scores.select(sort_inds);
  sum_masks = sum_masks.select(sort_inds);
  cate_labels = cate_labels.select(sort_inds);
  __TOC__(cssmcl_by_si)
  //// Matrix nms
  __TIC__(get_seg_masks_trans)
  auto n_samples =cate_labels.size();
  seg_masks.reshape({n_samples, seg_masks.shape[1] * seg_masks.shape[2]});
  auto seg_masks_trans = seg_masks.transpose2d();
  __TOC__(get_seg_masks_trans)
  __TIC__(mm)
  auto inter_matrix = mm(seg_masks, seg_masks_trans);
  __TOC__(mm)
  __TIC__(get_sum_masks_x)
  auto sum_masks_x = sum_masks.expand(n_samples);
  __TOC__(get_sum_masks_x)
  __TIC__(get_iou_matrix)
  auto iou_matrix = (inter_matrix._float()/ (sum_masks_x._float() + sum_masks_x.transpose2d()._float() - inter_matrix._float())).triu(1);
  __TOC__(get_iou_matrix)
  __TIC__(get_cate_labels_x)
  auto cate_labels_x = cate_labels.expand(n_samples);
  __TOC__(get_cate_labels_x)
  __TIC__(get_label_matrix)
  auto label_matrix = (cate_labels_x == cate_labels_x.transpose2d())._float().triu(1);
  __TOC__(get_label_matrix)
  __TIC__(get_decay_iou)
  auto decay_iou = iou_matrix * label_matrix;
  __TOC__(get_decay_iou)
  __TIC__(get_compensate_iou)
  auto compensate_iou = decay_iou.transpose2d().max().expand(n_samples).transpose2d();    
  __TOC__(get_compensate_iou)
  __TIC__(get_decay_matrix)
  auto decay_matrix = _exp(-2.0f * decay_iou.pow(2));
  __TOC__(get_decay_matrix)
  __TIC__(get_compensate_matrix)
  auto compensate_matrix = _exp(-2.0f * compensate_iou.pow(2));
  __TOC__(get_compensate_matrix)
  __TIC__(get_decay_coefficient)
  auto decay_coefficient = (decay_matrix/compensate_matrix).transpose2d().min();
  __TOC__(get_decay_coefficient)
  __TIC__(ud_cate_scores)
  cate_scores = cate_scores * decay_coefficient;
  __TOC__(ud_cate_scores)
  __TIC__(ud_keep)
  keep = cate_scores.filter(config.solo_param().update_thr());
  __TOC__(ud_keep)
  __TIC__(ud_sms)
  seg_masks_soft = seg_masks_soft.slice_dim0(keep, true);
  __TOC__(ud_sms)
  __TIC__(ud_cscl)
  cate_scores = cate_scores.bool_select(keep);
  cate_labels = cate_labels.bool_select(keep);
  __TOC__(ud_cscl)
  __TIC__(ud_si)
  sort_inds = cate_scores.argsort(1);
  if (sort_inds.size() > config.solo_param().max_per_img()) {
    sort_inds.resize(config.solo_param().max_per_img());
    sort_inds.reshape({config.solo_param().max_per_img()});
  }
  __TOC__(ud_si)
  __TIC__(ud_sms_by_si)
  seg_masks_soft = seg_masks_soft.slice_dim0(sort_inds);
  __TOC__(ud_sms_by_si)
  __TIC__(ud_cscl_by_si)
  cate_scores = cate_scores.select(sort_inds);
  cate_labels = cate_labels.select(sort_inds);
  __TOC__(ud_cscl_by_si)
  __TIC__(interpolate)
  Ndarray<float> seg_masks_interd = seg_masks_soft;
  if(ENV_PARAM(SOLO_2STAGE_INTERPOLATE) == 1)
    enable_2stage_interpolate = 1;
  if (enable_2stage_interpolate == 1)
  seg_masks_interd = interpolate(seg_masks_interd, 640, 640);
  //LOG(INFO) << size[1] << " " << size[0];
  seg_masks_interd = interpolate(seg_masks_interd,size[1],size[0]);
  __TOC__(interpolate)
  __TIC__(ud_sm)
  seg_masks = seg_masks_interd.filter(config.solo_param().mask_thr());
  __TOC__(ud_sm)
  //cate_preds.print();
  SoloResult result{640, 640, seg_masks, cate_labels, cate_scores};
  return result;
}

std::vector<SoloResult> solo_post_process_batch(
    const std::vector<vitis::ai::library::InputTensor>&
        input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config,
        std::vector<std::vector<int>> sizes, std::vector<int> ind) {
    
  auto batch = input_tensors[0].batch;
  auto ret = std::vector<SoloResult>{};
  for (auto bs = 0u; bs < batch; bs++) {
    ret.push_back(solo_post_process(input_tensors, output_tensors, config, sizes[bs], ind, bs));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
#endif


