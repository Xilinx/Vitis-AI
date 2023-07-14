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
#include "./x_autonomous3d_post.hpp"
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
#include "./utils.hpp"

using namespace std;
using namespace vitis::ai::x_autonomous3d;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_XNNPP_LOAD_FLOAT, "0");
DEF_ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO, "0");
DEF_ENV_PARAM(DEBUG_XNNPP_PROCESS, "0");
DEF_ENV_PARAM(DEBUG_PRINT, "0");

void print_vector(const std::string& name, const vector<float>& nums, int dim,
                  int line) {
  int start = 0;
  int end = line;
  if (line < 0) {
    end = nums.size() / dim;
    start = end + line;
  }
  std::cout << "name : " << name << " " << (line > 0 ? "first" : "last")
            << std::endl;
  for (int i = start; i < end; ++i) {
    for (int j = 0; j < dim; ++j) {
      std::cout << nums[i * dim + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "name : " << name << " end" << std::endl;
}

X_Autonomous3DPost::~X_Autonomous3DPost(){};

X_Autonomous3DPost::X_Autonomous3DPost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : input_tensors_(input_tensors), output_tensors_(output_tensors) {
  std::vector<std::string> names{"reg", "height", "dim",
                                 "rot", "hm",     "iou_quality"};
  for (auto i = 0u; i < names.size(); ++i) {
    for (auto j = 0u; j < output_tensors_.size(); j++) {
      if (output_tensors_[j].name.find(names[i]) != std::string::npos) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
            << "output tensor " << names[i] << " :" << output_tensors_[j].name
            << " h: " << output_tensors_[j].height
            << " w: " << output_tensors_[j].width
            << " c: " << output_tensors_[j].channel << " scale:"
            << vitis::ai::library::tensor_scale(output_tensors_[j]);
        output_tensor_map_[names[i]] = output_tensors_[j];
        break;
      }
    }
  }

  auto batch = output_tensors_[0].batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO)) << "batch:" << batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
      << "iou_quality.size:" << output_tensor_map_["iou_quality"].size / batch;
  ;
  iou_quality_cal_result_.resize(output_tensor_map_["iou_quality"].size /
                                 batch);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
      << "heatmap.size:" << output_tensor_map_["hm"].size / batch;
  scores_.resize(output_tensor_map_["hm"].size / batch);
}

void read_bin(const std::string& file, char* dst, int size) {
  ifstream in(file, ios::binary);
  if (!in) {
    LOG(ERROR) << "read " << file << " error!";
  }
  in.read(dst, size);
}

void transform_bbox_for_nms(float* bbox, int dim, float* bbox_for_nms) {
  bbox_for_nms[0] = bbox[0];
  bbox_for_nms[1] = bbox[1];
  bbox_for_nms[2] = bbox[2];
  bbox_for_nms[3] = bbox[4];
  bbox_for_nms[4] = bbox[3];
  bbox_for_nms[6] = -1.0 * bbox[6] - HALF_PI;
}

void decode_bbox(float* reg, float* height, float* rot, float* dim, int index,
                 int H, int W, vector<float>& bbox) {
  float voxel_size_x = 0.24;
  float voxel_size_y = 0.24;
  float pc_range_low_x = -74.88;
  float pc_range_low_y = -74.88;
  float factor = 1.0f;
  /// bbox: x, y, z
  int row = index / W;
  int col = index % W;
  bbox[0] = (reg[2 * index] + col) * factor * voxel_size_x + pc_range_low_x;
  bbox[1] = (reg[2 * index + 1] + row) * factor * voxel_size_y + pc_range_low_y;
  bbox[2] = height[index];

  // bbox[3] = std::exp(dim[index * 3 + 0]);
  // bbox[4] = std::exp(dim[index * 3 + 1]);
  // bbox[5] = std::exp(dim[index * 3 + 2]);
  bbox[3] = (dim[index * 3 + 0]);
  bbox[4] = (dim[index * 3 + 1]);
  bbox[5] = (dim[index * 3 + 2]);
  bbox[6] = atan2(rot[index * 2], rot[index * 2 + 1]);
}

void decode_bbox(int8_t* reg, float reg_scale, int8_t* height,
                 float height_scale, int8_t* rot, float rot_scale, int8_t* dim,
                 float dim_scale, int index, int H, int W,
                 vector<float>& bbox) {
  float voxel_size_x = 0.24;
  float voxel_size_y = 0.24;
  float pc_range_low_x = -74.88;
  float pc_range_low_y = -74.88;
  float factor = 1.0f;
  /// bbox: x, y, z
  int row = index / W;
  int col = index % W;
  bbox[0] = (reg[2 * index] * reg_scale + col) * factor * voxel_size_x +
            pc_range_low_x;
  bbox[1] = (reg[2 * index + 1] * reg_scale + row) * factor * voxel_size_y +
            pc_range_low_y;
  bbox[2] = height[index] * height_scale;

  bbox[3] = std::exp(float(dim[index * 3 + 0] * dim_scale));
  bbox[4] = std::exp(float(dim[index * 3 + 1] * dim_scale));
  bbox[5] = std::exp(float(dim[index * 3 + 2] * dim_scale));
  bbox[6] = atan2(float(rot[index * 2] * rot_scale),
                  float(rot[index * 2 + 1] * rot_scale));
}

static bool check_range(float x, float y, float z,
                        const vector<float>& pc_range) {
  return x >= pc_range[0] && x <= pc_range[3] && y >= pc_range[1] &&
         y <= pc_range[4] && z >= pc_range[2] && z <= pc_range[5];
}

void decode_xyz(int8_t* reg, float reg_scale, int8_t* height,
                float height_scale, int index, int H, int W, float* bbox) {
  float factor = 1.0f;
  float voxel_size_x = 0.24;
  float voxel_size_y = 0.24;
  float pc_range_low_x = -74.88;
  float pc_range_low_y = -74.88;
  /// bbox: x, y, z
  int row = index / W;
  int col = index % W;
  bbox[0] = (reg[2 * index] * reg_scale + col) * factor * voxel_size_x +
            pc_range_low_x;
  bbox[1] = (reg[2 * index + 1] * reg_scale + row) * factor * voxel_size_y +
            pc_range_low_y;
  bbox[2] = height[index] * height_scale;
}

vitis::ai::X_Autonomous3DResult process_kernel(
    std::map<std::string, DataInfo>& data_map_) {
  auto reg = data_map_["reg"];
  auto height = data_map_["height"];
  auto rot = data_map_["rot"];
  auto heatmap = data_map_["hm"];
  auto dim = data_map_["dim"];

  DataInfo iou_quality;
  iou_quality = data_map_["iou_quality"];

  if (ENV_PARAM(DEBUG_PRINT)) {
    print_vector("reg before transform", reg.data, reg.shape[2], 10);
    print_vector("height before transform", height.data, height.shape[2], 10);
    print_vector("dim before transform", dim.data, dim.shape[2], 10);
    print_vector("hm before transform", heatmap.data, heatmap.shape[2], 10);
    print_vector("rot before transform", rot.data, rot.shape[2], 10);
    print_vector("iou_quality before transform", iou_quality.data,
                 iou_quality.shape[2], 10);
  }

  // 0. params
  int H = height.shape[0];
  int W = height.shape[1];
  float voxel_size_x = 0.24;
  float voxel_size_y = 0.24;
  float pc_range_low_x = -74.88;
  float pc_range_low_y = -74.88;
  float factor = 1.0f;  // out_size_factor
  int featmap_size = H * W;
  float score_thresh = 0.5;
  LOG(INFO) << "score thresh:" << score_thresh;
  // float nms_thresh = 0.55;
  vector<float> nms_thresh_per_class{0.8, 0.55, 0.55};
  int bbox_pre_size = 4096;
  // int bbox_pre_size = 1000;
  // int bbox_max_num = 500;
  int bbox_max_num = 800;
  LOG(INFO) << "bbox_pre_size:" << bbox_pre_size;
  LOG(INFO) << "bbox_max_num:" << bbox_max_num;
  vector<float> post_pc_range{-80, -80, -10, 80, 80, 10};

  // 1. hm : sigmoid
  //    dim : exp
  //    rot : atan2 // torch.atan2

  __TIC__(CENTERPOINT_WAYMO_HEATMAP_CAL)
  iou_quality_cal(iou_quality.data);
  heatmap_calculate_with_iou_quality(heatmap.data, iou_quality.data,
                                     std::vector<float>{0.68, 0.71, 0.65}, 3);
  __TOC__(CENTERPOINT_WAYMO_HEATMAP_CAL)
  __TIC__(CENTERPOINT_WAYMO_EXP)
  exp_n(dim.data);
  __TOC__(CENTERPOINT_WAYMO_EXP)

  DataInfo rot_atan2;
  __TIC__(CENTERPOINT_WAYMO_ATAN2)
  rot_atan2.resize(rot.shape[0], rot.shape[1], 1);

  atan2_n(rot_atan2.data, rot.data);
  __TOC__(CENTERPOINT_WAYMO_ATAN2)

  // 2. get x, y
  __TIC__(CENTERPOINT_WAYMO_GET_XY)
  DataInfo bbox_xy;
  bbox_xy.resize(H, W, 2);
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      bbox_xy.at(i, j, 0) =
          (reg.at(i, j, 0) + j) * factor * voxel_size_x + pc_range_low_x;
      bbox_xy.at(i, j, 1) =
          (reg.at(i, j, 1) + i) * factor * voxel_size_y + pc_range_low_y;
      // bbox_xy.at(i, j, 0) = (reg.at(i, j, 0) + j);
      // bbox_xy.at(i, j, 1) = (reg.at(i, j, 1) + i);
    }
  }
  __TOC__(CENTERPOINT_WAYMO_GET_XY)
  if (ENV_PARAM(DEBUG_PRINT)) {
    print_vector("reg after transform", reg.data, reg.shape[2], 10);
    print_vector("height after transform", height.data, height.shape[2], 10);
    print_vector("dim after transform", dim.data, dim.shape[2], 10);
    print_vector("rot atan2 after transform", rot_atan2.data,
                 rot_atan2.shape[2], 10);
    print_vector("xy after transform", bbox_xy.data, bbox_xy.shape[2], 10);
    print_vector("iou_quality after transform", iou_quality.data,
                 iou_quality.shape[2], 10);
    print_vector("hm after transform", heatmap.data, heatmap.shape[2], 10);
  }

  // 3. select score
  __TIC__(CENTERPOINT_WAYMO_SELECT_SCORE)
  vector<ScoreIndex> all_selected;  // label, score, idx
  vector<vector<ScoreIndex>> all_classes_selected(3);
  int class_num = heatmap.shape[2];
  LOG(INFO) << "class_num:" << class_num;
  for (int i = 0; i < featmap_size; ++i) {
    auto start = heatmap.data.data() + i * class_num;
    auto max_result = std::max_element(start, start + class_num);
    bool score_ok = (*max_result > score_thresh);
    bool distance_ok = check_range(*(bbox_xy.data.data() + i * 2),
                                   *(bbox_xy.data.data() + i * 2 + 1),
                                   *(height.data.data() + i), post_pc_range);
    if (!distance_ok && ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO)) {
      std::cout << "idx: " << i
                << " distance check error! x:" << *(bbox_xy.data.data() + i * 2)
                << " y:" << *(bbox_xy.data.data() + i * 2 + 1)
                << " z:" << *(height.data.data() + i) << std::endl;
    }
    // if (!score_ok || !distance_ok) {
    // if (!distance_ok) {
    bool ok = score_ok && distance_ok;
    if (!ok) {
      continue;
    }
    int label = std::distance(start, max_result);
    float score = *max_result;
    all_selected.push_back(ScoreIndex{score, label, i});
    // all_classes_selected[label].push_back(ScoreIndex{score, label, i});
  }

  if (ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO)) {
    for (auto i = 0u; i < all_selected.size(); ++i) {
      std::cout << "all selected: " << i << " score: " << all_selected[i].score
                << " label: " << all_selected[i].label
                << " index: " << all_selected[i].index << std::endl;
      ;
    }
  }
  __TIC__(CENTERPOINT_WAYMO_TOP_K)

  // 4. top k of all selected scores
  int k = topk(all_selected, bbox_pre_size);
  all_selected.resize(k);
  std::stable_sort(all_selected.begin(), all_selected.end(),
                   ScoreIndex::compare);
  __TOC__(CENTERPOINT_WAYMO_TOP_K)
  __TOC__(CENTERPOINT_WAYMO_SELECT_SCORE)

  __TIC__(CENTERPOINT_WAYMO_NMS)
  vector<ScoreIndex> all_result;
  for (auto cls = 0u; cls < all_classes_selected.size(); ++cls) {
    vector<ScoreIndex> selected;
    selected.reserve(bbox_pre_size);
    for (auto i = 0u; i < all_selected.size(); ++i) {
      if (all_selected[i].label == (int)cls) {
        selected.emplace_back(all_selected[i]);
      }
    }
    LOG(INFO) << "cls = " << cls << " selected size:" << selected.size();
    // std::stable_sort(selected.begin(), selected.end(), ScoreIndex::compare);

    // auto& bbox_pre_per_class = all_bbox_selected[i];
    vector<vector<float>> bbox_per_class(selected.size());
    vector<vector<float>> bbox_for_nms_per_class(selected.size());
    vector<float> score_per_class(selected.size());
    vector<size_t> nms_result_per_class;
    // get_pre_bbox(bbox_xy, height, dim, rot_atan2, heatmap, selected,
    //             bbox_per_class, bbox_for_nms_per_class, score_per_class);
    for (auto i = 0u; i < selected.size(); ++i) {
      vector<float> bbox_pre(7);
      vector<float> bbox_for_nms(7);
      decode_bbox(reg.data.data(), height.data.data(), rot.data.data(),
                  dim.data.data(), selected[i].index, H, W, bbox_pre);
      transform_bbox_for_nms(bbox_pre.data(), 7, bbox_for_nms.data());
      bbox_per_class[i] = bbox_pre;
      bbox_for_nms_per_class[i] = bbox_for_nms;
      score_per_class[i] = selected[i].score;
    }
    LOG(INFO) << "bbox_pre " << cls << " size" << bbox_per_class.size();
    LOG(INFO) << "score_pre " << cls << " size" << score_per_class.size();
    applyNMS(bbox_for_nms_per_class, score_per_class, nms_thresh_per_class[cls],
             score_thresh, nms_result_per_class);
    LOG(INFO) << "class " << cls << " after nms size"
              << nms_result_per_class.size();
    for (auto i = 0u; i < nms_result_per_class.size(); ++i) {
      auto selected_idx = nms_result_per_class[i];
      all_result.push_back(selected[selected_idx]);
    }
  }
  __TOC__(CENTERPOINT_WAYMO_NMS)

  __TIC__(CENTERPOINT_WAYMO_GET_RESULT)
  // 6. get result;
  k = topk(all_result, bbox_max_num);
  all_result.resize(k);
  std::stable_sort(all_result.begin(), all_result.end(), ScoreIndex::compare);

  X_Autonomous3DResult bbox_final;
  // bbox_final.bboxes.resize(nms_result.size());
  for (auto i = 0u; i < all_result.size(); i++) {
    auto& selected = all_result[i];
    auto index = selected.index;
    vector<float> bbox_pre(7);
    decode_bbox(reg.data.data(), height.data.data(), rot.data.data(),
                dim.data.data(), index, H, W, bbox_pre);
    X_Autonomous3DResult::BBox bbox;
    bbox.bbox = bbox_pre;
    bbox.label = selected.label;
    bbox.score = selected.score;
    bbox_final.bboxes.push_back(bbox);
  }
  __TOC__(CENTERPOINT_WAYMO_GET_RESULT)
  return bbox_final;
}  // namespace ai

vitis::ai::X_Autonomous3DResult X_Autonomous3DPost::process_debug_float(
    size_t batch_index) {
  __TIC__(CENTERPOINT_WAYMO_TARNSFORM)
  std::vector<std::string> names{"reg", "height", "dim",
                                 "rot", "hm",     "iou_quality"};
  std::vector<int> sizes{2, 1, 3, 2, 3, 1};
  // std::vector<float> scales{0.0078125, 0.03125, 0.03125, 0.0078125, 0.0625};
  std::vector<float> scales{0.0078125, 0.03125, 0.03125,
                            0.0078125, 0.0625,  0.0078125};
  // std::vector<float> fix_scales{128, 32, 32, 128, 16};
  std::vector<float> fix_scales{128, 32, 32, 128, 16, 128};
  // std::vector<std::vector<float>> data(5);
  for (auto i = 0u; i < names.size(); ++i) {
    LOG(INFO) << "read " << names[i];
    auto size = output_tensor_map_[names[i]].size;
    vector<float> buffer(size);
    read_bin(names[i] + ".bin", (char*)buffer.data(), size * sizeof(float));
    auto ptr = (int8_t*)output_tensor_map_[names[i]].get_data(batch_index);
    for (auto j = 0u; j < size; ++j) {
      ptr[j] = std::round(buffer[j] * fix_scales[i]);
    }
    CHECK(std::ofstream(names[i] + "_dump_normal_int8.bin")
              .write(reinterpret_cast<char*>(ptr), sizeof(int8_t) * size)
              .good());
    // print_vector(names[i], data_map_[names[i]].data, sizes[i], 10);
  }
  return process_internal(batch_index);
}

vitis::ai::X_Autonomous3DResult X_Autonomous3DPost::process_internal(
    size_t batch_index) {
  if (ENV_PARAM(DEBUG_XNNPP_PROCESS)) {
    return process_internal_debug(batch_index);
  } else {
    return process_internal_simple(batch_index);
  }
}

vitis::ai::X_Autonomous3DResult X_Autonomous3DPost::process_internal_simple(
    size_t batch_index) {
  auto reg = output_tensor_map_["reg"];
  auto height = output_tensor_map_["height"];
  auto rot = output_tensor_map_["rot"];
  auto heatmap = output_tensor_map_["hm"];
  auto dim = output_tensor_map_["dim"];
  auto iou_quality = output_tensor_map_["iou_quality"];

  int8_t* reg_tensor_ptr = (int8_t*)reg.get_data(batch_index);
  float reg_tensor_scale = vitis::ai::library::tensor_scale(reg);
  int8_t* height_tensor_ptr = (int8_t*)height.get_data(batch_index);
  float height_tensor_scale = vitis::ai::library::tensor_scale(height);
  int8_t* rot_tensor_ptr = (int8_t*)rot.get_data(batch_index);
  float rot_tensor_scale = vitis::ai::library::tensor_scale(rot);
  int8_t* dim_tensor_ptr = (int8_t*)dim.get_data(batch_index);
  float dim_tensor_scale = vitis::ai::library::tensor_scale(dim);

  int8_t* hm_tensor_ptr = (int8_t*)heatmap.get_data(batch_index);
  float hm_tensor_scale = vitis::ai::library::tensor_scale(heatmap);
  int8_t* iou_quality_tensor_ptr = (int8_t*)iou_quality.get_data(batch_index);
  float iou_quality_tensor_scale =
      vitis::ai::library::tensor_scale(iou_quality);

  // 0. params
  int H = height.height;
  int W = height.width;
  float voxel_size_x = 0.24;
  float voxel_size_y = 0.24;
  float pc_range_low_x = -74.88;
  float pc_range_low_y = -74.88;
  float out_size_factor = 1.0f;
  int featmap_size = H * W;
  float score_thresh = 0.5;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
      << "score thresh:" << score_thresh;
  vector<float> nms_thresh_per_class{0.8, 0.55, 0.55};
  int bbox_pre_size = 4096;
  // int bbox_pre_size = 1000;
  // int bbox_max_num = 500;
  int bbox_max_num = 800;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
      << "bbox_pre_size:" << bbox_pre_size;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
      << "bbox_max_num:" << bbox_max_num;
  vector<float> post_pc_range{-80, -80, -10, 80, 80, 10};
  vector<float> iou_alpha = {0.68, 0.71, 0.65};
  int class_num = 3;

  // 1. calculate score with hm and iou_quality tensor
  __TIC__(CENTERPOINT_WAYMO_HEATMAP_CAL)
  // LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
  //    << "iou_quality.size:" << iou_quality.size;
  // vector<float> iou_quality_cal_result(iou_quality.size);
  // LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
  //    << "heatmap.size:" << heatmap.size;
  // vector<float> scores(heatmap.size);
  auto& iou_quality_cal_result = iou_quality_cal_result_;
  auto& scores = scores_;
  iou_quality_cal_result.assign(iou_quality_cal_result.size(), 0);
  scores.assign(scores.size(), 0);
  __TIC__(IOU_QUALITY_CAL)
  iou_quality_cal(iou_quality_tensor_ptr, iou_quality_tensor_scale,
                  iou_quality.size / iou_quality.batch, iou_quality_cal_result);
  __TOC__(IOU_QUALITY_CAL)
  __TIC__(HEATMAP_CAL)
  heatmap_calculate_with_iou_quality(hm_tensor_ptr, hm_tensor_scale,
                                     iou_quality_cal_result, iou_alpha,
                                     class_num, scores);
  __TOC__(HEATMAP_CAL)
  __TOC__(CENTERPOINT_WAYMO_HEATMAP_CAL)

  vector<float> decoded_xyz;
  // 2. select score
  __TIC__(CENTERPOINT_WAYMO_SELECT_SCORE)
  vector<ScoreIndex> all_selected;  // label, score, idx
  vector<vector<ScoreIndex>> all_classes_selected(3);
  for (int i = 0; i < featmap_size; ++i) {
    auto start = scores.data() + i * class_num;
    auto max_result = std::max_element(start, start + class_num);
    bool score_ok = (*max_result > score_thresh);
    bool ok = score_ok;
    if (score_ok) {
      // check distance ok
      // decode x, y, z
      vector<float> xyz(3);
      int h = i / W;
      int w = i % W;
      xyz[0] = (reg_tensor_ptr[i * 2] * reg_tensor_scale + w) *
                   out_size_factor * voxel_size_x +
               pc_range_low_x;
      xyz[1] = (reg_tensor_ptr[i * 2 + 1] * reg_tensor_scale + h) *
                   out_size_factor * voxel_size_y +
               pc_range_low_y;
      xyz[2] = height_tensor_ptr[i] * height_tensor_scale;

      bool distance_ok = check_range(xyz[0], xyz[1], xyz[2], post_pc_range);
      if (!distance_ok && ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO)) {
        std::cout << "idx: " << i
                  << " distance check error! x:" << decoded_xyz[i * 3]
                  << " y:" << decoded_xyz[i * 3 + 1]
                  << " z:" << decoded_xyz[i * 3 + 2] << std::endl;
      }
      ok = (ok && distance_ok);
    }
    if (!ok) {
      continue;
    }
    int label = std::distance(start, max_result);
    float score = *max_result;
    all_selected.push_back(ScoreIndex{score, label, i});
  }

  __TIC__(CENTERPOINT_WAYMO_TOP_K)
  // 4. top k of all selected scores
  int k = topk(all_selected, bbox_pre_size);
  all_selected.resize(k);
  std::stable_sort(all_selected.begin(), all_selected.end(),
                   ScoreIndex::compare);
  __TOC__(CENTERPOINT_WAYMO_TOP_K)
  __TOC__(CENTERPOINT_WAYMO_SELECT_SCORE)

  __TIC__(CENTERPOINT_WAYMO_NMS)
  vector<ScoreIndex> all_result;
  for (auto cls = 0u; cls < all_classes_selected.size(); ++cls) {
    vector<ScoreIndex> selected;
    selected.reserve(bbox_pre_size);
    for (auto i = 0u; i < all_selected.size(); ++i) {
      if (all_selected[i].label == (int)cls) {
        selected.emplace_back(all_selected[i]);
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
        << "cls = " << cls << " selected size:" << selected.size();
    // std::stable_sort(selected.begin(), selected.end(), ScoreIndex::compare);

    vector<vector<float>> bbox_per_class(selected.size());
    vector<vector<float>> bbox_for_nms_per_class(selected.size());
    vector<float> score_per_class(selected.size());
    vector<size_t> nms_result_per_class;
    for (auto i = 0u; i < selected.size(); ++i) {
      vector<float> bbox_pre(7);
      vector<float> bbox_for_nms(7);
      decode_bbox(reg_tensor_ptr, reg_tensor_scale, height_tensor_ptr,
                  height_tensor_scale, rot_tensor_ptr, rot_tensor_scale,
                  dim_tensor_ptr, dim_tensor_scale, selected[i].index, H, W,
                  bbox_pre);
      transform_bbox_for_nms(bbox_pre.data(), 7, bbox_for_nms.data());
      bbox_per_class[i] = bbox_pre;
      bbox_for_nms_per_class[i] = bbox_for_nms;
      score_per_class[i] = selected[i].score;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
        << "bbox_pre " << cls << " size" << bbox_per_class.size();
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
        << "score_pre " << cls << " size" << score_per_class.size();
    __TIC__(CENTERPOINT_WAYMO_NMS_API)
    applyNMS(bbox_for_nms_per_class, score_per_class, nms_thresh_per_class[cls],
             score_thresh, nms_result_per_class);
    __TOC__(CENTERPOINT_WAYMO_NMS_API)
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
        << "class " << cls << " after nms size" << nms_result_per_class.size();
    for (auto i = 0u; i < nms_result_per_class.size(); ++i) {
      auto selected_idx = nms_result_per_class[i];
      all_result.push_back(selected[selected_idx]);
    }
  }
  __TOC__(CENTERPOINT_WAYMO_NMS)

  __TIC__(CENTERPOINT_WAYMO_GET_RESULT)
  // 6. get result;
  k = topk(all_result, bbox_max_num);
  all_result.resize(k);
  std::stable_sort(all_result.begin(), all_result.end(), ScoreIndex::compare);

  X_Autonomous3DResult bbox_final;
  // bbox_final.bboxes.resize(nms_result.size());
  for (auto i = 0u; i < all_result.size(); i++) {
    auto& selected = all_result[i];
    auto index = selected.index;
    vector<float> bbox_pre(7);

    decode_bbox(reg_tensor_ptr, reg_tensor_scale, height_tensor_ptr,
                height_tensor_scale, rot_tensor_ptr, rot_tensor_scale,
                dim_tensor_ptr, dim_tensor_scale, index, H, W, bbox_pre);
    X_Autonomous3DResult::BBox bbox;
    bbox.bbox = bbox_pre;
    bbox.label = selected.label;
    bbox.score = selected.score;
    bbox_final.bboxes.push_back(bbox);
  }
  __TOC__(CENTERPOINT_WAYMO_GET_RESULT)
  return bbox_final;
}

vitis::ai::X_Autonomous3DResult X_Autonomous3DPost::process_internal_debug(
    size_t batch_index) {
  std::map<std::string, DataInfo> data_map_;
  __TIC__(CENTERPOINT_WAYMO_TRANSFORM)
  std::vector<std::string> names{"reg", "height", "dim",
                                 "rot", "hm",     "iou_quality"};
  for (auto i = 0u; i < names.size(); ++i) {
    for (auto j = 0u; j < output_tensors_.size(); j++) {
      if (output_tensors_[j].name.find(names[i]) != std::string::npos) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_CENTERPOINT_WAYMO))
            << "output tensor " << names[i] << " :" << output_tensors_[j].name
            << " h: " << output_tensors_[j].height
            << " w: " << output_tensors_[j].width
            << " c: " << output_tensors_[j].channel << " scale:"
            << vitis::ai::library::tensor_scale(output_tensors_[j]);
        data_map_.insert(std::make_pair(names[i], DataInfo()));
        data_map_[names[i]].tensor_index = j;
        data_map_[names[i]].resize(output_tensors_[j].height,
                                   output_tensors_[j].width,
                                   output_tensors_[j].channel);
        data_map_[names[i]].scale =
            vitis::ai::library::tensor_scale(output_tensors_[j]);

        output_tensor_map_[names[i]] = output_tensors_[j];
        break;
      }
    }
  }

  // 0. transform output
  for (auto it = data_map_.begin(); it != data_map_.end(); ++it) {
    int size = it->second.shape[0] * it->second.shape[1] * it->second.shape[2];
    auto src =
        (int8_t*)output_tensors_[it->second.tensor_index].get_data(batch_index);
    // LOG(INFO) << "dump " << it->first << ", scale:" << it->second.scale;
    // CHECK(std::ofstream(it->first + "_dump.bin").write(reinterpret_cast<char
    // *>(src), sizeof(int8_t) * size).good());

    for (int i = 0; i < size; ++i) {
      it->second.data[i] = src[i] * it->second.scale;
    }
    // CHECK(std::ofstream(it->first +
    // "_dump_float.bin").write(reinterpret_cast<char
    // *>(it->second.data.data()), sizeof(float) * size).good());
  }
  __TOC__(CENTERPOINT_WAYMO_TRANSFORM)
  return process_kernel(data_map_);
}

std::vector<vitis::ai::X_Autonomous3DResult> X_Autonomous3DPost::process(
    size_t batch_size) {
  __TIC__(X_Autonomous3D_total_batch)

  //  auto batch_size = input_tensors_[0].batch;
  auto ret = std::vector<vitis::ai::X_Autonomous3DResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; ++i) {
    // ret.emplace_back(process_internal(i));
    if (ENV_PARAM(DEBUG_XNNPP_LOAD_FLOAT)) {
      LOG(INFO) << "postprocess load float bin";
      ret.emplace_back(process_debug_float(i));
    } else {
      ret.emplace_back(process_internal(i));
    }
  }
  __TOC__(X_Autonomous3D_total_batch)
  return ret;
}

}  // namespace ai
}  // namespace vitis
