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

#include "./multitaskv3_imp.hpp"

#include "./prior_boxes.hpp"
#ifdef HAVE_NEON
#include <arm_neon.h>
#endif
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/math.hpp>
#include <vitis/ai/max_index.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;

namespace vitis {
namespace ai {
static void get_depth(int8_t* src, float* dst, size_t length, float scale) {
  for (size_t i = 0; i < length; ++i) {
    auto tmp = (float)src[i] * scale;
    dst[i] = (1.f / (1.f + exp(-tmp))) * 80.f;
    dst[i] = std::min(dst[i], 80.f);
    dst[i] = std::max(dst[i], 1.f);
    dst[i] *= 3;
  }
}

static void get_depth_ori(int8_t* src, float* dst, size_t length, float scale) {
  for (size_t i = 0; i < length; ++i) {
    auto tmp = (float)src[i] * scale;
    dst[i] = tmp;
  }
}

static void sigmoid_n(int8_t* src, float* dst, size_t length, float scale) {
  for (size_t i = 0; i < length; ++i) {
    auto tmp = (float)src[i] * scale;
    dst[i] = (1.0f / (1.0f + exp(-tmp)));
  }
}

static void convert_color(std::string& src, std::vector<uint8_t>& dest) {
  size_t pos = 0;
  while ((pos = src.find_first_of("0123456789", pos)) != std::string::npos) {
    dest.push_back(std::stoi(std::string(src, pos)));
    pos = src.find_first_of(" ", pos);
  }
}

static vector<shared_ptr<vector<float>>> CreatePriors(
    int image_width, int image_height,
    const google::protobuf::RepeatedPtrField<vitis::ai::proto::PriorBoxParam>&
        boxes) {
  vector<vitis::ai::multitaskv3::PriorBoxes> prior_boxes;
  for (const auto& box : boxes) {
    prior_boxes.emplace_back(vitis::ai::multitaskv3::PriorBoxes{
        image_width, image_height, box.layer_width(),
        box.layer_height(),  //
        vector<float>(box.variances().begin(), box.variances().end()),
        vector<float>(box.min_sizes().begin(), box.min_sizes().end()),
        vector<float>(box.max_sizes().begin(), box.max_sizes().end()),
        vector<float>(box.aspect_ratios().begin(), box.aspect_ratios().end()),
        box.offset(), box.step_width(), box.step_height(), box.flip(),
        box.clip()});
  }
  int num_priors = 0;
  for (auto& p : prior_boxes) {
    num_priors += p.priors().size();
  }

  auto priors = vector<shared_ptr<vector<float>>>{};
  priors.reserve(num_priors);
  for (auto i = 0U; i < prior_boxes.size(); ++i) {
    priors.insert(priors.end(), prior_boxes[i].priors().begin(),
                  prior_boxes[i].priors().end());
  }
  return priors;
}

MultiTaskv3PostProcessImp::MultiTaskv3PostProcessImp(
    const vitis::ai::proto::DpuModelParam& config,
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors)
    : num_detection_classes_{config.multi_task_param()
                                 .num_of_detection_classes()},
      num_segmention_classes_{
          config.multi_task_param().num_of_segmentation_classes()},
      input_tensors_(input_tensors),
      scolor1_{config.segmentation_param().color1()},
      scolor2_{config.segmentation_param().color2()},
      scolor3_{config.segmentation_param().color3()} {
  vector<string> output_names;
  vector<string> loc_names =
      vector<string>(config.multi_task_param().loc_name().begin(),
                     config.multi_task_param().loc_name().end());
  vector<string> conf_names =
      vector<string>(config.multi_task_param().conf_name().begin(),
                     config.multi_task_param().conf_name().end());
  vector<string> centerness_names =
      vector<string>(config.multi_task_param().centerness_name().begin(),
                     config.multi_task_param().centerness_name().end());
  string seg_name = string(config.multi_task_param().seg_name());
  string drivable_name = string(config.multi_task_param().drivable_name());
  string lane_name = string(config.multi_task_param().lane_name());
  string depth_name = string(config.multi_task_param().depth_name());
  CHECK(loc_names.size() != 0);
  CHECK(conf_names.size() != 0);
  CHECK(centerness_names.size() != 0);
  vector<vitis::ai::library::OutputTensor> loc_tensor;
  auto l_size = 0u;
  for (auto i = 0u; i < loc_names.size(); ++i) {
    for (auto j = 0u; j < output_tensors[0].size(); ++j) {
      if (output_tensors[0][j].name.find(loc_names[i]) != std::string::npos) {
        loc_tensor.emplace_back(output_tensors[0][j]);
        l_size += output_tensors[0][j].size / 4;
      }
    }
  }
  output_tensors_.push_back(loc_tensor);
  vector<vitis::ai::library::OutputTensor> conf_tensor;
  for (auto i = 0u; i < conf_names.size(); ++i) {
    for (auto j = 0u; j < output_tensors[0].size(); ++j) {
      if (output_tensors[0][j].name.find(conf_names[i]) != std::string::npos) {
        conf_tensor.emplace_back(output_tensors[0][j]);
      }
    }
  }
  output_tensors_.push_back(conf_tensor);

  vector<vitis::ai::library::OutputTensor> centerness_tensor;
  for (auto i = 0u; i < centerness_names.size(); ++i) {
    for (auto j = 0u; j < output_tensors[0].size(); ++j) {
      if (output_tensors[0][j].name.find(centerness_names[i]) !=
          std::string::npos) {
        centerness_tensor.emplace_back(output_tensors[0][j]);
      }
    }
  }
  output_tensors_.push_back(centerness_tensor);
  CHECK_EQ(loc_tensor.size(), conf_tensor.size());
  CHECK_EQ(centerness_tensor.size(), conf_tensor.size());

  vector<vitis::ai::library::OutputTensor> seg_tensor;
  for (auto j = 0u; j < output_tensors[0].size(); j++) {
    if (output_tensors[0][j].name.find(seg_name) != std::string::npos) {
      seg_tensor.push_back(output_tensors[0][j]);
    }
  }
  for (auto j = 0u; j < output_tensors[0].size(); j++) {
    if (output_tensors[0][j].name.find(lane_name) != std::string::npos) {
      seg_tensor.push_back(output_tensors[0][j]);
    }
  }
  for (auto j = 0u; j < output_tensors[0].size(); j++) {
    if (output_tensors[0][j].name.find(drivable_name) != std::string::npos) {
      seg_tensor.push_back(output_tensors[0][j]);
    }
  }
  for (auto j = 0u; j < output_tensors[0].size(); j++) {
    if (output_tensors[0][j].name.find(depth_name) != std::string::npos) {
      seg_tensor.push_back(output_tensors[0][j]);
    }
  }
  CHECK_EQ((int)seg_tensor.size(), 4);

  auto batch_size = input_tensors_[0][0].batch;
  output_tensors_.push_back(seg_tensor);
  all_loc_infos_.resize(batch_size);

  auto batch_idx = 0;
  for (auto& loc_infos : all_loc_infos_) {
    loc_infos.reserve(output_tensors_[0].size());
    loc_infos.assign(output_tensors_[0].size(),
                     vitis::ai::multitaskv3::SSDOutputInfo{});
    auto bbox_index = 0u;
    for (auto k = 0u; k < output_tensors_[0].size(); ++k) {
      loc_infos[k].base_ptr =
          (int8_t*)output_tensors_[0][k].get_data(batch_idx);
      loc_infos[k].ptr = loc_infos[k].base_ptr;
      loc_infos[k].index_begin = bbox_index;
      loc_infos[k].bbox_single_size = 4;
      loc_infos[k].index_size = output_tensors_[0][k].size / batch_size /
                                loc_infos[k].bbox_single_size;
      bbox_index += loc_infos[k].index_size;
      loc_infos[k].scale =
          vitis::ai::library::tensor_scale(output_tensors_[0][k]);
      loc_infos[k].size = output_tensors_[0][k].size / batch_size;
    }
    batch_idx++;
  }

  batch_idx = 0;
  all_conf_infos_.resize(batch_size);
  for (auto& conf_infos : all_conf_infos_) {
    auto score_index = 0u;
    conf_infos.reserve(output_tensors_[1].size());
    conf_infos.assign(output_tensors_[1].size(),
                      vitis::ai::multitaskv3::SSDOutputInfo{});
    for (auto k = 0u; k < output_tensors_[1].size(); ++k) {
      conf_infos[k].base_ptr =
          (int8_t*)output_tensors_[1][k].get_data(batch_idx);
      conf_infos[k].ptr = conf_infos[k].base_ptr;
      conf_infos[k].index_begin = score_index;
      conf_infos[k].index_size =
          output_tensors_[1][k].size / batch_size / num_detection_classes_;
      score_index += conf_infos[k].index_size;
      conf_infos[k].scale =
          vitis::ai::library::tensor_scale(output_tensors_[1][k]);
      conf_infos[k].size = output_tensors_[1][k].size / batch_size;
    }
    batch_idx++;
  }

  batch_idx = 0;
  all_centerness_infos_.resize(batch_size);
  for (auto& center_infos : all_centerness_infos_) {
    auto center_index = 0u;
    center_infos.reserve(output_tensors_[2].size());
    center_infos.assign(output_tensors_[2].size(),
                        vitis::ai::multitaskv3::SSDOutputInfo{});
    for (auto k = 0u; k < output_tensors_[2].size(); ++k) {
      center_infos[k].base_ptr =
          (int8_t*)output_tensors_[2][k].get_data(batch_idx);
      center_infos[k].ptr = center_infos[k].base_ptr;
      center_infos[k].index_begin = center_index;
      center_infos[k].index_size = output_tensors_[2][k].size / batch_size;
      center_index += center_infos[k].index_size;
      center_infos[k].scale =
          vitis::ai::library::tensor_scale(output_tensors_[2][k]);
      center_infos[k].size = output_tensors_[2][k].size / batch_size;
    }
    batch_idx++;
  }

  auto priors = CreatePriors((int)input_tensors[0][0].width,
                             (int)input_tensors[0][0].height,
                             config.multi_task_param().prior_box_param());
  detector_ = std::make_unique<vitis::ai::multitaskv3::SSDdetector>(
      num_detection_classes_,
      vitis::ai::multitaskv3::SSDdetector::CodeType::CENTER_SIZE,
      false,                                   //
      config.multi_task_param().keep_top_k(),  //
      std::vector<float>(config.multi_task_param().th_conf().begin(),
                         config.multi_task_param().th_conf().end()),  //
      config.multi_task_param().top_k(),                              //
      config.multi_task_param().nms_threshold(),                      //
      1.0, priors, vitis::ai::library::tensor_scale(output_tensors_[0][0]));

  conf_result.resize(priors.size() * num_detection_classes_);
  centerness_result.resize(priors.size());
  convert_color(scolor1_, color_c1);
  convert_color(scolor2_, color_c2);
  convert_color(scolor3_, color_c3);

  auto s = config.segmentation_param().color1();
  for (int i = 0; i < num_segmention_classes_; i++) {
    color_map_.push_back(color_c1[i]);
    color_map_.push_back(color_c2[i]);
    color_map_.push_back(color_c3[i]);
  }
  // detector_ = vitis::ai::multitaskv3::CreateSSDUniform(priors, config);
}

MultiTaskv3PostProcessImp::~MultiTaskv3PostProcessImp() {}

static void seg_color_c16(const uint8_t* color_map, int8_t* d, int g,
                          uint8_t* dst) {
#ifdef HAVE_NEON
  uint8x8_t l0 = vld1_u8(color_map);
  uint8x8_t l1 = vld1_u8(color_map + 8);
  uint8x8_t l2 = vld1_u8(color_map + 16);
  uint8x8_t l3 = vld1_u8(color_map + 24);
  uint8x8_t l4 = vld1_u8(color_map + 32);
  uint8x8_t l5 = vld1_u8(color_map + 40);

  uint8x8x3_t lut_low_color{l0, l1, l2};
  uint8x8x3_t lut_high_color{l3, l4, l5};

  // Internal result
  uint8x8_t result;  // register d0
  // Initilize index registers
  uint8x8_t task_idx = vcreate_u8(0x0706050403020100);
  uint8x8_t all_1 = vdup_n_u8(1);
  uint8x8_t all_2 = vdup_n_u8(2);  // register d2
  uint8x8_t all_3 = vdup_n_u8(3);  // register d3
  uint8x8_t all_24 = vdup_n_u8(24);
  uint8x8_t d13 = vcreate_u8(0x0000000006040200);
  uint8x8_t d14 = vcreate_u8(0x0400040004000400);
  uint8x8_t idx_0 = vcreate_u8(0x0a02110901100800);
  uint8x8_t idx_1 = vcreate_u8(0x05140c04130b0312);
  uint8x8_t idx_2 = vcreate_u8(0x170f07160e06150d);

  uint64_t temp = 0x0;
  int j = 0;

  for (int i = 0; i < g / 2; ++i) {
    // Use register d4 to d7 to store data
    int8x8x4_t interleave_data = vld4_s8(d);

    // Compare first two columns (column 0 and 1)
    result = vclt_s8(interleave_data.val[0], interleave_data.val[1]);
    result = vshr_n_u8(result, 7);

    uint8x8_t large_idx = vshl_n_u8(result, 3);  // register d8
    large_idx = vadd_u8(task_idx, large_idx);

    // d9 to store max value of column 0 and 1
    int8x8x2_t tab_d{interleave_data.val[0], interleave_data.val[1]};
    int8x8_t d9 = vtbl2_s8(tab_d, vreinterpret_s8_u8(large_idx));

    // Compare column 2
    large_idx = vclt_s8(d9, interleave_data.val[2]);
    large_idx = vshr_n_u8(large_idx, 7);
    large_idx = vshl_n_u8(large_idx, 3);
    large_idx = vadd_u8(task_idx, large_idx);
    uint8x8x2_t tab_i{result, all_2};
    result = vtbl2_u8(tab_i, large_idx);
    tab_d.val[0] = d9;
    tab_d.val[1] = interleave_data.val[2];
    d9 = vtbl2_s8(tab_d, vreinterpret_s8_u8(large_idx));

    // Compare column 3
    large_idx = vclt_s8(d9, interleave_data.val[3]);
    large_idx = vshr_n_u8(large_idx, 7);
    large_idx = vshl_n_u8(large_idx, 3);
    large_idx = vadd_u8(task_idx, large_idx);
    tab_i.val[0] = result;
    tab_i.val[1] = all_3;
    result = vtbl2_u8(tab_i, large_idx);
    tab_d.val[0] = d9;
    tab_d.val[1] = interleave_data.val[3];
    d9 = vtbl2_s8(tab_d, vreinterpret_s8_u8(large_idx));

    // Final compare, only lower 4 bytes are valid
    int8x8_t d10 = vrev16_s8(d9);
    uint8x8_t d11 = vclt_s8(d9, d10);
    d11 = vshr_n_u8(d11, 7);
    uint8x8_t d12 = vadd_u8(vtbl1_u8(d11, d13), d13);

    result = vadd_u8(result, d14);
    result = vtbl1_u8(result, d12);
    // print_u8x8(d0);

    d9 = vtbl1_s8(d9, vreinterpret_s8_u8(d12));
    // print_s8x8(d9);

    int max_g0 = ((vget_lane_s8(d9, 0) < vget_lane_s8(d9, 1))
                      ? (vget_lane_u8(result, 1) + 8)
                      : vget_lane_u8(result, 0));
    int max_g1 = ((vget_lane_s8(d9, 2) < vget_lane_s8(d9, 3))
                      ? (vget_lane_u8(result, 3) + 8)
                      : vget_lane_u8(result, 2));

    // cout << j << endl;
    // cout << max_g0 << ", " << max_g1 << endl;

    temp ^= (uint64_t)((max_g1 << 8) ^ (max_g0)) << (16 * (j++));

    // printf("%d, %0lx\n", j, temp);

    if (j % 4 == 0) {
      uint8x8_t creg = vcreate_u8(temp);
      // print_u8x8(creg);

      uint8x8_t creg_0 = vmul_u8(creg, all_3);
      uint8x8_t creg_1 = vadd_u8(creg_0, all_1);
      uint8x8_t creg_2 = vadd_u8(creg_1, all_1);
      uint8x8x3_t creg_t{creg_0, creg_1, creg_2};

      // print_u8x8(creg_0);
      // print_u8x8(creg_1);
      // print_u8x8(creg_2);

      uint8x8_t idx = vtbl3_u8(creg_t, idx_0);
      // print_u8x8(idx);
      uint8x8_t d0 = vtbl3_u8(lut_low_color, idx);
      // print_u8x8(d0);
      idx = vsub_u8(idx, all_24);
      // print_u8x8(idx);
      uint8x8_t d1 = vtbl3_u8(lut_high_color, idx);
      // print_u8x8(d1);
      d0 = vadd_u8(d0, d1);
      // print_u8x8(d0);
      // cout << endl;
      vst1_u8(dst, d0);
      dst += 8;

      idx = vtbl3_u8(creg_t, idx_1);
      // print_u8x8(idx);
      d0 = vtbl3_u8(lut_low_color, idx);
      idx = vsub_u8(idx, all_24);
      d1 = vtbl3_u8(lut_high_color, idx);
      d0 = vadd_u8(d0, d1);
      // print_u8x8(d0);
      vst1_u8(dst, d0);
      dst += 8;

      idx = vtbl3_u8(creg_t, idx_2);
      // print_u8x8(idx);
      d0 = vtbl3_u8(lut_low_color, idx);
      idx = vsub_u8(idx, all_24);
      d1 = vtbl3_u8(lut_high_color, idx);
      d0 = vadd_u8(d0, d1);
      // print_u8x8(d0);
      vst1_u8(dst, d0);
      dst += 8;

      j = 0;
      temp = 0;
    }

    // cout << max_g0 << ", " << max_g1 << endl;

    // Next 2 groups
    d += 32;
  }
#else
  //  assert(false && "not supported");
  //  abort();
  for (int i = 0; i < g; ++i) {
    int8_t max_val = 0;
    int offset = 0;
    // use magic num 16 is not a good job, we should fix it later.
    for (int j = 0; j < 16; ++j) {
      if (d[i * 16 + j] > max_val) {
        max_val = d[i * 16 + j];
        offset = j;
      }
    }
    dst[i * 3] = color_map[offset * 3];
    dst[i * 3 + 1] = color_map[offset * 3 + 1];
    dst[i * 3 + 2] = color_map[offset * 3 + 2];
  }
#endif
}

std::vector<Vehiclev3Result> MultiTaskv3PostProcessImp::process_det(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t batch_idx) {
  __TIC__(MULTITASK_DET)
  std::vector<float> cs(conf_result.size());
  CHECK_EQ(all_loc_infos_[batch_idx].size(), all_conf_infos_[batch_idx].size());
  __TIC__(ALL)
  for (auto k = 0u; k < all_conf_infos_[batch_idx].size(); k++) {
    auto offset =
        all_conf_infos_[batch_idx][k].index_begin * num_detection_classes_;
    __TIC__(SIG_1)
    sigmoid_n((int8_t*)all_conf_infos_[batch_idx][k].ptr,
              conf_result.data() + offset,
              all_conf_infos_[batch_idx][k].index_size * 3,
              all_conf_infos_[batch_idx][k].scale);
    __TOC__(SIG_1)
    __TIC__(SIG_2)
    sigmoid_n((int8_t*)all_centerness_infos_[batch_idx][k].ptr,
              centerness_result.data() +
                  all_centerness_infos_[batch_idx][k].index_begin,
              all_centerness_infos_[batch_idx][k].index_size,
              all_centerness_infos_[batch_idx][k].scale);
    __TOC__(SIG_2)
    __TIC__(MUL)
    for (auto i = 0u; i < centerness_result.size(); i++) {
      cs[i * 3 + 0] = conf_result[i * 3 + 0] * centerness_result[i];
      cs[i * 3 + 1] = conf_result[i * 3 + 1] * centerness_result[i];
      cs[i * 3 + 2] = conf_result[i * 3 + 2] * centerness_result[i];
    }
    __TOC__(MUL)
  }
  __TOC__(ALL)

  vector<Vehiclev3Result> v_result;
  std::map<uint32_t, vitis::ai::multitaskv3::SSDOutputInfo> bbox_layer_infos;
  // for (auto i : bbox_layer_indexes_) {
  for (auto i = 0u; i < all_loc_infos_[batch_idx].size(); ++i) {
    bbox_layer_infos.emplace(std::make_pair(i, all_loc_infos_[batch_idx][i]));
  }
  __TIC__(DETCT)
  // for(auto i = 0u; i < conf_result.size(); i++)
  detector_->Detect(bbox_layer_infos, cs.data(), v_result);
  __TOC__(DETCT)
  __TOC__(MULTITASK_DET)
  return v_result;
}

cv::Mat MultiTaskv3PostProcessImp::process_depth(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t tensor_ind, size_t batch_idx) {
  __TIC__(MULTITASK_DEPTH)
  // vector<float> rs(output_tensors[3][tensor_ind].size);
  vector<float> rs(  output_tensors[3][tensor_ind].channel
                   * output_tensors[3][tensor_ind].height 
                   * output_tensors[3][tensor_ind].width);
  auto out_scale =
      vitis::ai::library::tensor_scale(output_tensors[3][tensor_ind]);
  get_depth((int8_t*)output_tensors[3][tensor_ind].get_data(batch_idx),
            rs.data(), 
            // output_tensors[3][tensor_ind].size, 
            rs.size(), 
            out_scale);
  cv::Mat depth_results(output_tensors[3][tensor_ind].height,
                        output_tensors[3][tensor_ind].width, CV_8UC1);
  for (auto i = 0u; i < output_tensors[3][tensor_ind].width *
                            output_tensors[3][tensor_ind].height;
       i++) {
    depth_results.data[i] = (uint8_t)rs[i];
  }
  __TOC__(MULTITASK_DEPTH)
  return depth_results;
}

cv::Mat MultiTaskv3PostProcessImp::process_depth_ori(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t tensor_ind, size_t batch_idx) {
  __TIC__(MULTITASK_DEPTH)
  // vector<float> rs(output_tensors[3][tensor_ind].size);
  vector<float> rs(  output_tensors[3][tensor_ind].channel
                   * output_tensors[3][tensor_ind].height 
                   * output_tensors[3][tensor_ind].width);
  auto out_scale =
      vitis::ai::library::tensor_scale(output_tensors[3][tensor_ind]);
  get_depth_ori((int8_t*)output_tensors[3][tensor_ind].get_data(batch_idx),
                rs.data(), 
                // output_tensors[3][tensor_ind].size, 
                rs.size(), 
                out_scale);
  cv::Mat depth_results(output_tensors[3][tensor_ind].height,
                        output_tensors[3][tensor_ind].width, CV_8UC1);
  for (auto i = 0u; i < output_tensors[3][tensor_ind].width *
                            output_tensors[3][tensor_ind].height;
       i++) {
    depth_results.data[i] = (uint8_t)rs[i];
  }
  __TOC__(MULTITASK_DEPTH)
  return depth_results;
}

cv::Mat MultiTaskv3PostProcessImp::process_seg(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t tensor_ind, size_t batch_idx) {
  __TIC__(MULTITASK_SEG)
  /*
  size_t tensor_ind = 0;
  for (auto j = 0u; j < output_tensors[3].size(); ++j) {
    if (output_tensors[3][j].name.find(tensor_name) != std::string::npos) {
      tensor_ind = j;
    }
  }
  */
  cv::Mat seg_results(output_tensors[3][tensor_ind].height,
                      output_tensors[3][tensor_ind].width, CV_8UC1);
  // vector<uint8_t> seg_results(task->getOutputTensor()[0][2].width *
  //                            task->getOutputTensor()[0][2].height);
  vitis::ai::max_index_void(
      (int8_t*)output_tensors[3][tensor_ind].get_data(batch_idx),
      output_tensors[3][tensor_ind].width, output_tensors[3][tensor_ind].height,
      output_tensors[3][tensor_ind].channel, seg_results.data);
  __TOC__(MULTITASK_SEG)
  return seg_results;
}

cv::Mat MultiTaskv3PostProcessImp::process_seg_visualization(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t tensor_ind, size_t batch_idx) {
  if (num_segmention_classes_ != 16) {
    LOG(FATAL) << "only support channel = 16";
  }
  // size_t tensor_ind = 0;
  // for (auto j = 0u; j < output_tensors[3].size(); ++j) {
  //   if (output_tensors[3][j].name.find(tensor_name) != std::string::npos) {
  //     tensor_ind = j;
  //   }
  //}
  __TIC__(MULTITASK_SEG_VISUALIZATION)
  cv::Mat segmat(output_tensors[3][tensor_ind].height,
                 output_tensors[3][tensor_ind].width, CV_8UC3);
  seg_color_c16(color_map_.data(),
                (int8_t*)output_tensors[3][tensor_ind].get_data(batch_idx),
                output_tensors[3][tensor_ind].width *
                    output_tensors[3][tensor_ind].height,
                segmat.data);
  __TOC__(MULTITASK_SEG_VISUALIZATION)
  return segmat;
}

cv::Mat MultiTaskv3PostProcessImp::process_seg_visualization_c(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t tensor_ind, size_t batch_idx) {
  __TIC__(MULTITASK_SEG_VISUALIZATION)
  cv::Mat segmat(output_tensors[3][tensor_ind].height,
                 output_tensors[3][tensor_ind].width, CV_8UC3);
  size_t i = 0;
  for (unsigned int row_ind = 0; row_ind <
  output_tensors[3][tensor_ind].height; row_ind++)
    for (unsigned int col_ind = 0; col_ind <
    output_tensors[3][tensor_ind].width; col_ind++) {
      auto max_ind = std::max_element(
          ((int8_t*)output_tensors[3][tensor_ind].get_data(batch_idx)) + i,
          ((int8_t*)output_tensors[3][tensor_ind].get_data(batch_idx)) + i +
              output_tensors[3][tensor_ind].channel);
      uint8_t posit = std::distance(
          ((int8_t*)output_tensors[3][tensor_ind].get_data(batch_idx)) + i,
          max_ind);
      segmat.at<cv::Vec3b>(row_ind, col_ind) =
          cv::Vec3b((uint8_t)color_c1[posit], (uint8_t)color_c2[posit],
                    (uint8_t)color_c3[posit]);
      i+=output_tensors[3][tensor_ind].channel;
    }

  __TOC__(MULTITASK_SEG_VISUALIZATION)
  return segmat;
}

std::vector<MultiTaskv3Result> MultiTaskv3PostProcessImp::post_process(
    size_t batch_size) {
  __TIC__(POST_PROCESS)
  auto ret = std::vector<MultiTaskv3Result>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.push_back(MultiTaskv3Result{
        (int)input_tensors_[0][0].width,                             //
        (int)input_tensors_[0][0].height,                            //
        MultiTaskv3PostProcessImp::process_det(output_tensors_, i),  //
        // vr,  //
        MultiTaskv3PostProcessImp::process_seg(output_tensors_, 0, i),
        MultiTaskv3PostProcessImp::process_seg(output_tensors_, 1, i),
        MultiTaskv3PostProcessImp::process_seg(output_tensors_, 2, i),
        MultiTaskv3PostProcessImp::process_depth_ori(output_tensors_, 3, i)

    });
  }
  __TOC__(POST_PROCESS)
  return ret;
}

std::vector<MultiTaskv3Result>
MultiTaskv3PostProcessImp::post_process_visualization(size_t batch_size) {
  __TIC__(POST_VIS)
  auto ret = std::vector<MultiTaskv3Result>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.push_back(MultiTaskv3Result{
        (int)input_tensors_[0][0].width,                             //
        (int)input_tensors_[0][0].height,                            //
        MultiTaskv3PostProcessImp::process_det(output_tensors_, i),  //
        MultiTaskv3PostProcessImp::process_seg_visualization(output_tensors_, 0,
                                                             i),
        MultiTaskv3PostProcessImp::process_seg_visualization_c(output_tensors_,
                                                               1, i),
        MultiTaskv3PostProcessImp::process_seg_visualization_c(output_tensors_,
                                                               2, i),
        MultiTaskv3PostProcessImp::process_depth(output_tensors_, 3, i)

    });
  }
  __TOC__(POST_VIS)
  return ret;
}

}  // namespace ai
}  // namespace vitis

