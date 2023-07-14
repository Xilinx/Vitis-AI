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

#include "./multitask_imp.hpp"

#include "./prior_boxes.hpp"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <queue>
#include <vector>
#include <vitis/ai/math.hpp>
#include <vitis/ai/max_index.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;

namespace vitis {
namespace ai {

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
  vector<vitis::ai::multitask::PriorBoxes> prior_boxes;
  for (const auto& box : boxes) {
    prior_boxes.emplace_back(vitis::ai::multitask::PriorBoxes{
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

MultiTaskPostProcessImp::MultiTaskPostProcessImp(
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
  string seg_name = string(config.multi_task_param().seg_name());
  CHECK(loc_names.size() != 0);
  CHECK(conf_names.size() != 0);
  vector<vitis::ai::library::OutputTensor> loc_tensor;
  for (auto i = 0u; i < loc_names.size(); ++i) {
    for (auto j = 0u; j < output_tensors[0].size(); ++j) {
      if (output_tensors[0][j].name.find(loc_names[i]) != std::string::npos) {
        loc_tensor.emplace_back(output_tensors[0][j]);
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

  vector<vitis::ai::library::OutputTensor> seg_tensor;
  for (auto j = 0u; j < output_tensors[0].size(); j++) {
    if (output_tensors[0][j].name.find(seg_name) != std::string::npos) {
      seg_tensor.push_back(output_tensors[0][j]);
      break;
    }
  }
  auto batch_size = input_tensors_[0][0].batch;
  output_tensors_.push_back(seg_tensor);
  all_loc_infos_.resize(batch_size);

  auto batch_idx = 0;
  for (auto& loc_infos : all_loc_infos_) {
    loc_infos.reserve(output_tensors_[0].size());
    loc_infos.assign(output_tensors_[0].size(),
                     vitis::ai::multitask::SSDOutputInfo{});
    auto bbox_index = 0u;
    for (auto k = 0u; k < output_tensors_[0].size(); ++k) {
      loc_infos[k].base_ptr =
          (int8_t*)output_tensors_[0][k].get_data(batch_idx);
      loc_infos[k].ptr = loc_infos[k].base_ptr;
      loc_infos[k].index_begin = bbox_index;
      loc_infos[k].bbox_single_size = 6;
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
                      vitis::ai::multitask::SSDOutputInfo{});
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
  auto priors = CreatePriors((int)input_tensors[0][0].width,
                             (int)input_tensors[0][0].height,
                             config.multi_task_param().prior_box_param());
  detector_ = std::make_unique<vitis::ai::multitask::SSDdetector>(
      num_detection_classes_,
      vitis::ai::multitask::SSDdetector::CodeType::CENTER_SIZE,
      false,                                   //
      config.multi_task_param().keep_top_k(),  //
      std::vector<float>(config.multi_task_param().th_conf().begin(),
                         config.multi_task_param().th_conf().end()),  //
      config.multi_task_param().top_k(),                              //
      config.multi_task_param().nms_threshold(),                      //
      1.0, priors, vitis::ai::library::tensor_scale(output_tensors_[0][0]));

  softmax_result.resize(priors.size() * num_detection_classes_);
  vector<uint8_t> color_c1;
  vector<uint8_t> color_c2;
  vector<uint8_t> color_c3;
  convert_color(scolor1_, color_c1);
  convert_color(scolor2_, color_c2);
  convert_color(scolor3_, color_c3);

  auto s = config.segmentation_param().color1();
  for (int i = 0; i < num_segmention_classes_; i++) {
    color_map_.push_back(color_c1[i]);
    color_map_.push_back(color_c2[i]);
    color_map_.push_back(color_c3[i]);
  }
}

MultiTaskPostProcessImp::~MultiTaskPostProcessImp() {}

static void set_color(uint8_t* input, uint8_t* color, int g, uint8_t* output) {
#ifdef ENABLE_NEON
  uint8x8x3_t low_colors = vld3_u8(color);
  uint8x8x3_t high_colors = vld3_u8(color+24);
  uint8x8_t all_8 = vdup_n_u8(8);
  for (int i = 0; i < g/8; ++i) {
    uint8x8_t q = vld1_u8(input);
    uint8x8_t c1 = vtbl1_u8(low_colors.val[0], q);
    uint8x8_t c2 = vtbl1_u8(low_colors.val[1], q);
    uint8x8_t c3 = vtbl1_u8(low_colors.val[2], q);

    q = vsub_u8(q, all_8);
    uint8x8_t c4 = vtbl1_u8(high_colors.val[0], q);
    uint8x8_t c5 = vtbl1_u8(high_colors.val[1], q);
    uint8x8_t c6 = vtbl1_u8(high_colors.val[2], q);

    c1 = vadd_u8(c1, c4);
    c2 = vadd_u8(c2, c5);
    c3 = vadd_u8(c3, c6);
    vst3_u8(output, {c1, c2, c3});
    input += 8;
    output += 24;
  }
#else
  for (int i = 0; i < g; ++i) {
    memcpy(output+i*3, color+input[i]*3, 3);
  }
#endif

}

std::vector<VehicleResult> MultiTaskPostProcessImp::process_det(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t batch_idx) {
  __TIC__(MULTITASK_DET)
  CHECK_EQ(all_loc_infos_[batch_idx].size(), all_conf_infos_[batch_idx].size());
  for (auto k = 0u; k < all_conf_infos_[batch_idx].size(); k++) {
    auto offset =
        all_conf_infos_[batch_idx][k].index_begin * num_detection_classes_;
    vitis::ai::softmax((int8_t*)all_conf_infos_[batch_idx][k].ptr,
                       all_conf_infos_[batch_idx][k].scale,
                       num_detection_classes_,
                       all_conf_infos_[batch_idx][k].index_size,
                       softmax_result.data() + offset);
  }

  vector<VehicleResult> v_result;

  std::map<uint32_t, vitis::ai::multitask::SSDOutputInfo> bbox_layer_infos;
  // for (auto i : bbox_layer_indexes_) {
  for (auto i = 0u; i < all_loc_infos_[batch_idx].size(); ++i) {
    bbox_layer_infos.emplace(std::make_pair(i, all_loc_infos_[batch_idx][i]));
  }
  detector_->Detect(bbox_layer_infos, softmax_result.data(), v_result);
  __TOC__(MULTITASK_DET)
  return v_result;
}

cv::Mat MultiTaskPostProcessImp::process_seg(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t batch_idx) {
  __TIC__(MULTITASK_SEG)
  cv::Mat seg_results(output_tensors[2][0].height, output_tensors[2][0].width,
                      CV_8UC1);
  vitis::ai::max_index_void((int8_t*)output_tensors[2][0].get_data(batch_idx),
                            output_tensors[2][0].width,
                            output_tensors[2][0].height,
                            num_segmention_classes_, seg_results.data);
  __TOC__(MULTITASK_SEG)
  return seg_results;
}

cv::Mat MultiTaskPostProcessImp::process_seg_visualization(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t batch_idx) {
  if (num_segmention_classes_ != 16) {
    LOG(FATAL) << "only support channel = 16";
  }
  __TIC__(MULTITASK_SEG_VISUALIZATION)

  __TIC__(SEG_MAX_VALUE)
  std::vector<uint8_t> output =
           vitis::ai::max_index((int8_t*)output_tensors[2][0].get_data(batch_idx),
                                output_tensors[2][0].width, 
                                output_tensors[2][0].height,
                                output_tensors[2][0].channel);
  __TOC__(SEG_MAX_VALUE)
  
  __TIC__(SEG_CREATE_COLOR_IMG)
  cv::Mat segmat(output_tensors[2][0].height, output_tensors[2][0].width,
                 CV_8UC3);
  set_color(output.data(), color_map_.data(),
            output_tensors[2][0].width * output_tensors[2][0].height, segmat.data);
  __TOC__(SEG_CREATE_COLOR_IMG)

  __TOC__(MULTITASK_SEG_VISUALIZATION)
  return segmat;
}

std::vector<MultiTaskResult> MultiTaskPostProcessImp::post_process_seg(
    size_t batch_size) {
  auto ret = std::vector<MultiTaskResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.push_back(MultiTaskResult{
        (int)input_tensors_[0][0].width,                           //
        (int)input_tensors_[0][0].height,                          //
        MultiTaskPostProcessImp::process_det(output_tensors_, i),  //
        MultiTaskPostProcessImp::process_seg(output_tensors_, i)});
  }
  return ret;
}

std::vector<MultiTaskResult>
MultiTaskPostProcessImp::post_process_seg_visualization(size_t batch_size) {
  auto ret = std::vector<MultiTaskResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.push_back(MultiTaskResult{
        (int)input_tensors_[0][0].width,                           //
        (int)input_tensors_[0][0].height,                          //
        MultiTaskPostProcessImp::process_det(output_tensors_, i),  //
        MultiTaskPostProcessImp::process_seg_visualization(output_tensors_,
                                                           i)});
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
