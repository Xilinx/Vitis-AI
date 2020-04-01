/*
 * Copyright 2019 Xilinx Inc.
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
#ifdef HAVE_NEON
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
  output_names.push_back(string(config.multi_task_param().loc_name()));
  output_names.push_back(string(config.multi_task_param().conf_name()));
  output_names.push_back(string(config.multi_task_param().seg_name()));
  vector<vitis::ai::library::OutputTensor> temp_tensor;
  for (auto i = 0u; i < output_names.size(); i++){
    for (auto j = 0u; j < output_tensors[0].size(); j++){
      if (output_tensors[0][j].name.find(output_names[i]) != std::string::npos){
        temp_tensor.push_back(output_tensors[0][j]);
        break;
      }
    }
  }
  output_tensors_.push_back(temp_tensor);
  detector_ = std::make_unique<vitis::ai::multitask::SSDdetector>(
      num_detection_classes_,
      vitis::ai::multitask::SSDdetector::CodeType::CENTER_SIZE,
      false,                                   //
      config.multi_task_param().keep_top_k(),  //
      std::vector<float>(config.multi_task_param().th_conf().begin(),
                         config.multi_task_param().th_conf().end()),  //
      config.multi_task_param().top_k(),                              //
      config.multi_task_param().nms_threshold(),                      //
      1.0,
      CreatePriors((int)input_tensors[0][0].width,
                   (int)input_tensors[0][0].height,
                   config.multi_task_param().prior_box_param()),
      vitis::ai::library::tensor_scale(output_tensors_[0][0]));

  softmax_result.resize(output_tensors_[0][1].size);
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
      if (d[i*16+j] > max_val) {
        max_val = d[i*16+j];
        offset = j;
      }
    }
    dst[i*3] = color_map[offset*3];
    dst[i*3+1] = color_map[offset*3+1];
    dst[i*3+2] = color_map[offset*3+2];
  }
#endif
}

std::vector<VehicleResult> MultiTaskPostProcessImp::process_det(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t batch_idx) {
  __TIC__(MULTITASK_DET)
  vitis::ai::softmax(((int8_t*)output_tensors[0][1].get_data(batch_idx)),
                      vitis::ai::library::tensor_scale(output_tensors[0][1]),
                      num_detection_classes_,
                      output_tensors[0][1].size / num_detection_classes_,
                      softmax_result.data());
  vitis::ai::multitask::MultiDetObjects ssd_results;
  detector_->Detect((int8_t*)(output_tensors[0][0].get_data(batch_idx)), softmax_result.data(),
                    &ssd_results);
  vector<VehicleResult> v_result;
  // std::cout << ssd_results.size()  << std::endl;
  for (size_t i = 0; i < ssd_results.size(); ++i) {
    int label = get<0>(ssd_results[i]);
    float confidence = get<1>(ssd_results[i]);
    float xmin = get<2>(ssd_results[i]).x;
    float ymin = get<2>(ssd_results[i]).y;
    float width = get<2>(ssd_results[i]).width;
    float height = get<2>(ssd_results[i]).height;
    xmin = std::min(std::max(xmin, 0.f), 1.f);
    ymin = std::min(std::max(ymin, 0.f), 1.f);
    width = std::max(width, 0.f);
    width = ((width + xmin) < 1) ? width : (1 - xmin);
    height = std::max(height, 0.f);
    height = ((height + ymin) < 1) ? height : (1 - ymin);
    float angle = atan2(get<4>(ssd_results[i]), get<3>(ssd_results[i])) *
                  180.0 / 3.1415926;
    v_result.push_back({label, confidence, xmin, ymin, width, height, angle});
  }
  __TOC__(MULTITASK_DET)
  return v_result;
}

cv::Mat MultiTaskPostProcessImp::process_seg(
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t batch_idx) {
  __TIC__(MULTITASK_SEG)
  cv::Mat seg_results(output_tensors[0][2].height, output_tensors[0][2].width,
                      CV_8UC1);
  // vector<uint8_t> seg_results(task->getOutputTensor()[0][2].width *
  //                            task->getOutputTensor()[0][2].height);
  vitis::ai::max_index_void(
      (int8_t*)output_tensors[0][2].get_data(batch_idx), output_tensors[0][2].width,
      output_tensors[0][2].height, num_segmention_classes_, seg_results.data);
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
  cv::Mat segmat(output_tensors[0][2].height, output_tensors[0][2].width,
                 CV_8UC3);
  seg_color_c16(color_map_.data(), (int8_t*)output_tensors[0][2].get_data(batch_idx),
                output_tensors[0][2].width * output_tensors[0][2].height,
                segmat.data);
  __TOC__(MULTITASK_SEG_VISUALIZATION)
  return segmat;
}

std::vector<MultiTaskResult> MultiTaskPostProcessImp::post_process_seg() {
  auto batch_size = input_tensors_[0][0].batch;
  auto ret = std::vector<MultiTaskResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.push_back(MultiTaskResult{
      (int)input_tensors_[0][0].width,                        //
      (int)input_tensors_[0][0].height,                       //
      MultiTaskPostProcessImp::process_det(output_tensors_, i),  //
      MultiTaskPostProcessImp::process_seg(output_tensors_, i)});
  }
  return ret;
}

std::vector<MultiTaskResult> MultiTaskPostProcessImp::post_process_seg_visualization() {
  auto batch_size = input_tensors_[0][0].batch;
  auto ret = std::vector<MultiTaskResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.push_back(MultiTaskResult{
      (int)input_tensors_[0][0].width,                        //
      (int)input_tensors_[0][0].height,                       //
      MultiTaskPostProcessImp::process_det(output_tensors_, i),  //
      MultiTaskPostProcessImp::process_seg_visualization(output_tensors_, i)});
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
