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
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

#include "prior_boxes.hpp"
#include "refine_det_post.hpp"
#include "ssd_detector.hpp"
using namespace std;
DEF_ENV_PARAM(ENABLE_REFINE_DET_DEBUG, "0");
namespace vitis {
namespace ai {

static vector<shared_ptr<vector<float>>> CreatePriors(
    int image_width, int image_height,
    const vitis::ai::proto::DpuModelParam& config) {
  vector<vitis::nnpp::refinedet::PriorBoxes> prior_boxes;
  for (const auto& box : config.refine_det_param().prior_box_param()) {
    prior_boxes.emplace_back(vitis::nnpp::refinedet::PriorBoxes{
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

static std::unique_ptr<vitis::nnpp::refinedet::SSDdetector> createSSDDetector(
    const vector<shared_ptr<vector<float>>>& priors, float arm_scale,
    float odm_scale, const vitis::ai::proto::DpuModelParam& config) {
  const float NMS_THRESHOLD = config.refine_det_param().nms_threshold();
  const int num_classes = config.refine_det_param().num_classes();
  CHECK_EQ(num_classes, config.refine_det_param().conf_threshold().size())
      << "num_classes must = conf_threshold size";
  vector<float> th_conf(config.refine_det_param().conf_threshold().begin(),
                        config.refine_det_param().conf_threshold().end());

  const int KEEP_TOP_K = config.refine_det_param().keep_top_k();
  const int TOP_K = config.refine_det_param().top_k();
  // vector<float> th_conf(num_classes, CONF_THRESHOLD);
  if (ENV_PARAM(ENABLE_REFINE_DET_DEBUG) == 1)
    LOG(INFO) << " arm_scale " << arm_scale                       //
              << " odm_scale " << odm_scale                       //
              << " num_classes " << num_classes                   //
              << " KEEP_TOP_K " << KEEP_TOP_K                     //
              << " th_conf " << th_conf[0] << ", " << th_conf[1]  //
              << " TOP_K " << TOP_K                               //
              << " NMS_THRESHOLD " << NMS_THRESHOLD               //
              << " priors.size() " << priors.size()               //
        ;

  return std::unique_ptr<vitis::nnpp::refinedet::SSDdetector>(
      new vitis::nnpp::refinedet::SSDdetector(
          num_classes,
          vitis::nnpp::refinedet::SSDdetector::CodeType::CENTER_SIZE, false,
          KEEP_TOP_K, th_conf, TOP_K, NMS_THRESHOLD, 1.0, priors, arm_scale,
          odm_scale));
}

RefineDetPost::~RefineDetPost(){};

RefineDetPost::RefineDetPost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : num_classes_{config.refine_det_param().num_classes()},
      priors_{vitis::ai::CreatePriors((int)input_tensors[0].width,
                                      (int)input_tensors[0].height, config)},
      input_tensors_(input_tensors) {
  auto layername =
      std::vector<std::string>(config.refine_det_param().layer_name().begin(),
                               config.refine_det_param().layer_name().end());
  for (auto i = 0u; i < layername.size(); i++) {
    for (auto j = 0u; j < output_tensors.size(); j++) {
      if (output_tensors[j].name.find(layername[i]) != std::string::npos) {
        output_tensors_.emplace_back(output_tensors[j]);
        break;
      }
    }
  }
  detector_ = createSSDDetector(
      priors_, vitis::ai::library::tensor_scale(output_tensors_[0]),
      vitis::ai::library::tensor_scale(output_tensors_[2]), config);
}

RefineDetResult RefineDetPost::refine_det_post_process_internal(
    unsigned int idx) {
  int sWidth = input_tensors_[0].width;
  int sHeight = input_tensors_[0].height;
  size_t batch = input_tensors_[0].batch;
  constexpr int layer_index_arm_loc = 0;
  constexpr int layer_index_arm_conf = 1;
  const auto arm_loc_width = output_tensors_[layer_index_arm_loc].width;    // 1
  const auto arm_loc_height = output_tensors_[layer_index_arm_loc].height;  // 1
  const auto arm_loc_size =
      output_tensors_[layer_index_arm_loc].size / batch;  // 24480
  const auto arm_loc_channel = output_tensors_[layer_index_arm_loc].channel;
  const auto arm_loc_scale =
      vitis::ai::library::tensor_scale(output_tensors_[layer_index_arm_loc]);
  const auto arm_loc_addr =
      (int8_t*)output_tensors_[layer_index_arm_loc].get_data(idx);

  const auto arm_conf_width = output_tensors_[layer_index_arm_conf].width;  // 1
  const auto arm_conf_height =
      output_tensors_[layer_index_arm_conf].height;  // 1
  const auto arm_conf_size = output_tensors_[layer_index_arm_conf].size /
                             batch;  // 278080 = 25280 * 11
  const auto arm_conf_channel = output_tensors_[layer_index_arm_conf].channel;
  const auto arm_conf_scale =
      vitis::ai::library::tensor_scale(output_tensors_[layer_index_arm_conf]);
  const auto arm_conf_addr =
      (int8_t*)output_tensors_[layer_index_arm_conf].get_data(idx);

  constexpr int layer_index_odm_loc = 2;
  constexpr int layer_index_odm_conf = 3;
  const auto odm_loc_width = output_tensors_[layer_index_odm_loc].width;    // 1
  const auto odm_loc_height = output_tensors_[layer_index_odm_loc].height;  // 1
  const auto odm_loc_size =
      output_tensors_[layer_index_odm_loc].size / batch;  // 24480
  const auto odm_loc_channel = output_tensors_[layer_index_odm_loc].channel;
  const auto odm_loc_addr =
      (int8_t*)output_tensors_[layer_index_odm_loc].get_data(idx);
  const auto odm_loc_scale =
      vitis::ai::library::tensor_scale(output_tensors_[layer_index_odm_loc]);

  const auto odm_conf_width = output_tensors_[layer_index_odm_conf].width;  // 1
  const auto odm_conf_height =
      output_tensors_[layer_index_odm_conf].height;  // 1
  const auto odm_conf_size = output_tensors_[layer_index_odm_conf].size /
                             batch;  // 278080 = 25280 * 11
  const auto odm_conf_channel = output_tensors_[layer_index_odm_conf].channel;
  const auto odm_conf_scale =
      vitis::ai::library::tensor_scale(output_tensors_[layer_index_odm_conf]);
  const auto odm_conf_addr =
      (int8_t*)output_tensors_[layer_index_odm_conf].get_data(idx);
  auto conf_softmax = vector<float>(odm_conf_size);

  if (ENV_PARAM(ENABLE_REFINE_DET_DEBUG) == 1)
    LOG(INFO) << "odm_conf_width " << odm_conf_width << " "       //
              << "odm_conf_height " << odm_conf_height << " "     //
              << "odm_conf_size " << odm_conf_size << " "         //
              << "odm_conf_channel " << odm_conf_channel << " "   //
              << "odm_conf_addr " << (void*)odm_conf_addr << " "  //
              << "odm_conf_scale " << odm_conf_scale << " "       //
              << "odm_loc_width " << odm_loc_width << " "         //
              << "odm_loc_height " << odm_loc_height << " "       //
              << "odm_loc_size " << odm_loc_size << " "           //
              << "odm_loc_scale " << odm_loc_scale << " "         //
              << "odm_loc_channel " << odm_loc_channel << " "     //
              << "odm_loc_addr " << (void*)odm_loc_addr << " "    //
        ;
  if (ENV_PARAM(ENABLE_REFINE_DET_DEBUG) == 1)
    LOG(INFO) << "arm_conf_width " << arm_conf_width << " "            //
              << "arm_conf_height " << arm_conf_height << " "          //
              << "arm_conf_size " << arm_conf_size << " "              //
              << "arm_conf_channel " << arm_conf_channel << " "        //
              << "arm_conf_addr " << (void*)arm_conf_addr << " "       //
              << "arm_conf_scale " << arm_conf_scale << " "            //
              << "arm_loc_width " << arm_loc_width << " "              //
              << "arm_loc_height " << arm_loc_height << " "            //
              << "arm_loc_size " << arm_loc_size << " "                //
              << "arm_loc_scale " << arm_loc_scale << " "              //
              << "arm_loc_channel " << arm_loc_channel << " "          //
              << "arm_loc_addr " << (void*)arm_loc_addr << " "         //
              << "num_classes " << num_classes_ << " "                 //
              << "conf_softmax.size() " << conf_softmax.size() << " "  //
        ;
  __TIC__(REFINEDET_ARM_SOFTMAX)
  if (1) {
    vitis::ai::softmax(odm_conf_addr, odm_conf_scale, num_classes_,
                       conf_softmax.size() / num_classes_, &conf_softmax[0]);
  }
  __TOC__(REFINEDET_ARM_SOFTMAX)
  __TIC__(REFINEDET_NMS)
  vitis::nnpp::refinedet::MultiDetObjects results;
  detector_->Detect(arm_loc_addr, odm_loc_addr, &conf_softmax[0], &results);
  __TOC__(REFINEDET_NMS)

  if (ENV_PARAM(ENABLE_REFINE_DET_DEBUG) == 1) {
    for (size_t i = 0; i < results.size(); ++i) {
      int label = get<0>(results[i]);
      float x = get<2>(results[i]).x * (sWidth);
      float y = get<2>(results[i]).y * (sHeight);
      int xmin = x;
      int ymin = y;
      int xmax = x + (get<2>(results[i]).width) * (sWidth);
      int ymax = y + (get<2>(results[i]).height) * (sHeight);
      float confidence = get<1>(results[i]);
      xmin = std::min(std::max(xmin, 0), sWidth);
      xmax = std::min(std::max(xmax, 0), sWidth);
      ymin = std::min(std::max(ymin, 0), sHeight);
      ymax = std::min(std::max(ymax, 0), sHeight);

      // cout<<get<2>(results[i]).x<<"  "<<get<2>(results[i]).y<<"
      // "<<get<2>(results[i]).width<<"  "<<get<2>(results[i]).height<<endl;

      LOG(INFO) << "RESULT1: " << label << "\t" << xmin << "\t" << ymin << "\t"
                << xmax << "\t" << ymax << "\t" << confidence << "\n";
    }
  }
  auto bboxes = std::vector<RefineDetResult::BoundingBox>();
  bboxes.reserve(results.size());
  for (const auto& r : results) {
    bboxes.push_back(
        RefineDetResult::BoundingBox{get<2>(r).x, get<2>(r).y, get<2>(r).width,
                                     get<2>(r).height, get<0>(r), get<1>(r)});
  }
  return RefineDetResult{sWidth, sHeight, bboxes};
}

std::vector<RefineDetResult> RefineDetPost::refine_det_post_process(
    size_t batch_size) {
  __TIC__(RefineDet_total_batch)

  auto ret = std::vector<vitis::ai::RefineDetResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; ++i) {
    ret.emplace_back(refine_det_post_process_internal(i));
  }
  __TOC__(RefineDet_total_batch)
  return ret;
}

}  // namespace ai
}  // namespace vitis
