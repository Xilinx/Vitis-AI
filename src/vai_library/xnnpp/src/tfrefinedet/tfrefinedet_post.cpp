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
#include "tfrefinedet_post.hpp"

#include <sys/stat.h>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

#include "../medicaldetection/priorbox.hpp"

namespace vitis {
namespace ai {
namespace tfrefinedet {

DEF_ENV_PARAM(ENABLE_REFINE_DET_DEBUG, "0")

TFRefineDetPost::~TFRefineDetPost() {}

TFRefineDetPost::TFRefineDetPost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : input_tensors_(input_tensors) {
  const auto& param = config.tfrefinedet_param();
  auto layername = std::vector<std::string>(param.layer_name().begin(),
                                            param.layer_name().end());
  for (auto& name : layername) {
    for (size_t j = 0u; j < output_tensors.size(); j++) {
      if (output_tensors[j].name.find(name) != std::string::npos) {
        output_tensors_.emplace_back(output_tensors[j]);
        LOG_IF(INFO, ENV_PARAM(ENABLE_REFINE_DET_DEBUG))
            << name << " " << output_tensors[j].name;
        break;
      }
    }
  }
  const auto& pbparam = param.prior_box_param();
  medicaldetection::PriorBox pb(
      std::vector<int>(pbparam.input_shape().begin(),
                       pbparam.input_shape().end()),
      std::vector<int>(pbparam.feature_shapes().begin(),
                       pbparam.feature_shapes().end()),
      std::vector<int>(pbparam.min_sizes().begin(), pbparam.min_sizes().end()),
      std::vector<int>(pbparam.max_sizes().begin(), pbparam.max_sizes().end()),
      std::vector<float>(pbparam.aspect_ratios().begin(),
                         pbparam.aspect_ratios().end()),
      std::vector<int>(pbparam.steps().begin(), pbparam.steps().end()),
      pbparam.offset());
  detector_ = std::make_unique<vitis::ai::tfrefinedet::SSDDetector>(
      param.num_classes(), pb.get_pirors(), param.scale_xy(), param.scale_wh(),
      param.conf_threshold(), param.keep_top_k(), param.top_k(),
      param.nms_threshold());
}

std::vector<vitis::ai::RefineDetResult>
TFRefineDetPost::tfrefinedet_post_process(size_t batch_size) {
  auto ret = std::vector<vitis::ai::RefineDetResult>{};
  ret.reserve(batch_size);

  for (auto i = 0u; i < batch_size; ++i) {
    ret.emplace_back(tfrefinedet_post_process_internal(i));
  }
  return ret;
}

vitis::ai::RefineDetResult TFRefineDetPost::tfrefinedet_post_process_internal(
    unsigned int idx) {
  constexpr size_t arm_loc = 0;
  constexpr size_t arm_conf = 1;
  constexpr size_t odm_loc = 2;
  constexpr size_t odm_conf = 3;
  const auto arm_loc_scale = library::tensor_scale(output_tensors_[arm_loc]);
  const auto arm_loc_addr = (int8_t*)output_tensors_[arm_loc].get_data(idx);
  const auto arm_conf_scale = library::tensor_scale(output_tensors_[arm_conf]);
  const auto arm_conf_addr = (int8_t*)output_tensors_[arm_conf].get_data(idx);

  const auto odm_loc_scale = library::tensor_scale(output_tensors_[odm_loc]);
  const auto odm_loc_addr = (int8_t*)output_tensors_[odm_loc].get_data(idx);
  const auto odm_conf_scale = library::tensor_scale(output_tensors_[odm_conf]);
  const auto odm_conf_addr = (int8_t*)output_tensors_[odm_conf].get_data(idx);

  LOG_IF(INFO, ENV_PARAM(ENABLE_REFINE_DET_DEBUG))
      << "odm_conf_addr " << (void*)odm_conf_addr << " "  //
      << "odm_conf_scale " << odm_conf_scale << " "       //
      << "odm_loc_addr " << (void*)odm_loc_addr << " "    //
      << "odm_loc_scale " << odm_loc_scale << " "         //
      ;
  LOG_IF(INFO, ENV_PARAM(ENABLE_REFINE_DET_DEBUG))
      << "arm_conf_addr " << (void*)arm_conf_addr << " "  //
      << "arm_conf_scale " << arm_conf_scale << " "       //
      << "arm_loc_addr " << (void*)arm_loc_addr << " "    //
      << "arm_loc_scale " << arm_loc_scale << " "         //
      ;

  RefineDetResult results{int(input_tensors_[0].width),
                          int(input_tensors_[0].height)};

  detector_->detect(arm_loc_addr, odm_loc_addr, arm_conf_addr, odm_conf_addr,
                    arm_loc_scale, odm_loc_scale, arm_conf_scale,
                    odm_conf_scale, results);

  return results;
}

}  // namespace tfrefinedet
}  // namespace ai
}  // namespace vitis
