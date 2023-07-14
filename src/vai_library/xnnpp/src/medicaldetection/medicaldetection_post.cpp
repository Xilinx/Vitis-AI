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
#include "medicaldetection_post.hpp"

#include <sys/stat.h>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

#include "priorbox.hpp"

namespace vitis {
namespace ai {
namespace medicaldetection {

DEF_ENV_PARAM(ENABLE_REFINE_DET_DEBUG, "0")

MedicalDetectionPost::~MedicalDetectionPost() {}

MedicalDetectionPost::MedicalDetectionPost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, int& real_batch_sizex)
    : input_tensors_(input_tensors),
      output_tensors_(output_tensors),
      real_batch_size(real_batch_sizex) {
  vitis::ai::medicaldetection::PriorBox pb(
      std::vector<int>(config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .input_shape()
                           .begin(),
                       config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .input_shape()
                           .end()),
      std::vector<int>(config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .feature_shapes()
                           .begin(),
                       config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .feature_shapes()
                           .end()),
      std::vector<int>(config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .min_sizes()
                           .begin(),
                       config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .min_sizes()
                           .end()),
      std::vector<int>(config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .max_sizes()
                           .begin(),
                       config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .max_sizes()
                           .end()),
      std::vector<float>(config.medical_refine_det_param()
                             .medical_prior_box_param()
                             .aspect_ratios()
                             .begin(),
                         config.medical_refine_det_param()
                             .medical_prior_box_param()
                             .aspect_ratios()
                             .end()),
      std::vector<int>(config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .steps()
                           .begin(),
                       config.medical_refine_det_param()
                           .medical_prior_box_param()
                           .steps()
                           .end()),
      config.medical_refine_det_param().medical_prior_box_param().offset());
  priors_ = pb.get_pirors();
  detector_ = std::make_unique<vitis::ai::medicaldetection::SSDDetector>(
      config.medical_refine_det_param().num_classes() + 1, priors_,
      config.medical_refine_det_param().scale_xy(),
      config.medical_refine_det_param().scale_wh(),
      config.medical_refine_det_param().conf_threshold(),
      config.medical_refine_det_param().keep_top_k(),
      config.medical_refine_det_param().top_k(),
      config.medical_refine_det_param().nms_threshold());

  for (auto it = config.medical_refine_det_param().output_info().begin();
       it != config.medical_refine_det_param().output_info().end(); it++) {
    for (auto i = 0u; i < output_tensors.size(); i++) {
      if (output_tensors[i].name.find(it->name()) != std::string::npos) {
        if (it->type() == 1) {
          ARM_CONF_IDX = i;
          break;
        } else if (it->type() == 2) {
          ARM_LOC_IDX = i;
          break;
        } else if (it->type() == 3) {
          ODM_CONF_IDX = i;
          break;
        } else if (it->type() == 4) {
          ODM_LOC_IDX = i;
          break;
        }
      }
    }
  }
}

std::vector<vitis::ai::MedicalDetectionResult>
MedicalDetectionPost::medicaldetection_post_process() {
  auto ret = std::vector<vitis::ai::MedicalDetectionResult>{};
  ret.reserve(real_batch_size);

  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(medicaldetection_post_process(i));
  }
  return ret;
}

vitis::ai::MedicalDetectionResult
MedicalDetectionPost::medicaldetection_post_process(unsigned int idx) {
  const auto arm_loc_scale =
      vitis::ai::library::tensor_scale(output_tensors_[ARM_LOC_IDX]);
  const auto arm_loc_addr = (int8_t*)output_tensors_[ARM_LOC_IDX].get_data(idx);

  const auto arm_conf_scale =
      vitis::ai::library::tensor_scale(output_tensors_[ARM_CONF_IDX]);
  const auto arm_conf_addr =
      (int8_t*)output_tensors_[ARM_CONF_IDX].get_data(idx);

  const auto odm_loc_addr = (int8_t*)output_tensors_[ODM_LOC_IDX].get_data(idx);
  const auto odm_loc_scale =
      vitis::ai::library::tensor_scale(output_tensors_[ODM_LOC_IDX]);

  const auto odm_conf_scale =
      vitis::ai::library::tensor_scale(output_tensors_[ODM_CONF_IDX]);
  const auto odm_conf_addr =
      (int8_t*)output_tensors_[ODM_CONF_IDX].get_data(idx);

  if (ENV_PARAM(ENABLE_REFINE_DET_DEBUG) == 1) {
    LOG(INFO) << "odm_conf_addr " << (void*)odm_conf_addr << " "  //
              << "odm_conf_scale " << odm_conf_scale << " "       //
              << "odm_loc_addr " << (void*)odm_loc_addr << " "    //
              << "odm_loc_scale " << odm_loc_scale << " "         //
        ;
    LOG(INFO) << "arm_conf_addr " << (void*)arm_conf_addr << " "  //
              << "arm_conf_scale " << arm_conf_scale << " "       //
              << "arm_loc_addr " << (void*)arm_loc_addr << " "    //
              << "arm_loc_scale " << arm_loc_scale << " "         //
        ;
  }

  MedicalDetectionResult results{int(input_tensors_[0].width),
                                 int(input_tensors_[0].height)};

  detector_->detect(arm_loc_addr, odm_loc_addr, arm_conf_addr, odm_conf_addr,
                    arm_loc_scale, odm_loc_scale, arm_conf_scale,
                    odm_conf_scale, results);

  return results;
}

}  // namespace medicaldetection
}  // namespace ai
}  // namespace vitis
