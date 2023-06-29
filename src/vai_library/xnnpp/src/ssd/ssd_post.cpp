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
#include "./ssd_post.hpp"

#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
// #include <vitis/ai/env_config.hpp>
#include "./prior_boxes.hpp"
#include "./ssd_detector.hpp"

using namespace std;

namespace vitis {
namespace ai {

SSDPost::~SSDPost(){};

SSDPost::SSDPost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : num_classes_(config.ssd_param().num_classes()),
      is_tf_(config.is_tf()),
      is_mlperf_(config.ssd_param().is_mlperf()),
      // is_mlperf_(config.is_mlperf()),
      priors_{vitis::ai::dpssd::CreatePriors(
          (int)input_tensors[0].width, (int)input_tensors[0].height, is_tf_,
          is_mlperf_, config.ssd_param().prior_box_param())},
      input_tensors_(input_tensors),
      output_tensors_(output_tensors) {
  // default valid output tensors size = 2 used for common cases which do the
  // concat on DPU not in the post processing codes
  auto valid_output_tensors_size = 2;
  if (config.ssd_param().output_info().size() == 0 &&
      config.ssd_param().bbox_layer_index().size() == 0) {
    // Not use output_info and bbox_layer_indexes
    // 2 output layers: 0:bbox, 1:confidence
    bbox_layer_indexes_.emplace(0);
    conf_layer_indexes_.emplace(1);
    output_layer_infos_.reserve(valid_output_tensors_size);
    output_layer_infos_.assign(valid_output_tensors_size, vitis::ai::dpssd::SSDOutputInfo{});
    output_layer_infos_[0].output_tensor_index = 0;
    output_layer_infos_[0].type = 2;  // conf:1, bbox:2
    output_layer_infos_[0].order = 0;
    output_layer_infos_[1].output_tensor_index = 1;
    output_layer_infos_[1].type = 1;  // conf:1, bbox:2
    output_layer_infos_[1].order = 0;
  } else {  // Not all output layers are valid
    // valid output tensor information should be assigned in config prototxt
    valid_output_tensors_size = config.ssd_param().output_info().size();
    assert(valid_output_tensors_size % 2 == 0);
    output_layer_infos_.reserve(valid_output_tensors_size);
    output_layer_infos_.assign(valid_output_tensors_size, vitis::ai::dpssd::SSDOutputInfo{});

    // In output_layer_infos_, order as: bbox1 bbox2... conf1 conf2...
    int bbox_base_index = 0;
    int conf_base_index = valid_output_tensors_size / 2;
    for (auto it = config.ssd_param().output_info().begin();
         it != config.ssd_param().output_info().end(); it++) {
      for (auto i = 0u; i < output_tensors.size(); i++) {
        if (output_tensors[i].name.find(it->name()) != std::string::npos) {
          int index = 0;
          if (it->type() == 1) {  // conf
            conf_layer_indexes_.emplace(i);
            index = conf_base_index + it->order();
          } else if (it->type() == 2) {  // bbox
            bbox_layer_indexes_.emplace(i);
            index = bbox_base_index + it->order();
          }
          // std::cout << "tensor index : " << i << std::endl;
          output_layer_infos_[index].output_tensor_index = i;
          output_layer_infos_[index].type = it->type();  // conf:1, bbox:2
          output_layer_infos_[index].order = it->order();
          break;
        }
      }
    }
  }

  assert(bbox_layer_indexes_.size() == conf_layer_indexes_.size());

  auto batch_size = input_tensors_[0].batch;

  auto score_index = 0u;
  auto bbox_index = 0u;
  // auto k = 0u;

  // for (auto i = 0u; i < output_tensors_.size(); ++i) {
  for (auto k = 0; k < valid_output_tensors_size; ++k) {
    int i = output_layer_infos_[k].output_tensor_index;
    // std::cout << "output tensor index:" << i
    //          << ", ptr : " << (void *)output_tensors[i].get_data(0)
    //          << ", size: " << output_tensors[i].size
    //          << ", output_tensor_index: " <<
    //          output_tensors[i].output_tensor_index
    //          << ", width: " << output_tensors[i].width
    //          << ", height: " << output_tensors[i].height
    //          << ", name: " << output_tensors[i].name
    //          << std::endl;
    output_layer_infos_[k].base_ptr = (int8_t*)output_tensors_[i].get_data(0);
    output_layer_infos_[k].ptr = output_layer_infos_[k].base_ptr;
    // bool special = true;
    if (output_layer_infos_[k].type == 2) {  // bbox
      output_layer_infos_[k].index_begin = bbox_index;
      // if (special) { // special case
      //  output_layer_infos_[k].bbox_single_size = 6;
      //} else {
      output_layer_infos_[k].bbox_single_size = 4;
      //}
      output_layer_infos_[k].index_size =
          output_tensors_[i].size / batch_size /
          output_layer_infos_[k].bbox_single_size;
      bbox_index += output_layer_infos_[k].index_size;
    } else if (conf_layer_indexes_.count(i)) {
      output_layer_infos_[k].index_begin = score_index;
      output_layer_infos_[k].index_size =
          output_tensors_[i].size / batch_size / num_classes_;
      score_index += output_layer_infos_[k].index_size;
    }
    output_layer_infos_[k].scale =
        vitis::ai::library::tensor_scale(output_tensors_[i]);
    output_layer_infos_[k].size = output_tensors_[i].size / batch_size;
    // std::cout << "Init output info: " << k
    //         << ", output tensor index: " << i
    //         << ", ptr : " << (void *)output_layer_infos_[k].ptr
    //         << ", index_begin : " << output_layer_infos_[k].index_begin
    //         << ", index_size: " << output_layer_infos_[k].index_size
    //         << ", scale: " << output_layer_infos_[k].scale
    //         << ", size: " << output_layer_infos_[k].size
    //         << std::endl;
  }

  detector_ = vitis::ai::dpssd::CreateSSDUniform(priors_, config);
}

vitis::ai::SSDResult SSDPost::ssd_post_process_internal_uniform(
    unsigned int idx) {
  // std::cout << "begin priors" << std::endl;
  //// print priors
  // for (auto i = 0u; i < priors_.size(); ++i) {
  //  std::cout << "priors[" << i << "]:";
  //  for (auto j = 8u; j < priors_[i]->size(); ++j) {
  //    std::cout << (*priors_[i])[j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  bool need_copy_and_transpose = false;
  // If valid output tensors > 2, means we need to copy all layers and concate
  // score layers and loction layers separately. It indicates the model has more
  // opearations such as transpose which can't be computed ty DPU.
  if (output_layer_infos_.size() > 2 && is_mlperf_) {
    need_copy_and_transpose = true;
  }

  // set output pointer
  for (uint32_t k = 0; k < output_layer_infos_.size(); k++) {
    auto index = output_layer_infos_[k].output_tensor_index;
    output_layer_infos_[k].ptr = (int8_t*)output_tensors_[index].get_data(idx);
  }

  std::vector<std::vector<int8_t>> copyed_data(output_layer_infos_.size());

  if (need_copy_and_transpose) {
    __TIC__(SSD_copy)
    for (uint32_t k = 0; k < output_layer_infos_.size(); k++) {
      auto index = output_layer_infos_[k].output_tensor_index;
      copyed_data[k].reserve(output_tensors_[index].size);
      copyed_data[k].assign(output_tensors_[index].size, 0);
      // std::cout << "copyed_data[" << index << "] size "
      //           << copyed_data[index].size() << std::endl;
      output_layer_infos_[k].ptr = copyed_data[k].data();

      auto H = output_tensors_[index].height;
      auto W = output_tensors_[index].width;
      auto C = output_tensors_[index].channel;
      // std::cout << "H: " << H
      //          << "W: " << W
      //          << "C: " << C
      //          << std::endl;
      // output size = H * W * C; C = CH * Na;
      auto Na = 0u;
      auto CH = num_classes_;
      if (output_layer_infos_[k].type == 2) {  // bbox
        // Usualy CH=4, some special case it is 6;
        CH = output_layer_infos_[k].bbox_single_size;
      }
      Na = C / CH;
      // std::cout << "Na " << Na << std::endl;
      for (auto n = 0u; n < Na; n++) {
        for (auto hw = 0u; hw < H * W; ++hw) {
          for (auto ch = 0u; ch < CH; ++ch) {
            copyed_data[k][n * H * W * CH + hw * CH + ch] =
                ((int8_t*)output_tensors_[index].get_data(
                    idx))[hw * C + ch * Na + n];
          }
        }
      }
    }
    __TOC__(SSD_copy)
  }

  __TIC__(SSD_softmax)
  std::vector<float> softmax_data_(priors_.size() * num_classes_);
  // std::cout << "softmax_data size : " << softmax_data_.size() << std::endl;
  for (auto k = 0u; k < output_layer_infos_.size(); k++) {
    if (output_layer_infos_[k].type == 1) {  // conf
      auto offset = output_layer_infos_[k].index_begin * num_classes_;
      // std::cout << "offset " << offset << std::endl;
      vitis::ai::softmax((int8_t*)output_layer_infos_[k].ptr,
                         output_layer_infos_[k].scale, num_classes_,
                         output_layer_infos_[k].index_size,
                         softmax_data_.data() + offset);
    }
  }
  __TOC__(SSD_softmax)

  __TIC__(SSD_detect)
  std::vector<SSDResult::BoundingBox> bboxes;
  SSDResult results{(int)input_tensors_[0].width, (int)input_tensors_[0].height,
                    bboxes};
  std::map<uint32_t, vitis::ai::dpssd::SSDOutputInfo> bbox_layer_infos;
  // for (auto i : bbox_layer_indexes_) {
  for (auto i = 0u; i < output_layer_infos_.size(); ++i) {
    if (output_layer_infos_[i].type == 2) {  // bbox
      bbox_layer_infos.emplace(std::make_pair(i, output_layer_infos_[i]));
    }
  }
  detector_->detect(bbox_layer_infos, softmax_data_.data(), &results);

  __TOC__(SSD_detect)
  return results;
}

std::vector<vitis::ai::SSDResult> SSDPost::ssd_post_process(size_t batch_size) {
  __TIC__(SSD_total_batch)
  //  auto batch_size = input_tensors_[0].batch;
  auto ret = std::vector<vitis::ai::SSDResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; ++i) {
    ret.emplace_back(ssd_post_process_internal_uniform(i));
  }
  __TOC__(SSD_total_batch)
  return ret;
}

}  // namespace ai
}  // namespace vitis
