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

namespace  vitis {
namespace ai {


SSDPost::~SSDPost(){};

SSDPost::SSDPost(const std::vector<vitis::ai::library::InputTensor> &input_tensors,
                 const std::vector<vitis::ai::library::OutputTensor> &output_tensors,
                 const vitis::ai::proto::DpuModelParam &config)
    : num_classes_(config.ssd_param().num_classes()),
      is_tf_(config.is_tf()),
      is_mlperf_(config.ssd_param().is_mlperf()),
      //is_mlperf_(config.is_mlperf()),
      priors_{vitis::ai::dpssd::CreatePriors((int)input_tensors[0].width,
                                       (int)input_tensors[0].height,
                                       is_tf_, is_mlperf_,
                                       config.ssd_param().prior_box_param())},
      input_tensors_(input_tensors),
      output_tensors_(output_tensors) {

  if (config.ssd_param().output_info().size() == 0
      && config.ssd_param().bbox_layer_index().size() == 0) {
    // Not use output_info and bbox_layer_indexes 
    // 2 output layers: 0:bbox, 1:confidence
    bbox_layer_indexes_.emplace(0);
    conf_layer_indexes_.emplace(1);
  } else if (config.ssd_param().output_info().size() == 0) {
    // Not use output_info means all output layers are valid
    // but neither traditional order nor 2 output layers
    // e.g mlperf 12 output layers
    // e.g a model has 2 output layers but 0:confidence, 1:bbox 
    for (auto it = config.ssd_param().bbox_layer_index().begin();
         it != config.ssd_param().bbox_layer_index().end(); it++) {
      //std::cout << "Add bbox index : " << *it << std::endl;
      bbox_layer_indexes_.emplace(*it);
    }

    for (auto i = 0u; i < output_tensors.size(); i++) {
      if (bbox_layer_indexes_.count(i) == 0) {
        conf_layer_indexes_.emplace(i);
      }
    }
  } else {  // Not all output layers are valid
    for (auto it = config.ssd_param().output_info().begin();
         it != config.ssd_param().output_info().end(); it++) {
      for (auto i = 0u; i < output_tensors.size(); i++) {
        if (output_tensors[i].name.find(it->name()) !=  std::string::npos) {
          if (it->type() == 1) { //conf
            conf_layer_indexes_.emplace(i);
          } else if (it->type() == 2) { // bbox
            bbox_layer_indexes_.emplace(i);
          }
          break;
        }
      }
    }
  }

  assert(bbox_layer_indexes_.size() == conf_layer_indexes_.size());

  auto batch_size =  input_tensors_[0].batch;

  // output tensors may more than valid output tensors
  auto valid_output_tensors_size  = bbox_layer_indexes_.size() + conf_layer_indexes_.size();

  output_layer_infos_.reserve(valid_output_tensors_size);
  output_layer_infos_.assign(valid_output_tensors_size, SSDOutputInfo{});
  auto score_index = 0u;
  auto bbox_index = 0u;
  auto k = 0u;
 
  for (auto i = 0u; i < output_tensors_.size(); ++i) {
    //std::cout << "output tensor index:" << i
    //          << ", ptr : " << (void *)output_tensors[i].get_data(0)
    //          << ", size: " << output_tensors[i].size
    //          << ", channel: " << output_tensors[i].channel
    //          << ", width: " << output_tensors[i].width
    //          << ", height: " << output_tensors[i].height
    //          << ", name: " << output_tensors[i].name
    //          << std::endl;
    if (bbox_layer_indexes_.count(i) == 0 &&
	conf_layer_indexes_.count(i) == 0) {
      continue;
    }
    output_layer_infos_[k].channel = i;
    output_layer_infos_[k].base_ptr = (int8_t *)output_tensors_[i].get_data(0);
    output_layer_infos_[k].ptr =  output_layer_infos_[k].base_ptr;

    if (bbox_layer_indexes_.count(i)) {
      output_layer_infos_[k].index_begin = bbox_index;
      output_layer_infos_[k].index_size = output_tensors_[i].size / batch_size / 4;
      bbox_index += output_layer_infos_[i].index_size;
    } else if (conf_layer_indexes_.count(i)) {
      output_layer_infos_[k].index_begin = score_index;
      output_layer_infos_[k].index_size = output_tensors_[i].size /  batch_size / num_classes_;
      score_index += output_layer_infos_[i].index_size;
    }
    output_layer_infos_[k].scale = vitis::ai::library::tensor_scale(output_tensors_[i]);
    output_layer_infos_[k].size = output_tensors_[i].size / batch_size;
    //std::cout << "Init output info: " << k
    //          << ", output tensor index: " << i
    //          << ", ptr : " << (void *)output_layer_infos_[k].ptr
    //          << ", index_begin : " << output_layer_infos_[k].index_begin
    //          << ", index_size: " << output_layer_infos_[k].index_size
    //          << ", scale: " << output_layer_infos_[k].scale
    //          << ", size: " << output_layer_infos_[k].size
    //          << std::endl;
    k++;
    
  }

  detector_ = vitis::ai::dpssd::CreateSSDUniform(priors_, config);
}

vitis::ai::SSDResult SSDPost::ssd_post_process_internal_uniform( unsigned int idx) {
  bool need_copy_and_transpose = false;
  // If valid output tensors > 2, means we need to copy all layers and concate score
  // layers and loction layers separately.
  // It indicates the model has more opearations such as transpose which can't
  // be computed ty DPU.
  if (output_layer_infos_.size() > 2) {
    need_copy_and_transpose = true;
  }

  // set output pointer
  for (uint32_t k = 0; k < output_layer_infos_.size(); k++) {
    auto index = output_layer_infos_[k].channel;
    output_layer_infos_[k].ptr = (int8_t *)output_tensors_[index].get_data(idx);
  }

  std::vector<std::vector<int8_t>> copyed_data(output_layer_infos_.size());

  if (need_copy_and_transpose) {
__TIC__(SSD_copy)
    for (uint32_t k = 0; k < output_layer_infos_.size(); k++) {
      auto index = output_layer_infos_[k].channel;
      if (bbox_layer_indexes_.count(index) == 0 && conf_layer_indexes_.count(index) == 0) {
        continue;
      }
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
      if (bbox_layer_indexes_.count(index)) {
        CH = 4;
      }
      Na = C / CH;
      // std::cout << "Na " << Na << std::endl; 
      for(auto n = 0u; n < Na; n++) {
        for(auto hw = 0u; hw < H * W; ++hw) {
          for (auto ch = 0; ch < CH; ++ch) {
            copyed_data[k][n * H * W * CH + hw * CH + ch] =
                ((int8_t *)output_tensors_[index].get_data(idx))[hw * C + ch * Na + n];
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
    auto index = output_layer_infos_[k].channel;
    if (conf_layer_indexes_.count(index)) {
      auto offset = output_layer_infos_[k].index_begin * num_classes_;
      // std::cout << "offset " << offset << std::endl;
      vitis::ai::softmax((int8_t*)output_layer_infos_[k].ptr,
                          output_layer_infos_[k].scale, 
                          num_classes_,
                          output_layer_infos_[k].index_size,
                          softmax_data_.data() + offset);
    }
  }
  __TOC__(SSD_softmax)
 
  __TIC__(SSD_after)
  std::vector<SSDResult::BoundingBox> bboxes;
  SSDResult results{(int)input_tensors_[0].width,
                    (int)input_tensors_[0].height, bboxes};
  std::map<uint32_t, SSDOutputInfo> bbox_layer_infos;
  //for (auto i : bbox_layer_indexes_) {
  for (auto i = 0u; i < output_layer_infos_.size(); ++i) {
    if (bbox_layer_indexes_.count(output_layer_infos_[i].channel)) {
      bbox_layer_infos.emplace(std::make_pair(i, output_layer_infos_[i]));
    }
  }
  detector_->detect(bbox_layer_infos, softmax_data_.data(), &results);

  __TOC__(SSD_after)
  return results;
}

vitis::ai::SSDResult SSDPost::ssd_post_process(unsigned int idx) {
  return ssd_post_process_internal_uniform( idx);
}

std::vector<vitis::ai::SSDResult> SSDPost::ssd_post_process() {
  __TIC__(SSD_total_batch)
  auto batch_size = input_tensors_[0].batch;		
  auto ret = std::vector<vitis::ai::SSDResult>{};		
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; ++i) {     		
    ret.emplace_back(ssd_post_process(i));		
  }  		
  __TOC__(SSD_total_batch)
  return ret;
}


} // namespace ai
} // namespace vitis

