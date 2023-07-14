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

#include "./retinaface_post.hpp"
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

#include "./anchor.hpp"

DEF_ENV_PARAM(DEBUG_XNNPP_RETINAFACE, "0")
DEF_ENV_PARAM(DEBUG_XNNPP_RETINAFACE_ANCHOR, "0")
using namespace std;
namespace vitis {
namespace ai {
namespace retinaface {

static void my_softmax_2c(float *input1, float *input2, float* output1, float *output2) {
  float sum = 0.f;
  *output1 = exp(*input1);
  sum += *output1;
  *output2 = exp(*input2);
  sum += *output2;
  *output1 /= sum;
  *output2 /= sum;
}

static void my_softmax_2c(float *input, unsigned int group, float *output) {
  for (unsigned int i = 0; i < group; ++i) {
    my_softmax_2c(input + i * 4, input + i * 4 + 2, output + i * 4, output + i * 4 +2);
    my_softmax_2c(input + i * 4 + 1, input + i * 4 + 3, output + i * 4 + 1, output + i * 4 + 3);
    //if (i < 10) {
    //  LOG(INFO) << "softmax group:" << group
    //            << ", input1: " <<  *(input + i * 4)
    //            << ", input2: " <<  *(input + i * 4 + 2)
    //            << ", input3: " <<  *(input + i * 4 + 1)
    //            << ", input4: " <<  *(input + i * 4 + 3);

    //  LOG(INFO) << "softmax group:" << group
    //            << ", output1: " <<  *(output + i * 4)
    //            << ", output2: " <<  *(output + i * 4 + 2)
    //            << ", output3: " <<  *(output + i * 4 + 1)
    //            << ", output4: " <<  *(output + i * 4 + 3);
    //}
  }
} 
}}}

namespace vitis {
namespace ai {

static void debug_retinaface_output_info(const RetinaFaceOutputInfo &output_info) {
  LOG(INFO) << "name :" << output_info.layer_name;
  LOG(INFO) << "tensor_index:" << (int)output_info.output_tensor_index;
  LOG(INFO) << "type :" << (int)output_info.type;
  LOG(INFO) << "anchor type:" << (int)output_info.anchor_type;
  LOG(INFO) << "anchor index:" << (int)output_info.anchor_index;
  LOG(INFO) << "anchor stride:" << output_info.stride;
  LOG(INFO) << "scale :" << output_info.scale;
  LOG(INFO) << "size :" << output_info.size;

}

//static void read_float_bin(const char *file_path, float *dst, int32_t size) {
//  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
//         << "read file: " << file_path << ", size:" << size; 
//  std::ifstream in(file_path, ios::in|ios::binary);
//  in.read((char *)dst, sizeof(float) * size);
//  in.close();
//}

static void fill_retinaface_output_info(RetinaFaceOutputInfo &output_info, const std::string &name,  
                                        int8_t output_tensor_index, int8_t type, int8_t anchor_type, int8_t anchor_index,
                                        int stride, int8_t *base_ptr, int8_t *ptr, float scale, uint32_t size) {
  output_info.layer_name = name; 
  output_info.output_tensor_index = output_tensor_index; 
  output_info.type = type; 
  output_info.anchor_type = anchor_type;
  output_info.anchor_index = anchor_index; 
  output_info.stride = stride; 
  output_info.base_ptr = base_ptr; 
  output_info.ptr = ptr; 
  output_info.scale = scale; 
  output_info.size = size; 
} 

RetinaFacePost::~RetinaFacePost(){};
RetinaFacePost::RetinaFacePost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : input_tensors_(input_tensors),
      output_tensors_(output_tensors),
      nms_thresh_{config.retinaface_param().nms_threshold()},
      det_thresh_{config.retinaface_param().det_threshold()} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "config.retinaface_param().output_info().size(): "
        << config.retinaface_param().output_info().size();

  // 1. parse anchor param 
  // only set anchor_info of StrideLayers
  for (auto it = config.retinaface_param().anchor_param().begin();
         it != config.retinaface_param().anchor_param().end(); it++) {
    auto anchor_info = AnchorInfo{it->stride(), it->base_size(), 
                                  std::vector<float>(it->ratios().begin(), it->ratios().end()),
                                  std::vector<int>(it->scales().begin(), it->scales().end())};
    if(ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
      LOG(INFO) << "config.retinaface_param().anchor_param() it->stride: " << it->stride()
            << " it->base_size: " << it->base_size(); 
      for (auto i = 0u; i < anchor_info.ratios.size(); ++i) {
        LOG(INFO) << "ratios[" << i << "] :" << anchor_info.ratios[i];
      }
      for (auto i = 0u; i < anchor_info.scales.size(); ++i) {
        LOG(INFO) << "scales[" << i << "] :" << anchor_info.scales[i];
      }
    }

    stride_layers_.emplace(anchor_info.stride, StrideLayers{anchor_info, 0}); // set conf_data_size = 0
  } 

  // 2. generate anchors
  // note: need reverse
  priors_ = generate_anchors((int)input_tensors[0].width, 
                             (int)input_tensors[0].height,
                            stride_layers_);

  if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE_ANCHOR)) {
    LOG(INFO) << "priors_ size: " << priors_.size();
    for (auto i = 0u; i < priors_.size(); ++i) {
      LOG(INFO) << "priors_[" << i << "] : " 
                << priors_[i][0] << ", " 
                << priors_[i][1] << ", " 
                << priors_[i][2] << ", " 
                << priors_[i][3] ;
    }    
  }
  // 2.x debug output tensor
  if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
    for (auto i = 0u; i < output_tensors.size(); i++) {
      LOG(INFO) << "output_tensors[" << i << "].name " << output_tensors[i].name;
      LOG(INFO) << "size:"  << output_tensors[i].size;
      LOG(INFO) << "w:"  << output_tensors[i].width;
      LOG(INFO) << "h:"  << output_tensors[i].height;
      LOG(INFO) << "c:"  << output_tensors[i].channel;
    }
  }

  // 3. parse conf output layers info
  auto batch_size = input_tensors_[0].batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) << "batch size: " << batch_size;
  for (auto it = config.retinaface_param().output_info().begin();
         it != config.retinaface_param().output_info().end(); it++) {
  
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE))   
          << "config.retinaface_param().output_info() it->name : " << it->name()
          << " it->type : " << it->type() 
          << " it->stride : " << it->stride();
    auto stride = it->stride();
    if (stride_layers_.find(stride) == stride_layers_.end()) { // config stride invalid
      continue;
    }
    for (auto i = 0u; i < output_tensors.size(); i++) {
      //if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
      //  auto ret = output_tensors[i].name.find(it->name());
      //  LOG(INFO) << "output_tensors[" << i << "].name " << output_tensors[i].name;
      //  LOG(INFO) << "it->name: " << it->name();
      //  LOG(INFO) << "find result: " << ret;
      //}
      if (output_tensors[i].name.find(it->name()) != std::string::npos) {
        if (it->type() == 1) { // conf
          auto anchor_type = it->output_anchor_info().type();
          auto anchor_index = it->output_anchor_info().index();
          //uint32_t key = (stride << 16) + (anchor_index << 8) + anchor_type;
          stride_layers_[stride].conf_data_size += output_tensors[i].width * output_tensors[i].height;
          conf_layer_infos_.emplace_back(
                  RetinaFaceOutputInfo{output_tensors[i].name, 
                  (int8_t)i, (int8_t)it->type(), 
                  (int8_t)anchor_type, (int8_t)anchor_index,
                  stride, (int8_t *)output_tensors[i].get_data(0),
                  (int8_t *)output_tensors[i].get_data(0), 
                  // debug
                  //nullptr,
                  vitis::ai::library::tensor_scale(output_tensors_[i]),
                  (uint32_t)(output_tensors[i].size / batch_size)});
        } else if (it->type() == 2) { // bbox  
          fill_retinaface_output_info(stride_layers_[stride].bbox_layer, output_tensors[i].name,           
                  (int8_t)i, (int8_t)it->type(), -1, -1, stride, (int8_t *)output_tensors[i].get_data(0),
                  (int8_t *)output_tensors[i].get_data(0), 
                  vitis::ai::library::tensor_scale(output_tensors_[i]),
                  (uint32_t)(output_tensors[i].size / batch_size));
                
          if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
            debug_retinaface_output_info(stride_layers_[stride].bbox_layer);
          }
        } else if (it->type() == 3) { // landmark
          fill_retinaface_output_info(stride_layers_[stride].landmark_layer, output_tensors[i].name,
                  (int8_t)i, (int8_t)it->type(),-1, -1, stride, (int8_t *)output_tensors[i].get_data(0),
                  (int8_t *)output_tensors[i].get_data(0), 
                  vitis::ai::library::tensor_scale(output_tensors_[i]),
                  (uint32_t)(output_tensors[i].size / batch_size));
          if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
            debug_retinaface_output_info(stride_layers_[stride].landmark_layer);
          }
        } else {
          continue;
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) << "continue";
        break;
      } 
    }
  }

  if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
    for (auto it = stride_layers_.begin(); it != stride_layers_.end(); ++it) {
      LOG(INFO) << "stride_layers_[" << it->first << "]:";
      LOG(INFO) << "bbox_layer:"; 
      debug_retinaface_output_info(it->second.bbox_layer);
      LOG(INFO) << "landmark_layer:"; 
      debug_retinaface_output_info(it->second.landmark_layer);
    }
  }
  if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
    for (auto it = conf_layer_infos_.begin(); it != conf_layer_infos_.end(); ++it) {
      debug_retinaface_output_info(*it);
    }
  }

  // 3.1 init copy conf data
  auto softmax_data_size = 0;
  for (auto it = stride_layers_.begin(); it != stride_layers_.end(); ++it) {
    it->second.copyed_conf_data.resize(it->second.conf_data_size);
    softmax_data_size += it->second.conf_data_size;
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE))
          << "stride : " << it->first
          << " conf_data_size :" << it->second.conf_data_size
          << " size of copyed_conf_data :" << it->second.copyed_conf_data.size();
    // for debug
    it->second.copyed_bbox_data.resize(it->second.bbox_layer.size);
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE))
          << "stride : " << it->first
          << " size of copyed_bbox_data :" << it->second.copyed_bbox_data.size()
          << " ptr: " << it->second.copyed_bbox_data.data();

  }
  // 3.2 init softmax data buffer 
  softmax_data_.resize(softmax_data_size);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE))
        << "softmax_data_ size:" << softmax_data_.size();


  // 4. create detector
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) << "creating detector...";
  detector_ = vitis::ai::retinaface::create_retinaface_detector(priors_, config);
}

RetinaFaceResult RetinaFacePost::retinaface_post_process_internal(
    unsigned int idx) {
  __TIC__(RetinaFace_after)
  // -1 read bin and transpose
  // -1.1 bbox load 
  // read float value 
  //bool debug_load = false;
  //if (debug_load) {
  //  for(auto it = stride_layers_.begin(); it != stride_layers_.end(); ++it) {
  //    auto stride = it->first; 
  //    auto name = std::string("face_rpn_bbox_pred_stride") + std::to_string(stride) + ".bin";
  //    read_float_bin(name.c_str(), it->second.copyed_bbox_data.data(), 
  //                   it->second.copyed_bbox_data.size());
  //    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //          << "read values:";
  //    for(auto i = 0; i < 10; ++i) {
  //      LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //            << "i:" << i << ", value: " << it->second.copyed_bbox_data[i];
  //    }
  //  }
  //  // -1.2 conf load 
  //  std::vector<std::vector<float>> load_conf_buffers(conf_layer_infos_.size());
  //  auto i = 0;
  //  for(auto it = conf_layer_infos_.begin(); it < conf_layer_infos_.end(); ++it,++i) {
  //    auto &conf_layer = *it;
  //    auto stride = conf_layer.stride;
  //    auto anchor_type = conf_layer.anchor_type; // 0: bg, 1: fg
  //    auto anchor_index = conf_layer.anchor_index; // 0, 1
  //    auto prefix = std::string("face_rpn_cls_score_stride");
  //    auto type_str = std::to_string(stride) + "_anchor" 
  //                  + std::to_string(anchor_index) + "_" 
  //                  + std::string((anchor_type == 0) ? "bg" : "fg") 
  //                  + ".bin";
  //    auto name = prefix + type_str;

  //    load_conf_buffers[i].resize(conf_layer.size);
  //    if (debug_load) {
  //      conf_layer.load_data = load_conf_buffers[i].data();  
  //      read_float_bin(name.c_str(), conf_layer.load_data, conf_layer.size);
  //    } else {
  //    }
  //    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //          << "stride: " << stride << ", read values: ptr: 0x" << std::hex << conf_layer.load_data;
  //    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //          << "size: " << conf_layer.size;
  //    //for(auto i = 0u; i < conf_layer.size; ++i) {
  //    //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //    //        << "i:" << i << ", value: " << conf_layer.load_data[i];
  //    //}
  //  } 
  //}
  // 0. reset bbox layer and landmark layer ptr
  for (auto it = stride_layers_.begin(); it != stride_layers_.end(); ++it) {
    auto &landmark_layer = it->second.landmark_layer; 
    auto &bbox_layer = it->second.bbox_layer; 
    landmark_layer.ptr = (int8_t *)output_tensors_[landmark_layer.output_tensor_index].get_data(idx);
    bbox_layer.ptr = (int8_t *)output_tensors_[bbox_layer.output_tensor_index].get_data(idx);
  }
  
  __TIC__(RETINAFACE_COPY)
  // 1. copy conf data
  auto last_stride = stride_layers_.rbegin()->first;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) << "last stride: " << last_stride;
  auto anchor_type_num = 2; // num_classes
  //for(auto i = 0u; i < conf_layer_infos_.size(); ++i) {
  auto i = 0;
  for(auto it = conf_layer_infos_.begin(); it < conf_layer_infos_.end(); ++it, ++i) {
    auto &conf_layer = *it;
    auto stride = conf_layer.stride;
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
          << "stride = " << stride; 
    if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
      debug_retinaface_output_info(conf_layer);
    }
    auto &copyed_data = stride_layers_[stride].copyed_conf_data;
    auto anchor_type = conf_layer.anchor_type; // 0: bg, 1: fg
    auto anchor_index = conf_layer.anchor_index; // 0, 1
    auto tensor_index = conf_layer.output_tensor_index;
    auto &tensor = output_tensors_[tensor_index];
    // output shape: [batch, height, width, channel]
    // copyed channel : anchor_0_bg, anchor_0_fg, anchor_1_bg, anchor_1_fg
    auto wh = tensor.width * tensor.height;
    auto channel = tensor.channel;
    auto anchor_num = stride_layers_[stride].anchor_info.ratios.size() * stride_layers_[stride].anchor_info.scales.size();
    // e.g tensor       channel_index 
    //     anchor_0_bg  0
    //     anchor_0_fg  1
    //     anchor_1_bg  2
    //     ahchor_1_fg  3
    auto channel_size = anchor_type_num * anchor_num;
    auto channel_index = anchor_index * anchor_num + anchor_type;
    //LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
    //      << "copy values: ptr: 0x" << std::hex << conf_layer.load_data;
    //for (auto j = 0; j < 10; ++j) {
    //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
    //        << "conf_layer.load_data value[" << j << "] :" << *(conf_layer.load_data + j)  
    //        << ", load_conf_buffers[" << i << "][" << j << "] :" << load_conf_buffers[i][j];
    //}
    if ((stride == last_stride && anchor_type == 0) ||  // last stride and bg
        (stride != last_stride && anchor_type == 1)) {  // not last stride and fg
      
      for(auto k = 0u; k < wh; ++k) {
        //if (!debug_load) {
          copyed_data[k * channel_size + channel_index] = 
                    *(std::max_element((int8_t *)tensor.get_data(idx) + k * channel, 
                                        (int8_t *)tensor.get_data(idx) + (k + 1) * channel))
                    * conf_layer.scale;
        //} else {
          // debug
          //copyed_data[k * channel_size + channel_index] = 
          //          *(std::max_element(conf_layer.load_data + k * channel, 
          //                              conf_layer.load_data + (k + 1) * channel));
        //}

        //if (k < 10) {
        //LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        //      << "stride : " << stride
        //      << ((anchor_type == 0) ? " bg " : " fg ")
        //      << "anchor_index : " << (int)anchor_index
        //      << ", channel : " << channel 
        //      << ", channel_size : " << channel_size
        //      << ", channel_index : " << channel_index
        //      << ", k : " << k
        //      << ", index : " << k * channel_size + channel_index
        //      << ", value : " << copyed_data[k * channel_size + channel_index]
        //      << ", src value: " << *(conf_layer.load_data + k * channel)
        //      << " " << *(conf_layer.load_data + k * channel + 1)
        //      << " " << *(conf_layer.load_data + k * channel + 2);
; 
        //}
      }
    } else {
      for(auto k = 0u; k < wh; ++k) {
        //if (!debug_load) {
          copyed_data[k * channel_size + channel_index] 
                  = *((int8_t *)(tensor.get_data(idx)) + k) * conf_layer.scale; 
        //} else {
          // debug
          //copyed_data[k * channel_size + channel_index] 
          //        = *(conf_layer.load_data + k);
        //}
        //if (k < 10) {
        //LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        //      << "stride : " << stride
        //      << ((anchor_type == 0) ? " bg " : " fg ")
        //      << "anchor_index : " << (int)anchor_index
        //      << ", channel : " << channel 
        //      << ", channel_size : " << channel_size
        //      << ", channel_index : " << channel_index
        //      << ", k : " << k
        //      << ", index : " << k * channel_size + channel_index
        //      << ", value : " << copyed_data[k * channel_size + channel_index]
        //      << ", src value: " << *(conf_layer.load_data +k); 
        //}
      }
    }
  }
   __TOC__(RETINAFACE_COPY)

   __TIC__(RETINAFACE_SOFTMAX)
  // 3. softmax
  //std::vector<float> softmax_data_(priors_.size() * num_classes_);
  int32_t offset = 0;
  for(auto it = stride_layers_.begin(); it != stride_layers_.end(); ++it) {
    //vitis::ai::softmax((int8_t *)it->second.data(), 
    //vitis::ai::retinaface::softmax(it->second.copyed_conf_data.data(), 
    //                   1.0,
    //                   anchor_type_num, 
    //                   it->second.copyed_conf_data.size() / anchor_type_num, 
    //                   softmax_data_.data() + offset);
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
          << "stride = " << it->first << ", copyed_conf_data.size: " << it->second.copyed_conf_data.size();
    //if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
    //  for (auto k = 0u; k < it->second.copyed_conf_data.size(); ++k) {
    //     LOG(INFO) << "copyed_conf_data[" << k << "]: " << it->second.copyed_conf_data[offset + k];
    //  }
    //}
          
    auto anchor_num = it->second.anchor_info.ratios.size() * it->second.anchor_info.scales.size();
    vitis::ai::retinaface::my_softmax_2c(it->second.copyed_conf_data.data(), 
                                         it->second.copyed_conf_data.size() / anchor_type_num / anchor_num, 
                                         softmax_data_.data() + offset);
                                         
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
          << "softmax : stride = " << it->first
          << " offset = " << offset
          << " size = " << it->second.copyed_conf_data.size();
    //if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) {
    //  for (auto k = 0u; k < it->second.copyed_conf_data.size(); ++k) {
    //     LOG(INFO) << "softmax value[" << k << "]: " << *(softmax_data_.data() + offset + k);
    //  }
    //}
    offset += it->second.copyed_conf_data.size();
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "softmax_data_ size:" << softmax_data_.size();
  __TOC__(RETINAFACE_SOFTMAX)

  // 4. copy softmax
  __TIC__(RETINAFACE_SOFTMAX_COPY)
  std::vector<float> softmax_data_real(softmax_data_.size()/2);

  for (auto i = 0u; i < softmax_data_.size() / 4; ++i) {
    softmax_data_real[2 * i] = softmax_data_[4 * i + 2];
    softmax_data_real[2 * i + 1] = softmax_data_[4 * i + 3];
  } 

  //for (auto k = 0; k < 10; ++k) {
  //    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //        << "softmax_real_data[" << k << "]: " << *(softmax_data_.data() + k);
  //}

  //for (auto k = 480; k < 480 + 10; ++k) {
  //    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //        << "softmax_real_data[" << k << "]: " << *(softmax_data_.data() + k);
  //}
  //for (auto k = 480 + 1920; k < 480 + 1920 + 10; ++k) {
  //    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //        << "softmax_real_data[" << k << "]: " << *(softmax_data_.data() + k);
  //}
  __TOC__(RETINAFACE_SOFTMAX_COPY)

  // 5. detect 
  __TIC__(RETINAFACE_DETECT)
  std::vector<RetinaFaceResult::BoundingBox> bboxes;
  RetinaFaceResult results{(int)input_tensors_[0].width, (int)input_tensors_[0].height,
                    bboxes};
  detector_->detect(stride_layers_, softmax_data_real.data(), &results);
  __TOC__(RETINAFACE_DETECT)
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
         << "results size:" << results.bboxes.size();
  //std::map<uint32_t, RetinaFaceOutputInfo> bbox_layer_infos;
  //// for (auto i : bbox_layer_indexes_) {
  //for (auto i = 0u; i < conf_layer_infos_.size(); ++i) {
  //  if (conf_layer_infos_[i].type == 2) {  // bbox
  //    bbox_layer_infos.emplace(std::make_pair(i, conf_layer_infos_[i]));
  //  }
  //}
  __TOC__(RetinaFace_after)

  return results;
}


std::vector<vitis::ai::RetinaFaceResult> RetinaFacePost::retinaface_post_process(size_t batch_size) {
  __TIC__(RetinaFace_total_batch)
  //  auto batch_size = input_tensors_[0].batch;
  auto ret = std::vector<vitis::ai::RetinaFaceResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; ++i) {
    ret.emplace_back(retinaface_post_process_internal(i));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
         << "ret size:" << ret.size();
  __TOC__(RetinaFace_total_batch)
  return ret;
}

}  // namespace ai
}  // namespace vitis
