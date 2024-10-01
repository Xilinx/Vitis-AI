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
#include "./unet2d_imp.hpp"

#include <numeric>
#include <fstream>
#include <iostream>
#include<cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include <sys/stat.h>

namespace vitis {
namespace ai {

Unet2DImp::Unet2DImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Unet2D>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{
  batch_size = get_input_batch();
  scale_o = tensor_scale( output_tensors_[0] );
  scale_i = tensor_scale(  input_tensors_[0] );
}

Unet2DImp::~Unet2DImp() {}

std::vector<Unet2DResult> Unet2DImp::unet2d_post_process() {
  auto ret = std::vector<vitis::ai::Unet2DResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(unet2d_post_process(i));
  }
  return ret;
}

Unet2DResult Unet2DImp::unet2d_post_process(int idx) {
  int8_t* p = (int8_t*)output_tensors_[0].get_data(idx);
  Unet2DResult ret{int(input_tensors_[0].width), 
                   int(input_tensors_[0].height) , 
                   std::vector<float>(output_tensors_[0].height*output_tensors_[0].width)};

// 144 144 1 
// std::cout <<" size : " << output_tensors_[0].height << " " << output_tensors_[0].width << " " << output_tensors_[0].channel << "\n";

  int pos = 0;
  for(int h=0; h<(int)output_tensors_[0].height; ++h ){
    for(int w=0; w<(int)output_tensors_[0].width; ++w ){
      for(int c=0; c<(int)output_tensors_[0].channel; ++c ){
         ret.data[pos] = p[pos]*scale_o; 
         pos++;
      }
    }
  }  
  return ret;
}

void Unet2DImp::preprocess(float* input_img, int len, int idx) {
 // 224 144
 // Testing image dimensions: (7130, 144, 144, 4)
 // Testing mask dimensions:  (7130, 144, 144, 1)
 // mean 0; scale 1 : even don't care
 // INPUT CHANNELS:  "modality": {   // LABEL_CHANNELS: "labels": {
 //      "0": "FLAIR",               //      "0": "background",
 //      "1": "T1w",                 //      "1": "edema",
 //      "2": "t1gd",                //      "2": "non-enhancing tumor", 
 //      "3": "T2w" },               //      "3": "enhancing tumour"  }
  // img_temp[idx] = img[idx, x:(x + dx), y:(y + dy), :]

  __TIC__(pre)
  int8_t* dest = (int8_t*)input_tensors_[0].get_data(idx);
  int edge = (int)sqrt(len/input_tensors_[0].channel);
  int dx = (edge - input_tensors_[0].width)/2; 
  for(int h=0; h<(int)input_tensors_[0].height; ++h ){
    for(int w=0; w<(int)input_tensors_[0].width; ++w ){
      float* src_base = input_img + (edge*(dx+h) + dx+w)*input_tensors_[0].channel;
      int c_base = h*input_tensors_[0].width*input_tensors_[0].channel + w*input_tensors_[0].channel;
      for(int c=0; c<(int)input_tensors_[0].channel; ++c ){
         dest[ c_base + c] = (*(src_base+c)) * scale_i;
      }
    }
  }
  __TOC__(pre)
}

Unet2DResult Unet2DImp::run( float* input_img, int len) {
  return this->run(std::vector<float*>{input_img}, len)[0];
}

Unet2DResult Unet2DImp::run(const std::vector<float>& input_img) {
  return this->run(std::vector<float*>{(float*)input_img.data()}, input_img.size())[0];
}

std::vector<Unet2DResult> Unet2DImp::run( const std::vector<std::vector<float>> &input_img) {
  int len = input_img[0].size();
  std::vector<float*> img;
  for(int i=0; i<(int)input_img.size(); ++i) {
     img.emplace_back((float*)input_img[i].data());
  } 
  return this->run(img, len);
}

std::vector<Unet2DResult> Unet2DImp::run( const std::vector<float*> &input_img, int len) {

  __TIC__(Unet2D_total)
  __TIC__(Unet2D_setimg)

  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  for (auto i = 0; i < real_batch_size; i++) {
     preprocess(input_img[i], len, i);
  }
  __TOC__(Unet2D_setimg)
  __TIC__(Unet2D_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Unet2D_dpu)

  __TIC__(Unet2D_post)
  auto results = unet2d_post_process();
  __TOC__(Unet2D_post)

  __TOC__(Unet2D_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
