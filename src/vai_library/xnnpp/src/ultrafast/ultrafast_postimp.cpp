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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ultrafast_postimp.hpp"

#define ARR_WIDTHxCHANNEL_LEN 72
using namespace std;

namespace vitis {
namespace ai {

UltraFastPostImp::UltraFastPostImp(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config,
    int batch_sizex,
    int& real_batch_sizex,
    std::vector<cv::Size>& pic_sizex)
    : input_tensors_(input_tensors), 
      output_tensors_(output_tensors), 
      batch_size(batch_sizex),
      real_batch_size(real_batch_sizex),
      pic_size(pic_sizex)
{
    o_cls.resize(batch_size);
    const auto& layer_datai = input_tensors_[0];
    i_height= layer_datai.height;
    i_width = layer_datai.width;

    for(int i=0; i< batch_size; i++) {
       o_cls[i] =  (int8_t*)(output_tensors_[0].get_data(i));
    }
    o_height= output_tensors_[0].height;
    o_width = output_tensors_[0].width;
    o_scale =  tensor_scale(output_tensors_[0]);
    o_channel = output_tensors_[0].channel;

    griding_num = o_height -1;
    softmax_data.resize( (o_height-1)*o_width*o_channel);
    std::vector<float> tmpv{121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287};
    tmpv.swap(row_anchor_orig);
    for(unsigned int i=0; i<row_anchor_orig.size(); i++) row_anchor_orig[i]/=i_height;
    row_anchor.resize(row_anchor_orig.size(), 0.0);
    col_sample_w = (float(i_width-1))/(griding_num-1);  
}

UltraFastPostImp::~UltraFastPostImp(){}

static void softmax_c_with_sum(int8_t* input, int wc, int h_cls, float scale, float* output, float* loc_sum) 
{
  for(int k=0; k<wc; k++) {
    float sum = 0.f;
    for (int i = 0; i < h_cls-1; ++i) {
      output[k*(h_cls-1)+i] = exp(input[i*wc+k]*scale);
      sum += output[k*(h_cls-1)+i];
    }
    for (int i = 0; i < h_cls-1; ++i) { 
       output[k*(h_cls-1)+i] = output[k*(h_cls-1)+i]/sum ;
       loc_sum[k] += output[k*(h_cls-1)+i]*(i+1);
    }
  }
}

UltraFastResult UltraFastPostImp::post_process(unsigned int idx)
{
   UltraFastResult ret;
   ret.width =  i_width;
   ret.height = i_height;

   std::vector<float> loc(o_width*o_channel, 0);
   softmax_c_with_sum(o_cls[idx], o_width*o_channel, o_height, o_scale, softmax_data.data(), loc.data());

   for(unsigned int i=0; i<row_anchor.size(); i++) {
     row_anchor[i] = row_anchor_orig[i] * pic_size[idx].height;
   }

   using arr_int8=int8_t[ ARR_WIDTHxCHANNEL_LEN ];
   arr_int8* wcdata = (arr_int8*)o_cls[idx];
   std::vector<float> npsum(o_channel, 0.0);
   for(int i=0; i<o_width*o_channel; i++) {
      auto cmp = [i]( const arr_int8& in1,  const arr_int8& in2)->bool { return in1[i] < in2[i]; };
      int pos = std::max_element( wcdata, wcdata+o_height , cmp) - wcdata;
      if (pos == griding_num) {
          loc[i] = 0.0;
      } else {
          loc[i] = loc[i]*pic_size[idx].width/i_width*col_sample_w;  
          npsum[i%o_channel] += loc[i]; 
      }
   }
   for(int i=0; i<o_channel; i++) {
     std::vector<std::pair<float, float>> lane; 
     if ( npsum[i] >2 ) { 
        for(unsigned int j=0; j<row_anchor.size(); j++) {
           if (loc[j*o_channel+i] > 0) {
              lane.emplace_back( std::make_pair( loc[j*o_channel+i]-1, row_anchor[j] ));
           } else {
              lane.emplace_back( std::make_pair(-2, row_anchor[j] ));
           }
        }
     } else {
        for(unsigned int j=0; j<row_anchor.size(); j++) {
          lane.emplace_back( std::make_pair(-2, row_anchor[j] ));
        } 
     }
     ret.lanes.emplace_back(lane);
   }   
   return ret;
}

std::vector<UltraFastResult> UltraFastPostImp::post_process()
{
  auto ret = std::vector<vitis::ai::UltraFastResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(post_process(i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
