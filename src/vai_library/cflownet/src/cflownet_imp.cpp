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
#include "./cflownet_imp.hpp"
#include <sys/stat.h>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <random>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/nnpp/apply_nms.hpp>
#include <vitis/ai/math.hpp>

using namespace std;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_CFLOW_DEBUG, "0");
DEF_ENV_PARAM(ENABLE_CFLOW_CMPDATA, "0");
DEF_ENV_PARAM(ENABLE_CFLOW_ROUND, "1");

inline float sigmoid(float x) { return 1.0/(1.0+exp(-x) ); }

CflownetImp::CflownetImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Cflownet>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0])
{
  batch_size = get_input_batch();
  iData0.resize(batch_size);
  iData1.resize(batch_size);
  oData.resize(batch_size);

  for(int j=0; j<batch_size; j++) {
      oData[j] = (int8_t*)output_tensors_[0].get_data(j);
      iData0[j] = (int8_t*)input_tensors_[0].get_data(j);
      iData1[j] = (int8_t*)input_tensors_[1].get_data(j);
  }
  sWidth   = output_tensors_[0].width;
  sHeight  = output_tensors_[0].height;
  sChannel = output_tensors_[0].channel;
  sScaleo  = tensor_scale(output_tensors_[0]);
  sScalei0  = tensor_scale(input_tensors_[0]);
  sScalei1  = tensor_scale(input_tensors_[1]);
  if ( ENV_PARAM(ENABLE_CFLOW_DEBUG)) {
    // o-whc : 128 128 1 0.0625
    // i0-whc : 128 128 1  64
    // i1-whc : 1 6 1   32
    std::cout <<"o-whc : " << sWidth << " " << sHeight << " " << sChannel << " " << sScaleo <<"\n";
    std::cout <<"i0-whc : " << input_tensors_[0].width << " " << input_tensors_[0].height << " " << input_tensors_[0].channel << "  " << tensor_scale(input_tensors_[0])  << "\n";
    std::cout <<"i1-whc : " << input_tensors_[1].width << " " << input_tensors_[1].height << " " << input_tensors_[1].channel  << "  " << tensor_scale(input_tensors_[1]) <<"\n";
  }

  std::random_device rd{};
  std::mt19937 gen0{rd()}; 
  gen = gen0;
}

CflownetImp::~CflownetImp() {}

std::vector<CflownetResult> CflownetImp::cflownet_post_process() {
  auto ret = std::vector<vitis::ai::CflownetResult>{};
  ret.reserve(real_batch_size);
  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(cflownet_post_process(i));
  }
  return ret;
}

void CflownetImp::cflownet_pre_process(int idx, const float* in_p) {
  if (ENV_PARAM(ENABLE_CFLOW_CMPDATA)) {
    // 04fde9180ce0 
    unsigned char src[]={0x04, 0xfd, 0xe9, 0x18, 0x0c, 0xe0};
    for(int i=0;i<6;i++) iData1[idx][i] = (int8_t)src[i];
  } else {
    std::normal_distribution<> d{0.0,1.0};
    for(int i=0; i<6; i++) {
       iData1[idx][i] = std::clamp(int(std::round(d(gen)*sScalei1)), -128, 127);  
    }
  }
  auto total = sWidth*sHeight;
  if (ENV_PARAM(ENABLE_CFLOW_ROUND)) {
    for(int i=0; i<total; i++) iData0[idx][i] = round(in_p[i]*sScalei0); 
  } else  {
    for(int i=0; i<total; i++) iData0[idx][i] = int8_t(in_p[i]*sScalei0); 
  }
}

CflownetResult CflownetImp::cflownet_post_process(int idx) {
  CflownetResult  ret{int(input_tensors_[0].width), int(input_tensors_[0].height), std::vector<float>(sWidth*sHeight)};
  for(int i=0; i<sWidth*sHeight; i++) {
     ret.data[i] = sigmoid(oData[idx][i]*sScaleo);  
  }
  return ret;
}

CflownetResult CflownetImp::run( const float* in_p) {
  __TIC__(Cflownet_total)
  __TIC__(Cflownet_setimg)
  real_batch_size = 1;
  cflownet_pre_process(0, in_p);
  __TOC__(Cflownet_setimg)
  __TIC__(Cflownet_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Cflownet_dpu)

  __TIC__(Cflownet_post)
  auto result = cflownet_post_process(0);
  __TOC__(Cflownet_post)

  __TOC__(Cflownet_total)
  return result;
}

std::vector<CflownetResult> CflownetImp::run( const std::vector<const float*> in_ps) {
  __TIC__(Cflownet_total)
  __TIC__(Cflownet_setimg)
  real_batch_size = std::min(int(in_ps.size()), int(batch_size));

  for(int i=0; i<real_batch_size; i++) {
    cflownet_pre_process(i, in_ps[i]);
  }
  __TOC__(Cflownet_setimg)
  __TIC__(Cflownet_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(Cflownet_dpu)

  __TIC__(Cflownet_post)
  auto results = cflownet_post_process();
  __TOC__(Cflownet_post)

  __TOC__(Cflownet_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
