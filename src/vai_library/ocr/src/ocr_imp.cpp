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

#include <fstream>
#include <iostream>
#include <thread>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "./ocr_imp.hpp"

using namespace std;

DEF_ENV_PARAM(ENABLE_OCR_DEBUG, "0");
DEF_ENV_PARAM(XLNX_OCR_PRE_THREAD, "2");
DEF_ENV_PARAM(XLNX_OCR_PRE_ROUND, "1");
DEF_ENV_PARAM(XLNX_OCR_PRE_CVRESIZE, "0");

namespace vitis {
namespace ai {

OCRImp::OCRImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<OCR>(model_name, need_preprocess),
      input_tensors_ (configurable_dpu_task_->getInputTensor()[0]),
      output_tensors_(configurable_dpu_task_->getOutputTensor()[0]),
      cfg_(configurable_dpu_task_->getConfig())
{
  batch_size = (int)get_input_batch();
  target_h8.resize(batch_size);
  target_w8.resize(batch_size);
  ratioh.resize(batch_size);
  ratiow.resize(batch_size);
  oriimg.resize(batch_size);
  resize_oriimg_t.resize(batch_size);

  std::vector<float> vmean(cfg_.kernel(0).mean().begin(), cfg_.kernel(0).mean().end());
  mean.swap(vmean);
  std::vector<float> vscale(cfg_.kernel(0).scale().begin(), cfg_.kernel(0).scale().end());
  scale.swap(vscale);

  scale_i = tensor_scale( input_tensors_[0] ); 
  for(unsigned int i=0; i<scale.size(); i++) {
     scale[i]*=scale_i;
  }

  std::vector<float> vmean_scale{mean[0]*scale[0], mean[1]*scale[1], mean[2]*scale[2] };
  mean_scale.swap(vmean_scale);

  std::string model_namex(model_name);
  if (model_name.size() > 7 && model_name.substr( model_name.size()-7, 7) == ".xmodel") {
     size_t pos = 0;
     if ((pos = model_name.rfind("/")) != std::string::npos) {
        model_namex = model_name.substr(pos+1, model_name.size()-7-(pos+1) );
     } else {
        model_namex = model_name.substr(0, model_name.size()-7);
     }
  }
  std::string cfgpath = std::string(configurable_dpu_task_->get_graph()->get_attr<std::string>("dirname"))
                                 + "/" + model_namex + "_dict.txt";
  post_ = vitis::ai::OCRPost::create( configurable_dpu_task_->getInputTensor()[0], 
            configurable_dpu_task_->getOutputTensor()[0], 
            cfgpath, batch_size, real_batch_size,
            target_h8, target_w8,
            ratioh, ratiow,
            oriimg
          );
  XLNX_OCR_PRE_THREAD   = ENV_PARAM(XLNX_OCR_PRE_THREAD);
  XLNX_OCR_PRE_ROUND    = ENV_PARAM(XLNX_OCR_PRE_ROUND);  
  XLNX_OCR_PRE_CVRESIZE = ENV_PARAM(XLNX_OCR_PRE_CVRESIZE);
}

OCRImp::~OCRImp() {}

void OCRImp::cleanmem() {
  for(int i=0; i<real_batch_size; i++) {
    cleanmem(i);
  }
}

void OCRImp::cleanmem(unsigned int idx) {
  int8_t* p = (int8_t*)input_tensors_[0].get_data(idx);
  memset(p, 0, input_tensors_[0].width * input_tensors_[0].height* input_tensors_[0].channel );
}

void OCRImp::preprocess(const cv::Mat& input_img, int idx) {
  cv::Mat img;

  float ratio =  std::min(( input_tensors_[0].width*1.0)/input_img.cols, (input_tensors_[0].height*1.0)/input_img.rows);
  
  target_h8[idx] = int(input_img.rows*ratio);
  target_w8[idx] = int(input_img.cols*ratio); 

  if (target_h8[idx] %32 ) {
    target_h8[idx] = target_h8[idx] + (32-target_h8[idx] %32);
  }
  if (target_w8[idx] %32 ) {
    target_w8[idx] = target_w8[idx] + (32-target_w8[idx] %32);
  }
 
  ratiow[idx] = (float)input_img.cols/target_w8[idx];
  ratioh[idx] = (float)input_img.rows/target_h8[idx];
  // std::cout << "ratioh[idx]  img   " <<input_img.cols << " " <<  input_img.rows << " target " << target_w8[idx] << " " <<  target_h8[idx] << " " <<  ratiow[idx]  << " " << ratioh[idx]  << "\n"; 

  __TIC__(resize)
  if (cv::Size(target_w8[idx], target_h8[idx]) != input_img.size()) {
    cv::resize(input_img, img, cv::Size(target_w8[idx], target_h8[idx]), 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }

  __TOC__(resize)

  int channels =  input_tensors_[0].channel;
  uint8_t* input = img.data;
  int8_t* dest = (int8_t*)input_tensors_[0].get_data(idx);
  int rows1 = img.rows;
  int cols1 = img.cols;
  int cols =  input_tensors_[0].width;
  int cols1_channels = cols1*channels ;
  int cols_channels  = cols*channels ;

  __TIC__(imgtodpu)

  int size1 = rows1, start=0, len=0;
  std::vector<std::thread> vth;
  for(int i=0; i<XLNX_OCR_PRE_THREAD; i++) {
     start = i * size1/XLNX_OCR_PRE_THREAD;
     len = (i != XLNX_OCR_PRE_THREAD-1) ? size1/XLNX_OCR_PRE_THREAD : (size1- (size1/XLNX_OCR_PRE_THREAD*(XLNX_OCR_PRE_THREAD-1))) ;
     vth.emplace_back( std::thread( &OCRImp::preprocess_thread, this, start, len, input, dest, cols1_channels, cols_channels, cols1, channels));
  }
  for(int i=0; i<XLNX_OCR_PRE_THREAD; i++) {
     vth[i].join();
  }
  
  __TOC__(imgtodpu)

  thread resize_oriimg_t1(&OCRImp::resize_oriimg, this, img, idx);
  resize_oriimg_t[idx] = std::move(resize_oriimg_t1);
}

void OCRImp::resize_oriimg(cv::Mat img, int idx) {
  __TIC__(resize_oriimg)
  if (XLNX_OCR_PRE_CVRESIZE) {
    cv::resize(img, oriimg[idx], cv::Size(target_w8[idx]/4, target_h8[idx]/4), 0, 0, cv::INTER_CUBIC);
  } else {
    oriimg[idx] = cv::Mat(target_h8[idx]/4, target_w8[idx]/4,  CV_8UC3 );
    for(int i=0; i<target_h8[idx]/4; i++) {
      for(int j=0; j<target_w8[idx]/4; j++) {
         oriimg[idx].ptr<cv::Vec3b>(i)[j]  = img.ptr<cv::Vec3b>(i*4)[j*4];
      }
    }
  }
  __TOC__(resize_oriimg)
}

void OCRImp::preprocess_thread(int start, int len, uint8_t* input, int8_t* dest, int cols1_channels, int cols_channels, int cols1, int channels) {
  if (XLNX_OCR_PRE_ROUND==1) {
    for (auto h = start; h < start+len; ++h) {
      for (auto w = 0; w < cols1; ++w) {
        dest[h * cols_channels + w * channels + 2] = int(round(input[h * cols1_channels + w * channels + 0] * scale[0] - mean_scale[0]));
        dest[h * cols_channels + w * channels + 1] = int(round(input[h * cols1_channels + w * channels + 1] * scale[1] - mean_scale[1]));
        dest[h * cols_channels + w * channels + 0] = int(round(input[h * cols1_channels + w * channels + 2] * scale[2] - mean_scale[2]));
      } 
    }
  } else {
    for (auto h = start; h < start+len; ++h) {
      for (auto w = 0; w < cols1; ++w) {
        dest[h * cols_channels + w * channels + 2] = int(input[h * cols1_channels + w * channels + 0] * scale[0] - mean_scale[0]);
        dest[h * cols_channels + w * channels + 1] = int(input[h * cols1_channels + w * channels + 1] * scale[1] - mean_scale[1]);
        dest[h * cols_channels + w * channels + 0] = int(input[h * cols1_channels + w * channels + 2] * scale[2] - mean_scale[2]);
      }
    }
  }
}

OCRResult OCRImp::run( const cv::Mat &input_img) {
  if (ENV_PARAM(ENABLE_OCR_DEBUG) == 1) {
    {
      void* addr2 = configurable_dpu_task_->getInputTensor()[0][0].get_data(0);   (void)addr2; // printf("add-0-in: %p   ", addr2);
      std::vector<vitis::ai::library::InputTensor> inputs = configurable_dpu_task_->getInputTensor()[0];
      const auto& layer_data = inputs[0];
      int sWidth = layer_data.width;
      int sHeight= layer_data.height;
      auto channels = layer_data.channel;
      auto size = layer_data.size;
      float scale =  tensor_scale(layer_data);
      std::cout <<"net0in: sWidth heiht channel  scale :  " << sWidth << "  " << sHeight << "  " <<  channels << "  " << size << "  " << scale << "\n";  
      // 960  960  3  32
    }
    {
      for(int i=0; i<2; i++) {
        void* addr2 = configurable_dpu_task_->getOutputTensor()[0][0].get_data(0);   (void)addr2;  //  printf("add-0-out: %p   ", addr2);
        std::vector<vitis::ai::library::OutputTensor> outputs = configurable_dpu_task_->getOutputTensor()[0];
        const auto& layer_datao = outputs[i];
        int sWidth = layer_datao.width;
        int sHeight= layer_datao.height;
        auto channels = layer_datao.channel;
        auto size = layer_datao.size;
        float scale =  tensor_scale(layer_datao);
        std::cout <<"net0out: sWidth heiht channel  scale :  " << sWidth << "  " << sHeight << "  " <<  channels << "  " << size << "  " << scale << "\n"; 
      }
      // 480  480  37  0.25  (y2:  need from 480x480-->240x240)
      // 240  240  2  0.25     (y):  need sigmoid
      //   new :
      // net0in: sWidth heiht channel  scale :  960  960  3  2764800  32
      // net0out: sWidth heiht channel  scale :  240  240  2  115200  0.25
      // net0out: sWidth heiht channel  scale :  480  480  37  8524800  0.25
      
    }
  }
  __TIC__(OCR_total)
  __TIC__(OCR_setimg)

  real_batch_size = 1;
  cleanmem(0);
  preprocess(input_img, 0);

  __TOC__(OCR_setimg)
  __TIC__(OCR_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(OCR_dpu)

  resize_oriimg_t[0].join();
  __TIC__(OCR_post)
  auto result = post_->process(0);
  __TOC__(OCR_post)

  __TOC__(OCR_total)
  return result;
}

std::vector<OCRResult> OCRImp::run( const std::vector<cv::Mat> &input_img) {
  __TIC__(OCR_total)
  __TIC__(OCR_setimg)

  real_batch_size = std::min(int(input_img.size()), int(batch_size));
  cleanmem();
  for (auto i = 0; i < real_batch_size; i++) {
    preprocess(input_img[i], i);
  }

  __TOC__(OCR_setimg)
  __TIC__(OCR_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(OCR_dpu)

  for (auto i = 0; i < real_batch_size; i++) {
    resize_oriimg_t[i].join();
  }

  __TIC__(OCR_post)
  auto results = post_->process();
  __TOC__(OCR_post)

  __TOC__(OCR_total)
  return results;
}

}  // namespace ai
}  // namespace vitis

