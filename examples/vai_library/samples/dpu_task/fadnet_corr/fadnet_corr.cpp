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
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vitis/ai/library/tensor.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <math.h>
#include <utility>

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>

#include "fadnet_corr.hpp"

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

DEF_ENV_PARAM(DEBUG_FADNET, "0");
DEF_ENV_PARAM(DUMP_FADNET, "0");

using namespace std;
using namespace cv;

namespace vitis {
namespace ai {
class FadNetImp : public FadNet {
 public:
  FadNetImp(const std::string& model_name);

 public:
  virtual ~FadNetImp();
  virtual std::vector<cv::Mat> run(
      const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) override;
  virtual size_t get_input_batch() override;
  virtual int get_input_width() const override;
  virtual int get_input_height() const override;

 private:
  std::vector<cv::Mat> FADnet_run(
      const vector<pair<cv::Mat, cv::Mat>>& input_images);

 public:
  std::vector<std::unique_ptr<vitis::ai::DpuTask>> tasks_;
  std::vector<float> means_;
  std::vector<float> lscales_;
  std::vector<float> rscales_;
};

FadNet::FadNet(const std::string& model_name) {}

FadNet::~FadNet() {}

std::vector<cv::Mat> FadNetImp::run(
		const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) {
  return FADnet_run(imgs);
}

std::unique_ptr<FadNet> FadNet::create(const std::string& model_name) {
  return std::unique_ptr<FadNetImp>(new FadNetImp(model_name));
}

FadNetImp::FadNetImp(const std::string& model_name) : FadNet(model_name) {
  tasks_.emplace_back(vitis::ai::DpuTask::create(model_name));
  means_ = {103.53, 116.28, 123.675};
  vector<float> scale_ = {0.017429, 0.017507, 0.01712475};
  float scale0 = tensor_scale(tasks_[0]->getInputTensor(0u)[0]);
  float scale1 = tensor_scale(tasks_[0]->getInputTensor(0u)[1]);
  lscales_ = {scale_[0] * scale0, scale_[1] * scale0, scale_[2] * scale0};
  rscales_ = {scale_[0] * scale1, scale_[1] * scale1, scale_[2] * scale1};
  tasks_[0]->setMeanScaleBGR(means_, scale_);
}

FadNetImp::~FadNetImp() {}

void setImageRGB2(const Mat& img, int8_t* data, vector<float>& mean_, vector<float>& scale) {
  auto rows = img.rows;
  auto cols = img.cols;
  auto channels = img.channels();
  vitis::ai::NormalizeInputDataRGB(img.data, rows, cols, channels, img.step,
                        mean_, scale, data);
}

// run the fadnet
vector<Mat> FadNetImp::FADnet_run(
      const vector<pair<cv::Mat, cv::Mat>>& input_images) {
  vector<cv::Mat> left_mats;
  vector<cv::Mat> right_mats;
  auto input_tensor_left = tasks_[0]->getInputTensor(0u)[0];
  auto input_tensor_right = tasks_[0]->getInputTensor(0u)[1];
  int sWidth = input_tensor_left.width;
  int sHeight = input_tensor_left.height;

  __TIC__(FADNET_RESIZE)
  for(size_t i = 0; i < input_tensor_left.batch; ++i) {
    if(sWidth == input_images[i].first.cols && sHeight == input_images[i].first.rows) {
      left_mats.push_back(input_images[i].first);
    } else {
      cv::Mat left_mat;
      resize(input_images[i].first, left_mat, cv::Size(sWidth, sHeight));
      left_mats.push_back(left_mat);
    }
    if(sWidth == input_images[i].second.cols && sHeight == input_images[i].second.rows) {
      right_mats.push_back(input_images[i].second);
    } else {
      cv::Mat right_mat;
      resize(input_images[i].second, right_mat, cv::Size(sWidth, sHeight));
      right_mats.push_back(right_mat);
    }
  }
  __TOC__(FADNET_RESIZE)

  // ### kernel 0 part ###
  __TIC__(FADNET_SET_IMG_LEFT)
  tasks_[0]->setImageRGB(left_mats);
  __TOC__(FADNET_SET_IMG_LEFT)

  __TIC__(FADNET_SET_IMG_RIGHT)
#ifndef ENABLE_NEON
  for(size_t i = 0; i < input_tensor_left.batch; ++i) {
    int8_t* input_right_ptr = (int8_t*)input_tensor_right.get_data(i);
    for(int h = 0; h < sHeight; h++) {
      for(int w = 0; w < sWidth; w++) {
        int pos = 3*(h*sWidth+w);
        input_right_ptr[pos+2] = (int8_t)((right_mats[i].at<Vec3b>(h,w)[0] - means_[0])*lscales_[0]);
        input_right_ptr[pos+1] = (int8_t)((right_mats[i].at<Vec3b>(h,w)[1] - means_[1])*lscales_[1]);
        input_right_ptr[pos+0] = (int8_t)((right_mats[i].at<Vec3b>(h,w)[2] - means_[2])*lscales_[2]);
      }
    }
  }
#else
  for(size_t i = 0; i < input_tensor_left.batch; ++i) {
    int8_t* input_right_ptr = (int8_t*)input_tensor_right.get_data(i);
    setImageRGB2(right_mats[i], input_right_ptr, means_, rscales_);
  }
#endif
  __TOC__(FADNET_SET_IMG_RIGHT)
  //exit(0);
  
  __TIC__(FADNET_DPU_LAST)
  tasks_[0]->run(0u);
  __TOC__(FADNET_DPU_LAST)

  __TIC__(FADNET_POST_ARM)
  vector<Mat> rets;
  int ret_height = input_images[0].first.rows;
  int ret_width = input_images[0].first.cols;
  auto final_tensor = tasks_[0]->getOutputTensor(0u)[0];

  if(ENV_PARAM(DUMP_FADNET)) {
    for (size_t b = 0; b < final_tensor.batch; ++b) {
      std::ofstream ofs("tensor_res_" + to_string(b) + ".bin", ios::binary);
      ofs.write((char*)final_tensor.get_data(b), final_tensor.width * final_tensor.height);
      ofs.close();
    }
  }

  float f_scale = vitis::ai::library::tensor_scale(final_tensor);
  for (size_t b = 0; b < final_tensor.batch; ++b) {
    Mat final_img(final_tensor.height, final_tensor.width, CV_8UC1);
    Mat ret;
    auto final_data = (int8_t*)final_tensor.get_data(b);
    if (f_scale == 1.f) {
      final_img = Mat(Size(final_tensor.width, final_tensor.height), CV_8UC1, (void*)final_data);
    } else {
      for(size_t i = 0; i < final_tensor.width * final_tensor.height; ++i)
        final_img.data[i] = (uint8_t)(final_data[i] * f_scale);
    }
    resize(final_img, ret, cv::Size(ret_width, ret_height));
    rets.push_back(ret);
  }
  __TOC__(FADNET_POST_ARM)
  return rets;
}

size_t FadNetImp::get_input_batch() { return tasks_[0]->get_input_batch(0, 0); }

int FadNetImp::get_input_width() const {
  return tasks_[0]->getInputTensor(0u)[0].width;
}
int FadNetImp::get_input_height() const {
  return tasks_[0]->getInputTensor(0u)[0].height;
}

}
}
