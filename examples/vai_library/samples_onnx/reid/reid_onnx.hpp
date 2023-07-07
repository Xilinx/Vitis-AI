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

#pragma once
#include <assert.h>
#include <glog/logging.h>

#include <opencv2/imgproc/imgproc_c.h>
#include <algorithm>  // std::generate
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

std::vector<float> bn_means{
#include "means.inc"
};
std::vector<float> bn_vars{
#include "vars.inc"
};
std::vector<float> bn_weights{
#include "weights.inc"
};

// return value
struct OnnxReidResult {
  int width;
  int height;
  cv::Mat feat;
};

// model class
class OnnxReid : public OnnxTask {
 public:
  static std::unique_ptr<OnnxReid> create(const std::string& model_name) {
    return std::unique_ptr<OnnxReid>(new OnnxReid(model_name));
  }

 protected:
  explicit OnnxReid(const std::string& model_name);
  OnnxReid(const OnnxReid&) = delete;

 public:
  virtual ~OnnxReid() {}
  virtual std::vector<OnnxReidResult> run(const std::vector<cv::Mat>& mats);

 private:
  std::vector<OnnxReidResult> postprocess();
  OnnxReidResult postprocess(int idx);
  void preprocess(const cv::Mat& image, int idx);
  void preprocess(const std::vector<cv::Mat>& mats);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
  std::vector<float*> output_tensor_ptr;
};

//(image_data - mean) * scale, BRG2RGB and hwc2chw
static void set_input_image(const cv::Mat& image, float* data) {
  float mean[3] = {103.53f, 116.28f, 123.675f};
  float scales[3] = {0.017429f, 0.017507f, 0.01712475f};
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto c_t = abs(c - 2);  // BRG to RGB
        auto image_data =
            (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scales[c_t];
        data[c * image.rows * image.cols + h * image.cols + w] =
            (float)image_data;
      }
    }
  }
}

void OnnxReid::preprocess(const cv::Mat& image, int idx) {
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(getInputWidth(), getInputHeight()));
  set_input_image(resized_image, input_tensor_values.data() + batch_size * idx);
  return;
}

// preprocess
void OnnxReid::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index);
  }
}

float bn(float input, float weight, float mean, float var) {
  return ((input - mean) / sqrt(var + 1e-5)) * weight;
}

// postprocess
OnnxReidResult OnnxReid::postprocess(int idx) {
  int channels = output_shapes_[0][1];  // 2048
  float a[channels];

  float* fp = output_tensor_ptr[0] + idx * channels;
  for (int c = 0; c < channels; c++) {
    a[c] = fp[c];
    a[c] = bn(a[c], bn_weights[c], bn_means[c], bn_vars[c]);
  }

  Mat x = Mat(1, channels, CV_32F, a);
  Mat feat;
  normalize(x, feat);
  OnnxReidResult result{1, 1, feat};

  return result;
}

std::vector<OnnxReidResult> OnnxReid::postprocess() {
  std::vector<OnnxReidResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess(index));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

OnnxReid::OnnxReid(const std::string& model_name) : OnnxTask(model_name) {
  auto input_shape = input_shapes_[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  auto channel = input_shapes_[0][1];
  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];
  batch_size = channel * height * width;

  output_tensor_ptr.resize(1);
}

std::vector<OnnxReidResult> OnnxReid::run(const std::vector<cv::Mat>& mats) {
  __TIC__(total)
  __TIC__(preprocess)
  preprocess(mats);

  if (input_tensors.size()) {
    input_tensors[0] = Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]);
  } else {
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]));
  }
  __TOC__(preprocess)

  __TIC__(session_run)
  run_task(input_tensors, output_tensors);
  output_tensor_ptr[0] = output_tensors[0].GetTensorMutableData<float>();
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<OnnxReidResult> ret = postprocess();
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}

