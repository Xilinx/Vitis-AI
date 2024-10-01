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

// return value
struct OnnxMovenetResult {
  int width;
  int height;
  std::vector<cv::Point2f> poses;
};

std::vector<float> center_weight{
#include "center_weight.inc"
};

// model class
class OnnxMovenet : public OnnxTask {
 public:
  static std::unique_ptr<OnnxMovenet> create(const std::string& model_name) {
    return std::unique_ptr<OnnxMovenet>(new OnnxMovenet(model_name));
  }

 protected:
  explicit OnnxMovenet(const std::string& model_name);
  OnnxMovenet(const OnnxMovenet&) = delete;

 public:
  virtual ~OnnxMovenet() {}
  virtual std::vector<OnnxMovenetResult> run(const std::vector<cv::Mat>& mats);

 private:
  std::vector<OnnxMovenetResult> postprocess(const std::vector<cv::Mat>& mats);
  OnnxMovenetResult postprocess(const cv::Mat& mat, int idx);
  void preprocess(const cv::Mat& image, int idx);
  void preprocess(const std::vector<cv::Mat>& mats);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  int real_batch;
  int batch_size;
  std::vector<float*> output_tensor_ptr;
  float conf_threshold = 0.1;
};

void OnnxMovenet::preprocess(const cv::Mat& image, int idx) {
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(getInputWidth(), getInputHeight()));
  set_input_image_rgb(resized_image, input_tensor_values.data() + batch_size * idx,
                  std::vector<float>{127.5f, 127.5f, 127.5f},
                  std::vector<float>{0.00784314f, 0.00784314f, 0.00784314f}
                 );
  return;
}

// preprocess
void OnnxMovenet::preprocess(const std::vector<cv::Mat>& mats) {
  real_batch = std::min((int)input_shapes_[0][0], (int)mats.size());
  for (auto index = 0; index < real_batch; ++index) {
    preprocess(mats[index], index);
  }
}

static vector<int> getMaxPoint(float* center, const vector<float>& weights,
                               int width = 48) {
  int cx = -1, cy = -1;
  float max = -100.0f;
  for (auto i = 0u; i < weights.size(); ++i) {
    float val = center[i] * weights[i];
    if (val > max) {
      cy = i / width;
      cx = i % width;
      max = val;
    }
  }
  return vector<int>{cx, cy};
}
static vector<int> getMaxPoint(const vector<float>& center, int width = 48) {
  int cx = -1, cy = -1;
  float max = -100.0f;
  for (auto i = 0u; i < center.size(); ++i) {
    float val = center[i];
    if (val > max) {
      cy = i / width;
      cx = i % width;
      max = val;
    }
  }
  // std::cout<<"getmax: "<< cx<<" "<< cy<<"\n";
  return vector<int>{cx, cy};
}

// postprocess
OnnxMovenetResult OnnxMovenet::postprocess(const cv::Mat& mat, int idx0) {
  // Input Node Name/Shape (1):
  //         blob.1 : 1x3x192x192
  // Output Node Name/Shape (4):
  //         1548 : 1x17x48x48
  //         1607 : 1x1x48x48
  //         1665 : 1x34x48x48
  //         1723 : 1x34x48x48

  float* data_heatmap = output_tensor_ptr[0] + idx0 * output_shapes_[0][1] *
                                                   output_shapes_[0][2] *
                                                   output_shapes_[0][3];
  float* data_center = output_tensor_ptr[1] + idx0 * output_shapes_[1][1] *
                                                  output_shapes_[1][2] *
                                                  output_shapes_[1][3];
  float* data_reg = output_tensor_ptr[2] + idx0 * output_shapes_[2][1] *
                                               output_shapes_[2][2] *
                                               output_shapes_[2][3];
  float* data_offset = output_tensor_ptr[3] + idx0 * output_shapes_[3][1] *
                                                  output_shapes_[3][2] *
                                                  output_shapes_[3][3];

  int channel = output_shapes_[0][1];
  int width = output_shapes_[0][2];
  int height = output_shapes_[0][3];
  auto size = width * height;

  float* float_data_heatmap = data_heatmap;
  float* float_data_center = data_center;
  ;
  float* float_data_reg = data_reg;
  float* float_data_offset = data_offset;

  std::vector<cv::Point2f> poses;
  auto maxPoint = getMaxPoint(float_data_center, center_weight);
  for (auto i = 0; i < channel; ++i) {
    int start = size * i;
    int start_x = size * 2 * i;
    int start_y = size * 2 * i + size;
    auto reg_x_ori =
        (float_data_reg[start_x + width * maxPoint[1] + maxPoint[0]] + 0.5);
    auto reg_y_ori =
        (float_data_reg[start_y + width * maxPoint[1] + maxPoint[0]] + 0.5);
    auto reg_x = reg_x_ori + maxPoint[0];
    auto reg_y = reg_y_ori + maxPoint[1];
    vector<int> map_reg_x(width);
    vector<int> map_reg_y(width);
    for (auto iw = 0; iw < width; ++iw) {
      map_reg_x[iw] = (iw - reg_x) * (iw - reg_x);
      map_reg_y[iw] = (iw - reg_y) * (iw - reg_y);
    }
    vector<float> float_reg_map(width * width);
    for (auto iy = 0; iy < width; ++iy) {
      for (auto ix = 0; ix < width; ++ix) {
        auto val = sqrt(map_reg_x[ix] + map_reg_y[iy]) + 1.8;
        float_reg_map[iy * width + ix] = val;
      }
    }
    vector<float> float_tem_reg(width * width);
    for (auto idx = 0; idx < size; ++idx) {
      float_tem_reg[idx] = float_data_heatmap[start + idx] / float_reg_map[idx];
    }
    auto reg_max_point = getMaxPoint(float_tem_reg);
    auto score =
        float_data_heatmap[start + reg_max_point[1] * width + reg_max_point[0]];
    auto offset_x = float_data_offset[start_x + reg_max_point[1] * width +
                                      reg_max_point[0]];
    auto offset_y = float_data_offset[start_y + reg_max_point[1] * width +
                                      reg_max_point[0]];
    auto ret_x = (reg_max_point[0] + offset_x) / width;
    auto ret_y = (reg_max_point[1] + offset_y) / width;
    if (score < conf_threshold) {
      ret_x = -1;
      ret_y = -1;
    }
    poses.push_back(Point2f(ret_x * mat.cols, ret_y * mat.rows));
  }

  OnnxMovenetResult result{(int)getInputWidth(), (int)getInputHeight(), poses};
  return result;
}

std::vector<OnnxMovenetResult> OnnxMovenet::postprocess(
    const std::vector<cv::Mat>& mats) {
  std::vector<OnnxMovenetResult> ret;
  for (auto index = 0; index < (int)real_batch; ++index) {
    ret.emplace_back(postprocess(mats[index], index));
  }
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

OnnxMovenet::OnnxMovenet(const std::string& model_name) : OnnxTask(model_name) {
  auto input_shape = input_shapes_[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  auto channel = input_shapes_[0][1];
  auto height = input_shapes_[0][2];
  auto width = input_shapes_[0][3];
  batch_size = channel * height * width;

  output_tensor_ptr.resize(4);
}

std::vector<OnnxMovenetResult> OnnxMovenet::run(
    const std::vector<cv::Mat>& mats) {
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
  for (int i = 0; i < 4; i++) {
    output_tensor_ptr[i] = output_tensors[i].GetTensorMutableData<float>();
  }
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<OnnxMovenetResult> ret = postprocess(mats);
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}

