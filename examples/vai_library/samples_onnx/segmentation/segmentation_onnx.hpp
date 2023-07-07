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

#include <vitis/ai/nnpp/segmentation.hpp>
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

namespace onnx_segmentation {

class OnnxSegmentation : public OnnxTask {
 public:
  static std::unique_ptr<OnnxSegmentation> create(
      const std::string& model_name) {
    return std::unique_ptr<OnnxSegmentation>(new OnnxSegmentation(model_name));
  }

 protected:
  OnnxSegmentation(const std::string& model_name);
  OnnxSegmentation(const OnnxSegmentation&) = delete;

 public:
  virtual ~OnnxSegmentation() {}
  virtual std::vector<SegmentationResult> run(
      const std::vector<cv::Mat>& image);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;

  // int real_batch;
  int batch_size;
  std::vector<float*> output_tensor_ptr;
};

template <class T>
void max_index_c(T* d, int c, int g, uint8_t* results) {
  for (int i = 0; i < g; ++i) {
    auto it = std::max_element(d, d + c);
    results[i] = it - d;
    d += c;
  }
}

template <typename T>
std::vector<T> permute(const T* input, size_t C, size_t H, size_t W) {
  std::vector<T> output(C * H * W);
  for (auto c = 0u; c < C; c++) {
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        output[h * W * C + w * C + c] = input[c * H * W + h * W + w];
      }
    }
  }
  return output;
}

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

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

OnnxSegmentation::OnnxSegmentation(const std::string& model_name)
    : OnnxTask(model_name) {
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

std::vector<SegmentationResult> OnnxSegmentation::run(
    const std::vector<cv::Mat>& image) {
  __TIC__(total)
  __TIC__(preprocess)
  cv::Mat resize_image;
  auto input_shape = input_shapes_[0];
  auto batch = input_shape[0];
  auto channel = input_shape[1];
  auto height = input_shape[2];
  auto width = input_shape[3];
  auto size = cv::Size((int)width, (int)height);
  auto real_batch = std::min((uint32_t)image.size(), (uint32_t)batch);
  auto batch_size = channel * height * width;

  for (auto i = 0u; i < real_batch; ++i) {
    cv::resize(image[i], resize_image, size);
    set_input_image(resize_image, input_tensor_values.data() + i * batch_size);
  }
  input_tensors = convert_input(input_tensor_values, input_tensor_values.size(),
                                input_shape);

  __TOC__(preprocess)

  __TIC__(session_run)
  run_task(input_tensors, output_tensors);
  __TOC__(session_run)

  __TIC__(postprocess)
  output_tensor_ptr[0] = output_tensors[0].GetTensorMutableData<float>();
  auto output_batch_size =
      output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount() / batch;
  auto oc = output_shapes_[0][1];
  auto oh = output_shapes_[0][2];
  auto ow = output_shapes_[0][3];
  std::vector<SegmentationResult> results;
  for (auto i = 0u; i < real_batch; ++i) {
    auto hwc =
        permute(output_tensor_ptr[0] + i * output_batch_size, oc, oh, ow);
    cv::Mat result(oh, ow, CV_8UC1);
    max_index_c(hwc.data(), oc, oh * ow, result.data);
    results.emplace_back(SegmentationResult{(int)ow, (int)oh, result});
  }
  __TOC__(postprocess)
  __TOC__(total)
  return results;
}

}  // namespace onnx_segmentation

