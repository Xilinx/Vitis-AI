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
#include <algorithm>  // std::generate
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include <vitis/ai/profiling.hpp>
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"

using namespace std;

struct RcanOnnxResult {
  cv::Mat feat;
};

// chw -> hwc
void get_output_image(cv::Mat& image, const float* data) {
  auto H = image.rows;
  auto W = image.cols;
  auto HW = H * W;
  auto C = 3;
  std::vector<cv::Mat> mat_channels;
  auto op = [](const float a) -> uint8_t { return (uint8_t)a; };
  for (int c = 0; c < C; c++) {
    auto src = data + c * HW;
    cv::Mat dst(H, W, CV_8UC1);
    std::transform(src, src + HW, dst.data, op);
    mat_channels.push_back(dst);
  }
  cv::merge(mat_channels, image);
}

std::vector<RcanOnnxResult> postprocess(Ort::Value& output_tensor,
                                        int valid_batch) {
  std::vector<RcanOnnxResult> results;
  auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  // auto batch = output_shape[0];
  // auto channel = output_shape[1];
  auto width = output_shape[3];
  auto height = output_shape[2];
  // LOG(INFO) << "batch:" << batch << ", channel:" << channel
  //          << ", width:" << width << ", height:" << height;
  auto output_tensor_ptr = output_tensor.GetTensorMutableData<float>();
  for (auto index = 0; index < valid_batch; ++index) {
    cv::Mat result_img = cv::Mat(height, width, CV_8UC3);
    __TIC__(GET_OUTPUT_IMAGE)
    get_output_image(result_img, output_tensor_ptr);
    __TOC__(GET_OUTPUT_IMAGE)
    auto r = RcanOnnxResult{result_img};
    results.emplace_back(r);
  }
  return results;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

class RcanOnnx : public OnnxTask {
 public:
  static std::unique_ptr<RcanOnnx> create(const std::string& model_name) {
    return std::unique_ptr<RcanOnnx>(new RcanOnnx(model_name));
  }
  virtual ~RcanOnnx() {}

  RcanOnnx(const std::string& model_name) : OnnxTask(model_name) {}

  RcanOnnx(const RcanOnnx&) = delete;

  std::vector<RcanOnnxResult> run(const std::vector<cv::Mat> batch_images) {
    // print name/shape of inputs
    std::vector<std::string> input_names = get_input_names();

    // print name/shape of outputs
    std::vector<std::string> output_names = get_output_names();

    std::vector<std::vector<int64_t>> input_shapes = get_input_shapes();
    std::vector<std::vector<int64_t>> output_shapes = get_output_shapes();

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes[0];
    int total_number_elements = calculate_product(input_shape);
    std::vector<float> input_tensor_values(total_number_elements, 0.f);
    auto hw_batch = input_shape[0];
    auto valid_batch = std::min((int)hw_batch, (int)batch_images.size());
    __TIC__(PRE)
    this->preprocess(batch_images, input_tensor_values, input_shape,
                     valid_batch);
    __TOC__(PRE)

    std::vector<Ort::Value> input_tensors = convert_input(
        input_tensor_values, input_tensor_values.size(), input_shape);

    __TIC__(RUN)
    std::vector<Ort::Value> output_tensors;
    run_task(input_tensors, output_tensors);
    __TOC__(RUN)

    __TIC__(POST)
    auto results = this->postprocess(output_tensors[0], valid_batch);
    __TOC__(POST)
    return results;
  }

 protected:
  void preprocess(const std::vector<cv::Mat>& images,
                  std::vector<float>& input_tensor_values,
                  std::vector<int64_t>& input_shape, int valid_batch);

  std::vector<RcanOnnxResult> postprocess(Ort::Value& output_tensor,
                                          int valid_batch) {
    return ::postprocess(output_tensor, valid_batch);
  }
};

void RcanOnnx::preprocess(const std::vector<cv::Mat>& images,
                       std::vector<float>& input_tensor_values,
                       std::vector<int64_t>& input_shape, int valid_batch) {
  // auto batch = input_shape[0];
  auto channel = input_shape[1];
  auto height = input_shape[2];
  auto width = input_shape[3];
  auto batch_size = channel * height * width;

  auto size = cv::Size((int)width, (int)height);
  // CHECK_EQ(images.size(), batch)
  //    << "images number be read into input buffer must be equal to batch";

  for (auto index = 0; index < valid_batch; ++index) {
    cv::Mat resize_image;
    if (images[index].size() != size) {
      cv::resize(images[index], resize_image, size);
    } else {
      resize_image = images[index];
    }
    set_input_image_bgr(resize_image,
                    input_tensor_values.data() + batch_size * index,
                    std::vector<float>{0.f, 0.f, 0.f},
                    std::vector<float>{1.f, 1.f, 1.f}
                  );
  }
}

