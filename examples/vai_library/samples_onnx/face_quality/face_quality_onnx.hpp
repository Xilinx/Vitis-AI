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
#include <algorithm>  // std::generate
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"

DEF_ENV_PARAM(DEBUG_FACEQUALITY_ONNX, "0");

using namespace std;

static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
  cv::Mat resized_image;
  cv::resize(image, resized_image, size);
  return resized_image;
}

static void set_input_image(const cv::Mat& image, float* data) {
  float mean[3] = {127.5f, 127.5f, 127.5f};
  float scales[3] = {0.0078431f, 0.0078431f, 0.0078431f};
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < image.rows; h++) {
      for (int w = 0; w < image.cols; w++) {
        auto image_data = (image.at<cv::Vec3b>(h, w)[c] - mean[c]) * scales[c];
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

static float mapped_quality_day(float original_score) {
  return 1.0f / (1.0f + std::exp(-((3.0f * original_score - 600.0f) / 150.0f)));
}

struct FaceQualityOnnxResult {
  /// Width of a input image
  int width;
  /// Height of a input image
  int height;
  /// The quality of face. The value range is from 0 to 1. If the option
  /// "original_quality" in the model prototxt is false, it is a normal mode. If
  /// the option "original_quality" is true, the quality score can be larger
  /// than 1, this is a special mode only for accuracy test.
  float score;
  /// Five key points coordinate. An array of <x,y> has five elements where x
  /// and y are normalized relative to input image columns and rows. The value
  /// range is from 0 to 1.
  std::array<std::pair<float, float>, 5> points;
};

class FaceQualityOnnx : public OnnxTask {
 public:
  static std::unique_ptr<FaceQualityOnnx> create(
      const std::string& model_name) {
    return std::unique_ptr<FaceQualityOnnx>(new FaceQualityOnnx(model_name));
  }
  virtual ~FaceQualityOnnx() {}

  FaceQualityOnnx(const std::string& model_name) : OnnxTask(model_name) {}

  FaceQualityOnnx(const FaceQualityOnnx&) = delete;

  std::vector<FaceQualityOnnxResult> run(
      const std::vector<cv::Mat> batch_images) {
    // print name/shape of inputs
    std::vector<std::string> input_names = get_input_names();

    // print name/shape of outputs
    std::vector<std::string> output_names = get_output_names();

    std::vector<std::vector<int64_t>> input_shapes = get_input_shapes();
    std::vector<std::vector<int64_t>> output_shapes = get_output_shapes();
    // cout << "Output Node Name/Shape (" << output_names.size() << "):" <<
    // endl; for (size_t i = 0; i < output_names.size(); i++) {
    //  cout << "\t" << output_names[i] << " : " <<
    //  print_shape(output_shapes[i])
    //       << endl;
    //}

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 2);

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes[0];
    int total_number_elements = calculate_product(input_shape);
    std::vector<float> input_tensor_values(total_number_elements);
    auto hw_batch = input_shape[0];
    auto valid_batch = std::min((int)hw_batch, (int)batch_images.size());

    preprocess(batch_images, input_tensor_values, input_shape, valid_batch);

    std::vector<Ort::Value> input_tensors = convert_input(
        input_tensor_values, input_tensor_values.size(), input_shape);

    std::vector<Ort::Value> output_tensors;
    run_task(input_tensors, output_tensors);

    auto results = postprocess(output_tensors, valid_batch);
    return results;
  }

 protected:
  void preprocess(const std::vector<cv::Mat>& images,
                  std::vector<float>& input_tensor_values,
                  std::vector<int64_t>& input_shape, int valid_batch) {
    // auto batch = input_shape[0];
    auto channel = input_shape[1];
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto batch_size = channel * height * width;
    auto size = cv::Size(width, height);
    // CHECK_EQ(images.size(), batch)
    //    << "images number be read into input buffer must be equal to batch";

    for (auto index = 0; index < valid_batch; ++index) {
      auto resize_image = preprocess_image(images[index], size);
      set_input_image(resize_image,
                      input_tensor_values.data() + batch_size * index);
    }
  }

  std::vector<FaceQualityOnnxResult> postprocess(
      std::vector<Ort::Value>& output_tensors, int valid_batch) {
    std::vector<FaceQualityOnnxResult> results;
    auto input_width = getInputWidth();
    auto input_height = getInputHeight();

    auto point_layer_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto quality_layer_ptr = output_tensors[1].GetTensorMutableData<float>();

    auto output_point_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    auto hw_batch = output_point_shape[0];
    auto channel = output_point_shape[1];

    for (auto index = 0; index < valid_batch; ++index) {
      // 5 points
      auto points = std::unique_ptr<std::array<std::pair<float, float>, 5>>(
          new std::array<std::pair<float, float>, 5>());
      auto total_number_elements =
          output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
      // auto size = total_number_elements / batch * sizeof(float);
      auto size = total_number_elements / hw_batch;
      for (auto i = 0u; i < points->size(); i++) {
        auto x = point_layer_ptr[size * index + i] / input_width;
        auto y = point_layer_ptr[size * index + i + channel / 2] / input_height;
        (*points)[i] = std::make_pair(x, y);
      }
      // quality output
      float score_original = quality_layer_ptr[index];
      float score = score_original;
      if (!getenv((const char*)("ORIGINAL_QUALITY"))) {
        score = mapped_quality_day(score_original);
      }
      FaceQualityOnnxResult r = FaceQualityOnnxResult{
          (int)input_width, (int)input_height, score, *points};
      results.emplace_back(r);
    }
    return results;
  }
};

