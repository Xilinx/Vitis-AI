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

using namespace std;

static cv::Mat croppedImage(const cv::Mat& image, int height, int width);
static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size);
static void set_input_image(const cv::Mat& image, float* data);
static std::vector<float> softmax(float* data, int64_t size);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
  cv::Mat cropped_img;
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
  return cropped_img;
}

static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
  float smallest_side = 299;
  float scale =
      smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  // LOG_IF(INFO, ENV_PARAM(ENABLE_CLASSIFICATION_DEBUG))
  //    << "resize: Width = " << image.cols * scale
  //    << " Height = " << image.rows * scale;
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(ceil(image.cols * scale), ceil(image.rows * scale)));
  // LOG_IF(INFO, ENV_PARAM(ENABLE_CLASSIFICATION_DEBUG))
  //   << "after resize: " << resized_image.size;
  // croppedImage(resized_image, height, width, pro_res);
  auto result = croppedImage(resized_image, size.height, size.width);
  // if (ENV_PARAM(CLASSIFICATION_SET_INPUT)) {
  //  std::ifstream input_bin("input_fix.bin", std::ios::binary);
  //  if (input_bin.is_open()) {
  //    input_bin.seekg(0, input_bin.end);
  //    int fsize = input_bin.tellg();
  //    input_bin.seekg(0, input_bin.beg);
  //    input_bin.read((char*)pro_res.data, fsize);
  //    input_bin.close();
  //  } else {
  //    LOG(INFO) << "Pleasae rename your input file as input_fix.bin!";
  //  }
  //}
  return result;
}

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

static std::vector<float> softmax(float* data, int64_t size) {
  auto output = std::vector<float>(size);
  std::transform(data, data + size, output.begin(), expf);
  auto sum =
      std::accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K) {
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [&score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform(
      indices.begin(), indices.begin() + K, ret.begin(),
      [&score](int index) { return std::make_pair(index, score[index]); });
  return ret;
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

struct InceptionV3PtOnnxResult {
  struct Score {
    ///  The index of the result in the ImageNet.
    int index;
    ///  Confidence of this category.
    float score;
  };
  /**
   *A vector of object width confidence in the first k; k defaults to 5 and
   *can be modified through the model configuration file.
   */
  std::vector<Score> scores;
};

class InceptionV3PtOnnx : public OnnxTask {
 public:
  static std::unique_ptr<InceptionV3PtOnnx> create(
      const std::string& model_name) {
    return std::unique_ptr<InceptionV3PtOnnx>(
        new InceptionV3PtOnnx(model_name));
  }
  virtual ~InceptionV3PtOnnx() {}
  InceptionV3PtOnnx(const std::string& model_name) : OnnxTask(model_name) {}

  InceptionV3PtOnnx(const InceptionV3PtOnnx&) = delete;

  std::vector<InceptionV3PtOnnxResult> run(
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
    assert(input_names.size() == 1 && output_names.size() == 1);

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

    auto results = postprocess(output_tensors[0], valid_batch);
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

    auto size = cv::Size((int)width, (int)height);
    // CHECK_EQ(images.size(), batch)
    //    << "images number be read into input buffer must be equal to batch";

    for (auto index = 0; index < valid_batch; ++index) {
      auto resize_image = preprocess_image(images[index], size);
      set_input_image(resize_image,
                      input_tensor_values.data() + batch_size * index);
    }
  }

  std::vector<InceptionV3PtOnnxResult> postprocess(Ort::Value& output_tensor,
                                                   int valid_batch) {
    std::vector<InceptionV3PtOnnxResult> results;
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    // auto hw_batch = output_shape[0];
    auto channel = output_shape[1];
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<float>();
    for (auto index = 0; index < valid_batch; ++index) {
      auto softmax_output =
          softmax(output_tensor_ptr + channel * index, channel);
      auto tb_top5 = topk(softmax_output, 5);
      // std::cout << "batch_index: " << index << std::endl;
      // print_topk(tb_top5);
      InceptionV3PtOnnxResult r;

      for (const auto& v : tb_top5) {
        r.scores.push_back(InceptionV3PtOnnxResult::Score{v.first, v.second});
      }
      results.emplace_back(r);
    }
    return results;
  }
};

