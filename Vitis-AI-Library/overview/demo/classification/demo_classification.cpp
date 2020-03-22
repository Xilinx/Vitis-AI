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
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vart/dpu/dpu_runner_ext.hpp>
#include <vart/dpu/dpu_runner.hpp>

static cv::Mat read_image(const std::string& image_file_name);
static cv::Mat preprocess_image(cv::Mat input_image, cv::Size size);

static std::vector<float> convert_fixpoint_to_float(
    vitis::ai::TensorBuffer* tensor, float scale);

static std::vector<float> softmax(const std::vector<float>& input);

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);

static std::vector<std::pair<int, float>> post_process(
    vitis::ai::TensorBuffer* tensor_buffer, float scale);

static void print_topk(const std::vector<std::pair<int, float>>& topk);

static void setImageBGR(const cv::Mat& image, void* data1,
                        float scale_fix2float);

int main(int argc, char* argv[]) {
  {
    // a model dir name, e.g. inception_v1
    const auto model_dir_name = std::string("/usr/share/vitis_ai_library/models/resnet50");
    // create runner , input/output tensor buffer ;
    auto runners = vitis::ai::DpuRunner::create_dpu_runner(model_dir_name);
    auto runner = dynamic_cast<vart::dpu::DpuRunnerExt*>(runners[0].get());
    auto input_scale = runner->get_input_scale();
    auto output_scale = runner->get_output_scale();

    // a image file, e.g.
    // /usr/share/vitis_ai_library/demo/classification/demo_classification.jpg
    auto image_file_name = std::string(argv[1]);
    // load the image
    cv::Mat input_image = read_image(image_file_name);
    // get input tensor buffer
    auto input_tensors = runner->get_input_tensors();
    CHECK_EQ(input_tensors.size(), 1) << "only support classification model";
    auto input_tensor = input_tensors[0];
    auto height = input_tensor->get_dim_size(1);
    auto width = input_tensor->get_dim_size(2);
    auto input_size = cv::Size(width, height);
    auto input_tensor_buffer = runner->get_inputs()[0];
    auto output_tensor_buffer = runner->get_outputs()[0];
    // preprocess, i.e. resize if necessary
    cv::Mat image = preprocess_image(input_image, input_size);
    // set the input image and preprocessing
    void* data_in = nullptr;
    size_t size_in = 0u;
    std::tie(data_in, size_in) =
        input_tensor_buffer->data(std::vector<int>{0, 0, 0, 0});
    setImageBGR(image, data_in, input_scale[0]);
    auto v =
        runner->execute_async({input_tensor_buffer}, {output_tensor_buffer});
    auto status = runner->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";
    // post process
    auto topk = post_process(output_tensor_buffer, output_scale[0]);
    // print the result
    print_topk(topk);
  }
  LOG(INFO) << "bye";
  return 0;
}

static cv::Mat read_image(const std::string& image_file_name) {
  // read image from a file
  auto input_image = cv::imread(image_file_name);
  CHECK(!input_image.empty()) << "cannot load " << image_file_name;
  return input_image;
}

static cv::Mat preprocess_image(cv::Mat input_image, cv::Size size) {
  cv::Mat image;
  // resize it if size is not match
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  return image;
}

static std::vector<std::pair<int, float>> post_process(
    vitis::ai::TensorBuffer* tensor_buffer, float scale) {
  // run softmax
  auto softmax_input = convert_fixpoint_to_float(tensor_buffer, scale);
  auto softmax_output = softmax(softmax_input);
  constexpr int TOPK = 5;
  return topk(softmax_output, TOPK);
}

static std::vector<float> convert_fixpoint_to_float(
    vitis::ai::TensorBuffer* tensor_buffer, float scale) {
  //convert fixpoint to float 
  void* data = nullptr;
  size_t size = 0u;
  std::tie(data, size) = tensor_buffer->data(std::vector<int>{0, 0, 0, 0});
  signed char* data_c = (signed char*)data;
  auto ret = std::vector<float>(size);
  transform(data_c, data_c + size, ret.begin(),
            [scale](signed char v) { return ((float)v) * scale; });
  return ret;
}

static std::vector<float> softmax(const std::vector<float>& input) {
  // implement softmax
  auto output = std::vector<float>(input.size());
  std::transform(input.begin(), input.end(), output.begin(), expf);
  auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K) {
  //find top k
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

static void print_topk(const std::vector<std::pair<int, float>>& topk) {
  // print top k
  for (const auto& v : topk) {
    std::cout << "score[" << v.first << "] = " << v.second << std::endl;
  }
}

static void setImageBGR(const cv::Mat& image, void* data1,
                        float scale_fix2float) {
  //preprocess and set the input image 
  signed char* data = (signed char*)data1;
  int c = 0;
  for (auto row = 0; row < image.rows; row++) {
    for (auto col = 0; col < image.cols; col++) {
      auto v = image.at<cv::Vec3b>(row, col);
      auto B = (float)v[0];
      auto G = (float)v[1];
      auto R = (float)v[2];
      auto nB = (B - 104.0f) * 1.0f * scale_fix2float;
      auto nG = (G - 117.0f) * 1.0f * scale_fix2float;
      auto nR = (R - 123.0f) * 1.0f * scale_fix2float;
      nB = std::max(std::min(nB, 127.0f), -128.0f);
      nG = std::max(std::min(nG, 127.0f), -128.0f);
      nR = std::max(std::min(nR, 127.0f), -128.0f);
      data[c++] = (int)(nB);
      data[c++] = (int)(nG);
      data[c++] = (int)(nR);
    }
  }
}
