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
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>

#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"

static cv::Mat read_image(const std::string& image_file_name);
static cv::Mat preprocess_image(cv::Mat input_image, cv::Size size);
static std::vector<float> convert_fixpoint_to_float(vart::TensorBuffer* tensor,
                                                    float scale);
static std::vector<float> softmax(const std::vector<float>& input);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static std::vector<std::pair<int, float>> post_process(
    vart::TensorBuffer* tensor_buffer, float scale);
static void print_topk(const std::vector<std::pair<int, float>>& topk);

static const char* lookup(int index);

static void setImageBGR(const cv::Mat& image, void* data1, float scale) {
  // mean value and scale are model specific, we need to check the
  // model to get concrete value. For resnet50, they are 104, 107,
  // 123, and 0.5, 0.5, 0.5 respectively
  signed char* data = (signed char*)data1;
  int c = 0;
  for (auto row = 0; row < image.rows; row++) {
    for (auto col = 0; col < image.cols; col++) {
      auto v = image.at<cv::Vec3b>(row, col);
      // convert BGR to RGB, substract mean value and times scale;
      auto B = (float)v[0];
      auto G = (float)v[1];
      auto R = (float)v[2];
      auto nB = (B - 104.0f) * scale;
      auto nG = (G - 107.0f) * scale;
      auto nR = (R - 123.0f) * scale;
      nB = std::max(std::min(nB, 127.0f), -128.0f);
      nG = std::max(std::min(nG, 127.0f), -128.0f);
      nR = std::max(std::min(nR, 127.0f), -128.0f);
      data[c++] = (int)(nB);
      data[c++] = (int)(nG);
      data[c++] = (int)(nR);
    }
  }
}
// static std::ostream& operator<<(std::ostream& out,
//                                 const xir::Tensor* tensor) {
//   out << "xir::Tensor{";
//   out << tensor->get_name() << ":(";
//   auto dims = tensor->get_shape();
//   for (auto i = 0u; i < dims.size(); ++i) {
//     if (i != 0) {
//       out << ",";
//     }
//     out << dims[i];
//   }
//   out << ")";
//   out << "}";
//   return out;
// }
//
static std::unique_ptr<vart::TensorBuffer> create_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor) {
  return std::make_unique<vart::mm::HostFlatTensorBuffer>(tensor);
}

int main(int argc, char* argv[]) {
  {
    const auto image_file_name = std::string(argv[1]);  // std::string(argv[2]);
    const auto filename = "resnet50.xmodel";            //
    const auto kernel_name = std::string("resnet50_0");
    auto runner =
        vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, kernel_name);
    auto input_tensors = runner->get_input_tensors();
    auto output_tensors = runner->get_output_tensors();

    // create runner and input/output tensor buffers;
    auto input_scale = vart::get_input_scale(input_tensors);
    auto output_scale = vart::get_output_scale(output_tensors);

    // a image file, e.g.
    // /usr/share/VITIS_AI_SDK/samples/classification/images/001.JPEG
    // load the image
    cv::Mat input_image = read_image(image_file_name);

    // prepare input tensor buffer
    CHECK_EQ(input_tensors.size(), 1u) << "only support resnet50 model";
    auto input_tensor = input_tensors[0];
    auto height = input_tensor->get_shape().at(1);
    auto width = input_tensor->get_shape().at(2);
    auto input_tensor_buffer = create_cpu_flat_tensor_buffer(input_tensor);
    // prepare output tensor buffer
    CHECK_EQ(output_tensors.size(), 1u) << "only support resnet50 model";
    auto output_tensor = output_tensors[0];
    auto output_tensor_buffer = create_cpu_flat_tensor_buffer(output_tensor);
    // print intput and output dims
    //  LOG(INFO) << "inputs: " << input_tensor << ", outputs:" <<
    //  output_tensor;
    // proprocess, i.e. resize if necessary
    cv::Mat image = preprocess_image(input_image, cv::Size(width, height));
    // set the input image and preprocessing
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    std::tie(data_in, size_in) =
        input_tensor_buffer->data(std::vector<int>{0, 0, 0, 0});
    setImageBGR(image, (void*)data_in, input_scale[0]);
    // start the dpu
    auto v = runner->execute_async({input_tensor_buffer.get()},
                                   {output_tensor_buffer.get()});
    auto status = runner->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";
    // get output.
    // post process
    auto topk = post_process(output_tensor_buffer.get(), output_scale[0]);
    // print the result
    print_topk(topk);
  }
  // LOG(INFO) << "bye";
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
    vart::TensorBuffer* tensor_buffer, float scale) {
  // run softmax
  auto softmax_input = convert_fixpoint_to_float(tensor_buffer, scale);
  auto softmax_output = softmax(softmax_input);
  constexpr int TOPK = 5;
  return topk(softmax_output, TOPK);
}

static std::vector<float> convert_fixpoint_to_float(
    vart::TensorBuffer* tensor_buffer, float scale) {
  uint64_t data = 0u;
  size_t size = 0u;
  std::tie(data, size) = tensor_buffer->data(std::vector<int>{0, 0});
  signed char* data_c = (signed char*)data;
  auto ret = std::vector<float>(size);
  transform(data_c, data_c + size, ret.begin(),
            [scale](signed char v) { return ((float)v) * scale; });
  return ret;
}

static std::vector<float> softmax(const std::vector<float>& input) {
  auto output = std::vector<float>(input.size());
  std::transform(input.begin(), input.end(), output.begin(), expf);
  auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
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

static void print_topk(const std::vector<std::pair<int, float>>& topk) {
  for (const auto& v : topk) {
    std::cout << std::setiosflags(std::ios::left) << std::setw(11)
              << "score[" + std::to_string(v.first) + "]"
              << " =  " << std::setw(12) << v.second
              << " text: " << lookup(v.first)
              << std::resetiosflags(std::ios::left) << std::endl;
  }
}

static const char* lookup(int index) {
  static const char* table[] = {
#include "word_list.inc"
  };

  if (index < 0) {
    return "";
  } else {
    return table[index];
  }
};
