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
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>

static std::unique_ptr<vitis::ai::DpuTask> create_dpu_task();
static cv::Mat read_image(const std::string& image_file_name);
static cv::Size get_dpu_task_size(vitis::ai::DpuTask* task);
static cv::Mat preprocess_image(cv::Mat input_image, cv::Size size);
static std::vector<float> convert_fixpoint_to_float(
    const vitis::ai::library::OutputTensor& tensor);
static std::vector<float> softmax(const std::vector<float>& input);
static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);
static std::vector<std::pair<int, float>> post_process(
    const vitis::ai::library::OutputTensor& tensor);
static void print_topk(const std::vector<std::pair<int, float>>& topk);

int main(int argc, char* argv[]) {
  auto task = create_dpu_task();
  // a image file, e.g.
  // /usr/share/XILINX_AI_SDK/samples/classification/images/001.JPEG
  auto image_file_name = std::string(argv[1]);
  // load the image
  cv::Mat input_image = read_image(image_file_name);
  // proprocess, i.e. resize if necessary
  cv::Mat image = preprocess_image(input_image, get_dpu_task_size(task.get()));
  // set the input image
  task->setImageBGR(image);
  // start the dpu
  task->run();
  // get output.
  auto output_tensor = task->getOutputTensor();
  // post process
  auto topk = post_process(output_tensor[0]);
  // print the result
  print_topk(topk);
  return 0;
}

static std::unique_ptr<vitis::ai::DpuTask> create_dpu_task() {
  // a kernel name, e.g. resnet_50, inception_v1_0, inception_v2_0,
  // inception_v3_0, etc
  auto kernel_name = "inception_v1_0";
  // create a dpu task object.
  auto task = vitis::ai::DpuTask::create(kernel_name);
  // preprocessing, please check the caffe model, e.g. deploy.prototxt
  task->setMeanScaleBGR({104.0f, 107.0f, 123.0f}, {1.0f, 1.0f, 1.0f});
  return std::move(task);
}

static cv::Mat read_image(const std::string& image_file_name) {
  // read image from a file
  auto input_image = cv::imread(image_file_name);
  CHECK(!input_image.empty()) << "cannot load " << image_file_name;
  return input_image;
}

static cv::Size get_dpu_task_size(vitis::ai::DpuTask* task) {
  auto input_tensor = task->getInputTensor();
  CHECK_EQ(input_tensor.size(), 1) << " the dpu model must have only one input";
  auto width = input_tensor[0].width;
  auto height = input_tensor[0].height;
  auto size = cv::Size(width, height);
  return size;
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
    const vitis::ai::library::OutputTensor& tensor) {
  // run softmax
  auto softmax_input = convert_fixpoint_to_float(tensor);
  auto softmax_output = softmax(softmax_input);
  constexpr int TOPK = 5;
  return topk(softmax_output, TOPK);
}

static std::vector<float> convert_fixpoint_to_float(
    const vitis::ai::library::OutputTensor& tensor) {
  auto scale = vitis::ai::tensor_scale(tensor);
  auto data = (signed char*)tensor.data;
  auto size = tensor.width * tensor.height * tensor.channel;
  auto ret = std::vector<float>(size);
  transform(data, data + size, ret.begin(),
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
    std::cout << "score[" << v.first << "] = " << v.second << std::endl;
  }
}
