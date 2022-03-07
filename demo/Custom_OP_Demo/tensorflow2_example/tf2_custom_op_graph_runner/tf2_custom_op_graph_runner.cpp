/*
 * Copyright 2021 Xilinx Inc.
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

#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "vitis/ai/graph_runner.hpp"
using namespace std;
static void print_result(void* data1, size_t size);

// tf2_custom_op preprocess
static void preprocess_tf2_custom_op(
    const std::string& file,
    const std::vector<vart::TensorBuffer*>& input_tensor_buffers) {
  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto height = input_tensor->get_shape().at(1);
  auto width = input_tensor->get_shape().at(2);

  // get fix point
  CHECK(input_tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << input_tensor->get_name();
  int fixpos = input_tensor->get_attr<int>("fix_point");
  float fixed_scale = std::exp2f(1.0f * (float)fixpos);

  // read image file
  auto size = cv::Size(width, height);
  cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
  CHECK(!image.empty()) << "cannot read image from " << file;

  // resize image
  cv::Mat resize_image;
  if (size != image.size()) {
    cv::resize(image, resize_image, size);
  } else {
    image.copyTo(resize_image);
  }

  // prepare input tensor
  uint64_t data_in = 0u;
  size_t size_in = 0u;
  auto index = input_tensor->get_shape();
  std::fill(index.begin(), index.end(), 0);
  std::tie(data_in, size_in) = input_tensor_buffers[0]->data(index);
  unsigned char* data = (unsigned char*)data_in;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      auto image_data = resize_image.at<uchar>(h, w) * fixed_scale;
      data[h * width + w] = (unsigned char)image_data;
    }
  }
  return;
}

// tf2_custom_op postprocess
static void postprocess_tf2_custom_op(
    const std::vector<vart::TensorBuffer*>& output_tensor_buffers) {
  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto size = output_tensor_buffers.size();
  CHECK_EQ(size, 1) << "output_tensor_buffers.size() must be 1";

  // get result
  uint64_t data_out = 0u;
  size_t size_out = 0u;
  auto index = output_tensor->get_shape();
  std::fill(index.begin(), index.end(), 0);
  std::tie(data_out, size_out) = output_tensor_buffers[0]->data(index);
  auto elem_num = output_tensor->get_element_num();
  print_result((void*)data_out, elem_num);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: tf2_custom_op_graph_runner <model_file> <image_file>"
              << std::endl;
    abort();
  }
  // parse parameters
  std::string xmodel_file = std::string(argv[1]);
  std::string image_file = std::string(argv[2]);
  CHECK(!xmodel_file.empty()) << "invalid parameter model_file";
  CHECK(!image_file.empty()) << "invalid parameter image_file";
  std::cout << "model_file: " << xmodel_file << std::endl;
  std::cout << "image_file: " << image_file << std::endl;

  // create graph runner
  auto graph = xir::Graph::deserialize(xmodel_file);
  auto attrs = xir::Attrs::create();
  auto runner =
      vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
  CHECK(runner != nullptr);

  // get input/output tensor buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();

  // preprocess
  preprocess_tf2_custom_op(image_file, input_tensor_buffers);

  // sync input tensor buffers
  for (auto& input : input_tensor_buffers) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
  }

  // run graph runner
  auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run the graph";

  // sync output tensor buffers
  for (auto output : output_tensor_buffers) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                 output->get_tensor()->get_shape()[0]);
  }

  // postprocess
  postprocess_tf2_custom_op(output_tensor_buffers);

  return 0;
}

static void print_result(void* data1, size_t size) {
  const float* score = (const float*)data1;
  for (auto i = 0; i < size; ++i) {
    std::cout << setiosflags(ios::left) << std::setw(11)
              << "score[" + std::to_string(i) + "]"
              << " =  " << std::setw(12) << score[i] << resetiosflags(ios::left)
              << std::endl;
  }
  return;
}
