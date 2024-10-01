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

#include <opencv2/opencv.hpp>

#include "vitis/ai/graph_runner.hpp"

std::vector<std::string> supported_ext = {".jpg", ".png"};

static void get_image_files(const std::string& img_path,
                            const std::size_t batchsize,
                            std::vector<std::string>& images) {
  std::vector<std::string> files;
  cv::glob(img_path, files);

  for (auto f : files) {
    if (std::find(supported_ext.begin(), supported_ext.end(),
                  f.substr(f.rfind("."))) != supported_ext.end()) {
      images.push_back(f);
    }
  }

  auto num = images.size();
  if (num % batchsize) {
    auto append_num = batchsize - num % batchsize;
    while (append_num--) images.push_back(images[num - 1]);
  }

  return;
}

static void print_result(void* data1, size_t size) {
  const float* score = (const float*)data1;
  for (auto i = 0; i < size; ++i) {
    std::cout << std::setiosflags(std::ios::left) << std::setw(11)
              << "\tscore[" + std::to_string(i) + "]"
              << " =  " << std::setw(12) << score[i]
              << std::resetiosflags(std::ios::left) << std::endl;
  }

  return;
}

// tf2_custom_op preprocess
static void preprocess_tf2_custom_op(
    const std::vector<std::string>& files, const std::size_t file_index,
    const std::vector<vart::TensorBuffer*>& input_tensor_buffers) {
  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto input_shape = input_tensor->get_shape();

  auto height = input_shape.at(1);
  auto width = input_shape.at(2);
  auto size = cv::Size(width, height);

  // get fix point
  CHECK(input_tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << input_tensor->get_name();
  int fixpos = input_tensor->get_attr<int>("fix_point");
  float fixed_scale = std::exp2f(1.0f * (float)fixpos);

  for (auto i = 0; i < input_shape.at(0); ++i) {
    auto f_idx = file_index + i;
    cv::Mat image = cv::imread(files[f_idx], cv::IMREAD_GRAYSCALE);
    CHECK(!image.empty()) << "cannot read image from " << files[f_idx];

    cv::Mat resize_image;
    if (size != image.size()) {
      cv::resize(image, resize_image, size);
    } else {
      image.copyTo(resize_image);
    }

    uint64_t data_in = 0u;
    size_t size_in = 0u;
    auto index = input_shape;
    std::fill(index.begin(), index.end(), 0);
    index[0] = i;
    std::tie(data_in, size_in) = input_tensor_buffers[0]->data(index);
    unsigned char* data = (unsigned char*)data_in;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        auto image_data = resize_image.at<uchar>(h, w) * fixed_scale;
        data[h * width + w] = (unsigned char)image_data;
      }
    }
  }

  return;
}

// tf2_custom_op postprocess
static void postprocess_tf2_custom_op(
    const std::vector<std::string>& files, const std::size_t file_index,
    const std::vector<vart::TensorBuffer*>& output_tensor_buffers) {
  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto size = output_tensor_buffers.size();
  CHECK_EQ(size, 1) << "output_tensor_buffers.size() must be 1";
  auto batch = output_tensor->get_shape().at(0);

  for (auto i = 0; i < batch; ++i) {
    // get result
    uint64_t data_out = 0u;
    size_t size_out = 0u;
    auto index = output_tensor->get_shape();
    std::fill(index.begin(), index.end(), 0);
    index[0] = i;
    std::tie(data_out, size_out) = output_tensor_buffers[0]->data(index);
    auto elem_num = output_tensor->get_element_num() / batch;
    std::cout << "image file " << files[file_index + i]
              << " result:" << std::endl;
    print_result((void*)data_out, elem_num);
  }

  return;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: tf2_custom_op_graph_runner <model_file> <image_path>"
              << std::endl;
    abort();
  }
  // parse parameters
  std::string xmodel = std::string(argv[1]);
  std::string image_path = std::string(argv[2]);
  CHECK(!xmodel.empty()) << "invalid parameter model_file";
  CHECK(!image_path.empty()) << "invalid parameter image_path";
  std::cout << "model_file: " << xmodel << std::endl;
  std::cout << "image_path: " << image_path << std::endl;

  // create graph runner
  auto graph = xir::Graph::deserialize(xmodel);
  auto attrs = xir::Attrs::create();
  auto runner =
      vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
  CHECK(runner != nullptr);

  // get input/output tensor buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();
  auto batchsize = input_tensor_buffers[0]->get_tensor()->get_shape().at(0);

  std::vector<std::string> files;
  get_image_files(image_path, batchsize, files);

  auto num = files.size();
  auto loop_num = num / batchsize;

  for (auto i = 0; i < loop_num; ++i) {
    auto file_index = i * batchsize;
    // preprocess
    preprocess_tf2_custom_op(files, file_index, input_tensor_buffers);

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
    postprocess_tf2_custom_op(files, file_index, output_tensor_buffers);
  }

  return 0;
}
