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
#include <xrt.h>

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
#include <xir/graph/graph.hpp>

#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vart/softmax_runner.hpp"
#include "vitis/ai/collection_helper.hpp"

static cv::Mat read_image(const std::string& image_file_name);
static std::vector<std::pair<int, float>> topk(const float* score, size_t size,
                                               int K);
static void print_topk(const std::vector<std::pair<int, float>>& topk);
static void croppedImage(const cv::Mat& image, int height, int width,
                         cv::Mat& cropped_img);
static void vgg_preprocess(const cv::Mat& image, int height, int width,
                           cv::Mat& pro_res);
static const char* lookup(int index);
static void setImageRGB(const cv::Mat& image, void* data1, float scale);
static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "usage: " << argv[0]
         << " <resnet_v1_50_tf.xmodel> <sample_image>\n";
    return 0;
  }
  auto xmodel_file = std::string(argv[1]);
  const auto image_file_name = std::string(argv[2]);
  {
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto root = graph->get_root_subgraph();
    xir::Subgraph* subgraph = nullptr;
    xir::Subgraph* softmax_subgraph = nullptr;
    for (auto c : root->children_topological_sort()) {
      if (c->get_attr<std::string>("device") == "DPU" && subgraph == nullptr) {
        subgraph = c;
      }
      if (c->get_attr<std::string>("device") == "SMFC" &&
          softmax_subgraph == nullptr) {
        softmax_subgraph = c;
      }
    }
    auto attrs = xir::Attrs::create();
    std::unique_ptr<vart::RunnerExt> runner =
        vart::RunnerExt::create_runner(subgraph, attrs.get());
    // a image file, e.g.
    // /usr/share/VITIS_AI_SDK/samples/classification/images/001.JPEG
    // load the image
    cv::Mat input_image = read_image(image_file_name);
    // prepare input tensor buffer
    auto input_tensor_buffers = runner->get_inputs();
    auto output_tensor_buffers = runner->get_outputs();
    CHECK_EQ(input_tensor_buffers.size(), 1u)
        << "only support resnet_v1_50_tf model";
    CHECK_EQ(output_tensor_buffers.size(), 1u)
        << "only support resnet_v1_50_tf model";

    auto input_tensor = input_tensor_buffers[0]->get_tensor();
    auto batch = input_tensor->get_shape().at(0);
    auto height = input_tensor->get_shape().at(1);
    auto width = input_tensor->get_shape().at(2);
    auto input_scale = vart::get_input_scale(input_tensor);

    // proprocess, i.e. resize if necessary
    cv::Mat image;
    vgg_preprocess(input_image, height, width, image);
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    for (auto batch_idx = 0; batch_idx < batch; ++batch_idx) {
      std::tie(data_in, size_in) =
          input_tensor_buffers[0]->data(std::vector<int>{batch_idx, 0, 0, 0});
      CHECK_NE(size_in, 0u);
      setImageRGB(image, (void*)data_in, input_scale);
    }

    // start the dpu
    for (auto& input : input_tensor_buffers) {
      input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                   input->get_tensor()->get_shape()[0]);
    }
    auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
    auto status = runner->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";

    // post process
    auto sfm_runner(
        std::make_unique<vart::SoftmaxRunner>(softmax_subgraph, attrs.get()));
    auto sfm_tensor_buffer = sfm_runner->get_outputs()[0];
    auto sfm_v = sfm_runner->execute_async({output_tensor_buffers[0]},
                                           {sfm_tensor_buffer});
    auto sfm_status = runner->wait((int)sfm_v.first, -1);
    CHECK_EQ(sfm_status, 0) << "failed to run softmax";

    uint64_t sfm_output_addr = 0u;
    uint64_t sfm_output_size = 0u;
    auto input_dim_idx =
        get_index_zeros(output_tensor_buffers[0]->get_tensor());
    std::tie(sfm_output_addr, sfm_output_size) =
        sfm_tensor_buffer->data(input_dim_idx);
    const unsigned int cls = sfm_tensor_buffer->get_tensor()->get_shape()[1];
    // sorting
    auto topk_value = topk((float*)sfm_output_addr, cls, 5u);
    // print the result
    print_topk(topk_value);
  }
  return 0;
}

static cv::Mat read_image(const std::string& image_file_name) {
  // read image from a file
  auto input_image = cv::imread(image_file_name);
  CHECK(!input_image.empty()) << "cannot load " << image_file_name;
  return input_image;
}

static std::vector<std::pair<int, float>> topk(const float* score, size_t size,
                                               int K) {
  auto indices = std::vector<int>(size);
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
    std::cout << setiosflags(ios::left) << std::setw(11)
              << "score[" + std::to_string(v.first) + "]"
              << " =  " << std::setw(12) << v.second
              << " text: " << lookup(v.first) << resetiosflags(ios::left)
              << std::endl;
  }
}

static void croppedImage(const cv::Mat& image, int height, int width,
                         cv::Mat& cropped_img) {
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
}

static void vgg_preprocess(const cv::Mat& image, int height, int width,
                           cv::Mat& pro_res) {
  float smallest_side = 256;
  float scale =
      smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(image.cols * scale, image.rows * scale));
  croppedImage(resized_image, height, width, pro_res);
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

static void setImageRGB(const cv::Mat& image, void* data1, float scale) {
  signed char* data = (signed char*)data1;
  int c = 0;
  for (auto row = 0; row < image.rows; row++) {
    for (auto col = 0; col < image.cols; col++) {
      auto v = image.at<cv::Vec3b>(row, col);
      // convert BGR to RGB, substract mean value and times scale;
      auto B = (float)v[0];
      auto G = (float)v[1];
      auto R = (float)v[2];
      auto nB = (B - 103.94f) * scale;
      auto nG = (G - 116.78f) * scale;
      auto nR = (R - 123.68f) * scale;
      nB = std::max(std::min(nB, 127.0f), -128.0f);
      nG = std::max(std::min(nG, 127.0f), -128.0f);
      nR = std::max(std::min(nR, 127.0f), -128.0f);
      data[c++] = (int)(nR);
      data[c++] = (int)(nG);
      data[c++] = (int)(nB);
    }
  }
}

static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}
