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

#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vart/zero_copy_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "xir/sfm_controller.hpp"

static cv::Mat read_image(const std::string& image_file_name);

static std::unique_ptr<vart::TensorBuffer> allocate_tensor_buffer(
    xclDeviceHandle h, xclBufferHandle bo, size_t offset,
    const xir::Tensor* tensor);
static void mimic_hw_preprocessing(xclDeviceHandle h,         //
                                   xclBufferHandle input_bo,  //
                                   size_t offset,             //
                                   cv::Mat input_image,       //
                                   int width,                 //
                                   int height,                //
                                   float input_scale          //
);

static void mimic_hw_postprocessing(xclDeviceHandle h,          //
                                    xclBufferHandle output_bo,  //
                                    size_t offset,              //
                                    size_t size,                //
                                    float output_scale          //
);
static std::vector<float> convert_fixpoint_to_float(int8_t* data, size_t size,
                                                    float scale);
static std::vector<float> softmax(const std::vector<float>& input);
static cv::Mat preprocess_image(cv::Mat input_image, cv::Size size);
static std::vector<std::pair<int, float>> topk(const float* score, size_t size,
                                               int K);

static void print_topk(const std::vector<std::pair<int, float>>& topk);

static const char* lookup(int index);
static int get_fix_pos(const xir::Tensor* tensor);
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

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "usage: " << argv[0] << " <resnet50.xmodel> <sample_image>\n";
    return 0;
  }
  auto xmodel_file = std::string(argv[1]);
  const auto image_file_name = std::string(argv[2]);
  {
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto root = graph->get_root_subgraph();
    xir::Subgraph* subgraph = nullptr;
    for (auto c : root->children_topological_sort()) {
      if (c->get_attr<std::string>("device") == "DPU" && subgraph == nullptr) {
        subgraph = c;
      }
    }
    auto attrs = xir::Attrs::create();
    std::unique_ptr<vart::RunnerExt> runner =
        vart::RunnerExt::create_runner(subgraph, attrs.get());
    // prepare input tensor buffer
    // get the input and output buffer size for XRT BO allocation
    auto h = xclOpen(0, NULL, XCL_INFO);
    auto input_bo = xclAllocBO(h, vart::get_input_buffer_size(subgraph), 0, 0);
    auto input_tensors = runner->get_input_tensors();
    auto input_offsets = vart::get_input_offset(subgraph);
    auto input_tensor_buffer = allocate_tensor_buffer(
        // only support single input
        h, input_bo, input_offsets[0], input_tensors[0]);
    auto output_bo =
        xclAllocBO(h, vart::get_output_buffer_size(subgraph), 0, 0);
    auto output_offsets = vart::get_output_offset(subgraph);
    auto output_tensors = runner->get_output_tensors();
    auto output_tensor_buffer = allocate_tensor_buffer(
        // only support single output
        h, output_bo, output_offsets[0], output_tensors[0]);
    //
    auto input_tensor = input_tensors[0];
    auto height = input_tensor->get_shape().at(1);
    auto width = input_tensor->get_shape().at(2);
    auto input_scale = vart::get_input_scale(input_tensor);
    auto output_tensor = output_tensors[0];
    auto output_scale = vart::get_output_scale(output_tensor);
    auto output_shape = output_tensor->get_shape();
    auto output_softmax_size = output_shape[output_shape.size() - 1];

    // a image file, e.g.
    // /usr/share/VITIS_AI_SDK/samples/classification/images/001.JPEG
    cv::Mat input_image = read_image(image_file_name);
    mimic_hw_preprocessing(h, input_bo, input_offsets[0], input_image, width,
                           height, input_scale);
    auto v = runner->execute_async({input_tensor_buffer.get()},
                                   {output_tensor_buffer.get()});
    auto status = runner->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";
    // post process
    // softmax & topk
    mimic_hw_postprocessing(h, output_bo, output_offsets[0],
                            output_softmax_size, output_scale);
    xclFreeBO(h, input_bo);
    xclFreeBO(h, output_bo);
    xclClose(h);
  }
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

static std::unique_ptr<vart::TensorBuffer> allocate_tensor_buffer(
    xclDeviceHandle h, xclBufferHandle bo, size_t offset,
    const xir::Tensor* tensor) {
  return vart::assistant::XrtBoTensorBuffer::create({h, bo}, tensor);
}

static void mimic_hw_preprocessing(xclDeviceHandle h,         //
                                   xclBufferHandle input_bo,  //
                                   size_t offset,             //
                                   cv::Mat input_image,       //
                                   int width,                 //
                                   int height,                //
                                   float input_scale          //
) {
  cv::Mat image = preprocess_image(input_image, cv::Size(width, height));
  auto data = (int8_t*)xclMapBO(h, input_bo, true);  //
  auto data_in = data + offset;
  setImageBGR(image, (void*)data_in, input_scale);
  xclSyncBO(h, input_bo, XCL_BO_SYNC_BO_TO_DEVICE, width * height * 3, offset);
  xclUnmapBO(h, input_bo, data);
  return;
}

static void mimic_hw_postprocessing(xclDeviceHandle h,          //
                                    xclBufferHandle output_bo,  //
                                    size_t offset,              //
                                    size_t softmax_size,        //
                                    float output_scale          //
) {
  xclSyncBO(h, output_bo, XCL_BO_SYNC_BO_FROM_DEVICE,
            // TODO: hard coded value
            softmax_size, offset);
  auto data = (int8_t*)xclMapBO(h, output_bo, true);  //
  auto data_in = data + offset;
  // run softmax
  auto softmax_input =
      convert_fixpoint_to_float(data_in, softmax_size, output_scale);
  auto softmax_output = softmax(softmax_input);
  constexpr int TOPK = 5;
  auto r = topk(&softmax_output[0], softmax_size, TOPK);
  print_topk(r);
  return;
}

static std::vector<float> convert_fixpoint_to_float(int8_t* data, size_t size,
                                                    float scale) {
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

static int get_fix_pos(const xir::Tensor* tensor) {
  int fixpos = tensor->template get_attr<int>("fix_point");
  return fixpos;
}
