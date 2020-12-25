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
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/benchmark.hpp>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>

#include "dpu/dpu_session.hpp"
#include "vitis/dpu/tensor_buffer.hpp"
DEF_ENV_PARAM(TOTAL_COUNTER, "1");
static void setImageBGR(const cv::Mat& image, void* data1) {
  signed char* data = (signed char*)data1;
  int c = 0;
  for (auto row = 0; row < image.rows; row++) {
    for (auto col = 0; col < image.cols; col++) {
      auto v = image.at<cv::Vec3b>(row, col);
      // convert BGR to RGB, substract mean value and times scale;
      auto B = (float)v[0];
      auto G = (float)v[1];
      auto R = (float)v[2];
      auto nB = (B - 104.0f) * 1.0f;
      auto nG = (G - 107.0f) * 1.0f;
      auto nR = (R - 123.0f) * 1.0f;
      nB = std::max(std::min(nB, 127.0f), -128.0f);
      nG = std::max(std::min(nG, 127.0f), -128.0f);
      nR = std::max(std::min(nR, 127.0f), -128.0f);
      data[c++] = (int)(nR);
      data[c++] = (int)(nG);
      data[c++] = (int)(nB);
    }
  }
}
static std::ostream& operator<<(std::ostream& out, const xir::Tensor* tensor) {
  out << "xir::Tensor{";
  out << tensor->get_name() << ":(";
  auto dims = tensor->get_shape();
  for (auto i = 0u; i < dims.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << dims[i];
  }
  out << ")";
  out << "}";
  return out;
}

static std::unique_ptr<std::unique_ptr<vart::TensorBuffer>>
create_cpu_flat_tensor_buffer(const xir::Tensor* tensor) {
  auto size =
      tensor->get_element_num() * vitis::ai::size_of(tensor->get_data_type());
  void* data = (void*)(new char[size]);
  return std::make_unique<std::unique_ptr<vart::TensorBuffer>>(data, tensor);
}

static void free_cpu_flat_tensor_buffer(
    std::unique_ptr<vart::TensorBuffer>* tensor_buffer) {
  void* data_in;
  size_t size_in;
  std::tie(data_in, size_in) =
      tensor_buffer->data(std::vector<int>{0, 0, 0, 0});
  delete[]((char*)data_in);
}
struct adapter {
  explicit adapter(std::unique_ptr<vart::Runner>&& runner)
      : runner_{std::move(runner)},
        dims{runner_->get_input_tensors()[0]->get_shape()} {}
  void run(const cv::Mat& image) {
    // prepare input tensor buffer
    auto input_tensors = runner_->get_input_tensors();
    CHECK_EQ(input_tensors.size(), 1) << "only support resnet50 model";
    auto input_tensor = input_tensors[0];
    auto input_tensor_buffer = create_cpu_flat_tensor_buffer(input_tensor);
    // prepare output tensor buffer
    auto output_tensors = runner_->get_output_tensors();
    CHECK_EQ(output_tensors.size(), 1) << "only support resnet50 model";
    auto output_tensor = output_tensors[0];
    auto output_tensor_buffer = create_cpu_flat_tensor_buffer(output_tensor);
    // print intput and output dims
    LOG_IF(INFO, false) << "inputs: " << input_tensor
                        << ", outputs:" << output_tensor;
    void* data_in = nullptr;
    size_t size_in = 0u;
    std::tie(data_in, size_in) =
        input_tensor_buffer->data(std::vector<int>{0, 0, 0, 0});
    setImageBGR(image, data_in);
    // start the dpu
    auto v = runner_->execute_async({input_tensor_buffer.get()},
                                    {output_tensor_buffer.get()});
    auto status = runner_->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";
    // clean up
    free_cpu_flat_tensor_buffer(input_tensor_buffer.get());
    free_cpu_flat_tensor_buffer(output_tensor_buffer.get());
    return;
  }
  int getInputHeight() { return dims[1]; }
  int getInputWidth() { return dims[2]; }
  std::unique_ptr<vart::Runner> runner_;
  std::vector<int> dims;
};
int main(int argc, char* argv[]) {
  const auto filename = std::string("/usr/lib/resnet_50.xmodel");
  const auto kernel_name = std::string(
      "resnet_50_0");  // to be oboslete, for backward compatibility;
  auto session = vart::dpu::DpuSession::create(filename, kernel_name);
  auto sessionp = session.get();
  // create runner and input/output tensor buffers;
  auto input_scale = session->get_input_scale();
  auto output_scale = session->get_output_scale();
  return vitis::ai::main_for_performance(argc, argv, [sessionp]() {
    return std::make_unique<adapter>(sessionp->create_runner());
  });
  LOG(INFO) << "bye";
  return 0;
}
