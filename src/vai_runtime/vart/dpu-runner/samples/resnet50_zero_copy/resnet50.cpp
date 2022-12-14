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

#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "xir/sfm_controller.hpp"
using namespace std;

static cv::Mat read_image(const std::string& image_file_name);
static cv::Mat preprocess_image(cv::Mat input_image, cv::Size size);
static std::vector<std::pair<int, float>> topk(const float* score, size_t size,
                                               int K);

static void print_topk(const std::vector<std::pair<int, float>>& topk);

static const char* lookup(int index);
static int get_fix_pos(const xir::Tensor* tensor);

static void dump_out(const std::string filename, void* data, size_t size) {
  std::cout << "dump_out from data: 0x" << std::hex << data << std::dec
            << ", size=" << size << "to file " << filename << std::endl;
  auto mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  CHECK(std::ofstream(filename, mode).write((char*)data, size).good())
      << " faild to write to " << filename;
}
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

static uint64_t get_physical_address(const xclDeviceHandle& handle,
                                     const unsigned int bo) {
  xclBOProperties p;
  auto error_code = xclGetBOProperties(handle, bo, &p);
  uint64_t phy = 0u;
  if (error_code != 0) {
    LOG(INFO) << "cannot xclGetBOProperties !";
  }
  phy = error_code == 0 ? p.paddr : -1;
  return phy;
}

static void run_user_specific_ip(uint64_t dpu_output_phy_addr, unsigned int cls,
                                 unsigned int group, int fixpos) {
  // allocate memory.
  auto device_id = 0;
  auto handle = xclOpen(device_id, NULL, XCL_INFO);
  CHECK(handle != XRT_NULL_HANDLE);
  auto bo_handle = xclAllocBO(handle, 1u * 1024u * 1024u, 0, 0);
  CHECK(bo_handle != XRT_NULL_BO);
  auto phy_addr_for_softmax = get_physical_address(handle, bo_handle);
  auto bo_addr = xclMapBO(handle, bo_handle, true);
  CHECK(bo_addr != nullptr);
  auto sfm_controller = xir::SfmController::get_instance();
  auto fmap_size = cls * group;
  // start the SMFC IP with zero-copy, DPU output is directly
  // feed to SMFC's input
  size_t core_idx = 0u;
  const uint32_t offset = 0u;
  sfm_controller->run_xrt_cu(core_idx, dpu_output_phy_addr, cls, group, fixpos,
                             phy_addr_for_softmax, offset);
  xclSyncBO(handle, bo_handle, XCL_BO_SYNC_BO_FROM_DEVICE,
            fmap_size * sizeof(float), 0u);
  dump_out("softmax_out_resnet50.bin", (void*)bo_addr, cls * sizeof(float));
  {
    // sorting
    auto topk_value = topk((float*)bo_addr, cls, 5u);
    // print the result
    print_topk(topk_value);
  }
  xclUnmapBO(handle, bo_handle, bo_addr);
  xclFreeBO(handle, bo_handle);
  xclClose(handle);
  return;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "usage: " << argv[0] << " <resnet50.xmodel> <sample_image>\n";
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
      if (c->get_attr<std::string>("device") == "CPU" &&
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
    CHECK_EQ(input_tensor_buffers.size(), 1u) << "only support resnet50 model";
    CHECK_EQ(output_tensor_buffers.size(), 1u) << "only support resnet50 model";
    // CHECK_EQ(output_tensors[0]->get_shape().size(), 2u)
    //   << "only support resnet50 model";
    // CHECK_EQ(output_tensors[0]->get_shape()[1], 1000u)
    //    << "only support resnet50 model";

    auto input_tensor = input_tensor_buffers[0]->get_tensor();
    auto batch = input_tensor->get_shape().at(0);
    auto height = input_tensor->get_shape().at(1);
    auto width = input_tensor->get_shape().at(2);

    auto input_scale = vart::get_input_scale(input_tensor);
    // proprocess, i.e. resize if necessary
    cv::Mat image = preprocess_image(input_image, cv::Size(width, height));
    // set the input image and preprocessing
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    for (auto batch_idx = 0; batch_idx < batch; ++batch_idx) {
      std::tie(data_in, size_in) =
          input_tensor_buffers[0]->data(std::vector<int>{batch_idx, 0, 0, 0});
      CHECK_NE(size_in, 0u);
      setImageBGR(image, (void*)data_in, input_scale);
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
    uint64_t dpu_output_phy_addr = 0u;
    uint64_t dpu_output_size = 0u;
    std::tie(dpu_output_phy_addr, dpu_output_size) =
        output_tensor_buffers[0]->data_phy({0, 0});
    const unsigned int cls =
        output_tensor_buffers[0]->get_tensor()->get_shape()[1];
    const unsigned int group = 1u;
    const int fixpos = get_fix_pos(output_tensor_buffers[0]->get_tensor());
    uint64_t dpu_output_virt_addr = 0u;
    uint64_t dpu_output_virt_size = 0u;
    std::tie(dpu_output_virt_addr, dpu_output_virt_size) =
        output_tensor_buffers[0]->data({0, 0});
    auto dump_size = output_tensor_buffers[0]->get_tensor()->get_data_size();
    dump_out("softmax_in_resnet50.bin", (void*)dpu_output_virt_addr, dump_size);
    // softmax & topk
    run_user_specific_ip(dpu_output_phy_addr, cls, group, fixpos);
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
