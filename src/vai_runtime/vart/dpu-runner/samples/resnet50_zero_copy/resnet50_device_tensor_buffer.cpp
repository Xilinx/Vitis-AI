/*
 * Copyright 2022 Xilinx Inc.
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

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xir/graph/graph.hpp>

#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"
#include "vart/tensor_buffer_unowned_device.hpp"
#include "xir/sfm_controller.hpp"

struct Simulate_Host_Phy {
 public:
  Simulate_Host_Phy(xclDeviceHandle xcl_handle, size_t size)
      : handle_{xcl_handle}, size_{size} {
    bo_ = xclAllocBO(xcl_handle, size, 0, 0);
    CHECK(bo_ != XRT_NULL_BO);
    data_ = (void*)xclMapBO(xcl_handle, bo_, true);
    CHECK(nullptr != (void*)data_);

    xclBOProperties p;
    auto error_code = xclGetBOProperties(xcl_handle, bo_, &p);
    if (error_code != 0) {
      phy_ = -1;
      LOG(INFO) << "cannot xclGetBOProperties!";
    } else {
      phy_ = p.paddr;
      LOG(INFO) << "simulate host phy successfully!"
                << " phy_=0x" << std::hex << phy_ << " data_=0x" << data_
                << std::dec;
    }
  }
  ~Simulate_Host_Phy() {
    xclUnmapBO(handle_, bo_, (void*)data_);
    xclFreeBO(handle_, bo_);
    LOG(INFO) << "destroy simulate host phy";
  }

 public:
  uint64_t get_host_phy_addr() { return phy_; }
  void* get_host_virt_addr() { return data_; }
  void sync_data_for_write() {
    xclSyncBO(handle_, bo_, XCL_BO_SYNC_BO_TO_DEVICE, size_, 0);
  }
  void sync_data_for_read() {
    xclSyncBO(handle_, bo_, XCL_BO_SYNC_BO_FROM_DEVICE, size_, 0);
  }
  void dump_data(const std::string filename) {
    if (filename.empty()) return;

    LOG(INFO) << "dump data from phy: 0x" << std::hex << phy_ << ", virt: 0x"
              << data_ << std::dec;
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    CHECK(std::ofstream(filename, mode).write((char*)data_, size_).good())
        << " faild to write to " << filename;
  }

 private:
  xclDeviceHandle handle_;
  xclBufferHandle bo_;
  size_t size_;
  uint64_t phy_;
  void* data_;
};

static void preprocess(const std::vector<uint64_t>& data,
                       const std::vector<cv::Mat>& images, int width,
                       int height, float scale);
static void setImageBGR(const cv::Mat& image, void* data1, float scale);
static void run_user_specific_ip(const uint64_t input_phy_addr,
                                 const uint64_t output_phy_addr,
                                 const unsigned int cls,
                                 const unsigned int group, const int fixpos);
static int get_fix_pos(const xir::Tensor* tensor);
static const char* lookup(int index);
static std::vector<std::pair<int, float>> topk(const float* score, size_t size,
                                               int K);
static void print_topk(const std::string& image_file,
                       const std::vector<std::pair<int, float>>& topk);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name> <image_url>"
              << std::endl;
    abort();
  }

  // get xmodel file
  auto xmodel_file = std::string(argv[1]);
  std::cout << "xmodel file: " << xmodel_file << std::endl;

  // get image files
  std::vector<cv::Mat> images;
  std::vector<std::string> image_files;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      LOG(INFO) << "Can't load image: " << argv[i];
      continue;
    }
    images.emplace_back(img);
    image_files.emplace_back(argv[i]);
  }

  // get xrt handle
  auto handle = xclOpen(0, NULL, XCL_INFO);

  // create dpu runner
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

  // get input tensor information
  auto input_tensor_buffers = runner->get_inputs();
  auto input_tensor_shape = input_tensor_buffers[0]->get_tensor()->get_shape();
  auto input_scale =
      vart::get_input_scale(input_tensor_buffers[0]->get_tensor());
  auto batchsize =
      std::min((size_t)images.size(), (size_t)input_tensor_shape[0]);
  std::vector<uint64_t> in_batch_virt;
  for (auto i = 0; i < batchsize; i++) {
    std::vector<std::int32_t> idx{i, 0, 0, 0};
    uint64_t data = 0u;
    size_t size = 0u;
    std::tie(data, size) = input_tensor_buffers[0]->data(idx);
    in_batch_virt.emplace_back(data);
  }

  // only process the first batchsize images
  images.resize(batchsize);
  image_files.resize(batchsize);
  std::cout << "image files: " << std::endl;
  for (auto img : image_files) {
    std::cout << "\t" << img << std::endl;
  }

  // get output tensor information
  auto output_tensors = runner->get_output_tensors();
  auto out_tensor_shape = output_tensors[0]->get_shape();
  auto buffersize = output_tensors[0]->get_data_size() / out_tensor_shape[0];

  // create output tensor buffers
  std::vector<vart::TensorBuffer*> output_tensor_buffers;
  // simulate a host phy addr for output tensor
  uint64_t out_batch_addr[batchsize];
  std::vector<std::unique_ptr<Simulate_Host_Phy>> hp_out;
  hp_out.reserve(batchsize);
  for (auto i = 0; i < batchsize; i++) {
    hp_out.emplace_back(
        std::make_unique<Simulate_Host_Phy>(handle, buffersize));
    out_batch_addr[i] = hp_out[i]->get_host_phy_addr();
  }
  auto output_tb = vart::TensorBuffer::create_unowned_device_tensor_buffer(
      output_tensors[0], out_batch_addr, batchsize);
  output_tensor_buffers.emplace_back(output_tb.get());

  // postprocess by HW softmax runner
  // simulate a host phy addr for output tensor of softmax runner
  unsigned int cls = out_tensor_shape[1];
  auto sim_hp =
      std::make_unique<Simulate_Host_Phy>(handle, cls * sizeof(float));
  auto softmax_phy_addr = sim_hp->get_host_phy_addr();
  auto softmax_virt_data = sim_hp->get_host_virt_addr();
  int fixpos = get_fix_pos(output_tensors[0]);

  // images process
  preprocess(in_batch_virt, images, input_tensor_shape[2],
             input_tensor_shape[1], input_scale);

  // execute dpu runner
  for (auto& input : input_tensor_buffers) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
  }
  auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run dpu";

  for (auto i = 0; i < batchsize; i++) {
    //hp_out[i]->sync_data_for_read();
    //hp_out[i]->dump_data("image_" + std::to_string(i + 1) + "_softmax_in.bin");

    // create and execute softmax runner
    run_user_specific_ip(out_batch_addr[i], softmax_phy_addr, cls, 1u, fixpos);

    sim_hp->sync_data_for_read();
    //sim_hp->dump_data("image_" + std::to_string(i + 1) + "_softmax_out.bin");

    // get and print topk result
    auto topk_value = topk((float*)softmax_virt_data, cls, 5u);
    print_topk(image_files[i], topk_value);
  }

  xclClose(handle);

  return 0;
}

static void preprocess(const std::vector<uint64_t>& data,
                       const std::vector<cv::Mat>& images, const int width,
                       const int height, const float scale) {
  CHECK_LE(images.size(), data.size())
      << "number of images should not exceed count of data ptr";
  auto batch = data.size();
  for (auto i = 0; i < batch; i++) {
    cv::Mat reimage;
    auto size = cv::Size(width, height);
    if (size != images[i].size()) {
      cv::resize(images[i], reimage, size);
    } else {
      reimage = images[i];
    }
    setImageBGR(reimage, (void*)data[i], scale);
  }

  return;
}

static void setImageBGR(const cv::Mat& image, void* data1, float scale) {
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
  return;
}

static void run_user_specific_ip(const uint64_t input_phy_addr,
                                 const uint64_t output_phy_addr,
                                 const unsigned int cls,
                                 const unsigned int group, const int fixpos) {
  auto sfm_controller = xir::SfmController::get_instance();
  auto fmap_size = cls * group;
  sfm_controller->run_xrt_cu(0u, input_phy_addr, cls, group, fixpos,
                             output_phy_addr, 0u);

  return;
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

static void print_topk(const std::string& image_file,
                       const std::vector<std::pair<int, float>>& topk) {
  std::cout << "result for image " << image_file << ":" << std::endl;
  for (const auto& v : topk) {
    std::cout << std::setiosflags(std::ios::left) << std::setw(11)
              << "score[" + std::to_string(v.first) + "]"
              << " =  " << std::setw(12) << v.second
              << " text: " << lookup(v.first)
              << std::resetiosflags(std::ios::left) << std::endl;
  }
  return;
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
}

static int get_fix_pos(const xir::Tensor* tensor) {
  int fixpos = tensor->template get_attr<int>("fix_point");
  return fixpos;
}
