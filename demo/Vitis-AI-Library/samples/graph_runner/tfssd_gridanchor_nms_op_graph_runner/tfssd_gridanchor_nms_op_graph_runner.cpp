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
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/graph_runner.hpp"

// env var control the nms confidence threshold: 0.001 for accuracy test

using namespace std;

const float scale_0 = 0.00784314; 
std::vector<cv::Mat> images;

static void preprocess_tfssdop(const std::vector<std::string>& files, 
		                const std::vector<vart::TensorBuffer*>& input_tensor_buffers);
static void postprocess_tfssdop(const std::vector<vart::TensorBuffer*>& output_tensor_buffers);
static int get_fix_point(const xir::Tensor* tensor);
static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor);
static void read_images(const std::vector<std::string>& files, size_t batch);
static void set_input_image(const cv::Mat& image, void* data1, float scale);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }
  std::string g_xmodel_file = std::string(argv[1]);
  std::vector<std::string> g_image_files;
  for (auto i = 2; i < argc; i++) {
    g_image_files.push_back(std::string(argv[i]));
  }

  //create graph runner
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  auto attrs = xir::Attrs::create();
  auto runner =
      vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
  CHECK(runner != nullptr);

  //get input/output tensor buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();

  //preprocess and fill input
  preprocess_tfssdop(g_image_files, input_tensor_buffers);
  //sync input tensor buffers
  for (auto& input : input_tensor_buffers) {
      input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                   input->get_tensor()->get_shape()[0]);
    }

  //run graph runner
  auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run the graph";

  //sync output tensor buffers
  for (auto output : output_tensor_buffers) {
      output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                   output->get_tensor()->get_shape()[0]);
    }

  //postprocess and print tfssdop result
  postprocess_tfssdop(output_tensor_buffers);

  return 0;
}

static void preprocess_tfssdop(const std::vector<std::string>& files, 
		                const std::vector<vart::TensorBuffer*>& input_tensor_buffers) {
  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto batch = input_tensor->get_shape().at(0);
  auto height = input_tensor->get_shape().at(1);
  auto width = input_tensor->get_shape().at(2);

  int fixpos = get_fix_point(input_tensor);
  float input_fixed_scale = std::exp2f(1.0f * (float)fixpos);   (void)input_fixed_scale;

  auto size = cv::Size(width, height);
  images.resize(batch);
  read_images(files, batch);
  CHECK_EQ(images.size(), batch) 
	  << "images number be read into input buffer must be equal to batch";

  for (int index = 0; index < batch; ++index) {
    cv::Mat resize_image;
    if (size != images[index].size()) {
      cv::resize(images[index], resize_image, size);
    } else {
      images[index].copyTo(resize_image);
    }
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    auto idx = get_index_zeros(input_tensor);
    idx[0] = (int)index;
    std::tie(data_in, size_in) = input_tensor_buffers[0]->data(idx);

    set_input_image(resize_image, (void*)data_in, input_fixed_scale);
  }
}

//tfssdop postprocess
static void postprocess_tfssdop(const std::vector<vart::TensorBuffer*>& output_tensor_buffers) {

  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto batch = output_tensor->get_shape().at(0);
  auto size = output_tensor_buffers.size();
  CHECK_EQ(size, 1) << "output_tensor_buffers.size() must be 1";

  for (int batch_index = 0; batch_index < batch; ++batch_index) {
      uint64_t data_out = 0u;
      size_t size_out = 0u;
      auto idx = get_index_zeros(output_tensor);
      idx[0] = (int)batch_index;
      std::tie(data_out, size_out) = output_tensor_buffers[0]->data(idx);
      float* fpout = (float*)data_out;  (void)fpout;
      for(unsigned int i=0; i<size_out/(6*sizeof(float)); i++) {
         std::cout <<fpout[0] << "     " << fpout[1] << "  "
                   << fpout[2] << "  " << fpout[3] << "  " << fpout[4] << "      " << fpout[5] << "\n";
         fpout+=6;
      }
  }
}

static int get_fix_point(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point"))
        << "get tensor fix_point error! has no fix_point attr, tensor name is "
        << tensor->get_name();
  return tensor->template get_attr<int>("fix_point");
}

static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}

void read_images(const std::vector<std::string>& files,
		                                        size_t batch) {
  for (auto index = 0u; index < batch; ++index) {
    const auto& file = files[index % files.size()];
    images[index] = cv::imread(file);
    CHECK(!images[index].empty()) << "cannot read image from " << file;
  }
}

static void set_input_image(const cv::Mat& image, void* data1, float scale) {
  float mean[3] = {127.5, 127.5, 127.5 };
  signed char* data = (signed char*)data1;
  scale*=scale_0;
  for (int h = 0; h < image.rows; h++) {
    for (int w = 0; w < image.cols; w++) {
      for (int c = 0; c < 3; c++) {
        auto image_data = (image.at<cv::Vec3b>(h, w)[c] - mean[c]) * scale;
        image_data = std::max(std::min(image_data, 127.0f), -128.0f);
        data[h * image.cols * 3 + w * 3 + abs(c-2) ] = (int)image_data;
      }
    }
  }
}



