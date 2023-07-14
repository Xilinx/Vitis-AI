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
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "vitis/ai/graph_runner.hpp"

const std::vector<std::string> charactor_0 = {
	 "unknown", "jing", "hu", "jin", "yu", "ji", "jin", "meng", "liao", "ji",
         "hei", "su", "zhe", "wan", "min", "gan",
         "lu", "yu", "e", "xiang", "yue", "gui", "qiong", "chuan", "gui", "yun",
         "zang", "shan", "gan", "qing", "ning", "xin"};

const std::vector<std::string> charactor_1 = {
	 "unknown", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L",
         "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

const std::vector<std::string> charactor_2 = {
	 "unknown", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
         "C", "D", "E",
         "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
         "V", "W", "X", "Y", "Z"};

const std::vector<std::string> color = {"Blue", "Yellow"};

static int get_fix_point(const xir::Tensor* tensor);
static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor);
static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
                                        size_t batch);
static void set_input_image(const cv::Mat& image, void* data1, float scale);
static std::vector<std::pair<int, float>> topk(void* data1, size_t size,
                                               int K);
static size_t find_tensor_index(const char* tensor_name,
		                         const std::vector<vart::TensorBuffer*>& outputs);
//platenum preprocess
static void preprocess_platenum(const std::vector<std::string>& files, 
		                const std::vector<vart::TensorBuffer*>& input_tensor_buffers) {
  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto batch = input_tensor->get_shape().at(0);
  auto height = input_tensor->get_shape().at(1);
  auto width = input_tensor->get_shape().at(2);

  int fixpos = get_fix_point(input_tensor);
  float input_fixed_scale = std::exp2f(1.0f * (float)fixpos);

  auto size = cv::Size(width, height);
  auto images = read_images(files, batch);
  CHECK_EQ(images.size(), batch) 
	  << "images number be read into input buffer must be equal to batch";

  for (int index = 0; index < batch; ++index) {
    cv::Mat resize_image;
    if (size != images[index].size()) {
      cv::resize(images[index], resize_image, size, 0);
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

//platenum postprocess
static void postprocess_platenum(const std::vector<vart::TensorBuffer*>& output_tensor_buffers) {
  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto batch = output_tensor->get_shape().at(0);
  auto size = output_tensor_buffers.size();
  CHECK_EQ(size, 8) << "output_tensor_buffers.size() must be 8";
  for (auto i = 1u; i < size; ++i) {
    CHECK_EQ(output_tensor_buffers[i]->get_tensor()->get_shape().at(0), batch) 
	    << "all output_tensor_buffer batch number must be equal";
  }
  std::vector<std::pair<int, float>> ret;
  for (int batch_index = 0; batch_index < batch; ++batch_index) {
    for (auto tb_index = 0u; tb_index < size; ++tb_index) {
      uint64_t data_out = 0u;
      size_t size_out = 0u;
      auto idx = get_index_zeros(output_tensor_buffers[tb_index]->get_tensor());
      idx[0] = (int)batch_index;
      std::tie(data_out, size_out) = output_tensor_buffers[tb_index]->data(idx);
      auto elem_num = output_tensor_buffers[tb_index]->get_tensor()->get_element_num() / batch;
      auto tb_top1 = topk((void*)data_out, elem_num, 1)[0];
      ret.push_back(tb_top1);
    }
  }
  for (int batch_index = 0; batch_index < batch; ++batch_index) {
    std::string plate_number = "";
    std::string plate_color = "";
    //output_tensor_buffers maybe out of order, need find correct output_tensor_buffer result by tensor name
    plate_number += charactor_0[ret[batch_index * size + find_tensor_index("prob1", output_tensor_buffers)].first];
    plate_number += charactor_1[ret[batch_index * size + find_tensor_index("prob2", output_tensor_buffers)].first];
    plate_number += charactor_2[ret[batch_index * size + find_tensor_index("prob3", output_tensor_buffers)].first];
    plate_number += charactor_2[ret[batch_index * size + find_tensor_index("prob4", output_tensor_buffers)].first];
    plate_number += charactor_2[ret[batch_index * size + find_tensor_index("prob5", output_tensor_buffers)].first];
    plate_number += charactor_2[ret[batch_index * size + find_tensor_index("prob6", output_tensor_buffers)].first];
    plate_number += charactor_2[ret[batch_index * size + find_tensor_index("prob7", output_tensor_buffers)].first];
    plate_color = color[ret[batch_index * size + find_tensor_index("prob8", output_tensor_buffers)].first];
    std::cout << "batch_index: " << batch_index << std::endl;
    std::cout << "plate_color: " << plate_color << std::endl;
    std::cout << "plate_number: " << plate_number << std::endl;
  }
}

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
  preprocess_platenum(g_image_files, input_tensor_buffers);

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

  //postprocess and print platenum result
  postprocess_platenum(output_tensor_buffers);

  return 0;
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

static std::vector<cv::Mat> read_images(const std::vector<std::string>& files,
		                                        size_t batch) {
  std::vector<cv::Mat> images(batch);
  for (auto index = 0u; index < batch; ++index) {
    const auto& file = files[index % files.size()];
    images[index] = cv::imread(file);
    CHECK(!images[index].empty()) << "cannot read image from " << file;
  }
  return images;
}

static void set_input_image(const cv::Mat& image, void* data1, float scale) {
  float mean[3] = {128.0, 128.0, 128.0};
  signed char* data = (signed char*)data1;
  for (int h = 0; h < image.rows; h++) {
    for (int w = 0; w < image.cols; w++) {
      for (int c = 0; c < 3; c++) {
        auto image_data = (image.at<cv::Vec3b>(h, w)[c] - mean[c]) * scale;
        image_data = std::max(std::min(image_data, 127.0f), -128.0f);
        data[h * image.cols * 3 + w * 3 + c] = (int)image_data;
      }
    }
  }
}

static std::vector<std::pair<int, float>> topk(void* data1, size_t size,
		                                               int K) {
  const float* score = (const float*)data1;
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

static size_t find_tensor_index(const char* tensor_name,
                         const std::vector<vart::TensorBuffer*>& outputs) {
  auto it = std::find_if(outputs.begin(), outputs.end(),
                         [&tensor_name](const vart::TensorBuffer* tb) {
                         return tb->get_tensor()->get_name() == tensor_name;
                        });
  CHECK(it != outputs.end()) << "cannot find tensorbuffer. tensor_name=" << tensor_name;
  return it - outputs.begin();
}
