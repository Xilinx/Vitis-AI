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
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vart/dpu/dpu_runner_ext.hpp>
#include <vart/dpu/vitis_dpu_runner_factory.hpp>
#include <xir/tensor/tensor.hpp>

static cv::Mat preprocess_image(cv::Mat input_image, cv::Size size);

static std::vector<float> convert_fixpoint_to_float(
    vart::TensorBuffer* tensor, float scale, int batch_idx);

static std::vector<float> softmax(const std::vector<float>& input);

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K);

static std::vector<std::vector<std::pair<int, float>>> post_process(
    vart::TensorBuffer* tensor_buffer, float scale);

static void setImageBGR(const cv::Mat& image, int8_t * data1,
                        float scale_fix2float);

int main(int argc, char* argv[]) {
  {
    // a model name, e.g. /usr/share/vitis_ai_library/models/resnet50/resnet50.elf
    string model_path = argv[1];
    // a kernel name, e.g. resnet50_0
    string kernel_name = argv[2];

    // create runner , input/output tensor buffer ;
    auto runner_base = 
      vart::dpu::DpuRunnerFactory::create_dpu_runner(model_path, kernel_name);
    auto runner = dynamic_cast<vart::dpu::DpuRunnerExt*>(runner_base.get());
    auto input_scale = runner->get_input_scale();
    auto output_scale = runner->get_output_scale();

    // load the image
    std::vector<cv::Mat> imgs;
    std::vector<string> imgs_names;
    for (int i = 3; i < argc; i++) {
      // image file names, e.g.
      // /usr/share/vitis_ai_library/demo/classification/demo_classification.jpg
      auto img = cv::imread(argv[i]);
      if (img.empty()) {
        cout << "Cannot load " << argv[i] << endl;
        continue;
      }
      imgs.push_back(img);
      imgs_names.push_back(argv[i]);
    }
    if (imgs.empty()) {
      cerr << "No image load success!" << endl;
      abort();
    }
    // get input tensor buffer
    auto input_tensors = runner->get_input_tensors();
    CHECK_EQ(input_tensors.size(), 1u) << "only support classification model";
    auto input_tensor = input_tensors[0];
    auto height = input_tensor->get_dim_size(1);
    auto width = input_tensor->get_dim_size(2);
    auto batch = input_tensor->get_dim_size(0);
    auto input_size = cv::Size(width, height);
    auto inputs = runner->get_inputs();
    auto outputs = runner->get_outputs();
    int j = 0 ;
    for (long unsigned int i = 0; i < imgs.size(); i++) {
      // preprocess, i.e. resize if necessary
      cv::Mat image = preprocess_image(imgs[i], input_size);
      // set the input image and preprocessing
      int8_t * data_in = (int8_t *)inputs[0]->data({j, 0, 0, 0}).first;
      setImageBGR(image, data_in, input_scale[0]);
      j++;
      if (j < batch && i < imgs.size() - 1) {
        continue;
      }

      auto v = runner->execute_async(inputs, outputs);
      auto status = runner->wait((int)v.first, -1);
      CHECK_EQ(status, 0) << "failed to run dpu";
      // post process
      auto topk = post_process(outputs[0], output_scale[0]);
      // print the result
      for (int k = 0; k < j; k++){
        cout << "batch_index " << k << " "                    //
          << "image_name " << imgs_names[i - j + 1 + k] << " " //
          << std::endl;
        // print top k
        for (const auto& v : topk[k]) {
          cout << "score[" << v.first << "] = " << v.second << endl;
        }
      }
      j = 0;
    }
  }
  cout << "bye" << endl;
  return 0;
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

static std::vector<std::vector<std::pair<int, float>>> post_process(
    vart::TensorBuffer* tensor_buffer, float scale) {
  // run softmax
  std::vector<std::vector<std::pair<int, float>>> res;
  constexpr int TOPK = 5;
  for (auto i = 0; i < tensor_buffer->get_tensor()->get_dim_size(0); i++){
    auto softmax_input = convert_fixpoint_to_float(tensor_buffer, scale , i);
    auto softmax_output = softmax(softmax_input);
    res.push_back(topk(softmax_output, TOPK));
  }
  return res;              
}

static std::vector<float> convert_fixpoint_to_float(
    vart::TensorBuffer* tensor_buffer, float scale, int batch_idx) {
  //convert fixpoint to float 
  uint64_t data = 0u;
  size_t size = 0u;
  std::tie(data, size) = tensor_buffer->data({0, 0, 0, 0});
  auto batch = tensor_buffer->get_tensor()->get_dim_size(0);
  int8_t * data_c = (int8_t *)data;
  size /= batch; 
  std::vector<float> ret(size);
  std::transform(data_c + batch_idx * size , data_c + (batch_idx + 1)* size,
    ret.begin(), [scale](signed char v) { return ((float)v) * scale; });
  return ret;
}

static std::vector<float> softmax(const std::vector<float>& input) {
  // implement softmax
  auto output = std::vector<float>(input.size());
  std::transform(input.begin(), input.end(), output.begin(), expf);
  auto sum = accumulate(output.begin(), output.end(), 0.0f, std::plus<float>());
  std::transform(output.begin(), output.end(), output.begin(),
                 [sum](float v) { return v / sum; });
  return output;
}

static std::vector<std::pair<int, float>> topk(const std::vector<float>& score,
                                               int K) {
  //find top k
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [&score](int a, int b) { return score[a] > score[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  std::transform( indices.begin(), indices.begin() + K, ret.begin(),
      [&score](int index) { return make_pair(index, score[index]); });
  return ret;
}

static void setImageBGR(const cv::Mat& image, int8_t * data,
                        float scale_fix2float) {
  //preprocess and set the input image 
  int c = 0;
  for (auto row = 0; row < image.rows; row++) {
    for (auto col = 0; col < image.cols; col++) {
      auto v = image.at<cv::Vec3b>(row, col);
      auto B = (float)v[0];
      auto G = (float)v[1];
      auto R = (float)v[2];
      auto nB = (B - 104.0f) * 1.0f * scale_fix2float;
      auto nG = (G - 117.0f) * 1.0f * scale_fix2float;
      auto nR = (R - 123.0f) * 1.0f * scale_fix2float;
      nB = std::max(std::min(nB, 127.0f), -128.0f);
      nG = std::max(std::min(nG, 127.0f), -128.0f);
      nR = std::max(std::min(nR, 127.0f), -128.0f);
      data[c++] = (int)(nB);
      data[c++] = (int)(nG);
      data[c++] = (int)(nR);
    }
  }
}
