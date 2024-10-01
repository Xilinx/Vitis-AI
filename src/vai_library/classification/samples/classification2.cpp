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

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/configurable_dpu_task.hpp>

using namespace std;
class MyInterface {
 public:
  virtual int getInputWidth() const = 0;
  virtual int getInputHeight() const = 0;
};
class MyClassification : public vitis::ai::TConfigurableDpuTask<MyInterface> {
 public:
  MyClassification(const std::string &model_name, bool need_preprocess)
      : vitis::ai::TConfigurableDpuTask<MyInterface>(model_name,
                                                     need_preprocess),
        TOP_K{configurable_dpu_task_->getConfig()
                  .classification_param()
                  .top_k()} {
    LOG(INFO) << "TOP_K is " << TOP_K;
  }
  virtual ~MyClassification(){};
  std::array<std::pair<int, float>, 5> run(const cv::Mat &input_image) {
    cv::Mat image;
    int width = getInputWidth();
    int height = getInputHeight();
    auto size = cv::Size(width, height);
    if (size != input_image.size()) {
      cv::resize(input_image, image, size, 0);
    } else {
      image = input_image;
    }
    // Set the input image
    configurable_dpu_task_->setInputImageBGR(image);
    // Start the dpu
    configurable_dpu_task_->run(0);
    // Run softmax
    auto output_tensor = configurable_dpu_task_->getOutputTensor()[0];
    // Run softmax
    auto scale = vitis::ai::tensor_scale(output_tensor[0]);
    auto data = (signed char *)output_tensor[0].data;
    auto cls = output_tensor[0].channel;
    auto score = std::vector<float>(cls);
    auto sum = 0.0f;
    for (auto i = 0u; i < score.size(); ++i) {
      score[i] = expf(((float)data[i]) * scale);
      sum = sum + score[i];
    }
    for (auto i = 0u; i < score.size(); ++i) {
      score[i] = score[i] / sum;
    }
    // Print top-5
    auto indices = std::vector<int>(score.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&score](int a, int b) { return score[a] > score[b]; });
    auto ret = std::array<std::pair<int, float>, 5>{};
    for (auto i = 0u; i < 5; ++i) {
      ret[i] = std::make_pair(indices[i], score[indices[i]]);
    }
    return ret;
  }

 private:
  int TOP_K;  // un-used, just for an example how to read it from a
              // configuration.
};

int main(int argc, char *argv[]) {
  // A kernel name, e.g. resnet_50, inception_v1_0, inception_v2_0,
  // inception_v3_0, etc
  auto model_name = "inception_v1";
  // A image file, e.g.
  // /usr/share/XILINX_AI_SDK/samples/classification/images/001.JPEG
  auto image_file_name = argv[1];
  auto need_preprocess = true;
  auto task = std::unique_ptr<MyClassification>(
      new MyClassification(model_name, need_preprocess));
  // Read image from a file
  auto input_image = cv::imread(image_file_name);
  auto result = task->run(input_image);
  for (auto i = 0u; i < 5; ++i) {
    std::cout << "score[" << result[i].first << "]=" << result[i].second
              << "\n";
  }
  return 0;
}
