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
#include <memory>
#include <queue>
#include <vitis/ai/benchmark.hpp>
#include <vitis/ai/classification.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/globalavepool.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/nnpp/classification.hpp>

#include "./find_model.hpp"

namespace vitis {
namespace ai {

static ClassificationResult topk(const float* softres, int channel, int k,
                                 int width, int height) {
  auto topkres = ClassificationResult{width, height};
  topkres.scores.reserve(k);
  std::priority_queue<std::pair<float, int>> q;
  for (int i = 0; i < channel; ++i) {
    q.push(std::pair<float, int>(softres[i], i));
  }

  for (int i = 0; i < k; ++i) {
    std::pair<float, int> maxprob = q.top();
    topkres.scores.emplace_back(
        ClassificationResult::Score{maxprob.second, softres[maxprob.second]});
    q.pop();
  }
  //  DLOG(INFO) << "topkres.size = " << topkres.scores.size();
  return topkres;
}

class ClassificationSqueezenet {
 public:
  ClassificationSqueezenet(std::string model)
      : kernel_name(model), task(DpuTask::create(find_model(kernel_name))) {}
  int getInputWidth() { return task->getInputTensor(0u)[0].width; }
  int getInputHeight() { return task->getInputTensor(0u)[0].height; }
  size_t get_input_batch() const { return task->get_input_batch(0, 0); }
  ClassificationResult classification_post_process1(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors) {
    auto top_k = 5;

    std::vector<int8_t> data(output_tensors[0].channel);
    globalAvePool((int8_t*)output_tensors[0].get_data(0),
                  output_tensors[0].channel, output_tensors[0].width,
                  output_tensors[0].height, data.data());

    std::vector<float> softres(output_tensors[0].channel);
    softmax(data.data(), tensor_scale(output_tensors[0]),
            output_tensors[0].channel, 1, &softres[0]);
    // std::cout << std::endl;
    return topk(&softres[0], output_tensors[0].channel, top_k,
                input_tensors[0].width, input_tensors[0].height);
  }

  std::vector<ClassificationResult> classification_post_process2(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors) {
    auto top_k = 5;
    std::vector<ClassificationResult> res;
    std::vector<int8_t> data(output_tensors[0].channel);
    for (size_t i = 0; i < output_tensors[0].batch; i++) {
      globalAvePool((int8_t*)output_tensors[0].get_data(i),
                    output_tensors[0].channel, output_tensors[0].width,
                    output_tensors[0].height, data.data());

      std::vector<float> softres(output_tensors[0].channel);
      softmax(data.data(), tensor_scale(output_tensors[0]),
              output_tensors[0].channel, 1, &softres[0]);
      // std::cout << std::endl;
      res.push_back(topk(&softres[0], output_tensors[0].channel, top_k,
                         input_tensors[0].width, input_tensors[0].height));
    }
    return res;
  }

  static void croppedImage(const cv::Mat& image, int height, int width,
                           cv::Mat& cropped_img) {
    int offset_h = (image.rows - height) / 2;
    int offset_w = (image.cols - width) / 2;
    cv::Rect box(offset_w, offset_h, width, height);
    cropped_img = image(box).clone();
  }

  std::vector<ClassificationResult> run(
      const std::vector<cv::Mat>& input_image) {
    std::vector<cv::Mat> images;

    for (const auto& i : input_image) {
      cv::Mat image;
      int width = getInputWidth();
      int height = getInputHeight();
      auto size = cv::Size(width, height);
      if (size != i.size()) {
        croppedImage(i, height, width, image);
      } else {
        image = i;
      }
      images.push_back(image);
    }

    task->setImageBGR(images);

    task->run(0u);

    auto ret = classification_post_process2(task->getInputTensor(0u),
                                            task->getOutputTensor(0u));
    return ret;
  }

  ClassificationResult run(const cv::Mat& input_image) {
    cv::Mat image;
    int width = getInputWidth();
    int height = getInputHeight();
    auto size = cv::Size(width, height);
    if (size != input_image.size()) {
      croppedImage(input_image, height, width, image);
    } else {
      image = input_image;
    }
    task->setImageBGR(image);

    task->run(0u);

    auto ret = classification_post_process1(task->getInputTensor(0u),
                                            task->getOutputTensor(0u));
    return ret;
  }

 private:
  std::string kernel_name;
  std::unique_ptr<DpuTask> task;
};
}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  std::string model = argv[1];
  return vitis::ai::main_for_performance(argc, argv, [model] {
    return std::unique_ptr<vitis::ai::ClassificationSqueezenet>(
        new vitis::ai::ClassificationSqueezenet(model));
  });
}
