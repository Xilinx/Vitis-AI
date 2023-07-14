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
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/globalavepool.hpp>

using namespace std;

// Only kernel_h/kernel_w with 2*2 to 8*8 is supported when pool type
// is average pooling. The squeezenet needs to calculate the data of
// the ave pool is 14*14, we have to calculate by ARM.

int main(int argc, char *argv[]) {
  // A kernel name, it must be samed as the dnnc result.
  auto kernel_name = "squeezenet_0";
  // A image file, e.g.
  // /usr/share/XILINX_AI_SDK/samples/classification/images/001.JPEG
  auto image_file_name = argv[1];
  // Create a dpu task object.
  auto task = vitis::ai::DpuTask::create(kernel_name);
  // Preprocessing, please check
  // /etc/dpu_model_param_.conf.d/inception_v1.prototxt or your caffe
  // model, e.g. deploy.prototxt
  task->setMeanScaleBGR({104.0f, 107.0f, 123.0f}, {1.0f, 1.0f, 1.0f});
  // Read image from a file
  auto input_image = cv::imread(image_file_name);
  if (input_image.empty()) {
    cerr << "cannot load " << argv[1] << endl;
    abort();
  }
  // Resize it if size is not match
  cv::Mat image;
  auto input_tensor = task->getInputTensor();
  CHECK_EQ(input_tensor.size(), 1) << " the dpu model must have only one input";
  auto width = input_tensor[0].width;
  auto height = input_tensor[0].height;
  auto size = cv::Size(width, height);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  // Set the input image
  task->setImageBGR(image);
  // Start the dpu
  task->run();
  // Get output.
  auto output_tensor = task->getOutputTensor();

  // Calc ave pool
  auto pre_data = (int8_t *)output_tensor[0].data;
  auto &cls = output_tensor[0].channel;
  auto &ot_w = output_tensor[0].width;
  auto &ot_h = output_tensor[0].height;

  std::vector<int8_t> data(cls);
  vitis::ai::globalAvePool(pre_data, cls, ot_w, ot_h, data.data());

  // Run softmax
  auto scale = vitis::ai::tensor_scale(output_tensor[0]);
  auto score = std::vector<float>(cls);
  auto sum = 0.0f;
  for (auto i = 0u; i < score.size(); ++i) {
    score[i] = expf(((float)data[i]) * scale);
    sum = sum + score[i];
  }
  for (auto i = 0u; i < score.size(); ++i) {
    score[i] = score[i] / sum;
  }

  // Sort to find the top-5 scores
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&score](int a, int b) { return score[a] > score[b]; });
  for (auto i = 0u; i < 5; ++i) {
    std::cout << "score[" << indices[i] << "]=" << score[indices[i]] << "\n";
  }
  return 0;
}
