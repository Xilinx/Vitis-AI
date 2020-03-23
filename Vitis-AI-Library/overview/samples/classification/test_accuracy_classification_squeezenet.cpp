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
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <glog/logging.h>

#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/globalavepool.hpp>

using namespace std;

// Only kernel_h/kernel_w with 2*2 to 8*8 is supported when pool type
// is average pooling. The squeezenet needs to calculate the data of
// the ave pool is 14*14, we have to calculate by ARM.

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "Please input your image path as the first param!" << endl;
    cout << "The second param is a txt to store results!" << endl;
  }

  auto kernel_name = "squeezenet"+string("_acc");
  std::string path = argv[1];
  std::ofstream out_fs(argv[2], std::ofstream::out);
  vector<cv::String> files;
  cv::glob(path, files);
  int length = path.size();
  auto task = vitis::ai::DpuTask::create(kernel_name);
  task->setMeanScaleBGR({104.0f, 107.0f, 123.0f}, {1.0f, 1.0f, 1.0f});
  auto input_tensor = task->getInputTensor(0u);
  auto width = input_tensor[0].width;
  auto height = input_tensor[0].height;
  auto size = cv::Size(width, height);
  auto scale = vitis::ai::library::tensor_scale(task->getOutputTensor(0u)[0]);
  int count = files.size();
  auto cls = task->getOutputTensor(0u)[0].channel;
  auto ot_w = task->getOutputTensor(0u)[0].width;
  auto ot_h = task->getOutputTensor(0u)[0].height;

  cerr << "The image count = " << count << endl;

  for (int i = 0; i < count; i++) {
    auto input_image = imread(files[i]);
    cv::Mat image;
    if (size != input_image.size()) {
      cv::resize(input_image, image, size);
    } else {
      image = input_image;
    }
    task->setImageBGR(image);
    task->run(0u);
    auto pre_data = (int8_t*)task->getOutputTensor(0u)[0].get_data(0);
    std::vector<int8_t> data(cls);
    vitis::ai::globalAvePool(pre_data, cls, ot_w, ot_h, data.data());

    auto score = std::vector<float>(cls);
    auto sum = 0.0f;
    for (auto i = 0u; i < score.size(); ++i) {
      score[i] = expf(((float)data[i]) * scale);
      sum = sum + score[i];
    }
    for (auto i = 0u; i < score.size(); ++i) {
      score[i] = score[i] / sum;
    }
    auto indices = std::vector<int>(score.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&score](int a, int b) { return score[a] > score[b]; });

    for (size_t j = 0; j < 5; ++j) {
      out_fs << cv::String(files[i]).substr(length) << " " << indices[j]
             << endl;
    }
  }

  out_fs.close();
  return 0;
}
