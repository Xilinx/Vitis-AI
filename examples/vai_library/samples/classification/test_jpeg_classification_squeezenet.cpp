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

#include "./find_model.hpp"

using namespace std;

// Only kernel_h/kernel_w with 2*2 to 8*8 is supported when pool type
// is average pooling. The squeezenet needs to calculate the data of
// the ave pool is 14*14, we have to calculate by ARM.

float get_avgpool_dpu_coefficient(const std::vector<std::int32_t>& kernels) {
  auto rec = kernels[0] * kernels[1];
  float multi_factor = 0;
  float shift_factor = 0;
  auto diff = 1.f;
  auto max_factor = std::ceil(std::log2(rec * 128));
  for (auto shift_factor_ = 0; shift_factor_ < max_factor; shift_factor_++) {
    auto factor = std::round(std::exp2(shift_factor_) / rec);
    auto diff_ = std::abs(factor / std::exp2(shift_factor_) - 1.f / rec);
    if (diff_ < diff) {
      multi_factor = factor;
      diff = diff_;
      shift_factor = shift_factor_;
    }
  }
  return multi_factor / std::exp2(shift_factor);
}
void AvePool(int8_t* src, int channel, int width, int height, float* dst) {
  auto scale = get_avgpool_dpu_coefficient({width, height});
  float sum;
  for (int i = 0; i < channel; i++) {
    sum = 0.0f;
    for (int j = 0; j < width * height; j++) {
      sum += src[i + channel * j];
    }
    dst[i] = sum * scale;
  }
}

int main(int argc, char* argv[]) {
  // A kernel name, it must be samed as the dnnc result.
  string kernel_name = argv[1];
  // A image file, e.g.
  // /usr/share/XILINX_AI_SDK/samples/classification/images/001.JPEG
  auto image_file_name = argv[2];
  // Create a dpu task object.
  auto task = vitis::ai::DpuTask::create(find_model(kernel_name));
  if (!task) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  // Preprocessing, please check
  // /etc/dpu_model_param_.conf.d/inception_v1.prototxt or your caffe
  // model, e.g. deploy.prototxt
  // Read image from a file
  auto input_image = cv::imread(image_file_name);
  if (input_image.empty()) {
    cerr << "cannot load " << image_file_name << endl;
    abort();
  }
  // Resize it if size is not match
  cv::Mat image;
  auto input_tensor = task->getInputTensor(0u);
  // CHECK_EQ((int)input_tensor.size(), 1)
  //    << " the dpu model must have only one input";
  auto width = input_tensor[0].width;
  auto height = input_tensor[0].height;
  auto size = cv::Size(width, height);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  // Set the input image
  if (kernel_name.compare("squeezenet_pt") == 0) {
    task->setMeanScaleBGR({103.53f, 116.28f, 123.675f},
                          {0.017429f, 0.017507f, 0.01712475f});
    task->setImageRGB(image);
  } else if (kernel_name.compare("squeezenet") == 0) {
    task->setMeanScaleBGR({104.0f, 107.0f, 123.0f}, {1.0f, 1.0f, 1.0f});
    task->setImageBGR(image);
  } else
    cout << "Model name should be squeezenet or squeezenet_pt" << endl;
  // Start the dpu
  task->run(0u);
  // Get output.
  auto output_tensor = task->getOutputTensor(0u);

  // Calc ave pool
  auto pre_data = (int8_t*)output_tensor[0].get_data(0);
  auto& cls = output_tensor[0].channel;
  auto& ot_w = output_tensor[0].width;
  auto& ot_h = output_tensor[0].height;
  std::vector<float> data(cls);
  if (kernel_name.compare("squeezenet_pt") == 0) {
    AvePool(pre_data, cls, ot_w, ot_h, data.data());
  } else {
    std::vector<int8_t> data_i(cls);
    vitis::ai::globalAvePool(pre_data, cls, ot_w, ot_h, data_i.data());
    data = std::vector<float>(data_i.begin(), data_i.end());
  }
  // Run softmax
  auto scale = vitis::ai::library::tensor_scale(output_tensor[0]);
  auto score = std::vector<float>(cls);
  auto sum = 0.0f;
  for (auto i = 0u; i < score.size(); ++i) {
    score[i] = expf(((float)data[i]) * scale);
    sum = sum + score[i];
  }
  for (auto i = 0u; i < score.size(); ++i) {
    score[i] = score[i] / sum;
  }
  string s = argv[1];
  s = s.substr(s.find_last_of("/") + 1);
  // Sort to find the top-5 scores
  auto indices = std::vector<int>(score.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&score](int a, int b) { return score[a] > score[b]; });
  for (auto i = 0u; i < 5; ++i) {
    std::cout << s << " " << indices[i] << " " << score[indices[i]] << "\n";
  }
  return 0;
}
