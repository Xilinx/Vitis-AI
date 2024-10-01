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
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/globalavepool.hpp>

#include "./find_model.hpp"

using namespace std;
using namespace cv;
// Only kernel_h/kernel_w with 2*2 to 8*8 is supported when pool type
// is average pooling. The squeezenet needs to calculate the data of
// the ave pool is 14*14, we have to calculate by ARM.

static void croppedImage(const cv::Mat& image, int height, int width,
                         cv::Mat& cropped_img) {
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
}

static void preprocess(const cv::Mat& image, int height, int width,
                       cv::Mat& pro_res) {
  float smallest_side = 256;
  float scale =
      smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(image.cols * scale, image.rows * scale));
  croppedImage(resized_image, height, width, pro_res);
}

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
  if (argc < 4) {
    cout << "First param is the model name." << endl
         << "Second param is your image path." << endl
         << "Third param is a txt to store results!" << endl;
  }
  string model = string(argv[1]);
  auto kernel_name = model + string("_acc");
  std::string path = argv[2];
  std::ofstream out_fs(argv[3], std::ofstream::out);
  vector<cv::String> files;
  cv::glob(path, files);
  int length = path.size();

  auto task = vitis::ai::DpuTask::create(find_model(kernel_name));
  if (!task) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  if (model.compare("squeezenet_pt") == 0)
    task->setMeanScaleBGR({103.53f, 116.28f, 123.675f},
                          {0.017429f, 0.017507f, 0.01712475f});
  else if (model.compare("squeezenet") == 0)
    task->setMeanScaleBGR({104.0f, 107.0f, 123.0f}, {1.0f, 1.0f, 1.0f});
  else
    cout << "Model name should be squeezenet or squeezenet_pt" << endl;

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
      preprocess(input_image, height, width, image);
    } else {
      image = input_image;
    }
    if (model.compare("squeezenet_pt") == 0)
      task->setImageRGB(image);
    else if (model.compare("squeezenet") == 0)
      task->setImageBGR(image);
    task->run(0u);
    auto pre_data = (int8_t*)task->getOutputTensor(0u)[0].get_data(0);
    std::vector<float> data(cls);
    if (model.compare("squeezenet_pt") == 0) {
      AvePool(pre_data, cls, ot_w, ot_h, data.data());
    } else {
      std::vector<int8_t> data_i(cls);
      vitis::ai::globalAvePool(pre_data, cls, ot_w, ot_h, data_i.data());
      data = std::vector<float>(data_i.begin(), data_i.end());
    }
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
