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

#include <math.h>

#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/arflow.hpp>

using namespace std;
using namespace cv;
std::vector<float> convert_fixpoint_to_float(
    const vitis::ai::library::OutputTensor& tensor) {
  auto scale = vitis::ai::library::tensor_scale(tensor);
  auto data = (signed char*)tensor.get_data(0);
  auto size = tensor.width * tensor.height * tensor.channel;
  auto ret = std::vector<float>(size);
  transform(data, data + size, ret.begin(),
            [scale](signed char v) { return ((float)v) * scale; });
  return ret;
}

cv::Mat resize_flow(const cv::Mat& flow, cv::Size shape) {
  cv::Mat resize_flow;
  cv::resize(flow, resize_flow, shape);
  std::vector<float> scale{float(flow.cols) / float(shape.width),
                           float(flow.rows) / float(shape.height)};
  resize_flow.forEach<cv::Point2f>(
      [scale](cv::Point2f& val, const int* position) {
        val.x /= scale[0];
        val.y /= scale[1];
      });

  return resize_flow;
}
void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
  int i = h * 6.0;
  float f = (h * 6.0) - i;
  float p = v * (1.0 - s);
  float q = v * (1.0 - s * f);
  float t = v * (1.0 - s * (1.0 - f));
  if (i % 6 == 0) {
    r = v;
    g = t;
    b = p;
  } else if (i == 1) {
    r = q;
    g = v;
    b = p;
  } else if (i == 2) {
    r = p;
    g = v;
    b = t;
  } else if (i == 3) {
    r = p;
    g = q;
    b = v;
  } else if (i == 4) {
    r = t;
    g = p;
    b = v;
  } else if (i == 5) {
    r = v;
    g = p;
    b = q;
  }
  if (std::abs(s) < DBL_EPSILON) {
    r = v;
    g = v;
    b = v;
  }
}

cv::Mat flow_to_image(cv::Mat flow) {
  float max_flow = 256.;
  float n = 8.;
  auto src = cv::Mat(flow.size(), CV_8UC3);
  auto input = (float*)flow.data;
  auto data = src.data;
  auto height = flow.size().height;
  auto width = flow.size().width;
  for (auto h = 0; h < height; h++) {
    for (auto w = 0; w < width; w++) {
      auto u = input[h * width * 2 + w * 2 + 0];
      auto v = input[h * width * 2 + w * 2 + 1];

      auto mag = sqrt(u * u + v * v);
      auto angle = atan2(v, u);
      auto im_h = angle / (2 * M_PI) + 1 - (int)(angle / (2 * M_PI) + 1);
      auto im_s = std::min(std::max(mag * n / max_flow, 0.f), 1.f);
      auto im_v = std::min(std::max(n - im_s, 0.f), 1.f);
      float r, g, b;
      hsv_to_rgb(im_h, im_s, im_v, r, g, b);

      data[h * width * 3 + w * 3 + 2] = r * 255.;
      data[h * width * 3 + w * 3 + 1] = g * 255.;
      data[h * width * 3 + w * 3 + 0] = b * 255.;
    }
  }

  return src;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cerr << "usage: " << argv[0]
         << "<model_name> <image_0_file_url> <image_1_file_url> " << endl;
    abort();
  }
  Mat img_0 = cv::imread(argv[2]);
  Mat img_1 = cv::imread(argv[3]);
  if (img_0.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }
  if (img_1.empty()) {
    cerr << "cannot load " << argv[3] << endl;
    abort();
  }
  auto arflow = vitis::ai::ARFlow::create(argv[1], true);
  if (!arflow) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  auto result = arflow->run(img_0, img_1)[0];
  auto data = convert_fixpoint_to_float(result);
  auto flow_Mat = cv::Mat(result.height, result.width, CV_32FC2, data.data());
  auto res = flow_to_image(resize_flow(flow_Mat, img_0.size()));
  imwrite(string(argv[1]) + "_result.png", res);
  cout << "The result is written in " << string(argv[1]) + "_result.png"
       << endl;

  return 0;
}
