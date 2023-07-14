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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#if _WIN32
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif
#include <vitis/ai/profiling.hpp>
#include "./rcan_onnx.hpp"
#include "vitis/ai/env_config.hpp"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> " << std::endl;
    abort();
  }
#if _WIN32
  auto model_name = strconverter.from_bytes(std::string(argv[1]));
#else
  auto model_name = std::string(argv[1]);
#endif
  cv::Mat image = cv::imread(argv[2]);
  CHECK(!image.empty()) << "cannot read image from " << argv[2];

  auto rcan_onnx = RcanOnnx::create(model_name);

  auto batch = rcan_onnx->get_input_batch();
  std::vector<cv::Mat> images(batch);
  for (auto i = 0u; i < batch; ++i) {
    image.copyTo(images[i]);
  }
  __TIC__(ONNX_RUN)
  auto results = rcan_onnx->run(images);
  __TOC__(ONNX_RUN)

  for (auto i = 0u; i < results.size(); ++i) {
    std::cout << "batch " << i << std::endl;
    cv::imwrite(std::string("batch_") + std::to_string(i) + ".jpg",
                results[i].feat);
  }
  return 0;
}

