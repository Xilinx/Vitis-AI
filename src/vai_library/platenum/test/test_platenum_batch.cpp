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
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/platenum.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
using namespace vitis::ai;
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }
  auto det = vitis::ai::PlateNum::create(argv[1], true);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  int width = det->getInputWidth();
  int height = det->getInputHeight();
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;
  std::vector<cv::Mat> arg_input_images;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  size_t batch = det->get_input_batch();
  for (size_t batch_idx = 0; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

  vector<vitis::ai::PlateNumResult> nres = det->run(batch_images);

  // 86 233 156 258
  for (size_t i = 0; i < batch; i++) {
    std::cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
              << "]"  //                            //
              << "result.width " << nres[i].width << " "                //
              << "result.height " << nres[i].height << " "              //
              << "result.plate_color " << nres[i].plate_color << " "    //
              << "result.plate_number " << nres[i].plate_number << " "  //
              << std::endl;
  }
  return 0;
}
