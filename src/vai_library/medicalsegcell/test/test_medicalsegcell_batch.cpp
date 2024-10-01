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

#include <sys/stat.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/medicalsegcell.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name> <result_dir> <image_url> [<image_url> ...]"
              << std::endl;
    abort();
  }

  auto seg = vitis::ai::MedicalSegcell::create(argv[1] );
  if (!seg) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  std::vector<cv::Mat> arg_input_images;
  std::vector<cv::Size> arg_input_images_size;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 1; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_size.push_back(img.size());
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  std::vector<cv::Size> batch_images_size;
  auto batch = seg->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
    batch_images_size.push_back(
        arg_input_images_size[batch_idx % arg_input_images.size()]);
  }

  std::string path{argv[2]};
  auto ret = mkdir(path.c_str(), 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << path << std::endl;
    return -1;
  }

  auto result = seg->run(batch_images);

  cv::Mat img_save;
  for (auto batch_idx = 0u; batch_idx < result.size(); batch_idx++) {
    std::string filenamepart1 = batch_images_names[batch_idx].substr(
        batch_images_names[batch_idx].find_last_of('/') + 1);
    filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));

    std::string fname( path + "/" + filenamepart1 + "_result.png");
    std::cout << "fname :" << fname << "\n";
    cv::imwrite(fname, result[batch_idx].segmentation);
  }
  return 0;
}

