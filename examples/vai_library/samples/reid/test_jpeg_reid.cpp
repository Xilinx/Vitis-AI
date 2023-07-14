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

#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/reid.hpp>

DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");
DEF_ENV_PARAM(SAMPLES_BATCH_NUM, "0");

using namespace std;
using namespace cv;

double cosine_distance(Mat feat1, Mat feat2) { return 1 - feat1.dot(feat2); }

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cerr << "need at least two images" << endl;
    return -1;
  }
  auto model_name = argv[1];
  auto det = vitis::ai::Reid::create(model_name);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  if (ENV_PARAM(SAMPLES_ENABLE_BATCH)) {
    std::vector<std::string> image_x_files;
    std::vector<std::string> image_y_files;
    for (int i = 2; i < argc; i = i + 2) {
      image_x_files.push_back(std::string(argv[i]));
    }
    for (int i = 3; i < argc; i = i + 2) {
      image_y_files.push_back(std::string(argv[i]));
    }
    if (image_x_files.empty() || image_y_files.empty()) {
      std::cerr << "no input file" << std::endl;
      exit(1);
    }
    if (image_x_files.size() != image_y_files.size()) {
      std::cerr << "input images should be pair" << std::endl;
      exit(1);
    }

    auto batch = det->get_input_batch();
    if (ENV_PARAM(SAMPLES_BATCH_NUM)) {
      unsigned int batch_set = ENV_PARAM(SAMPLES_BATCH_NUM);
      assert(batch_set <= batch);
      batch = batch_set;
    }

    std::vector<std::string> batch_x_files(batch);
    std::vector<cv::Mat> x_images(batch);
    for (auto index = 0u; index < batch; ++index) {
      const auto& file = image_x_files[index % image_x_files.size()];
      batch_x_files[index] = file;
      x_images[index] = cv::imread(file);
      CHECK(!x_images[index].empty()) << "cannot read image from " << file;
    }
    std::vector<std::string> batch_y_files(batch);
    std::vector<cv::Mat> y_images(batch);
    for (auto index = 0u; index < batch; ++index) {
      const auto& file = image_y_files[index % image_y_files.size()];
      batch_y_files[index] = file;
      y_images[index] = cv::imread(file);
      CHECK(!y_images[index].empty()) << "cannot read image from " << file;
    }
    auto y_results = det->run(x_images);
    auto x_results = det->run(y_images);
    assert(x_results.size() == batch);
    for (auto i = 0u; i < x_results.size(); i++) {
      double dismat = cosine_distance(x_results[i].feat, y_results[i].feat);
      LOG(INFO) << "batch: " << i;
      LOG(INFO) << "distmat : " << std::fixed << std::setprecision(3) << dismat;
      std::cout << std::endl;
    }
  } else {
    Mat imgx = imread(argv[2]);
    if (imgx.empty()) {
      cerr << "can't load image! " << argv[1] << endl;
      return -1;
    }
    Mat imgy = imread(argv[3]);
    if (imgy.empty()) {
      cerr << "can't load image! " << argv[2] << endl;
      return -1;
    }
    Mat featx = det->run(imgx).feat;
    Mat featy = det->run(imgy).feat;
    double dismat = cosine_distance(featx, featy);
    printf("dismat : %.3lf \n", dismat);
  }
  return 0;
}
