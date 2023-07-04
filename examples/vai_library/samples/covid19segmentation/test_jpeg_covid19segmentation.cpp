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

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/covid19segmentation.hpp>
#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");
DEF_ENV_PARAM(SAMPLES_BATCH_NUM, "0");

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name> <image_url>"
              << std::endl;
    abort();
  }

  auto det = vitis::ai::Covid19Segmentation::create(argv[1]);  // Init
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  if (ENV_PARAM(SAMPLES_ENABLE_BATCH)) {
    std::vector<std::string> image_files;
    for (int i = 2; i < argc; ++i) {
      image_files.push_back(std::string(argv[i]));
    }
    if (image_files.empty()) {
      std::cerr << "no input file" << std::endl;
      exit(1);
    }

    auto batch = det->get_input_batch();
    if (ENV_PARAM(SAMPLES_BATCH_NUM)) {
      unsigned int batch_set = ENV_PARAM(SAMPLES_BATCH_NUM);
      assert(batch_set <= batch);
      batch = batch_set;
    }

    std::vector<std::string> batch_files(batch);
    std::vector<cv::Mat> images(batch);
    for (auto index = 0u; index < batch; ++index) {
      const auto& file = image_files[index % image_files.size()];
      batch_files[index] = file;
      images[index] = cv::imread(file);
      CHECK(!images[index].empty()) << "cannot read image from " << file;
    }

    auto results = det->run_8UC3(images);

    assert(results.size() == batch);
    for (auto i = 0u; i < results.size(); i++) {
      LOG(INFO) << "batch: " << i << "     image: " << batch_files[i];
      std::cout << "Run and get a visualization result" << std::endl;
      resize(results[i].positive_classification,
             results[i].positive_classification,
             cv::Size{results[i].width,
                      results[i].height});  // Resize the result Mat as same as
                                            // input size;
      cv::imwrite(
          std::to_string(i) + "_" +
              batch_files[i].substr(0, batch_files[i].size() - 4) +
              "_classification.jpg",
          results[i].positive_classification);  // Save the result as an image;
      resize(results[i].infected_area_classification,
             results[i].infected_area_classification,
             cv::Size{results[i].width,
                      results[i].height});  // Resize the result Mat as same as
                                            // input size;
      cv::imwrite(
          std::to_string(i) + "_" +
              batch_files[i].substr(0, batch_files[i].size() - 4) +
              "_infected_area.jpg",
          results[i]
              .infected_area_classification);  // Save the result as an image;
      std::cout << std::endl;
    }
  } else {
    auto image = cv::imread(argv[2]);  // Load an input image;
    if (image.empty()) {
      std::cerr << "cannot load " << argv[2] << std::endl;
      abort();
    }
    std::cout << "Run and get a visualization result" << std::endl;
    auto resultshow = det->run_8UC3(image);
    resize(
        resultshow.positive_classification, resultshow.positive_classification,
        cv::Size{resultshow.width,
                 resultshow
                     .height});  // Resize the result Mat as same as input size;
    cv::imwrite(
        "classification.jpg",
        resultshow.positive_classification);  // Save the result as an image;
    resize(
        resultshow.infected_area_classification,
        resultshow.infected_area_classification,
        cv::Size{resultshow.width,
                 resultshow
                     .height});  // Resize the result Mat as same as input size;
    cv::imwrite(
        "infected_area.jpg",
        resultshow
            .infected_area_classification);  // Save the result as an image;
  }
  return 0;
}
