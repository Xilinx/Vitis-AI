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
#include <vitis/ai/refinedet.hpp>

using namespace std;

int main(int argc, char* argv[]) {
  auto det = vitis::ai::RefineDet::create(argv[1]);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto image_file = string(argv[2]);
  auto image = cv::imread(image_file);
  cout << "load image" << endl;
  if (image.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto results = det->run(image);
  LOG(INFO) << "results.size() = " << results.bboxes.size();
  if (1) {
    auto img = image.clone();
    for (auto& box : results.bboxes) {
      float x = box.x * (img.cols);
      float y = box.y * (img.rows);
      int xmin = x;
      int ymin = y;
      int xmax = x + (box.width) * (img.cols);
      int ymax = y + (box.height) * (img.rows);
      float score = box.score;
      xmin = std::min(std::max(xmin, 0), img.cols);
      xmax = std::min(std::max(xmax, 0), img.cols);
      ymin = std::min(std::max(ymin, 0), img.rows);
      ymax = std::min(std::max(ymax, 0), img.rows);

      LOG(INFO) << "RESULT " << box.label << " :\t" << xmin << "\t" << ymin
                << "\t" << xmax << "\t" << ymax << "\t" << score << "\n";
      auto label = box.label;
      if (label == 1) {
        cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(0, 255, 0), 1, 1, 0);
      } else if (label == 2) {
        cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(255, 0, 0), 1, 1, 0);
      } else if (label == 3) {
        cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(0, 0, 255), 1, 1, 0);
      } else {
        cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(255.0 / label, 255 - label * 10, label), 1, 1,
                      0);
      }
    }
    auto out = image_file.substr(0, image_file.size() - 4) + "_out.jpg";
    LOG(INFO) << "write result to " << out;
    cv::imwrite(out, img);
  }
  return 0;
}
