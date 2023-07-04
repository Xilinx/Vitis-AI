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
#include <vitis/ai/yolov2.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage : " << argv[1] << " <image_file_url> " << std::endl;
    abort();
  }
  const auto image_file_name = std::string(argv[1]);

  auto img = cv::imread(image_file_name);
  if (img.empty()) {
    cerr << "cannot load " << image_file_name << endl;
    abort();
  }
  auto model_name = "yolov2_voc";
  auto model = vitis::ai::YOLOv2::create(model_name);
  if (!model) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  auto result = model->run(img);
  for (const auto &bbox : result.bboxes) {
    int label = bbox.label;
    float xmin = bbox.x * img.cols + 1;
    float ymin = bbox.y * img.rows + 1;
    float xmax = xmin + bbox.width * img.cols;
    float ymax = ymin + bbox.height * img.rows;
    if (xmax > img.cols) xmax = img.cols;
    if (ymax > img.rows) ymax = img.rows;
    float confidence = bbox.score;

    cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
              1, 0);
  }
  //  imshow("", img);
  // waitKey(0);
  imwrite("result.jpg", img);

  return 0;
}
