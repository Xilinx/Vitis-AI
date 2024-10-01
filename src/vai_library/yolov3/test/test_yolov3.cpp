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
#include <opencv2/opencv.hpp>
#include <vitis/ai/yolov3.hpp>

#include "../src/utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "usage: " << argv[0] << " image_file_url " << endl;
    abort();
  }
  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto yolo = vitis::ai::YOLOv3::create(argv[1], true);
  if (!yolo) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  //  auto yolo =
  //    vitis::ai::YOLOv3::create(xilinx::ai::YOLOV3_VOC_416x416_TF, true);

  auto results = yolo->run(img);

  for (auto &box : results.bboxes) {
    int label = box.label;
    float xmin = box.x * img.cols + 1;
    float ymin = box.y * img.rows + 1;
    float xmax = xmin + box.width * img.cols;
    float ymax = ymin + box.height * img.rows;
    if (xmin < 0.) xmin = 1.;
    if (ymin < 0.) ymin = 1.;
    if (xmax > img.cols) xmax = img.cols;
    if (ymax > img.rows) ymax = img.rows;
    float confidence = box.score;

    cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
              1, 0);
  }
  //    imshow("", img);
  //    waitKey(0);
  imwrite("result.jpg", img);

  return 0;
}
