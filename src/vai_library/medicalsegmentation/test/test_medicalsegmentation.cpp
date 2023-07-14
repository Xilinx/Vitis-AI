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
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <vitis/ai/medicalsegmentation.hpp>

using namespace cv;
using namespace std;

Scalar colors[] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
                   Scalar(255, 0, 255), Scalar(0, 255, 255)};
std::vector<string> classTypes = {"BE", "cancer", "HGD", "polyp", "suspicious"};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <img_url>" << std::endl;  //
    abort();
  }

  Mat img = cv::imread(argv[1]);
  if (img.empty()) {
    cerr << "cannot load " << argv[1] << endl;
    abort();
  }

  std::string name(argv[1]);

  std::string filenamepart1 = name.substr(name.find_last_of('/') + 1);
  filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));

  auto seg =
      vitis::ai::MedicalSegmentation::create("FPN_Res18_Medical_segmentation");
  if (!seg) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  cv::Size size_orig = img.size();

  cv::Mat img_save;
  auto result = seg->run(img);

  /* simple version
  for(int i=0; i<5; i++) {
    for (auto y = 0; y < result.segmentation[i].rows; y++) {
      for (auto x = 0; x < result.segmentation[i].cols; x++) {
        result.segmentation[i].at<uchar>(y, x) *= 100;
      }
    }
    std::stringstream ss; ss << "segres_" << i << ".jpg" ;
    cv::imwrite(ss.str(), result.segmentation[i]); // Save the result as an
  image;
  }
  */

  // if dir doesn't exist, create it.
  for (int i = 0; i < 6; i++) {
    std::string path = "results";
    if (i != 0) {
      path = path + "/" + classTypes[i - 1];
    }
    auto ret = mkdir(path.c_str(), 0777);
    if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
      std::cout << "error occured when mkdir " << path << std::endl;
      return -1;
    }
  }

  for (int i = 0; i < 5; i++) {
    std::string fname("results/" + classTypes[i] + "/" + filenamepart1 +
                      ".png");
    cv::resize(result.segmentation[i], img_save, size_orig, 0, 0,
               cv::INTER_LINEAR);
    cv::imwrite(fname, img_save);          // Save the result as an image;
    auto img_true = cv::imread(fname, 0);  // gray
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::findContours(img_true, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
    if (contours.size()) {
        cv::drawContours(img, contours, -1, colors[i], 3);

        auto midVal = int(contours[0].size()/2);
        cv::putText(img,
                  classTypes[i],
                  cv::Point(contours[0][midVal].x, contours[0][midVal].y),
                  cv::FONT_HERSHEY_SIMPLEX,
                  1,
                  cv::Scalar(255,255,255),
                  2,
                  cv::LINE_AA);
    }
  }
  std::string fname("results/" + filenamepart1 + "_overlayer.png");
  cv::imwrite(fname, img);
  return 0;
}

