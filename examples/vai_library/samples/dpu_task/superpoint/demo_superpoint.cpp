/*
 * Copyright 2019 Xilinx Inc.
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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "./superpoint.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  string model_name = argv[1];
  Mat img = imread(argv[2], cv::IMREAD_GRAYSCALE);
  {
    auto superpoint = vitis::ai::SuperPoint::create(model_name);

    vector<Mat> imgs;
    for(size_t i = 0; i < superpoint->get_input_batch(); ++i)
      imgs.push_back(img);
    auto result = superpoint->run(imgs);
    for(size_t i = 0; i < superpoint->get_input_batch(); ++i) {
      LOG(INFO) << "res scales: " << result[i].scale_h << " " << result[i].scale_w;
      for(size_t k = 0; k < result[i].keypoints.size(); ++k)
        circle(imgs[i], Point(result[i].keypoints[k].first*result[i].scale_w,
               result[i].keypoints[k].second*result[i].scale_h), 1, Scalar(0, 0, 255), -1);
      imwrite(string("result_superpoint_")+to_string(i)+".jpg", imgs[i]);
      //imshow(std::string("result ") + std::to_string(c), result[c]);
      //waitKey(0);
    }
  }
  LOG(INFO) << "BYEBYE";
  return 0;
}

