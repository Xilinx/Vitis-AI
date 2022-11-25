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

using namespace std;
using namespace cv;

#include "./psmnet.hpp"
int main(int argc, char* argv[]) {
  Mat img_l0 = cv::imread(argv[1]);
  Mat img_r0 = cv::imread(argv[2]);

  {
    auto psmnet = vitis::ai::PsmNet::create();
    vector<pair<Mat, Mat>> imgs;
    for (size_t i = 0; i < psmnet->get_input_batch(); ++i)
      imgs.push_back(make_pair(img_l0, img_r0));
    if (1) {
      psmnet->run(imgs);
      auto result = psmnet->get_result();
      int c = 0;
      for (auto r : result) {
        // imshow(std::string("result ") + std::to_string(c), result[c]);
        imwrite(std::string("result_psmnet_") + std::to_string(c) + ".jpg", r);
        c++;
      }
      if (c != 0) {
        waitKey(0);
      }
    }
  }
  LOG(INFO) << "BYEBYE";
  return 0;
}
