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
#include <google/protobuf/text_format.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "fadnet_v2.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  // A kernel name, it should be samed as the dnnc result. e.g.
  // /usr/share/vitis_ai_library/models/FADNet_0_pt/FADNet_0_pt.xmodel

  Mat img_l = cv::imread(argv[1]);
  Mat img_r = cv::imread(argv[2]);

  auto fadnet = vitis::ai::FadNetV2::create();
  if (!fadnet) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  // Create a dpu task object.
  vector<pair<Mat, Mat>> imgs;
  for (size_t i = 0; i < fadnet->get_input_batch(); ++i)
    imgs.push_back(make_pair(img_l, img_r));

  // Execute the FADnet post-processing.
  auto result = fadnet->run(imgs);
  //imshow("", result[0]);
  //waitKey(0);
  imwrite("result_fadnet.jpg", result[0]);
  LOG(INFO) << "write result succeed";

  return 0;
}
