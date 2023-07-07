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
#include <google/protobuf/text_format.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "fadnet_corr.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  // A kernel name, it should be samed as the dnnc result. e.g.
  // /usr/share/vitis_ai_library/models/FADNet_0_pt/FADNet_0_pt.xmodel
  //auto kernel_name_0 = argv[1];
  //auto kernel_name_1 = argv[2];
  //auto kernel_name_2 = argv[3];
  string model_name = argv[1];

  Mat img_l = cv::imread(argv[2]);
  Mat img_r = cv::imread(argv[3]);

  auto fadnet = vitis::ai::FadNet::create(model_name);
  size_t batch = fadnet->get_input_batch();
  // Create a dpu task object.
  vector<pair<Mat, Mat>> imgs;
  for(size_t i = 0; i < batch; ++i)
    imgs.push_back(make_pair(img_l, img_r));

  // Execute the FADnet post-processing.
  auto result = fadnet->run(imgs);
  //imshow("", result[0]);
  //waitKey(0);
  imwrite("result_fadnet.jpg", result[0]);
  LOG(INFO) << "write result succeed";

  return 0;
}
