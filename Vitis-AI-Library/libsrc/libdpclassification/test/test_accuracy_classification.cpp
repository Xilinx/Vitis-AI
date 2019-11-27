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
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <xilinx/ai/classification.hpp>

#include <xilinx/ai/proto/dpu_model_param.pb.h>

using namespace std;
using namespace cv;
namespace xilinx {
namespace ai {
extern "C" xilinx::ai::proto::DpuModelParam *
find(const std::string &model_name);
}
} // namespace xilinx
int main(int argc, char *argv[]) {
  if (argc < 3)
    cout << "Please input the model name as the first param!" << endl
	 << "And input your image path as the second param!" << endl;

  cv::String path = argv[2];
  int length = path.size();

  auto g_model_name = argv[1];
  auto model1 = xilinx::ai::find(g_model_name);
  model1->mutable_classification_param()->set_test_accuracy(true);

  auto det = xilinx::ai::Classification::create(g_model_name);

  vector<cv::String> files;
  cv::glob(path, files);
  int count = files.size();
  cerr << "The image count = " << count << endl;
  for (int i = 0; i < count; i++) {
    auto image = imread(files[i]);
    auto res = det->run(image);
    for (size_t j = 0; j < res.scores.size(); ++j) {
      int index = res.scores[j].index;
      cout << String(files[i]).substr(length) << " " << index << " "
           << res.scores[j].score << " " << endl;
    }
  }
}
