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
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/classification.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cout << "Please input the model name as the first param!" << endl
         << "And input your image path as the second param!" << endl
         << "The third param is a txt to store results!" << endl;
  }

  cv::String path = argv[2];
  std::ofstream out_fs(argv[3], std::ofstream::out);
  int length = path.size();

  auto g_model_name = argv[1] + string("_acc");
  auto det = vitis::ai::Classification::create(g_model_name);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }   

  vector<cv::String> files;
  cv::glob(path, files);
  int count = files.size();
  cerr << "The image count = " << count << endl;
  for (int i = 0; i < count; i++) {
    auto image = imread(files[i]);
    if (image.empty()) {
      cerr << "cannot load " << files[i] << endl;
      abort();
    }
    auto res = det->run(image);
    for (size_t j = 0; j < res.scores.size(); ++j) {
      int index = res.scores[j].index;
      cout << String(files[i]).substr(length) << " " << index << " "
           << res.scores[j].score << " " << endl;
      out_fs << cv::String(files[i]).substr(length) << " " << index << endl;
    }
  }
  out_fs.close();
}
