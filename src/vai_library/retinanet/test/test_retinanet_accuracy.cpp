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
#include <opencv2/opencv.hpp>
#include <vitis/ai/retinanet.hpp>
extern int GLOBAL_ENABLE_C_SOFTMAX;
using namespace cv;
using namespace std;
#define RESULT_FILE "accuracy_result.txt"
void parseImage(vitis::ai::RetinaNet *retinanet, cv::Mat &img,
                const std::string &single_name, std::ofstream &out);
std::vector<std::string> get_single_name(const std::string &line);

int main(int argc, char *argv[]) {
  GLOBAL_ENABLE_C_SOFTMAX = 2;
    // argv[1] is the file list name
  if (argc != 3) {
    std::cout << "usage: test_retinanet_accuracy model_type file_list_file_nam"
         << std::endl;
    return -1;
  }

  auto retinanet = vitis::ai::RetinaNet::create(argv[1], true);
  if (!retinanet) { // supress coverity complain
     std::cout <<"create error" << std::endl;
     abort();
  }

  std::ofstream out_fs(RESULT_FILE, std::ofstream::out);

  std::ifstream fs(argv[2]);
  std::string line;
  std::string single_name;
  while (getline(fs, line)) {
    auto image = cv::imread(get_single_name(line)[1]);
    if (image.empty()) {
      std::cout << "cannot read image: " << line << std::endl;
      continue;
    }
    single_name = get_single_name(line)[0];
    std::cout << "image id : " << single_name
        << std::endl;
    parseImage(retinanet.get(), image, single_name, out_fs);
  }
  fs.close();
  out_fs.close();

  return 0;
}

std::vector<std::string> get_single_name(const std::string &line) {
  std::vector<std::string> res;
  std::size_t found = line.rfind(' ');
  if (found != std::string::npos) {
    res.push_back(line.substr(0, found));
    res.push_back(line.substr(found + 1));
  }
  return res;
}

void parseImage(vitis::ai::RetinaNet *retinanet, cv::Mat &img,
                const std::string &single_name, std::ofstream &out) {
  int width = retinanet->getInputWidth();
  int height = retinanet->getInputHeight();

  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

  auto results = retinanet->run(img_resize);

  float h_scale = img.rows * 1.0 / height;
  float w_scale = img.cols * 1.0 / width;
  for (auto &it : results) {
    out << single_name << " " << it.label
        << " " << it.score
        << " " << (int)(it.x * w_scale)
        << " " << (int)(it.y * h_scale)
        << " " << (int)(it.w * w_scale)
        << " " << (int)(it.h * h_scale)
        << std::endl;
  }
}
