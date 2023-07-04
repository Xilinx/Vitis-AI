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
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vitis/ai/ssd.hpp>
static std::vector<std::string> label_map{"backgroud", "person"};
using namespace cv;
using namespace std;

void parseImage(vitis::ai::SSD *ssd, cv::Mat &img,
                const std::string &single_name, std::ofstream &out);
std::string get_single_name(const std::string &line);

extern int GLOBAL_ENABLE_C_SOFTMAX;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " image_list_file  output_file"
              << std::endl;
    return -1;
  }
  GLOBAL_ENABLE_C_SOFTMAX = 2;

  auto ssd = vitis::ai::SSD::create("ssd_pedestrian_pruned_0_97_acc", true);
  if (!ssd) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  std::ifstream fs(argv[1]);
  std::ofstream out_fs(argv[2], std::ofstream::out);
  std::string line;
  std::string single_name;
  while (getline(fs, line)) {
    // LOG(INFO) << "line = [" << line << "]";
    auto image = cv::imread(line);
    if (image.empty()) {
      cerr << "cannot read image: " << line;
      continue;
    }
    single_name = get_single_name(line);
    parseImage(ssd.get(), image, single_name, out_fs);
  }
  fs.close();
  out_fs.close();

  return 0;
}

std::string get_single_name(const std::string &line) {
  std::size_t found = line.rfind('/');
  auto single_name = line;
  if (found != std::string::npos) {
    single_name = line.substr(found + 1);
  }
  found = single_name.rfind('.');
  if (found != std::string::npos) {
    single_name = single_name.substr(0, found);
  }

  return single_name;
}

void parseImage(vitis::ai::SSD *ssd, cv::Mat &img,
                const std::string &single_name, std::ofstream &out) {
  int width = ssd->getInputWidth();
  int height = ssd->getInputHeight();

  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

  auto results = ssd->run(img_resize);

  for (auto &it : results.bboxes) {
    if (it.label < (int)label_map.size()) {
      out << single_name << " " << label_map[it.label] << " " << it.score << " "
          << it.x * img.cols << " " << it.y * img.rows << " "
          << (it.x + it.width) * img.cols << " "
          << (it.y + it.height) * img.rows << std::endl;
    } else {
      std::cerr << "Label not in map." << std::endl;
    }
  }
}
