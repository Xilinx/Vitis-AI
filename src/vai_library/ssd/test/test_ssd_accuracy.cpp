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
#include <vitis/ai/ssd.hpp>
extern int GLOBAL_ENABLE_C_SOFTMAX;
using namespace cv;
using namespace std;
#define RESULT_FILE "accuracy_result.txt"
void parseImage(vitis::ai::SSD *ssd, cv::Mat &img,
                const std::string &single_name, std::ofstream &out);
std::string get_single_name(const std::string &line);

int main(int argc, char *argv[]) {
  GLOBAL_ENABLE_C_SOFTMAX = 2;
  /*std::map<std::string, std::string> typemap{
      {"ADAS_VEHICLE_V3_480x360", vitis::ai::SSD_ADAS_VEHICLE_V3_480x360},
      {"TRAFFIC_480x360", vitis::ssd::TRAFFIC_480x360},
      {"ADAS_PEDESTRIAN_640x360", vitis::ssd::ADAS_PEDESTRIAN_640x360},
      {"MOBILENET_V2_480x360", vitis::ssd::MOBILENET_V2_480x360},
      {"VOC_300x300_TF", vitis::ssd::VOC_300x300_TF}};
  */
  // argv[1] is the file list name
  if (argc != 3) {
    cerr << "usage: test_ssd_accuracy model_type file_list_file_name " << endl
         << "model_type is one of below  " << endl
         << "   ssd_adas_pruned_0_95" << endl
         << "   ssd_traffic_pruned_0_9" << endl
         << "   ssd_pedestrian_pruned_0_97" << endl
         << "   ssd_mobilenet_v2" << endl;
    return -1;
  }

  auto ssd = vitis::ai::SSD::create(argv[1], true);
  if (!ssd) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  std::ofstream out_fs(RESULT_FILE, std::ofstream::out);

  std::ifstream fs(argv[2]);
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
  if (found != std::string::npos) {
    return line.substr(found + 1);
  }
  return line;
}

void parseImage(vitis::ai::SSD *ssd, cv::Mat &img,
                const std::string &single_name, std::ofstream &out) {
  int width = ssd->getInputWidth();
  int height = ssd->getInputHeight();

  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

  auto results = ssd->run(img_resize);

  for (auto &it : results.bboxes) {
    out << single_name << " " << it.label << " " << it.score << " "
        << it.x * img.cols << " " << it.y * img.rows << " "
        << (it.x + it.width) * img.cols << " " << (it.y + it.height) * img.rows
        << std::endl;
  }
}
