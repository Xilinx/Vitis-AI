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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vitis/ai/platedetect.hpp>
using namespace std;
using namespace vitis::ai;
using namespace cv;

string RESULT_FILE_PATH = "accuracy_result_plate/";

void parseImage(vitis::ai::PlateDetect* det, cv::Mat& img,
                const std::string& single_name);
std::string get_single_name(const std::string& line);

int main(int argc, char* argv[]) {
  if (argc != 4) {
    cerr << "usage: test_platedetect_accuracy model_name file_list_file_name "
            "folder_name"
         << endl;
    return -1;
  }
  auto det = vitis::ai::PlateDetect::create(argv[1], true);
  if (!det) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  RESULT_FILE_PATH = argv[3];
  std::ifstream fs(argv[2]);
  CHECK_EQ(system(string("rm -rf " + string(argv[3])).c_str()), 0);
  CHECK_EQ(system(string("mkdir -p " + string(argv[3])).c_str()), 0);
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
    cout << single_name << endl;
    parseImage(det.get(), image, single_name);
  }
  fs.close();

  return 0;
}

std::string get_single_name(const std::string& line) {
  std::size_t found = line.rfind('/');
  if (found != std::string::npos) {
    return line.substr(found + 1);
  }
  return line;
}

static std::vector<std::string> split(const std::string& s,
                                      const std::string& delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

void parseImage(vitis::ai::PlateDetect* det, cv::Mat& img,
                const std::string& single_name) {
  int width = det->getInputWidth();
  int height = det->getInputHeight();

  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

  auto result = det->run(img);

  // there is no label here, we always use "1" to comply with standard result
  // format std::cout << single_name << " 1 " << result.box.score << " "
  ofstream out(RESULT_FILE_PATH + "/" + split(single_name, ".")[0] + ".txt");
  out << result.top_left.x * img.cols << " " << result.top_left.y * img.rows
      << " " << result.top_right.x * img.cols << " "
      << result.top_right.y * img.rows << " "
      << result.bottom_right.x * img.cols << " "
      << result.bottom_right.y * img.rows << " "
      << result.bottom_left.x * img.cols << " "
      << result.bottom_left.y * img.rows << " " << std::endl;
  out.close();
}
