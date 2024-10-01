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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/segmentation.hpp>
using namespace std;
using namespace cv;

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
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

void LoadImageNames(std::string const &filename,
                    std::vector<std::string> &images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE *fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << "\n model name: " << argv[1] << "\nimage_list_file " << argv[2] << "\nresult_path: " << argv[3] << std::endl;
    return -1;
  }
  auto det = vitis::ai::Segmentation::create(argv[1]);  // Init
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  string g_output_dir = argv[3];
  vector<string> names;
  LoadImageNames(argv[2], names);
  // ofstream out(argv[2]);
  for (auto& name : names) {
    cout << name << endl;
    cv::Mat img_resize;
    cv::Mat image = cv::imread(name);
    auto result = det->run_8UC1(image);
    // std::vector<uint8_t> data1;
    auto namesp = split(name, "/");
    // data1.assign(result.segmentation.begin(), result.segmentation.end());
    // cv::Mat img(result.rows, result.cols, CV_8UC1, data1.data());
    cv::Mat img;
    cv::resize(result.segmentation, img, cv::Size(2048, 1024), 0, 0,
               cv::INTER_NEAREST);
    cv::imwrite(g_output_dir + "/" + namesp[namesp.size() - 1], img);
  }

  return 0;
}
