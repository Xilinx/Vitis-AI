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

#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/segmentation.hpp>

using namespace std;
using namespace cv;

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

void LoadImageNames(std::string const& filename,
                    std::vector<std::string>& images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
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

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << "\n model name: " << argv[1]
              << "\nimage_list_file " << argv[2] << "\nresult_path: " << argv[3]
              << std::endl;
    return -1;
  }
  auto det = vitis::ai::Segmentation::create(argv[1]);  // Init
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  string g_output_dir = argv[3];
  string mkdir = "mkdir -p " + g_output_dir + "/";
  vector<string> names;
  LoadImageNames(argv[2], names);
  auto a = system(mkdir.c_str());
  if (a == -1) exit(0);
  vector<string> categories_path{"OP1", "OP2", "OP3", "OP4", "OP5", "OP6"};
  for (auto i = 0u; i < categories_path.size(); i++) {
    string mksubdir = "mkdir -p ./" + g_output_dir + "/" + categories_path[i];
    a = system(mksubdir.c_str());
    if (a == -1) exit(0);
  }
  for (auto& name : names) {
    cout << name << endl;
    cv::Mat img_resize;
    cv::Mat image = cv::imread(name);
    auto result = det->run_8UC1(image);
    auto namesp = split(name, "/");
    // auto i_w = det->getInputWidth();
    // auto i_h = det->getInputHeight();
    cv::Mat img;
    cv::resize(result.segmentation, img, cv::Size(640, 480), 0, 0,
               cv::INTER_NEAREST);
    cv::imwrite(g_output_dir + "/" + namesp[namesp.size() - 3] + "/" +
                    namesp[namesp.size() - 1],
                img);
    cout << g_output_dir + "/" + namesp[namesp.size() - 3] + "/" +
                namesp[namesp.size() - 1]
         << endl;
  }

  return 0;
}
