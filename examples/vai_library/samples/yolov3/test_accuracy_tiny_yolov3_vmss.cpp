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
#include <string>
#include <vitis/ai/yolov3.hpp>

using namespace std;
using namespace cv;

static std::vector<std::string> vmss_map {
    "KELLOGS", "CHOCOLATE", "CANDLE", "SHAMPOO", "BULB", "PLIERS", "DETERGENT", "KOOLAID",
    "LIPSTICK", "BOX"};

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
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " image_list_file  output_file"
              << std::endl;
    return -1;
  }
  vector<string> names;
  LoadImageNames(argv[1], names);
  ofstream out(argv[2]);

  auto yolo = vitis::ai::YOLOv3::create("tiny_yolov3_vmss_acc", true);
  if (!yolo) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  for (auto& name : names) {
    cv::Mat img = cv::imread(name);
    auto results = yolo->run(img);
    auto namesp = split(name, "/");
    auto single_name = namesp[namesp.size() - 1];
    single_name = split(single_name, ".")[0];
    for (auto &box : results.bboxes) {
      std::string label_name = vmss_map[box.label];

      float xmin = box.x * img.cols + 1;
      float ymin = box.y * img.rows + 1;
      float xmax = (box.x + box.width) * img.cols + 1;
      float ymax = (box.y + box.height) * img.rows + 1;
      if (xmin < 0) xmin = 1;
      if (ymin < 0) ymin = 1;
      if (xmax > img.cols) xmax = img.cols;
      if (ymax > img.rows) ymax = img.rows;
      float confidence = box.score;

      out << single_name << " " << label_name << " " << confidence << " "
          << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
      out.flush();
    }
  }
  out.close();
  return 0;
}
