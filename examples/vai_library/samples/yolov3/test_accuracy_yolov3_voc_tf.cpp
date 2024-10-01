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
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/yolov3.hpp>

static std::vector<std::string> VOC_map{
    "aeroplane",   "bicycle", "bird",  "boat",      "bottle",
    "bus",         "car",     "cat",   "chair",     "cow",
    "diningtable", "dog",     "horse", "motorbike", "person",
    "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"};

using namespace std;
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

inline std::string get_name(std::string name) {
  auto namesp = split(name, "/");
  auto single_name = namesp[namesp.size() - 1];
  single_name = split(single_name, ".")[0];
  return single_name;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " image_list_file  output_file"
              << std::endl;
  }
  std::ofstream out_fs(argv[2], std::ofstream::out);

  auto det = vitis::ai::YOLOv3::create("yolov3_voc_tf_acc", true);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  } 
  vector<string> names;
  LoadImageNames(argv[1], names);

  for (auto& name : names) {
    auto image = cv::imread(name);
    auto res = det->run(image);
    for (auto &box : res.bboxes) {
      std::string label_name = VOC_map[box.label];
      float xmin = box.x * image.cols + 1;
      float ymin = box.y * image.rows + 1;
      float xmax = (box.x + box.width) * image.cols + 1;
      float ymax = (box.y + box.height) * image.rows + 1;
      if (xmin < 0) xmin = 1;
      if (ymin < 0) ymin = 1;
      if (xmax > image.cols) xmax = image.cols;
      if (ymax > image.rows) ymax = image.rows;
      float confidence = box.score;
      out_fs << get_name(name) << " " << label_name << " " << confidence << " "
             << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
    }
  }
  out_fs.close();
}
