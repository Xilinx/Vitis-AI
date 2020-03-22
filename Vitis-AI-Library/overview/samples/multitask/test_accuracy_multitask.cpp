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
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vitis/ai/multitask.hpp>
using namespace std;
using namespace cv;

extern int GLOBAL_ENABLE_C_SOFTMAX;

static std::vector<std::string> label_map{
    "background",   "person", "car",  "truck",
    "bus", "bike",   "sign",  "light"};

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0)
    return elems;
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
  vector<string> names;
  GLOBAL_ENABLE_C_SOFTMAX = 2;
  if (argc < 4) {
    cout << "Please input 3 parameters into teminal.\n"
         << "First is the input-images list, second is an output-boxes txt, third is the output-images folder."
         << endl;
  }
  LoadImageNames(argv[1], names);
  ofstream out(argv[2]);
  string g_model_name = "multi_task_acc";
  string out_img_path = string(argv[3]);
  auto model = vitis::ai::MultiTask::create(g_model_name);
  for (auto name : names) {
    cv::Mat img = cv::imread(name);
    auto results = model->run_8UC1(img);
    auto namesp = split(name, "/");
    auto single_name = namesp[namesp.size() - 1];
    single_name = split(single_name, ".")[0];
    for (auto &box : results.vehicle) {

      std::string label_name = label_map[box.label];
      float xmin = box.x * img.cols;
      float ymin = box.y * img.rows;
      float xmax = (box.x + box.width) * img.cols;
      float ymax = (box.y + box.height) * img.rows;
      float confidence = box.score;
      //std::cout << single_name << " " << label_name << " " << confidence << " "
      //          << xmin << " " << ymin << " " << xmax << " " << ymax
      //          << std::endl;
      out << single_name+".png" << " " << label_name << " " << confidence << " "
          << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
      out.flush();
    }
    imwrite(out_img_path + single_name + ".png", results.segmentation);
  }
  out.close();
  return 0;
}
