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
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "vitis/ai/nnpp/multitaskv3.hpp"
#include "multitaskv3_onnx.hpp"
#include <unistd.h>
#include <cassert>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace onnx_multitaskv3;
static std::vector<std::string> label_map{"car", "sign", "person"};

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


void getAllFiles(string path, vector<string>& files) {
    DIR *basedir;
    struct dirent *entry;
    if((basedir=opendir(path.c_str()))==NULL){
        perror("Open dir error...");
        exit(1);
    }
    while((entry=readdir(basedir))!=NULL){
        string name = entry->d_name;
        string ext = name.substr(name.find_last_of(".") + 1);
        if(strcmp(entry->d_name,".")==0||strcmp(entry->d_name,"..")==0)
            continue;
        else if(entry->d_type==8 && ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png") || (ext == "bmp")  ))//file
            files.push_back(path+"/"+entry->d_name);
        else if(entry->d_type==10)
            continue;
        else if(entry->d_type==4){
            getAllFiles(path+"/"+entry->d_name,files);
        }
    }
    closedir(basedir);
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
  vector<string> names;
  if (argc < 5) {
    cout << "Please input 3 parameters into teminal.\n"
         << "First argu specify model, second is the input images folder path, third is an output-boxes txt."
         << endl;
  }
  string g_model_name = argv[1];
  string bpath = argv[3];
  ofstream out(argv[4]);
  LoadImageNames(argv[2], names);
  auto model = OnnxMultiTaskv3::create(g_model_name);
  int i = 0;
  cout << argv[2] << " "  << names.size() << " images" << endl;
  for (auto name : names) {
    name = bpath + "/" + name;
    cv::Mat img = cv::imread(name);
    std::vector<cv::Mat> images;
    images.push_back(img);
    auto results = model->run(images)[0];
    auto namesp = split(name, "/");
    auto single_name = namesp[namesp.size() - 1];
    single_name = split(single_name, ".")[0];
    for (auto& box : results.vehicle) {
      std::string label_name = "none";
      label_name = label_map[box.label];
      float xmin = box.x * img.cols;
      float ymin = box.y * img.rows;
      float xmax = (box.x + box.width) * img.cols;
      float ymax = (box.y + box.height) * img.rows;
      float confidence = box.score;
      out << single_name << " " << label_name << " " << confidence << " "
          << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
      std::cout << i << " " << single_name << " " << label_name << " " << confidence << " "
          << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
      out.flush();
    }
    i++;
  }
  out.close();
  return 0;
}
