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

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <vitis/ai/medicaldetection.hpp>

using namespace cv;
using namespace std;

std::vector<string> classTypes =  {"BE", "cancer", "HGD" , "polyp", "suspicious"};

void LoadImageNames(std::string const &filename, std::vector<std::string> &images) {
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
  if (argc < 4) {
    std::cerr << "usage : " << argv[0] << " model_name  <image_list_file> <output_file>" << std::endl
              << "          model_name is RefineDet_Medical  " << std::endl;
    abort();
  }

  std::ofstream out_fs(argv[3], std::ofstream::out);
  if(!out_fs)  {
      std::cout<<"Can't open the file medicaldet.result!";      
      abort();
  }

  // auto det = vitis::ai::MedicalDetection::create("RefineDet_Medical");
  std::string model = argv[1] + std::string("_acc");
  auto det = vitis::ai::MedicalDetection::create(model);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  std::string name(argv[2]);
  std::string fpath = name.substr(0, name.find_last_of('/')+1 );
  fpath.append("images/");

  vector<string> names;
  LoadImageNames(argv[2], names);
 
  for (auto& name : names) {
    std::string picname(fpath+name+".jpg");
    cv::Mat img = cv::imread(picname);

    auto result = det->run(img);
    for(auto& res: result.bboxes) {
       out_fs << name << " " << classTypes[res.label-1] << " " << res.score << " " 
              << res.x*img.cols<< " " 
              << res.y*img.rows<< " " 
              << (res.x + res.width)*img.cols << " " 
              << (res.y + res.height)*img.rows << std::endl;

    } 
  }

  out_fs.close();
  return 0;
}

