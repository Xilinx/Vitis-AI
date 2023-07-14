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

#include <sys/stat.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>
#include <vector>
#include <vitis/ai/ocr.hpp>

using namespace cv;
using namespace std;
using namespace vitis::ai;

void save_result(const std::string& path, const std::string& fn, OCRResult& res, cv::Mat& img);

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << " usage: " << argv[0] << " <model_name>  <result_dir>"
              << " [<img_url> ... ]" << std::endl;
    abort();
  }

  auto ocr = vitis::ai::OCR::create(argv[1]);
  if (!ocr) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  std::vector<cv::Mat> images;
  std::vector<std::string> names;

  for (auto i = 3; i < argc; i++) {
    auto img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    images.push_back(img);
    names.push_back(argv[i]);
  }

  if (images.empty()) {
    std::cerr << "No image load cussess !" << std::endl;
    abort();
  }

  auto results = ocr->run(images); 

  std::string resultdir(argv[2]);
  auto ret = mkdir(resultdir.c_str(), 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << resultdir << std::endl;
    return -1;
  }

  int i=0;
  for (auto &res : results) {
    std::string filenamepart1 = names[i].substr(names[i].find_last_of('/') + 1);
    filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));
    save_result(resultdir, filenamepart1, res, images[i++]);
  }
  return 0;
}

void save_result(const std::string& path, const std::string& fn, OCRResult& res, cv::Mat& img)
{
   std::string res_txt{path+"/res_"+fn+".txt"};
   std::string res_jpg{path+"/res_"+fn+".jpg"};

   ofstream Fout;
   Fout.open(res_txt, ios_base::out);
   if(!Fout) {
      cout<<"Can't open the file! " << res_txt << "\n";
      exit (-1);
   }
   std::string str;
   for(unsigned int i=0; i<res.words.size(); i++) {
      for(auto& it: res.box[i]) {
         str+= std::to_string(it.x)+","+std::to_string(it.y)+",";
      } 
      str+=res.words[i]+"\r\n";
      Fout << str;
      str = "";
      cv::polylines(img, res.box[i], true, cv::Scalar(0, 0, 255), 2 );
      cv::putText(img, res.words[i], cv::Point(res.box[i][0].x+1, res.box[i][0].y+1 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
      cv::putText(img, res.words[i], cv::Point(res.box[i][0].x, res.box[i][0].y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);
   }
   Fout.close();
   cv::imwrite(res_jpg, img);
}

