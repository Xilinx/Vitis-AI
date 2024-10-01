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
#include <sstream>
#include <fstream>
#include <vector>
#include <vitis/ai/ocr.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace vitis::ai;

void save_result(const std::string& path, const std::string& fn, OCRResult& res, cv::Mat& img);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << " usage: " << argv[0] << " <model_name> <img_url>  [ result_dir ] " << std::endl;  //
    abort();
  }

  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }
  
  std::string name(argv[2]);

  std::string filenamepart1 = name.substr(name.find_last_of('/') + 1);
  filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));

  auto ocr = vitis::ai::OCR::create(argv[1]);
  if (!ocr) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  OCRResult result;
  result  = ocr->run(img); (void)result;
  /*
   * struct OCRResult{
   *     std::vector<std::string> words;
   *     std::vector<std::vector<cv::Point>> box;
   */
  for(unsigned int i=0; i<result.words.size(); i++) {
      std::string str;
      for(auto& it: result.box[i]) {
         str+= std::to_string(it.x)+","+std::to_string(it.y)+",";
      } 
      str+=result.words[i];
      std::cout << str <<"\n";
  }

  if (argc == 4) {
    std::string resultdir(argv[3]);
    auto ret = mkdir(resultdir.c_str(), 0777);
    if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
      std::cout << "error occured when mkdir " << resultdir << std::endl;
      return -1;
    }
    save_result(resultdir, filenamepart1, result, img );
  }
  return 0;
}

void save_result(const std::string& path, const std::string& fn, OCRResult& res, cv::Mat& img )
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
   cv::Mat img1 = img.clone();
   for(unsigned int i=0; i<res.words.size(); i++) {
      for(auto& it: res.box[i]) {
         str+= std::to_string(it.x)+","+std::to_string(it.y)+",";
      } 
      str+=res.words[i]+"\r\n";
      Fout << str;
      str = "";
      cv::polylines(img1, res.box[i], true, cv::Scalar(0, 0, 255), 2 );
      cv::putText(img1, res.words[i], cv::Point(res.box[i][0].x+1, res.box[i][0].y+1 ), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
      cv::putText(img1, res.words[i], cv::Point(res.box[i][0].x, res.box[i][0].y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);
   }
   Fout.close();
   cv::imwrite(res_jpg, img1);

}

