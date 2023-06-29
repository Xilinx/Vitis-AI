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

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/ocr.hpp>

using namespace std;
using namespace cv;
using namespace vitis::ai;

void save_result(const std::string& path, const std::string& fn, OCRResult& res)
{
   std::string res_txt{path+"/res_"+fn+".txt"};

   ofstream Fout;
   Fout.open(res_txt, ios_base::out);
   if(!Fout) {
      cout<<"Can't open the file! " << res_txt << "\n";
      exit (-1);
   }
   std::string str;
   if (res.words.size() != res.box.size()) {
     std::cout <<"Big error ! words and box size not match " << res.words.size() << " " << res.box.size() << "\n";
   }
   for(unsigned int i=0; i<res.words.size(); i++) {
      for(auto& it: res.box[i]) {
         str+= std::to_string(it.x)+","+std::to_string(it.y)+",";
      } 
      str+=res.words[i]+"\n";
      Fout << str;
      str = "";
   }
   Fout.close();
}

void LoadListNames(const std::string& filename,  std::vector<std::string> &vlist)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in);
  std::string str;
  if(!Tin)  {
     std::cout<<"Can't open the file " << filename << "\n";      exit(-1);
  }
  while( getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name> <image_list_file>  <result_dir>" 
              << "\n        model_name is ocr_pt " << std::endl;
    abort();
  }

  auto ocr = vitis::ai::OCR::create(argv[1]);
  if (!ocr) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  vector<string> names;
  LoadListNames(argv[2], names);

  std::string result_path(argv[3]);

  // if dir doesn't exist, create it.
  auto ret = mkdir(result_path.c_str(), 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
      std::cout << "error occured when mkdir " << result_path << std::endl;
      return -1;
  }
  std::string in_path(argv[2]);
  std::string dirpart1 = in_path.substr(0, in_path.find_last_of('/')+1 );
  if (dirpart1.empty()) {
     dirpart1="./";
  }

  int total = (int)names.size();
  int loop = 0;
  for (auto& name : names) {
    std::string fname{dirpart1+"images/"+name};
    std::cout << "imgfile :" << fname << "    "  << loop++ << "/" << total << "\n";
    cv::Mat img = cv::imread(fname );
    auto result = ocr->run(img); (void)result;
    std::string  name1  = name.substr(0, name.find_last_of('.') );
    save_result( result_path, name1, result);
  }

  return 0;
}

