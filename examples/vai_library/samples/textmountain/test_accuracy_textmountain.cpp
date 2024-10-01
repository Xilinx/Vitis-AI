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
#include <filesystem>
#include <vitis/ai/textmountain.hpp>

using namespace std;
using namespace cv;

std::vector<std::string> pics;
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

std::string getrealname(std::string& name) {
    return name.substr(0, name.find_last_of('.'));
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name> <test_dir> <list_file> <result_dir>" 
              << "\n        model_name is textmountain_pt " << std::endl;
    abort();
  }
  auto net = vitis::ai::TextMountain::create(argv[1]);
  if (!net) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  LoadListNames( std::string(argv[3]), pics);

  auto ret = mkdir(argv[4], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[4] << std::endl;
    return -1;
  }
  ofstream Tout;
  for(unsigned int k=0; k<pics.size(); ++k) {
     std::string matpath(argv[2]);
     matpath=matpath+"/"+pics[k];
     cv::Mat mat = cv::imread( matpath );
     auto res = net->run(mat);
     std::string txtname(argv[4]);

     txtname = txtname + "/res_" +  getrealname(pics[k]) + ".txt" ;
     Tout.open(txtname, ios_base::out);
     if(!Tout) {
        cout<<"Can't open the file! " << txtname << "\n";
        return -1;
     }
     for(unsigned int j=0; j<res.res.size(); ++j) {
        Tout << res.res[j].box[0].x << "," << res.res[j].box[0].y <<","
             << res.res[j].box[1].x << "," << res.res[j].box[1].y <<","
             << res.res[j].box[2].x << "," << res.res[j].box[2].y <<","
             << res.res[j].box[3].x << "," << res.res[j].box[3].y <<","
             << res.res[j].score <<"\n";
     }
     Tout.close();
  }
  return 0;
}

