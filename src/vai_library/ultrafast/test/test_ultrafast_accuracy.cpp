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
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <unordered_set>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <vitis/ai/ultrafast.hpp>

using namespace cv;
using namespace std;

void LoadListNames(const std::string& filename, std::vector<std::string>& vlist) {
  ifstream Tin;
  Tin.open(filename, ios_base::in);
  std::string str;
  if (!Tin) {
    std::cout << "Can't open the file " << filename << "\n";
    exit(-1);
  }
  while (getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
}

void create_dir(const std::string& path){
   auto ret = mkdir(path.c_str(), 0755);
   if ( ret == -1 && errno == ENOENT ){
      std::string path1 = path.substr(0, path.find_last_of('/'));
      create_dir(path1);
      create_dir(path);
      return;
   }
   if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
     std::cout << "error occured when mkdir " << path << "   " << errno << std::endl;
     exit(-1);
   }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << " usage: " << argv[0] << " <base_dir> <input_list_file> <output_dir>" << std::endl; //
    abort();
  }

  auto det = vitis::ai::UltraFast::create("ultrafast_pt");
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  std::vector<std::string> vlist;
  LoadListNames(argv[2], vlist);
  std::string basedir(argv[1]);
  std::string outdir(argv[3]);
  std::unordered_set<std::string> path_db;

  create_dir(outdir);
  create_dir(outdir+"/culane");

  int batch = det->get_input_batch();
  for(int i=0; i<(int)vlist.size(); ) {
    std::vector<Mat> imgs;
    int real_batch = 0;
    for(int j=0; j<batch && i+j<(int)vlist.size(); j++){
      std::string imgp(basedir+"/"+vlist[i+j]);
      Mat img = cv::imread(imgp);
      if (img.empty()) {
        cerr << "cannot load " << imgp << endl;
        abort();
      }
      imgs.emplace_back(img);
      real_batch++;
    }
    auto results = det->run(imgs);  

    for(int j=0; j<real_batch; j++){
      ofstream Tout;
      std::string toutname(outdir+"/culane"+vlist[i+j]);
      std::string path1 = toutname.substr(0, toutname.find_last_of('/'));

      if (path_db.find( path1) == path_db.end()) {
        if (access(path1.c_str(), F_OK)!=0) {
            create_dir(path1);
        }
        path_db.emplace(path1);
      }
      toutname = toutname.substr(0, toutname.find_last_of('.')) + ".lines.txt";
      Tout.open(toutname, ios_base::out);
      if (!Tout) {
        cout << "Can't open the file! " << toutname << "\n";
        return -1;
      }
      
      for(auto &lane: results[j].lanes) {
         int num = 0;
         for(auto &v0: lane) {
            if(v0.first >0) {
               num++;
               if (num>2) {
                  for(auto &v: lane) {
                     if(v.first >0) {
                        Tout << (int)v.first << " " << (int)v.second << " " ;
                     } 
                  }
                  Tout <<"\n";
                  break;
               } 
            } 
         }
      }
      Tout.close();
    }
    i+=real_batch;
  }

  return 0;
}

