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
#include <thread>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <stdlib.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vitis/ai/cflownet.hpp>

using namespace cv;
using namespace std;
using namespace vitis::ai;

namespace fs=std::filesystem;

std::string out_dir;
std::string db_dir;
std::vector<std::string> vlist;

int getfloatfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size/4;
}

template<typename T>
void myreadfile(T* dest, const std::string& filename)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {
     cout<<"Can't open the file! " << filename << std::endl;
     return;
  }
  int size1 = getfloatfilelen(filename);
  Tin.read( (char*)dest, size1*sizeof(float));
}

template<typename T>
void mywritefile(T* src, int size1, std::string filename)
{
  ofstream Tout;
  Tout.open(filename, ios_base::out|ios_base::binary);
  if(!Tout)  {
     cout<<"Can't open the file! " << filename << "\n";
     return;
  }
  Tout.write( (char*)src, size1*sizeof(T));
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

void accuracy_thread(Cflownet* det, int i, int t_n);

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage : " << argv[0] << " <db_file_list> <db_dir> <output_dir> [thread_num]" << std::endl;
    abort();
  }

  LoadListNames(argv[1],  vlist);
  db_dir = std::string(argv[2]);
  out_dir = std::string(argv[3]);

  int t_n = 1; (void)t_n;
  if (argc==5) {
    t_n = atoi(argv[4]);
  }

  auto ret = mkdir(argv[3], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
     std::cout << "error occured when mkdir " << argv[3] << std::endl;
     return -1;
  }

  std::vector<std::thread> vth;
  std::vector< std::unique_ptr<Cflownet>> vssd;
  std::vector<std::string> out_dir_t(t_n);
  for(int i=0; i<t_n; i++) {
    auto det = vitis::ai::Cflownet::create("cflownet_pt");
    if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
    }  
    vssd.emplace_back(std::move(det));
    vth.emplace_back( std::thread( &accuracy_thread, vssd[i].get(), i , t_n ));
  }
  for(int i=0; i<t_n; i++) {
    vth[i].join();
  }
  return 0;
}

void accuracy_thread(Cflownet* det, int i, int t_n) 
{
  int total_num = vlist.size();
  int split = total_num/t_n;
  int start = i*split;
  int end    = ((i != t_n-1) ? split : total_num-i*split);
  stringstream ss;
  std::vector<float> vf(128*128);
  for(int j=start; j<start+end; j++) {
    ss.str("");
    ss << db_dir << "/" << vlist[j];
    std::string fn = ss.str();
    myreadfile(vf.data() , ss.str());
    auto result = det->run(vf.data());
    ss.str("");
    ss << out_dir <<"/" <<  vlist[j];
    mywritefile( result.data.data(), 128*128, ss.str());
  }
}

