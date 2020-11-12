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

#include <vitis/ai/pointpillars.hpp>
#include <opencv2/opencv.hpp>
#include <vitis/ai/profiling.hpp>
#include <sys/stat.h>
#include <future>
#include <csignal>
#include <execinfo.h>
#include <iostream>

// #define PREP_DEMO_DATA 1    // only for demo program

using namespace std;
using namespace cv;
using namespace vitis::ai;

void get_display_data(DISPLAY_PARAM&);

template<typename T>
void myreadfile(T* dest, int size1, std::string filename)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {
     cout<<"Can't open the file! " << filename << std::endl;
     return;
  }
  Tin.read( (char*)dest, size1*sizeof(T));
}

int getfloatfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size/4;
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

std::string getrealname(std::string& name) {
    std::string filenamepart1 = name.substr(name.find_last_of('/') + 1);
    filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));
#ifdef PREP_DEMO_DATA
    filenamepart1 = filenamepart1.substr(4);
#endif
    return filenamepart1;
}

int main( int argc, char *argv[])
{
  if (argc != 6) {
    std::cout << "usage: " << argv[0] << " <model_0> <modle_1> <bin_list> <rgb_list> <result_dir> \n"
              << "      model_0 is: pointpillars_kitti_12000_0_pt\n"
              << "      model_1 is: pointpillars_kitti_12000_1_pt\n";
    abort();
  }
  auto net = vitis::ai::PointPillars::create(argv[1], argv[2] );  

  std::vector<std::string> bins;
  std::vector<std::string> rgbs;
  LoadListNames( std::string(argv[3]), bins);
  LoadListNames( std::string(argv[4]), rgbs);
  if (bins.size() > rgbs.size() ) {
     std::cout <<"Bin list is longer than rgb list. please check. \n";
     abort();
  }

  DISPLAY_PARAM g_test;
  get_display_data(g_test);

  auto ret = mkdir(argv[5], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[4] << std::endl;
    return -1;
  }

  V1F PointCloud; 
  cv::Mat bevmat;
  ofstream Tout;
  ANNORET annoret;

  for(unsigned int k=0; k<bins.size(); ++k) {
     auto& b = bins[k];
     int len = getfloatfilelen( b );
     PointCloud.resize( len );
     myreadfile(PointCloud.data(), len, b);
     cv::Mat rgbmat = cv::imread( rgbs[k] );
     auto res = net->run(PointCloud);
     annoret.clear();
     net->do_pointpillar_display(res, 0, g_test, rgbmat, bevmat, rgbmat.cols, rgbmat.rows, annoret );
     std::string txtname(argv[5]); 
     txtname = txtname + "/" +  getrealname(b) + ".txt" ;
     Tout.open(txtname, ios_base::out);
     if(!Tout) {
        cout<<"Can't open the file! " << txtname << "\n";
        return -1;
     }
     for (unsigned int i=0; i<annoret.name.size(); i++) {
        Tout << annoret.name[i]      << " ";
        // Tout << annoret.truncated[i] << " ";
        // Tout << annoret.occluded[i]  << " ";
        Tout << "-1 -1 ";
        Tout << annoret.alpha[i]     << " ";
        for (int j=0; j<4; j++) {
           Tout << annoret.bbox[i][j] << " ";
        } 
        for (int j=0; j<3; j++) {
           Tout << annoret.dimensions[i][j] << " ";
        } 
        for (int j=0; j<3; j++) {
           Tout << annoret.location[i][j] << " ";
        } 
        Tout << annoret.rotation_y[i]     << " ";
        Tout << annoret.score[i]          << "\n";
     }
     Tout.close();
  }
  return 0;	  
}

void get_display_data(DISPLAY_PARAM& g_v)
{
  g_v.P2.emplace_back(std::vector<float>{ 721.54, 0, 609.56, 44.857   });
  g_v.P2.emplace_back(std::vector<float>{ 0, 721.54, 172.854,  0.21638   });
  g_v.P2.emplace_back(std::vector<float>{ 0, 0,  1, 0.002746   });
  g_v.P2.emplace_back(std::vector<float>{ 0, 0, 0, 1   });

  g_v.rect.emplace_back(std::vector<float>{ 0.999924, 0.009838, -0.007445,   0 });
  g_v.rect.emplace_back(std::vector<float>{ -0.00987, 0.99994, -0.00427846,  0 });
  g_v.rect.emplace_back(std::vector<float>{ 0.007403, 0.004351614, 0.999963,  0 });
  g_v.rect.emplace_back(std::vector<float>{ 0, 0, 0, 1  });

  g_v.Trv2c.emplace_back(std::vector<float>{0.007534,  -0.99997 ,  -0.0006166,  -0.00407 });
  g_v.Trv2c.emplace_back(std::vector<float>{ 0.0148,  0.000728,  -0.99989,  -0.07632  });
  g_v.Trv2c.emplace_back(std::vector<float>{  0.99986,  0.0075238,  0.0148,  -0.27178 });
  g_v.Trv2c.emplace_back(std::vector<float>{ 0, 0, 0, 1 });

  g_v.p2rect.resize(4);
  for(int i=0; i<4; i++) {
    g_v.p2rect[i].resize(4);
    for(int j=0; j<4; j++) {
      for(int k=0; k<4; k++) {
        g_v.p2rect[i][j] += g_v.P2[i][k]*g_v.rect[k][j];
      }
    }
  }
  return;
}

