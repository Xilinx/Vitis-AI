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
#include <glog/logging.h>
#include <fstream>
#include <iterator>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "pointpillars_onnx.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

std::vector<std::string> bins;
std::vector<std::string> rgbs;
std::vector<std::string> calibs;

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
    return filenamepart1;
}

V2F get_vec(std::string& ins, int itype)
{
  V2F ret(4, V1F(4, 0));
  ret[3] = std::vector<float>{0, 0, 0, 1};
  std::istringstream iss(ins);

  vector<string> subs;
  std::copy(istream_iterator<string>(iss),
          istream_iterator<string>(),
          back_inserter(subs));
  if (itype==12) {
    ret[0][0] = std::stof(subs[1]);
    ret[0][1] = std::stof(subs[2]);
    ret[0][2] = std::stof(subs[3]);
    ret[0][3] = std::stof(subs[4]);

    ret[1][0] = std::stof(subs[5]);
    ret[1][1] = std::stof(subs[6]);
    ret[1][2] = std::stof(subs[7]);
    ret[1][3] = std::stof(subs[8]);

    ret[2][0] = std::stof(subs[9]);
    ret[2][1] = std::stof(subs[10]);
    ret[2][2] = std::stof(subs[11]);
    ret[2][3] = std::stof(subs[12]);
  } else {
    ret[0][0] = std::stof(subs[1]);
    ret[0][1] = std::stof(subs[2]);
    ret[0][2] = std::stof(subs[3]);

    ret[1][0] = std::stof(subs[4]);
    ret[1][1] = std::stof(subs[5]);
    ret[1][2] = std::stof(subs[6]);

    ret[2][0] = std::stof(subs[7]);
    ret[2][1] = std::stof(subs[8]);
    ret[2][2] = std::stof(subs[9]);
  }
  return ret;
}

void get_display_data(int k, DISPLAY_PARAM& g_v)
{
  std::vector<std::string> lines;
  LoadListNames( calibs[k], lines);
  g_v.P2    = get_vec(lines[2], 12);
  g_v.rect  = get_vec(lines[4], 9 );
  g_v.Trv2c = get_vec(lines[5], 12);

  g_v.p2rect.resize(4);
  for(int i=0; i<4; i++) {
    g_v.p2rect[i].resize(4);
    for(int j=0; j<4; j++) {
      for(int kk=0; kk<4; kk++) {
        g_v.p2rect[i][j] += g_v.P2[i][kk]*g_v.rect[kk][j];
      }
    }
  }
  return;
}


int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "usage: " << argv[0] << " <model> <bin_list> <rgb_list> <calib_list> <result_dir> \n";
    abort();
  }

  std::string model = argv[1];
  auto net = OnnxPointpillars::create(model);

  LoadListNames( std::string(argv[2]), bins);
  LoadListNames( std::string(argv[3]), rgbs);
  LoadListNames( std::string(argv[4]), calibs);
  if ( bins.size() !=calibs.size()) {
     std::cout <<"list not same. please check. \n";
     abort();
  }

  auto ret = mkdir(argv[5], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[5] << std::endl;
    return -1;
  }

  V1F PointCloud;
  cv::Mat bevmat;
  ofstream Tout;
  ANNORET annoret;

  std::vector<float*> vf(1);
  V1I vi(1);

  for(unsigned int k=0; k<bins.size(); ++k) {
     auto& b = bins[k];
     int len = getfloatfilelen( b );
     PointCloud.resize( len );
     myreadfile(PointCloud.data(), len, b);
     vf[0] = PointCloud.data();
     vi[0] = len;
     
     auto res = net->run(vf, vi);
     annoret.clear();
     DISPLAY_PARAM g_test;
     get_display_data(k, g_test);
     cv::Mat rgbmat = cv::imread( rgbs[k] );

     net->do_pointpillar_display(res[0], 0, g_test, rgbmat, bevmat, rgbmat.cols, rgbmat.rows, annoret );
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

