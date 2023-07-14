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
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <vitis/ai/monodepth2.hpp>

using namespace cv;
using namespace std;

typedef struct {
    double r,g,b;
} COLOUR;

COLOUR GetColour(double v,double vmin,double vmax)
{
   COLOUR c = {1.0,1.0,1.0}; // white
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4 * (v - vmin - 0.5 * dv) / dv;
      c.b = 0;
   } else {
      c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c.b = 0;
   }
   return(c);
}

template<typename T>
void mywritefile(T* src, int size1, const std::string& filename)
{
  ofstream Tout;
  Tout.open(filename, ios_base::out|ios_base::binary);
  if(!Tout)  {
     cout<<"Can't open the file! " << filename << "\n";
     return;
  }
  Tout.write( (char*)src, size1*sizeof(T));
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <img_url>" << std::endl;  //
    abort();
  }

  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto net = vitis::ai::Monodepth2::create(argv[1]);
  if (!net) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto result = net->run(img);

  float* p = result.mat.ptr<float>(0);
  mywritefile( p, 192*640, "result_monodepth2.bin" );

  cv::Mat imgo{ 192, 640, CV_8UC3 , cvScalar(0)};
  for (int i=0; i<192; i++) {
    for(int j=0; j<640; j++) {
      COLOUR c = GetColour( p[i*640+j], 0.01, 9.99 );
      imgo.at<cv::Vec3b>(i, j) = cv::Vec3b(c.b*255, c.g*255, c.r*255 );
    }
  }
  imwrite("result_monodepth2.png", imgo);

  return 0;
}

