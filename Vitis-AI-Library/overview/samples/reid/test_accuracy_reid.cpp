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

#include <vitis/ai/reid.hpp>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

double dismat[3368][19732];

double cosine_distance(Mat feat1, Mat feat2){
    return 1 - feat1.dot(feat2);
}

int main(int argc, char *argv[]) {
  auto det = vitis::ai::Reid::create("reid");
  vector<Mat> featx;
  vector<Mat> featy;
  ifstream imagex(argv[1]);
  ifstream imagey(argv[2]);
  string line;
  while(getline(imagex, line)) {
      Mat img = imread(line);
      if(img.empty()){
          cerr<<"can't load image! "<<line<<endl;
          continue;
      }
      Mat feat = det->run(img).feat;
      featx.emplace_back(feat);
  }
  while(getline(imagey, line)) {
      Mat img = imread(line);
      if(img.empty()){
          cerr<<"can't load image! "<<line<<endl;
          continue;
      }
      Mat feat = det->run(img).feat;
      featy.emplace_back(feat);
  }                                                                                                
  int x = featx.size();                                                                            
  int y = featy.size();                                                                            
  FILE* out_fs = fopen(argv[3], "w");
  for(int i = 0; i < x; ++i){                                                                      
      for(int j = 0; j < y; ++j){                                                                  
          dismat[i][j] = cosine_distance(featx[i], featy[j]);                                      
          fprintf(out_fs, "%.3lf ", dismat[i][j]);                                                          
      }                                                                                            
      fprintf(out_fs, "\n");                                                          
  }     
  return 0;
}
