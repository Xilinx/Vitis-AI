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
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

double cosine_distance(Mat feat1, Mat feat2){
    return 1 - feat1.dot(feat2);
}

int main(int argc, char *argv[]) {
  if(argc < 3){
      cerr<<"need two images"<<endl;
      return -1;
  }
  Mat imgx = imread(argv[2]);
  if(imgx.empty()){
      cerr<<"can't load image! "<<argv[2]<<endl;
      return -1;
  }
  Mat imgy = imread(argv[3]);
  if(imgy.empty()){
      cerr<<"can't load image! "<<argv[3]<<endl;
      return -1;
  }
  auto det = vitis::ai::Reid::create(argv[1]);
  Mat featx = det->run(imgx).feat;
  Mat featy = det->run(imgy).feat;
  double dismat= cosine_distance(featx, featy);                                      
  printf("dismat : %.3lf \n", dismat);                                                          
  return 0;
}
