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
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <vitis/ai/ultrafast.hpp>

using namespace cv;
using namespace std;

Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(255, 255, 0), Scalar(0, 0, 255) };

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " [ <img_url> ... ]" << std::endl; //
    abort();
  }

  std::vector<cv::Mat> arg_input_images;
  std::vector<cv::Size> arg_input_images_size;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 1; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_size.push_back(img.size());
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  auto det = vitis::ai::UltraFast::create("ultrafast_pt");
  if (!det) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  std::vector<cv::Size> batch_images_size;
  auto batch = det->get_input_batch();

  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(arg_input_images_names[batch_idx % arg_input_images.size()]);
    batch_images_size.push_back(arg_input_images_size[batch_idx % arg_input_images.size()]);
  }

  auto results = det->run(batch_images); 

  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << batch_images_names[batch_idx] << std::endl;
    int iloop = 0;
    for(auto& lane: results[batch_idx].lanes) {
      std::cout <<"lane: " << iloop << "\n";
      for(auto &v: lane) {
        if(v.first >0) {
          cv::circle(batch_images[batch_idx], cv::Point(v.first, v.second), 5, colors[iloop], -1);
        } 
        std::cout << "    ( " << v.first << ", " << v.second << " )\n";
      } 
      iloop++;
    } 
    std::string filenamepart1 = batch_images_names[batch_idx].substr( batch_images_names[batch_idx].find_last_of('/')+1 );
    filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));

    cv::imwrite(filenamepart1+"_result.jpg", batch_images[batch_idx]);
  }

  return 0;
}

