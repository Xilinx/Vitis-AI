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
#include <opencv2/opencv.hpp>
#include <vitis/ai/medicaldetection.hpp>

using namespace cv;
using namespace std;

Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 0, 255), Scalar(0, 255, 255)};
std::vector<string> classTypes =  {"BE", "cancer", "HGD" , "polyp", "suspicious"};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] 
              << " [<img_url> ... ]" << std::endl;
    abort();
  }

  auto det = vitis::ai::MedicalDetection::create("RefineDet_Medical");
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
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
    for(auto& res: results[batch_idx].bboxes) {
      std::cout << classTypes[res.label-1] << "   " 
                << res.score << "   " 
                << res.x << " "
                << res.y << " "
                << res.width << " "
                << res.height << "\n";
      rectangle(batch_images[batch_idx], 
                Point(res.x*batch_images[batch_idx].cols, 
                      res.y*batch_images[batch_idx].rows), 
                Point((res.x+res.width)*batch_images[batch_idx].cols, 
                      (res.y+res.height)*batch_images[batch_idx].rows),
                colors[res.label-1], 
                1, 1, 0);
    }
    std::string name(batch_images_names[batch_idx]);
    std::string fpart1 = name.substr(0, name.find_last_of('.'));
    std::string fname(fpart1+"_result.jpg");
    cv::imwrite(fname, batch_images[batch_idx]);   
  }

  return 0;
}


