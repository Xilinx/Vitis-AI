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
#include <fstream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/monodepth2.hpp>

using namespace cv;
using namespace std;

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
    std::cerr << "usage :" << argv[0] << " <image_url> [<image_url> ...]"
              << std::endl;
    abort();
  }

  auto net = vitis::ai::Monodepth2::create("monodepth2_pt");
  if (!net) { // supress coverity complain
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
  auto batch = net->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
    batch_images_size.push_back(
        arg_input_images_size[batch_idx % arg_input_images.size()]);
  }

  auto result = net->run(batch_images);

  for (auto batch_idx = 0u; batch_idx < result.size(); batch_idx++) {
    float* p = result[batch_idx].mat.ptr<float>(0);
    std::string name("result_");
    name.append(batch_images_names[batch_idx]);
    name.append(".bin");
    mywritefile( p, 192*640, name );
  }
  return 0;
}

