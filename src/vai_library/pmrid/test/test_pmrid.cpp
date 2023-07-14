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

#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/pmrid.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cerr << "usage: " << argv[0] << " image_file_url " << endl;
    abort();
  }
  std::string filename = argv[2];
  cv::Mat img;
  img.create(3000, 4000, CV_16UC1);
  auto mode = std::ios_base::in | std::ios_base::binary;
  auto flag =
      std::ifstream(filename, mode).read((char*)img.data, 24000000).good();

  if (!flag) {
    LOG(INFO) << "fail to read! filename=" << argv[2];
    abort();
  }
  auto info = std::stoi(argv[3]);
  auto runner = vitis::ai::PMRID::create(argv[1], true);
  if (!runner) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  auto result = runner->run(img, info);
  auto output_file = filename + std::string(".out");
  LOG(INFO) << "dump output to " << output_file;
  CHECK(std::ofstream(output_file, std::ios_base::out | std::ios_base::binary |
                                       std::ios_base::trunc)
            .write((char*)result.data(), sizeof(float) * result.size())
            .good())
      << " faild to write to " << output_file;

  return 0;
}
