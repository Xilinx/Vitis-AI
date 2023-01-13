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

#include "reid_onnx.hpp"

double cosine_distance(Mat feat1, Mat feat2) { return 1 - feat1.dot(feat2); }

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>" << " <list_file1> <list_file2> <outfile> " << std::endl;
    abort();
  }

  FILE *out_fs = fopen(argv[4], "w");
  if(out_fs==NULL) {
     std::cout <<"bad file for result \n";
     return -1;
  }

  auto det = OnnxReid::create(argv[1]);
  size_t batch = det->get_input_batch();
  vector<Mat> featx;
  vector<Mat> featy;
  ifstream imagex(argv[2]);
  ifstream imagey(argv[3]);
  vector<Mat> images;
  string line;
  while (getline(imagex, line)) {
    auto image = cv::imread(line);
    if (image.empty()) {
      cerr << "cannot read image: " << line;
      continue;
    }
    images.push_back(image);
  }

  imagex.close();
  vector<Mat> inputs;
  for (size_t i = 0; i < images.size(); i++) {
    inputs.push_back(images[i]);
    if (inputs.size() < batch && i < images.size() - 1) {
      continue;
    }
    auto rets = det->run(inputs);
    for (auto ret : rets) {
      featx.push_back(ret.feat);
    }
    inputs.clear();
  }
  images.clear();
  while (getline(imagey, line)) {
    auto image = cv::imread(line);
    if (image.empty()) {
      cerr << "cannot read image: " << line;
      continue;
    }
    images.push_back(image);
  }
  imagey.close();
  for (size_t i = 0; i < images.size(); i++) {
    inputs.push_back(images[i]);
    if (inputs.size() < batch && i < images.size() - 1) {
      continue;
    }
    auto rets = det->run(inputs);
    for (auto ret : rets) {
      featy.push_back(ret.feat);
    }
    inputs.clear();
  }

  const int x = featx.size();
  const int y = featy.size();
  for (int i = 0; i < x; ++i) {
    for (int j = 0; j < y; ++j) {
      double rd = cosine_distance(featx[i], featy[j]);
      fprintf(out_fs, "%.3lf ", rd);
    }
    fprintf(out_fs, "\n");
  }
  fclose(out_fs);
  return 0;
}

