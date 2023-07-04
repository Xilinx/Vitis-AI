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

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/medicalsegmentation.hpp>

using namespace std;
using namespace cv;

Scalar colors[] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
                   Scalar(255, 0, 255), Scalar(0, 255, 255)};
std::vector<string> classTypes = {"BE", "cancer", "HGD", "polyp", "suspicious"};

void LoadImageNames(std::string const &filename,
                    std::vector<std::string> &images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE *fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name> <image_list_file>  <result_dir>" 
              << "\n        model_name is FPN_Res18_Medical_segmentation" << std::endl;
    abort();
  }

  auto seg =
      vitis::ai::MedicalSegmentation::create(argv[1]);
  if (!seg) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  vector<string> names;
  LoadImageNames(argv[2], names);

  std::string pathbase(argv[3]);

  // if dir doesn't exist, create it.
  for (int i = 0; i < 6; i++) {
    std::string path = pathbase + "/results";
    if (i != 0) {
      path = path + "/" + classTypes[i - 1];
    }
    auto ret = mkdir(path.c_str(), 0777);
    if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
      std::cout << "error occured when mkdir " << path << std::endl;
      return -1;
    }
  }
  for (auto& name : names) {
    cv::Mat img_save;
    cv::Mat img = cv::imread(name);
    cv::Size size_orig = img.size();

    std::string filenamepart1 = name.substr(name.find_last_of('/') + 1);
    filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));

    auto result = seg->run(img);
    for (int i = 0; i < 5; i++) {
      std::string fname(pathbase + "/results/" + classTypes[i] + "/" +
                        filenamepart1 + ".png");
      cv::resize(result.segmentation[i], img_save, size_orig, 0, 0,
                 cv::INTER_LINEAR);
      cv::imwrite(fname, img_save);
    }
  }

  return 0;
}
