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
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/facelandmark.hpp>
using namespace vitis::ai;

using namespace std;
using namespace cv;

static string &replace_all(string &str, const string &old_value,
                           const string &new_value) {
  while (true) {
    string::size_type pos(0);
    if ((pos = str.find(old_value)) != string::npos)
      str.replace(pos, old_value.length(), new_value);
    else
      break;
  }
  return str;
}

std::string get_single_name(const std::string &line) {
  std::size_t found = line.rfind('/');
  if (found != std::string::npos) {
    return line.substr(found + 1);
  }
  return line;
}
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

//const string g_output_dir = "./result/";
//const string g_output_img = "./result_img/";
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << " usage: " << argv[0] << " <model_name> <image_list> <output_file>"
              << std::endl;  //
    abort();
  }
//  auto output_dir = std::string(argv[3]);

  auto landmark = FaceLandmark::create(argv[1]);
  if (!landmark) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  int width = landmark->getInputWidth();
  int height = landmark->getInputHeight();

  vector<string> names;
  LoadImageNames(argv[2], names);
  
//  auto ret = access(output_dir.c_str(), F_OK);
//  if (ret == -1) {
//    std::cout << "make output_dir: " << output_dir.c_str() << std::endl;
//    mkdir(output_dir.c_str(), 0755);
//  }

  std::ofstream out(argv[3], std::ofstream::out);

  for (auto& name : names) {
    cv::Mat image = cv::imread(name);
    auto single_name = get_single_name(name);
    replace_all(single_name, ".jpg", "");
//    ofstream out(output_dir + "/" + single_name + ".txt");
    out << single_name << " ";
    if (image.empty()) {
      std::cout << "cannot load " << name << std::endl;
      continue;
    }

    cv::Mat img_resize;
    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);

    auto result = landmark->run(img_resize);
    auto points = result.points;

    for (int i = 0; i < 5; ++i) {
      auto pt = cv::Point{(int)(points[i].first * img_resize.cols),
                          (int)(points[i].second * img_resize.rows)};
      out << (int)(points[i].first * img_resize.cols) << " ";
      cv::circle(img_resize, pt, 3, cv::Scalar(0, 255, 255));
    }
    for (int i = 0; i < 5; i++) {
      out << (int)(points[i].second * img_resize.rows) << " ";
    }
    out << std::endl;
    // cv::imwrite(g_output_img + name + "_out.jpg", img_resize);
//    out.close();
  }
  out.close();
  return 0;
}
