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

const string g_output_dir = "./result/";
const string g_output_img = "./result_img/";
const string g_input_img = "./landmark_image/";
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <img_url> [<img_url> ...]"
              << std::endl;  //
    abort();
  }

  auto landmark = FaceLandmark::create("face_landmark");
  if (!landmark) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  int width = landmark->getInputWidth();
  int height = landmark->getInputHeight();

  std::cout << "InputWidth " << width << " "
            << "InputHeight " << height << " " << std::endl;

  vector<string> names;
  LoadImageNames(argv[1], names);
  for (auto& name : names) {
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ":"
              << "image " << name << " " << std::endl;
    cv::Mat image = cv::imread(g_input_img + name);

    replace_all(name, ".jpg", "");
    std::cout << "output path " << g_output_dir + name + ".txt" << endl;
    ofstream out(g_output_dir + name + ".txt");
    if (image.empty()) {
      std::cout << "cannot load " << name << std::endl;
      continue;
    }

    cv::Mat img_resize;
    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    // cv::imwrite("resize_after.jpg", img_resize);
    // std::cout << "resize success,will be run!"<< std::endl; //

    auto result = landmark->run(img_resize);
    auto points = result.points;

    // std::cout << "score " << result.score << " "
    //           << "gender " << (float)result.gender << " " //
    //           << "age " << (int)result.age << " ";        //

    std::cout << "points ";  //
    for (int i = 0; i < 5; ++i) {
      std::cout << points[i].first << " " << points[i].second << " ";
      auto pt = cv::Point{(int)(points[i].first * img_resize.cols),
                          (int)(points[i].second * img_resize.rows)};
      out << (int)(points[i].first * img_resize.cols) << " ";
      cv::circle(img_resize, pt, 3, cv::Scalar(0, 255, 255));
    }
    for (int i = 0; i < 5; i++) {
      out << (int)(points[i].second * img_resize.rows) << " ";
    }
    std::cout << std::endl;
    cv::imwrite(g_output_img + name + "_out.jpg", img_resize);
    out.close();
  }
  return 0;
}
