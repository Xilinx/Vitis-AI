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
#include <vector>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/profiling.hpp>

extern int GLOBAL_ENABLE_C_SOFTMAX;

using namespace std;
using namespace cv;
static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
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

int main(int argc, char *argv[]) {
  GLOBAL_ENABLE_C_SOFTMAX = 2;

  if (argc < 4) {
    std::cerr << "usage ：" << argv[0]
              << " image_list_file output_file model_directory" << std::endl;
    abort();
  }
  bool preprocess = !(getenv("PRE") != nullptr);
  auto v = vitis::ai::FaceDetect::create(argv[3], preprocess);
  if (!v) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
            << "]"  //
                    //<< "width " << width << " " //
                    //<< "height " << height << " "            //
            << "pre " << preprocess << " "
            << "v.get() " << (void *)v.get() << " "  //
            << std::endl;
  int width = v->getInputWidth();
  int height = v->getInputHeight();

  vector<string> names;
  LoadImageNames(argv[1], names);
  ofstream out(argv[2]);
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;
  for (auto& name : names) {
    // cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ":"
    //    << "image " << argv[i] << " "; //
    // cv::Mat image = cv::imread(argv[i]);
    cv::Mat img_resize;

    cout << name << endl;
    cv::Mat image = cv::imread(name);
    cv::Mat canvas(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    // cv::imwrite("out1.jpg", canvas);
    float scale = (float)image.cols / (float)image.rows;
    float network_scale = (float)width / (float)height;
    if (scale >= network_scale) {
      resize(canvas, canvas,
             cv::Size(image.cols, ceil((float)image.cols / network_scale)));
    } else {
      resize(canvas, canvas,
             cv::Size(ceil((float)image.rows * network_scale), image.rows));
    }
    auto namesp = split(name, "/FDDB_images/");
    auto logic_name = split(namesp[1], ".")[0];

    cout << logic_name << endl;

    cout << image.size() << " " << canvas.size() << endl;

    image.copyTo(canvas(cv::Rect_<int>(0, 0, image.cols, image.rows)));
    cv::resize(canvas, img_resize, cv::Size(height, width));

    __TIC__(FACE_DET_TOTLE)
    auto result = v->run(img_resize);
    __TOC__(FACE_DET_TOTLE)
    out << logic_name << endl;
    out << result.rects.size() << endl;
    for (const auto &r : result.rects) {
      cout << " " << r.score << " :q"  //
           << r.x << " "               //
           << r.y << " "               //
           << r.width << " "           //
           << r.height << " "          //
           << endl;
      ;
      cv::rectangle(image,
                    cv::Rect{cv::Point(r.x * canvas.cols, r.y * canvas.rows),
                             cv::Size{(int)(r.width * canvas.cols),
                                      (int)(r.height * canvas.rows)}},
                    0xff);
      out << r.x * canvas.cols << " " << r.y * canvas.rows << " "
          << r.width * canvas.cols << " " << r.height * canvas.rows << " "
          << r.score << " " << endl;
    }
    // cv::imwrite("out.jpg", image);
    canvas.release();
    cout << std::endl;
  }

  return 0;
}
