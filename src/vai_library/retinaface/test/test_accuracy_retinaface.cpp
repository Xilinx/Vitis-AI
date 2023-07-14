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
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/retinaface.hpp>

using namespace cv;
using namespace std;
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
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_list> <output_file>" << std::endl;
    abort();
  }

  string kernel = argv[1];
  vector<string> names;

  auto retinaface = vitis::ai::RetinaFace::create(kernel, true);
  LoadImageNames(argv[2], names);
  auto output_file = std::string(argv[3]);

  int width = retinaface->getInputWidth();
  int height = retinaface->getInputHeight();

  std::cout << "input width " << width << " "    //
            << "input height " << height << " "  //
            << std::endl;

  int valid_width = 640;
  int valid_height = 360;
  std::cout << "valid_width " << valid_width << " "    //
            << "valid_height " << valid_height << " "  //
            << std::endl;

  ofstream out(output_file);
  for (auto name : names) {
    cv::Mat img = cv::imread(name);
    if (img.empty()) {
      std::cout << "cannot load " << name << std::endl;
      continue;
    }

    std::cout << "image width " << img.cols << " "
              << "image height " << img.rows << " "
              << std::endl;


    float scale = 360.0 / img.rows;
    if (img.cols * scale > 640) {
      scale = 640.0 / img.cols;
    }
    cout << "scale :" << scale << endl;
    out << name << endl;
    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(img.cols * scale, img.rows * scale),
               0, 0, cv::INTER_LINEAR);
    std::cout << "resize width " << img_resize.cols<< " "    //
            << " height " << img_resize.rows << std::endl;
    cv::Mat input_image(384, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat roi = input_image(cv::Range{0, img_resize.rows}, cv::Range{0, img_resize.cols});
    img_resize.copyTo(roi);
    auto results = retinaface->run(input_image);
    out << results.bboxes.size() << endl;
    for (auto i = 0u; i < results.bboxes.size(); ++i) {
      auto &box = results.bboxes[i];
      float fxmin = box.x * width / scale;  
      float fymin = box.y * height / scale; 
      float fwidth = box.width * width / scale; 
      float fheight = box.height * height / scale;
      //if (flip && n == 1) {
      //  fxmin = (img_resize.cols - box.x * width) / scale;
      //}
      float fxmax = fxmin + fwidth; 
      float fymax = fymin + fheight;
      float confidence = box.score;

      int xmin = round(fxmin * 100.0) / 100.0;
      int ymin = round(fymin * 100.0) / 100.0;
      int xmax = round(fxmax * 100.0) / 100.0;
      int ymax = round(fymax * 100.0) / 100.0;
      int w = round(fwidth * 100.0) / 100.0;
      int h = round(fheight * 100.0) / 100.0;

      xmin = std::min(std::max(xmin, 0), img.cols);
      xmax = std::min(std::max(xmax, 0), img.cols);
      ymin = std::min(std::max(ymin, 0), img.rows);
      ymax = std::min(std::max(ymax, 0), img.rows);
      w = std::min(std::max(w, 0), img.rows);
      h = std::min(std::max(h, 0), img.rows);

      out << "RESULT: " << "\t" << xmin << "\t" << ymin << "\t" << w 
           << "\t" << h << "\t" << confidence << "\n";
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
                  1, 0);
      out << endl;
    }
  }
  out.close();
  return 0;
}
