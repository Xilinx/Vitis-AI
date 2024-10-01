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

#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/efficientdet_d2.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace cv;
using namespace std;

DEF_ENV_PARAM(DEBUG_ACC, "0");
DEF_ENV_PARAM(DEBUG_ACC_SAVE, "0");

static void LoadImageNames(std::string const& filename,
                           std::vector<std::string>& images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
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

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "usage :" << argv[0] << " <model_name> "
              << "<database_path> <image_list> <output_file>" << std::endl;
    abort();
  }

  string kernel = string(argv[1]) + "_acc";
  vector<string> names;

  auto efficientdet_d2 = vitis::ai::EfficientDetD2::create(kernel, true);
  if (!efficientdet_d2) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto database_path = std::string(argv[2]);

  LoadImageNames(argv[3], names);
  auto output_file = std::string(argv[4]);

  // int width = efficientdet_d2->getInputWidth();
  // int height = efficientdet_d2->getInputHeight();
  int batch = efficientdet_d2->get_input_batch();

  // std::cout << "input width " << width << " "    //
  //          << "input height " << height << " "  //
  //          << std::endl;

  ofstream out(output_file);
  int size = names.size();
  for (auto i = 0; i < size; i = i + batch) {
    auto valid_batch = size - i >= batch ? batch : size - i;
    auto batch_names = std::vector<std::string>(valid_batch);
    auto batch_images = std::vector<cv::Mat>(valid_batch);
    // auto batch_input_images = std::vector<cv::Mat>(valid_batch);
    auto batch_scales = std::vector<float>(valid_batch);

    for (auto j = 0; j < valid_batch; ++j) {
      auto name = names[i + j];
      auto load_name = database_path + "/" + name + ".jpg";
      cv::Mat img = cv::imread(load_name);
      if (img.empty()) {
        std::cout << "cannot load " << load_name << std::endl;
        exit(0);
      }
      if (ENV_PARAM(DEBUG_ACC)) {
        LOG(INFO) << "load:" << name;
      }
      // std::cout << "image width " << img.cols << " "
      //          << "image height " << img.rows << " "
      //          << std::endl;
      batch_names[j] = name;
      batch_images[j] = img;
      // batch_input_images[j] = img;
      // batch_scales[j] = scale;
      // LOG(INFO) << "batch:" << j << ", scale:" << scale;
    }
    //
    // auto results = efficientdet_d2->run(input_image);
    // auto batch_results = efficientdet_d2->run(batch_input_images);
    auto batch_results = efficientdet_d2->run(batch_images);
    for (auto b = 0; b < valid_batch; ++b) {
      auto& result = batch_results[b];
      for (auto i = 0u; i < result.bboxes.size(); ++i) {
        auto& box = result.bboxes[i];
        auto label = box.label + 1;

        float fxmin = box.x * batch_images[b].cols;
        float fymin = box.y * batch_images[b].rows;

        float fwidth = box.width * batch_images[b].cols;
        float fheight = box.height * batch_images[b].rows;

        float fxmax = fxmin + fwidth;
        float fymax = fymin + fheight;
        float confidence = box.score;

        int xmin = round(fxmin * 100.0) / 100.0;
        int ymin = round(fymin * 100.0) / 100.0;
        int xmax = round(fxmax * 100.0) / 100.0;
        int ymax = round(fymax * 100.0) / 100.0;
        // int w = round(fwidth * 100.0) / 100.0;
        // int h = round(fheight * 100.0) / 100.0;

        xmin = std::min(std::max(xmin, 0), batch_images[b].cols);
        xmax = std::min(std::max(xmax, 0), batch_images[b].cols);
        ymin = std::min(std::max(ymin, 0), batch_images[b].rows);
        ymax = std::min(std::max(ymax, 0), batch_images[b].rows);

        // out << batch_names[b] << " " << xmin << " " << ymin << " " << xmax
        //    << " " << ymax << " " << confidence << " " << label << "\n";
        out << batch_names[b] << " " << fxmin << " " << fymin << " " << fwidth
            << " " << fheight << " " << confidence << " " << label << "\n";
        rectangle(batch_images[b], Point(xmin, ymin), Point(xmax, ymax),
                  Scalar(0, 255, 0), 1, 1, 0);
      }
      if (ENV_PARAM(DEBUG_ACC_SAVE)) {
        cv::imwrite(batch_names[b] + "_result.jpg", batch_images[b]);
      }
    }
  }
  out.close();
  return 0;
}
