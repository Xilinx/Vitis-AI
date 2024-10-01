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
#include <json-c/json.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/tfssd.hpp>

using namespace cv;
using namespace std;

void parseImage(vitis::ai::TFSSD *tfssd, cv::Mat &img,
                const std::string &single_name, std::ofstream &out);
std::string get_single_name(const std::string &line);
bool is_print = false;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " model_name image_list_file  output_file" << std::endl;
    return -1;
  }
  auto model_name = argv[1] + string("_acc");
  auto tfssd = vitis::ai::TFSSD::create(model_name, true);
  if (!tfssd) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  std::ifstream fs(argv[2]);
  std::ofstream out_fs(argv[3], std::ofstream::out);
  std::string line;
  std::string single_name;
  out_fs << "[";
  while (getline(fs, line)) {
    auto image = cv::imread(line);
    if (image.empty()) {
      cerr << "cannot read image: " << line;
      continue;
    }
    single_name = get_single_name(line);
    parseImage(tfssd.get(), image, single_name, out_fs);
  }
  out_fs << "]";
  fs.close();
  out_fs.close();
  return 0;
}

std::string get_single_name(const std::string &line) {
  std::size_t found = line.rfind('/');
  if (found != std::string::npos) {
    return line.substr(found + 1 + 13, line.length() - found - 5);
  }
  return line;
}

void parseImage(vitis::ai::TFSSD *tfssd, cv::Mat &img,
                const std::string &single_name, std::ofstream &out) {
  // int width = tfssd->getInputWidth();
  // int height = tfssd->getInputHeight();
  int width = img.cols;
  int height = img.rows;

  auto result = tfssd->run(img);
  for (auto &r : result.bboxes) {
    if (is_print) {
      out << ",\n";
    } else {
      is_print = true;
    }
    json_object *objitem = json_object_new_object();
    json_object *str_image_id = json_object_new_int(std::stoi(single_name));
    json_object_object_add(objitem, "image_id", str_image_id);
    json_object *str_category_id = json_object_new_int(r.label);
    json_object_object_add(objitem, "category_id", str_category_id);
    json_object *str_score = json_object_new_double(r.score);
    json_object_object_add(objitem, "score", str_score);
    json_object *bbox_array = json_object_new_array();
    json_object_array_add(bbox_array, json_object_new_double(r.x * width));
    json_object_array_add(bbox_array, json_object_new_double(r.y * height));
    json_object_array_add(bbox_array, json_object_new_double(r.width * width));
    json_object_array_add(bbox_array,
                          json_object_new_double(r.height * height));
    json_object_object_add(objitem, "bbox", bbox_array);
    out << json_object_to_json_string(objitem);
    json_object_put(objitem);
  }
}
