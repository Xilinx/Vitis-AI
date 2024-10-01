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
#include <vitis/ai/openpose.hpp>

using namespace cv;
using namespace std;

using Peak = std::tuple<int, float, cv::Point2f>;
using PosePoint = vitis::ai::OpenPoseResult::PosePoint;

void parseImage(vitis::ai::OpenPose *openpose, cv::Mat &img,
                const std::string &single_name, std::ofstream &out);
std::string get_single_name(const std::string &line);
int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " image_list_file  output_file"
              << std::endl;
    return -1;
  }
  auto openpose = vitis::ai::OpenPose::create("openpose_pruned_0_3");
  if (!openpose) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  std::ifstream fs(argv[1]);
  std::ofstream out_fs(argv[2], std::ofstream::out);
  std::string line;
  std::string single_name;
  out_fs << "[";
  bool is_print = false;
  while (getline(fs, line)) {
    if (is_print) {
      out_fs << ", ";
    } else
      is_print = true;
    auto image = cv::imread(line);
    if (image.empty()) {
      cerr << "cannot read image: " << line;
      continue;
    }
    single_name = get_single_name(line);
    parseImage(openpose.get(), image, single_name, out_fs);
  }
  out_fs << "]";
  fs.close();
  out_fs.close();
  return 0;
}

std::string get_single_name(const std::string &line) {
  std::size_t found = line.rfind('/');
  if (found != std::string::npos) {
    return line.substr(found + 1);
  }
  return line;
}

void parseImage(vitis::ai::OpenPose *openpose, cv::Mat &img,
                const std::string &single_name, std::ofstream &out) {
  int width = openpose->getInputWidth();
  int height = openpose->getInputHeight();
  Mat res_img;
  resize(img, res_img, Size(width, height));

  auto results = openpose->run(res_img);
  float scale_x = float(img.cols) / float(width);
  float scale_y = float(img.rows) / float(height);

  json_object *str_imageid = json_object_new_string(
      single_name.substr(0, single_name.size() - 4).c_str());
  json_object *value = json_object_new_object();
  json_object_object_add(value, "image_id", str_imageid);
  json_object *human = json_object_new_object();
  vector<int> a = {5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 0, 1};
  for (size_t k = 1; k < results.poses.size(); ++k) {
    PosePoint posePoint;
    vector<PosePoint> jo(14, posePoint);
    vector<PosePoint> j = results.poses[k];
    string humanid = "human";
    humanid += to_string(results.poses.size() - k);
    json_object *point_array = json_object_new_array();
    for (int i = 0; i < 14; ++i) {
      jo[i] = j[a[i]];
      json_object_array_add(point_array,
                            json_object_new_double(jo[i].point.x * scale_x));
      json_object_array_add(point_array,
                            json_object_new_double(jo[i].point.y * scale_y));
      json_object_array_add(point_array, json_object_new_int(jo[i].type));
    }
    json_object_object_add(human, humanid.c_str(), point_array);
  }

  json_object_object_add(value, "keypoint_annotations", human);
  out << json_object_to_json_string(value);
  json_object_put(value);
}
