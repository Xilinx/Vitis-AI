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

#include <glog/logging.h>
#include <json-c/json.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/openpose.hpp>

using namespace cv;
using namespace std;

using Peak = std::tuple<int, float, cv::Point2f>;
using PosePoint = vitis::ai::OpenPoseResult::PosePoint;

bool g_is_print = false;

void parseImage(vitis::ai::OpenPose *openpose, const vector<cv::Mat> &img,
                const vector<std::string> &single_name, std::ofstream &out);
std::string get_single_name(const std::string &line);
int main(int argc, char *argv[]) {
  auto openpose = vitis::ai::OpenPose::create("openpose_pruned_0_3");
  if (!openpose) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  size_t batch = openpose->get_input_batch();
  std::ofstream out_fs(argv[2], std::ofstream::out);
  std::ifstream fs(argv[1]);
  std::string line;
  std::string single_name;
  out_fs << "[";
  vector<Mat> input_images;
  vector<string> single_names;
  while (getline(fs, line)) {
    auto image = cv::imread(line);
    if (image.empty()) {
      cerr << "cannot read image: " << line;
      continue;
    }
    input_images.push_back(image);
    single_names.push_back(get_single_name(line));
    if (input_images.size() < batch) {
      continue;
    }
    CHECK(input_images.size() == single_names.size())
        << "images'size is not equall names's";
    parseImage(openpose.get(), input_images, single_names, out_fs);
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

void parseImage(vitis::ai::OpenPose *openpose, const vector<cv::Mat> &imgs,
                const vector<std::string> &single_names, std::ofstream &out) {
  size_t batch = openpose->get_input_batch();
  CHECK(imgs.size() == batch) << "input images'size is not equall batch";

  auto results_vec = openpose->run(imgs);
  for (size_t k = 0; k < results_vec.size(); ++k) {
    auto results = results_vec[k];
    auto single_name = single_names[k];
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
                              json_object_new_double(jo[i].point.x));
        json_object_array_add(point_array,
                              json_object_new_double(jo[i].point.y));
        json_object_array_add(point_array, json_object_new_int(jo[i].type));
      }
      json_object_object_add(human, humanid.c_str(), point_array);
    }

    json_object_object_add(value, "keypoint_annotations", human);
    if (g_is_print) {
      out << ", ";
    } else {
      g_is_print = true;
    }
    out << json_object_to_json_string(value);
    json_object_put(value);
  }
}
