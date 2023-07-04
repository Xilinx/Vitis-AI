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
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/yolov2.hpp>
using namespace std;
using namespace cv;

static std::vector<std::string> VOC_map{
    "aeroplane",   "bicycle", "bird",  "boat",      "bottle",
    "bus",         "car",     "cat",   "chair",     "cow",
    "diningtable", "dog",     "horse", "motorbike", "person",
    "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"};

int main(int argc, char *argv[]) {
  std::map<std::string, std::string> namemap{
      {"BASELINE", "yolov2_voc_baseline"},
      {"COMPRESS22G", "yolov2_voc_compress22G"},
      {"COMPRESS24G", "yolov2_voc_compress24G"},
      {"COMPRESS26G", "yolov2_voc_compress26G"},
  };
  if (argc != 3) {
    cerr << "usage: test_ssd_accuracy model_type image_path " << endl
         << "model_type is one of below  " << endl
         << "   BASELINE" << endl
         << "   COMPRESS22G" << endl
         << "   COMPRESS24G" << endl
         << "   COMPRESS26G" << endl;

    return -1;
  }
  string g_model_name = "none";
  g_model_name = argv[1];
  if (namemap.find(g_model_name) == namemap.end()) {
    cerr << "model_type is one of below  " << endl
         << "   BASELINE" << endl
         << "   COMPRESS22G" << endl
         << "   COMPRESS24G" << endl
         << "   COMPRESS26G" << endl
         << " it is " << g_model_name << endl;
    return -1;
  }

  cv::String path = argv[2];
  int length = path.size();
  vector<cv::String> files;
  cv::glob(path, files);

  auto yolo =
      vitis::ai::YOLOv2::create(namemap[g_model_name] + string("_acc"), true);
  if (!yolo) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  int count = files.size();
  cerr << "The image count = " << count << endl;
  for (int i = 0; i < count; i++) {
    auto img = imread(files[i]);
    auto results = yolo->run(img);
    for (auto &box : results.bboxes) {
      std::string label_name = VOC_map[box.label];
      // int label = box.label;
      float xmin = box.x * img.cols + 1;
      float ymin = box.y * img.rows + 1;
      float xmax = (box.x + box.width) * img.cols + 1;
      float ymax = (box.y + box.height) * img.rows + 1;
      if (xmin < 0) xmin = 1;
      if (ymin < 0) ymin = 1;
      if (xmax > img.cols) xmax = img.cols;
      if (ymax > img.rows) ymax = img.rows;
      float confidence = box.score;
      string imgname = String(files[i]).substr(length);
      std::cout << imgname.substr(0, 6) << " " << label_name << " "
                << confidence << " " << xmin << " " << ymin << " " << xmax
                << " " << ymax << std::endl;
    }
  }
  return 0;
}
