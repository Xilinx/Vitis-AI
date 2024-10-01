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
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/yolov3.hpp>
using namespace std;
using namespace cv;

static std::vector<std::string> VOC_map{
    "aeroplane",   "bicycle", "bird",  "boat",      "bottle",
    "bus",         "car",     "cat",   "chair",     "cow",
    "diningtable", "dog",     "horse", "motorbike", "person",
    "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"};

static std::vector<std::string> cityscapes_map_yolo{"car", "person", "cycle"};

static std::vector<std::string> bdd_map_yolo{
    "bike",  "bus",   "car",  "motor", "person",
    "rider", "light", "sign", "train", "truck"};

static std::vector<std::string> coco_map_yolo{
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"};

vector<int> coco_id_map_dict() {
  vector<int> category_ids;
  category_ids = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                  62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                  80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
  return category_ids;
}

int imagename_to_id(string imagename) {
  int idx1 = imagename.find_last_of('.');
  int idx2 = imagename.find_last_of('_');
  string id = imagename.substr(idx2 + 1, idx1 - idx2 - 1);
  int image_id = atoi(id.c_str());
  return image_id;
}

static std::vector<std::string> split(const std::string& s,
                                      const std::string& delim) {
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

void LoadImageNames(std::string const& filename,
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
  std::map<std::string, std::string> namemap{
      {"YOLOV3_VOC_416x416", "yolov3_voc_416"},
      {"YOLOV3_ADAS_512x256", "yolov3_adas_512x256"},
      {"YOLOV3_ADAS_512x288", "yolov3_adas_512x288"},
      {"yolov4", "yolov4"},
  };
  if (argc != 4) {
    cerr << "usage: test_accuracy_yolov3 model_type file_list_file_name "
         << endl
         << "model_type is one of below  " << endl
         << "   YOLOV3_VOC_416x416" << endl
         << "   YOLOV3_ADAS_512x256" << endl
         << "   YOLOV3_ADAS_512x288" << endl;

    return -1;
  }
  string g_model_name = "none";
  g_model_name = argv[3];
  if (namemap.find(g_model_name) == namemap.end()) {
    cerr << "model_type is one of below  " << endl
         << "   YOLOV3_VOC_416x416" << endl
         << "   YOLOV3_ADAS_512x256" << endl
         << "   YOLOV3_ADAS_512x288" << endl
         << " it is " << g_model_name << endl;
    return -1;
  }

  vector<string> names;
  LoadImageNames(argv[1], names);
  ofstream out(argv[2]);
  auto model_name = namemap[g_model_name] + string("_acc");
  auto yolo = vitis::ai::YOLOv3::create(model_name, true);
  if (!yolo) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  if (g_model_name == "yolov4") {
    auto ccoco_id_map_dict = coco_id_map_dict();
    out << "[" << endl;
    for (auto& name : names) {
      cv::Mat img = cv::imread(name);
      auto results = yolo->run(img);
      for (auto& box : results.bboxes) {
        float xmin = box.x * img.cols;
        float ymin = box.y * img.rows;
        float xmax = (box.x + box.width) * img.cols;
        float ymax = (box.y + box.height) * img.rows;
        if (xmin < 0) xmin = 1;
        if (ymin < 0) ymin = 1;
        if (xmax > img.cols) xmax = img.cols;
        if (ymax > img.rows) ymax = img.rows;
        float confidence = box.score;
        cout << fixed << setprecision(0)
             << "{\"image_id\":" << imagename_to_id(name)
             << ", \"category_id\":" << ccoco_id_map_dict[box.label]
             << ", \"bbox\":[" << fixed << setprecision(6) << xmin << ", "
             << ymin << ", " << xmax - xmin << ", " << ymax - ymin
             << "], \"score\":" << confidence << "}," << endl;
        out << fixed << setprecision(0)
            << "{\"image_id\":" << imagename_to_id(name)
            << ", \"category_id\":" << ccoco_id_map_dict[box.label]
            << ", \"bbox\":[" << fixed << setprecision(6) << xmin << ", "
            << ymin << ", " << xmax - xmin << ", " << ymax - ymin
            << "], \"score\":" << confidence << "}," << endl;

        out.flush();
      }
    }
    out.seekp(-2L, ios::end);
    out << endl << "]" << endl;
    out.close();
    return 0;
  }
  for (auto name : names) {
    cv::Mat img = cv::imread(name);
    auto results = yolo->run(img);
    auto namesp = split(name, "/");
    auto single_name = namesp[namesp.size() - 1];
    single_name = split(single_name, ".")[0];
    for (auto& box : results.bboxes) {
      std::string label_name = "none";
      if (g_model_name == "YOLOV3_VOC_416x416") {
        label_name = VOC_map[box.label];
      } else if (g_model_name == "YOLOV3_ADAS_512x256") {
        label_name = cityscapes_map_yolo[box.label];
      } else if (g_model_name == "YOLOV3_ADAS_512x288") {
        label_name = bdd_map_yolo[box.label];
      } else {
        cout << "^_^" << endl;
        return 0;
      }

      // out << single_name << " " << label_name << " " << box.score << " "
      //    << box.x*img.cols << " " << box.y*img.rows << " "
      //    << (box.x + box.width)*img.cols << " " << (box.y +
      //    box.height)*img.rows
      //    << std::endl;
      // out.flush();

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
      std::cout << single_name << " " << label_name << " " << confidence << " "
                << xmin << " " << ymin << " " << xmax << " " << ymax
                << std::endl;
      out << single_name << " " << label_name << " " << confidence << " "
          << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
      out.flush();
    }
  }
  out.close();
  return 0;
}
