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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/posedetect.hpp>

using namespace std;
static void DrawLines(cv::Mat& img,
                      const vitis::ai::PoseDetectResult::Pose14Pt& results);

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
  auto det = vitis::ai::PoseDetect::create("ssd_pedestrian_pruned_0_97");
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  vector<string> names;
  LoadImageNames(argv[1], names);

  for (auto & name : names) {
    cout << name << endl;
    cv::Mat image = cv::imread(name);

    auto result = det->run(image);
    auto ret = system("mkdir -p ./result/");
    if (ret != 0) {
      cerr << "Can not create result directory!" << endl;
    }

    auto result_path = "./result/";
    auto namesp = split(name, "/");
    auto truename = split(namesp[namesp.size() - 1], ".")[0];
    ofstream fo(result_path + truename + ".txt");
    vector<float> pose14pt_arry((float*)&result.pose14pt,
                                (float*)&result.pose14pt + 28);
    for (size_t i = 0; i < pose14pt_arry.size(); i = i + 2) {
      fo << pose14pt_arry[i] * 128.f << " " << pose14pt_arry[i + 1] * 224.f
         << endl;
      cout << "(" << pose14pt_arry[i] * 128.f << ","
           << pose14pt_arry[i + 1] * 224.f << ")" << endl;
    }

    DrawLines(image, result.pose14pt);
    // return 0;

    imwrite("res.jpg", image);
  }
  return 0;
}

using namespace cv;
static inline void DrawLine(Mat& img, Point2f point1, Point2f point2,
                            Scalar colour, int thickness, float scale_w,
                            float scale_h) {
  if ((point1.x * img.cols > scale_w || point1.y * img.rows > scale_h) &&
      (point2.x * img.cols > scale_w || point2.y * img.rows > scale_h))
    cv::line(img, Point2f(point1.x * img.cols, point1.y * img.rows),
             Point2f(point2.x * img.cols, point2.y * img.rows), colour,
             thickness);
}

static void DrawLines(Mat& img,
                      const vitis::ai::PoseDetectResult::Pose14Pt& results) {
  float scale_w = 1;
  float scale_h = 1;

  float mark = 5.f;
  float mark_w = mark * scale_w;
  float mark_h = mark * scale_h;
  std::vector<Point2f> pois(14);
  for (size_t i = 0; i < pois.size(); ++i) {
    pois[i].x = ((float*)&results)[i * 2] * img.cols;
    // std::cout << ((float*)&results)[i * 2] << " " << ((float*)&results)[i * 2
    // + 1] << std::endl;
    pois[i].y = ((float*)&results)[i * 2 + 1] * img.rows;
  }
  for (size_t i = 0; i < pois.size(); ++i) {
    circle(img, pois[i], 3, Scalar::all(255));
  }
  DrawLine(img, results.right_shoulder, results.right_elbow, Scalar(255, 0, 0),
           2, mark_w, mark_h);
  DrawLine(img, results.right_elbow, results.right_wrist, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_hip, results.right_knee, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_knee, results.right_ankle, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.left_elbow, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_elbow, results.left_wrist, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_hip, results.left_knee, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_knee, results.left_ankle, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.head, results.neck, Scalar(0, 255, 255), 2, mark_w,
           mark_h);
  DrawLine(img, results.right_shoulder, results.neck, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.neck, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_shoulder, results.right_hip, Scalar(0, 255, 255),
           2, mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.left_hip, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_hip, results.left_hip, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
}
