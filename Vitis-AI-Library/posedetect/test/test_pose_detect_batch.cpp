/*
 * Copyright 2019 Xilinx Inc.
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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;
static void DrawLines(cv::Mat &img,
                      const vitis::ai::PoseDetectResult::Pose14Pt &results);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }

  auto det = vitis::ai::PoseDetect::create(argv[1]);


  std::vector<cv::Mat> arg_input_images;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_names.push_back(argv[i]);

  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }


  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_image_names;
  auto batch = det->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(arg_input_images[batch_idx % arg_input_images.size()]);
    batch_image_names.push_back(arg_input_images_names[batch_idx % arg_input_images.size()]);
  }



  auto results = det->run(batch_images);

  int i = 0;
  for (const auto &result : results) {
    auto image = batch_images[i];
    std::cout << "result: " << i << std::endl;
    std::cout << "(" << result.pose14pt.right_shoulder.x << ","
              << result.pose14pt.right_shoulder.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.right_elbow.x << ","
              << result.pose14pt.right_elbow.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.right_wrist.x << ","
              << result.pose14pt.right_wrist.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.left_shoulder.x << ","
              << result.pose14pt.left_shoulder.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.left_elbow.x << ","
              << result.pose14pt.left_elbow.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.left_wrist.x << ","
              << result.pose14pt.left_wrist.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.right_hip.x << ","
              << result.pose14pt.right_hip.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.right_knee.x << ","
              << result.pose14pt.right_knee.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.right_ankle.x << ","
              << result.pose14pt.right_ankle.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.left_hip.x << ","
              << result.pose14pt.left_hip.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.left_knee.x << ","
              << result.pose14pt.left_knee.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.left_ankle.x << ","
              << result.pose14pt.left_ankle.y << ")" << std::endl;
    std::cout << "(" << result.pose14pt.head.x << "," << result.pose14pt.head.y
              << ")" << std::endl;
    std::cout << "(" << result.pose14pt.neck.x << "," << result.pose14pt.neck.y
              << ")" << std::endl;
    DrawLines(image, result.pose14pt);
    i++;
  }
  return 0;
}

using namespace cv;
static inline void DrawLine(Mat &img, Point2f point1, Point2f point2,
                            Scalar colour, int thickness, float scale_w,
                            float scale_h) {
  if ((point1.x * img.cols > scale_w || point1.y * img.rows > scale_h) &&
      (point2.x * img.cols > scale_w || point2.y * img.rows > scale_h))
    cv::line(img, Point2f(point1.x * img.cols, point1.y * img.rows),
             Point2f(point2.x * img.cols, point2.y * img.rows), colour,
             thickness);
}

static void DrawLines(Mat &img,
                      const vitis::ai::PoseDetectResult::Pose14Pt &results) {
  float scale_w = 1;
  float scale_h = 1;

  float mark = 5.f;
  float mark_w = mark * scale_w;
  float mark_h = mark * scale_h;
  std::vector<Point2f> pois(14);
  for (size_t i = 0; i < pois.size(); ++i) {
    pois[i].x = ((float *)&results)[i * 2] * img.cols;
    pois[i].y = ((float *)&results)[i * 2 + 1] * img.rows;
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
