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

#include "./facerecog_imp.hpp"
#include "util.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/attrs/attrs.hpp>

DEF_ENV_PARAM(ENABLE_DEBUG_FACE_RECOG, "0");
using Eigen::Map;
using Eigen::Matrix3f;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using std::vector;

namespace vitis {
namespace ai {

FaceRecogImp::FaceRecogImp(const std::string &feature_model_name,
                           bool need_preprocess) {
    auto attrs = xir::Attrs::create();
    landmark_ = std::move(FaceLandmark::create("face_landmark", attrs.get(),
                                               need_preprocess));
    feature_ = std::move(FaceFeature::create(feature_model_name, attrs.get(),
                                             need_preprocess));
}

FaceRecogImp::FaceRecogImp(const std::string &feature_model_name,
                           xir::Attrs *attrs,
                           bool need_preprocess)
    : landmark_{FaceLandmark::create("face_landmark",
                                     attrs,
                                     need_preprocess)},
      feature_{FaceFeature::create(feature_model_name, attrs, need_preprocess)} { //
}

FaceRecogImp::FaceRecogImp(const std::string &landmark_model_name,
                           const std::string &feature_model_name,
                           bool need_preprocess) {
    auto attrs = xir::Attrs::create();
    landmark_ = std::move(FaceLandmark::create(landmark_model_name, attrs.get(),
                                               need_preprocess));
    feature_ = std::move(FaceFeature::create(feature_model_name, attrs.get(),
                                             need_preprocess));
}

FaceRecogImp::FaceRecogImp(const std::string &landmark_model_name,
                           const std::string &feature_model_name,
                           xir::Attrs *attrs,
                           bool need_preprocess)
    : landmark_{FaceLandmark::create(landmark_model_name,
                                      attrs,
                                     need_preprocess)},
      feature_{FaceFeature::create(feature_model_name, attrs,  need_preprocess)} { //
}

FaceRecogImp::~FaceRecogImp() {}

int FaceRecogImp::getInputWidth() const { return landmark_->getInputWidth(); }

int FaceRecogImp::getInputHeight() const { return landmark_->getInputHeight(); }

size_t FaceRecogImp::get_input_batch() const { return landmark_->get_input_batch(); }

FaceRecogFixedResult FaceRecogImp::_runNormalFixed(const cv::Mat &img_expanded,
                                                   int inner_x, int inner_y,
                                                   int inner_w, int inner_h) {
  static int debug_counter = 0;
  CHECK_NE(img_expanded.rows, 0) << "image must not be empty";
  CHECK_NE(img_expanded.cols, 0) << "image must not be empty";
  CHECK_GE(inner_x, 0) << "inner_x must >= 0";
  CHECK_GE(inner_y, 0) << "inner_y must >= 0";
  CHECK_GE(inner_w, 0) << "inner_w must >= 0";
  CHECK_GE(inner_h, 0) << "inner_h must >= 0";
  CHECK_LE(inner_x + inner_w, img_expanded.cols) << "inner_w must <= cols";
  CHECK_LE(inner_y + inner_h, img_expanded.rows) << "inner_h must <= rows";
  auto valid_face_tl_x = inner_x;
  auto valid_face_tl_y = inner_y;
  auto valid_face_width = inner_w;
  auto valid_face_height = inner_h;
  __TIC__(ATT_RESIZE)
  // cv::Mat img_expanded =cv::Mat(rows, cols, CV_8UC3, const_cast<uint8_t
  // *>(input), stride);
  if (ENV_PARAM(ENABLE_DEBUG_FACE_RECOG)) {
    debug_counter++;
    LOG(INFO) << "get img_expanded success will get Rect Image "
              << "img_expanded.cols " << img_expanded.cols << " " //
              << "img_expanded.rows " << img_expanded.rows << " " //
              << "inner_x " << inner_x << " "                     //
              << "inner_y " << inner_y << " "                     //
              << "inner_w " << inner_w << " "                     //
              << "inner_h " << inner_h << " "                     //
              << std::endl;
  }
  cv::Mat img = img_expanded(cv::Rect_<int>(
      valid_face_tl_x, valid_face_tl_y, valid_face_width, valid_face_height));
  cv::Mat resize_5pt;
  /*cv::resize(img, resize_5pt,
             cv::Size(landmark_->getInputWidth(), landmark_->getInputHeight()),
             0, 0, cv::INTER_NEAREST);*/
  cv::resize(img, resize_5pt,
             cv::Size(landmark_->getInputWidth(), landmark_->getInputHeight()),
             0, 0, cv::INTER_LINEAR);

  __TOC__(ATT_RESIZE)
  __TIC__(ATT_RUN)
  auto landmarkResult = landmark_->run(resize_5pt);
  __TOC__(ATT_RUN)
  __TIC__(ATT_POST_PROCESS)

  cv::Mat aligned;
  auto points = landmarkResult.points;

  if (ENV_PARAM(ENABLE_DEBUG_FACE_RECOG)) {
    cv::imwrite(std::string{"face_recog_expand-"} +
                    std::to_string(debug_counter) + ".jpg",
                img_expanded);
    auto img1 =
        img_expanded(cv::Rect{inner_x, inner_y, inner_w, inner_h}).clone();
    cv::imwrite(std::string{"face_recog-"} + std::to_string(debug_counter) +
                    ".jpg",
                img1);

    for (int i = 0; i < 5; i++) {
      auto point1 = cv::Point{static_cast<int>(points[i].first * img.cols),
                              static_cast<int>(points[i].second * img.rows)};
      cv::circle(img1, point1, 3, cv::Scalar(255, 8, 18), -1);
      std::cout << "points[i].first " << points[i].first << " "   //
                << "points[i].second " << points[i].second << " " //
                << std::endl;
    }
    cv::imwrite(std::string{"face_recog_out-"} + std::to_string(debug_counter) +
                    ".jpg",
                img1);
  }

  vector<float> points_src(10);
  //  valid_face_tl_x=0.0; //added by lyn
  // valid_face_tl_y=0.0; //added by lyn
  for (int i = 0; i < 5; i++) {
    points_src[2 * i] = points[i].first * img.cols + valid_face_tl_x;
    points_src[2 * i + 1] = points[i].second * img.rows + valid_face_tl_y;
  }

  __TOC__(ATT_POST_PROCESS)

  __TIC__(RECOG_ALIGN)

  // need aligned;
  MatrixXf m = get_rotate_matrix(points_src);
  vector<float> data(m.size());
  for (auto i = 0; i < m.rows(); ++i) {
    for (auto j = 0; j < m.cols(); ++j) {
      data[i * m.cols() + j] = m(i, j);
    }
  }
  cv::Mat rotate_mat = cv::Mat(2, 3, CV_32FC1, data.data());
  cv::warpAffine(
      img_expanded, aligned, rotate_mat,
      cv::Size(feature_->getInputWidth(), feature_->getInputHeight()));
  if (ENV_PARAM(ENABLE_DEBUG_FACE_RECOG)) {
    cv::imwrite("face_recog_out_aligned-" + std::to_string(debug_counter) +
                    ".jpg",
                aligned);
  }
  __TOC__(RECOG_ALIGN)
  auto features = feature_->run_fixed(aligned);

  return FaceRecogFixedResult{getInputWidth(),
                              getInputHeight(),
                              features.scale,
                              std::move(features.feature)};
}

std::vector<FaceRecogFixedResult> 
FaceRecogImp::_runNormalFixed(
        const std::vector<cv::Mat> &imgs_expanded,
        const std::vector<cv::Rect> &inner_bboxes) {
  static int debug_counter = 0;
  CHECK_EQ(imgs_expanded.size(), inner_bboxes.size()) 
          << "image vector size should equal to inner boxes vector size";
  int size = imgs_expanded.size();

  std::vector<cv::Mat> imgs(size);
  std::vector<cv::Mat> resize_5pt(size);
  std::vector<cv::Mat> aligned_imgs(size);

  for (auto i = 0; i < size; ++i) {
    CHECK_NE(imgs_expanded[i].rows, 0) << "image must not be empty";
    CHECK_NE(imgs_expanded[i].cols, 0) << "image must not be empty";
    CHECK_GE(inner_bboxes[i].x, 0) << "image[" << i << "] inner_x must >= 0";
    CHECK_GE(inner_bboxes[i].y, 0) << "image[" << i << "] inner_y must >= 0";
    CHECK_GE(inner_bboxes[i].width, 0) << "image[" << i << "] inner_w must >= 0";
    CHECK_GE(inner_bboxes[i].height, 0) << "image[" << i << "] inner_h must >= 0";
    CHECK_LE(inner_bboxes[i].x + inner_bboxes[i].width, imgs_expanded[i].cols) << "image[" << i << "]"
         << "inner_w must <= cols";
    CHECK_LE(inner_bboxes[i].y + inner_bboxes[i].height, imgs_expanded[i].rows) << "image[" << i << "]"
         << "inner_h must <= rows";
    __TIC__(ATT_RESIZE)
    // cv::Mat img_expanded =cv::Mat(rows, cols, CV_8UC3, const_cast<uint8_t
    // *>(input), stride);
    if (ENV_PARAM(ENABLE_DEBUG_FACE_RECOG)) {
      debug_counter++;
      LOG(INFO) << "get imgs_expanded[" << i << "]"
                << " success will get Rect Image "
                << "cols " << imgs_expanded[i].cols << " " //
                << "rows " << imgs_expanded[i].rows << " " //
                << "inner_x " << inner_bboxes[i].x << " "                     //
                << "inner_y " << inner_bboxes[i].y << " "                     //
                << "inner_w " << inner_bboxes[i].width << " "                     //
                << "inner_h " << inner_bboxes[i].height << " "                     //
                << std::endl;
    }
    imgs[i] = imgs_expanded[i](cv::Rect_<int>(
        inner_bboxes[i].x, inner_bboxes[i].y, inner_bboxes[i].width, inner_bboxes[i].height));

    cv::resize(imgs[i], resize_5pt[i],
               cv::Size(landmark_->getInputWidth(), landmark_->getInputHeight()),
               0, 0, cv::INTER_LINEAR);

    __TOC__(ATT_RESIZE)
  }
  __TIC__(ATT_RUN)
  auto landmarkResult_vector = landmark_->run(resize_5pt);
  __TOC__(ATT_RUN)
  __TIC__(ATT_POST_PROCESS)

  for (auto i = 0; i < size; ++i) {
    cv::Mat aligned;
    auto points = landmarkResult_vector[i].points;

    if (ENV_PARAM(ENABLE_DEBUG_FACE_RECOG)) {
      cv::imwrite(std::string{"face_recog_expand-"} +
                      std::to_string(i) + "-" +
                      std::to_string(debug_counter) + ".jpg",
                  imgs_expanded[i]);
      auto img1 =
          imgs_expanded[i](cv::Rect{inner_bboxes[i].x, inner_bboxes[i].y, 
                                    inner_bboxes[i].width, inner_bboxes[i].height}).clone();
      cv::imwrite(std::string{"face_recog-"} + 
                      std::to_string(i) + "-" +
                      std::to_string(debug_counter) +
                      ".jpg",
                  img1);
  
      for (int j = 0; j < 5; j++) {
        auto point1 = cv::Point{static_cast<int>(points[j].first * imgs[i].cols),
                                static_cast<int>(points[j].second * imgs[i].rows)};
        cv::circle(img1, point1, 3, cv::Scalar(255, 8, 18), -1);
        std::cout << "image: " << i << " "
                  << "points[" << j << "].first " << points[j].first << " "   //
                  << "points[" << j << "].second " << points[j].second << " " //
                  << std::endl;
      }
      cv::imwrite(std::string{"face_recog_out-"} + 
                  std::to_string(i) + "-" +
                  std::to_string(debug_counter) +
                      ".jpg",
                  img1);
    }

    vector<float> points_src(10);
    //  valid_face_tl_x=0.0; //added by lyn
    // valid_face_tl_y=0.0; //added by lyn
    for (int j = 0; j < 5; j++) {
      points_src[2 * j] = points[j].first * imgs[i].cols + inner_bboxes[i].x;
      points_src[2 * j + 1] = points[j].second * imgs[i].rows + inner_bboxes[i].y;
    }

    __TOC__(ATT_POST_PROCESS)
    __TIC__(RECOG_ALIGN)
    // need aligned;
    MatrixXf m = get_rotate_matrix(points_src);
    vector<float> data(m.size());
    for (auto n = 0; n < m.rows(); ++n) {
      for (auto k = 0; k < m.cols(); ++k) {
        data[n * m.cols() + k] = m(n, k);
      }
    }
    cv::Mat rotate_mat = cv::Mat(2, 3, CV_32FC1, data.data());
    cv::warpAffine(
        imgs_expanded[i], aligned_imgs[i], rotate_mat,
        cv::Size(feature_->getInputWidth(), feature_->getInputHeight()));
    if (ENV_PARAM(ENABLE_DEBUG_FACE_RECOG)) {
      cv::imwrite("face_recog_out_aligned-" + 
                  std::to_string(i) + "-" +
                  std::to_string(debug_counter) +
                      ".jpg",
                  aligned_imgs[i]);
    }
    __TOC__(RECOG_ALIGN)
  }

  auto features_vector = feature_->run_fixed(aligned_imgs);
  auto results = std::vector<FaceRecogFixedResult>(features_vector.size());
  for (auto i = 0u; i < features_vector.size(); ++i) {
    results[i].width = getInputWidth();
    results[i].height = getInputHeight();
    results[i].scale = features_vector[i].scale;
    results[i].feature = std::move(features_vector[i].feature);
  }
  return results;
}

FaceRecogFixedResult FaceRecogImp::run_fixed(const cv::Mat &img, int inner_x,
                                             int inner_y, int inner_w,
                                             int inner_h) {
  return _runNormalFixed(img, inner_x, inner_y, inner_w, inner_h);
}

std::vector<FaceRecogFixedResult> 
FaceRecogImp::run_fixed(const std::vector<cv::Mat> &imgs, 
                        const std::vector<cv::Rect> &inner_bboxes) {
  return _runNormalFixed(imgs, inner_bboxes);
}

FaceRecogFloatResult FaceRecogImp::run(const cv::Mat &img, int inner_x,
                                       int inner_y, int inner_w, int inner_h) {

  auto fixedResult = run_fixed(img, inner_x, inner_y, inner_w, inner_h);

  auto result =
      FaceRecogFloatResult{getInputWidth(),
                           getInputHeight(),
                           std::unique_ptr<FaceRecogFloatResult::vector_t>(
                               new FaceRecogFloatResult::vector_t())};
  for (auto i = 0u; i < result.feature->size(); i++) {
    (*result.feature.get())[i] =
        (float)((*fixedResult.feature.get())[i]) * fixedResult.scale;
  }

  return result;
}

std::vector<FaceRecogFloatResult> 
FaceRecogImp::run(const std::vector<cv::Mat> &imgs, 
                  const std::vector<cv::Rect> &inner_bboxes) {
  auto fixedResult = run_fixed(imgs, inner_bboxes);

  auto result = std::vector<FaceRecogFloatResult>(fixedResult.size());

  for (auto n = 0u; n < fixedResult.size(); ++n ) { 
    result[n] =
        FaceRecogFloatResult{getInputWidth(),
                             getInputHeight(),
                             std::unique_ptr<FaceRecogFloatResult::vector_t>(
                                 new FaceRecogFloatResult::vector_t())};
    for (auto i = 0u; i < result[n].feature->size(); i++) {
      (*result[n].feature.get())[i] =
        (float)((*fixedResult[n].feature.get())[i]) * fixedResult[n].scale;
    }
  }
  return result;
}

} // namespace ai
} // namespace vitis
