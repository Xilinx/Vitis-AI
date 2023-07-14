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
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "facedetectrecog_imp.hpp"
#include <eigen3/Eigen/Dense>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/attrs/attrs.hpp>
//#include <vitis/ai/expand_and_align.hpp>

DEF_ENV_PARAM_2(VAI_LIBRARY_MODELS_DIR, ".", std::string)
DEF_ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG, "0");
using Eigen::Map;
using Eigen::Matrix3f;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using std::vector;

namespace vitis {
namespace ai {

static float ref_points[] = {
    30.29459953, 65.53179932, 48.02519989, 33.54930115, 62.72990036,
    51.69630051, 51.50139999, 71.73660278, 92.3655014, 92.20410156};

static float ref_matrix[][40] = {{
    30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    -30.2946, -65.5318, -48.0252, -33.5493, -62.7299,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1},
    {-30.2946, -65.5318, -48.0252, -33.5493, -62.7299,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    51.6963, 51.5014, 71.7366, 92.3655, 92.2041,
    30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1}};

static Matrix3f solve_rotate(MatrixXf A, VectorXf b) {
  VectorXf r = A.colPivHouseholderQr().solve(b);
  Matrix3f t_inv;
  t_inv << r(0), -r(1), 0, r(1), r(0), 0, r(2), r(3), 1;
  Matrix3f t = t_inv.inverse();
  t(0, 2) = 0;
  t(1, 2) = 0;
  t(2, 2) = 1;
  return t;
}

static MatrixXf get_rotate_matrix(const std::vector<float>& points) {

  Map<MatrixXf> m_points(const_cast<float *>(points.data()), 2, 5);
  MatrixXf m_points_t = m_points.transpose();
  Map<VectorXf> v_points(m_points_t.data(), m_points_t.size());

  Map<MatrixXf> m_ref0(ref_matrix[0], 10, 4);
  Map<MatrixXf> m_ref1(ref_matrix[1], 10, 4);

  Matrix3f m_t0 = solve_rotate(m_ref0, v_points);
  Matrix3f m_t1 = solve_rotate(m_ref1, v_points);
  m_t1 << -m_t1.leftCols(1), m_t1.rightCols(2);

  MatrixXf m_points_ext(5, 3);
  m_points_ext << m_points_t, MatrixXf::Ones(5, 1);

  MatrixXf m_pt0 = m_points_ext * m_t0;
  MatrixXf m_pt1 = m_points_ext * m_t1;

  Map<MatrixXf> m_ref_points(ref_points, 5, 2);

  float norm0 = (m_pt0.block(0, 0, 5, 2) - m_ref_points).norm();
  float norm1 = (m_pt1.block(0, 0, 5, 2) - m_ref_points).norm();

  MatrixXf m_rotate = (norm0 <= norm1) ?
      m_t0.block(0, 0, 3, 2).transpose() : m_t1.block(0, 0, 3, 2).transpose();

  return m_rotate;
}

static int div_ceil(int a, int b) {  //
  return a / b + (a % b == 0 ? 0 : 1);
}

static int div_floor(int a, int b) {  //
  return a / b;
}

// static int div_round(int a, int b) {  //
//   return a / b + (a % b < b / 2 ? 0 : 1);
// }

static int in_range(int a, int min_value, int max_value) {
  return std::min(std::max(a, min_value), max_value);
}

static std::pair<int,int> expand(int total, int x, int w, float ratio) {
  const int d = w * ratio;
  const int x1 = x - d;
  const int x2 = x + w + d;
  const int x1_c = in_range(x1, 0, total);
  const int x2_c = in_range(x2, 0, total);
  return std::make_pair(x1_c, x2_c - x1);
}

static std::pair<int, int> align(int total, int x, int w, int a) {
  // 如果总宽度不是对齐像素的整数倍，最多 a-1 个像素会被丢弃掉
  const int total_in_a = div_floor(total, a);
  // 对齐的时候，左边尽量多扩一些
  const int x1_in_a = div_floor(x, a);
  // 对齐的时候，右边也尽量多扩一些。
  const int x2_in_a = div_ceil(x + w, a);
  const int x1_c_in_a = in_range(x1_in_a, 0, total_in_a);
  const int x2_c_in_a = in_range(x2_in_a, 0, total_in_a);
  const int aligned_x = x1_c_in_a * a;
  const int aligned_w = (x2_c_in_a - x1_c_in_a) * a;
  return std::make_pair(aligned_x, aligned_w);
}

static std::tuple<int, int, int, int> expand_and_align_1(int total, int x,
                                                         int w, float ratio,
                                                         int a) {
  int new_x = 0;
  int new_w = 0;
  std::tie(new_x, new_w) = expand(total, x, w, ratio);
  int aligned_x = 0;
  int aligned_w = 0;
  std::tie(aligned_x, aligned_w) = align(total, new_x, new_w, a);
  int relative_x = x - aligned_x;
  // 注意，对其之后，原来的 w 有可能超过对其之后的边界。如果输入的 w
  // 没有超过对齐之后的边界，relative_w 应该不变。
  int relative_w = in_range(w, 0, aligned_w);
  return std::make_tuple(aligned_x, aligned_w, relative_x, relative_w);
}

static std::pair<cv::Rect, cv::Rect> expand_and_align(
        int width, int height, int bbx, int bby, int bbw, int bbh,
        float ratio_x, float ratio_y, int aligned_x, int aligned_y) {

  // Expanded rect in original image
  std::tuple<int, int, int, int> expanded;

  // Cropped rect in expanded image
  std::tuple<int, int, int, int> relative;

  std::tie(std::get<0>(expanded), std::get<2>(expanded), std::get<0>(relative),
           std::get<2>(relative)) =
      expand_and_align_1(width, bbx, bbw, ratio_x, aligned_x);
  std::tie(std::get<1>(expanded), std::get<3>(expanded), std::get<1>(relative),
           std::get<3>(relative)) =
      expand_and_align_1(height, bby, bbh, ratio_y, aligned_y);
  return std::make_pair(cv::Rect{std::get<0>(expanded), std::get<1>(expanded),
                                 std::get<2>(expanded), std::get<3>(expanded)},
                        cv::Rect{std::get<0>(relative), std::get<1>(relative),
                                 std::get<2>(relative), std::get<3>(relative)});
}

static std::vector<std::string> find_model_search_path() {
  auto ret = std::vector<std::string>{};
  ret.push_back(ENV_PARAM(VAI_LIBRARY_MODELS_DIR));
  ret.push_back("/usr/share/vitis_ai_library/models");
  ret.push_back("/usr/share/vitis_ai_library/.models");
  return ret;
}

static size_t filesize(const std::string& filename) {
  size_t ret = 0u;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = S_ISREG(statbuf.st_mode) ? statbuf.st_size : 0u;
  }
  return ret;
}
static std::string find_model(const std::string& name) {
//# Disable the unused functions when DPUV1 Enable
#ifndef ENABLE_DPUCADX8G_RUNNER
  if (filesize(name) > 0u) {
    return name;
  }

  auto ret = std::string();
  for (const auto& p : find_model_search_path()) {
    ret = p + "/" + name + "/" + name;
    const auto xmodel_name = ret + ".xmodel";
    if (filesize(xmodel_name) > 0u) {
      return xmodel_name;
    }
    const auto elf_name = ret + ".elf";
    if (filesize(elf_name) > 0u) {
      return elf_name;
    }
  }
#else
  //# Get the config prototxt from dir path
  std::string tmp_name = name;
  while (tmp_name.back() == '/') {
    tmp_name.pop_back();
  }
  std::string last_element(tmp_name.substr(tmp_name.rfind("/") + 1));
  auto config_file = name + "/" + last_element + ".prototxt";

  if (filesize(config_file) > 0u) {
    return config_file;
  }

  //# Get model path from standard path
  auto ret = std::string();
  for (const auto& p : find_model_search_path()) {
    ret = p + "/" + name + "/" + name;
    const auto config_file = ret + ".prototxt";
    if (filesize(config_file) > 0u) {
      return config_file;
    }
  }
#endif

  std::stringstream str;
  str << "cannot find model <" << name << "> after checking following dir:";
  for (const auto& p : find_model_search_path()) {
    str << "\n\t" << p;
  }
  LOG(FATAL) << str.str();
  return std::string{""};
}

static std::string find_name_in_line(const std::string& line, const std::string& type_name){
  std::string ret;
  auto pos = line.find(type_name);
  
  if (pos != std::string::npos) {
    auto subline = line.substr(pos + 1); // maybe: " : ./face_landmark.xmodel\n"
    pos = subline.find(':');
    CHECK_NE(pos, std::string::npos) << "Type : " << type_name << " has no value!";
    auto raw = subline.substr(pos+1);
    auto begin = raw.find_first_not_of(" \t");
    auto end = raw.find_last_not_of(" \t");
    ret = raw.substr(begin, end);
    LOG_IF(INFO, ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) 
          << "name :" << ret;
  }
  return ret;
}

static std::vector<std::string> get_real_model_names(const std::string &fake_name) {
  std::vector<std::string> ret(3);
  auto full_name = find_model(fake_name);
  LOG_IF(INFO, ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) 
        << "model name: " << full_name;
  std::ifstream ifile(full_name);
  char s[1024];
  std::string detect_name;
  std::string landmark_name;
  std::string feature_name;
  int cnt = 0;
  std::vector<std::string> types{"FACEDETECT", "FACELANDMARK", "FACEFEATURE"}; 
  while(!ifile.eof()) {
    ifile.getline(s, 1024);
    for (auto i = 0; i < 3; ++i) {
      auto name = find_name_in_line(s, types[i]);  
      if (!name.empty()) {
        ret[i] = name;
        cnt++;
        break;
      }
    }
    if (cnt == 3) {
      break;
    }
  }
  cnt = 0;
  for (auto i = 0; i < 3; ++i) {
    if (!ret[i].empty()) {
      cnt++;
    }
  }

  if (cnt != 3) {
    LOG(FATAL) << "Reach end of file: " << full_name 
               << ", but only read " << cnt << "model names";
  }
  return ret;
}

FaceDetectRecogImp::FaceDetectRecogImp(const std::string &detect_model_name, 
                                       const std::string &landmark_model_name, 
                                       const std::string &feature_model_name, 
                                       bool need_preprocess)
  {
    auto attrs = xir::Attrs::create();
    detect_ = FaceDetect::create(detect_model_name, need_preprocess);

    landmark_ = FaceLandmark::create(landmark_model_name, attrs.get(),
                                     need_preprocess);
    feature_ = FaceFeature::create(feature_model_name, attrs.get(),
                                   need_preprocess);
}

FaceDetectRecogImp::FaceDetectRecogImp(const std::string &model_name, 
                                       bool need_preprocess)
  {
    auto real_model_names = get_real_model_names(model_name);
    auto attrs = xir::Attrs::create();
    CHECK_EQ(real_model_names.size(), 3) << "real model names should equal to 3";
    detect_ = FaceDetect::create(real_model_names[0], need_preprocess);

    landmark_ = FaceLandmark::create(real_model_names[1], attrs.get(),
                                     need_preprocess);
    feature_ = FaceFeature::create(real_model_names[2], attrs.get(),
                                   need_preprocess);
}
FaceDetectRecogImp::~FaceDetectRecogImp() {}

int FaceDetectRecogImp::getInputWidth() const { return detect_->getInputWidth(); }

int FaceDetectRecogImp::getInputHeight() const { return detect_->getInputHeight(); }

size_t FaceDetectRecogImp::get_input_batch() const { return detect_->get_input_batch(); }

static bool CheckBboxValid(const FaceDetectResult::BoundingBox &bbox) {
  return (bbox.x >= 0.f && bbox.x <= 1.0f) && 
         (bbox.y >= 0.f && bbox.y <= 1.0f) && 
         (bbox.x + bbox.width >=0.f && bbox.x + bbox.width <=1.0f) && 
         (bbox.y + bbox.height >=0.f && bbox.y + bbox.height <=1.0f);
}

std::vector<FaceFeatureFixedResult> 
FaceDetectRecogImp::run_recog_fixed_batch_internal(
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
    if (ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) {
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

    if (ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) {
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

    std::vector<float> points_src(10);
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
    std::vector<float> data(m.size());
    for (auto n = 0; n < m.rows(); ++n) {
      for (auto k = 0; k < m.cols(); ++k) {
        data[n * m.cols() + k] = m(n, k);
      }
    }
    cv::Mat rotate_mat = cv::Mat(2, 3, CV_32FC1, data.data());
    cv::warpAffine(
        imgs_expanded[i], aligned_imgs[i], rotate_mat,
        cv::Size(feature_->getInputWidth(), feature_->getInputHeight()));
    if (ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) {
      cv::imwrite("face_recog_out_aligned-" + 
                  std::to_string(i) + "-" +
                  std::to_string(debug_counter) +
                      ".jpg",
                  aligned_imgs[i]);
    }
    __TOC__(RECOG_ALIGN)
  }

  auto features_vector = feature_->run_fixed(aligned_imgs);
  auto results = std::vector<FaceFeatureFixedResult>(features_vector.size());
  for (auto i = 0u; i < features_vector.size(); ++i) {
    results[i].width = getInputWidth();
    results[i].height = getInputHeight();
    results[i].scale = features_vector[i].scale;
    results[i].feature = std::move(features_vector[i].feature);
  }
  return results;
}

std::vector<FaceDetectRecogFixedResult> FaceDetectRecogImp::run_fixed_internal(
    const std::vector<cv::Mat> &input_images) {
  auto batch = get_input_batch(); // detect batch
  auto input_batch = (input_images.size() < batch) ? input_images.size() : batch;
  auto size = cv::Size(detect_->getInputWidth(), detect_->getInputHeight());

  std::vector<cv::Mat> images_detect(input_batch);
  std::vector<std::pair<int, int>> original_size(input_batch);

  // 1. resize image as facedetect size
  for (auto n = 0u; n < input_batch; ++n) {
    original_size[n] = std::make_pair(input_images[n].cols, input_images[n].rows);    
    if (size != input_images[n].size()) {
      cv::resize(input_images[n], images_detect[n], size, 0);
    } else {
      images_detect[n] = input_images[n];
    }
  }
  
  // 2. face detect
  std::vector<FaceDetectResult> det_batch_result;
  if (input_batch == 1) {
    auto det_result = detect_->run(images_detect[0]);
    det_batch_result.push_back(det_result);
  } else {
    det_batch_result = detect_->run(images_detect); // use batch API
  }

  // 3. check bbox valid and expand
  std::vector<FaceDetectResult> detect_valid_results(input_batch);
  std::vector<std::set<int>> valid_rect_indexes(input_batch);
  for (auto n = 0u; n < input_batch; ++n) {
    for (auto i = 0u; i < det_batch_result[n].rects.size(); ++i) {
      if (CheckBboxValid(det_batch_result[n].rects[i])) {
        //LOG(INFO) << "hello: valid_rect_indexes[" << n << "] size: " << valid_rect_indexes[n].size();
        valid_rect_indexes[n].insert(i);
      }
    }
  }

  for (auto n = 0u; n < input_batch; ++n) {
    detect_valid_results[n] = det_batch_result[n];
    int count = 0;
    detect_valid_results[n].rects.resize(valid_rect_indexes[n].size());
    for (auto it = valid_rect_indexes[n].begin(); it != valid_rect_indexes[n].end(); ++it) {
      detect_valid_results[n].rects[count++] = det_batch_result[n].rects[*it];
      LOG_IF(INFO, ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) << "count :" << count -1 << ", *it: " << *it;
      LOG_IF(INFO, ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) 
            << "detect[" << n << "] rect[" << count-1 << "] :" 
            << "x = " << detect_valid_results[n].rects[count-1].x
            << "y = " << detect_valid_results[n].rects[count-1].y
            << "w = " << detect_valid_results[n].rects[count-1].width
            << "h = " << detect_valid_results[n].rects[count-1].height;
      LOG_IF(INFO, ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) 
            << "detect[" << n << "] rect[" << count-1 << "] :" 
            << "x = " << detect_valid_results[n].rects[count-1].x * input_images[n].cols
            << "y = " << detect_valid_results[n].rects[count-1].y * input_images[n].rows
            << "w = " << detect_valid_results[n].rects[count-1].width * input_images[n].cols
            << "h = " << detect_valid_results[n].rects[count-1].height * input_images[n].rows;
    }
  }
  std::vector<std::vector<std::pair<cv::Rect, cv::Rect>>> expanded_bboxes(input_batch);
  //std::vector<std::vector<ExpandAndAlign>> expanded_bboxes(input_batch);
  for (auto n = 0u; n < input_batch; ++n) {
    for (auto i = 0u; i < detect_valid_results[n].rects.size(); ++i) {
        auto expanded = expand_and_align(
        //auto expanded = expand_crop(
                input_images[n].cols, input_images[n].rows,
                detect_valid_results[n].rects[i].x * input_images[n].cols,
                detect_valid_results[n].rects[i].y * input_images[n].rows,
                detect_valid_results[n].rects[i].width * input_images[n].cols,
                detect_valid_results[n].rects[i].height * input_images[n].rows,
                0.2, 0.2, 16, 8);
        expanded_bboxes[n].push_back(expanded);
        LOG_IF(INFO, ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) 
              << "expanded bboxes: "<< expanded_bboxes[n][i].first
              << " , " << expanded_bboxes[n][i].second;
    }
  }
  // 4. facerecog
  // note: recog should run with batch, not input batch
  // maybe input one image but detect more than one faces
  auto all_valid_bboxes_size = 0;
  for (auto n = 0u; n < input_batch; ++n) {
      all_valid_bboxes_size += detect_valid_results[n].rects.size();
  }
  LOG_IF(INFO, ENV_PARAM(ENABLE_DEBUG_FACE_DETECT_RECOG)) 
         << "all valid bboxes size: " << all_valid_bboxes_size; 
  auto recog_batch = landmark_->get_input_batch();
  std::vector<cv::Mat> recog_batch_images(recog_batch);
  std::vector<cv::Rect> recog_batch_inner_rects(recog_batch);
  std::vector<FaceFeatureFixedResult> recog_results(all_valid_bboxes_size);
  auto cnt = 0u;
  auto batch_index = 0u;
  for (auto n = 0u; n < input_batch; ++n) {
    for (auto i = 0u; i < expanded_bboxes[n].size(); ++i) {
      recog_batch_images[batch_index] = input_images[n](expanded_bboxes[n][i].first);
      recog_batch_inner_rects[batch_index] = expanded_bboxes[n][i].second;
      batch_index++;
      if (batch_index == recog_batch) {
        auto recog_batch_results = 
            run_recog_fixed_batch_internal(recog_batch_images, recog_batch_inner_rects);
        for (auto c = 0u; c < recog_batch; ++c) {
          //recog_results[cnt] = recog_batch_results[c];
          recog_results[cnt] = FaceFeatureFixedResult{recog_batch_results[c].width,
                                                      recog_batch_results[c].height,
                                                      recog_batch_results[c].scale,
                                                      std::move(recog_batch_results[c].feature)};
          cnt++;
        }
        batch_index = 0;
      }
    }
  }
  if (batch_index != 0) {
    recog_batch_images.resize(batch_index);
    recog_batch_inner_rects.resize(batch_index);
    auto recog_batch_results = 
        run_recog_fixed_batch_internal(recog_batch_images, recog_batch_inner_rects);
    for (auto c = 0u; c < batch_index; ++c) {
      recog_results[cnt] = FaceFeatureFixedResult{recog_batch_results[c].width,
                                                  recog_batch_results[c].height,
                                                  recog_batch_results[c].scale,
                                                  std::move(recog_batch_results[c].feature)};
      cnt++;
    }
  }
  // 5. merge results 
  std::vector<FaceDetectRecogFixedResult> results(input_batch);
  auto index = 0u;
  for (auto n = 0u; n < input_batch; ++n) {
    results[n].width = original_size[n].first; 
    results[n].height = original_size[n].second; 
    // shouldn't use det_batch_result directly
    //results[n].detect_result = detect_valid_results[n]; 
    results[n].rects = detect_valid_results[n].rects; 
    results[n].features.resize(detect_valid_results[n].rects.size());
    //std::vector<FaceDetectRecogFixedResult::vector_t> recog_features(detect_valid_results[n].rects.size());
    for (auto i = index; i < index + detect_valid_results[n].rects.size(); ++i) {  
      if (i >= recog_results.size()) {
        break;
      }
      results[n].feature_scale = recog_results[i].scale; 
      //recog_features[i - index] = *(recog_results[i].feature);
      results[n].features[i - index] = *(recog_results[i].feature);
    }
    //results[n].features = recog_features;
    index += detect_valid_results[n].rects.size();
  }

  return results; 
}

std::vector<FaceDetectRecogFixedResult> FaceDetectRecogImp::run_fixed(
    const std::vector<cv::Mat> &input_images) {
  __TIC__(FACE_DETECT_RECOG_E2E)
  auto ret = run_fixed_internal(input_images);
  __TOC__(FACE_DETECT_RECOG_E2E)
  return ret;
}

FaceDetectRecogFixedResult FaceDetectRecogImp::run_fixed(const cv::Mat &input_image) {
  __TIC__(FACE_DETECT_RECOG_E2E)
  std::vector<cv::Mat> input_images(1);
  input_images[0] = input_image;
  auto ret = run_fixed_internal(input_images);
  __TOC__(FACE_DETECT_RECOG_E2E)
  return ret[0];
}

std::vector<FaceDetectRecogFloatResult> FaceDetectRecogImp::run(
    const std::vector<cv::Mat> &input_images) {
  __TIC__(FACE_DETECT_RECOG_E2E)
  auto ret_fixed = run_fixed_internal(input_images); 
  std::vector<FaceDetectRecogFloatResult> ret(ret_fixed.size());
  for (auto n = 0u; n < ret_fixed.size(); ++n) {
    ret[n].width = input_images[n].cols;
    ret[n].height = input_images[n].rows;
    ret[n].rects = ret_fixed[n].rects;
    ret[n].features.resize(ret_fixed[n].features.size());
    for (auto i = 0u; i < ret[n].features.size(); ++i) {
      //auto feature = std::unique_ptr<std::array<float, 512>>(new std::array<float, 512>());
      for (auto j = 0; j < 512; ++j) {
        ret[n].features[i][j] = 
            ret_fixed[n].features[i][j] * ret_fixed[n].feature_scale;
      } 
    }
  }
  __TOC__(FACE_DETECT_RECOG_E2E)
  return ret;
}

FaceDetectRecogFloatResult FaceDetectRecogImp::run(const cv::Mat &input_image) {
  __TIC__(FACE_DETECT_RECOG_E2E)
  std::vector<cv::Mat> input_images(1);
  input_images[0] = input_image;
  auto ret_fixed = run_fixed_internal(input_images);
  FaceDetectRecogFloatResult ret;
  ret.width = input_image.cols;
  ret.height = input_image.rows;
  ret.rects = ret_fixed[0].rects;
  ret.features.resize(ret_fixed[0].features.size());
  for (auto i = 0u; i < ret.features.size(); ++i) {
    //auto feature = std::unique_ptr<std::array<float, 512>>(new std::array<float, 512>());
    for (auto j = 0u; j < 512; ++j) {
        ret.features[i][j] = 
          ret_fixed[0].features[i][j] * ret_fixed[0].feature_scale;
    }
  }
  __TOC__(FACE_DETECT_RECOG_E2E)
  return ret;
}

float FaceDetectRecogImp::getThreshold() const {
  return detect_->getThreshold();
}

void FaceDetectRecogImp::setThreshold(float threshold) {
  detect_->setThreshold(threshold);  
}

}  // namespace ai
}  // namespace vitis
