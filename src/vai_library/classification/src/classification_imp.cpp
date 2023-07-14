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

#include "./classification_imp.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/globalavepool.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(ENABLE_CLASSIFICATION_DEBUG, "0");
DEF_ENV_PARAM(CLASSIFICATION_SET_INPUT, "0");

ClassificationImp::ClassificationImp(const std::string& model_name,
                                     bool need_preprocess)
    : Classification(model_name, need_preprocess),
      preprocess_type{configurable_dpu_task_->getConfig()
                          .classification_param()
                          .preprocess_type()},
      TOP_K{configurable_dpu_task_->getConfig().classification_param().top_k()},
      test_accuracy{configurable_dpu_task_->getConfig()
                        .classification_param()
                        .test_accuracy()} {}

ClassificationImp::~ClassificationImp() {}

static void croppedImage(const cv::Mat& image, int height, int width,
                         cv::Mat& cropped_img) {
  int offset_h = (image.rows - height) / 2;
  int offset_w = (image.cols - width) / 2;
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
}

static void inception_preprocess(const cv::Mat& image, int height, int width,
                                 cv::Mat& pro_res,
                                 float central_fraction = 0.875,
                                 bool iscentral_crop = true) {
  LOG_IF(INFO, ENV_PARAM(ENABLE_CLASSIFICATION_DEBUG))
      << "incepiton_preprocess";
  if (iscentral_crop) {
    float img_hd = image.rows;
    float img_wd = image.cols;
    int offset_h = (img_hd - img_hd * central_fraction) / 2;
    int offset_w = (img_wd - img_wd * central_fraction) / 2;
    cv::Rect box(offset_w, offset_h, image.cols - offset_w * 2,
                 image.rows - offset_h * 2);
    cv::Mat res_crop = image(box).clone();
    cv::resize(res_crop, pro_res, cv::Size(width, height));
  } else {
    cv::resize(image, pro_res, cv::Size(width, height));
  }
}

static void vgg_preprocess(const cv::Mat& image, int height, int width,
                           cv::Mat& pro_res) {
  float smallest_side = 256;
  float scale =
      smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(image.cols * scale, image.rows * scale));
  croppedImage(resized_image, height, width, pro_res);
}

static void inception_pt(const cv::Mat& image, int height, int width,
                         cv::Mat& pro_res) {
  float smallest_side = 299;
  float scale =
      smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  LOG_IF(INFO, ENV_PARAM(ENABLE_CLASSIFICATION_DEBUG))
      << "resize: Width = " << image.cols * scale
      << " Height = " << image.rows * scale;
  cv::Mat resized_image;
  cv::resize(image, resized_image,
             cv::Size(ceil(image.cols * scale), ceil(image.rows * scale)));
  LOG_IF(INFO, ENV_PARAM(ENABLE_CLASSIFICATION_DEBUG))
      << "after resize: " << resized_image.size;
  croppedImage(resized_image, height, width, pro_res);
  if (ENV_PARAM(CLASSIFICATION_SET_INPUT)) {
    std::ifstream input_bin("input_fix.bin", std::ios::binary);
    if (input_bin.is_open()) {
      input_bin.seekg(0, input_bin.end);
      int fsize = input_bin.tellg();
      input_bin.seekg(0, input_bin.beg);
      input_bin.read((char*)pro_res.data, fsize);
      input_bin.close();
    } else {
      LOG(INFO) << "Pleasae rename your input file as input_fix.bin!";
    }
  }
}

static void efficientnet_preprocess(const cv::Mat& input, int height, int width,
                                    cv::Mat& output) {
  int CROP_PADDING = 32;
  CHECK_EQ(height, width) << "width must be equal with height";
  int output_size = height;
  int input_height = input.rows;
  int input_width = input.cols;

  float scale = (float)(output_size) / (output_size + CROP_PADDING);
  int padded_center_crop_size =
      (int)(scale *
            ((input_height > input_width) ? input_width : input_height));
  int offset_height = ((input_height - padded_center_crop_size) + 1) / 2;
  int offset_width = ((input_width - padded_center_crop_size) + 1) / 2;

  cv::Mat cropped_img;
  cv::Rect box(offset_width, offset_height, padded_center_crop_size,
               padded_center_crop_size);
  cropped_img = input(box).clone();

  cv::resize(cropped_img, output, cv::Size(output_size, output_size), 0, 0,
             cv::INTER_CUBIC);
}

static void ofa_resnet50_preprocess(const cv::Mat& image, int height, int width,
                                    cv::Mat& pro_res) {
  int smallest_side = 183;
  cv::Mat resized_image;
  cv::Size size;  // w,h
  if ((image.cols <= image.rows && image.cols == smallest_side) ||
      (image.cols >= image.rows && image.rows == smallest_side)) {
    resized_image = image;
  } else {
    if (image.cols < image.rows) {
      size = cv::Size(smallest_side, float(smallest_side) * float(image.rows) /
                                         float(image.cols));
    } else {
      size =
          cv::Size(float(smallest_side) * float(image.cols) / float(image.rows),
                   smallest_side);
    }
    cv::resize(image, resized_image, size);
  }

  int offset_h = round(float(resized_image.rows - height) / 2.0f);
  int offset_w = round(float(resized_image.cols - width) / 2.0f);
  cv::Rect box(offset_w, offset_h, width, height);
  pro_res = resized_image(box).clone();
}

static void ofa_resnet50_pt2_preprocess(const cv::Mat& input, int height, int width,
                                    cv::Mat& output) {
  int newheight  = std::ceil(input.rows / 0.875);
  int newwidth = std::ceil(input.cols / 0.875);
  cv::Mat middle_img, cropped_img;
  cv::resize(input, middle_img, cv::Size(newwidth, newheight), 0, 0,
             cv::INTER_AREA);
  croppedImage(middle_img, input.rows, input.cols, cropped_img);
  cv::resize(cropped_img, output, cv::Size(width, height), 0, 0,
             cv::INTER_CUBIC);
}

static void ofadepthwise_preprocess(const cv::Mat& input, int height, int width,
                                    cv::Mat& output) {
  int shorter = std::min(input.rows, input.cols);
  int size_init = std::ceil(320.0 / 0.875);
  int newwidth = size_init * ((input.cols * 1.0) / (shorter * 1.0));
  int newheight = size_init * ((input.rows * 1.0) / (shorter * 1.0));
  cv::Mat middle_img, cropped_img;
  cv::resize(input, middle_img, cv::Size(newwidth, newheight), 0, 0,
             cv::INTER_AREA);
  croppedImage(middle_img, 320, 320, cropped_img);
  cv::resize(cropped_img, output, cv::Size(width, height), 0, 0,
             cv::INTER_CUBIC);
}

vitis::ai::ClassificationResult ClassificationImp::run(
    const cv::Mat& input_image) {
  cv::Mat image;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  if (size == input_image.size() && (preprocess_type != 8)) {
    image = input_image;
  } else {
    switch (preprocess_type) {
      case 0:
        cv::resize(input_image, image, size);
        break;
      case 1:
        if (test_accuracy) {
//# DPUV1 directly uses the resized image dataset so no need of crop
#ifdef ENABLE_DPUCADX8G_RUNNER
          cv::resize(input_image, image, size);
#else
          croppedImage(input_image, height, width, image);
#endif
        } else {
          cv::resize(input_image, image, size);
        }
        break;
      case 2:
        vgg_preprocess(input_image, height, width, image);
        break;
      case 3:
        inception_preprocess(input_image, height, width, image);
        break;
      case 4:
        inception_pt(input_image, height, width, image);
        break;
      case 5:
        vgg_preprocess(input_image, height, width, image);
        break;
      case 6:
        efficientnet_preprocess(input_image, height, width, image);
        break;
      case 7:
        ofa_resnet50_preprocess(input_image, height, width, image);
        break;
      case 8:
        ofadepthwise_preprocess(input_image, height, width, image);
        break;
      case 9:
        cv::resize(input_image, image, size);
        break;
      case 10:
        ofa_resnet50_pt2_preprocess(input_image, height, width, image);
        break;
      default:
        break;
    }
  }
  //__TIC__(CLASSIFY_E2E_TIME)
  __TIC__(CLASSIFY_SET_IMG)
  if (preprocess_type == 2 || preprocess_type == 3 || preprocess_type == 4 ||
      preprocess_type == 6 || preprocess_type == 7 || preprocess_type == 8 ||
      preprocess_type == 9 || preprocess_type == 10) {
    configurable_dpu_task_->setInputImageRGB(image);
  } else {
    configurable_dpu_task_->setInputImageBGR(image);
  }
  __TOC__(CLASSIFY_SET_IMG)

  auto postprocess_index = 0;
  if (configurable_dpu_task_->getConfig()
          .classification_param()
          .has_avg_pool_param()) {
    __TIC__(CLASSIFY_DPU_0)
    configurable_dpu_task_->run(0);
    __TOC__(CLASSIFY_DPU_0)

    auto avg_scale = configurable_dpu_task_->getConfig()
                         .classification_param()
                         .avg_pool_param()
                         .scale();
    __TIC__(CLASSIFY_AVG_POOL)
    vitis::ai::globalAvePool(
        (int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(0),
        // 1024, 9, 3,
        configurable_dpu_task_->getOutputTensor()[0][0].channel,
        configurable_dpu_task_->getOutputTensor()[0][0].width,
        configurable_dpu_task_->getOutputTensor()[0][0].height,
        (int8_t*)configurable_dpu_task_->getInputTensor()[1][0].get_data(0),
        avg_scale);
    __TOC__(CLASSIFY_AVG_POOL)

    __TIC__(CLASSIFY_DPU_1)
    configurable_dpu_task_->run(1);
    __TOC__(CLASSIFY_DPU_1)
    postprocess_index = 1;
  } else {
    __TIC__(CLASSIFY_DPU)
    configurable_dpu_task_->run(0);
    __TOC__(CLASSIFY_DPU)
  }
  __TIC__(CLASSIFY_POST_ARM)
  auto ret = classification_post_process(
      configurable_dpu_task_->getInputTensor()[postprocess_index],
      configurable_dpu_task_->getOutputTensor()[postprocess_index],
      configurable_dpu_task_->getConfig());

  if (configurable_dpu_task_->getOutputTensor()[postprocess_index][0].channel ==
      1001) {
    for (auto& s : ret[0].scores) {
      s.index--;
    }
  }

  ret[0].type = 0;
  if (!configurable_dpu_task_->getConfig()
           .classification_param()
           .label_type()
           .empty()) {
    auto label_type =
        configurable_dpu_task_->getConfig().classification_param().label_type();
    if (label_type == "CIFAR10") {
      ret[0].type = 1;
    } else if (label_type == "FMNIST") {
      ret[0].type = 2;
    } else if (label_type == "ORIEN") {
      ret[0].type = 3;
    } else if (label_type == "ALVEOCLS") {
      ret[0].type = 4;
    } else if (label_type == "TF1KERAS") {
      ret[0].type = 5;
    } else if (label_type == "CAR_COLOR_CHEN") {
      ret[0].type = 6;
    }
  }

  __TOC__(CLASSIFY_POST_ARM)
  //__TOC__(CLASSIFY_E2E_TIME)

  return ret[0];
}

std::vector<ClassificationResult> ClassificationImp::run(
    const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);

  for (auto i = 0u; i < input_images.size(); i++) {
    if (size == input_images[i].size() && (preprocess_type != 8)) {
      images.push_back(input_images[i]);
    } else {
      cv::Mat image;
      switch (preprocess_type) {
        case 0:
          cv::resize(input_images[i], image, size);
          break;
        case 1:
          if (test_accuracy) {
//# DPUV1 directly uses the resized image dataset so no need of crop
#ifdef ENABLE_DPUCADX8G_RUNNER
            cv::resize(input_images[i], image, size);
#else
            croppedImage(input_images[i], height, width, image);
#endif
          } else {
            cv::resize(input_images[i], image, size);
          }
          break;
        case 2:
          vgg_preprocess(input_images[i], height, width, image);
          break;
        case 3:
          inception_preprocess(input_images[i], height, width, image);
          break;
        case 4:
          inception_pt(input_images[i], height, width, image);
          break;
        case 5:
          vgg_preprocess(input_images[i], height, width, image);
          break;
        case 6:
          efficientnet_preprocess(input_images[i], height, width, image);
          break;
        case 7:
          ofa_resnet50_preprocess(input_images[i], height, width, image);
          break;
        case 8:
          ofadepthwise_preprocess(input_images[i], height, width, image);
          break;
        case 9:
          cv::resize(input_images[i], image, size);
          break;
      case 10:
        ofa_resnet50_pt2_preprocess(input_images[i], height, width, image);
        break;
        default:
          break;
      }
      images.push_back(image);
    }
  }

  __TIC__(CLASSIFY_SET_IMG)
  if (preprocess_type == 2 || preprocess_type == 3 || preprocess_type == 4 ||
      preprocess_type == 6 || preprocess_type == 7 || preprocess_type == 8 ||
      preprocess_type == 9 || preprocess_type == 10) {
    configurable_dpu_task_->setInputImageRGB(images);
  } else {
    configurable_dpu_task_->setInputImageBGR(images);
  }
  __TOC__(CLASSIFY_SET_IMG)

  auto postprocess_index = 0;
  if (configurable_dpu_task_->getConfig()
          .classification_param()
          .has_avg_pool_param()) {
    __TIC__(CLASSIFY_DPU_0)
    configurable_dpu_task_->run(0);
    __TOC__(CLASSIFY_DPU_0)

    auto avg_scale = configurable_dpu_task_->getConfig()
                         .classification_param()
                         .avg_pool_param()
                         .scale();
    auto batch_size = configurable_dpu_task_->getInputTensor()[0][0].batch;

    __TIC__(CLASSIFY_AVG_POOL)
    for (auto batch_idx = 0u; batch_idx < batch_size; batch_idx++) {
      vitis::ai::globalAvePool(
          (int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(
              batch_idx),
          // 1024, 9, 3,
          configurable_dpu_task_->getOutputTensor()[0][0].channel,
          configurable_dpu_task_->getOutputTensor()[0][0].width,
          configurable_dpu_task_->getOutputTensor()[0][0].height,
          (int8_t*)configurable_dpu_task_->getInputTensor()[1][0].get_data(
              batch_idx),
          avg_scale);
    }
    __TOC__(CLASSIFY_AVG_POOL)

    __TIC__(CLASSIFY_DPU_1)
    configurable_dpu_task_->run(1);
    __TOC__(CLASSIFY_DPU_1)
    postprocess_index = 1;
  } else {
    __TIC__(CLASSIFY_DPU)
    configurable_dpu_task_->run(0);
    __TOC__(CLASSIFY_DPU)
  }

  __TIC__(CLASSIFY_POST_ARM)
  auto rets = classification_post_process(
      configurable_dpu_task_->getInputTensor()[postprocess_index],
      configurable_dpu_task_->getOutputTensor()[postprocess_index],
      configurable_dpu_task_->getConfig());

  for (auto& ret : rets) {
    if (configurable_dpu_task_->getOutputTensor()[postprocess_index][0]
            .channel == 1001) {
      for (auto& s : ret.scores) {
        s.index--;
      }
    }

    ret.type = 0;
    if (!configurable_dpu_task_->getConfig()
             .classification_param()
             .label_type()
             .empty()) {
      auto label_type = configurable_dpu_task_->getConfig()
                            .classification_param()
                            .label_type();
      if (label_type == "CIFAR10") {
        ret.type = 1;
      } else if (label_type == "FMNIST") {
        ret.type = 2;
      } else if (label_type == "ORIEN") {
        ret.type = 3;
      } else if (label_type == "ALVEOCLS") {
        ret.type = 4;
      } else if (label_type == "TF1KERAS") {
        ret.type = 5;
      } else if (label_type == "CAR_COLOR_CHEN") {
        ret.type = 6;
      }
    }
  }
  __TOC__(CLASSIFY_POST_ARM)
  //__TOC__(CLASSIFY_E2E_TIME)
  return rets;
}

std::vector<ClassificationResult> ClassificationImp::run(
    const std::vector<vart::xrt_bo_t>& input_bos) {
  auto postprocess_index = 0;
  if (configurable_dpu_task_->getConfig()
          .classification_param()
          .has_avg_pool_param()) {
    __TIC__(CLASSIFY_DPU_0)
    configurable_dpu_task_->run_with_xrt_bo(input_bos);
    __TOC__(CLASSIFY_DPU_0)

    auto avg_scale = configurable_dpu_task_->getConfig()
                         .classification_param()
                         .avg_pool_param()
                         .scale();
    auto batch_size = configurable_dpu_task_->getInputTensor()[0][0].batch;

    __TIC__(CLASSIFY_AVG_POOL)
    for (auto batch_idx = 0u; batch_idx < batch_size; batch_idx++) {
      vitis::ai::globalAvePool(
          (int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(
              batch_idx),
          // 1024, 9, 3,
          configurable_dpu_task_->getOutputTensor()[0][0].channel,
          configurable_dpu_task_->getOutputTensor()[0][0].width,
          configurable_dpu_task_->getOutputTensor()[0][0].height,
          (int8_t*)configurable_dpu_task_->getInputTensor()[1][0].get_data(
              batch_idx),
          avg_scale);
    }
    __TOC__(CLASSIFY_AVG_POOL)
    __TIC__(CLASSIFY_DPU_1)
    configurable_dpu_task_->run(1);
    __TOC__(CLASSIFY_DPU_1)
    postprocess_index = 1;
  } else {
    __TIC__(CLASSIFY_DPU)
    configurable_dpu_task_->run_with_xrt_bo(input_bos);
    __TOC__(CLASSIFY_DPU)
  }

  __TIC__(CLASSIFY_POST_ARM)
  auto rets = classification_post_process(
      configurable_dpu_task_->getInputTensor()[postprocess_index],
      configurable_dpu_task_->getOutputTensor()[postprocess_index],
      configurable_dpu_task_->getConfig());

  for (auto& ret : rets) {
    if (configurable_dpu_task_->getOutputTensor()[postprocess_index][0]
            .channel == 1001) {
      for (auto& s : ret.scores) {
        s.index--;
      }
    }

    ret.type = 0;
    if (!configurable_dpu_task_->getConfig()
             .classification_param()
             .label_type()
             .empty()) {
      auto label_type = configurable_dpu_task_->getConfig()
                            .classification_param()
                            .label_type();
      if (label_type == "CIFAR10") {
        ret.type = 1;
      } else if (label_type == "FMNIST") {
        ret.type = 2;
      } else if (label_type == "ORIEN") {
        ret.type = 3;
      } else if (label_type == "ALVEOCLS") {
        ret.type = 4;
      } else if (label_type == "TF1KERAS") {
        ret.type = 5;
      } else if (label_type == "CAR_COLOR_CHEN") {
        ret.type = 6;
      }
    }
  }
  __TOC__(CLASSIFY_POST_ARM)
  return rets;
}

}  // namespace ai
}  // namespace vitis
