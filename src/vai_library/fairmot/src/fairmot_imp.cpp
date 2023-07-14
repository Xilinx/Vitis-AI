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
#include "./fairmot_imp.hpp"

#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

static float sigmoid(float p, float scale) {
  return 1.0 / (1 + exp(-(p * scale) * 1.0));
}

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1, x2);
  float right = min(x1 + w1, x2 + w2);
  return right - left;
}

static float cal_iou(BoundingBox box, BoundingBox truth) {
  float w = overlap(box.x, box.width, truth.x, truth.width);
  float h = overlap(box.y, box.height, truth.y, truth.height);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area =
      box.width * box.height + truth.width * truth.height - inter_area;
  return inter_area * 1.0 / union_area;
}

FairMotImp::FairMotImp(const std::string& model_name, bool need_preprocess)
    : FairMot(model_name, need_preprocess) {}

FairMotImp::~FairMotImp() {}

FairMotResult fairmot_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config, int iw, int ih,
    size_t batch_idx) {
  int sWidth = input_tensors[0][0].width;
  int sHeight = input_tensors[0][0].height;

  const float NMS_THRESHOLD = config.fair_mot_param().nms_threshold();
  const float CONF_THRESHOLD = config.fair_mot_param().conf_threshold();

  auto layername =
      std::vector<std::string>(config.fair_mot_param().layer_name().begin(),
                               config.fair_mot_param().layer_name().end());
  std::vector<vitis::ai::library::OutputTensor> output_tensors;
  for (auto i = 0u; i < layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted[0].size(); j++) {
      if (output_tensors_unsorted[0][j].name.find(layername[i]) !=
          std::string::npos) {
        output_tensors.push_back(output_tensors_unsorted[0][j]);
        break;
      }
    }
  }

  auto* data_reg = (int8_t*)output_tensors[0].get_data(batch_idx);  // offset 2
  auto* data_hm = (int8_t*)output_tensors[1].get_data(batch_idx);   // score 1
  auto* data_wh = (int8_t*)output_tensors[2].get_data(batch_idx);   // wh 2
  auto* data_id = (int8_t*)output_tensors[3].get_data(batch_idx);   // feat 256
  float scale_reg = vitis::ai::library::tensor_scale(output_tensors[0]);
  float scale_hm = vitis::ai::library::tensor_scale(output_tensors[1]);
  float scale_wh = vitis::ai::library::tensor_scale(output_tensors[2]);
  float scale_id = vitis::ai::library::tensor_scale(output_tensors[3]);
  int channel_id = output_tensors[3].channel;
  // int channel_id =256;
  //
  int width = output_tensors[0].width;
  int height = output_tensors[0].height;
  auto size = width * height;
  float scores[size];
  float float_data_wh[size * 2];
  float float_data_reg[size * 2];
  transform(data_hm, data_hm + size, scores,
            std::bind2nd(multiplies<float>(), scale_hm));
  transform(data_wh, data_wh + size * 2, float_data_wh,
            std::bind2nd(multiplies<float>(), scale_wh));
  transform(data_reg, data_reg + size * 2, float_data_reg,
            std::bind2nd(multiplies<float>(), scale_reg));

  vector<pair<int, BoundingBox>> bbox_index_vec;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      int idx = h * width + w;
      scores[idx] = sigmoid(scores[idx], 1.0f);
      if (scores[idx] < CONF_THRESHOLD) continue;
      float rx =
          float(w) / float(width) + float_data_reg[2 * idx] / float(width);
      float ry = float(h) / float(height) +
                 float_data_reg[2 * idx + 1] / float(height);
      float rw = float_data_wh[2 * idx] / width;
      float rh = float_data_wh[2 * idx + 1] / height;
      rx = rx - rw / 2;
      ry = ry - rh / 2;
      rx = rx * width * 4 / iw;
      ry = ry * height * 4 / ih;
      rw = rw * width * 4 / iw;
      rh = rh * height * 4 / ih;
      BoundingBox bbox = BoundingBox{rx, ry, rw, rh, 1, scores[idx]};
      bbox_index_vec.push_back(make_pair(idx, bbox));
    }
  }
  // nms
  std::stable_sort(
      bbox_index_vec.begin(), bbox_index_vec.end(),
      [](const pair<int, BoundingBox>& lhs, const pair<int, BoundingBox>& rhs) {
        return lhs.second.score > rhs.second.score;
      });
  const size_t count = bbox_index_vec.size();
  vector<bool> exist_box(count, true);
  vector<Mat> feats;
  vector<BoundingBox> bboxes;
  for (size_t i = 0; i < count; ++i) {
    int idx = bbox_index_vec[i].first;
    if (!exist_box[i]) continue;
    bboxes.push_back(bbox_index_vec[i].second);
    float float_feat[channel_id];
    transform(data_id + idx * channel_id, data_id + (idx + 1) * channel_id,
              float_feat, std::bind2nd(multiplies<float>(), scale_id));
    Mat feat(1, channel_id, CV_32F, float_feat);
    normalize(feat, feat);
    feats.push_back(feat.clone());
    for (size_t j = i + 1; j < count; ++j) {
      if (!exist_box[j]) continue;
      float ovr = cal_iou(bbox_index_vec[i].second, bbox_index_vec[j].second);
      if (ovr >= NMS_THRESHOLD) exist_box[j] = false;
    }
  }
  FairMotResult result{sWidth, sHeight, feats, bboxes};
  return result;
}

std::vector<FairMotResult> fairmot_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, vector<int> ws,
    vector<int> hs) {
  auto batch = input_tensors[0][0].batch;
  auto ret = std::vector<FairMotResult>{};
  ret.reserve(batch);
  for (auto i = 0u; i < batch; i++) {
    ret.emplace_back(fairmot_post_process(input_tensors, output_tensors, config,
                                          ws[i], hs[i], i));
  }
  return ret;
}

FairMotResult FairMotImp::run(const cv::Mat& input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  int iWidth = input_image.cols;
  int iHeight = input_image.rows;
  int new_width = sWidth;
  int new_height = sHeight;
  auto size = cv::Size(sWidth, sHeight);
  if (size != input_image.size()) {
    float iratio = float(iWidth) / float(iHeight);
    float sratio = float(sWidth) / float(sHeight);
    Size new_size = size;
    if (iratio > sratio) {
      new_height = sWidth / iratio;
      new_size = Size(new_width, new_height);
      cv::resize(input_image, image, new_size);
      int pad_height = sHeight - new_height;
      Mat pad_img = Mat(pad_height, sWidth, CV_8UC3, Scalar(128, 128, 128));
      vconcat(image, pad_img, image);
      CHECK(image.size() == size) << image.size() << " vs: " << size;
    } else if (iratio < sratio) {
      new_width = sHeight * iratio;
      new_size = Size(new_width, new_height);
      cv::resize(input_image, image, new_size);
      int pad_width = sWidth - new_width;
      Mat pad_img = Mat(sHeight, pad_width, CV_8UC3, Scalar(128, 128, 128));
      hconcat(image, pad_img, image);
      CHECK(image.size() == size) << image.size() << " vs: " << size;
    } else
      cv::resize(input_image, image, new_size);
  } else {
    image = input_image;
  }
  __TIC__(FAIRMOT_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(FAIRMOT_SET_IMG)
  __TIC__(FAIRMOT_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FAIRMOT_DPU)
  __TIC__(FAIRMOT_POST_PROCESS)
  auto ret = fairmot_post_process(configurable_dpu_task_->getInputTensor(),
                                  configurable_dpu_task_->getOutputTensor(),
                                  configurable_dpu_task_->getConfig(),
                                  new_width, new_height, 0);
  __TOC__(FAIRMOT_POST_PROCESS)
  return ret;
}

std::vector<FairMotResult> FairMotImp::run(
    const std::vector<cv::Mat>& input_images) {
  vector<cv::Mat> images;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<int> newWidths;
  vector<int> newHeights;
  auto size = cv::Size(sWidth, sHeight);
  for (auto& input_image : input_images) {
    Mat image;
    int iWidth = input_image.cols;
    int iHeight = input_image.rows;
    int new_width = sWidth;
    int new_height = sHeight;
    if (size != input_image.size()) {
      float iratio = float(iWidth) / float(iHeight);
      float sratio = float(sWidth) / float(sHeight);
      Size new_size = size;
      if (iratio > sratio) {
        new_height = sWidth / iratio;
        new_size = Size(new_width, new_height);
        cv::resize(input_image, image, new_size);
        int pad_height = sHeight - new_height;
        Mat pad_img = Mat(pad_height, sWidth, CV_8UC3, Scalar(128, 128, 128));
        vconcat(image, pad_img, image);
        CHECK(image.size() == size) << image.size() << " vs: " << size;
      } else if (iratio < sratio) {
        new_width = sHeight * iratio;
        new_size = Size(new_width, new_height);
        cv::resize(input_image, image, new_size);
        int pad_width = sWidth - new_width;
        Mat pad_img = Mat(sHeight, pad_width, CV_8UC3, Scalar(128, 128, 128));
        hconcat(image, pad_img, image);
        CHECK(image.size() == size) << image.size() << " vs: " << size;
      } else
        cv::resize(input_image, image, size);
    } else {
      image = input_image;
    }
    images.push_back(image.clone());
    newWidths.push_back(new_width);
    newHeights.push_back(new_height);
  }
  __TIC__(FAIRMOT_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(FAIRMOT_SET_IMG)
  __TIC__(FAIRMOT_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FAIRMOT_DPU)
  __TIC__(FAIRMOT_POST_PROCESS)
  auto ret = fairmot_post_process(configurable_dpu_task_->getInputTensor(),
                                  configurable_dpu_task_->getOutputTensor(),
                                  configurable_dpu_task_->getConfig(),
                                  newWidths, newHeights);
  __TOC__(FAIRMOT_POST_PROCESS)
  return ret;
}

}  // namespace ai
}  // namespace vitis
