/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ssd.h"

namespace detect {

using namespace cv;
using namespace std;

SSDdetector::SSDdetector(unsigned int num_classes, CodeType code_type,
                         bool variance_encoded_in_target,
                         unsigned int keep_top_k,
                         const vector<float>& confidence_threshold,
                         unsigned int nms_top_k, float nms_threshold, float eta,
                         const vector<shared_ptr<vector<float>>>& priors,
                         float scale, bool clip)
    : num_classes_(num_classes),
      code_type_(code_type),
      variance_encoded_in_target_(variance_encoded_in_target),
      keep_top_k_(keep_top_k),
      confidence_threshold_(confidence_threshold),
      nms_top_k_(nms_top_k),
      nms_threshold_(nms_threshold),
      eta_(eta),
      priors_(priors),
      scale_(scale),
      clip_(clip) {
  num_priors_ = priors_.size();
  nms_confidence_ = *min_element(confidence_threshold_.begin() + 1,
                                 confidence_threshold_.end());
}

template <typename T>
void SSDdetector::Detect(const T* loc_data, const float* conf_data,
                         MultiDetObjects* result) {
  decoded_bboxes_.clear();
  const T(*bboxes)[4] = (const T(*)[4])loc_data;

  unsigned int num_det = 0;
  vector<vector<int>> indices(num_classes_);
  vector<vector<pair<float, int>>> score_index_vec(num_classes_);

  // Get top_k scores (with corresponding indices).
  GetMultiClassMaxScoreIndexMT(conf_data, 1, num_classes_ - 1,
                               &score_index_vec);

  for (auto c = 1u; c < num_classes_; ++c) {
    // Perform NMS for one class
    ApplyOneClassNMS(bboxes, conf_data, c, score_index_vec[c], &(indices[c]));

    num_det += indices[c].size();
  }

  if (keep_top_k_ > 0 && num_det > keep_top_k_) {
    vector<tuple<float, int, int>> score_index_tuples;
    for (auto label = 0u; label < num_classes_; ++label) {
      const vector<int>& label_indices = indices[label];
      for (auto j = 0u; j < label_indices.size(); ++j) {
        auto idx = label_indices[j];
        auto score = conf_data[idx * num_classes_ + label];
        score_index_tuples.emplace_back(score, label, idx);
      }
    }

    // Keep top k results per image.
    sort(score_index_tuples.begin(), score_index_tuples.end(),
         [](const tuple<float, int, int>& lhs,
            const tuple<float, int, int>& rhs) {
           return get<0>(lhs) > get<0>(rhs);
         });
    score_index_tuples.resize(keep_top_k_);

    indices.clear();
    indices.resize(num_classes_);
    for (auto& item : score_index_tuples) {
      indices[get<1>(item)].push_back(get<2>(item));
    }

    num_det = keep_top_k_;
  }

  for (auto label = 1u; label < indices.size(); ++label) {
    for (auto idx : indices[label]) {
      auto score = conf_data[idx * num_classes_ + label];
      if (score < confidence_threshold_[label]) {
        continue;
      }
      auto& bbox = decoded_bboxes_[idx];
      bbox[0] = max(min(bbox[0], 1.f), 0.f);
      bbox[1] = max(min(bbox[1], 1.f), 0.f);
      bbox[2] = max(min(bbox[2], 1.f), 0.f);
      bbox[3] = max(min(bbox[3], 1.f), 0.f);

      auto box_rect =
          Rect_<float>(Point2f(bbox[0], bbox[1]), Point2f(bbox[2], bbox[3]));
      result->emplace_back(label, score, box_rect);
    }
  }
}

template <typename T>
void SSDdetector::ApplyOneClassNMS(
    const T (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices) {
  // Do nms.
  float adaptive_threshold = nms_threshold_;
  indices->clear();
  unsigned int i = 0;
  while (i < score_index_vec.size()) {
    const int idx = score_index_vec[i].second;
    if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
      DecodeBBox(bboxes, idx, true);
    }

    bool keep = true;
    for (auto k = 0u; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes, idx, kept_idx);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    ++i;
    if (keep && eta_ < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta_;
    }
  }
}

void SSDdetector::GetOneClassMaxScoreIndex(
    const float* conf_data, int label,
    vector<pair<float, int>>* score_index_vec) {
  conf_data += label;
  for (int i = 0; i < num_priors_; ++i) {
    auto score = *conf_data;
    if (score > nms_confidence_) {
      score_index_vec->emplace_back(score, i);
    }
    conf_data += num_classes_;
  }
  stable_sort(score_index_vec->begin(), score_index_vec->end(),
              [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
                return lhs.first > rhs.first;
              });

  if (nms_top_k_ < score_index_vec->size()) {
    score_index_vec->resize(nms_top_k_);
  }
}

void SSDdetector::GetMultiClassMaxScoreIndex(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec) {
  for (auto i = start_label; i < start_label + num_classes; ++i) {
    GetOneClassMaxScoreIndex(conf_data, i, &((*score_index_vec)[i]));
  }
}

/**
 * @brief GetMultiClassMaxScoreIndexMT - get multiple max score index in
 * multiple work thread
 */
void SSDdetector::GetMultiClassMaxScoreIndexMT(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec, int threads) {
  int thread_classes = num_classes / threads;
  int last_thread_classes = num_classes % threads + thread_classes;

  vector<thread> workers;

  auto c = start_label;
  for (auto i = 0; i < threads - 1; ++i) {
    workers.emplace_back(&SSDdetector::GetMultiClassMaxScoreIndex, this,
                         conf_data, c, thread_classes, score_index_vec);
    c += thread_classes;
  }
  workers.emplace_back(&SSDdetector::GetMultiClassMaxScoreIndex, this,
                       conf_data, c, last_thread_classes, score_index_vec);

  for (auto& worker : workers) {
    if (worker.joinable()) worker.join();
  }
}

/**
 * @brief BBoxSize - calculate the size of bounding box
 */
void BBoxSize(vector<float>& bbox, bool normalized) {
  float width = bbox[2] - bbox[0];
  float height = bbox[3] - bbox[1];
  if (width > 0 && height > 0) {
    if (normalized) {
      bbox[4] = width * height;
    } else {
      bbox[4] = (width + 1) * (height + 1);
    }
  } else {
    bbox[4] = 0.f;
  }
}

/**
 * @brief IntersectBBoxSize - calculate the intersect of two bounding boxes
 */
float IntersectBBoxSize(const vector<float>& bbox1, const vector<float>& bbox2,
                        bool normalized) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] || bbox2[1] > bbox1[3] ||
      bbox2[3] < bbox1[1]) {
    // Return 0 if there is no intersection.
    return 0.f;
  }

  vector<float> intersect_bbox(5);
  intersect_bbox[0] = max(bbox1[0], bbox2[0]);
  intersect_bbox[1] = max(bbox1[1], bbox2[1]);
  intersect_bbox[2] = min(bbox1[2], bbox2[2]);
  intersect_bbox[3] = min(bbox1[3], bbox2[3]);
  BBoxSize(intersect_bbox, normalized);
  return intersect_bbox[4];
}

/**
 * @brief JaccardOverlap - calculate Jaccard Overlap
 */
template <typename T>
float SSDdetector::JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
                                  bool normalized) {
  const vector<float>& bbox1 = decoded_bboxes_[idx];
  const vector<float>& bbox2 = decoded_bboxes_[kept_idx];
  float intersect_size = IntersectBBoxSize(bbox1, bbox2, normalized);
  return intersect_size <= 0
             ? 0
             : intersect_size / (bbox1[4] + bbox2[4] - intersect_size);
}

template float SSDdetector::JaccardOverlap(const int (*bboxes)[4], int idx,
                                           int kept_idx, bool normalized);
template float SSDdetector::JaccardOverlap(const int8_t (*bboxes)[4], int idx,
                                           int kept_idx, bool normalized);

template <typename T>
void SSDdetector::DecodeBBox(const T (*bboxes)[4], int idx, bool normalized) {
  vector<float> bbox(5, 0);
  // scale bboxes
  transform(bboxes[idx], bboxes[idx] + 4, bbox.begin(),
            bind2nd(multiplies<float>(), scale_));

  auto& prior_bbox = priors_[idx];

  if (code_type_ == CodeType::CORNER) {
    if (variance_encoded_in_target_) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      transform(bbox.begin(), bbox.end(), prior_bbox->begin() + 4, bbox.begin(),
                multiplies<float>());
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    }
  } else if (code_type_ == CodeType::CENTER_SIZE) {
    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    if (variance_encoded_in_target_) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decode_bbox_center_x = bbox[0] * (*prior_bbox)[10] + (*prior_bbox)[8];
      decode_bbox_center_y = bbox[1] * (*prior_bbox)[11] + (*prior_bbox)[9];
      decode_bbox_width = exp(bbox[2]) * (*prior_bbox)[10];
      decode_bbox_height = exp(bbox[3]) * (*prior_bbox)[11];
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox_center_x =
          (*prior_bbox)[4] * bbox[0] * (*prior_bbox)[10] + (*prior_bbox)[8];
      decode_bbox_center_y =
          (*prior_bbox)[5] * bbox[1] * (*prior_bbox)[11] + (*prior_bbox)[9];
      decode_bbox_width = exp((*prior_bbox)[6] * bbox[2]) * (*prior_bbox)[10];
      decode_bbox_height = exp((*prior_bbox)[7] * bbox[3]) * (*prior_bbox)[11];
    }

    bbox[0] = decode_bbox_center_x - decode_bbox_width / 2.;
    bbox[1] = decode_bbox_center_y - decode_bbox_height / 2.;
    bbox[2] = decode_bbox_center_x + decode_bbox_width / 2.;
    bbox[3] = decode_bbox_center_y + decode_bbox_height / 2.;
  } else if (code_type_ == CodeType::CORNER_SIZE) {
    if (variance_encoded_in_target_) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      bbox[0] *= (*prior_bbox)[10];
      bbox[1] *= (*prior_bbox)[11];
      bbox[2] *= (*prior_bbox)[10];
      bbox[3] *= (*prior_bbox)[11];
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      bbox[0] *= (*prior_bbox)[10];
      bbox[1] *= (*prior_bbox)[11];
      bbox[2] *= (*prior_bbox)[10];
      bbox[3] *= (*prior_bbox)[11];
      transform(bbox.begin(), bbox.end(), prior_bbox->begin() + 4, bbox.begin(),
                multiplies<float>());
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    }
  } else {
    // LOG(FATAL) << "Unknown LocLossType.";
  }

  BBoxSize(bbox, normalized);

  decoded_bboxes_.emplace(idx, move(bbox));
}

template void SSDdetector::DecodeBBox(const int (*bboxes)[4], int idx,
                                      bool normalized);
template void SSDdetector::DecodeBBox(const int8_t (*bboxes)[4], int idx,
                                      bool normalized);

/**
 * @brief PriorBoxes - construct a PriorBoxes object
 *
 */
PriorBoxes::PriorBoxes(int image_width, int image_height, int layer_width,
                       int layer_height, const vector<float>& variances,
                       const vector<float>& min_sizes,
                       const vector<float>& max_sizes,
                       const vector<float>& aspect_ratios, float offset,
                       float step_width, float step_height, bool flip,
                       bool clip)
    : offset_(offset), clip_(clip) {
  // Store image dimensions and layer dimensions
  image_dims_ = make_pair(image_width, image_height);
  layer_dims_ = make_pair(layer_width, layer_height);

  // Compute step width and height
  if (step_width == 0 || step_height == 0) {
    step_dims_ =
        make_pair(static_cast<float>(image_dims_.first) / layer_dims_.first,
                  static_cast<float>(image_dims_.second) / layer_dims_.second);
  } else {
    step_dims_ = make_pair(step_width, step_height);
  }

  // Store box variances
  if (variances.size() == 4) {
    variances_ = variances;
  } else if (variances.size() == 1) {
    variances_.resize(4);
    fill_n(variances_.begin(), 4, variances[0]);
  } else {
    variances_.resize(4);
    fill_n(variances_.begin(), 4, 0.1f);
  }

  // Generate boxes' dimensions
  for (auto i = 0u; i < min_sizes.size(); ++i) {
    // first prior: aspect_ratio = 1, size = min_size
    boxes_dims_.emplace_back(min_sizes[i], min_sizes[i]);
    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
    if (!max_sizes.empty()) {
      boxes_dims_.emplace_back(sqrt(min_sizes[i] * max_sizes[i]),
                               sqrt(min_sizes[i] * max_sizes[i]));
    }
    // rest of priors
    for (auto ar : aspect_ratios) {
      float w = min_sizes[i] * sqrt(ar);
      float h = min_sizes[i] / sqrt(ar);
      boxes_dims_.emplace_back(w, h);
      if (flip) boxes_dims_.emplace_back(h, w);
    }
  }

  // automatically create priors
  CreatePriors();
}

/**
 * @brief CreatePriors - create a PriorBoxes
 *
 */
void PriorBoxes::CreatePriors() {
  for (int h = 0; h < layer_dims_.second; ++h) {
    for (int w = 0; w < layer_dims_.first; ++w) {
      float center_x = (w + offset_) * step_dims_.first;
      float center_y = (h + offset_) * step_dims_.second;
      for (auto& dims : boxes_dims_) {
        auto box = make_shared<vector<float>>(12);
        // xmin, ymin, xmax, ymax
        (*box)[0] = (center_x - dims.first / 2.) / image_dims_.first;
        (*box)[1] = (center_y - dims.second / 2.) / image_dims_.second;
        (*box)[2] = (center_x + dims.first / 2.) / image_dims_.first;
        (*box)[3] = (center_y + dims.second / 2.) / image_dims_.second;

        if (clip_) {
          for (int i = 0; i < 4; ++i) (*box)[i] = min(max((*box)[i], 0.f), 1.f);
        }
        // variances
        copy_n(variances_.begin(), 4, box->data() + 4);
        // centers and dimensions
        (*box)[8] = 0.5f * ((*box)[0] + (*box)[2]);
        (*box)[9] = 0.5f * ((*box)[1] + (*box)[3]);
        (*box)[10] = (*box)[2] - (*box)[0];
        (*box)[11] = (*box)[3] - (*box)[1];

        priors_.push_back(move(box));
      }
    }
  }
}

/**
 * @brief Create - Create prior boxes according to SSD type
 *
 * @param vecPriors - the vector of prior boxes
 * @param type - SSD type: vehicle or person
 */
void PriorBoxes::Create(vector<shared_ptr<vector<float>>>& vec_priors,
                        SSD_TYPE type) {
  vector<float> variances{0.1, 0.1, 0.2, 0.2};
  vector<PriorBoxes> prior_boxes;
  if (type == SSD_TYPE::VEHICLE) {
    prior_boxes.emplace_back(PriorBoxes{480,
                                        360,
                                        60,
                                        45,
                                        variances,
                                        {15.0, 30},
                                        {33.0, 60},
                                        {2},
                                        0.5,
                                        8.0,
                                        8.0});
    // fc7_mbox_priorbox
    prior_boxes.emplace_back(PriorBoxes{
        480, 360, 30, 23, variances, {66.0}, {127.0}, {2, 3}, 0.5, 16, 16});
    // conv6_2_mbox_priorbox
    prior_boxes.emplace_back(PriorBoxes{
        480, 360, 15, 12, variances, {127.0}, {188.0}, {2, 3}, 0.5, 32, 32});
    // conv7_2_mbox_priorbox
    prior_boxes.emplace_back(PriorBoxes{
        480, 360, 8, 6, variances, {188.0}, {249.0}, {2, 3}, 0.5, 64, 64});
    // conv8_2_mbox_priorbox
    prior_boxes.emplace_back(PriorBoxes{
        480, 360, 6, 4, variances, {249.0}, {310.0}, {2}, 0.5, 100, 100});
    // conv9_2_mbox_priorbox
    prior_boxes.emplace_back(PriorBoxes{
        480, 360, 4, 2, variances, {310.0}, {372.0}, {2}, 0.5, 300, 300});

  } else if (type == SSD_TYPE::PERSON) {
    prior_boxes.emplace_back(PriorBoxes{640,
                                        360,
                                        80,
                                        45,
                                        variances,
                                        {18, 30, 50},
                                        {40, 60, 80},
                                        {2},
                                        0.5,
                                        8,
                                        8});
    prior_boxes.emplace_back(PriorBoxes{
        640, 360, 40, 23, variances, {72.0}, {160.0}, {2, 3}, 0.5, 16, 16});
    prior_boxes.emplace_back(PriorBoxes{
        640, 360, 20, 12, variances, {160.0}, {240.0}, {2, 3}, 0.5, 32, 32});
    prior_boxes.emplace_back(PriorBoxes{
        640, 360, 10, 6, variances, {240.0}, {320.0}, {2, 3}, 0.5, 64, 64});
    prior_boxes.emplace_back(PriorBoxes{
        640, 360, 8, 4, variances, {320.0}, {400.0}, {2}, 0.5, 100, 100});
    prior_boxes.emplace_back(PriorBoxes{
        640, 360, 6, 2, variances, {350.0}, {480.0}, {2}, 0.5, 300, 300});
  }

  int num_priors = 0;
  for (auto& p : prior_boxes) {
    num_priors += p.priors().size();
  }

  vec_priors.clear();
  vec_priors.reserve(num_priors);
  for (auto i = 0U; i < prior_boxes.size(); ++i) {
    vec_priors.insert(vec_priors.end(), prior_boxes[i].priors().begin(),
                      prior_boxes[i].priors().end());
  }
}

/**
 * @brief Init - initialize SSD model
 *
 * @param kernel_name - the kernel name which is generated by DNNC
 * @param input_node - name of the input node in SSD model
 * @param output_loc - name of the location node in SSD model
 * @param output_conf - name of the confidence node in SSD model
 * @param th_nms - NMS threshold
 * @param top_k - the number of top_k
 * @param class_k - the number of class
 */
void SSD::Init(const string& path, const string& input_node,
               const string& output_loc, const string& output_conf,
               float th_nms, int top_k, int class_k) {
  input_node_ = input_node;
  GraphInfo shapes;
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  string ssdPath;
  kernel_name_ = "ssd_person";
  //  if (path.c_str()[path.length() - 1] == '/') {
  //    ssdPath = path + "ssd" + "/";
  //  } else {
  //    ssdPath = path + "/" + "ssd" + "/";
  //  }

  graph = xir::Graph::deserialize(path);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "ssd should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  ssd_runner = vart::Runner::create_runner(subgraph[0], "run");
  getTensorShape(ssd_runner.get(), &shapes, 1, {output_loc, output_conf});
  output_mapping = shapes.output_mapping;
  if ((kernel_name_ == "ssd") || (kernel_name_ == "vehicle")) {
    type_ = SSD_TYPE::VEHICLE;
    num_classes_ = 4;
    th_conf_ = {0, 0.3, 0.5, 0.25};
  } else {
    type_ = SSD_TYPE::PERSON;
    num_classes_ = 2;
    th_conf_ = {0, 0.3};
  }
  PriorBoxes::Create(priors_, type_);

  softmax_data_ = new float[priors_.size() * num_classes_];
  int batch = ssd_runner->get_input_tensors()[0]->get_shape().at(0);
  conf_size = shapes.outTensorList[1].size;
  loc_size = shapes.outTensorList[0].size;
  loc = new int8_t[loc_size * batch];
  conf = new int8_t[conf_size * batch];
  loc_scale =
      get_output_scale(ssd_runner->get_output_tensors()[output_mapping[0]]);
  detector_ = new SSDdetector(num_classes_, SSDdetector::CodeType::CENTER_SIZE,
                              false, class_k, th_conf_, top_k, th_nms, 1.0,
                              priors_, loc_scale);
}
/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const int8_t* data, size_t size, float scale,
                    float* result) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp(data[i] * scale);
    sum += result[i];
  }

  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

/**
 * @brief Run - target detection using SSD model
 *
 * @param img - the input image
 * @param results - the detected targets
 *
 */
void SSD::Run(Mat& img, MultiDetObjects* results) {
  auto inputTensors = cloneTensorBuffer(ssd_runner->get_input_tensors());
  auto outputTensors = cloneTensorBuffer(ssd_runner->get_output_tensors());
  auto input_scale = get_input_scale(ssd_runner->get_input_tensors()[0]);
  auto conf_output_scale =
      get_output_scale(ssd_runner->get_output_tensors()[output_mapping[1]]);

  int batchSize = ssd_runner->get_input_tensors()[0]->get_shape().at(0);
  int inSize = inshapes[0].size;
  int inHeight = inshapes[0].height;
  int inWidth = inshapes[0].width;
  int8_t* imageInputs = new int8_t[inSize * batchSize];
  float mean[3] = {104, 117, 123};
  Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
  cv::resize(img, image2, Size(inWidth, inHeight), 0, 0, cv::INTER_LINEAR);
  // for (int h =0; h< inHeight;h++)
  //    for (int w =0; w< inWidth;w++)
  //        for (int c =0; c< 3; c++)
  //            imageInputs[h*inWidth*3+w*3+c]
  //            =image2.at<Vec3b>(h,w)[c]-mean[c];

  for (int h = 0; h < inHeight; h++) {
    for (int w = 0; w < inWidth; w++) {
      for (int c = 0; c < 3; c++) {
        imageInputs[h * inWidth * 3 + w * 3 + c] =
            (int8_t)((image2.at<Vec3b>(h, w)[c] - mean[c]) * input_scale);
      }
    }
  }

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
      imageInputs, inputTensors[0].get()));
  outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
      loc, outputTensors[output_mapping[0]].get()));
  outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
      conf, outputTensors[output_mapping[1]].get()));

  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  inputsPtr.push_back(inputs[0].get());
  outputsPtr.push_back(outputs[0].get());
  outputsPtr.push_back(outputs[1].get());

  auto job_id = ssd_runner->execute_async(inputsPtr, outputsPtr);
  ssd_runner->wait(job_id.first, -1);
  for (int i = 0; i < conf_size / num_classes_; i++)
    CPUCalcSoftmax(&conf[i * num_classes_], num_classes_, conf_output_scale,
                   &softmax_data_[i * num_classes_]);
  _T(detector_->Detect(loc, softmax_data_, results));
  delete []imageInputs;
  return;
}

/**
 * @brief Finalize - release resource
 *
 */
void SSD::Finalize() {
  delete[] softmax_data_;
  delete[] loc;
  delete[] conf;
}

/**
 * @brief DrawBoxes - draw detected boxes on the image
 *
 * @param img - the input image
 * @param results - the detected targets
 *
 */
void SSD::DrawBoxes(Mat& img, MultiDetObjects& results) {
  for (size_t i = 0; i < results.size(); ++i) {
    int label = get<0>(results[i]);
    int xmin = get<2>(results[i]).x * img.cols;
    int ymin = get<2>(results[i]).y * img.rows;
    int xmax = xmin + (get<2>(results[i]).width) * img.cols;
    int ymax = ymin + (get<2>(results[i]).height) * img.rows;

    xmin = min(max(xmin, 0), img.cols);
    xmax = min(max(xmax, 0), img.cols);
    ymin = min(max(ymin, 0), img.rows);
    ymax = min(max(ymax, 0), img.rows);

    if (label == 1) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
                1, 0);
    } else if (label == 2) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(255, 0, 0), 1,
                1, 0);
    } else if (label == 3) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 1,
                1, 0);
    }
  }
}

SSD::~SSD() {}

}  // namespace detect
