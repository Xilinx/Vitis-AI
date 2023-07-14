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
#include "ftd_structure.hpp"

#include <glog/logging.h>

namespace xilinx {
namespace ai {

FTD_Structure::FTD_Structure(const SpecifiedCfg &specified_cfg) {
  CHECK(id_record.empty()) << "id_record must be empty when initial";
  id_record.push_back(0);
  iou_threshold = 0.3f;
  score_threshold = 0.f;
  specified_cfg_ = specified_cfg;
}

FTD_Structure::~FTD_Structure() { this->clear(); }

void FTD_Structure::clear() {
  tracks.clear();
  id_record.clear();
  remove_id_this_frame.clear();
  id_record.push_back(0);
}

void FindRemain(std::vector<int> &input, std::vector<int> &output, int len) {
  output.clear();
  for (int i = 0; i < len; i++) {
    if (input.empty() ||
        std::find(input.begin(), input.end(), i) == input.end()) {
      output.push_back(i);
    }
  }
  CHECK((int)(input.size() + output.size()) == len)
      << "error happen in function FindRemain";
}

float GetIou(const cv::Rect_<float> &rect1, const cv::Rect_<float> &rect2) {
  float inner = (rect1 & rect2).area();
  float univer = rect1.area() + rect2.area() - inner;
  return (inner / univer);
}

float GetCoverRatio(const cv::Rect_<float> &rect1,
                    const cv::Rect_<float> &rect2) {
  float inner = (rect1 & rect2).area();
  return (inner / rect1.area());
}

void FTD_Structure::GetOut(std::vector<OutputCharact> &output_characts) {
  CHECK(output_characts.size() == 0) << "error output_characts size";
  for (auto ti = tracks.begin(); ti != tracks.end();) {
    auto status = (*ti)->GetStatus();
#ifdef _FTD_DEBUG_
    if (status == 1)
      LOG(INFO) << "show " << (*ti)->GetId();
    else if (status == 0)
      LOG(INFO) << "hide " << (*ti)->GetId();
    else
      LOG(INFO) << "remove " << (*ti)->GetId();
#endif
    switch (status) {
      case 1: {
        auto oout = (*ti)->GetOut();
        std::get<1>(oout) = (std::get<1>(oout) & roi_range);
        output_characts.push_back(oout);
        ti++;
      } break;
      case 0:
        ti++;
        break;
      case -1:;
      default: {
        // recode remove_id
        if ((*ti)->GetShown() == true)
          remove_id_this_frame.push_back((*ti)->GetId());
        // recycle id
        if ((*ti)->GetShown() == false)
          id_record.insert(id_record.end() - 1, (*ti)->GetId());
        ti = tracks.erase(ti);
      } break;
    }
  }
}

std::vector<OutputCharact> FTD_Structure::Update(
    uint64_t frame_id, bool detect_flag, int mode,
    std::vector<InputCharact> &input_characts) {
  remove_id_this_frame.clear();
#ifdef _FTD_DEBUG_
  LOG(INFO) << "frame " << frame_id << " detect_flag " << detect_flag;
#endif
  // get range of frame and check detect_flag
  std::vector<OutputCharact> output_characts;
  if (detect_flag == false)
    CHECK(input_characts.size() == 0) << "error input_characts size";
  roi_range = cv::Rect_<float>(0.f, 0.f, 1.f, 1.f);
// show and prune predict
#ifdef _FTD_DEBUG_
  LOG(INFO) << "there are already " << tracks.size()
            << " trajectory(id predict_bbox):";
#endif
  for (auto ti = tracks.begin(); ti != tracks.end();) {
    (*ti)->Predict();
    auto track_id = (*ti)->GetId();
    auto track_rect = std::get<0>((*ti)->GetCharact());
    if (track_rect.width <= 0.f || track_rect.height <= 0.f) {
#ifdef _FTD_DEBUG_
      LOG(INFO) << "trajectory " << track_id << " predict fail, remove "
                << track_id;
#endif
      // recode remove_id
      if ((*ti)->GetShown() == true) remove_id_this_frame.push_back(track_id);
      // recycle id
      if ((*ti)->GetShown() == false)
        id_record.insert(id_record.end() - 1, (*ti)->GetId());
      ti = tracks.erase(ti);
    } else {
#ifdef _FTD_DEBUG_
      LOG(INFO) << track_id << " " << track_rect;
#endif
      ti++;
    }
  }
  if (detect_flag == false) {
    for (auto ti : tracks) ti->UpdateWithoutDetect();
    GetOut(output_characts);
    return output_characts;
  }
// show detect
#ifdef _FTD_DEBUG_
  LOG(INFO) << "there are " << input_characts.size()
            << " new detections(bbox):";
#endif
  for (auto ici = input_characts.begin(); ici != input_characts.end();) {
    auto rect = std::get<0>(*ici);
    auto ici_score = std::get<1>(*ici);
    rect = rect & roi_range;
    if (rect.width <= 0.f || rect.height <= 0.f ||
        ici_score < score_threshold) {
      ici = input_characts.erase(ici);
    } else {
      std::get<0>(*ici) = rect;
#ifdef _FTD_DEBUG_
      LOG(INFO) << rect;
#endif
      ici++;
    }
  }

  /*cal iou between predict and det*/
  std::vector<float> iou_score;
  std::vector<bool> iou_flag;
  std::vector<int> iou_index;
  int index = 0;
  for (auto &t : tracks) {
    for (auto &ic : input_characts) {
      auto rect_t = std::get<0>(t->GetCharact());
      auto label_t = std::get<2>(t->GetCharact());
      auto rect_i = std::get<0>(ic);
      auto label_i = std::get<2>(ic);
      iou_score.push_back(label_t == label_i ? GetIou(rect_t, rect_i) : 0.f);
      // LOG(INFO)<<iou_score.back();
      iou_flag.push_back(true);
      iou_index.push_back(index++);
    }
  }
  // CHECK SIZE for all matrix
  int divide1 = tracks.size();
  int divide2 = input_characts.size();
  CHECK((int)iou_score.size() == divide1 * divide2) << "iou_score size error";
  CHECK((int)iou_flag.size() == divide1 * divide2) << "iou_flag size error";
  CHECK((int)iou_index.size() == divide1 * divide2) << "iou_index size error";

  // sort iou_matrix to index
  std::sort(iou_index.begin(), iou_index.end(),
            [&iou_score](int a, int b) { return iou_score[a] > iou_score[b]; });
  // greedy find match track and detect number
  std::vector<int> match_track;
  std::vector<int> match_detect;
  for (auto iii = iou_index.begin();
       (iii != iou_index.end()) && (iou_score[*iii] >= iou_threshold); iii++) {
    if (!iou_flag[*iii]) continue;
    match_track.push_back(*iii / divide2);
    match_detect.push_back(*iii % divide2);
    for (int i = 0; i < divide2; i++)
      iou_flag[match_track.back() * divide2 + i] = false;
    for (int i = 0; i < divide1; i++)
      iou_flag[match_detect.back() + i * divide2] = false;
  }
  CHECK(match_track.size() == match_detect.size())
      << "match_track and match_detect must have the same size";

  // find unmatch_track and unmatch_detect number by order
  std::vector<int> unmatch_track;
  FindRemain(match_track, unmatch_track, divide1);
  std::vector<int> unmatch_detect;
  FindRemain(match_detect, unmatch_detect, divide2);

  /*new detection with new id*/
  for (unsigned int i = 0; i < unmatch_detect.size(); i++) {
    // reid for new detection here
    tracks.push_back(std::make_shared<FTD_Trajectory>(specified_cfg_));
    tracks.back()->Init(input_characts[unmatch_detect[i]], id_record, mode);
  }

  /*unmatch_track update with filter*/
  for (unsigned int i = 0; i < unmatch_track.size(); i++) {
    tracks[unmatch_track[i]]->UpdateTrack();
  }

  /*strategy for match_track detect and unmatch detect*/
  for (unsigned int i = 0; i < match_track.size(); i++) {
    tracks[match_track[i]]->UpdateDetect(input_characts[match_detect[i]]);
  }

  GetOut(output_characts);
  return output_characts;
}

std::vector<int> FTD_Structure::GetRemoveID() { return remove_id_this_frame; }

}  // namespace ai
}  // namespace xilinx
