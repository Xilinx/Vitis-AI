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
#include "ftd_trajectory.hpp"

#include <glog/logging.h>

namespace xilinx {
namespace ai {

FTD_Trajectory::FTD_Trajectory(SpecifiedCfg &specified_cfg) {
  this->B2G = std::get<1>(specified_cfg)[0];
  this->G2B = std::get<1>(specified_cfg)[1];
  this->B2D = std::get<1>(specified_cfg)[2];
  specified_cfg_ = specified_cfg;
}

void FTD_Trajectory::Predict() { std::get<0>(charact) = filter.GetPre(); }

int FTD_Trajectory::GetId() { return id; }

InputCharact &FTD_Trajectory::GetCharact() { return charact; }

void FTD_Trajectory::Init(const InputCharact &input_charact,
                          std::vector<uint64_t> &id_record, int mode) {
  // Init id and charact
  CHECK(!id_record.empty()) << "id_record must not be empty";
  if (id_record.size() == 1) {
    id = id_record[0];
    id_record[0] += 1;
  } else {
    id = id_record[0];
    id_record.erase(id_record.begin());
  }
  charact = input_charact;
#ifdef _FTD_DEBUG_
  LOG(INFO) << "Init a new trajectory(id " << id << ", bbox "
            << std::get<0>(charact) << ", label " << std::get<2>(charact)
            << ")";
#endif
  // Init FTD_Tracker and FTD_Filter
  filter.Init(std::get<0>(charact), mode, specified_cfg_);
  // Init others
  status = 0;
  leap = 1;
  have_been_shown = false;
  if (leap >= FTD_Trajectory::B2G) {
    status = 1;
    leap = 0;
    have_been_shown = true;
  }
}

void FTD_Trajectory::UpdateDetect(const InputCharact &input_charact) {
  // Update charact
  CHECK(std::get<2>(charact) == std::get<2>(input_charact))
      << "UpdateDetect must have the same label";
  charact = input_charact;
#ifdef _FTD_DEBUG_
  LOG(INFO) << "trajectory " << id << " update_detector with bbox "
            << std::get<0>(charact);
#endif
  // Init FTD_Tracker and Update FTD_Filter
  filter.UpdateDetect(std::get<0>(charact));
  // Update life
  if (status == 1) {
    leap = 0;
  } else if (status == 0) {
    leap = leap < 0 ? 0 : (leap + 1);
    if (leap >= FTD_Trajectory::B2G) {
      leap = 0;
      status = 1;
      have_been_shown = true;
    }
  }
}

void FTD_Trajectory::UpdateTrack() {
  // update FTD_Tracker and Update FTD_Filter
  filter.UpdateFilter();
#ifdef _FTD_DEBUG_
  LOG(INFO) << "trajectory " << id << " update_filter";
#endif
  std::get<3>(charact) = -1;
  // Update life
  if (status == 1) {
    leap -= 1;
    if (leap <= 0 - FTD_Trajectory::G2B) {
      leap = 0;
      status = 0;
    }
  } else if (status == 0) {
    leap = leap > 0 ? 0 : (leap - 1);
    if (leap <= 0 - FTD_Trajectory::B2D) {
      leap = 0;
      status = -1;
    }
  }
}

void FTD_Trajectory::UpdateWithoutDetect() {
  // update FTD_Tracker and Update FTD_Filter
  filter.UpdateFilter();
#ifdef _FTD_DEBUG_
  LOG(INFO) << "trajectory " << id << " update_without_detect";
#endif
  std::get<3>(charact) = -1;
}

int FTD_Trajectory::GetStatus() { return status; }

bool FTD_Trajectory::GetShown() { return have_been_shown; }

OutputCharact FTD_Trajectory::GetOut() {
  return std::make_tuple(id, filter.GetPost(), std::get<1>(charact),
                         std::get<2>(charact), std::get<3>(charact));
}

}  // namespace ai
}  // namespace xilinx
