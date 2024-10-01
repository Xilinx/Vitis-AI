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

#include "tracker_imp.hpp"
#include <thread>
#include "ftd/ftd_structure.hpp"
#include "state_map.hpp"

namespace vitis {
namespace ai {

ReidTrackerImp::ReidTrackerImp(uint64_t mode, const SpecifiedCfg& cfg) {
  mode_ = mode;
  ftd_ = new FTD_Structure(cfg);
  if (mode & MODE_MULTIDETS) {
    sm_ = new StateMap();
    sm_->addBadState(StateMap::DET_TO);
    sm_->addBadState(StateMap::TRC_ED);

    undet_tracks_ =
        new RingQueue<std::pair<uint64_t, std::vector<OutputCharact>>>(100);
  }
}

ReidTrackerImp::~ReidTrackerImp() {
  delete ftd_;
  if (mode_ & MODE_MULTIDETS) {
    delete sm_;
    delete undet_tracks_;
  }
}

std::vector<int> ReidTrackerImp::GetRemoveID() { return ftd_->GetRemoveID(); }

void ReidTrackerImp::clear() {
  ftd_->clear();
  if (mode_ & MODE_MULTIDETS) {
    undet_tracks_->clear();
    sm_->clearAll();
    sm_->addBadState(StateMap::DET_TO);
    sm_->addBadState(StateMap::TRC_ED);
  } else {
    lastframe_id = 0;
  }
}

std::vector<OutputCharact> ReidTrackerImp::patchFrame(const uint64_t frame_id) {
  std::vector<InputCharact> empty_charact;
  std::vector<OutputCharact> det_track;
  if (!(mode_ & MODE_MULTIDETS)) {
    det_track = ftd_->Update(frame_id, false, true, empty_charact);
  }
  return det_track;
}

std::vector<OutputCharact> ReidTrackerImp::track(
    const uint64_t frame_id, std::vector<InputCharact>& input_characts,
    const bool is_detection, const bool is_normalized) {
  std::vector<OutputCharact> det_track;
  if (mode_ & MODE_MULTIDETS) {
    return det_track;
  }

  if (lastframe_id && (frame_id > lastframe_id + 1) &&
      (mode_ & MODE_AUTOPATCH)) {
    exit(1);
    for (uint64_t i = 1; i < frame_id - lastframe_id; i++) {
      patchFrame(lastframe_id + i);
    }
  }
  det_track =
      ftd_->Update(frame_id, is_detection, is_normalized, input_characts);

  lastframe_id = frame_id;
  return det_track;
}

bool ReidTrackerImp::addDetStart(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    return sm_->add(frame_id, StateMap::DET_ST);
  }
  return false;
}

bool ReidTrackerImp::setDetEnd(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    return sm_->set(frame_id, StateMap::DET_ED);
  }
  return false;
}

bool ReidTrackerImp::setDetTimeout(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    return sm_->set(frame_id, StateMap::DET_TO);
  }
  return false;
}

bool ReidTrackerImp::setTrackLock(int frame_id, int timeout, int interval) {
  if (mode_ & MODE_MULTIDETS) {
    int time = 0;
    while (sm_->getCurId() != frame_id) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_REID_TRACKER))
          << "Track: wait for mutex, fid: " << frame_id
          << " cur_id: " << sm_->getCurId();
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      if ((time += interval) > timeout) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_REID_TRACKER))
            << "Track: setTrackLock timeout.";
        return false;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_REID_TRACKER))
        << "Track: start track, fid: " << frame_id;
    sm_->updateLastTrackedId(frame_id);
    return sm_->set(frame_id, StateMap::TRC_ST);
  }
  return false;
}

bool ReidTrackerImp::releaseTrackLock(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_REID_TRACKER))
        << "Track: end track, fid: " << frame_id;
    bool success = sm_->set(frame_id, StateMap::TRC_ED);
    if (success) {
      sm_->clearBadStates();
    }
    return success;
  }
  return false;
}

std::vector<OutputCharact> ReidTrackerImp::trackWithoutLock(
    const uint64_t frame_id, std::vector<InputCharact>& input_characts,
    const bool is_detection, const bool is_normalized) {
  if (mode_ & MODE_MULTIDETS) {
    // Track for the missing frames between (last_tracked_id, frame_id)
    int last_tracked_id = sm_->getLastTrackedId();
    if (mode_ & MODE_AUTOPATCH) {
      while (last_tracked_id != -1 && (int)frame_id > ++last_tracked_id) {
        std::vector<InputCharact> empty_charact;
        auto undet_track =
            ftd_->Update(last_tracked_id, false, is_normalized, empty_charact);
        LOG_IF(INFO, ENV_PARAM(DEBUG_REID_TRACKER))
            << "do undet_track for " << last_tracked_id
            << " frame_id: " << frame_id;
        undet_tracks_->push(
            std::make_pair<uint64_t, std::vector<OutputCharact>>(
                last_tracked_id, std::move(undet_track)));
      }
    }
    return ftd_->Update(frame_id, is_detection, is_normalized, input_characts);
  }
  return std::vector<OutputCharact>();
};

std::vector<OutputCharact> ReidTrackerImp::trackWithLock(
    const uint64_t frame_id, std::vector<InputCharact>& input_characts,
    const bool is_detection, const bool is_normalized) {
  std::vector<OutputCharact> det_track;
  if (mode_ & MODE_MULTIDETS) {
    setTrackLock(frame_id);
    det_track =
        trackWithoutLock(frame_id, input_characts, is_detection, is_normalized);
    releaseTrackLock(frame_id);
  }
  return det_track;
};

std::vector<OutputCharact> ReidTrackerImp::outputUndetTracks(
    uint64_t frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    if (undet_tracks_->size() != 0) {
      while (undet_tracks_->front()->first <= frame_id) {
        auto result = undet_tracks_->pop();
        if (result->first == frame_id) {
          return result->second;
        }
      }
    }
    DLOG(WARNING) << "no results found for frame_id: " << frame_id;
  }
  return std::vector<OutputCharact>();
};

void ReidTrackerImp::printUndetTracks() {
  if (mode_ & MODE_MULTIDETS) {
    if (undet_tracks_->size() != 0) {
      while (undet_tracks_->front()) {
        auto result = undet_tracks_->pop();
        LOG_IF(INFO, ENV_PARAM(DEBUG_REID_TRACKER))
            << "UndetTracks: " << result->first;
      }
      return;
    }
    DLOG(WARNING) << "no results found";
  }
  return;
};

void ReidTrackerImp::printState() {
  if (mode_ & MODE_MULTIDETS) {
    sm_->print();
  }
}

}  // namespace ai
}  // namespace vitis
