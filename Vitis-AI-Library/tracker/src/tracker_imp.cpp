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
#include "tracker_imp.hpp"
#include "ftd/ftd_structure.hpp"
#include "state_map.hpp"
#include <thread>

namespace xilinx {
namespace ai {

TrackerImp::TrackerImp(uint64_t mode, const SpecifiedCfg &cfg) {
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

TrackerImp::~TrackerImp() {
  delete ftd_;
  if (mode_ & MODE_MULTIDETS) {
    delete sm_;
    delete undet_tracks_;
  }
}

std::vector<int> TrackerImp::GetRemoveID() { return ftd_->GetRemoveID(); }

void TrackerImp::clear() {
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

std::vector<OutputCharact> TrackerImp::patchFrame(const uint64_t frame_id) {
  std::vector<InputCharact> empty_charact;
  std::vector<OutputCharact> det_track;
  if (!(mode_ & MODE_MULTIDETS)) {
    det_track = ftd_->Update(frame_id, false, true, empty_charact);
  }
  return det_track;
}

std::vector<OutputCharact>
TrackerImp::track(const uint64_t frame_id,
                  std::vector<InputCharact> &input_characts,
                  const bool is_detection, const bool is_normalized) {
  std::vector<OutputCharact> det_track;
  if (mode_ & MODE_MULTIDETS) {
    return det_track;
  }

  if (lastframe_id && (frame_id > lastframe_id + 1) &&
      (mode_ & MODE_AUTOPATCH)) {
    for (uint64_t i = 1; i < frame_id - lastframe_id; i++) {
      patchFrame(lastframe_id + i);
    }
  }
  det_track =
      ftd_->Update(frame_id, is_detection, is_normalized, input_characts);

  lastframe_id = frame_id;
  return det_track;
}

bool TrackerImp::addDetStart(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    return sm_->add(frame_id, StateMap::DET_ST);
  }
  return false;
}

bool TrackerImp::setDetEnd(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    return sm_->set(frame_id, StateMap::DET_ED);
  }
  return false;
}

bool TrackerImp::setDetTimeout(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    return sm_->set(frame_id, StateMap::DET_TO);
  }
  return false;
}

bool TrackerImp::setTrackLock(int frame_id, int timeout, int interval) {
  if (mode_ & MODE_MULTIDETS) {
    int time = 0;
    while (sm_->getCurId() != frame_id) {
      DLOG(INFO) << "Track: wait for mutex, fid: " << frame_id
                 << " cur_id: " << sm_->getCurId();
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      if ((time += interval) > timeout) {
        DLOG(INFO) << "Track: setTrackLock timeout.";
        return false;
      }
    }
    DLOG(INFO) << "Track: start track, fid: " << frame_id;
    sm_->updateLastTrackedId(frame_id);
    return sm_->set(frame_id, StateMap::TRC_ST);
  }
  return false;
}

bool TrackerImp::releaseTrackLock(int frame_id) {
  if (mode_ & MODE_MULTIDETS) {
    DLOG(INFO) << "Track: end track, fid: " << frame_id;
    bool success = sm_->set(frame_id, StateMap::TRC_ED);
    if (success) {
      sm_->clearBadStates();
    }
    return success;
  }
  return false;
}

std::vector<OutputCharact> TrackerImp::trackWithoutLock(
    const uint64_t frame_id, std::vector<InputCharact> &input_characts,
    const bool is_detection, const bool is_normalized) {
  if (mode_ & MODE_MULTIDETS) {
    // Track for the missing frames between (last_tracked_id, frame_id)
    int last_tracked_id = sm_->getLastTrackedId();
    if (mode_ & MODE_AUTOPATCH) {
      while (last_tracked_id != -1 && (int)frame_id > ++last_tracked_id) {
        std::vector<InputCharact> empty_charact;
        auto undet_track =
            ftd_->Update(last_tracked_id, false, is_normalized, empty_charact);
        DLOG(INFO) << "do undet_track for " << last_tracked_id
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

std::vector<OutputCharact>
TrackerImp::trackWithLock(const uint64_t frame_id,
                          std::vector<InputCharact> &input_characts,
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

std::vector<OutputCharact> TrackerImp::outputUndetTracks(uint64_t frame_id) {
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

void TrackerImp::printUndetTracks() {
  if (mode_ & MODE_MULTIDETS) {
    if (undet_tracks_->size() != 0) {
      while (undet_tracks_->front()) {
        auto result = undet_tracks_->pop();
        LOG(INFO) << "UndetTracks: " << result->first;
      }
      return;
    }
    DLOG(WARNING) << "no results found";
  }
  return;
};

void TrackerImp::printState() {
  if (mode_ & MODE_MULTIDETS) {
    sm_->print();
  }
}

} // namespace ai
} // namespace xilinx
