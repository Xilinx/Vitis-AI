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
#pragma once
#include <array>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "../include/xilinx/ai/tracker.hpp"
#include "common.hpp"
#include "ftd/ftd_structure.hpp"
#include "ring_queue.hpp"
#include "state_map.hpp"

namespace xilinx {
namespace ai {

class TrackerImp : public Tracker {
 public:
  TrackerImp(uint64_t, const SpecifiedCfg &);
  TrackerImp(const TrackerImp &) = delete;
  TrackerImp &operator=(const TrackerImp &) = delete;
  /// Destructor
  virtual ~TrackerImp();

  /**
   * @brief Function to get the ids that are removed in this frame.
   */
  virtual std::vector<int> GetRemoveID() override;

  /**
   * @brief Function to clear the state of Tracker.
   */
  virtual void clear() override;

  /**
   * @brief Function to patch the missing frame.
   *
   * @parama frame_id the frame_id of the missing frame.
   *
   */
  virtual std::vector<OutputCharact> patchFrame(
      const uint64_t frame_id) override;

  virtual std::vector<OutputCharact> track(
      const uint64_t frame_id, std::vector<InputCharact> &input_characts,
      const bool is_detection, const bool is_normalized) override;

  virtual bool addDetStart(int frame_id) override;
  virtual bool setDetEnd(int frame_id) override;
  virtual bool setDetTimeout(int frame_id) override;
  virtual bool setTrackLock(int frame_id, int timeout = 1000,
                            int interval = 1) override;
  virtual bool releaseTrackLock(int frame_id) override;
  virtual std::vector<OutputCharact> trackWithoutLock(
      const uint64_t frame_id, std::vector<InputCharact> &input_characts,
      const bool is_detection = true, const bool is_normalized = true) override;
  virtual std::vector<OutputCharact> trackWithLock(
      const uint64_t frame_id, std::vector<InputCharact> &input_characts,
      const bool is_detection = true, const bool is_normalized = true);
  virtual std::vector<OutputCharact> outputUndetTracks(
      uint64_t frame_id) override;
  virtual void printState() override;
  virtual void printUndetTracks() override;

 private:
  FTD_Structure *ftd_ = NULL;
  StateMap *sm_ = NULL;
  uint64_t mode_ = 0;
  uint64_t lastframe_id = 0;
  RingQueue<std::pair<uint64_t, std::vector<OutputCharact>>> *undet_tracks_ =
      NULL;
};

}  // namespace ai
}  // namespace xilinx
