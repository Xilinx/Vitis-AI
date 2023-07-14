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

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>

namespace xilinx {
namespace ai {

/**
 * @brief Base class of object tracker
 *
 */

class Tracker {
 public:
  /// InputCharact: bbox, score, label, local_id
  typedef std::tuple<cv::Rect_<float>, float, int, int> InputCharact;
  /// OutputCharact: gid, bbox, score, label, local_id
  typedef std::tuple<uint64_t, cv::Rect_<float>, float, int, int> OutputCharact;
  typedef std::tuple<std::array<int, 4>, std::array<int, 3>> SpecifiedCfg;

  /**
   *@enum TRACKER mode
   *
   *@brief MODE_MULTIDETS: if use in multi-detector mode. mainly use in multi
   *threads environment MODE_AUTOPATCH: if auto patch the missing frame.
   *
   */
  enum TRACKER_MODE { MODE_MULTIDETS = 0x01, MODE_AUTOPATCH = 0x10 };

  /**
   * @brief Constructor function for creating Tracker instance.
   *
   * @param mode. see enum TRACKER mode defination
   * @param cfg  Configure value. Please use default value.
   */

  static std::shared_ptr<Tracker> create(
      uint64_t mode = 0,
      const SpecifiedCfg &cfg = SpecifiedCfg(std::array<int, 4>({6, 6, 1, 1}),
                                             std::array<int, 3>({3, 2, 1})));
  Tracker();
  Tracker(const Tracker &) = delete;
  Tracker &operator=(const Tracker &) = delete;
  /// Destructor
  virtual ~Tracker();

  /**
   * @brief Function to get the ids that are removed in this frame.
   */
  virtual std::vector<int> GetRemoveID() = 0;

  /**
   * @brief Function to clear the state of Tracker.
   */
  virtual void clear() = 0;

  /**
   * @brief Function to patch the missing frame.
   *
   * @parama frame_id the frame_id of the missing frame.
   *
   */
  virtual std::vector<OutputCharact> patchFrame(const uint64_t frame_id) = 0;

  /**
   * @brief Function to do the track : only use in !MODE_MULTIDETS mode.
   *
   * @param frame_id The frame_id of the frame to track
   * @param input_characts The input data for tracking, come from frame
   * detection.
   * @param is_detection If this frame is detection frame ( or patch frame ).
   * @param is_normalized If the bbox of input data is normalized.
   *
   * @return the result of the track. As the tuple defined, it returns
   *         the global track id(gid), bbox, score, label, local track
   * id(local_id) if local_id is <0, it means this id is not detected and
   * patched by Tracker.
   */
  virtual std::vector<OutputCharact> track(
      const uint64_t frame_id, std::vector<InputCharact> &input_characts,
      const bool is_detection, const bool is_normalized) = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *    Notify the tracker that detection starts, only the thread with minimal
   *      detecting frame_id can lock the tracker
   *
   * @parama frame_id the frame_id.
   */
  virtual bool addDetStart(int frame_id) = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *    Notify the tracker that detection ends
   *
   * @parama frame_id the frame_id.
   */
  virtual bool setDetEnd(int frame_id) = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *    set the detect time out
   *
   * @parama frame_id the frame_id.
   */
  virtual bool setDetTimeout(int frame_id) = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *    Add lock for tracking
   *
   * @param frame_id
   * @param timeout the time to stop getting the track lock. uint: ms
   * @param interval the time interval to try getting the track lock. uint: ms
   *
   * @return false if fail to get the track lock
   */
  virtual bool setTrackLock(int frame_id, int timeout = 1000,
                            int interval = 1) = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *    release lock for tracking
   *
   * @parama frame_id the frame_id.
   */
  virtual bool releaseTrackLock(int frame_id) = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *  Track without lock, need to handle the lock mannually in multi-thread
   *  tracking
   *
   * @param frame_id
   * @param input_characts the vector of input charact{bbox, score,
   * label,local_id}
   * @is_detection.  this is detection frame or patch frame
   * @is_normalized.  if use normilized rect
   * @return the output characts{global_id, tracked_bbox, score, label,
   * local_id}
   */
  virtual std::vector<OutputCharact> trackWithoutLock(
      const uint64_t frame_id, std::vector<InputCharact> &input_characts,
      const bool is_detection = true, const bool is_normalized = true) = 0;
  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *  Track with lock, will be blocked if other threads are tracking previous
   * frame
   *
   * @param frame_id
   * @param input_characts the vector of input charact{bbox, score,
   * label,local_id}
   * @is_detection.  this is detection frame or patch frame
   * @is_normalized.  if use normilized rect
   * @return the output characts{global_id, tracked_bbox, score, label,
   * local_id}
   */
  virtual std::vector<OutputCharact> trackWithLock(
      const uint64_t frame_id, std::vector<InputCharact> &input_characts,
      const bool is_detection = true, const bool is_normalized = true) = 0;
  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   * Output the track results of un-detection frames, only the recent 100
   * undet-frames will be kept
   *
   * @param frame_id
   * @return the output charact{global_id, tracked_bbox, score, label,
   * local_id}, will return empty vector if the frame has empty track results or
   * frame_id is wrong
   */
  virtual std::vector<OutputCharact> outputUndetTracks(uint64_t frame_id) = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *   print state
   */
  virtual void printState() = 0;

  /**
   * @brief Function : only use in MODE_MULTIDETS mode.
   *   print all undetected tracks
   *
   */
  virtual void printUndetTracks() = 0;
};

}  // namespace ai
}  // namespace xilinx
