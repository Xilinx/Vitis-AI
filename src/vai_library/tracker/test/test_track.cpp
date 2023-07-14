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
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <tuple>
#include <vector>
#include <xilinx/ai/tracker.hpp>

#include "../src/common.hpp"

using namespace cv;
using namespace xilinx::ai;
using namespace std;
using namespace xilinx;
using namespace std::chrono;

/*
test_1 is example for simple single track mode.
just use track() to do the track logic.
And if you want to patch the frame, you can use patchFrame().
For example, you have original frameid of 1 3 5 7 9 11,
and the frame data you actually get is 1 3 7 9 11, that means, 5 is lost,
you can patch this frame by patchFrame(5).
You can also patch frame automatically without explicitly call the patchFrame(),
just by using the MODE_AUTOPATCH when calling create().
But in this mode, the frame id is assumed to be consequently.
that is, if your original frame is 1 3 5 7 9 11, and you lost the 5,
the patched frame is 2 4 5 6 8 10, not only the "5".


For multi-threads mode, there are 2 solutions.
one solution is: for multi-threads detection, each detecion thread has its own
output queue. (of course, each queue is sorted by frame id) tracker can pick the
smallest frame id from these queue, then remove it by pop. then repeat these
actions.  if one queue is empty, tracker has to wait for it. For example, there
are 2 detection threads det1 & det2. assume output queue1 has frame 1 3 5 7 9,
and output queue2 has frame 2 4 6 8 10, tracker pick 1 from queue1, then pick 2
from queue2, ....... so the sequence tracker picked is 1 2 3 4 5 6 7 8 9 10.

another solution is: you needn't create your own queues for each threads.
Please refer test_2() for an example how to use track in multi-threads
environment.
*/

int test_1();
int test_2();

std::vector<xilinx::ai::Tracker::InputCharact> input_characts;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  if (test_1()) return -1;
  if (test_2()) return -1;
  return 0;
}

int test_1() {
  /* detect bbox of each frame. format:
      bbox of face-0 of frame-1
      bbox of face-1 of frame-1
      bbox of face-0 of frame-2
      bbox of face-1 of frame-2
      ......
  */
  cv::Rect2f vbboxd[]{cv::Rect2f(0.128125, 0.39722222, 0.1, 0.21944444),
                      cv::Rect2f(0.34531251, 0.55555558, 0.09375, 0.21944444),
                      cv::Rect2f(0.12656251, 0.39444444, 0.1015625, 0.22222222),
                      cv::Rect2f(0.34531251, 0.55555558, 0.09375, 0.21666667),
                      cv::Rect2f(0.1296875, 0.39444444, 0.1, 0.21944444),
                      cv::Rect2f(0.34531251, 0.55555558, 0.0921875, 0.21944444),
                      cv::Rect2f(0.128125, 0.39444444, 0.1015625, 0.21944444),
                      cv::Rect2f(0.34531251, 0.55555558, 0.09375, 0.22222222)};

  /* real result of track bbox. format:
      bbox of face-0 of frame-3
      bbox of face-1 of frame-3
      bbox of face-0 of frame-4
      bbox of face-1 of frame-4
     because frame 1&2 has no track result.
  */
  cv::Rect2f vbboxt[]{
      cv::Rect2f(0.12916669, 0.39444441, 0.10000001, 0.21944445),
      cv::Rect2f(0.34544277, 0.55509269, 0.092187494, 0.21944445),
      cv::Rect2f(0.12843755, 0.39402795, 0.1015625, 0.21944439),
      cv::Rect2f(0.34500006, 0.555, 0.09375, 0.22222221)};

  auto GAP =
      0.0001;  // if diffence of bbox is smaller than this, we think it equal

  // create tracker
  std::shared_ptr<xilinx::ai::Tracker> tracker = xilinx::ai::Tracker::create();

  auto j = 0;
  for (int i = 0; i < 4; i++) {
    input_characts.clear();
    // prepare the detection information
    input_characts.emplace_back(vbboxd[i * 2], 0.99, -1, 1);
    input_characts.emplace_back(vbboxd[i * 2 + 1], 0.99, -1, 2);
    // call the track
    std::shared_ptr<std::vector<xilinx::ai::Tracker::OutputCharact>>
        track_results =
            std::make_shared<std::vector<xilinx::ai::Tracker::OutputCharact>>(
                tracker->track(i + 1, input_characts, true, true));
    // analysis track result
    for (auto &c : *track_results) {
      auto bbox = std::get<1>(c);

#if 0
        auto gid = std::get<0>(c);
        auto lid = std::get<4>(c);
        std::cout << "Track result for " << i << "th: gid: " << gid << " lid: " << lid
                  << " " << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << std::endl;
        std::cout << "diff: " << abs(vbboxt[j].x-bbox.x) << " " << abs(vbboxt[j].y-bbox.y) << " "
                  << abs(vbboxt[j].width-bbox.width) << " " << abs(vbboxt[j].height-bbox.height) << std::endl;
#endif

      // if track result is differ than the expected result, alert.
      if (abs(vbboxt[j].x - bbox.x) >= GAP ||
          abs(vbboxt[j].y - bbox.y) >= GAP ||
          abs(vbboxt[j].width - bbox.width) >= GAP ||
          abs(vbboxt[j].height - bbox.height) >= GAP) {
        std::cout << "Bad for i:" << i << " j:" << j << std::endl;
        return -1;
      }
      j++;
    }
  }
  // an example for how to patch frame. assume the 5th frame is missing and we
  // need patch it.
  std::shared_ptr<std::vector<xilinx::ai::Tracker::OutputCharact>>
      track_results =
          std::make_shared<std::vector<xilinx::ai::Tracker::OutputCharact>>(
              tracker->patchFrame(5));
  for (auto &c : *track_results) {
    auto bbox = std::get<1>(c);
    auto gid = std::get<0>(c);
    auto lid = std::get<4>(c);
    std::cout << "Track result of patchFrame gid: " << gid << " lid: " << lid
              << " " << bbox.x << " " << bbox.y << " " << bbox.width << " "
              << bbox.height << std::endl;
  }
  input_characts.clear();
  return 0;
}

void detect(std::shared_ptr<Tracker> tk, const string &test_input) {
  ifstream ifs(test_input);
  CHECK(ifs) << " Fail to open file: " << test_input;
  int fid;
  int delay = 10;
  while (ifs >> fid) {
    LOG(INFO) << "fid:\t" << fid;

    LOG(INFO) << "*** addDetStart(" << fid << ")";
    bool success1 = tk->addDetStart(fid);
    tk->printState();
    if (!success1) {
      LOG(WARNING) << "addDetStart Failed: " << fid;
      continue;
    }
    // here, simulate the detect() operation which consume time.
    std::this_thread::sleep_for(milliseconds(delay));

    LOG(INFO) << "*** setDetEnd(" << fid << ")";
    tk->setDetEnd(fid);
    tk->printState();

    LOG(INFO) << "*** setTrackLock(" << fid << ")";
    bool success2 = tk->setTrackLock(fid);
    tk->printState();

    if (success2) {
      tk->trackWithoutLock(fid, input_characts);

      LOG(INFO) << "*** releaseTrackLock(" << fid << ")";
      tk->releaseTrackLock(fid);
      tk->printState();
    } else {
      LOG(INFO) << "*** setTrackLock(" << fid << ") failed!";
    }
  }
  ifs.close();
}

int test_2() {
  FLAGS_alsologtostderr = true;
  LOG(INFO) << "TEST START";

  // create tracker
  std::shared_ptr<xilinx::ai::Tracker> tk =
      xilinx::ai::Tracker::create(xilinx::ai::Tracker::MODE_MULTIDETS);
  std::thread th1(detect, tk, "../test/test_input/test_input1.txt");
  std::thread th2(detect, tk, "../test/test_input/test_input2.txt");

  th1.join();
  th2.join();
  tk->printUndetTracks();

  LOG(INFO) << "TEST END";
  input_characts.clear();
  return 0;
}
