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
#include <glog/logging.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/nnpp/segmentation.hpp>
#include <vitis/ai/polypsegmentation.hpp>
extern int g_last_frame_id;

namespace vitis {
namespace ai {

static std::vector<std::string> split(const std::string& s,
                                      const std::string& delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

struct PolypSegmentationAcc : public AccThread {
  PolypSegmentationAcc(std::string output_file) : AccThread() {
    dpu_result.frame_id = -1;
    of = output_file + "/Kvasir/";
    auto ret = mkdir(of.c_str(), 0777);
    if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
      std::cout << "error occured when mkdir " << of << std::endl;
    }
  }

  virtual ~PolypSegmentationAcc() {}

  static std::shared_ptr<PolypSegmentationAcc> instance(
      std::string output_file) {
    static std::weak_ptr<PolypSegmentationAcc> the_instance;
    std::shared_ptr<PolypSegmentationAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<PolypSegmentationAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (SegmentationResult*)dpu_result.result_ptr.get();
    imwrite(of + "/" + split(dpu_result.single_name, ".")[0] + ".png",
            result->segmentation);
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000)))
      process_result(dpu_result);
    return 0;
  }

  DpuResultInfo dpu_result;
  std::string of;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [&] { return vitis::ai::PolypSegmentation::create(argv[1]); },
      vitis::ai::PolypSegmentationAcc::instance(argv[3]), 2);
}
