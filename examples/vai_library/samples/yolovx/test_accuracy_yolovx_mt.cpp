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
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/nnpp/yolovx.hpp>
#include <vitis/ai/yolovx.hpp>

using namespace std;
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
struct YOLOvXAcc : public AccThread {
  YOLOvXAcc(std::string output_path_)
      : AccThread(), output_path(output_path_ + "/") {
    dpu_result.frame_id = -1;
  }

  virtual ~YOLOvXAcc() {}

  static std::shared_ptr<YOLOvXAcc> instance(std::string output_file) {
    static std::weak_ptr<YOLOvXAcc> the_instance;
    std::shared_ptr<YOLOvXAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<YOLOvXAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (YOLOvXResult*)dpu_result.result_ptr.get();
    std::ofstream of(
        output_path + split(dpu_result.single_name, ".")[0] + ".txt",
        std::ofstream::out);
    for (auto& res : result->bboxes) {
      auto box = res.box;
      float confidence = res.score;
      of << "Car -10 -10 -10 " << std::fixed << std::setprecision(2) << box[0]
         << " " << box[1] << " " << box[2] << " " << box[3]
         << " -1000 -1000 -1000 -1000 -1000 -1000 -1000 "
         << std::setprecision(6) << confidence << endl;
    }
    of.close();
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000))) {
      process_result(dpu_result);
    }
    return 0;
  }
  std::string output_path;
  DpuResultInfo dpu_result;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] { return vitis::ai::YOLOvX::create(string(argv[1]) + "_acc"); },
      vitis::ai::YOLOvXAcc::instance(argv[3]), 2);
}
