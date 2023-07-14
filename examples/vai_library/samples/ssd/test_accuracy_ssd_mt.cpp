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

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/ssd.hpp>
extern int g_last_frame_id;
extern int GLOBAL_ENABLE_C_SOFTMAX;

std::string model_name;

namespace vitis {
namespace ai {

static std::map<std::string, std::vector<std::string>> label_map{
    {"ssd_pedestrian_pruned_0_97", {"backgroud", "person"}}};

std::string get_single_name(const std::string& line) {
  std::size_t found = line.rfind('/');
  if (found != std::string::npos) {
    return line.substr(found + 1);
  }
  return line;
}

std::string get_single_name_no_suffix(const std::string& line) {
  auto single_name = get_single_name(line);
  auto found = single_name.rfind('.');
  if (found != std::string::npos) {
    single_name = single_name.substr(0, found);
  }
  return single_name;
}

struct SSDAcc : public AccThread {
  SSDAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~SSDAcc() { of.close(); }

  static std::shared_ptr<SSDAcc> instance(std::string output_file) {
    static std::weak_ptr<SSDAcc> the_instance;
    std::shared_ptr<SSDAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<SSDAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (SSDResult*)dpu_result.result_ptr.get();
    for (auto& it : result->bboxes) {
      if (label_map.count(model_name)) {
        of << get_single_name_no_suffix(dpu_result.single_name) << " "
           << label_map[model_name][it.label] << " " << it.score << " "
           << it.x * dpu_result.w << " " << it.y * dpu_result.h << " "
           << (it.x + it.width) * dpu_result.w << " "
           << (it.y + it.height) * dpu_result.h << std::endl;
      } else if ("mlperf_ssd_resnet34_tf" == model_name) {
        of << get_single_name(dpu_result.single_name) << " " << it.label << " "
           << it.score << " " << it.x << " " << it.y << " " << it.width << " "
           << it.height << std::endl;
      } else {
        of << get_single_name(dpu_result.single_name) << " " << it.label << " "
           << it.score << " " << it.x * dpu_result.w << " "
           << it.y * dpu_result.h << " " << (it.x + it.width) * dpu_result.w
           << " " << (it.y + it.height) * dpu_result.h << std::endl;
      }
    }
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (getQueue()->pop(dpu_result, std::chrono::milliseconds(5000))) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "[" << name() << "] process result id :" << dpu_result.frame_id
          << ", dpu queue size " << getQueue()->size();
      process_result(dpu_result);
    }
    return 0;
  }

  DpuResultInfo dpu_result;
  std::ofstream of;
};  // namespace ai

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  GLOBAL_ENABLE_C_SOFTMAX = 2;
  model_name = argv[1];
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [&] { return vitis::ai::SSD::create(model_name + "_acc"); },
      vitis::ai::SSDAcc::instance(argv[3]), 2);
}
