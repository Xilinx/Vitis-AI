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
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vitis/ai/demo_accuracy.hpp>

#include "vitis/ai/nnpp/platenum.hpp"
#include "vitis/ai/platenum.hpp"

extern int g_last_frame_id;
std::string RESULT_FILE_PATH = "accuracy_result_plate/";
extern int enable_platenum_acc;
std::string model_name;

namespace vitis {
namespace ai {

static std::string get_single_name(const std::string& line) {
  std::size_t found = line.rfind('/');
  if (found != std::string::npos) {
    return line.substr(found + 1);
  }
  return line;
}

static std::string get_single_name_no_suffix(const std::string& line) {
  auto single_name = get_single_name(line);
  auto found = single_name.rfind('.');
  if (found != std::string::npos) {
    single_name = single_name.substr(0, found);
  }
  return single_name;
}

struct PlateNumAccThread : public AccThread {
  PlateNumAccThread(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~PlateNumAccThread() { of.close(); }

  static std::shared_ptr<PlateNumAccThread> instance(std::string output_file) {
    static std::weak_ptr<PlateNumAccThread> the_instance;
    std::shared_ptr<PlateNumAccThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<PlateNumAccThread>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (PlateNumResult*)dpu_result.result_ptr.get();
    std::ofstream out(RESULT_FILE_PATH + "/" +
                      get_single_name_no_suffix(dpu_result.single_name) +
                      ".txt");
    out << result->plate_number;
    std::cout << result->plate_number << std::endl;
    out.close();
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
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Please input a model name as the first param!" << std::endl;
    std::cout << "Please input your image path list as the second param!"
              << std::endl;
    std::cout << "The third param is the output dir!" << std::endl;
    std::cout << "The fourth param is thread nums, eg: '-t 4', default single "
                 "thread if not filled "
              << std::endl;
  }
  enable_platenum_acc = 1;
  model_name = argv[1];
  RESULT_FILE_PATH = argv[3];
  CHECK_EQ(system(std::string("rm -rf " + std::string(argv[3])).c_str()), 0);
  CHECK_EQ(system(std::string("mkdir -p " + std::string(argv[3])).c_str()), 0);

  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [&] { return vitis::ai::PlateNum::create(model_name); },
      vitis::ai::PlateNumAccThread::instance(argv[3]), 2);
}
