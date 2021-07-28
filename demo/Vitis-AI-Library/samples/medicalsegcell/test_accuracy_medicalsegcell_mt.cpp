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
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/medicalsegcell.hpp>
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

struct MedicalSegcellAcc : public AccThread {
  MedicalSegcellAcc(std::string output_path)
      : AccThread(), resultpath(output_path) {
    dpu_result.frame_id = -1;
  }

  virtual ~MedicalSegcellAcc() {}

  static std::shared_ptr<MedicalSegcellAcc> instance(std::string output_file) {
    static std::weak_ptr<MedicalSegcellAcc> the_instance;
    std::shared_ptr<MedicalSegcellAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<MedicalSegcellAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (MedicalSegcellResult*)dpu_result.result_ptr.get();

    cv::Mat img_save;
    std::string fname(resultpath + "/pred_" +
                      split(dpu_result.single_name, ".")[0] + ".png");
    cv::resize(result->segmentation, img_save,
               cv::Size(dpu_result.w, dpu_result.h), 0, 0, cv::INTER_LINEAR);
    cv::imwrite(fname, img_save);
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000)))
      process_result(dpu_result);
    return 0;
  }

  DpuResultInfo dpu_result;
  std::string resultpath;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  std::string model_name(argv[1]);

  auto ret = mkdir(argv[3], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[3] << std::endl;
    return -1;
  }
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] { return vitis::ai::MedicalSegcell::create(model_name + "_acc"); },
      vitis::ai::MedicalSegcellAcc::instance(argv[3]), 2);
}
