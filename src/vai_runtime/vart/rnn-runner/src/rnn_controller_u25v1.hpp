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

#pragma once

#include "xrnn_controller.hpp"

namespace vart {
namespace xrnn {

class RnnControllerU25 : public XrnnController {
 public:
  explicit RnnControllerU25(size_t device_core_id,
                            std::unique_ptr<xir::XrtCu>&& xrt_cu);
  ~RnnControllerU25() override;
  RnnControllerU25(const RnnControllerU25& other) = delete;
  RnnControllerU25& operator=(const RnnControllerU25& rhs) = delete;

  void init(char* ddr, uint64_t size) override;
  void update(int frame, ModelConfig* mc, uint32_t* p_ddr,
              size_t size) override;
  void run(char* in, uint64_t isize, char* out, uint64_t osize, int batch,
           int frame, int thread_index) override;
  std::string get_board_name() override;
  int get_batch_size() override;

 private:
  std::vector<uint32_t> get_reg_data(int frame, int thread_index);
  size_t get_base_addr(unsigned batch_num);
  std::string get_addr_name();

  const size_t idx_ = 0;
  std::unique_ptr<xir::XrtCu> xrt_cu_;
  std::shared_ptr<xir::DeviceMemory> memory_;

  int batch_ = 1;
  int nlayers_ = 1;

  static std::mutex mutex_;
};

}  // namespace xrnn
}  // namespace vart
