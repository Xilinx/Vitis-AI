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

#include "rnn_controller_u50lv.hpp"

namespace vart {
namespace xrnn {

static std::vector<size_t> U25_DDR_BASE_CU0{
    0x800000000,
};

static std::vector<size_t> U25_DDR_INIT_ADDR_CU0{
    0x800000000,
};

class RnnControllerU25v2 : public RnnControllerU50LV {
 public:
  explicit RnnControllerU25v2(size_t device_core_id,
                              std::unique_ptr<xir::XrtCu>&& xrt_cu);
  ~RnnControllerU25v2() override;
  RnnControllerU25v2(const RnnControllerU25v2& other) = delete;
  RnnControllerU25v2& operator=(const RnnControllerU25v2& rhs) = delete;

  std::string get_board_name() override;
  int get_batch_size() override;

 protected:
  virtual std::vector<uint32_t> get_reg_data(int frame,
                                             int thread_index) override;
  virtual size_t get_base_addr(unsigned batch_num) override;
  virtual const std::vector<size_t>& get_init_addr() override;

 private:
  using MapBaseAddr = std::map<const std::string, std::vector<size_t>*>;
  MapBaseAddr batch_addr_map{{"u25_cu0", &U25_DDR_BASE_CU0}};
  MapBaseAddr init_addr_map{{"u25_cu0", &U25_DDR_INIT_ADDR_CU0}};
};

}  // namespace xrnn
}  // namespace vart
