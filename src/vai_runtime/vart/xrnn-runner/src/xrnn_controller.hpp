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
#include "xrnn_hw.hpp"
//#include "xrnn_xrt_cu.hpp"
#include "xrt_cu.hpp"
#include "xir/buffer_object.hpp"
#include "xir/device_memory.hpp"

#include "model_config.hpp"

#include <string>
#include <map>
#include <mutex>
#include <uuid/uuid.h>

#include <xclbin.h>
#include <xrt.h>
#include <ert.h>
#include <xclhal2.h>

namespace vart {
namespace xrnn {


enum MODEL_TYPE {SENTIMENT, SATISFACTION, OPENIE, UNKOWN=255};


class XrnnController {
public:
  explicit XrnnController(size_t device_core_id,
                          const std::string& model_type,
                          std::unique_ptr<xir::XrtCu>&& xrt_cu);
  virtual ~XrnnController();
  XrnnController(const XrnnController& other) = delete;
  XrnnController& operator=(const XrnnController& rhs) = delete;

public:
  void init(char *ddr, uint64_t size);
  void update(int frame, ModelConfig *mc, uint32_t* p_ddr, size_t size);
  void run(char* in, uint64_t isize,
           char* out, uint64_t osize,
           int batch, int frame, int thread_index);
  std::string get_board_name();
  int get_batch_size();
  
private:
  MODEL_TYPE get_model_type(const std::string& model_name);
  std::string get_model_name(MODEL_TYPE model_type);
  std::vector<uint32_t> get_reg_data(int frame, int thread_index);
  size_t get_base_addr(unsigned batch_num);
  std::string get_addr_name();

private:
  const size_t idx_ = 0;
  std::unique_ptr<xir::XrtCu> xrt_cu_;
  std::shared_ptr<xir::DeviceMemory> memory_;

  MODEL_TYPE model_type_;
  int batch_ = 1;

  static std::mutex mutex_;
};

} // namespace xrnn
} // namespace vart
