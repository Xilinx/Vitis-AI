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
#include <ert.h>
#include <uuid/uuid.h>
#include <xclbin.h>
#include <xclhal2.h>
#include <xrt.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "model_config.hpp"
#include "rnn_model_parser.hpp"
#include "vitis/ai/weak.hpp"
#include "xir/buffer_object.hpp"
#include "xir/device_memory.hpp"
#include "xir/xrt_device_handle.hpp"
#include "xrt_cu.hpp"

DEF_ENV_PARAM(DEBUG_XRNN_CONTROLLER, "0");
DEF_ENV_PARAM(XRNN_MAX_THREAD, "16");
DEF_ENV_PARAM_2(XRNN_KERNEL_NAME, "xrnn", std::string);
DEF_ENV_PARAM_2(XRNN_INSTANCE_NAME, "xrnn_1", std::string);

namespace vart {
namespace xrnn {

enum MODEL_TYPE { SENTIMENT, SATISFACTION, OPENIE, UNKOWN = 255 };

class XrnnController {
 public:
  XrnnController() = default;
  virtual ~XrnnController() = default;
  XrnnController(const XrnnController& other) = delete;
  XrnnController& operator=(const XrnnController& rhs) = delete;

  virtual void init(char* ddr, uint64_t size) = 0;
  // TODO : abidk : update and run should be refactored
  virtual void update(int frame, ModelConfig* mc, uint32_t* p_ddr,
                      size_t size) = 0;
  virtual void run(char* in, uint64_t isize, char* out, uint64_t osize,
                   int batch, int frame, int thread_index) = 0;
  virtual std::string get_board_name() = 0;
  virtual int get_batch_size() = 0;

  virtual void set_model_config(const ModelConfig* model_config) {
    model_config_ = model_config;
  };
  virtual const ModelConfig* get_model_config() const { return model_config_; };

 protected:
  const ModelConfig* model_config_;
};

using GenFunc =
    std::function<std::unique_ptr<XrnnController>(unsigned int, std::string)>;

class RnnControllerCreate {
  static std::map<std::string, GenFunc>& getRegistry();

 public:
  // static std::map<std::string, GenFunc> registry;
  static void Register(std::string device_name, GenFunc func);
  static std::unique_ptr<XrnnController> Create(std::string device_name,
                                                unsigned int device_core_id,
                                                std::string device);

  static std::unique_ptr<XrnnController> Create(RNNModelParser* parser);
};

}  // namespace xrnn
}  // namespace vart

#define REGISTER_RNN_CONTROLLER(key, value)                                    \
  struct Register##value {                                                     \
    Register##value() {                                                        \
      vart::xrnn::RnnControllerCreate::Register(                               \
          key, [](unsigned int device_core_id, const std::string& device) {    \
            return make_unique<value>(device_core_id,                          \
                                      std::make_unique<xir::XrtCu>(device));   \
          });                                                                  \
    }                                                                          \
  };                                                                           \
  static Register##value dummyReg##value;
