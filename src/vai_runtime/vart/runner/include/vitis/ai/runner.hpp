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
#pragma once

#include <cstdint>
#include <utility>
namespace vitis {
namespace ai {

template <typename InputType, typename OutputType = InputType>
class Runner {
 public:
  virtual ~Runner() = default;

  /**
   * @brief execute_async
   *
   * @param in inputs with a customized type
   *
   * @param out outputs with a customized type
   *
   * @return pair<jodid, status> status 0 for exit successfully, others for
   * customized warnings or errors
   *
   */
  virtual std::pair<std::uint32_t, int> execute_async(InputType input,
                                                      OutputType output) = 0;

  /**
   * @brief wait
   *
   * @details modes: 1. Blocking wait for specific ID. 2. Non-blocking wait for
   * specific ID. 3. Blocking wait for any ID. 4. Non-blocking wait for any ID
   *
   * @param jobid job id, neg for any id, others for specific job id
   *
   * @param timeout timeout, neg for block for ever, 0 for non-block, pos for
   * block with a limitation(ms).
   *
   * @return status 0 for exit successfully, others for customized warnings or
   * errors
   *
   */
  virtual int wait(int jobid, int timeout = -1) = 0;
};
}  // namespace ai
}  // namespace vitis
