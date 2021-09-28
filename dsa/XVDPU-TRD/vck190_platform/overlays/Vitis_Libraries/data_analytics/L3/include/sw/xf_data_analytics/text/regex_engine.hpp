/*
 * Copyright 2020 Xilinx, Inc.
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
#ifndef _RE_ENGNINE_L3_
#define _RE_ENGNINE_L3_

#include <iostream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <queue>
#include <cstring>
#include "xf_data_analytics/text/reEngine_config.hpp"
#include "xf_data_analytics/text/helper.hpp"
#include "xf_utils_sw/logger.hpp"

namespace xf {
namespace data_analytics {
namespace text {
namespace re {

/**
 * @class RegexEngine regex_engine.hpp "xf_data_analytics/text/regex_engine.hpp"
 * @brief Offload regex match with FPGA.
 */
class RegexEngine {
   private:
    // variable for platform
    /// Error code
    cl_int err;
    /// Context
    cl_context ctx;
    /// Device ID
    cl_device_id dev_id;
    /// Command queue
    cl_command_queue cq;
    /// Program
    cl_program prg;
    /// Path to FPGA binary
    std::string xclbin_path;

    /// Kernel name
    const char* kernel_name = "reEngineKernel";

    /// Object for generating the configurations
    reConfig reCfg;

    const int kInstrDepth;
    const int kCharClassNum;
    const int kCaptureGrpNum;
    const int kMsgSize;
    const int kMaxSliceSize;
    const int kMaxSliceNum;

    /**
     * @brief Performs the actual matching process
     *
     * @param msg_buff Buffer contains messages line-by-line, pads white-space if not hit the boundary of 64-bit
     * @param offt_buff Buffer contains starting address of each line, aligned with 64-bit
     * @param len_buff Buffer contains length of each message, in byte
     * @param out_buff Buffer contains output results including match flag & corresponding start/end offset address for
     * each capturing group
     * @param cfg_buff Buffer contains configuration header, instructions, and bit-set map
     * @param total_lnm Total number of lines in current message block
     */
    ErrCode match_all(const uint64_t* msg_buff,
                      uint32_t* offt_buff,
                      uint16_t* len_buff,
                      uint32_t* out_buff,
                      const uint64_t* cfg_buff,
                      uint32_t total_lnm);

    /**
     * @brief Calculates the number of sections for current input message block
     *
     * @param len_buff Buffer contains length of each message, in byte
     * @param lnm Number of messages (lines) in current block
     * @param slice_nm Max number of lines of each section in current block
     * @param lnm_per_sec Number of lines in each section
     * @param pos_per_sec Starting address of each section within current block
     */
    uint32_t findSecNum(
        uint16_t* len_buff, uint32_t lnm, uint32_t* slice_nm, uint16_t* lnm_per_sec, uint32_t* pos_per_sec);

   public:
    /**
     * @brief Default constructor for loading and programming binary
     *
     * @param xclbin Path to FPGA binary.
     * @param dev_index The index of Xilinx OpenCL device.
     *
     * @param instr_depth Max number of instructions.
     * @param char_class_num Max number of character classes.
     * @param capture_grp_num Max number of capturing groups.
     *
     * @param msg_size Max size for each message, in number of bytes.
     * @parma max_slice_size Max message slice size, in number of bytes.
     * @param max_slice_num  Max message slice number.
     */
    RegexEngine(const std::string& xclbin,
                const int dev_index = 0,
                const int instr_depth = 4096,
                const int char_class_num = 128,
                const int capture_grp_num = 512,
                const int msg_size = 4096,
                const int max_slice_size = 5242880,
                const int max_slice_num = 256);
    /**
     * @brief Default de-constructor for releasing program/command-queue/contect
     */
    ~RegexEngine();
    /**
     * @brief Pre-compiles pattern and gives error code correspondingly
     *
     * @param pattern Input regular expression
     */
    ErrCode compile(std::string pattern);
    /**
     * @brief Gets the number of capturing groups for current pattern
     */
    uint32_t getCpgpNm() const;
    /**
     * @brief Prepares configurations for ``RegexEngineKernel``, and performs the actual matching process.
     *
     * @param total_lnm Total number of lines in current message block
     * @param msg_buff Buffer for saving messages line-by-line, pads white-space if not hit the boundary of 64-bit
     * @param offt_buff Buffer for saving starting address of each line, aligned with 64-bit
     * @param len_buff Buffer for saving length of each message, in byte
     * @param out_buff Buffer for saving output results including match flag & corresponding start/end offset
     * address for each capturing group
      */
    ErrCode match(
        uint32_t total_lnm, const uint64_t* msg_buff, uint32_t* offt_buff, uint16_t* len_buff, uint32_t* out_buff);
};
} // namespace re
} // namespace text
} // namespace data_analytics
} // namespace xf

#endif
