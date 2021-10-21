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
#ifndef _LOG_ANALYZER_HPP_
#define _LOG_ANALYZER_HPP_

#include <iostream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <queue>
#include <cstring>
#include "xclhost.hpp"
#include "log_analyzer_config.hpp"
#include "ap_int.h"
typedef ap_uint<512> uint512;
namespace xf {
namespace search {
enum { TH1 = 16, TH2 = 4, Bank1 = 32, Bank2 = 24 };

class logAnalyzer {
   private:
    // variable for platform
    cl_int err;
    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cq;
    cl_program prg;
    std::string xclbin_path;

    x_utils::MM mm;

    // configuration
    logAnalyzerConfig cfg;

    //
    ErrCode analyze_all(uint64_t* cfg_buff,
                        uint64_t* msg_cfg,
                        uint16_t* msg_len_buff,
                        // buffer for GeoIP search
                        uint64_t* net_high16,
                        uint512* net_low21,
                        // buffer for geo info in JSON format
                        uint8_t* geo_buff,
                        uint64_t* geo_len_buff,
                        uint8_t* out_buff);

    uint32_t findSliceNum(
        uint16_t* len_buff, uint32_t lnm, uint32_t* slice_lnm, uint16_t* lnm_per_slc, uint32_t* pos_per_slc);

    int geoCSV2JSON(uint8_t* geo_db_buff, uint32_t* geo_oft_buff, uint8_t* geo_buff, uint64_t* geo_len_buff);

    int geoIPConvert(uint8_t* geo_db_buff, uint32_t* geo_oft_buff, uint64_t* net_high16, uint512* net_low21);

   public:
    uint32_t msg_lnm;
    uint32_t geo_lnm;
    // constructor
    logAnalyzer(std::string xclbin);
    // de-constructor
    ~logAnalyzer();
    // compile
    ErrCode compile(std::string pattern);
    uint32_t getCpgpNm() const;
    /**
      * @brief analyze function
      */
    ErrCode analyze(uint64_t* msg_buff,
                    uint16_t* msg_len_buff,
                    uint32_t msg_lnm,
                    uint8_t* geo_db_buff,
                    uint32_t* geo_oft_buff,
                    uint32_t geo_lnm,
                    uint8_t* out_buff);
};
} // namespace serach
} // namespace xf

#endif
