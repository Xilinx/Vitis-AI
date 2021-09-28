/*
 * Copyright 2019 Xilinx, Inc.
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

/**
 * @file utils.hpp
 * @brief This file is part of Vitis Data Analytics Library, contains utilities.
 */

#ifndef _XF_DATA_ANALYTICS_L1_UTILS_HPP_
#define _XF_DATA_ANALYTICS_L1_UTILS_HPP_

// ------------------------------------------------------------

template <typename MType>
union f_cast;

template <>
union f_cast<ap_uint<8> > {
    uint8_t f;
    uint8_t i;
};

template <>
union f_cast<ap_uint<32> > {
    uint32_t f;
    uint32_t i;
};

template <>
union f_cast<ap_uint<64> > {
    uint64_t f;
    uint64_t i;
};

template <>
union f_cast<double> {
    double f;
    uint64_t i;
};

template <>
union f_cast<float> {
    float f;
    uint32_t i;
};

#endif // _XF_DATA_ANALYTICS_UTILS_HPP_
