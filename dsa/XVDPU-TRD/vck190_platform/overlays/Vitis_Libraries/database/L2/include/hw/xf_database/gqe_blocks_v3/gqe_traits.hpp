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
#ifndef XF_DATABASE_GQE_TRAITS_HPP
#define XF_DATABASE_GQE_TRAITS_HPP

#include <hls_stream.h>
#include <ap_int.h>

#include "xf_database/types.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace database {
namespace gqe {

/**
 *
 * @brief templated class helpers for calculation of log2().
 *
 * @tparam number to be calculated for log2().
 *
 */
template <int n>
struct Log2 {
    static const int value = (1 << Log2<n - 1>::value) >= n ? Log2<n - 1>::value : (Log2<n - 1>::value + 1);
};

template <>
struct Log2<1> {
    static const int value{0};
};

/**
 *
 * @brief templated class helpers for finding mask for specific bit-width.
 *
 * @tparam the bit-width.
 *
 */
template <int n>
struct MaskU32 {
    static const uint32_t mask = (~0U) << (32 - n) >> (32 - n);
};

} // namespace gqe
} // namespace database
} // namespace xf

#endif
