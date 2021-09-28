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
 * @brief This file is part of Vitis Database Library, contains utilities.
 */

#ifndef XF_DATABASE_UTILS_H
#define XF_DATABASE_UTILS_H

#include "xf_database/types.hpp"

#ifndef __SYNTHESIS__
// for assert function.
#include <cassert>
#define XF_DATABASE_ASSERT(b) assert((b))
#else
#define XF_DATABASE_ASSERT(b) ((void)0)
#endif

#if __cplusplus >= 201103L
#define XF_DATABASE_STATIC_ASSERT(b, m) static_assert((b), m)
#else
#define XF_DATABASE_STATIC_ASSERT(b, m) XF_DATABASE_ASSERT((b) && (m))
#endif

#define XF_DATABASE_MACRO_QUOTE(s) #s
#define XF_DATABASE_MACRO_STR(s) XF_DATABASE_MACRO_QUOTE(s)

namespace xf {
namespace database {
namespace details {
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

} // namespace details
} // namespace database
} // namespace xf
#endif // XF_DATABASE_UTILS_H
// -*- cpp -*-
// vim: ts=8:sw=2:sts=2:ft=cpp
