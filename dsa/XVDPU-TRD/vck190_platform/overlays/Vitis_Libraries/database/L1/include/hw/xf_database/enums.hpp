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
 * @file enums.hpp
 * @brief Header of common enums.
 *
 * This file is part of Vitis Database Library, and should contain only contain
 * definitions of enums and constants, so that it can be included in both host
 * and kenrel code without introducing complex dependency.
 */

#ifndef XF_DATABASE_ENUMS_H
#define XF_DATABASE_ENUMS_H

namespace xf {
namespace database {

// This sub-namespace allows enums to be imported to other
// namespace all together.
namespace enums {

/** @brief Aggregate operators, used by all aggregate functions.
 *
 * hash-aggregate only supports first four operators.
 */
enum AggregateOp {
    AOP_MIN = 0,
    AOP_MAX,
    AOP_SUM,
    AOP_COUNT,
    AOP_COUNTNONZEROS,
    AOP_MEAN,
    AOP_VARIANCE,
    AOP_NORML1,
    AOP_NORML2
};

/**
 * @brief Comparison operator of filter.
 */
enum FilterOp {
    FOP_DC = 0, ///< don't care, always true.
    FOP_EQ,     ///< equal
    FOP_NE,     ///< not equal
    FOP_GT,     ///< greater than, signed.
    FOP_LT,     ///< less than, signed.
    FOP_GE,     ///< greater than or equal, signed.
    FOP_LE,     ///< less than or equal, signed.
    FOP_GTU,    ///< greater than, unsigned.
    FOP_LTU,    ///< less than, unsigned.
    FOP_GEU,    ///< greater than or equal, unsigned.
    FOP_LEU     ///< less than or equal, unsigned.
};

/**
 * @brief JoinType operators
 *
 * CAUTION: hash-multi-cond-join only supports first three operators.
 */
enum JoinType { JT_INNER, JT_SEMI, JT_ANTI, JT_LEFT, JT_RIGHT };

/// @brief width of comparison operator in bits.
enum { FilterOpWidth = 4 };

/// @brief Sort Order enum.
enum SortOrder { SORT_ASCENDING = 1, SORT_DESCENDING = 0 };

} // namespace enums

using namespace enums;

} // namespace database
} // namespace xf

#endif // XF_DATABASE_ENUMS_H
