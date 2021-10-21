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
 * @file aggregate.hpp
 * @brief AGGREGATE template function implementation (including its overloads).
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_AGGREGATE_H
#define XF_DATABASE_AGGREGATE_H

#ifndef __cplusplus
#error "Database Library only works with C++."
#endif

// for wide output
#include <ap_int.h>

// for stream
#include <hls_stream.h>

// for sqrt
#include <hls_math.h>

#include "xf_database/enums.hpp"
#include "xf_database/types.hpp"
#include "xf_database/utils.hpp"

namespace xf {
namespace database {
namespace details {
// max
template <typename T>
inline void aggr_max(hls::stream<T>& in_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<T>& out_strm,
                     hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    T max = 0;
    if (!e) {
        max = in_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline II = 1
        T arg = in_strm.read();
        e = in_e_strm.read();
        max = ((max > arg) ? max : arg);
    }
    out_strm.write(max);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// min
template <typename T>
inline void aggr_min(hls::stream<T>& in_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<T>& out_strm,
                     hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    T min = 0;
    if (!e) {
        min = in_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline II = 1
        T arg = in_strm.read();
        e = in_e_strm.read();
        min = ((min < arg) ? min : arg);
    }
    out_strm.write(min);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// sum
template <typename T, typename T2>
inline void aggr_sum(hls::stream<T>& in_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<T2>& out_strm,
                     hls::stream<bool>& out_e_strm) {
    T2 sum = 0;
    bool e = in_e_strm.read();
AGGREGATE_SUM_P:
    while (!e) {
#pragma HLS pipeline
        sum += in_strm.read();
        e = in_e_strm.read();
    }
    out_strm.write(sum);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// count
template <typename T>
inline void aggr_count(hls::stream<T>& in_strm,
                       hls::stream<bool>& in_e_strm,
                       hls::stream<uint64_t>& out_strm,
                       hls::stream<bool>& out_e_strm) {
    uint64_t cnt = 0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        T t = in_strm.read();
        e = in_e_strm.read();
        ++cnt;
    }
    out_strm.write(cnt);
    out_e_strm.write(false);
    out_e_strm.write(true);
}
template <typename T>
inline void aggr_count(hls::stream<T>& in_strm,
                       hls::stream<bool>& in_e_strm,
                       hls::stream<T>& out_strm,
                       hls::stream<bool>& out_e_strm) {
    T cnt = 0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        T t = in_strm.read();
        e = in_e_strm.read();
        ++cnt;
    }
    out_strm.write(cnt);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// numNonZeros
template <typename T>
inline void aggr_cnz(hls::stream<T>& in_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<uint64_t>& out_strm,
                     hls::stream<bool>& out_e_strm) {
    uint64_t cnt = 0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        T t = in_strm.read();
        e = in_e_strm.read();
        cnt += (t != 0);
    }
    out_strm.write(cnt);
    out_e_strm.write(false);
    out_e_strm.write(true);
}
template <typename T>
inline void aggr_cnz(hls::stream<T>& in_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<T>& out_strm,
                     hls::stream<bool>& out_e_strm) {
    T cnt = 0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        T t = in_strm.read();
        e = in_e_strm.read();
        cnt += (t != 0);
    }
    out_strm.write(cnt);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// mean
template <typename T>
inline void aggr_mean(hls::stream<T>& in_strm,
                      hls::stream<bool>& in_e_strm,
                      hls::stream<T>& out_strm,
                      hls::stream<bool>& out_e_strm) {
    double ret = 0;
    uint64_t cnt = 0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        ret += in_strm.read();
        e = in_e_strm.read();
        ++cnt;
    }
    ret /= cnt;
    out_strm.write((T)ret);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// variance
template <typename T>
inline void aggr_variance(hls::stream<T>& in_strm,
                          hls::stream<bool>& in_e_strm,
                          hls::stream<T>& out_strm,
                          hls::stream<bool>& out_e_strm) {
    uint64_t cnt = 0;
    double mean = 0.0;
    double temp_power = 0.0;
    double variance = 0.0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        T t = in_strm.read();
        e = in_e_strm.read();
        mean += t;
        ++cnt;
        temp_power += t * t;
    }
    mean /= cnt;
    temp_power /= cnt;
    variance = temp_power - (mean * mean);
    out_strm.write((T)variance);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// normL1
template <typename T>
inline void aggr_normL1(hls::stream<T>& in_strm,
                        hls::stream<bool>& in_e_strm,
                        hls::stream<T>& out_strm,
                        hls::stream<bool>& out_e_strm) {
    double ret = 0.0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        T t = in_strm.read();
        e = in_e_strm.read();
        ret += (t > 0) ? t : (T)-t;
    }
    out_strm.write((T)ret);
    out_e_strm.write(false);
    out_e_strm.write(true);
}

// normL2
template <typename T>
inline void aggr_normL2(hls::stream<T>& in_strm,
                        hls::stream<bool>& in_e_strm,
                        hls::stream<T>& out_strm,
                        hls::stream<bool>& out_e_strm) {
    double ret = 0.0;
    bool e = in_e_strm.read();
AGGREGATE_MEAN_P:
    while (!e) {
#pragma HLS pipeline
        T t = in_strm.read();
        e = in_e_strm.read();
        ret += t * t;
    }
    ret = hls::sqrt(ret);
    out_strm.write((T)ret);
    out_e_strm.write(false);
    out_e_strm.write(true);
}
} // namespace details
} // name space database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Overload for most common aggregations.
 *
 * As shown below in the parameters, this function can calculate one of a range of statistics, including
 * minimal, maximal, average (mean), variance, L1 norm, L2 norm. It can also calculate the sum and count.
 *
 * The limitation in this function is that the output data type must match with the input data type. In some cases,
 * the sum or count may overflow the output type, but it can be safely covered by other aggregation overloads.
 *
 * Note that minimum, maximum, sum, count, number of non-zero, L1 norm as well as L2 norm aggregate functions will
 * all be returned as zero when the input is empty.
 *
 * For group-by aggregation, please refer to the ``hashGroupAggregateMPU`` primitive.
 *
 * @tparam op the aggregate operator: AOP_SUM, AOP_MAX, AOP_MIN, AOP_MEAN,
 *    AOP_VARIANCE, AOP_NORML1 or AOP_NORML2
 * @tparam T the data type of input and output streams
 *
 * @param in_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param out_strm output data stream
 * @param out_e_strm end flag stream for output data
 */
template <AggregateOp op, typename T>
void aggregate(hls::stream<T>& in_strm,
               hls::stream<bool>& in_e_strm,
               hls::stream<T>& out_strm,
               hls::stream<bool>& out_e_strm) {
    XF_DATABASE_STATIC_ASSERT((op == AOP_MAX) || (op == AOP_MIN) || (op == AOP_MEAN) || (op == AOP_VARIANCE) ||
                                  (op == AOP_NORML1) || (op == AOP_NORML2) || (op == AOP_SUM) || (op == AOP_COUNT),
                              "Invalid operator");
    if (op == AOP_MAX) {
        details::aggr_max(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_MIN) {
        details::aggr_min(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_MEAN) {
        details::aggr_mean(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_VARIANCE) {
        details::aggr_variance(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_NORML1) {
        details::aggr_normL1(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_NORML2) {
        details::aggr_normL2(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_SUM) {
        details::aggr_sum(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_COUNT) {
        details::aggr_count(in_strm, in_e_strm, out_strm, out_e_strm);
    }
}

/**
 * @brief Aggregate function overload for SUM operation.
 *
 * The output type can be inferred to be different from input type, this allows the sum value to have more precision
 * bits than input, and avoid overflow.
 *
 * Note that sum aggregate function will be returned as zero when the input is empty.
 *
 * For group-by aggregation, please refer to the ``hashGroupAggregateMPU`` primitive.
 *
 * @tparam op the aggregate operator: AOP_SUM
 * @tparam T the data type of input stream, inferred from argument
 * @tparam T2 the data type of output stream, inferred from argument
 *
 * @param in_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param out_strm output data stream
 * @param out_e_strm end flag stream for output data
 */
template <AggregateOp op, typename T, typename T2>
void aggregate(hls::stream<T>& in_strm,
               hls::stream<bool>& in_e_strm,
               hls::stream<T2>& out_strm,
               hls::stream<bool>& out_e_strm) {
    XF_DATABASE_STATIC_ASSERT((op == AOP_SUM), "Invalid operator");
    if (op == AOP_SUM) {
        details::aggr_sum(in_strm, in_e_strm, out_strm, out_e_strm);
    }
}

/**
 * @brief Aggregate function overload for counting.
 *
 * This function counts the number of input rows, or number of non-zero input rows, and returns the count as
 * ``uint64_t`` value.
 *
 * Note that count aggregate function will be returned as zero when the input is empty.
 *
 * For group-by aggregation, please refer to the ``hashGroupAggregateMPU`` primitive.
 *
 * @tparam op the aggregate operator: AOP_COUNT or AOP_COUNTNONZEROS
 * @tparam T the data type of input stream, inferred from argument
 *
 * @param in_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param out_strm output data stream
 * @param out_e_strm end flag stream for output data
 */
template <AggregateOp op, typename T>
void aggregate(hls::stream<T>& in_strm,
               hls::stream<bool>& in_e_strm,
               hls::stream<uint64_t>& out_strm,
               hls::stream<bool>& out_e_strm) {
    XF_DATABASE_STATIC_ASSERT((op == AOP_COUNT) || (op == AOP_COUNTNONZEROS), "Invalid operator");
    if (op == AOP_COUNT) {
        details::aggr_count(in_strm, in_e_strm, out_strm, out_e_strm);
    } else if (op == AOP_COUNTNONZEROS) {
        details::aggr_cnz(in_strm, in_e_strm, out_strm, out_e_strm);
    }
}

} // namespace database
} // namespace xf
#endif
