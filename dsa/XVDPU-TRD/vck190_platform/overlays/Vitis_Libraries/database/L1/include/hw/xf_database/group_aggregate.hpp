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
 * @file group_aggregate.hpp
 * @brief GROUP AGGREGATE template function implementation.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_GROUP_AGGREGATE_H
#define XF_DATABASE_GROUP_AGGREGATE_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#include "xf_database/enums.hpp"
#include "xf_database/types.hpp"
#include "xf_database/utils.hpp"

namespace xf {
namespace database {
namespace details {
// max
template <typename T, typename KEY_T>
inline void group_aggr_max(hls::stream<T>& din_strm,
                           hls::stream<bool>& in_e_strm,
                           hls::stream<T>& dout_strm,
                           hls::stream<bool>& out_e_strm,
                           hls::stream<KEY_T>& kin_strm,
                           hls::stream<KEY_T>& kout_strm) {
    bool e = in_e_strm.read();
    T max;
    KEY_T idx_key;
    if (!e) {
        max = din_strm.read();
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            // same key1 group
            max = ((max > arg) ? max : arg);
            if (e) {
                dout_strm.write(max);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //                std::cout << idx_key <<", " << max << std::endl;
            }
        } else {
            dout_strm.write(max);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //            std::cout << idx_key <<", " << max << std::endl;
            idx_key = idx_arg;
            max = arg;
        }
    }
    out_e_strm.write(1);
}

// min
template <typename T, typename KEY_T>
inline void group_aggr_min(hls::stream<T>& din_strm,
                           hls::stream<bool>& in_e_strm,
                           hls::stream<T>& dout_strm,
                           hls::stream<bool>& out_e_strm,
                           hls::stream<KEY_T>& kin_strm,
                           hls::stream<KEY_T>& kout_strm) {
    bool e = in_e_strm.read();
    T min;
    KEY_T idx_key;
    if (!e) {
        min = din_strm.read();
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            // same key1 group
            min = ((min < arg) ? min : arg);
            if (e) {
                dout_strm.write(min);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //                std::cout << idx_key <<", " << min << std::endl;
            }
        } else {
            dout_strm.write(min);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //            std::cout << idx_key <<", " << min << std::endl;
            idx_key = idx_arg;
            min = arg;
        }
    }
    out_e_strm.write(1);
}

// sum
template <typename T, typename T2, typename KEY_T>
inline void group_aggr_sum(hls::stream<T>& din_strm,
                           hls::stream<bool>& in_e_strm,
                           hls::stream<T2>& dout_strm,
                           hls::stream<bool>& out_e_strm,
                           hls::stream<KEY_T>& kin_strm,
                           hls::stream<KEY_T>& kout_strm) {
    double agg_sum = 0;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        agg_sum = din_strm.read();
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline II = 15
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            agg_sum += arg;
            if (e) {
                dout_strm.write((T2)agg_sum);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //               std::cout << idx_key <<", " << agg_sum << std::endl;
            }
        } else {
            dout_strm.write((T2)agg_sum);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //           std::cout << idx_key <<", " << agg_sum << std::endl;
            idx_key = idx_arg;
            agg_sum = arg;
        }
    }
    out_e_strm.write(1);
}

// count
template <typename T, typename KEY_T>
inline void group_aggr_count(hls::stream<T>& din_strm,
                             hls::stream<bool>& in_e_strm,
                             hls::stream<uint64_t>& dout_strm,
                             hls::stream<bool>& out_e_strm,
                             hls::stream<KEY_T>& kin_strm,
                             hls::stream<KEY_T>& kout_strm) {
    uint64_t cnt = 0;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        din_strm.read();
        cnt++;
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            cnt++;
            if (e) {
                dout_strm.write(cnt);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //               std::cout << idx_key <<", " << cnt << std::endl;
            }
        } else {
            dout_strm.write(cnt);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //           std::cout << idx_key <<", " << cnt << std::endl;
            idx_key = idx_arg;
            cnt = 1;
        }
    }
    out_e_strm.write(1);
}

// count for merge left join
template <typename T, typename KEY_T>
inline void group_aggr_count(hls::stream<T>& din_strm,
                             hls::stream<bool>& isnull_strm,
                             hls::stream<bool>& in_e_strm,
                             hls::stream<uint64_t>& dout_strm,
                             hls::stream<bool>& out_e_strm,
                             hls::stream<KEY_T>& kin_strm,
                             hls::stream<KEY_T>& kout_strm) {
    uint64_t cnt = 0;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        din_strm.read();
        if (isnull_strm.read() == 0) {
            cnt++;
        }
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        e = in_e_strm.read();
        bool isnull = isnull_strm.read();
        if (idx_arg == idx_key) {
            if (isnull == 0) {
                cnt++;
            }
            if (e) {
                dout_strm.write(cnt);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //               std::cout << idx_key <<", " << cnt << std::endl;
            }
        } else {
            dout_strm.write(cnt);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //           std::cout << idx_key <<", " << cnt << std::endl;
            idx_key = idx_arg;
            if (isnull == 0) {
                cnt = 1;
            } else {
                cnt = 0;
            }
        }
    }
    out_e_strm.write(1);
}

// numNonZeros
template <typename T, typename KEY_T>
inline void group_aggr_cnz(hls::stream<T>& din_strm,
                           hls::stream<bool>& in_e_strm,
                           hls::stream<uint64_t>& dout_strm,
                           hls::stream<bool>& out_e_strm,
                           hls::stream<KEY_T>& kin_strm,
                           hls::stream<KEY_T>& kout_strm) {
    uint64_t cnt = 0;
    T ret;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        ret = din_strm.read();
        cnt += (ret != 0);
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline
        KEY_T idx_arg = kin_strm.read();
        ret = din_strm.read();
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            cnt += (ret != 0);
            if (e) {
                dout_strm.write(cnt);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //               std::cout << idx_key <<", " << cnt << std::endl;
            }
        } else {
            dout_strm.write(cnt);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //           std::cout << idx_key <<", " << cnt << std::endl;
            idx_key = idx_arg;
            cnt = (ret != 0);
        }
    }
    out_e_strm.write(1);
}

// mean
template <typename T, typename KEY_T>
inline void group_aggr_mean(hls::stream<T>& din_strm,
                            hls::stream<bool>& in_e_strm,
                            hls::stream<T>& dout_strm,
                            hls::stream<bool>& out_e_strm,
                            hls::stream<KEY_T>& kin_strm,
                            hls::stream<KEY_T>& kout_strm) {
    double agg_sum = 0;
    T ret;
    uint64_t cnt = 0;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        agg_sum = din_strm.read();
        idx_key = kin_strm.read();
        e = in_e_strm.read();
        cnt++;
    }
    while (!e) {
#pragma HLS pipeline II = 15
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            agg_sum += arg;
            cnt++;
            if (e) {
                ret = agg_sum / cnt;
                cnt = 0;
                dout_strm.write(ret);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //               std::cout << idx_key <<", " << ret << std::endl;
            }
        } else {
            ret = agg_sum / cnt;
            dout_strm.write(ret);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //            std::cout << idx_key <<", " << ret << std::endl;
            idx_key = idx_arg;
            agg_sum = arg;
            cnt = 1;
        }
    }
    out_e_strm.write(1);
}

// variance
template <typename T, typename KEY_T>
inline void group_aggr_variance(hls::stream<T>& din_strm,
                                hls::stream<bool>& in_e_strm,
                                hls::stream<T>& dout_strm,
                                hls::stream<bool>& out_e_strm,
                                hls::stream<KEY_T>& kin_strm,
                                hls::stream<KEY_T>& kout_strm) {
    uint64_t cnt = 0;
    double agg_sum = 0.0;
    double mean = 0.0;
    double temp_power = 0.0;
    double variance = 0.0;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        agg_sum = din_strm.read();
        temp_power = agg_sum * agg_sum;
        idx_key = kin_strm.read();
        e = in_e_strm.read();
        cnt++;
    }
    while (!e) {
#pragma HLS pipeline II = 15
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            agg_sum += arg;
            temp_power += arg * arg;
            cnt++;
            if (e) {
                mean = agg_sum / cnt;
                temp_power /= cnt;
                variance = temp_power - (mean * mean);
                dout_strm.write(variance);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //               std::cout << idx_key <<", " << variance << std::endl;
            }
        } else {
            mean = agg_sum / cnt;
            temp_power /= cnt;
            variance = temp_power - (mean * mean);
            dout_strm.write(variance);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //            std::cout << idx_key <<", " << variance << std::endl;
            idx_key = idx_arg;
            agg_sum = arg;
            temp_power = arg * arg;
            cnt = 1;
        }
    }
    out_e_strm.write(1);
}

// normL1
template <typename T, typename KEY_T>
inline void group_aggr_normL1(hls::stream<T>& din_strm,
                              hls::stream<bool>& in_e_strm,
                              hls::stream<T>& dout_strm,
                              hls::stream<bool>& out_e_strm,
                              hls::stream<KEY_T>& kin_strm,
                              hls::stream<KEY_T>& kout_strm) {
    double agg_sum = 0.0;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        T t = din_strm.read();
        agg_sum = (t > 0) ? t : -t;
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline II = 15
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        arg = (arg > 0) ? arg : (T)-arg;
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            agg_sum += arg;
            if (e) {
                dout_strm.write((T)agg_sum);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //                std::cout << idx_key <<", " << agg_sum << std::endl;
            }
        } else {
            dout_strm.write((T)agg_sum);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //            std::cout << idx_key <<", " << agg_sum << std::endl;
            idx_key = idx_arg;
            agg_sum = arg;
        }
    }
    out_e_strm.write(1);
}

// normL2
template <typename T, typename KEY_T>
inline void group_aggr_normL2(hls::stream<T>& din_strm,
                              hls::stream<bool>& in_e_strm,
                              hls::stream<T>& dout_strm,
                              hls::stream<bool>& out_e_strm,
                              hls::stream<KEY_T>& kin_strm,
                              hls::stream<KEY_T>& kout_strm) {
    double agg_sum = 0.0;
    KEY_T idx_key;
    bool e = in_e_strm.read();
    if (!e) {
        T t = din_strm.read();
        agg_sum = t * t;
        idx_key = kin_strm.read();
        e = in_e_strm.read();
    }
    while (!e) {
#pragma HLS pipeline II = 15
        KEY_T idx_arg = kin_strm.read();
        T arg = din_strm.read();
        double argm = arg * arg;
        e = in_e_strm.read();
        if (idx_arg == idx_key) {
            agg_sum += argm;
            if (e) {
                double ret = hls::sqrt(agg_sum);
                dout_strm.write((T)ret);
                kout_strm.write(idx_key);
                out_e_strm.write(0);
                //                std::cout << idx_key <<", " << ret << std::endl;
            }
        } else {
            double ret = hls::sqrt(agg_sum);
            dout_strm.write((T)ret);
            kout_strm.write(idx_key);
            out_e_strm.write(0);
            //            std::cout << idx_key <<", " << ret << std::endl;
            idx_key = idx_arg;
            agg_sum = argm;
        }
    }
    out_e_strm.write(1);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief group aggregate function that returns same type as input
 *
 * @tparam op the aggregate operator: AOP_MAX, AOP_MIN, AOP_MEAN, AOP_VARIANCE,
 * AOP_NORML1 or AOP_NORML2
 * @tparam T the data type of input and output streams
 * @tparam KEY_T the input and output indexing key type
 *
 * @param din_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param dout_strm output data stream
 * @param out_e_strm end flag stream for output data
 * @param kin_strm input indexing key stream
 * @param kout_strm output indexing key stream
 */
template <AggregateOp op, typename T, typename KEY_T>
void groupAggregate(hls::stream<T>& din_strm,
                    hls::stream<bool>& in_e_strm,
                    hls::stream<T>& dout_strm,
                    hls::stream<bool>& out_e_strm,
                    hls::stream<KEY_T>& kin_strm,
                    hls::stream<KEY_T>& kout_strm) {
    XF_DATABASE_STATIC_ASSERT((op == AOP_MAX) || (op == AOP_MIN) || (op == AOP_MEAN) || (op == AOP_VARIANCE) ||
                                  (op == AOP_NORML1) || (op == AOP_NORML2),
                              "Invalid operator");
    if (op == AOP_MAX) {
        details::group_aggr_max(din_strm, in_e_strm, dout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_MIN) {
        details::group_aggr_min(din_strm, in_e_strm, dout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_MEAN) {
        details::group_aggr_mean(din_strm, in_e_strm, dout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_VARIANCE) {
        details::group_aggr_variance(din_strm, in_e_strm, dout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_NORML1) {
        details::group_aggr_normL1(din_strm, in_e_strm, dout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_NORML2) {
        details::group_aggr_normL2(din_strm, in_e_strm, dout_strm, out_e_strm, kin_strm, kout_strm);
    }
}

/**
 * @brief group aggregate function that returns different type as input
 *
 * @tparam op the aggregate operator: AOP_SUM
 * @tparam T the input stream type, inferred from argument
 * @tparam T2 the output stream type, inferred from argument
 * @tparam KEY_T the input and output stream type, inferred from argument
 *
 * @param in_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param out_strm output data stream
 * @param out_e_strm end flag stream for output data
 * @param kin_strm input indexing key stream
 * @param kout_strm output indexing key stream
 */
template <AggregateOp op, typename T, typename T2, typename KEY_T>
void groupAggregate(hls::stream<T>& in_strm,
                    hls::stream<bool>& in_e_strm,
                    hls::stream<T2>& out_strm,
                    hls::stream<bool>& out_e_strm,
                    hls::stream<KEY_T>& kin_strm,
                    hls::stream<KEY_T>& kout_strm) {
    XF_DATABASE_STATIC_ASSERT((op == AOP_SUM), "Invalid operator");
    if (op == AOP_SUM) {
        details::group_aggr_sum(in_strm, in_e_strm, out_strm, out_e_strm, kin_strm, kout_strm);
    }
}

/**
 * @brief aggregate function that counts and returns uint64_t
 *
 * @tparam op the aggregate operator: AOP_COUNT or AOP_COUNTNONZEROS
 * @tparam T the input stream type, inferred from argument
 * @tparam KEY_T the input and output stream type, inferred from argument
 *
 * @param in_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param out_strm output data stream
 * @param out_e_strm end flag stream for output data
 * @param kin_strm input indexing key stream
 * @param kout_strm output indexing key stream
 */
template <AggregateOp op, typename T, typename KEY_T>
void groupAggregate(hls::stream<T>& in_strm,
                    hls::stream<bool>& in_e_strm,
                    hls::stream<uint64_t>& out_strm,
                    hls::stream<bool>& out_e_strm,
                    hls::stream<KEY_T>& kin_strm,
                    hls::stream<KEY_T>& kout_strm) {
    XF_DATABASE_STATIC_ASSERT((op == AOP_COUNT) || (op == AOP_COUNTNONZEROS), "Invalid operator");
    if (op == AOP_COUNT) {
        details::group_aggr_count(in_strm, in_e_strm, out_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_COUNTNONZEROS) {
        details::group_aggr_cnz(in_strm, in_e_strm, out_strm, out_e_strm, kin_strm, kout_strm);
    }
}
/**
 * @brief aggregate function that counts and returns uint64_t
 *
 * @tparam op the aggregate operator: AOP_COUNT
 * @tparam T the input stream type, inferred from argument
 * @tparam KEY_T the input and output stream type, inferred from argument
 *
 * @param in_strm input data stream
 * @param isnull_strm flag to indicate the input data is null or not
 * @param in_e_strm end flag stream for input data
 * @param out_strm output data stream
 * @param out_e_strm end flag stream for output data
 * @param kin_strm input indexing key stream
 * @param kout_strm output indexing key stream
 */
template <AggregateOp op, typename T, typename KEY_T>
void groupAggregate(hls::stream<T>& in_strm,
                    hls::stream<bool>& isnull_strm,
                    hls::stream<bool>& in_e_strm,
                    hls::stream<uint64_t>& out_strm,
                    hls::stream<bool>& out_e_strm,
                    hls::stream<KEY_T>& kin_strm,
                    hls::stream<KEY_T>& kout_strm) {
    XF_DATABASE_STATIC_ASSERT((op == AOP_COUNT), "Invalid operator");
    if (op == AOP_COUNT) {
        details::group_aggr_count(in_strm, isnull_strm, in_e_strm, out_strm, out_e_strm, kin_strm, kout_strm);
    }
}

} // namespace database
} // namespace xf
#endif
