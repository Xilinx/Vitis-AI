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
 * @file naive_bayes.hpp
 * @brief Multinomial Naive Bayes function implementation.
 *
* This file is part of Vitis Data Analytics Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_NAIVE_BAYES_HPP_
#define _XF_DATA_ANALYTICS_L1_NAIVE_BAYES_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_data_analytics/common/math_helper.hpp"

namespace xf {
namespace data_analytics {
namespace classification {
namespace internal {

// -------------  max Value and min Value ------------------------------//
template <typename DT>
struct maxMinValue {};

template <>
struct maxMinValue<double> {
    static const double POSITIVE_INFINITY;
    static const double NEGATIVE_INFINITY;
};
const double maxMinValue<double>::POSITIVE_INFINITY = 1.79769e+308;
const double maxMinValue<double>::NEGATIVE_INFINITY = -1.79769e+308;

template <typename DT, int Dim>
DT maxValueTree(DT* x, int& k, const int s) {
#pragma HLS inline
    const int dr = (Dim + 1) >> 1;
    int kl = 0;
    int kr = 0;
    DT left = maxValueTree<DT, Dim / 2>(x, kl, s);
    DT right = maxValueTree<DT, dr>(x, kr, s + Dim / 2);
    k = (right - left > 1e-8) ? kr + Dim / 2 : kl;
    return (right - left > 1e-8) ? right : left;
}

template <>
double maxValueTree<double, 2>(double* x, int& k, const int s) {
#pragma HLS inline
    k = (x[1 + s] - x[0 + s] > 1e-8) ? 1 : 0;
    return (x[s + 1] - x[s + 0] > 1e-8) ? x[s + 1] : x[s + 0];
}
template <>
double maxValueTree<double, 1>(double* x, int& k, const int s) {
#pragma HLS inline
    k = 0;
    return x[0 + s];
}

template <typename MType>
union f_cast {
    MType f;
    MType i;
};
template <>
union f_cast<unsigned int> {
    unsigned int f;
    unsigned int i;
};
template <>
union f_cast<unsigned long long> {
    unsigned long long f;
    unsigned long long i;
};
template <>
union f_cast<double> {
    double f;
    unsigned long long i;
};
template <>
union f_cast<float> {
    float f;
    unsigned int i;
};

struct MULTINB_DATA {
    ap_uint<12> type;
    ap_uint<20> term;
    ap_uint<32> tf;
};

template <typename DT, int Dim>
DT sum_tree(DT* x, const int s) {
#pragma HLS inline
    DT left = sum_tree<DT, (Dim >> 1)>(x, s);
    DT right = sum_tree<DT, ((Dim + 1) >> 1)>(x, s + (Dim >> 1));
    return (left + right);
}

template <>
ap_uint<32> sum_tree<ap_uint<32>, 2>(ap_uint<32>* x, const int s) {
#pragma HLS inline
    return (x[0 + s] + x[1 + s]);
}
template <>
ap_uint<32> sum_tree<ap_uint<32>, 1>(ap_uint<32>* x, const int s) {
#pragma HLS inline
    return x[0 + s];
}
template <>
ap_uint<64> sum_tree<ap_uint<64>, 2>(ap_uint<64>* x, const int s) {
#pragma HLS inline
    return (x[0 + s] + x[1 + s]);
}
template <>
ap_uint<64> sum_tree<ap_uint<64>, 1>(ap_uint<64>* x, const int s) {
#pragma HLS inline
    return x[0 + s];
}
template <>
double sum_tree<double, 2>(double* x, const int s) {
#pragma HLS inline
    return (x[0 + s] + x[1 + s]);
}
template <>
double sum_tree<double, 1>(double* x, const int s) {
#pragma HLS inline
    return x[0 + s];
}

template <int IN_NM>
ap_uint<3> mux(ap_uint<IN_NM> rd) {
#pragma HLS inline
    ap_uint<3> o = 0;
    if (IN_NM == 8) {
        o[0] = rd[1] | rd[3] | rd[5] | rd[7];
        o[1] = rd[2] | rd[3] | rd[6] | rd[7];
        o[2] = rd[4] | rd[5] | rd[6] | rd[7];
    } else if (IN_NM == 4) {
        o[0] = rd[1] | rd[3];
        o[1] = rd[2] | rd[3];
    } else if (IN_NM == 2) {
        o[0] = rd[1];
    } else {
        o = 0;
    }
    return o;
}

template <int CH_NM>
ap_uint<CH_NM> mul_ch_read(ap_uint<CH_NM> empty) {
    ap_uint<CH_NM> rd = 0;
#pragma HLS inline
    for (int i = 0; i < CH_NM; i++) {
#pragma HLS unroll
        ap_uint<CH_NM> t_e = 0;
        if (i > 0) t_e = empty(i - 1, 0);
        rd[i] = t_e > 0 ? (bool)0 : (bool)empty[i];
    }
    return rd;
}

/**
 * @brief initUram, clear the content of given URAM entitiy
 */
template <int PU>
void initUram(const int num_of_class,
              const int num_of_term,
#ifndef __SYNTHESIS__
              ap_uint<72>* lh_vector[PU],
              ap_uint<96>* prior_vector[PU + 1]
#else
              ap_uint<72> lh_vector[PU][(1 << 20) / PU],
              ap_uint<96> prior_vector[PU + 1][1 << 12]
#endif
              ) {

    const int depth0 = (1 << 17);
    for (int i = 0; i < depth0; i++) {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < PU; j++) {
#pragma HLS unroll
            lh_vector[j][i] = 0;
        }
    }

    const int depth1 = (1 << 12);
    for (int i = 0; i < depth1; i++) {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j <= PU; j++) {
#pragma HLS unroll
            prior_vector[j][i] = 0;
        }
    }
}

/**
 * @breif dispatchUnit, distribute one source data stream to multiple output stream based on its tag
 */
template <int WL>
void dispatchUnit(hls::stream<ap_uint<64> >& i_data_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<MULTINB_DATA> o_data_strm[1 << WL],
                  hls::stream<bool> o_e_strm[1 << WL]) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
LOOP_DISPATCH:
    while (!last) {
#pragma HLS PIPELINE II = 1

        ap_uint<64> m = i_data_strm.read();
        struct MULTINB_DATA d = {m(63, 52), m(51, 32), m(31, 0)};
        ap_uint<20> dt = d.term - 1;
        ap_uint<WL> idx = dt.range(WL - 1, 0);

        last = i_e_strm.read();

        o_data_strm[idx].write(d);
        o_e_strm[idx].write(false);
    }

    struct MULTINB_DATA t = {0, 0, 0};
    for (int i = 0; i < (1 << WL); i++) {
#pragma HLS unroll
        o_data_strm[i].write(t);
        o_e_strm[i].write(true);
    }
}

/**
 * @brief merge8To1, merge 8 source streams into one destination, read next stream until current stream is empty
 */
void merge8To1(hls::stream<MULTINB_DATA>& i0_data_strm,
               hls::stream<MULTINB_DATA>& i1_data_strm,
               hls::stream<MULTINB_DATA>& i2_data_strm,
               hls::stream<MULTINB_DATA>& i3_data_strm,
               hls::stream<MULTINB_DATA>& i4_data_strm,
               hls::stream<MULTINB_DATA>& i5_data_strm,
               hls::stream<MULTINB_DATA>& i6_data_strm,
               hls::stream<MULTINB_DATA>& i7_data_strm,
               hls::stream<bool>& i0_e_strm,
               hls::stream<bool>& i1_e_strm,
               hls::stream<bool>& i2_e_strm,
               hls::stream<bool>& i3_e_strm,
               hls::stream<bool>& i4_e_strm,
               hls::stream<bool>& i5_e_strm,
               hls::stream<bool>& i6_e_strm,
               hls::stream<bool>& i7_e_strm,

               hls::stream<MULTINB_DATA>& o_data_strm,
               hls::stream<bool>& o_e_strm) {
    MULTINB_DATA data_arry[8];
#pragma HLS ARRAY_PARTITION variable = data_arry complete dim = 1
    ap_uint<8> empty_e = 0;
    ;
    ap_uint<8> rd_e = 0;
    ;
    ap_uint<8> last = 0;
LOOP_MERGE8_1:
    do {
#pragma HLS loop_tripcount min = 1 max = 5000
#pragma HLS PIPELINE II = 1
        empty_e[0] = !i0_e_strm.empty() && !last[0];
        empty_e[1] = !i1_e_strm.empty() && !last[1];
        empty_e[2] = !i2_e_strm.empty() && !last[2];
        empty_e[3] = !i3_e_strm.empty() && !last[3];
        empty_e[4] = !i4_e_strm.empty() && !last[4];
        empty_e[5] = !i5_e_strm.empty() && !last[5];
        empty_e[6] = !i6_e_strm.empty() && !last[6];
        empty_e[7] = !i7_e_strm.empty() && !last[7];
        rd_e = mul_ch_read<8>(empty_e);
        if (rd_e[0]) {
            data_arry[0] = i0_data_strm.read();
            last[0] = i0_e_strm.read();
        }
        if (rd_e[1]) {
            data_arry[1] = i1_data_strm.read();
            last[1] = i1_e_strm.read();
        }
        if (rd_e[2]) {
            data_arry[2] = i2_data_strm.read();
            last[2] = i2_e_strm.read();
        }
        if (rd_e[3]) {
            data_arry[3] = i3_data_strm.read();
            last[3] = i3_e_strm.read();
        }
        if (rd_e[4]) {
            data_arry[4] = i4_data_strm.read();
            last[4] = i4_e_strm.read();
        }
        if (rd_e[5]) {
            data_arry[5] = i5_data_strm.read();
            last[5] = i5_e_strm.read();
        }
        if (rd_e[6]) {
            data_arry[6] = i6_data_strm.read();
            last[6] = i6_e_strm.read();
        }
        if (rd_e[7]) {
            data_arry[7] = i7_data_strm.read();
            last[7] = i7_e_strm.read();
        }

        ap_uint<3> id = mux<8>(rd_e);
        MULTINB_DATA d = data_arry[id];
        bool valid_n = last[id];
        if (!valid_n && rd_e != 0) {
            o_data_strm.write(d);
            o_e_strm.write(false);
        }
    } while (last != 255);

    o_e_strm.write(true);
}

/**
 * @brief counterUnit, count the number of each term for each class, and then store into LH table
 */
template <typename DT, int DT_WIDTH, int WL>
void counterUnit(const int num_of_class,
                 const int num_of_term,

                 hls::stream<MULTINB_DATA>& i_data_strm,
                 hls::stream<bool>& i_e_strm,

                 hls::stream<MULTINB_DATA>& o_data_strm,
                 hls::stream<bool>& o_e_strm,

                 ap_uint<72>* lh_vector) {
    const int depth = num_of_term;

    ap_uint<72> elem = 0;
    ap_uint<72> elem_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<20> line_idx_temp[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

    bool last = i_e_strm.read();
COUNTER_UNIT_CORE_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = lh_vector inter false

        MULTINB_DATA data = i_data_strm.read();
        last = i_e_strm.read();

        ap_uint<12> type = data.type;
        ap_uint<20> line_idx;
        ap_uint<1> bit_idx;
        ap_uint<20> dt = data.term - 1;
        if (DT_WIDTH <= 36) {
            line_idx = type * depth + dt.range(19, WL + 1);
            bit_idx = dt[WL];
        } else {
            line_idx = type * depth + dt;
            bit_idx = 0;
        }

        // read uram
        if (line_idx == line_idx_temp[0]) {
            elem = elem_temp[0];
        } else if (line_idx == line_idx_temp[1]) {
            elem = elem_temp[1];
        } else if (line_idx == line_idx_temp[2]) {
            elem = elem_temp[2];
        } else if (line_idx == line_idx_temp[3]) {
            elem = elem_temp[3];
        } else if (line_idx == line_idx_temp[4]) {
            elem = elem_temp[4];
        } else if (line_idx == line_idx_temp[5]) {
            elem = elem_temp[5];
        } else if (line_idx == line_idx_temp[6]) {
            elem = elem_temp[6];
        } else if (line_idx == line_idx_temp[7]) {
            elem = elem_temp[7];
        } else {
            elem = lh_vector[line_idx];
        }

        ap_uint<72> elem_next;
        if (DT_WIDTH <= 36) {
            for (int i = 0; i < 2; i++) {
#pragma HLS unroll
                f_cast<DT> cc1, cc2, cc3;
                cc1.i = elem(32 * i + 31, 32 * i);
                cc2.i = data.tf;
                cc3.f = cc1.f + cc2.f;
                elem_next(32 * i + 31, 32 * i) = ((bit_idx == i) ? cc3.i : elem(32 * i + 31, 32 * i));
            }
        } else {
            f_cast<DT> cc1, cc2, cc3;
            cc1.i = elem(63, 0);
            cc2.i = data.tf;
            cc3.f = cc1.f + cc2.f;
            elem_next(63, 0) = cc3.i;
        }
        elem_next(71, 64) = 0;

        // write back
        lh_vector[line_idx] = elem_next;

        // right shift temp
        for (int i = 7; i > 0; i--) {
#pragma HLS unroll
            elem_temp[i] = elem_temp[i - 1];
            line_idx_temp[i] = line_idx_temp[i - 1];
        }
        elem_temp[0] = elem_next;
        line_idx_temp[0] = line_idx;

        o_data_strm.write(data);
        o_e_strm.write(false);
    }

    o_e_strm.write(true);
}

/**
 * @brief collectUnit, count the number of train sample, and get the maximum feature value
 */
void collectUnit(const int num_of_class,

                 hls::stream<MULTINB_DATA>& i_data_strm,
                 hls::stream<bool>& i_e_strm,

                 hls::stream<ap_uint<96> >& o_prior_strm,
                 hls::stream<ap_uint<32> >& o_dist_sum_strm,

                 ap_uint<96>* prior_vector) {
    ap_uint<32> dist_tag = 0;
    ap_uint<96> elem = 0;
    ap_uint<96> elem_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<12> line_idx_temp[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

    bool last = i_e_strm.read();
COLLECT_UNIT_CORE_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = prior_vector inter false

        MULTINB_DATA data = i_data_strm.read();
        last = i_e_strm.read();

        ap_uint<12> line_idx = data.type;
        ap_uint<20> term = data.term;

        if (term != (ap_uint<20>)-1) {
            dist_tag = (dist_tag < term) ? (ap_uint<32>)term : dist_tag;
        }

        // read uram
        if (line_idx == line_idx_temp[0]) {
            elem = elem_temp[0];
        } else if (line_idx == line_idx_temp[1]) {
            elem = elem_temp[1];
        } else if (line_idx == line_idx_temp[2]) {
            elem = elem_temp[2];
        } else if (line_idx == line_idx_temp[3]) {
            elem = elem_temp[3];
        } else if (line_idx == line_idx_temp[4]) {
            elem = elem_temp[4];
        } else if (line_idx == line_idx_temp[5]) {
            elem = elem_temp[5];
        } else if (line_idx == line_idx_temp[6]) {
            elem = elem_temp[6];
        } else if (line_idx == line_idx_temp[7]) {
            elem = elem_temp[7];
        } else {
            elem = prior_vector[line_idx];
        }

        ap_uint<96> elem_next;
        elem_next(63, 0) = elem(31, 0) + data.tf; // No. of total term for class y
        elem_next(95, 64) = (term == (ap_uint<20>)-1) ? elem(95, 64) + 1 : elem(95, 64); // No. of class y

        // write back
        prior_vector[line_idx] = elem_next;

        // right shift temp
        for (int i = 7; i > 0; i--) {
#pragma HLS unroll
            elem_temp[i] = elem_temp[i - 1];
            line_idx_temp[i] = line_idx_temp[i - 1];
        }
        elem_temp[0] = elem_next;
        line_idx_temp[0] = line_idx;
    }

    o_dist_sum_strm.write(dist_tag);

    for (int i = 0; i < num_of_class; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<96> tmp = prior_vector[i];
        o_prior_strm.write(tmp);
    }
}

/**
 * @brief priorGather, calculate the total samples and the maximum feature
 */
template <int PU>
void priorGather(const int num_of_class,
                 hls::stream<ap_uint<96> > i_prior_strm[PU],
                 hls::stream<ap_uint<32> > i_dist_sum_strm[PU],
                 ap_uint<64>& term_sum,
                 ap_uint<96>* prior_prob) {
#pragma HLS INLINE off
    ap_uint<64> temp_sum = 0;
    for (int i = 0; i < num_of_class; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<96> tmp0[PU];
#pragma HLS ARRAY_PARTITION variable = tmp0 complete dim = 0
        ap_uint<64> tmp1[PU];
#pragma HLS ARRAY_PARTITION variable = tmp1 complete dim = 0
        ap_uint<32> tmp2[PU];
#pragma HLS ARRAY_PARTITION variable = tmp2 complete dim = 0
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            tmp0[p] = i_prior_strm[p].read();
            tmp1[p] = tmp0[p](63, 0);
            tmp2[p] = tmp0[p](95, 64);
        }

        ap_uint<96> new_elem = 0;
        new_elem(63, 0) = sum_tree<ap_uint<64>, PU>(tmp1, 0);
        new_elem(95, 64) = sum_tree<ap_uint<32>, PU>(tmp2, 0);

        temp_sum(63, 32) = temp_sum(63, 32) + new_elem(95, 64); // the number of samples

        prior_prob[i] = new_elem;
    }

    ap_uint<32> tmp3 = 0;
    for (int p = 0; p < PU; p++) {
#pragma HLS PIPELINE II = 1
        ap_uint<32> t = i_dist_sum_strm[p].read();
        tmp3 = (tmp3 < t) ? t : tmp3;
    }
    temp_sum(31, 0) = tmp3;

    term_sum = temp_sum;
}

/**
 * @brief trainCore, the top function of multinomial Naive Bayes training
 */
template <typename DT, int WL, int PU>
void trainCore(const int num_of_class,
               const int num_of_term,
               hls::stream<ap_uint<64> > i_data_strm[PU],
               hls::stream<bool> i_e_strm[PU],

               ap_uint<64>& term_sum,
#ifndef __SYNTHESIS__
               ap_uint<72>* lh_vector[PU],
               ap_uint<96>* prior_vector[PU + 1]
#else
               ap_uint<72> lh_vector[PU][(1 << 20) / PU],
               ap_uint<96> prior_vector[PU + 1][1 << 12]
#endif
               ) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<MULTINB_DATA> dispatch_data_array[PU][PU];
#pragma HLS stream variable = dispatch_data_array depth = 8
    hls::stream<bool> dispatch_e_array[PU][PU];
#pragma HLS stream variable = dispatch_e_array depth = 8

    hls::stream<MULTINB_DATA> merge_data_array[PU];
#pragma HLS stream variable = merge_data_array depth = 8
    hls::stream<bool> merge_e_array[PU];
#pragma HLS stream variable = merge_e_array depth = 8

    hls::stream<MULTINB_DATA> counter_data_array[PU];
#pragma HLS stream variable = counter_data_array depth = 8
    hls::stream<bool> counter_e_array[PU];
#pragma HLS stream variable = counter_e_array depth = 8

    hls::stream<ap_uint<96> > collect_prior_strm[PU];
#pragma HLS stream variable = collect_prior_strm depth = 32

    hls::stream<ap_uint<32> > dist_sum_strm[PU];
#pragma HLS stream variable = dist_sum_strm depth = 2

    for (int i = 0; i < PU; i++) {
#pragma HLS UNROLL
        dispatchUnit<WL>(i_data_strm[i], i_e_strm[i], dispatch_data_array[i], dispatch_e_array[i]);
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS UNROLL
        merge8To1(dispatch_data_array[0][i], dispatch_data_array[1][i], dispatch_data_array[2][i],
                  dispatch_data_array[3][i], dispatch_data_array[4][i], dispatch_data_array[5][i],
                  dispatch_data_array[6][i], dispatch_data_array[7][i], dispatch_e_array[0][i], dispatch_e_array[1][i],
                  dispatch_e_array[2][i], dispatch_e_array[3][i], dispatch_e_array[4][i], dispatch_e_array[5][i],
                  dispatch_e_array[6][i], dispatch_e_array[7][i], merge_data_array[i], merge_e_array[i]);
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS UNROLL
        counterUnit<DT, 32, WL>(num_of_class, num_of_term, merge_data_array[i], merge_e_array[i], counter_data_array[i],
                                counter_e_array[i], lh_vector[i]);

        collectUnit(num_of_class, counter_data_array[i], counter_e_array[i], collect_prior_strm[i], dist_sum_strm[i],
                    prior_vector[i]);
    }

    priorGather<PU>(num_of_class, collect_prior_strm, dist_sum_strm, term_sum, prior_vector[PU]);
}

/**
 * @brief trainWriteOut, calculate the log of prior probablity and likehood probablity, and then write into stream
 */
template <typename DT, int DT_WIDTH, int PU>
void trainWriteOut(const int num_of_class,
                   const int num_of_term,
                   const ap_uint<64> term_sum,

                   hls::stream<int>& o_terms_strm,
                   hls::stream<ap_uint<64> > o_d0_strm[PU],
                   hls::stream<ap_uint<64> > o_d1_strm[PU],
#ifndef __SYNTHESIS__
                   ap_uint<72>* lh_vector[PU],
                   ap_uint<96>* prior_vector
#else
                   ap_uint<72> lh_vector[PU][(1 << 20) / PU],
                   ap_uint<96> prior_vector[1 << 12]
#endif
                   ) {
    ;
    const int nread = (term_sum(31, 0) + PU - 1) / PU;
    o_terms_strm.write(term_sum(31, 0));

    for (int c = 0; c < num_of_class; c++) {
#pragma HLS loop_tripcount min = 2 avg = 2 max = 2
        ap_uint<96> Ny = prior_vector[c];
        for (ap_uint<32> n = 0; n < nread; n++) {
#pragma HLS loop_tripcount min = 5904 avg = 5904 max = 5904
#pragma HLS PIPELINE II = 1
            for (int p = 0; p < PU; p++) {
#pragma HLS unroll
                f_cast<DT> cc0;
                if (DT_WIDTH <= 36) {
                    ap_uint<72> t = lh_vector[p][c * num_of_term + n(31, 1)];
                    cc0.i = t(32 * n[0] + 31, 32 * n[0]);
                } else {
                    cc0.i = lh_vector[p][c * num_of_term + n].range(63, 0);
                }

                double numer = (double)(cc0.f + 1);
                double denom = (double)(Ny(63, 0) + term_sum(31, 0));
                double lh = numer / denom;

                f_cast<double> cc1;
                cc1.f = xf::data_analytics::internal::m::log(lh);
                o_d0_strm[p].write(cc1.i);
            }
        }
    }

    ap_uint<64> tmp[PU];
#pragma HLS ARRAY_PARTITION variable = tmp complete dim = 0
    for (int c = 0; c < num_of_class; c++) {
#pragma HLS PIPELINE II = 1
        ap_uint<96> Ny = prior_vector[c];
        double prior = (double)Ny(95, 64) / (double)term_sum(63, 32);

        f_cast<double> cc1;
        cc1.f = xf::data_analytics::internal::m::log(prior);
        tmp[c % PU] = cc1.i;

        if (c % PU == PU - 1) {
            for (int p = 0; p < PU; p++) {
#pragma HLS unroll
                o_d1_strm[p].write(tmp[p]);
            }
        }
    }

    const int remain = num_of_class % PU;
    if (remain != 0) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            ap_uint<64> t = (p < remain) ? tmp[p] : (ap_uint<64>)0;
            o_d1_strm[p].write(t);
        }
    }
}

} // namespace internal
} // namespace classification
} // namespace data_analytics
} // namespace xf

namespace xf {
namespace data_analytics {
namespace classification {
namespace internal {

/**
 * @brief loadTheta, likehood probability matrix loader, it will read num_of_class * num_of_term times
 */
template <int GRP_NM>
void loadTheta(const int num_of_class,
               const int num_of_term,
               hls::stream<ap_uint<64> >& i_theta_strm,
#ifndef __SYNTHESIS__
               ap_uint<72>* lh_vector[GRP_NM][256 / GRP_NM]
#else
               ap_uint<72> lh_vector[GRP_NM][256 / GRP_NM][4096]
#endif
               ) {
#pragma HLS inline off

    const int num_per_grp = 256 / GRP_NM;
    const int line = (num_of_class + GRP_NM - 1) / GRP_NM;
    const int depth_per_line = 4096 / line;

    for (int i = 0; i < num_of_class; i++) {
        for (int j = 0; j < num_of_term; j++) {
#pragma HLS PIPELINE II = 1
            ap_uint<72> tmp;
            tmp(71, 64) = 0;
            tmp(63, 0) = i_theta_strm.read();

            int offset = (i / GRP_NM) * depth_per_line;
            int z_addr = offset + j / num_per_grp;

            lh_vector[i % GRP_NM][j % num_per_grp][z_addr] = tmp;
        }
    }
}

/**
 * @brief loadPrior, prior probability vector loader, it will read num_of_class times
 */
template <int GRP_NM>
void loadPrior(const int num_of_class,
               hls::stream<ap_uint<64> >& i_prior_strm,
#ifndef __SYNTHESIS__
               ap_uint<64>* prior_vector[GRP_NM]
#else
               ap_uint<64> prior_vector[GRP_NM][1024 / GRP_NM]
#endif
               ) {
#pragma HLS inline off

    for (int i = 0; i < num_of_class; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<64> tmp = i_prior_strm.read();
        prior_vector[i % GRP_NM][i / GRP_NM] = tmp;
    }
}

template <int GRP_NM>
void initResVector(
#ifndef __SYNTHESIS__
    ap_uint<64>* result_vector[GRP_NM]
#else
    ap_uint<64> result_vector[GRP_NM][1024 / GRP_NM]
#endif
    ) {

    for (int i = 0; i < 1024 / GRP_NM; i++) {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < GRP_NM; j++) {
#pragma HLS unroll
            result_vector[j][i] = 0;
        }
    }
}

template <int GRP_NM>
void loadModel(const int num_of_class,
               const int num_of_term,
               hls::stream<ap_uint<64> >& i_theta_strm,
               hls::stream<ap_uint<64> >& i_prior_strm,
#ifndef __SYNTHESIS__
               ap_uint<72>* lh_vector[GRP_NM][256 / GRP_NM],
               ap_uint<64>* prior_vector[GRP_NM],
               ap_uint<64>* result_vector[GRP_NM]
#else
               ap_uint<72> lh_vector[GRP_NM][256 / GRP_NM][4096],
               ap_uint<64> prior_vector[GRP_NM][1024 / GRP_NM],
               ap_uint<64> result_vector[GRP_NM][1024 / GRP_NM]
#endif
               ) {
#pragma HLS inline off
#pragma HLS DATAFLOW

    loadTheta<GRP_NM>(num_of_class, num_of_term, i_theta_strm, lh_vector);

    loadPrior<GRP_NM>(num_of_class, i_prior_strm, prior_vector);

    initResVector<GRP_NM>(result_vector);
}

/**
 * @brief uramAccess, likehood probability is read line by line, GRP_NM*CH_NM per cycle
 */
template <int CH_NM, int GRP_NM>
void uramAccess(const int num_of_class,
                const int num_of_term,
                hls::stream<ap_uint<32> > i_data_strm[CH_NM],
                hls::stream<bool>& i_e_strm,

                hls::stream<double> o_prob_strm[GRP_NM][CH_NM],
                hls::stream<ap_uint<32> > o_data_strm[GRP_NM][CH_NM],
                hls::stream<bool>& o_e_strm,
#ifndef __SYNTHESIS__
                ap_uint<72>* lh_vector[GRP_NM][256 / GRP_NM]
#else
                ap_uint<72> lh_vector[GRP_NM][256 / GRP_NM][4096]
#endif
                ) {
#pragma HLS inline off

    const int num_per_grp = 256 / GRP_NM;
    const int line = (num_of_class + GRP_NM - 1) / GRP_NM;
    const int depth_per_line = 4096 / line;

    int r = 0;

    bool last = i_e_strm.read();
    while (!last) {
        last = i_e_strm.read();

        ap_uint<32> sample[CH_NM];
#pragma HLS ARRAY_PARTITION variable = sample complete dim = 0
        for (int j = 0; j < CH_NM; j++) {
#pragma HLS UNROLL
            sample[j] = i_data_strm[j].read();
        }

        for (int i = 0; i < line; i++) {
#pragma HLS PIPELINE II = 1
            for (int j = 0; j < CH_NM; j++) {
#pragma HLS UNROLL
                for (int k = 0; k < GRP_NM; k++) {
#pragma HLS UNROLL
                    f_cast<double> cc0;
                    cc0.i = lh_vector[k][r + j][depth_per_line * i + r / num_per_grp];
                    o_prob_strm[k][j].write(cc0.f);
                    o_data_strm[k][j].write(sample[j]);
                }
            }
            o_e_strm.write(false);
        }

        if (r + CH_NM < num_of_term)
            r += CH_NM;
        else
            r = 0;
    }

    o_e_strm.write(true);
}

/*
 * @brief treeCluster, do vector multipler and tree-adder, burst write once if storing 4 data
 */
template <int CH_NM, int GRP_NM>
void treeCluster(const int num_of_class,
                 const int num_of_term,
                 hls::stream<double> i_prob_strm[GRP_NM][CH_NM],
                 hls::stream<ap_uint<32> > i_data_strm[GRP_NM][CH_NM],
                 hls::stream<bool>& i_e_strm,

                 hls::stream<ap_uint<10> >& o_level_strm,
                 hls::stream<bool>& o_sample_end_strm,
                 hls::stream<double> o_data_strm[GRP_NM][4],
                 hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

    const int line1 = (num_of_class + GRP_NM - 1) / GRP_NM;
    const int line2 = (num_of_term + CH_NM - 1) / CH_NM;
    const int size = line1 * line2;

#ifndef __SYNTHESIS__
    ap_uint<64>* tmp_vector[GRP_NM][3];
    for (int i = 0; i < GRP_NM; i++) {
        for (int j = 0; j < 3; j++) {
            tmp_vector[i][j] = (ap_uint<64>*)malloc(1024 / GRP_NM * sizeof(ap_uint<64>));
            memset(tmp_vector[i][j], 0, 1024 / GRP_NM * sizeof(ap_uint<64>));
        }
    }
#else
    ap_uint<64> tmp_vector[GRP_NM][3][1024 / GRP_NM];
#pragma HLS bind_storage variable = tmp_vector type = ram_2p impl = lutram
#pragma HLS array_partition variable = tmp_vector complete dim = 1
#pragma HLS array_partition variable = tmp_vector complete dim = 2

#endif

    ap_uint<10> line_idx = 0;

    double tmp0[CH_NM];
#pragma HLS ARRAY_PARTITION variable = tmp0 complete dim = 0
    ap_uint<32> tmp1[CH_NM];
#pragma HLS ARRAY_PARTITION variable = tmp1 complete dim = 0
    double tmp2[CH_NM];
#pragma HLS ARRAY_PARTITION variable = tmp2 complete dim = 0

    int r = 0;

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1
        last = i_e_strm.read();
        ap_uint<10> level = line_idx % line1;
        bool is_sample_end = (line_idx / line1 == (line2 - 1));

        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
            for (int j = 0; j < CH_NM; j++) {
#pragma HLS UNROLL
                tmp0[j] = i_prob_strm[i][j].read();
                tmp1[j] = i_data_strm[i][j].read();
                tmp2[j] = tmp0[j] * tmp1[j].to_int();
            }

            f_cast<double> tmp3;
            tmp3.f = sum_tree<double, CH_NM>(tmp2, 0);
            if (((r + 1) % 4 != 0) && ((r + 1) < line2)) tmp_vector[i][r % 4][level] = tmp3.i;
            // reach one burst
            if (((r + 1) % 4 == 0) || (r + 1) == line2) {
                for (int j = 0; j < 3; j++) {
#pragma HLS UNROLL
                    f_cast<double> cc0;
                    if (j < r % 4)
                        cc0.i = tmp_vector[i][j][level];
                    else
                        cc0.i = 0;

                    o_data_strm[i][j].write(cc0.f);
                }
                o_data_strm[i][3].write(tmp3.f);
            }
        }

        if (((r + 1) % 4 == 0) || (r + 1) == line2) {
            o_sample_end_strm.write(is_sample_end);
            o_level_strm.write(level);
            o_e_strm.write(false);
        }

        if ((r + 1 < line2) && ((level + 1) == line1))
            r++;
        else if ((line_idx + 1) == size && (r + 1 == line2))
            r = 0;

        if (line_idx + 1 < size)
            line_idx++;
        else
            line_idx = 0;
    }

    o_e_strm.write(true);
}

/*
 * @brief treeAdder, do tree-adder for 4-channel
 */
template <int GRP_NM>
void treeAdder(hls::stream<ap_uint<10> >& i_level_strm,
               hls::stream<bool>& i_sample_end_strm,
               hls::stream<double> i_data_strm[GRP_NM][4],
               hls::stream<bool>& i_e_strm,

               hls::stream<ap_uint<10> >& o_level_strm,
               hls::stream<bool>& o_sample_end_strm,
               hls::stream<double> o_data_strm[GRP_NM],
               hls::stream<bool>& o_e_strm) {
    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1
        last = i_e_strm.read();

        double tmp0[GRP_NM][4];
#pragma HLS ARRAY_PARTITION variable = tmp0 complete dim = 0
        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
            for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
                tmp0[i][j] = i_data_strm[i][j].read();
            }

            double tmp1 = sum_tree<double, 4>(tmp0[i], 0);
            o_data_strm[i].write(tmp1);
        }

        o_level_strm.write(i_level_strm.read());
        o_sample_end_strm.write(i_sample_end_strm.read());
        o_e_strm.write(false);
    }

    o_e_strm.write(true);
}

/*
 * @brief accmCluster, accumulate for each sample, do GRP_NM line per cycle
 */
template <int CH_NM, int GRP_NM>
void accmCluster(hls::stream<ap_uint<10> >& i_level_strm,
                 hls::stream<bool>& i_sample_end_strm,
                 hls::stream<double> i_data_strm[GRP_NM],
                 hls::stream<bool>& i_e_strm,

                 hls::stream<double> o_accm_strm[GRP_NM],
                 hls::stream<bool>& o_e_strm,
#ifndef __SYNTHESIS__
                 ap_uint<64>* result_vector[GRP_NM]
#else
                 ap_uint<64> result_vector[GRP_NM][1024 / GRP_NM]
#endif
                 ) {
#pragma HLS inline off

    ap_uint<64> elem[GRP_NM];
#pragma HLS ARRAY_PARTITION variable = elem complete dim = 0
    ap_uint<64> elem_temp[GRP_NM][12];
#pragma HLS ARRAY_PARTITION variable = elem_temp complete dim = 0
    ap_uint<10> line_idx_temp[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
#pragma HLS ARRAY_PARTITION variable = line_idx_temp complete dim = 0

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 4
#pragma HLS DEPENDENCE variable = result_vector inter false

        last = i_e_strm.read();
        bool sample_end = i_sample_end_strm.read();
        ap_uint<10> line_idx = i_level_strm.read();

        double in[GRP_NM];
#pragma HLS array_partition variable = in complete dim = 0
        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
            in[i] = i_data_strm[i].read();
        }

        // read cache
        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
            if (line_idx == line_idx_temp[0]) {
                elem[i] = elem_temp[i][0];
            } else if (line_idx == line_idx_temp[1]) {
                elem[i] = elem_temp[i][1];
            } else if (line_idx == line_idx_temp[2]) {
                elem[i] = elem_temp[i][2];
            } else if (line_idx == line_idx_temp[3]) {
                elem[i] = elem_temp[i][3];
            } else if (line_idx == line_idx_temp[4]) {
                elem[i] = elem_temp[i][4];
            } else if (line_idx == line_idx_temp[5]) {
                elem[i] = elem_temp[i][5];
            } else if (line_idx == line_idx_temp[6]) {
                elem[i] = elem_temp[i][6];
            } else if (line_idx == line_idx_temp[7]) {
                elem[i] = elem_temp[i][7];
            } else if (line_idx == line_idx_temp[8]) {
                elem[i] = elem_temp[i][8];
            } else if (line_idx == line_idx_temp[9]) {
                elem[i] = elem_temp[i][9];
            } else if (line_idx == line_idx_temp[10]) {
                elem[i] = elem_temp[i][10];
            } else if (line_idx == line_idx_temp[11]) {
                elem[i] = elem_temp[i][11];
            } else {
                elem[i] = result_vector[i][line_idx];
            }
        }

        double cc2[GRP_NM];
#pragma HLS array_partition variable = cc2 complete dim = 0
        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
            f_cast<double> cc0;
            cc0.i = elem[i];
            cc2[i] = cc0.f + in[i];
        }

        ap_uint<64> elem_next[GRP_NM];
#pragma HLS ARRAY_PARTITION variable = elem_next complete dim = 0
        if (!sample_end) { // Not sample end
            for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
                f_cast<double> cc3;
                cc3.f = cc2[i];
                elem_next[i] = cc3.i;
            }

        } else { // sample end
            for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
                o_accm_strm[i].write(cc2[i]);
                elem_next[i] = 0;
            }

            o_e_strm.write(false);
        }

        // write cache
        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS UNROLL
            result_vector[i][line_idx] = elem_next[i];
        }

        // right shift temp
        for (int j = 0; j < GRP_NM; j++) {
#pragma HLS unroll
            for (int i = 11; i > 0; i--) {
#pragma HLS unroll
                elem_temp[j][i] = elem_temp[j][i - 1];
            }
            elem_temp[j][0] = elem_next[j];
        }

        for (int i = 11; i > 0; i--) {
#pragma HLS unroll
            line_idx_temp[i] = sample_end ? (ap_uint<10>)-1 : line_idx_temp[i - 1];
        }
        line_idx_temp[0] = sample_end ? (ap_uint<10>)-1 : line_idx;
    }

    o_e_strm.write(true);
}

/**
 * @brief finalProb, calculate the final probability by adding the prior probability for each sample
 */
template <int GRP_NM>
void finalProb(const int num_of_class,
               hls::stream<double> i_data_strm[GRP_NM],
               hls::stream<bool>& i_e_strm,

               hls::stream<double> o_data_strm[GRP_NM],
               hls::stream<bool>& o_e_strm,
#ifndef __SYNTHESIS__
               ap_uint<64>* prior_vector[GRP_NM]
#else
               ap_uint<64> prior_vector[GRP_NM][1024 / GRP_NM]
#endif
               ) {
#pragma HLS inline off

    ap_uint<10> line_idx = 0;

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        last = i_e_strm.read();

        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS unroll
            f_cast<double> cc0;
            double tmp0 = i_data_strm[i].read();
            cc0.i = prior_vector[i][line_idx / GRP_NM];
            double tmp2 = tmp0 + cc0.f;

            o_data_strm[i].write(tmp2);
        }

        if (line_idx + GRP_NM < num_of_class)
            line_idx += GRP_NM;
        else
            line_idx = 0;

        o_e_strm.write(false);
    }

    o_e_strm.write(true);
}
/**
 * @brief treeProcess, process core
 */
template <int CH_NM, int GRP_NM>
void treeProcess(const int num_of_class,
                 const int num_of_term,
                 hls::stream<double> i_prob_strm[GRP_NM][CH_NM],
                 hls::stream<ap_uint<32> > i_data_strm[GRP_NM][CH_NM],
                 hls::stream<bool>& i_e_strm,

                 hls::stream<double> o_data_strm[GRP_NM],
                 hls::stream<bool>& o_e_strm,
#ifndef __SYNTHESIS__
                 ap_uint<64>* prior_vector[GRP_NM],
                 ap_uint<64>* result_vector[GRP_NM]
#else
                 ap_uint<64> prior_vector[GRP_NM][1024 / GRP_NM],
                 ap_uint<64> result_vector[GRP_NM][1024 / GRP_NM]
#endif
                 ) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<ap_uint<10> > addT_level_strm;
#pragma HLS stream variable = addT_level_strm depth = 8
    hls::stream<bool> addT_sample_end_strm;
#pragma HLS stream variable = addT_sample_end_strm depth = 8
    hls::stream<double> addT_data_strm[GRP_NM][4];
#pragma HLS stream variable = addT_data_strm depth = 8
    hls::stream<bool> addT_e_strm;
#pragma HLS stream variable = addT_e_strm depth = 8

    hls::stream<ap_uint<10> > tA_level_strm;
#pragma HLS stream variable = tA_level_strm depth = 8
    hls::stream<bool> tA_sample_end_strm;
#pragma HLS stream variable = tA_sample_end_strm depth = 8
    hls::stream<double> tA_data_strm[GRP_NM];
#pragma HLS stream variable = tA_data_strm depth = 8
    hls::stream<bool> tA_e_strm;
#pragma HLS stream variable = tA_e_strm depth = 8

    hls::stream<double> accmT_data_strm[GRP_NM];
#pragma HLS stream variable = accmT_data_strm depth = 8
    hls::stream<bool> accmT_e_strm;
#pragma HLS stream variable = accmT_e_strm depth = 8

    treeCluster<CH_NM, GRP_NM>(num_of_class, num_of_term, i_prob_strm, i_data_strm, i_e_strm, addT_level_strm,
                               addT_sample_end_strm, addT_data_strm, addT_e_strm);

    treeAdder<GRP_NM>(addT_level_strm, addT_sample_end_strm, addT_data_strm, addT_e_strm, tA_level_strm,
                      tA_sample_end_strm, tA_data_strm, tA_e_strm);

    accmCluster<CH_NM, GRP_NM>(tA_level_strm, tA_sample_end_strm, tA_data_strm, tA_e_strm, accmT_data_strm,
                               accmT_e_strm, result_vector);

    finalProb<GRP_NM>(num_of_class, accmT_data_strm, accmT_e_strm, o_data_strm, o_e_strm, prior_vector);
}

/**
 * @brief argmaxClassifiler, argmax function
 */
template <int GRP_NM>
void argmaxClassifiler(const int num_of_class,
                       hls::stream<double> i_data_strm[GRP_NM],
                       hls::stream<bool>& i_e_strm,

                       hls::stream<ap_uint<10> >& o_class_strm,
                       hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

    ap_uint<10> cls_cnt = 0;
    double cache_word = maxMinValue<double>::NEGATIVE_INFINITY;
    int offset = 0;

    double prob_line[GRP_NM];
#pragma HLS array_partition variable = prob_line complete dim = 0

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS pipeline II = 6
        last = i_e_strm.read();

        for (int i = 0; i < GRP_NM; i++) {
#pragma HLS unroll
            double tmp = i_data_strm[i].read();
            if (cls_cnt + i < num_of_class) { // read all, but drop them when cls_cnt > num_of_class
                prob_line[i] = tmp;
            } else {
                prob_line[i] = maxMinValue<double>::NEGATIVE_INFINITY;
            }
        }

        int offset_t;
        double maxValue_t = maxValueTree<double, GRP_NM>(prob_line, offset_t, 0);

        if (maxValue_t - cache_word > 1e-8) { // Must be greater than this threshold value
            offset = cls_cnt + offset_t;
        }

        if (cls_cnt + GRP_NM < num_of_class) {
            cache_word = (maxValue_t - cache_word > 1e-8) ? maxValue_t : cache_word;
            cls_cnt += GRP_NM;
        } else {
            ap_uint<10> predict_cls = offset % (1 << 10);
            o_class_strm.write(predict_cls);
            o_e_strm.write(false);

            cache_word = maxMinValue<double>::NEGATIVE_INFINITY;
            cls_cnt = 0;
        }
    }

    o_e_strm.write(true);
}

/**
 * @brief predictCore, the top function of multinomial Naive Bayes Prediction
 */
template <int CH_NM, int GRP_NM>
void predictCore(const int num_of_class,
                 const int num_of_term,
                 hls::stream<ap_uint<32> > i_data_strm[CH_NM],
                 hls::stream<bool>& i_e_strm,

                 hls::stream<ap_uint<10> >& o_class_strm,
                 hls::stream<bool>& o_e_strm,
#ifndef __SYNTHESIS__
                 ap_uint<72>* lh_vector[GRP_NM][256 / GRP_NM],
                 ap_uint<64>* prior_vector[GRP_NM],
                 ap_uint<64>* result_vector[GRP_NM]
#else
                 ap_uint<72> lh_vector[GRP_NM][256 / GRP_NM][4096],
                 ap_uint<64> prior_vector[GRP_NM][1024 / GRP_NM],
                 ap_uint<64> result_vector[GRP_NM][1024 / GRP_NM]
#endif
                 ) {
#pragma HLS inline off
#pragma HLS dataflow

    hls::stream<double> uram_prob_strm[GRP_NM][CH_NM];
#pragma HLS stream variable = uram_prob_strm depth = 32
    hls::stream<ap_uint<32> > uram_data_strm[GRP_NM][CH_NM];
#pragma HLS stream variable = uram_data_strm depth = 32
    hls::stream<bool> uram_e_strm;
#pragma HLS stream variable = uram_e_strm depth = 32

    hls::stream<double> tree_data_strm[GRP_NM];
#pragma HLS stream variable = tree_data_strm depth = 8
    hls::stream<bool> tree_e_strm;
#pragma HLS stream variable = tree_e_strm depth = 8

    uramAccess<CH_NM, GRP_NM>(num_of_class, num_of_term, i_data_strm, i_e_strm, uram_prob_strm, uram_data_strm,
                              uram_e_strm, lh_vector);

    treeProcess<CH_NM, GRP_NM>(num_of_class, num_of_term, uram_prob_strm, uram_data_strm, uram_e_strm, tree_data_strm,
                               tree_e_strm, prior_vector, result_vector);

    argmaxClassifiler<GRP_NM>(num_of_class, tree_data_strm, tree_e_strm, o_class_strm, o_e_strm);
}
}
}
}
}

namespace xf {
namespace data_analytics {
namespace classification {
/**
 * @brief naiveBayesTrain, top function of multinomial Naive Bayes Training.
 *
 * This function will firstly load train dataset from the i_data_strm, then counte the frequency for each hit term.
 * After scaning all sample, the likehood probability matrix and prior probability will be output from two independent
 * stream
 *
 * @tparam DT_WIDTH the width of type DT, in bits
 * @tparam WL the width of bit to enable dispatcher, only 3 is supported so far
 * @tparam DT the data type of internal counter for terms, can be 32/64-bit integer, float or double
 *
 * @param num_of_class the number of class in sample dataset, should be exactly same with real dataset
 * @param num_of_term the number of terms, must be larger than the number of feature, and num_of_class * num_of_term
 * <= (1 << (20-WL)) must be satisfied.
 * @param i_data_strm input data stream of ap_uint<64> in multiple channel
 * @param i_e_strm end flag stream for each input data channel
 * @param o_terms_strm the output number of statistic feature
 * @param o_data0_strm the output likehood matrix
 * @param o_data1_strm the output prior probablity vector
 *
 */
template <int DT_WIDTH = 32, int WL = 3, typename DT = unsigned int>
void naiveBayesTrain(const int num_of_class,
                     const int num_of_term,
                     hls::stream<ap_uint<64> > i_data_strm[1 << WL],
                     hls::stream<bool> i_e_strm[1 << WL],

                     hls::stream<int>& o_terms_strm,
                     hls::stream<ap_uint<64> > o_data0_strm[1 << WL],
                     hls::stream<ap_uint<64> > o_data1_strm[1 << WL]) {
#pragma HLS inline off

    const int PU = (1 << WL);
    const int DEPTH_URAM = (1 << 20) / PU;
    const int DEPTH_BRAM = (1 << 12);

#ifndef __SYNTHESIS__

    ap_uint<72>* lh_vector[PU];
    ap_uint<96>* prior_vector[PU + 1];

    for (int i = 0; i < PU; i++) {
        lh_vector[i] = (ap_uint<72>*)malloc(DEPTH_URAM * sizeof(ap_uint<72>));
        memset(lh_vector[i], 0, DEPTH_URAM * sizeof(ap_uint<72>));
    }

    for (int i = 0; i <= PU; i++) {
        prior_vector[i] = (ap_uint<96>*)malloc(DEPTH_BRAM * sizeof(ap_uint<96>));
        memset(prior_vector[i], 0, DEPTH_BRAM * sizeof(ap_uint<96>));
    }

#else

    ap_uint<72> lh_vector[PU][DEPTH_URAM];
    ap_uint<96> prior_vector[PU + 1][DEPTH_BRAM];
#pragma HLS bind_storage variable = lh_vector type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = lh_vector complete dim = 1
#pragma HLS bind_storage variable = prior_vector type = ram_2p impl = bram
#pragma HLS ARRAY_PARTITION variable = prior_vector complete dim = 1

#endif

    ap_uint<64> term_sum = 0;

    internal::initUram<PU>(num_of_class, num_of_term, lh_vector, prior_vector);

    internal::trainCore<DT, WL, PU>(num_of_class, num_of_term, i_data_strm, i_e_strm, term_sum, lh_vector,
                                    prior_vector);

    internal::trainWriteOut<DT, DT_WIDTH, PU>(num_of_class, num_of_term, term_sum, o_terms_strm, o_data0_strm,
                                              o_data1_strm, lh_vector, prior_vector[PU]);
}

/**
 * @brief naiveBayesPredict, top function of multinomial Naive Bayes Prediction
 *
 * The function will firstly load the train model into on-chip memory, and calculate the classfication results
 * for each sample using argmax function.
 *
 * @tparam CH_NM the number of channel for input sample data, should be power of 2
 * @tparam GRP_NM the unroll factor for handling the classes simultaneously, must be power of 2 in 1~256
 *
 * @param num_of_term the number of class, should be exactly same with the input dataset
 * @param num_of_term the number of feature, should be exactly same with the input dataset
 * @param i_theta_strm the input likehood probability stream, [num_of_class][num_of_term]
 * @param i_prior_strm the input prior probability stream, [num_of_class]
 * @param i_data_strm the input of test data stream
 * @param i_e_strm end flag stream for i_data_strm
 * @param o_class_strm the prediction result for each input sample
 * @param o_e_strm end flag stream for o_class_strm
 */
template <int CH_NM, int GRP_NM>
void naiveBayesPredict(const int num_of_class,
                       const int num_of_term,
                       hls::stream<ap_uint<64> >& i_theta_strm,
                       hls::stream<ap_uint<64> >& i_prior_strm,

                       hls::stream<ap_uint<32> > i_data_strm[CH_NM],
                       hls::stream<bool>& i_e_strm,

                       hls::stream<ap_uint<10> >& o_class_strm,
                       hls::stream<bool>& o_e_strm) {
#pragma HLS inline off

#ifndef __SYNTHESIS__

    ap_uint<72>* lh_vector[GRP_NM][256 / GRP_NM];
    ap_uint<64>* prior_vector[GRP_NM];
    ap_uint<64>* result_vector[GRP_NM];

    for (int i = 0; i < GRP_NM; i++) {
        for (int j = 0; j < 256 / GRP_NM; j++) {
            lh_vector[i][j] = (ap_uint<72>*)malloc(4096 * sizeof(ap_uint<72>));
            memset(lh_vector[i][j], 0, 4096 * sizeof(ap_uint<72>));
        }

        prior_vector[i] = (ap_uint<64>*)malloc(1024 / GRP_NM * sizeof(ap_uint<64>));
        memset(prior_vector[i], 0, 1024 / GRP_NM * sizeof(ap_uint<64>));
        result_vector[i] = (ap_uint<64>*)malloc(1024 / GRP_NM * sizeof(ap_uint<64>));
        memset(result_vector[i], 0, 1024 / GRP_NM * sizeof(ap_uint<64>));
    }

#else

    ap_uint<72> lh_vector[GRP_NM][256 / GRP_NM][4096];
#pragma HLS bind_storage variable = lh_vector type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = lh_vector complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lh_vector complete dim = 2
    ap_uint<64> prior_vector[GRP_NM][1024 / GRP_NM];
#pragma HLS bind_storage variable = prior_vector type = ram_1p impl = lutram
#pragma HLS ARRAY_PARTITION variable = prior_vector complete dim = 1
    ap_uint<64> result_vector[GRP_NM][1024 / GRP_NM];
#pragma HLS bind_storage variable = result_vector type = ram_2p impl = lutram
#pragma HLS array_partition variable = result_vector complete dim = 1

#endif
    internal::loadModel<GRP_NM>(num_of_class, num_of_term, i_theta_strm, i_prior_strm, lh_vector, prior_vector,
                                result_vector);

    internal::predictCore<CH_NM, GRP_NM>(num_of_class, num_of_term, i_data_strm, i_e_strm, o_class_strm, o_e_strm,
                                         lh_vector, prior_vector, result_vector);
}
}
}
}
#endif
