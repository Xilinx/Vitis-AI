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

#ifndef _XF_DATABASE_MERGE_SORT_HPP_
#define _XF_DATABASE_MERGE_SORT_HPP_

#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace database {
namespace details {

template <typename T>
bool selectLeft(const T left, const T right, const bool ascending) {
#pragma HLS inline
    if (ascending) {
        return (left < right);
    } else {
        return (left > right);
    }
}

template <typename DT, typename KT>
struct Pair {
    enum { DW = 8 * sizeof(DT), KW = 8 * sizeof(KT) };
    ap_uint<1 + DW + KW> v;

    void data(DT d) { v.range(DW + KW - 1, KW) = d; }
    void key(KT k) { v.range(KW - 1, 0) = k; }
    void end(bool e) { v[DW + KW] = e; }

    Pair() {}
    Pair(const Pair& o) : v(o.v) {}
    Pair(DT d, KT k, bool e) {
        data(d);
        key(k);
        end(e);
    }

    DT data() { return v.range(DW + KW - 1, KW); };
    KT key() { return v.range(KW - 1, 0); };
    bool end() { return v[DW + KW]; }
};

template <typename Data_Type, typename Key_Type>
void mergeSortPair(hls::stream<Pair<Data_Type, Key_Type> >& left_strm,
                   hls::stream<Pair<Data_Type, Key_Type> >& right_strm,
                   hls::stream<Pair<Data_Type, Key_Type> >& out_strm,
                   bool sign) {
    typedef Pair<Data_Type, Key_Type> PT;

    Key_Type iLeft = 0;
    Key_Type iRight = 0;
    Key_Type Result = 0;
    Data_Type dLeft = 0;
    Data_Type dRight = 0;
    Data_Type dResult = 0;

    bool left_strm_end = 0;
    bool right_strm_end = 0;

    // control output the last element
    bool output_last_left = 0;
    bool output_last_right = 0;

    // read 1st left key
    PT ldp = left_strm.read();
    left_strm_end = ldp.end();
    if (!left_strm_end) {
        iLeft = ldp.key();
        dLeft = ldp.data();
    } else {
        output_last_left = 1;
    }

    // read 1st right key
    PT rdp = right_strm.read();
    right_strm_end = rdp.end();
    if (!right_strm_end) {
        iRight = rdp.key();
        dRight = rdp.data();
    } else {
        output_last_right = 1;
    }

merge_loop:
    while (!left_strm_end || !right_strm_end) {
#pragma HLS PIPELINE II = 1

        bool sl = selectLeft(iLeft, iRight, sign);
        if ((sl && !left_strm_end) || (!sl && right_strm_end && output_last_right && !left_strm_end)) {
            // output left && read next left
            Result = iLeft;
            dResult = dLeft;

            PT dp = left_strm.read();
            left_strm_end = dp.end();
            if (!left_strm_end) {
                iLeft = dp.key();
                dLeft = dp.data();
            } else {
                output_last_left = true;
            }
        } else if ((sl && left_strm_end && output_last_left && !right_strm_end) || (!sl && !right_strm_end)) {
            // output right && read risidual right
            Result = iRight;
            dResult = dRight;

            PT dp = right_strm.read();
            right_strm_end = dp.end();
            if (!right_strm_end) {
                iRight = dp.key();
                dRight = dp.data();
            } else {
                output_last_right = true;
            }
        } else if (sl && left_strm_end && !output_last_left && !right_strm_end) {
            // output last left && no read
            Result = iLeft;
            dResult = dLeft;
            output_last_left = true;
        } else if (!sl && right_strm_end && !output_last_right && !left_strm_end) {
            // output last right && no read
            Result = iRight;
            dResult = dRight;
            output_last_right = true;
        } else {
            // left_strm_end && right_strm_end
            // no operation, break loop
        }
        out_strm.write({dResult, Result, 0});
    }

    // output cached element
    if (selectLeft(iLeft, iRight, sign)) {
        // whether output last left
        if (!output_last_left) {
            out_strm.write({dLeft, iLeft, 0});
        }
        // whether output last right
        if (!output_last_right) {
            out_strm.write({dRight, iRight, 0});
        }
    } else {
        // whether output last right
        if (!output_last_right) {
            out_strm.write({dRight, iRight, 0});
        }
        // whether output last left
        if (!output_last_left) {
            out_strm.write({dLeft, iLeft, 0});
        }
    }

    // bye!
    out_strm.write({0, 0, 1});
}

template <int _n>
inline bool andTree(bool flag[], int _o = 0) {
#pragma HLS inline
    return andTree<_n / 2>(flag, _o) && andTree<_n / 2>(flag, _o + (_n / 2));
}

template <>
inline bool andTree<2>(bool flag[], int _o) {
#pragma HLS inline
    return flag[_o] && flag[_o + 1];
}

// helper class for template-expansion of the tree
template <typename Data_Type, typename Key_Type, int Ch_Num>
struct mergeTreeS {
    static void f(hls::stream<Pair<Data_Type, Key_Type> > inStrm[Ch_Num],
                  bool order,
                  hls::stream<Pair<Data_Type, Key_Type> >& outStrm) {
#pragma HLS inline

#pragma HLS dataflow

        hls::stream<Pair<Data_Type, Key_Type> > left_strm;
#pragma HLS stream variable = left_strm depth = 4

        hls::stream<Pair<Data_Type, Key_Type> > right_strm;
#pragma HLS stream variable = right_strm depth = 4

        mergeTreeS<Data_Type, Key_Type, Ch_Num / 2>::f(inStrm, order, left_strm);
        mergeTreeS<Data_Type, Key_Type, Ch_Num / 2>::f(&inStrm[Ch_Num / 2], order, right_strm);

        mergeSortPair<Data_Type, Key_Type>(left_strm, right_strm, outStrm, order);
    }
};

// specialized helper class for template-expansion of the tree
template <typename Data_Type, typename Key_Type>
struct mergeTreeS<Data_Type, Key_Type, 2> {
    static void f(hls::stream<Pair<Data_Type, Key_Type> > inStrm[2],
                  bool order,
                  hls::stream<Pair<Data_Type, Key_Type> >& outStrm) {
        mergeSortPair<Data_Type, Key_Type>(inStrm[0], inStrm[1], outStrm, order);
    }
};
} // namespace details
} // namespace database
} // namespace xf

#endif
