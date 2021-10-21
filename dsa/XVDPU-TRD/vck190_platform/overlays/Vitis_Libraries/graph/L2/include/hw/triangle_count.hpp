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
 * @file triangle_count.hpp
 *
 */

#ifndef __XF_GRAPH_TRIANGLE_COUNT_HPP_
#define __XF_GRAPH_TRIANGLE_COUNT_HPP_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
#include "iostream"
#endif

namespace xf {
namespace graph {
namespace internal {
namespace triangle_count {
/**
 * @brief orderStrmInterNum the intersection number of ordered (ascending)
 * streams
 */
template <typename DT>
void orderStrmInterNum(hls::stream<DT>& value1Strm,
                       hls::stream<bool>& end1Strm,
                       hls::stream<DT>& value2Strm,
                       hls::stream<bool>& end2Strm,
                       hls::stream<int>& interNum) {
    bool end1 = end1Strm.read();
    bool end2 = end2Strm.read();
    DT value1 = 0, value2 = 0;
    int num = 0;
    while (!end1 || !end2) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 20 max = 20
        if (value1 == value2) {
            if (!end1) {
                value1 = value1Strm.read();
                end1 = end1Strm.read();
            }
            if (!end2) {
                value2 = value2Strm.read();
                end2 = end2Strm.read();
            }
        } else if (value1 > value2) {
            if (!end2) {
                value2 = value2Strm.read();
                end2 = end2Strm.read();
            } else {
                value1 = value1Strm.read();
                end1 = end1Strm.read();
            }
        } else {
            if (!end1) {
                value1 = value1Strm.read();
                end1 = end1Strm.read();
            } else {
                value2 = value2Strm.read();
                end2 = end2Strm.read();
            }
        }
        if (value1 == value2) num++;
#ifndef __SYNTHESIS__
//	std::cout << "value1=" << value1 << ",value2=" << value2 << ",num=" << num << std::endl;
#endif
    }
    interNum.write(num);
#ifndef __SYNTHESIS__
// std::cout << "single triangle count = " << num << std::endl;
#endif
}

template <typename uint512>
void burstRead2Strm(int len, uint512* inArr, hls::stream<uint512>& outStrm) {
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS pipeline ii = 1
        outStrm.write(inArr[i]);
    }
}

template <typename DT, typename uint512, int T, int W>
void Arr2Strm(
    int len, int beginAddr, hls::stream<uint512>& inStrm, hls::stream<DT>& outStrm, hls::stream<bool>& endStrm) {
    uint512 tmp;
    for (int i = beginAddr; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 10000 max = 10000
        if (i % T == 0 || i == beginAddr) {
            tmp = inStrm.read();
        }
        DT val = tmp.range(i % T * W + W - 1, i % T * W);
        outStrm.write(val);
        endStrm.write(false);
    }
    endStrm.write(true);
}

template <typename DT>
void coreControlImpl(hls::stream<DT>& offset1Strm,
                     hls::stream<bool>& offset1EndStrm,
                     hls::stream<DT>& row1Strm,
                     hls::stream<bool>& row1EndStrm,
                     hls::stream<DT>& copyNumStrm,
                     hls::stream<DT>& row1OutStrm,
                     hls::stream<bool>& row1OutBlockEndStrm,
                     hls::stream<bool>& row1OutEndStrm) {
    DT begin = 0;
    while (!offset1EndStrm.read()) {
#pragma HLS loop_tripcount min = 1000 max = 1000
        DT end = offset1Strm.read();
        row1OutEndStrm.write(false);
        copyNumStrm.write(end - begin);
        for (int i = begin; i < end; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline ii = 1
            DT row1 = row1Strm.read();
            row1EndStrm.read();
            row1OutStrm.write(row1);
            row1OutBlockEndStrm.write(false);
        }
        if (begin != end) {
            row1OutBlockEndStrm.write(true);
        }
        begin = end;
    }
    row1EndStrm.read();
    row1OutEndStrm.write(true);
}

template <typename DT, int ML>
void row1CopyImpl(hls::stream<DT>& col1Strm,
                  hls::stream<bool>& col1BlockEndStrm,
                  hls::stream<bool>& col1EndStrm,
                  hls::stream<DT>& copyNumStrm,
                  hls::stream<DT>& row1Strm,
                  hls::stream<bool>& row1BlockEndStrm,
                  hls::stream<bool>& row1EndStrm) {
    while (!col1EndStrm.read()) {
#pragma HLS loop_tripcount min = 1000 max = 1000
        DT copyNum = copyNumStrm.read();
        int cnt = 0;
        for (int i = 0; i < copyNum; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
            if (copyNum > 1) row1EndStrm.write(false);
#ifndef __SYNTHESIS__
            static DT col1Arr[ML];
#else
            DT col1Arr[ML];
#pragma HLS resource variable = col1Arr core = RAM_1P_URAM
#endif
            if (i == 0) {
                while (!col1BlockEndStrm.read()) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 10 max = 10
                    DT col1 = col1Strm.read();
                    col1Arr[cnt++] = col1;
                    if (cnt > 1 && copyNum > 1) {
                        row1BlockEndStrm.write(false);
                        row1Strm.write(col1);
                    }
                }
            } else {
                if (i + 1 < copyNum) {
                    for (int j = i + 1; j < cnt; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 10 max = 10
                        DT col1 = col1Arr[j];
                        row1BlockEndStrm.write(false);
                        row1Strm.write(col1Arr[j]);
                    }
                }
            }
            if (copyNum > 1) {
                row1BlockEndStrm.write(true);
            }
        }
    }
    row1EndStrm.write(true);
}

template <typename DT>
void offset2NoOrderReadPart1(hls::stream<DT>& offset1Strm,
                             hls::stream<bool>& offset1EndStrm,
                             hls::stream<DT>& row1Strm,
                             hls::stream<bool>& row1EndStrm,
                             hls::stream<DT>& row1ReduceStrm,
                             hls::stream<bool>& row1ReduceEndStrm) {
    DT begin = 0;
    while (!offset1EndStrm.read()) {
#pragma HLS loop_tripcount min = 1000 max = 1000
        DT end = offset1Strm.read();
        for (int i = begin; i < end; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline ii = 1
            DT row1 = row1Strm.read();
            row1EndStrm.read();
            if (begin + 1 < end) {
                row1ReduceStrm.write(row1);
                row1ReduceEndStrm.write(false);
            }
        }
        begin = end;
    }
    row1EndStrm.read();
    row1ReduceEndStrm.write(true);
}

template <typename DT, typename DT2, typename uint512, int T, int W>
void offset2NoOrderReadPart2(uint512* offset2Arr,
                             hls::stream<DT>& row1ReduceStrm,
                             hls::stream<bool>& row1ReduceEndStrm,
                             hls::stream<DT2>& offset2Strm,
                             hls::stream<bool>& offset2EndStrm) {
    while (!row1ReduceEndStrm.read()) {
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS pipeline ii = 1
        DT row1 = row1ReduceStrm.read();
        int index = row1 / T;
        int index1 = row1 % T;
        DT2 val = offset2Arr[index].range((index1 + 1) * W - 1, index1 * W);
        offset2Strm.write(val);
        offset2EndStrm.write(false);
    }
    offset2EndStrm.write(true);
}

template <typename DT, typename DT2, typename uint512, int T, int W>
void coreControlImpl2(uint512* offset2Arr,
                      hls::stream<DT>& offset1Strm,
                      hls::stream<bool>& offset1EndStrm,
                      hls::stream<DT>& row1Strm,
                      hls::stream<bool>& row1EndStrm,
                      hls::stream<DT2>& offset2Strm,
                      hls::stream<bool>& offset2EndStrm) {
#pragma HLS dataflow
    hls::stream<DT> row1ReduceStrm("row1ReduceStrm");
#pragma HLS stream variable = row1ReduceStrm depth = 512

    hls::stream<bool> row1ReduceEndStrm("row1ReduceEndStrm");
#pragma HLS stream variable = row1ReduceEndStrm depth = 512
#pragma HLS resource variable = row1ReduceEndStrm core = FIFO_LUTRAM
    offset2NoOrderReadPart1<DT>(offset1Strm, offset1EndStrm, row1Strm, row1EndStrm, row1ReduceStrm, row1ReduceEndStrm);
    offset2NoOrderReadPart2<DT, DT2, uint512, T, W>(offset2Arr, row1ReduceStrm, row1ReduceEndStrm, offset2Strm,
                                                    offset2EndStrm);
}

template <typename DT, typename DT2, int W>
void row2ImplPart1(hls::stream<DT2>& offset2Strm,
                   hls::stream<bool>& offset2EndStrm,
                   hls::stream<DT>& row2AddrStrm,
                   hls::stream<bool>& row2AddrEndStrm,
                   hls::stream<DT>& row2DistStrm,
                   hls::stream<bool>& row2DistEndStrm) {
    while (!offset2EndStrm.read()) {
#pragma HLS loop_tripcount min = 10000 max = 10000
        DT2 offset2 = offset2Strm.read();
        DT begin = offset2.range(W - 1, 0);
        DT end = offset2.range(2 * W - 1, W);
        row2DistStrm.write(end - begin);
        row2DistEndStrm.write(false);
        for (DT i = begin; i < end; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline ii = 1
            row2AddrStrm.write(i);
            row2AddrEndStrm.write(false);
        }
    }
    row2AddrEndStrm.write(true);
    row2DistEndStrm.write(true);
}

template <typename DT, typename DT2, typename uint512, int T, int W>
void row2ImplPart2(uint512* row2Arr,
                   hls::stream<DT>& row2AddrStrm,
                   hls::stream<bool>& row2AddrEndStrm,
                   hls::stream<DT>& row2ContinueStrm) {
    int last_i = -1;
    uint512 last_v;
    bool flag;
    flag = row2AddrEndStrm.read();
    bool rd_flag_success = 1;
    while (!flag) {
#pragma HLS loop_tripcount min = 100000 max = 100000
#pragma HLS pipeline ii = 1
        if (rd_flag_success) {
            DT addr = row2AddrStrm.read();
            int index = addr / T;
            int offset = addr % T;
            if (index != last_i) {
                last_v = row2Arr[index];
                last_i = index;
            }
            row2ContinueStrm.write(last_v.range(W * (offset + 1) - 1, W * offset));
        }
        rd_flag_success = row2AddrEndStrm.read_nb(flag);
    }
}

template <typename DT>
void row2ImplPart3(hls::stream<DT>& row2ContinueStrm,
                   hls::stream<DT>& row2DistStrm,
                   hls::stream<bool>& row2DistEndStrm,
                   hls::stream<DT>& row2Strm,
                   hls::stream<bool>& row2BlockEndStrm) {
    while (!row2DistEndStrm.read()) {
#pragma HLS loop_tripcount min = 10000 max = 10000
        DT distance = row2DistStrm.read();
        for (int i = 0; i < distance; i++) {
#pragma HLS loop_tripcount min = 10 max = 10
#pragma HLS pipeline ii = 1
            row2BlockEndStrm.write(false);
            row2Strm.write(row2ContinueStrm.read());
        }
        row2BlockEndStrm.write(true);
    }
}

template <typename DT, typename DT2, typename uint512, int T, int W>
void row2Impl(uint512* row2Arr,
              hls::stream<DT2>& offset2Strm,
              hls::stream<bool>& offset2EndStrm,
              hls::stream<DT>& row2Strm,
              hls::stream<bool>& row2BlockEndStrm) {
#pragma HLS dataflow
    hls::stream<DT> row2AddrStrm("row2AddrStrm");
    hls::stream<bool> row2AddrEndStrm("row2AddrEndStrm");
    hls::stream<DT> row2DistStrm("row2DistStrm");
    hls::stream<bool> row2DistEndStrm("row2DistEndStrm");
    hls::stream<DT> row2ContinueStrm("row2ContinueStrm");
#pragma HLS stream variable = row2AddrStrm depth = 512
#pragma HLS stream variable = row2AddrEndStrm depth = 512
#pragma HLS resource variable = row2AddrEndStrm core = FIFO_LUTRAM
#pragma HLS stream variable = row2DistStrm depth = 512
#pragma HLS stream variable = row2DistEndStrm depth = 512
#pragma HLS resource variable = row2DistEndStrm core = FIFO_LUTRAM
#pragma HLS stream variable = row2ContinueStrm depth = 512
    row2ImplPart1<DT, DT2, W>(offset2Strm, offset2EndStrm, row2AddrStrm, row2AddrEndStrm, row2DistStrm,
                              row2DistEndStrm);
    row2ImplPart2<DT, DT2, uint512, T, W>(row2Arr, row2AddrStrm, row2AddrEndStrm, row2ContinueStrm);
    row2ImplPart3<DT>(row2ContinueStrm, row2DistStrm, row2DistEndStrm, row2Strm, row2BlockEndStrm);
}

template <typename DT>
void mergeImpl(hls::stream<DT>& row1Strm,
               hls::stream<bool>& row1BlockEndStrm,
               hls::stream<bool>& row1EndStrm,
               hls::stream<DT>& row2Strm,
               hls::stream<bool>& row2BlockEndStrm,
               hls::stream<int>& tcStrm,
               hls::stream<bool>& tcEndStrm) {
    while (!row1EndStrm.read()) {
#pragma HLS loop_tripcount min = 10000 max = 10000
        orderStrmInterNum<DT>(row1Strm, row1BlockEndStrm, row2Strm, row2BlockEndStrm, tcStrm);
        tcEndStrm.write(false);
    }
    tcEndStrm.write(true);
}

template <class DT>
void tcAccUnit(hls::stream<int>& tcStrm, hls::stream<bool>& tcEndStrm, uint64_t* triangles) {
    while (!tcEndStrm.read()) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 10000 max = 10000
        triangles[0] += tcStrm.read();
    }
}

template <typename DT, typename uint512, int T, int W>
void burstWrite2Buf(int len, hls::stream<uint512>& buf1, uint512* buf2) {
    ap_uint<1024> tmpIn;
    uint512 tmpOut;
    tmpIn.range(1023, 512) = buf1.read();
    for (int i = 0; i < (len - 1 + T / 2) / (T / 2); i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 1000 max = 1000
        if ((i % 2 == 1) && ((i / 2 + 1) * T) < len) {
            tmpIn.range(1023, 512) = buf1.read();
        } else {
            tmpIn.range(511, 0) = tmpIn.range(1023, 512);
        }
        for (int j = 0; j < 8; j++) {
            tmpOut.range(W * 2 * (j + 1) - 1, W * 2 * j) =
                tmpIn.range(W * j + 2 * W - 1 + (i % 2) * 256, W * j + (i % 2) * 256);
        }
        buf2[i] = tmpOut;
    }
}
} // namespace triangle_count
} // namespace internal

inline void preProcessData(int len, ap_uint<512>* buf1, ap_uint<512>* buf2) {
#pragma HLS dataflow
    hls::stream<ap_uint<512> > buf1Strm;
#pragma HLS stream variable = buf1Strm depth = 16
#pragma HLS resource variable = buf1Strm core = FIFO_LUTRAM

    internal::triangle_count::burstRead2Strm<ap_uint<512> >((len - 1 + 16) / 16, buf1, buf1Strm);
    internal::triangle_count::burstWrite2Buf<ap_uint<32>, ap_uint<512>, 16, 32>(len, buf1Strm, buf2);
}

/**
 * @brief triangleCount the triangle counting algorithm is implemented, the input is the matrix in CSC format.
 *
 * @tparam LEN the depth of stream
 * @tparam ML URAM depth in the design
 *
 * @param numVertex length of column offsets
 * @param numEdge length of row indices
 * @param offset0 column offsets (begin+end) value
 * @param index0 row indices value
 * @param offset1 column offsets (begin+end) value
 * @param index1 row indices value
 * @param offset2 8 column offsets (begin+end) values
 * @param index2 16 row indices values
 * @param triangles return triangle number
 *
 */
template <int LEN, int ML>
void triangleCount(int numVertex,
                   int numEdge,
                   ap_uint<512>* offset0,
                   ap_uint<512>* index0,
                   ap_uint<512>* offset1,
                   ap_uint<512>* index1,
                   ap_uint<512>* offset2,
                   ap_uint<512>* index2,
                   uint64_t* triangles) {
#pragma HLS dataflow
    const int T = 512 / 32;

    hls::stream<ap_uint<512> > offset1BufStrm;
#pragma HLS stream variable = offset1BufStrm depth = 16
#pragma HLS resource variable = offset1BufStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<512> > offset1Buf2Strm;
#pragma HLS stream variable = offset1Buf2Strm depth = 16
#pragma HLS resource variable = offset1Buf2Strm core = FIFO_LUTRAM
    hls::stream<ap_uint<512> > row1BufStrm;
#pragma HLS stream variable = row1BufStrm depth = 16
#pragma HLS resource variable = row1BufStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<512> > row1Buf2Strm;
#pragma HLS stream variable = row1Buf2Strm depth = 16
#pragma HLS resource variable = row1Buf2Strm core = FIFO_LUTRAM
    hls::stream<ap_uint<32> > offset1Strm("offset1Strm");
#pragma HLS stream variable = offset1Strm depth = LEN
#pragma HLS resource variable = offset1Strm core = FIFO_LUTRAM
    hls::stream<bool> offset1EndStrm("offset1EndStrm");
#pragma HLS stream variable = offset1EndStrm depth = LEN
#pragma HLS resource variable = offset1EndStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<32> > row1Strm("row1Strm");
#pragma HLS stream variable = row1Strm depth = LEN
#pragma HLS resource variable = row1Strm core = FIFO_LUTRAM
    hls::stream<bool> row1EndStrm("row1EndStrm");
#pragma HLS stream variable = row1EndStrm depth = LEN
#pragma HLS resource variable = row1EndStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<32> > offset1Strm2("offset1Strm2");
#pragma HLS stream variable = offset1Strm2 depth = LEN
#pragma HLS resource variable = offset1Strm2 core = FIFO_LUTRAM
    hls::stream<bool> offset1EndStrm2("offset1EndStrm2");
#pragma HLS stream variable = offset1EndStrm2 depth = LEN
#pragma HLS resource variable = offset1EndStrm2 core = FIFO_LUTRAM
    hls::stream<ap_uint<32> > row1Strm2("row1Strm2");
#pragma HLS stream variable = row1Strm2 depth = LEN
#pragma HLS resource variable = row1Strm2 core = FIFO_LUTRAM
    hls::stream<bool> row1EndStrm2("row1EndStrm2");
#pragma HLS stream variable = row1EndStrm2 depth = LEN
#pragma HLS resource variable = row1EndStrm2 core = FIFO_LUTRAM

    hls::stream<ap_uint<64> > offset2Strm("offset2Strm");
#pragma HLS stream variable = offset2Strm depth = 1024
    hls::stream<bool> offset2EndStrm("offset2EndStrm");
#pragma HLS stream variable = offset2EndStrm depth = 2048
#pragma HLS resource variable = offset2EndStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<32> > copyNumStrm("copyNumStrm");
#pragma HLS stream variable = copyNumStrm depth = LEN
#pragma HLS resource variable = copyNumStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<32> > col1Strm("col1Strm");
#pragma HLS stream variable = col1Strm depth = LEN
#pragma HLS resource variable = col1Strm core = FIFO_LUTRAM
    hls::stream<bool> col1BlockEndStrm("col1BlockEndStrm");
#pragma HLS stream variable = col1BlockEndStrm depth = LEN
#pragma HLS resource variable = col1BlockEndStrm core = FIFO_LUTRAM
    hls::stream<bool> col1EndStrm("col1EndStrm");
#pragma HLS stream variable = col1EndStrm depth = LEN
#pragma HLS resource variable = col1EndStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<32> > col1CopyStrm("col1CopyStrm");
#pragma HLS stream variable = col1CopyStrm depth = 2048
    hls::stream<bool> col1CopyBlockEndStrm("col1CopyBlockEndStrm");
#pragma HLS stream variable = col1CopyBlockEndStrm depth = 4096
#pragma HLS resource variable = col1CopyBlockEndStrm core = FIFO_LUTRAM
    hls::stream<bool> col1CopyEndStrm("col1CopyEndStrm");
#pragma HLS stream variable = col1CopyEndStrm depth = 1024
#pragma HLS resource variable = col1CopyEndStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<32> > row2Strm("row2Strm");
#pragma HLS stream variable = row2Strm depth = 2048
    hls::stream<bool> row2BlockEndStrm("row2BlockEndStrm");
#pragma HLS stream variable = row2BlockEndStrm depth = 4096
#pragma HLS resource variable = row2BlockEndStrm core = FIFO_LUTRAM
    hls::stream<int> tcStrm("tcStrm");
#pragma HLS stream variable = tcStrm depth = LEN
#pragma HLS resource variable = tcStrm core = FIFO_LUTRAM
    hls::stream<bool> tcEndStrm("tcEndStrm");
#pragma HLS stream variable = tcEndStrm depth = LEN
#pragma HLS resource variable = tcEndStrm core = FIFO_LUTRAM

#ifndef __SYNTHESIS__
    std::cout << "offset Arr2Strm" << std::endl;
#endif
    int numVertex16 = (numVertex + T - 1) / T;
    int numEdge16 = (numEdge + T - 1) / T;
    internal::triangle_count::burstRead2Strm<ap_uint<512> >(numVertex16, offset0, offset1BufStrm);
    internal::triangle_count::Arr2Strm<ap_uint<32>, ap_uint<512>, T, 32>(numVertex, 1, offset1BufStrm, offset1Strm,
                                                                         offset1EndStrm);

#ifndef __SYNTHESIS__
    std::cout << "row Arr2Strm" << std::endl;
#endif
    internal::triangle_count::burstRead2Strm<ap_uint<512> >(numEdge16, index0, row1BufStrm);
    internal::triangle_count::Arr2Strm<ap_uint<32>, ap_uint<512>, T, 32>(numEdge, 0, row1BufStrm, row1Strm,
                                                                         row1EndStrm);

#ifndef __SYNTHESIS__
    std::cout << "offset Arr2Strm2" << std::endl;
#endif
    internal::triangle_count::burstRead2Strm<ap_uint<512> >(numVertex16, offset1, offset1Buf2Strm);
    internal::triangle_count::Arr2Strm<ap_uint<32>, ap_uint<512>, T, 32>(numVertex, 1, offset1Buf2Strm, offset1Strm2,
                                                                         offset1EndStrm2);

#ifndef __SYNTHESIS__
    std::cout << "row Arr2Strm2" << std::endl;
#endif
    internal::triangle_count::burstRead2Strm<ap_uint<512> >(numEdge16, index1, row1Buf2Strm);
    internal::triangle_count::Arr2Strm<ap_uint<32>, ap_uint<512>, T, 32>(numEdge, 0, row1Buf2Strm, row1Strm2,
                                                                         row1EndStrm2);

#ifndef __SYNTHESIS__
    std::cout << "coreControlImpl" << std::endl;
#endif
    internal::triangle_count::coreControlImpl<ap_uint<32> >(offset1Strm, offset1EndStrm, row1Strm, row1EndStrm,
                                                            copyNumStrm, col1Strm, col1BlockEndStrm, col1EndStrm);

#ifndef __SYNTHESIS__
    std::cout << "coreControlImpl2" << std::endl;
#endif
    internal::triangle_count::coreControlImpl2<ap_uint<32>, ap_uint<2 * 32>, ap_uint<512>, T / 2, 32 * 2>(
        offset2, offset1Strm2, offset1EndStrm2, row1Strm2, row1EndStrm2, offset2Strm, offset2EndStrm);
#ifndef __SYNTHESIS__
    std::cout << "row1CopyImpl" << std::endl;
#endif
    internal::triangle_count::row1CopyImpl<ap_uint<32>, ML>(col1Strm, col1BlockEndStrm, col1EndStrm, copyNumStrm,
                                                            col1CopyStrm, col1CopyBlockEndStrm, col1CopyEndStrm);

#ifndef __SYNTHESIS__
    std::cout << "row2Impl" << std::endl;
#endif
    internal::triangle_count::row2Impl<ap_uint<32>, ap_uint<2 * 32>, ap_uint<512>, T, 32>(
        index2, offset2Strm, offset2EndStrm, row2Strm, row2BlockEndStrm);
#ifndef __SYNTHESIS__
    std::cout << "mergeImpl" << std::endl;
#endif
    internal::triangle_count::mergeImpl<ap_uint<32> >(col1CopyStrm, col1CopyBlockEndStrm, col1CopyEndStrm, row2Strm,
                                                      row2BlockEndStrm, tcStrm, tcEndStrm);

#ifndef __SYNTHESIS__
    std::cout << "tcAccUnit" << std::endl;
#endif

    uint64_t TC[1] = {0};
    internal::triangle_count::tcAccUnit<void>(tcStrm, tcEndStrm, TC);
    triangles[0] = TC[0];
}
} // namespace graph
} // namespace xf
#endif
