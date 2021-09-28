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

#ifndef _XF_DATABASE_COMPOUND_SORT_HPP_
#define _XF_DATABASE_COMPOUND_SORT_HPP_

#include <ap_int.h>
#include "xf_database/insert_sort.hpp"
#include "xf_database/merge_sort.hpp"

namespace xf {
namespace database {
namespace details {
template <typename KEY_TYPE, int INSERT_LEN>
void genData(hls::stream<KEY_TYPE>& inKeyStrm,
             hls::stream<bool>& inEndStrm,
             int& keyLength,
             hls::stream<KEY_TYPE>& outKeyStrm,
             hls::stream<bool>& outEndStrm) {
    int index = 0;
    bool flag = inEndStrm.read();
    while (!flag) {
#pragma HLS loop_tripcount max = 100000 min = 100000
#pragma HLS pipeline ii = 1
        outKeyStrm.write(inKeyStrm.read());
        outEndStrm.write(false);
        flag = inEndStrm.read();
        index += 1;
    }
    keyLength = index;
#ifndef __SYNTHESIS__
    std::cout << "keyLength=" << keyLength << std::endl;
#endif
    outEndStrm.write(true);
}

template <typename KEY_TYPE, typename uint64>
void insert2Storage(hls::stream<KEY_TYPE>& KeyStrm,
                    hls::stream<bool>& endBlockStrm,
                    hls::stream<bool>& endStrm,
                    uint64* value) {
    int index = 0;
    uint64 tmp = 0;
    while (!endStrm.read()) {
#pragma HLS loop_tripcount max = 100 min = 100
        while (!endBlockStrm.read()) {
#pragma HLS loop_tripcount max = 1000 min = 1000
#pragma HLS pipeline
            if (index % 2) {
                tmp.range(63, 32) = KeyStrm.read();
                value[index / 2] = tmp;
            } else
                tmp.range(31, 0) = KeyStrm.read();
            index++;
        }
    }
    if (index % 2) value[index / 2] = tmp;
}

template <typename KEY_TYPE, int INSERT_LEN, int N>
void channel1toNCore(hls::stream<KEY_TYPE>& inKeyStrm,
                     hls::stream<bool>& inEndStrm,
                     hls::stream<KEY_TYPE> outKeyStrm[N],
                     hls::stream<bool> outEndBlockStrm[N],
                     hls::stream<bool> outEndStrm[N]) {
    int flag = 0;
    bool blockFlag = false;
    int index = 0;
    bool endFlag = inEndStrm.read();
    if (endFlag) flag = -1;
    while (!endFlag) {
#pragma HLS loop_tripcount max = 100000 min = 100000
#pragma HLS pipeline
        if ((index % INSERT_LEN == 0) && blockFlag) {
            outEndBlockStrm[flag].write(true);
            blockFlag = false;
            flag = (flag + 1) % N;
        } else {
            blockFlag = true;
            outKeyStrm[flag].write(inKeyStrm.read());
            outEndBlockStrm[flag].write(false);
            if (index % INSERT_LEN == 0) {
                outEndStrm[flag].write(false);
            }
            index += 1;
            endFlag = inEndStrm.read();
        }
    }
    if (flag >= 0) outEndBlockStrm[flag].write(true);
    for (int i = 0; i < N; i++) {
#pragma HLS pipeline
        if (i > flag) {
            outEndBlockStrm[i].write(true);
            outEndStrm[i].write(false);
        }
        outEndStrm[i].write(true);
    }
}

template <typename KEY_TYPE>
void mergeSortWrapper(hls::stream<KEY_TYPE>& inKeyStrm0,
                      hls::stream<bool>& inEndBlockStrm0,
                      hls::stream<bool>& inEndStrm0,
                      hls::stream<KEY_TYPE>& inKeyStrm1,
                      hls::stream<bool>& inEndBlockStrm1,
                      hls::stream<bool>& inEndStrm1,
                      hls::stream<KEY_TYPE>& outKeyStrm,
                      hls::stream<bool>& outEndBlockStrm,
                      hls::stream<bool>& outEndStrm,
                      bool order) {
    bool flag = false;
    while (!inEndStrm0.read()) {
#pragma HLS loop_tripcount max = 100 min = 100
        flag = inEndStrm1.read();
        outEndStrm.write(false);
        mergeSort<KEY_TYPE>(inKeyStrm0, inEndBlockStrm0, inKeyStrm1, inEndBlockStrm1, outKeyStrm, outEndBlockStrm,
                            order);
    }
    if (!flag) inEndStrm1.read();
    outEndStrm.write(true);
}

template <typename KEY_TYPE, int INSERT_LEN>
void merge1to4Wrapper(hls::stream<KEY_TYPE>& inKeyStrm,
                      hls::stream<bool>& inEndStrm,
                      hls::stream<KEY_TYPE>& outKeyStrm,
                      hls::stream<bool>& outEndBlockStrm,
                      hls::stream<bool>& outEndStrm,
                      bool order) {
#pragma HLS dataflow
    hls::stream<KEY_TYPE> keyStrm0[8];
    hls::stream<bool> endBlockStrm0[8];
    hls::stream<bool> endStrm0[8];
    hls::stream<KEY_TYPE> keyStrm1[4];
    hls::stream<bool> endBlockStrm1[4];
    hls::stream<bool> endStrm1[4];
    hls::stream<KEY_TYPE> keyStrm2[2];
    hls::stream<bool> endBlockStrm2[2];
    hls::stream<bool> endStrm2[2];
#pragma HLS stream variable = keyStrm0 depth = 2048
#pragma HLS stream variable = endBlockStrm0 depth = 2048
#pragma HLS stream variable = endStrm0 depth = 4
#pragma HLS bind_storage variable = endBlockStrm0 type = fifo impl = lutram
#pragma HLS stream variable = keyStrm1 depth = 4
#pragma HLS stream variable = endBlockStrm1 depth = 4
#pragma HLS stream variable = endStrm1 depth = 4
#pragma HLS stream variable = keyStrm2 depth = 4
#pragma HLS stream variable = endBlockStrm2 depth = 4
#pragma HLS stream variable = endStrm2 depth = 4
    channel1toNCore<KEY_TYPE, INSERT_LEN, 8>(inKeyStrm, inEndStrm, keyStrm0, endBlockStrm0, endStrm0);
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        mergeSortWrapper<KEY_TYPE>(keyStrm0[i * 2], endBlockStrm0[i * 2], endStrm0[i * 2], keyStrm0[i * 2 + 1],
                                   endBlockStrm0[i * 2 + 1], endStrm0[i * 2 + 1], keyStrm1[i], endBlockStrm1[i],
                                   endStrm1[i], order);
    }
    for (int i = 0; i < 2; i++) {
#pragma HLS unroll
        mergeSortWrapper<KEY_TYPE>(keyStrm1[i * 2], endBlockStrm1[i * 2], endStrm1[i * 2], keyStrm1[i * 2 + 1],
                                   endBlockStrm1[i * 2 + 1], endStrm1[i * 2 + 1], keyStrm2[i], endBlockStrm2[i],
                                   endStrm2[i], order);
    }

    mergeSortWrapper<KEY_TYPE>(keyStrm2[0], endBlockStrm2[0], endStrm2[0], keyStrm2[1], endBlockStrm2[1], endStrm2[1],
                               outKeyStrm, outEndBlockStrm, outEndStrm, order);
}

/**
 * @brief sortPart1 sort each 4 * INSERT_LEN and store key and data to URAM
 *
 * @tparam KEY_TYPE key type
 * @tparam INSERT_LEN the length of insert sort
 *
 * @param inKeyStrm input key stream
 * @param inEndStrm end flag stream for input key
 * @param order 1:sort ascending 0:sort descending
 * @param value return sort result
 *
 */
template <typename KEY_TYPE, typename uint64, int INSERT_LEN>
void sortPart1(
    hls::stream<KEY_TYPE>& inKeyStrm, hls::stream<bool>& inEndStrm, int& keyLength, bool order, uint64* value) {
#pragma HLS dataflow
    hls::stream<KEY_TYPE> keyStrm0("keyStrm0");
    hls::stream<bool> endBlockStrm0("endBlockStrm0");
    hls::stream<bool> endStrm0("endStrm0");
    hls::stream<KEY_TYPE> keyStrm1;
    hls::stream<bool> endBlockStrm1("endBlockStrm1");
    hls::stream<bool> endStrm1("endStrm1");
    hls::stream<KEY_TYPE> keyStrm2;
    hls::stream<bool> endBlockStrm2("endBlockStrm2");
    hls::stream<bool> endStrm2("endStrm2");
#pragma HLS stream variable = keyStrm0 depth = 8
#pragma HLS stream variable = endBlockStrm0 depth = 8
#pragma HLS stream variable = endStrm0 depth = 8
#pragma HLS stream variable = keyStrm1 depth = 8
#pragma HLS stream variable = endBlockStrm1 depth = 8
#pragma HLS stream variable = endStrm1 depth = 8
#pragma HLS stream variable = keyStrm2 depth = 8
#pragma HLS stream variable = endBlockStrm2 depth = 8
#pragma HLS stream variable = endStrm2 depth = 8
    genData<KEY_TYPE, INSERT_LEN>(inKeyStrm, inEndStrm, keyLength, keyStrm0, endStrm0);
    insertSort<KEY_TYPE, INSERT_LEN>(keyStrm0, endStrm0, keyStrm1, endStrm1, order);
    merge1to4Wrapper<KEY_TYPE, INSERT_LEN>(keyStrm1, endStrm1, keyStrm2, endBlockStrm2, endStrm2, order);
    insert2Storage<KEY_TYPE, uint64>(keyStrm2, endBlockStrm2, endStrm2, value);
}

template <typename KEY_TYPE, typename uint64>
void readArrayCore(int keyLength,
                   uint64* value,
                   int begin_cnt,
                   int block_cnt,
                   hls::stream<KEY_TYPE>& keyStrm,
                   hls::stream<bool>& endStrm) {
    uint64 tmp = 0;
loop_read:
    for (int k = 0; k < block_cnt; k++) {
#pragma HLS pipeline ii = 2
        if (begin_cnt * 2 + k * 2 >= keyLength) break;
        tmp = value[begin_cnt + k];
        keyStrm.write(tmp.range(31, 0));
        endStrm.write(false);
        if (begin_cnt * 2 + k * 2 + 1 >= keyLength) break;
        keyStrm.write(tmp.range(63, 32));
        endStrm.write(false);
    }
    endStrm.write(true);
}

template <typename KEY_TYPE, typename uint64, int K, int BLOCK_LEN>
void readArray2NStream(int keyLength, uint64* value, hls::stream<KEY_TYPE> keyStrm[K], hls::stream<bool> endStrm[K]) {
#pragma HLS dataflow
    for (int i = 0; i < K; i++) {
#pragma HLS unroll
        readArrayCore<KEY_TYPE, uint64>(keyLength, value, i * BLOCK_LEN / 2, BLOCK_LEN / 2, keyStrm[i], endStrm[i]);
    }
}

template <typename KEY_TYPE, int N, int BEGIN_CNT>
void mergeSortN(bool order, hls::stream<KEY_TYPE>* keyStrm, hls::stream<bool>* endStrm) {
#pragma HLS dataflow
    for (int i = 0; i < N / 2; i++) {
#pragma HLS unroll
        mergeSort<KEY_TYPE>(keyStrm[i * 2 + BEGIN_CNT], endStrm[i * 2 + BEGIN_CNT], keyStrm[i * 2 + 1 + BEGIN_CNT],
                            endStrm[i * 2 + 1 + BEGIN_CNT], keyStrm[N + i + BEGIN_CNT], endStrm[N + i + BEGIN_CNT],
                            order);
    }
}

template <typename KEY_TYPE, int N>
void outputStream(hls::stream<KEY_TYPE>* inKeyStrm,
                  hls::stream<bool>* inEndStrm,
                  hls::stream<KEY_TYPE>& outKeyStrm,
                  hls::stream<bool>& outEndStrm) {
    while (!inEndStrm[N].read()) {
#pragma HLS pipeline
#pragma HLS loop_tripcount max = 100000 min = 100000
        outKeyStrm.write(inKeyStrm[N].read());
        outEndStrm.write(false);
    }
    outEndStrm.write(true);
}

/**
 * @brief sortPart2 merge sort the key and data from URAM and output by stream.
 *
 * @tparam KEY_TYPE key type
 * @tparam LEN Maximum support sort length
 * @tparam BLOCK_LEN each block key and data length
 *
 * @param order 1:sort ascending 0:sort descending
 * @param keyLength Actual key length
 * @param value sort key and data from value
 * @param outDataStrm output data stream
 * @param outKeyStrm output key stream
 * @param outEndStrm end flag stream for output
 */
template <typename KEY_TYPE, typename uint64, int LEN, int BLOCK_LEN>
void sortPart2(
    bool order, int keyLength, uint64 value[LEN], hls::stream<KEY_TYPE>& outKeyStrm, hls::stream<bool>& outEndStrm) {
#pragma HLS dataflow
    const int K = LEN / BLOCK_LEN;
    hls::stream<KEY_TYPE> keyStrm[2 * K - 1];
    hls::stream<bool> endStrm[2 * K - 1];
#pragma HLS stream variable = keyStrm depth = 16
#pragma HLS stream variable = endStrm depth = 16

    readArray2NStream<KEY_TYPE, uint64, K, BLOCK_LEN>(keyLength, value, keyStrm, endStrm);
    mergeSortN<KEY_TYPE, K, 0>(order, keyStrm, endStrm);                                 // 16384
    if (K > 2) mergeSortN<KEY_TYPE, K / 2, K>(order, keyStrm, endStrm);                  // 32768
    if (K > 4) mergeSortN<KEY_TYPE, K / 4, K + K / 2>(order, keyStrm, endStrm);          // 65536
    if (K > 8) mergeSortN<KEY_TYPE, K / 8, 2 * K - K / 4>(order, keyStrm, endStrm);      // 131072
    if (K > 16) mergeSortN<KEY_TYPE, K / 16, 2 * K - K / 8>(order, keyStrm, endStrm);    // 262144
    if (K > 32) mergeSortN<KEY_TYPE, K / 32, 2 * K - K / 16>(order, keyStrm, endStrm);   // 524288
    if (K > 64) mergeSortN<KEY_TYPE, K / 64, 2 * K - K / 32>(order, keyStrm, endStrm);   // 1048576
    if (K > 128) mergeSortN<KEY_TYPE, K / 128, 2 * K - K / 64>(order, keyStrm, endStrm); // 2097152
    outputStream<KEY_TYPE, 2 * K - 2>(keyStrm, endStrm, outKeyStrm, outEndStrm);
}
} // details

/**
 * @brief compoundSort sort the key based on insert sort and merge sort.
 *
 * @tparam KEY_TYPE key type
 * @tparam SORT_LEN Maximum support sort length, between 16K to 2M, but it must be an integer power of 2.
 * @tparam INSERT_LEN insert sort length, maximum length 1024 (recommend)
 *
 * @param order 1:sort ascending 0:sort descending
 * @param inKeyStrm input key stream
 * @param inEndStrm end flag stream for input key
 * @param outKeyStrm output key-sorted stream
 * @param outEndStrm end flag stream for output key
 */
template <typename KEY_TYPE, int SORT_LEN, int INSERT_LEN>
void compoundSort(bool order,
                  hls::stream<KEY_TYPE>& inKeyStrm,
                  hls::stream<bool>& inEndStrm,
                  hls::stream<KEY_TYPE>& outKeyStrm,
                  hls::stream<bool>& outEndStrm) {
#ifdef __SYNTHESIS__
    ap_uint<64> value[SORT_LEN / 2];
#else
    ap_uint<64>* value;
    value = new ap_uint<64>[ SORT_LEN / 2 ];
#endif
    int keyLength = 0;
    const int AP = SORT_LEN / INSERT_LEN / 8;
#pragma HLS bind_storage variable = value type = ram_2p impl = uram
#pragma HLS array_partition variable = value dim = 1 block factor = AP
    details::sortPart1<KEY_TYPE, ap_uint<64>, INSERT_LEN>(inKeyStrm, inEndStrm, keyLength, order, value);
    details::sortPart2<KEY_TYPE, ap_uint<64>, SORT_LEN, INSERT_LEN * 8>(order, keyLength, value, outKeyStrm,
                                                                        outEndStrm);

#ifndef __SYNTHESIS__
    delete[] value;
#endif
}

} // database
} // xf
#endif //_KERNEL_SORT_HPP_
