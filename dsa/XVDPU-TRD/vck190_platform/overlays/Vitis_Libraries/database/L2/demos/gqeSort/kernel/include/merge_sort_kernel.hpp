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

#ifndef _XF_DATABASE_MERGE_SORT_KERNEL_HPP_
#define _XF_DATABASE_MERGE_SORT_KERNEL_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "merge_sort_v0.hpp"
#include "stream_to_axi.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#define _DBG_PRINTF(...) void(0)
//#define _DBG_PRINTF(...) fprintf(stderr, __VA_ARGS__)
#endif

namespace xf {
namespace database {
namespace details {

template <typename DT, typename KT>
struct BiPair {
    enum { DW = 8 * sizeof(DT), KW = 8 * sizeof(KT) };
    ap_uint<2 + DW * 2 + KW * 2> v;

    BiPair() {}
    BiPair(const BiPair& o) : v(o.v) {}
    BiPair(ap_uint<DW * 2 + KW * 2> bp, bool odd, bool end) {
        v.range(DW * 2 + KW * 2 - 1, 0) = bp;
        v[DW * 2 + KW * 2] = odd;
        v[DW * 2 + KW * 2 + 1] = end;
    }

    KT key0() { return v.range(KW - 1, 0); }
    DT data0() { return v.range(KW + DW - 1, KW); }
    KT key1() { return v.range(KW * 2 + DW - 1, KW + DW); }
    DT data1() { return v.range(KW * 2 + DW * 2 - 1, KW * 2 + DW); }
    bool odd() { return v[DW * 2 + KW * 2]; }
    bool end() { return v[DW * 2 + KW * 2 + 1]; }
};

// half_strm are used to notify FIFO filing before key or data stream is empty.
template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width, int Ch_Num, int B_Len>
void balancingLoader(ap_uint<2 * (Data_Width + Key_Width)>* inBuff,
                     const unsigned int eachLen,
                     const unsigned int totalLen,
                     hls::stream<BiPair<Data_Type, Key_Type> > packStrm[Ch_Num],
                     hls::stream<bool> halfStrm[Ch_Num]) {
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
    std::cout << "balancingLoader " << std::endl;
#endif
    unsigned int nBlk = totalLen / (eachLen * Ch_Num);

    unsigned int eachCnt[Ch_Num];
#pragma HLS array_partition variable = eachCnt dim = 0
    bool hasFinished[Ch_Num];
#pragma HLS array_partition variable = hasFinished dim = 0
    unsigned int loadNum[Ch_Num];
#pragma HLS array_partition variable = loadNum dim = 0
    unsigned int baseAddr[Ch_Num];
#pragma HLS array_partition variable = baseAddr dim = 0
    unsigned int endBound[Ch_Num];
#pragma HLS array_partition variable = endBound dim = 0

    for (unsigned int c = 0; c < Ch_Num; c++) {
#pragma HLS pipeline II = 1
        hasFinished[c] = false;
        loadNum[c] = 0;
        baseAddr[c] = c * (eachLen / 2);
        if (totalLen > (nBlk * eachLen * Ch_Num + (c + 1) * eachLen)) {
            eachCnt[c] = (nBlk + 1) * eachLen;
        } else if (totalLen > (nBlk * eachLen * Ch_Num + c * eachLen) &&
                   totalLen <= (nBlk * eachLen * Ch_Num + (c + 1) * eachLen)) {
            eachCnt[c] = nBlk * eachLen + (totalLen - nBlk * eachLen * Ch_Num - c * eachLen);
        } else {
            eachCnt[c] = nBlk * eachLen;
        }
        endBound[c] = (c + 1) * (eachLen / 2);
    }

    // synth version, using empty of halfStrm.
    int burstBiPair;
    bool end = false;
    // first feed around without setting halfStrm.
    for (int j = 0; j < Ch_Num; ++j) {
        // burst half amount of the FIFO
        // full burst
        if (((int)totalLen - j * (int)eachLen - (int)loadNum[j]) >= (B_Len * 2)) {
            burstBiPair = B_Len;
            // last burst
        } else {
            burstBiPair = ((int)totalLen - j * (int)eachLen - (int)loadNum[j]) / 2;
            hasFinished[j] = true;
        }
        for (int i = 0; i < burstBiPair; ++i) {
#pragma HLS pipeline II = 1
            ap_uint<2 * (Data_Width + Key_Width)> t = inBuff[baseAddr[j] + i];
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
            std::cout << "key_l " << t.range(Key_Width - 1, 0) << "    end 0" << std::endl;
            std::cout << "key_h " << t.range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width) << "    odd 0"
                      << std::endl;
#endif
            BiPair<Data_Type, Key_Type> bp = {t, 0, 0};
            packStrm[j].write(bp);
        }
        // send end-flag
        if (hasFinished[j]) {
            // total length is odd
            if ((((int)totalLen - j * (int)eachLen - (int)loadNum[j]) % 2) > 0 && burstBiPair >= 0) {
                ap_uint<2 * (Data_Width + Key_Width)> t = inBuff[baseAddr[j] + burstBiPair];
                BiPair<Data_Type, Key_Type> bp = {t, 0, 1};
                packStrm[j].write(bp);
            } else {
                BiPair<Data_Type, Key_Type> bp = {0, 1, 1};
                packStrm[j].write(bp);
            }
        }
        if (burstBiPair > 0) {
            loadNum[j] += burstBiPair * 2;
            baseAddr[j] += burstBiPair;
        }
        // XXX first round, do not lock halfStrm
    }
    end = andTree<Ch_Num>(hasFinished);
    // try load the second half, controlled by halfStrm
    int ch = 0;
    while (!end) {
        // half empty and not finished
        // XXX only `full` signal works fine in data-flow producer side...
        if (!halfStrm[ch].full() && !hasFinished[ch]) {
            // burst half amount of the FIFO
            // full burst
            if ((eachCnt[ch] - loadNum[ch]) >= (B_Len * 2)) {
                burstBiPair = B_Len;
                // last burst
            } else {
                burstBiPair = (eachCnt[ch] - loadNum[ch]) / 2;
                hasFinished[ch] = true;
            }
            // jump to next block when hitting the end boundary of current channel
            if (baseAddr[ch] == endBound[ch]) {
                endBound[ch] += eachLen * Ch_Num / 2;
                baseAddr[ch] = endBound[ch] - eachLen / 2;
            }
            for (int i = 0; i < burstBiPair; ++i) {
#pragma HLS pipeline II = 1
                ap_uint<2 * (Data_Width + Key_Width)> t = inBuff[baseAddr[ch] + i];
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
                std::cout << "key_l " << t.range(Key_Width - 1, 0) << "    end 0" << std::endl;
                std::cout << "key_h " << t.range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width) << "    odd 0"
                          << std::endl;
#endif
                BiPair<Data_Type, Key_Type> bp = {t, 0, 0};
                packStrm[ch].write(bp);
            }
            loadNum[ch] += burstBiPair * 2;
            baseAddr[ch] += burstBiPair;
            // XXX set every B_Len, last feed may not have signal.
            if (burstBiPair == B_Len) {
                halfStrm[ch].write(true);
            }
            if (hasFinished[ch]) {
                // total length is odd
                if (((eachCnt[ch] - loadNum[ch]) % 2) > 0) {
                    ap_uint<2 * (Data_Width + Key_Width)> t = inBuff[baseAddr[ch]];
                    BiPair<Data_Type, Key_Type> bp = {t, 0, 1};
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
                    std::cout << "key_l " << t.range(Key_Width - 1, 0) << "    end 1" << std::endl;
                    std::cout << "key_h " << t.range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width)
                              << "    odd 0" << std::endl;
#endif
                    packStrm[ch].write(bp);
                } else {
                    BiPair<Data_Type, Key_Type> bp = {0, 1, 1};
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
                    std::cout << "key_l 0    end 1" << std::endl;
                    std::cout << "key_h 0    odd 1" << std::endl;
#endif
                    packStrm[ch].write(bp);
                }
            }
            // XXX while-loop exit condition here, may need vitis HLS
            end = andTree<Ch_Num>(hasFinished);
        }
        // move to next anyway
        ch = (ch + 1) % Ch_Num;
    }
    // XXX one more to compensate the first round
    for (int j = 0; j < Ch_Num; ++j) {
        halfStrm[j].write(true);
    }
}

template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width, int Ch_Num, int B_Len>
void downSizer(hls::stream<BiPair<Data_Type, Key_Type> >& packInStrm,
               hls::stream<bool>& halfStrm,
               const unsigned int eachLen,
               const unsigned int totalLen,
               hls::stream<Pair<Data_Type, Key_Type> >& pairStrm) {
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
    std::cout << "downSizer " << std::endl;
#endif
    BiPair<Data_Type, Key_Type> bp;
    bp = packInStrm.read();
    unsigned int endCnt = 0;
    unsigned int endNum = totalLen / (eachLen * Ch_Num) + ((totalLen % (eachLen * Ch_Num)) > 0);

    bool end = bp.end();
    unsigned int loadNum = 1;
    unsigned int loadCnt = 0;
    bool blockEnd = false;
    while (!end) {
#pragma HLS pipeline II = 2
        if (!blockEnd) {
            Pair<Data_Type, Key_Type> dp0 = {bp.data0(), bp.key0(), false};
            Pair<Data_Type, Key_Type> dp1 = {bp.data1(), bp.key1(), false};
            // write
            pairStrm.write(dp0);
            pairStrm.write(dp1);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
            std::cout << "key_l " << bp.key0() << "    end 0" << std::endl;
            std::cout << "key_h " << bp.key1() << "    end 0" << std::endl;
#endif
            loadCnt += 2;
        } else {
            Pair<Data_Type, Key_Type> dp = {0, 0, true};
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
            std::cout << "key_l 0    end 1" << std::endl;
#endif
            pairStrm.write(dp);
            blockEnd = false;
            loadCnt = 0;
            endCnt++;
        }
        if (loadCnt == eachLen) {
            blockEnd = true;
        }
        // next
        if (!blockEnd) {
            bp = packInStrm.read();
            end = bp.end();
            // pull next half
            if (loadNum == (B_Len - 1)) {
                bool pullNext;
                // XXX must be non-blocking to prevent unnecessary check.
                halfStrm.read_nb(pullNext);
                loadNum = 0;
            } else {
                ++loadNum;
            }
        }
    }

    bool pullNext;
    // XXX must be non-blocking to prevent unnecessary check.
    halfStrm.read_nb(pullNext);
    // has 1 more pair left
    if (!bp.odd()) {
        Pair<Data_Type, Key_Type> dp = {bp.data0(), bp.key0(), false};
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
        std::cout << "key_l " << bp.key0() << "    end 0" << std::endl;
#endif
        pairStrm.write(dp);
    }
    // send out rest end flags
    unsigned char endLeft = (unsigned char)(endNum - endCnt);
    for (unsigned int i = 0; i < endLeft; i++) {
        Pair<Data_Type, Key_Type> dp = {0, 0, true};
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
        std::cout << "key_l 0    end 1" << std::endl;
#endif
        pairStrm.write(dp);
    }
}

template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width, int Ch_Num, int B_Len>
void dataDispatcher(ap_uint<2 * (Data_Width + Key_Width)>* inBuff,
                    const unsigned int eachLen,
                    const unsigned int totalLen,
                    hls::stream<Pair<Data_Type, Key_Type> > outPairStrm[Ch_Num]) {
#pragma HLS dataflow
    typedef Pair<Data_Type, Key_Type> PT;

    enum { Part = 2, Buff_Len = B_Len * Part };
#if __cplusplus >= 201103L
    static_assert(Part >= 2, "Part cannot be less than 2");
#endif

    hls::stream<BiPair<Data_Type, Key_Type> > packInStrm[Ch_Num];
#pragma HLS stream variable = packInStrm depth = Buff_Len
#pragma HLS resource variable = packInStrm core = FIFO_BRAM

    // XXX the key here is depth, when it is not full, we known at least B_Len can be read in.
    hls::stream<bool> halfStrm[Ch_Num];
#pragma HLS stream variable = halfStrm depth = 1
//#pragma HLS resource variable = halfStrm core = FIFO_LUTRAM
// XXX: work-around as 1-depth FIFO LUTRAM cannot be correctly synthesized by HLS
#pragma HLS resource variable = halfStrm core = FIFO_SRL

    balancingLoader<Data_Type, Key_Type, Data_Width, Key_Width, Ch_Num, B_Len>(inBuff, eachLen, totalLen, packInStrm,
                                                                               halfStrm);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_SORT_KERNEL_DEBUG__ == 1
    for (int i = 0; i < Ch_Num; i++) {
        std::cout << "packInStrm[" << i << "].size() = " << packInStrm[i].size() << "    halfStrm[" << i
                  << "].size() = " << halfStrm[i].size() << std::endl;
    }
#endif

    for (int i = 0; i < Ch_Num; i++) {
#pragma HLS unroll
        downSizer<Data_Type, Key_Type, Data_Width, Key_Width, Ch_Num, B_Len>(packInStrm[i], halfStrm[i], eachLen,
                                                                             totalLen, outPairStrm[i]);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_SORT_KERNEL_DEBUG__ == 1
        std::cout << "outPairStrm[" << i << "].size() = " << outPairStrm[i].size() << "    halfStrm[" << i
                  << "].size() = " << halfStrm[i].size() << std::endl;
#endif
    }
}

template <typename Data_Type, typename Key_Type, int Ch_Num>
void mergeSortTreeWrapper(hls::stream<Pair<Data_Type, Key_Type> > packInStrm[Ch_Num],
                          unsigned int eachLen,
                          unsigned int totalLen,
                          bool order,
                          hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm) {
    unsigned int rnd = totalLen / (eachLen * Ch_Num) + ((totalLen % (eachLen * Ch_Num)) > 0);

    // internal loop for dealing with long input
    for (unsigned int r = 0; r < rnd; r++) {
#pragma HLS loop_tripcount max = 100 min = 100
        mergeTreeS<Data_Type, Key_Type, Ch_Num>::f(packInStrm, order, packOutStrm);
    }
}

template <typename Data_Type, typename Key_Type, int Ch_Num>
void eatEnd(hls::stream<Pair<Data_Type, Key_Type> >& packInStrm,
            unsigned int eachLen,
            unsigned int totalLen,
            hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm) {
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
    std::cout << "eatEnd " << std::endl;
#endif
    unsigned int rnd = totalLen / (eachLen * Ch_Num) + ((totalLen % (eachLen * Ch_Num)) > 0);
    unsigned int eat = 0;

    while (eat < rnd) {
#pragma HLS pipeline II = 1
        Pair<Data_Type, Key_Type> t = packInStrm.read();
        if (!t.end()) {
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
            std::cout << "key " << t.key() << std::endl;
#endif
            packOutStrm.write(t);
        } else {
            eat++;
        }
    }
    Pair<Data_Type, Key_Type> endFlag = {0, 0, true};
    packOutStrm.write(endFlag);
}

} // namespace details

/**
 * @brief mergeSortKernel Implementation of merge sort tree
 *
 * @tparam Data_Type Type of data
 * @tparam Key_Type Type of key
 * @tparam Ch_Num Number of channels
 * @tparam Data_Width Bit-width of data
 * @tparam Key_Width Bit-width of key
 *
 * @param inBuff Input buffer
 * @param eachLen Length of each channel with sorted inputs
 * @param totalLen Number of keys (along with data) needs to be read in total
 * @param order 1:sort ascending 0:sort descending
 * @param outBuff Output sorted result buffer
 *
 */
template <typename Data_Type, typename Key_Type, int Ch_Num, int B_Len, int Data_Width, int Key_Width>
void mergeSortKernel(ap_uint<2 * (Data_Width + Key_Width)>* inBuff,
                     const unsigned int eachLen,
                     const unsigned int totalLen,
                     const unsigned int order,
                     const unsigned int outBuffOff,
                     ap_uint<2 * (Key_Width + Data_Width)>* outBuff) {
#pragma HLS dataflow

    enum { O_Fifo_Depth = B_Len * 2 };

    bool sign = order & 0x01UL;

    hls::stream<details::Pair<Data_Type, Key_Type> > inStrm[Ch_Num];
#pragma HLS stream variable = inStrm depth = 4
#pragma HLS resource variable = inStrm core = FIFO_LUTRAM

    hls::stream<details::Pair<Data_Type, Key_Type> > sortedStrm;
#pragma HLS stream variable = sortedStrm depth = 4
#pragma HLS resource variable = sortedStrm core = FIFO_LUTRAM

    hls::stream<details::Pair<Data_Type, Key_Type> > outStrm;
#pragma HLS stream variable = outStrm depth = 4
#pragma HLS resource variable = outStrm core = FIFO_LUTRAM

#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
    std::cout << "Merge kernel info:" << std::endl;
    std::cout << "inBuff =" << inBuff << std::endl;
    std::cout << "eachLen =" << eachLen << std::endl;
    std::cout << "totalLen =" << totalLen << std::endl;
    std::cout << "order =" << sign << std::endl;
    std::cout << "outBuff =" << outBuff << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << "key_l = " << inBuff[i].range(Key_Width - 1, 0) << std::endl;
        std::cout << "key_h = " << inBuff[i].range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width) << std::endl;
    }
#endif

    details::dataDispatcher<Data_Type, Key_Type, Data_Width, Key_Width, Ch_Num, B_Len>(inBuff, eachLen, totalLen,
                                                                                       inStrm);

    // cassading expansion of struct in to tree.
    details::mergeSortTreeWrapper<Data_Type, Key_Type, Ch_Num>(inStrm, eachLen, totalLen, sign, sortedStrm);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_SORT_KERNEL_DEBUG__ == 1
    std::cout << "sortedStrm.size() = " << sortedStrm.size() << std::endl;
#endif

    details::eatEnd<Data_Type, Key_Type, Ch_Num>(sortedStrm, eachLen, totalLen, outStrm);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_SORT_KERNEL_DEBUG__ == 1
    std::cout << "outStrm.size() = " << outStrm.size() << std::endl;
#endif

    xf::common::utils_hw::streamToAxi<B_Len>(outStrm, outBuffOff, outBuff);

#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << "key_l = " << outBuff[i].range(Key_Width - 1, 0) << std::endl;
        std::cout << "key_h = " << outBuff[i].range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width)
                  << std::endl;
    }
#endif
}

/**
 * @brief mergeSortKernelDualOut Implementation of merge sort tree with 2 output AXI ports in accordance with AVX2
 * format.
 *
 * @tparam Data_Type Type of data
 * @tparam Key_Type Type of key
 * @tparam Ch_Num Number of channels
 * @tparam Data_Width Bit-width of data
 * @tparam Key_Width Bit-width of key
 *
 * @param inBuff Input buffer
 * @param eachLen Length of each channel with sorted inputs
 * @param totalLen Number of keys (along with data) needs to be read in total
 * @param order 1:sort ascending 0:sort descending
 * @param keyBuff Output sorted key buffer
 * @param dataBuff Output sorted data buffer
 *
 */
template <typename Data_Type, typename Key_Type, int Ch_Num, int B_Len, int Data_Width, int Key_Width>
void mergeSortKernelDualOut(ap_uint<2 * (Data_Width + Key_Width)>* inBuff,
                            const unsigned int eachLen,
                            const unsigned int totalLen,
                            const unsigned int order,
                            ap_uint<2 * Data_Width>* dataBuff,
                            ap_uint<2 * Key_Width>* keyBuff) {
#pragma HLS dataflow

    enum { O_Fifo_Depth = B_Len * 2 };

    bool sign = order & 0x01UL;

    hls::stream<details::Pair<Data_Type, Key_Type> > inStrm[Ch_Num];
#pragma HLS stream variable = inStrm depth = 4
#pragma HLS resource variable = inStrm core = FIFO_LUTRAM

    hls::stream<details::Pair<Data_Type, Key_Type> > sortedStrm;
#pragma HLS stream variable = sortedStrm depth = 4
#pragma HLS resource variable = sortedStrm core = FIFO_LUTRAM

    hls::stream<details::Pair<Data_Type, Key_Type> > outStrm;
#pragma HLS stream variable = outStrm depth = 4
#pragma HLS resource variable = outStrm core = FIFO_LUTRAM

#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
    std::cout << "Merge kernel info:" << std::endl;
    std::cout << "inBuff =" << inBuff << std::endl;
    std::cout << "eachLen =" << eachLen << std::endl;
    std::cout << "totalLen =" << totalLen << std::endl;
    std::cout << "order =" << sign << std::endl;
    std::cout << "dataBuff =" << dataBuff << std::endl;
    std::cout << "keyBuff =" << keyBuff << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << "key_l = " << inBuff[i].range(Key_Width - 1, 0) << std::endl;
        std::cout << "key_h = " << inBuff[i].range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width) << std::endl;
    }
#endif

    details::dataDispatcher<Data_Type, Key_Type, Data_Width, Key_Width, Ch_Num, B_Len>(inBuff, eachLen, totalLen,
                                                                                       inStrm);

    // cassading expansion of struct in to tree.
    details::mergeSortTreeWrapper<Data_Type, Key_Type, Ch_Num>(inStrm, eachLen, totalLen, sign, sortedStrm);

    details::eatEnd<Data_Type, Key_Type, Ch_Num>(sortedStrm, eachLen, totalLen, outStrm);

    xf::common::utils_hw::streamToAxi<B_Len, 2 * (Data_Width + Key_Width), Data_Type, Key_Type>(outStrm, keyBuff,
                                                                                                dataBuff);

#if !defined(__SYNTHESIS__) && __XF_DATABASE_MERGE_KERNEL_HOST_DEBUG__ == 1
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << "key_l = " << keyBuff[i].range(Key_Width - 1, 0) << std::endl;
        std::cout << "key_h = " << keyBuff[i].range(2 * Key_Width - 1, Key_Width) << std::endl;
    }
#endif
}

} // database
} // xf

#endif
