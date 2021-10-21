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
#ifndef _XF_DATABASE_INSERT_SORT_KERNEL_HPP_
#define _XF_DATABASE_INSERT_SORT_KERNEL_HPP_

#include "insert_sort_v0.hpp"
#include "merge_sort_v0.hpp"
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace database {
namespace details {

template <typename Data_Type, typename Key_Type, int Sort_Len, int Data_Width, int Key_Width, int Burst_Len>
void readIn(ap_uint<2 * (Data_Width + Key_Width)>* inBuff,
            unsigned int totalLength,
            hls::stream<ap_uint<2 * (Data_Width + Key_Width)> >& packBiStrm) {
    ap_uint<Key_Width> nBursts = totalLength / (Burst_Len * 2);
    unsigned int loadNum = 0;

    // full burst
    for (unsigned int n = 0; n < nBursts; n++) {
        for (unsigned int i = 0; i < Burst_Len; i++) {
#pragma HLS pipeline II = 1
            ap_uint<2 * (Data_Width + Key_Width)> biPair = inBuff[loadNum++];
            packBiStrm.write(biPair);
        }
    }

    // deal with the left data
    if ((totalLength % (Burst_Len * 2)) > 0) {
        unsigned int burstLen = ((totalLength % (Burst_Len * 2)) + 1) / 2;
        for (unsigned int i = 0; i < burstLen; i++) {
#pragma HLS pipeline II = 1
            ap_uint<2 * (Data_Width + Key_Width)> biPair = inBuff[loadNum++];
            packBiStrm.write(biPair);
        }
    }
}

template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width>
void downSizer(hls::stream<ap_uint<2 * (Data_Width + Key_Width)> >& packBiInStrm,
               unsigned int totalLength,
               hls::stream<Pair<Data_Type, Key_Type> >& packStrm) {
    unsigned int num = totalLength / 2;
    typedef Pair<Data_Type, Key_Type> PT;

    // full bi-pair
    for (unsigned int i = 0; i < num; i++) {
#pragma HLS pipeline II = 2
        ap_uint<2 * (Data_Width + Key_Width)> biPair = packBiInStrm.read();
        PT dp0 = {biPair.range(Data_Width + Key_Width - 1, Key_Width), biPair.range(Key_Width - 1, 0), 0},
           dp1 = {biPair.range(2 * (Data_Width + Key_Width) - 1, Data_Width + 2 * Key_Width),
                  biPair.range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width), 0};
        packStrm.write(dp0);
        packStrm.write(dp1);
    }

    // deal with last pair (if exists)
    if ((totalLength % 2) > 0) {
        ap_uint<2 * (Data_Width + Key_Width)> biPair = packBiInStrm.read();
        PT dp = {biPair.range(Data_Width + Key_Width - 1, Key_Width), biPair.range(Key_Width - 1, 0), 0};
        packStrm.write(dp);
    }

    // send end flag
    packStrm.write(PT(0, 0, 1));
}

template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width, int Sort_Len, int Ch_Num>
void channel1toNCore(hls::stream<Pair<Data_Type, Key_Type> >& packInStrm,
                     hls::stream<Pair<Data_Type, Key_Type> > packOutStrm[Ch_Num]) {
    ap_uint<5> ch = 0;
    Pair<Data_Type, Key_Type> packIn = packInStrm.read();
    Pair<Data_Type, Key_Type> packOut;
    unsigned int loadNum = 0;

    // split inserted data block-by-block
    bool end = packIn.end();
    while (!end) {
#pragma HLS loop_tripcount max = 100000 min = 100000
#pragma HLS pipeline II = 1
        if (loadNum < Sort_Len) {
            packOut.key(packIn.key());
            packOut.data(packIn.data());
            packOut.end(0);
            packOutStrm[ch].write(packOut);
            packIn = packInStrm.read();
            end = packIn.end();
            loadNum++;
        } else {
            packOut.key(0);
            packOut.data(0);
            packOut.end(1);
            packOutStrm[ch].write(packOut);
            ch = (ch + 1) % Ch_Num;
            loadNum = 0;
        }
    }

    // send end flags for the remaining streams
    for (unsigned char n = 0; n < Ch_Num; n++) {
#pragma HLS pipeline II = 1
        if (n >= ch) {
            packOut.key(0);
            packOut.data(0);
            packOut.end(1);
            packOutStrm[n].write(packOut);
        }
    }
}

template <typename Key_Type, typename Data_Type, int Block_Len>
void mergeSortWrapper(hls::stream<Pair<Data_Type, Key_Type> >& packLeftStrm,
                      hls::stream<Pair<Data_Type, Key_Type> >& packRightStrm,
                      unsigned int totalLength,
                      hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm,
                      bool order) {
    unsigned int rnd = totalLength / Block_Len + ((totalLength % Block_Len) > 0);

    // internal loop for dealing with long input
    for (unsigned int r = 0; r < rnd; r++) {
#pragma HLS loop_tripcount max = 100 min = 100
        mergeSortPair<Data_Type, Key_Type>(packLeftStrm, packRightStrm, packOutStrm, order);
    }
}

// double-buff insertSort's output before 4->1 mergeTree
template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width, int Sort_Len>
void merge1to4Wrapper(hls::stream<Pair<Data_Type, Key_Type> >& packInStrm,
                      unsigned int totalLength,
                      hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm,
                      bool order) {
#pragma HLS dataflow

    enum { PINGPONG_LEN = 2 * Sort_Len };

    hls::stream<Pair<Data_Type, Key_Type> > packStrm0[4];
#pragma HLS stream variable = packStrm0 depth = PINGPONG_LEN
    //#pragma HLS resource variable = packStrm0 core = FIFO_BRAM

    hls::stream<Pair<Data_Type, Key_Type> > packStrm1[2];
#pragma HLS stream variable = packStrm1 depth = 4
    //#pragma HLS resource variable = packStrm1 core = FIFO_LUTRAM

    // split 1 stream into 4
    channel1toNCore<Data_Type, Key_Type, Data_Width, Key_Width, Sort_Len, 4>(packInStrm, packStrm0);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "channel1toNCore.packStrm0[0].size = " << packStrm0[0].size() << std::endl;
    std::cout << "channel1toNCore.packStrm0[1].size = " << packStrm0[1].size() << std::endl;
    std::cout << "channel1toNCore.packStrm0[2].size = " << packStrm0[2].size() << std::endl;
    std::cout << "channel1toNCore.packStrm0[3].size = " << packStrm0[3].size() << std::endl;
#endif
    // first-stage merge, 4 -> 2
    mergeSortWrapper<Data_Type, Key_Type, 4 * Sort_Len>(packStrm0[0], packStrm0[1], totalLength, packStrm1[0], order);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "mergeSortWrapper.packStrm1[0].size = " << packStrm1[0].size() << std::endl;
#endif
    mergeSortWrapper<Data_Type, Key_Type, 4 * Sort_Len>(packStrm0[2], packStrm0[3], totalLength, packStrm1[1], order);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "mergeSortWrapper.packStrm1[1].size = " << packStrm1[1].size() << std::endl;
#endif
    // second-stage merge, 1 -> 1
    mergeSortWrapper<Data_Type, Key_Type, 4 * Sort_Len>(packStrm1[0], packStrm1[1], totalLength, packOutStrm, order);
}

// Emits chunks of (4 * Sort_Len) or inter
// with (totalLength + (4 * Sort_Len - 1)) / (4 * Sort_Len) end-flags.
template <typename Data_Type,
          typename Key_Type,
          int Sort_Len,
          int Uram_Num,
          int Data_Width,
          int Key_Width,
          int Burst_Len>
void insert2StorageWrapper(ap_uint<2 * (Data_Width + Key_Width)>* inBuff,
                           bool order,
                           unsigned int totalLength,
                           hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm) {
#pragma HLS dataflow

    enum { IN_BUFF_DEPTH = 2 * Burst_Len };

    hls::stream<ap_uint<2 * (Data_Width + Key_Width)> > packBiInStrm;
#pragma HLS stream variable = packBiInStrm depth = IN_BUFF_DEPTH
    //#pragma HLS resource variable = packBiInStrm core = FIFO_BRAM

    hls::stream<Pair<Data_Type, Key_Type> > packInStrm;
#pragma HLS stream variable = packInStrm depth = 4
    //#pragma HLS resource variable = packInStrm core = FIFO_LUTRAM
    hls::stream<Pair<Data_Type, Key_Type> > packInsertStrm;
#pragma HLS stream variable = packInsertStrm depth = 4
    //#pragma HLS resource variable = packInsertStrm core = FIFO_LUTRAM

    readIn<Data_Type, Key_Type, Sort_Len, Data_Width, Key_Width, Burst_Len>(inBuff, totalLength, packBiInStrm);
    downSizer<Data_Type, Key_Type, Data_Width, Key_Width>(packBiInStrm, totalLength, packInStrm);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "downSizer.packInStrm.size = " << packInStrm.size() << std::endl;
#endif
    insertSort<Data_Type, Key_Type, Sort_Len>(packInStrm, packInsertStrm, order);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "insertSort.packInsertStrm.size = " << packInsertStrm.size() << std::endl;
#endif
    merge1to4Wrapper<Data_Type, Key_Type, Data_Width, Key_Width, Sort_Len>(packInsertStrm, totalLength, packOutStrm,
                                                                           order);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "merge1to4Wrapper.packOutStrm.size = " << packOutStrm.size() << std::endl;
#endif
}

// Emits chunks of (4 * Sort_Len * Uram_Num) or inter
// for (totalLength + (4 * Sort_Len * Uram_Num - 1))/(4 *Sort_Len * Uram_Num) times
template <typename Data_Type, typename Key_Type, int Uram_Num, int Block_Len>
void mergeSortTreeWrapper(hls::stream<Pair<Data_Type, Key_Type> > packInStrm[Uram_Num],
                          unsigned int totalLength,
                          bool order,
                          hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm) {
    unsigned int rnd = totalLength / (Block_Len * Uram_Num) + ((totalLength % (Block_Len * Uram_Num)) > 0);

    // internal loop for dealing with long input
    for (unsigned int r = 0; r < rnd; r++) {
#pragma HLS loop_tripcount max = 100 min = 100
        mergeTreeS<Data_Type, Key_Type, Uram_Num>::f(packInStrm, order, packOutStrm);
    }
}

template <typename Data_Type, typename Key_Type, int Uram_Num, int Block_Len, int Data_Width, int Key_Width>
void upSizer(hls::stream<Pair<Data_Type, Key_Type> >& packStrm,
             unsigned int totalLength,
             hls::stream<ap_uint<2 * (Data_Width + Key_Width)> >& packBiOutStrm) {
    ap_uint<2 * (Data_Width + Key_Width)> biPair;
    // deal with full block
    unsigned int rnd = totalLength / (Block_Len * Uram_Num);
    unsigned int nBiPair = Block_Len * Uram_Num / 2;
    for (unsigned int r = 0; r < rnd; r++) {
#pragma HLS loop_flatten off
        for (unsigned int i = 0; i < nBiPair; i++) {
#pragma HLS pipeline II = 2
            Pair<Data_Type, Key_Type> dp0 = packStrm.read();
            biPair.range(Key_Width - 1, 0) = dp0.key();
            biPair.range(Data_Width + Key_Width - 1, Key_Width) = dp0.data();

            Pair<Data_Type, Key_Type> dp1 = packStrm.read();
            biPair.range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width) = dp1.key();
            biPair.range(2 * (Data_Width + Key_Width) - 1, Data_Width + 2 * Key_Width) = dp1.data();
            packBiOutStrm.write(biPair);
        }
        Pair<Data_Type, Key_Type> drop = packStrm.read();
    }

    // deal with partial block
    unsigned int left = totalLength % (Block_Len * Uram_Num);
    if (left > 0) {
        ap_uint<1> enable = 0;
        for (unsigned int i = 0; i <= left; i++) {
#pragma HLS pipeline II = 1
            if (i < left) {
                Pair<Data_Type, Key_Type> dp = packStrm.read();
                biPair.range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width) = dp.key();
                biPair.range(2 * (Data_Width + Key_Width) - 1, Data_Width + 2 * Key_Width) = dp.data();
                // emit bi-pair
                if (enable) {
                    packBiOutStrm.write(biPair);
                }
                // right shift pair
                biPair.range(Data_Width + Key_Width - 1, 0) =
                    biPair.range(2 * (Data_Width + Key_Width) - 1, Data_Width + Key_Width);
                enable++;
            } else {
                Pair<Data_Type, Key_Type> drop = packStrm.read();
                // left shift & emit pair
                if (enable) {
                    biPair.range(2 * (Data_Width + Key_Width) - 1, Data_Width + Key_Width) =
                        biPair.range(Data_Width + Key_Width - 1, 0);
                    packBiOutStrm.write(biPair);
                }
            }
        }
    }
}

template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width, int Burst_Len>
void writeOut(hls::stream<ap_uint<2 * (Data_Width + Key_Width)> >& packStrm,
              unsigned int totalLength,
              unsigned int outBuffOff,
              ap_uint<2 * (Data_Width + Key_Width)>* outBuff) {
    ap_uint<Key_Width> nBursts = totalLength / (Burst_Len * 2);
    unsigned int loadNum = outBuffOff;

    // full burst
    for (unsigned int n = 0; n < nBursts; n++) {
        for (unsigned char i = 0; i < Burst_Len; i++) {
#pragma HLS pipeline II = 1
            ap_uint<2 * (Data_Width + Key_Width)> biPair = packStrm.read();
            outBuff[loadNum++] = biPair;
        }
    }

    // partial burst
    if ((totalLength % (Burst_Len * 2)) > 0) {
        unsigned int burstLen = ((totalLength % (Burst_Len * 2)) + 1) / 2;
        for (unsigned int i = 0; i < burstLen; i++) {
#pragma HLS pipeline II = 1
            ap_uint<2 * (Data_Width + Key_Width)> biPair = packStrm.read();
            outBuff[loadNum++] = biPair;
        }
    }
}

template <typename Data_Type,
          typename Key_Type,
          int Sort_Len,
          int Uram_Num,
          int Data_Width,
          int Key_Width,
          int Burst_Len>
void mergeStorageWrapper(hls::stream<Pair<Data_Type, Key_Type> > packInStrm[Uram_Num],
                         bool order,
                         unsigned int totalLength,
                         unsigned int outBuffOff,
                         ap_uint<2 * (Data_Width + Key_Width)>* outBuff) {
#pragma HLS dataflow

    enum { OUT_BUFF_LEN = 2 * Burst_Len };

    hls::stream<Pair<Data_Type, Key_Type> > packOutStrm;
#pragma HLS stream variable = packOutStrm depth = 4
    //#pragma HLS resource variable = packOutStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<2 * (Data_Width + Key_Width)> > packBiOutStrm;
#pragma HLS stream variable = packBiOutStrm depth = OUT_BUFF_LEN
    //#pragma HLS resource variable = packBiOutStrm core = FIFO_BRAM

    mergeSortTreeWrapper<Data_Type, Key_Type, Uram_Num, Sort_Len * 4>(packInStrm, totalLength, order, packOutStrm);

#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    for (int i = 0; i < Uram_Num; i++) {
        std::cout << "mergeSortTreeWrapper.packInStrm[" << i << "].size = " << packInStrm[i].size() << std::endl;
    }
    std::cout << "mergeSortTreeWrapper.packOutStrm.size = " << packOutStrm.size() << std::endl;
#endif

    upSizer<Data_Type, Key_Type, Uram_Num, Sort_Len * 4, Data_Width, Key_Width>(packOutStrm, totalLength,
                                                                                packBiOutStrm);
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "upSizer.packBiOutStrm.size = " << packBiOutStrm.size() << std::endl;
#endif
    writeOut<Data_Type, Key_Type, Data_Width, Key_Width, Burst_Len>(packBiOutStrm, totalLength, outBuffOff, outBuff);
}

template <typename Data_Type, typename Key_Type, int Data_Width, int Key_Width, int Block_Len, int Uram_Num>
void dispatcher(hls::stream<Pair<Data_Type, Key_Type> >& packInStrm,
                unsigned int totalLength,
                hls::stream<unsigned int> lenStrm[Uram_Num],
                hls::stream<Pair<Data_Type, Key_Type> > packOutStrm[Uram_Num]) {
    ap_uint<Uram_Num> ch = 0;
    Pair<Data_Type, Key_Type> packIn;
    Pair<Data_Type, Key_Type> packOut;

    // dispatch inserted data to individual merge-tree leaves
    unsigned int rnd = totalLength / Block_Len;
    for (unsigned int r = 0; r < rnd; r++) {
        lenStrm[ch].write(Block_Len);
        for (unsigned int i = 0; i <= Block_Len; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
#pragma HLS pipeline II = 1
            packIn = packInStrm.read();
            packOut.key(packIn.key());
            packOut.data(packIn.data());
            packOut.end(packIn.end());
            packOutStrm[ch].write(packOut);
            if (i == Block_Len) {
                ch = (ch + 1) % Uram_Num;
            }
        }
    }

    // deal with the remaining inserted inputs
    unsigned int left = totalLength % Block_Len;
    if (left > 0) {
        lenStrm[ch].write(left);
        for (unsigned int i = 0; i <= left; i++) {
#pragma HLS pipeline II = 1
            packIn = packInStrm.read();
            packOut.key(packIn.key());
            packOut.data(packIn.data());
            packOut.end(packIn.end());
            packOutStrm[ch].write(packOut);
        }
        // emit end flags for rest channels
        for (unsigned char n = 0; n < Uram_Num; n++) {
#pragma HLS pipeline II = 1
            if (n > ch) {
                lenStrm[n].write(0);
                packOut.key(0);
                packOut.data(0);
                packOut.end(1);
                packOutStrm[n].write(packOut);
            }
        }
    } else {
        if (totalLength % (Block_Len * Uram_Num)) {
            for (unsigned char n = 0; n < Uram_Num; n++) {
#pragma HLS pipeline II = 1
                if (n >= ch) {
                    lenStrm[n].write(0);
                    packOut.key(0);
                    packOut.data(0);
                    packOut.end(1);
                    packOutStrm[n].write(packOut);
                }
            }
        }
    }
}

template <typename Data_Type, typename Key_Type, int Block_Len, int Data_Width, int Key_Width, int Uram_Num>
void pingpong1Way(hls::stream<Pair<Data_Type, Key_Type> >& packInStrm,
                  hls::stream<unsigned int>& lenStrm,
                  unsigned int rnd,
                  hls::stream<Pair<Data_Type, Key_Type> >& packOutStrm) {
    ap_uint<Data_Width + Key_Width> pingBuff[Block_Len];
#pragma HLS resource variable = pingBuff core = RAM_2P_URAM
#pragma HLS dependence variable = pingBuff inter false
#pragma HLS dependence variable = pingBuff intra false
    ap_uint<Data_Width + Key_Width> pongBuff[Block_Len];
#pragma HLS resource variable = pongBuff core = RAM_2P_URAM
#pragma HLS dependence variable = pongBuff inter false
#pragma HLS dependence variable = pongBuff intra false

    Pair<Data_Type, Key_Type> packIn;
    ap_uint<Data_Width + Key_Width> pairIn;
    Pair<Data_Type, Key_Type> packOut;
    ap_uint<Data_Width + Key_Width> pairOut;

    ap_uint<1> pingpong = 0;
    unsigned int inCnt;
    unsigned int outCnt;
    unsigned int readNum[2];
#pragma HLS array_partition variable = readNum dim = 0

    for (int r = 0; r <= rnd; r++) {
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1 && __XF_DATABASE_INSERT_KERNEL_DATA_DEBUG__ == 1
        std::cout << "Round " << r << std::endl;
#endif
        inCnt = 0;
        outCnt = 0;
        // write only for 1st round
        if (r < 1) {
            readNum[pingpong] = lenStrm.read();
            if (!pingpong) {
                while (inCnt <= readNum[pingpong]) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = pingBuff inter false
#pragma HLS dependence variable = pingBuff intra false
                    if (inCnt < readNum[pingpong]) {
                        if (packInStrm.read_nb(packIn)) {
                            pairIn.range(Key_Width - 1, 0) = packIn.key();
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1 && __XF_DATABASE_INSERT_KERNEL_DATA_DEBUG__ == 1
                            std::cout << "key = " << pairIn.range(Key_Width - 1, 0) << std::endl;
#endif
                            pairIn.range(Data_Width + Key_Width - 1, Key_Width) = packIn.data();
                            pingBuff[inCnt++] = pairIn;
                        }
                    } else if (inCnt == readNum[pingpong]) {
                        if (packInStrm.read_nb(packIn)) {
                            inCnt++;
                        }
                    }
                }
            }
            pingpong++;
            // read & write for intermediate rounds
        } else if (r < rnd) {
            readNum[pingpong] = lenStrm.read();
            // write pong & read ping
            if (pingpong) {
                while (inCnt <= readNum[pingpong] || outCnt <= readNum[!pingpong]) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = pingBuff inter false
#pragma HLS dependence variable = pingBuff intra false
                    if (inCnt < readNum[pingpong]) {
                        if (packInStrm.read_nb(packIn)) {
                            pairIn.range(Key_Width - 1, 0) = packIn.key();
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1 && __XF_DATABASE_INSERT_KERNEL_DATA_DEBUG__ == 1
                            std::cout << "key = " << pairIn.range(Key_Width - 1, 0) << std::endl;
#endif
                            pairIn.range(Data_Width + Key_Width - 1, Key_Width) = packIn.data();
                            pongBuff[inCnt++] = pairIn;
                        }
                    } else if (inCnt == readNum[pingpong]) {
                        if (packInStrm.read_nb(packIn)) {
                            inCnt++;
                        }
                    }

                    pairOut = pingBuff[outCnt];
                    packOut.key(pairOut.range(Key_Width - 1, 0));
                    packOut.data(pairOut.range(Data_Width + Key_Width - 1, Key_Width));
                    packOut.end(0);
                    if (outCnt < readNum[!pingpong]) {
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    } else if (outCnt == readNum[!pingpong]) {
                        packOut.data(0);
                        packOut.key(0);
                        packOut.end(1);
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    }
                }
                // write ping & read pong
            } else {
                while (inCnt <= readNum[pingpong] || outCnt <= readNum[!pingpong]) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = pingBuff inter false
#pragma HLS dependence variable = pingBuff intra false
                    if (inCnt < readNum[pingpong]) {
                        if (packInStrm.read_nb(packIn)) {
                            pairIn.range(Key_Width - 1, 0) = packIn.key();
#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1 && __XF_DATABASE_INSERT_KERNEL_DATA_DEBUG__ == 1
                            std::cout << "key = " << pairIn.range(Key_Width - 1, 0) << std::endl;
#endif
                            pairIn.range(Data_Width + Key_Width - 1, Key_Width) = packIn.data();
                            pingBuff[inCnt++] = pairIn;
                        }
                    } else if (inCnt == readNum[pingpong]) {
                        if (packInStrm.read_nb(packIn)) {
                            inCnt++;
                        }
                    }

                    pairOut = pongBuff[outCnt];
                    packOut.key(pairOut.range(Key_Width - 1, 0));
                    packOut.data(pairOut.range(Data_Width + Key_Width - 1, Key_Width));
                    packOut.end(0);
                    if (outCnt < readNum[!pingpong]) {
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    } else if (outCnt == readNum[!pingpong]) {
                        packOut.data(0);
                        packOut.key(0);
                        packOut.end(1);
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    }
                }
            }
            pingpong++;
            outCnt = 0;
            // read only for last round
        } else {
            // read ping
            if (pingpong) {
                while (outCnt <= readNum[!pingpong]) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = pingBuff inter false
#pragma HLS dependence variable = pingBuff intra false
                    pairOut = pingBuff[outCnt];
                    packOut.key(pairOut.range(Key_Width - 1, 0));
                    packOut.data(pairOut.range(Data_Width + Key_Width - 1, Key_Width));
                    packOut.end(0);
                    if (outCnt < readNum[!pingpong]) {
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    } else if (outCnt == readNum[!pingpong]) {
                        packOut.data(0);
                        packOut.key(0);
                        packOut.end(1);
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    }
                }
                // read pong
            } else {
                while (outCnt <= readNum[!pingpong]) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = pingBuff inter false
#pragma HLS dependence variable = pingBuff intra false
                    pairOut = pongBuff[outCnt];
                    packOut.key(pairOut.range(Key_Width - 1, 0));
                    packOut.data(pairOut.range(Data_Width + Key_Width - 1, Key_Width));
                    packOut.end(0);
                    if (outCnt < readNum[!pingpong]) {
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    } else if (outCnt == readNum[!pingpong]) {
                        packOut.data(0);
                        packOut.key(0);
                        packOut.end(1);
                        if (packOutStrm.write_nb(packOut)) {
                            outCnt++;
                        }
                    }
                }
            }
        }
    }
}

template <typename Data_Type, typename Key_Type, int Block_Len, int Data_Width, int Key_Width, int Uram_Num>
void pingpongURAM(hls::stream<Pair<Data_Type, Key_Type> >& packInStrm,
                  unsigned int totalLength,
                  hls::stream<Pair<Data_Type, Key_Type> > packOutStrm[Uram_Num]) {
#pragma HLS dataflow

    hls::stream<Pair<Data_Type, Key_Type> > packSplitStrm[Uram_Num];
#pragma HLS stream variable = packSplitStrm depth = 64
#pragma HLS resource variable = packSplitStrm core = FIFO_LUTRAM
    hls::stream<unsigned int> lenStrm[Uram_Num];
#pragma HLS stream variable = lenStrm depth = 4
    //#pragma HLS resource variable = lenStrm core = FIFO_LUTRAM

    unsigned int left = totalLength % (Block_Len * Uram_Num);
    unsigned int rnd = totalLength / (Block_Len * Uram_Num) + (left > 0);

    dispatcher<Data_Type, Key_Type, Data_Width, Key_Width, Block_Len, Uram_Num>(packInStrm, totalLength, lenStrm,
                                                                                packSplitStrm);

#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    std::cout << "dispatcher.packInStrm.size = " << packInStrm.size() << std::endl;
    for (unsigned int i = 0; i < Uram_Num; i++) {
        std::cout << "dispatcher.packSplitStrm[" << i << "].size = " << packSplitStrm[i].size() << std::endl;
    }
#endif

    for (unsigned int i = 0; i < Uram_Num; i++) {
#pragma HLS unroll
        pingpong1Way<Data_Type, Key_Type, Block_Len, Data_Width, Key_Width, Uram_Num>(packSplitStrm[i], lenStrm[i], rnd,
                                                                                      packOutStrm[i]);

#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
        std::cout << "pingpong1Way.packSplitStrm[" << i << "].size = " << packSplitStrm[i].size() << std::endl;
        std::cout << "pingpong1Way.packOutStrm[" << i << "].size = " << packOutStrm[i].size() << std::endl;
#endif
    }
}

} // namespace details

/**
 * @brief insertSortKernel sort each 4 * Sort_Len * Uram_Num key
 * along with its data, and save the sorted items to DDR
 *
 * @tparam Data_Type Type of data
 * @tparam Key_Type Type of key
 * @tparam Sort_Len Length of insert sort
 * @tparam Data_Width Bit width of data
 * @tparam Key_Width Bit width of key
 * @tparam Burst_Len Burst length
 * @tparam Uram_Num Number of URAMs at the middle to buffer the intermediate results
 *
 * @param inBuff Input data & key, expand port-width considering duty cycle of AXI_ready
 * @param totalLength Total number of key & data to be sorted
 * @param order 1:sort ascending 0:sort descending
 * @param outBuff Output sorted result, expand port-width considering duty cycle of AXI_ready
 *
 */
template <typename Data_Type,
          typename Key_Type,
          int Sort_Len,
          int Data_Width,
          int Key_Width,
          int Burst_Len,
          int Uram_Num>
void insertSortKernel(ap_uint<2 * (Data_Width + Key_Width)>* inBuff,
                      unsigned int totalLength,
                      unsigned int order,
                      unsigned int outBuffOff,
                      ap_uint<2 * (Data_Width + Key_Width)>* outBuff) {
#pragma HLS dataflow

    bool sign = order & 0x01UL;

    hls::stream<details::Pair<Data_Type, Key_Type> > packInStrm;
#pragma HLS stream variable = packInStrm depth = 4
    //#pragma HLS resource variable = packInStrm core = FIFO_LUTRAM
    hls::stream<details::Pair<Data_Type, Key_Type> > packOutStrm[Uram_Num];
#pragma HLS stream variable = packOutStrm depth = 4
//#pragma HLS resource variable = packOutStrm core = FIFO_LUTRAM

#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_HOST_DEBUG__ == 1
    std::cout << "Insert kernel info:" << std::endl;
    std::cout << "inBuff =" << inBuff << std::endl;
    std::cout << "totalLength =" << totalLength << std::endl;
    std::cout << "order =" << sign << std::endl;
    std::cout << "outBuff =" << outBuff << std::endl;
    for (int i = 0; i < 64; i++) {
        std::cout << "key_l = " << inBuff[i].range(Key_Width - 1, 0) << std::endl;
        std::cout << "key_h = " << inBuff[i].range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width) << std::endl;
    }
#endif

    details::insert2StorageWrapper<Data_Type, Key_Type, Sort_Len, Uram_Num, Data_Width, Key_Width, Burst_Len>(
        inBuff, sign, totalLength, packInStrm);

    // (totalLength + (4 * Sort_Len - 1)) / (4 * Sort_Len) end-flags in total

    details::pingpongURAM<Data_Type, Key_Type, 4 * Sort_Len, Data_Width, Key_Width, Uram_Num>(packInStrm, totalLength,
                                                                                              packOutStrm);

#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_DEBUG__ == 1
    for (unsigned int i = 0; i < Uram_Num; i++) {
        std::cout << "pingpongURAM.packOutStrm.size = " << packOutStrm[i].size() << std::endl;
    }
#endif

    // (totalLength + (4 * Sort_Len * Uram_Num - 1)) / (4 * Sort_Len * Uram_Num) end-flags from each URAM leave.

    details::mergeStorageWrapper<Data_Type, Key_Type, Sort_Len, Uram_Num, Data_Width, Key_Width, Burst_Len>(
        packOutStrm, sign, totalLength, outBuffOff, outBuff);

#if !defined(__SYNTHESIS__) && __XF_DATABASE_INSERT_KERNEL_HOST_DEBUG__ == 1
    for (int i = 0; i < 64; i++) {
        std::cout << "key_l = " << outBuff[i].range(Key_Width - 1, 0) << std::endl;
        std::cout << "key_h = " << outBuff[i].range(Data_Width + 2 * Key_Width - 1, Data_Width + Key_Width)
                  << std::endl;
    }
#endif
}
} // namespace database
} // namespace xf
#endif
