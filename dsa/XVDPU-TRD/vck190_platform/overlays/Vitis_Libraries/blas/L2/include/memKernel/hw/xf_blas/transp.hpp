/**********
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
 * **********/
/**
 *  @brief DDR-to-DDR matrix transposer and formatter
 *
 *  $DateTime: 2018/01/09 05:28:55 $
 */

#ifndef XF_BLAS_TRANSP_HPP
#define XF_BLAS_TRANSP_HPP

#include "assert.h"
#include "hls_stream.h"
#include "types.hpp"
#include "kargs.hpp"

namespace xf {

namespace blas {

///////////////////////////////////////////////////////////////////////////
// DDR-to-DDR transposer
//
//   DDR matrix is decomposed into the blocks:
//                l_colBlocks
//    ---------------------------------
//    | Trans | Trans | Trans | Trans |
//    | Block | Block | Block | Block|
//    ---------------------------------  l_rowBlocks
//    | Trans | Trans | Trans | Trans |
//    | Block | Block | Block | Block|
//    ---------------------------------
//    | Trans | Trans | Trans | Trans |
//    | Block | Block | Block | Block|
//    ---------------------------------
//
//
///////////////////////////////////////////////////////////////////////////
template <typename t_FloatType, unsigned int t_DdrWidth, unsigned int t_colMemWords, unsigned int t_rowMemWords>
class Transp {
   public:
    typedef WideType<t_FloatType, t_DdrWidth> DdrWideType;
    typedef typename DdrWideType::t_TypeInt DdrIntType;
    typedef hls::stream<typename WideType<t_FloatType, t_DdrWidth>::t_TypeInt> DdrStream;
    static const unsigned short t_colBlockLength = t_DdrWidth * t_colMemWords;
    static const unsigned short t_rowBlockLength = t_DdrWidth * t_rowMemWords;
    typedef TranspArgs TranspArgsType;
    typedef DdrMatrixShape::FormatType MatFormatType;

   public:
    void processWithReuse(DdrStream& fromMemStream,
                          DdrStream& toMemStream,
                          unsigned int numOfBlocks,
                          unsigned numOfReuse) {
        DdrStream p_wrStream, p_s1_1, p_s1_2, p_s2_1, p_s2_2, p_outStream;
#pragma HLS STREAM variable = p_wrStream depth = 2
#pragma HLS STREAM variable = p_s1_1 depth = t_colMemWords * t_rowBlockLength
#pragma HLS STREAM variable = p_s1_2 depth = t_colMemWords * t_rowBlockLength
#pragma HLS STREAM variable = p_s2_1 depth = 2
#pragma HLS STREAM variable = p_s2_2 depth = 2
#pragma HLS STREAM variable = p_outStream depth = 4
#pragma HLS INLINE
        shuffle_input(fromMemStream, p_wrStream, numOfBlocks);
        split(p_wrStream, p_s1_1, p_s1_2, numOfBlocks);
        WR_bufferWithReuse(p_s1_1, p_s2_1, ((numOfBlocks / 2) + (numOfBlocks % 2)), numOfReuse - 1);
        WR_bufferWithReuse(p_s1_2, p_s2_2, numOfBlocks / 2, numOfReuse - 1);
        mergeWithReuse(p_s2_1, p_s2_2, p_outStream, numOfBlocks, numOfReuse - 1);
        shuffle_output(p_outStream, toMemStream, numOfBlocks * numOfReuse);
    }

    void shuffle_input(DdrStream& fromMemStream, DdrStream& wrStream, unsigned int numOfBlocks) {
        bool finish = false;
        enum STATE { IDLE = 0, SHUFFLE } mState = IDLE;

        unsigned int blockCounter = 0;
        unsigned short dimCounter = 0;
        unsigned short memWordCounter = 0;
        unsigned short shiftLeftNum = 0;
        DdrWideType word2WR;
        DdrWideType wordFromMem;
#pragma HLS ARRAY_PARTITION variable = wordFromMem complete dim = 1

        while (!finish) {
#pragma HLS PIPELINE II = 1 enable_flush
            switch (mState) {
                case IDLE:
                    memWordCounter = 0;
                    dimCounter = 0;
                    blockCounter++;
                    wordFromMem = fromMemStream.read();
                    mState = SHUFFLE;
                    break;
                case SHUFFLE:
                    shiftLeftNum = dimCounter % t_DdrWidth;
                    for (unsigned short w = 0; w < t_DdrWidth; ++w) {
#pragma HLS UNROLL
                        word2WR[w] = wordFromMem[(w - shiftLeftNum + t_DdrWidth) % t_DdrWidth];
                    }
                    wrStream.write(word2WR);
                    if ((dimCounter == (t_rowBlockLength - 1)) && (memWordCounter == (t_colMemWords - 1))) {
                        if (blockCounter >= numOfBlocks) {
                            finish = true;
                        } else {
                            // mState = IDLE;
                            memWordCounter = 0;
                            dimCounter = 0;
                            blockCounter++;
                            wordFromMem = fromMemStream.read();
                        }
                    } else {
                        wordFromMem = fromMemStream.read();
                        if (memWordCounter == (t_colMemWords - 1)) {
                            memWordCounter = 0;
                            dimCounter++;
                        } else {
                            memWordCounter++;
                        }
                    }
                    break;
            }
        }
        return;
    }

    void WR_buffer(DdrStream& wrStream, DdrStream& shuffleStream, unsigned int numOfBlocks) {
        bool finish = (numOfBlocks == 0);
        enum STATE { IDLE = 0, WRITE, READ } mState = IDLE;

        t_FloatType localBuffer[t_DdrWidth][t_rowBlockLength * t_colBlockLength / t_DdrWidth];
#pragma HLS ARRAY_PARTITION variable = localBuffer dim = 1

        unsigned short writeIndex = 0;
        unsigned short readIndexBase[t_DdrWidth];
        unsigned short readIndexBaseIncr[t_DdrWidth];
        unsigned short readIndex[t_DdrWidth];

        DdrWideType wordFromShuffle;
        DdrWideType word2Shuffle;

        unsigned int blockCounter = 0;
        unsigned short dimCounter = 0;
        unsigned short memWordCounter = 0;

        while (!finish) {
#pragma HLS PIPELINE II = 1 enable_flush
            switch (mState) {
                case IDLE:
                    memWordCounter = 0;
                    dimCounter = 0;
                    writeIndex = 0;
                    blockCounter++;
                    wordFromShuffle = wrStream.read();
                    for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                        readIndexBase[i] = i * t_colMemWords;
                    }
                    mState = WRITE;
                    break;
                case WRITE:
                    for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                        localBuffer[i][writeIndex] = wordFromShuffle[i];
                    }
                    if ((memWordCounter != (t_colMemWords - 1)) || (dimCounter != (t_rowBlockLength - 1))) {
                        wordFromShuffle = wrStream.read();
                        if (memWordCounter == (t_colMemWords - 1)) {
                            memWordCounter = 0;
                            dimCounter++;
                        } else {
                            memWordCounter++;
                        }
                        writeIndex++;
                    } else {
                        memWordCounter = 0;
                        dimCounter = 0;
                        for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                            readIndex[i] = readIndexBase[i];
                            readIndexBaseIncr[i] = readIndexBase[i] + 1;
                        }
                        unsigned short tmp = readIndexBase[t_DdrWidth - 1];
                        for (unsigned short i = t_DdrWidth - 1; i > 0; --i) {
                            readIndexBase[i] = readIndexBase[i - 1];
                        }
                        readIndexBase[0] = tmp;
                        mState = READ;
                    }
                    break;
                case READ:
                    for (unsigned int i = 0; i < t_DdrWidth; ++i) {
                        word2Shuffle[i] = localBuffer[i][readIndex[i]];
                    }
                    shuffleStream.write(word2Shuffle);
                    if ((memWordCounter == (t_rowMemWords - 1)) && (dimCounter == (t_colBlockLength - 1))) {
                        if (blockCounter >= numOfBlocks) {
                            finish = true;
                        }
                        mState = IDLE;
                    } else if (memWordCounter == (t_rowMemWords - 1)) {
                        if ((dimCounter % t_DdrWidth) == (t_DdrWidth - 1)) {
                            for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                                readIndex[i] = readIndexBaseIncr[i];
                                readIndexBase[i] = readIndexBaseIncr[(i + t_DdrWidth - 1) % t_DdrWidth];
                            }
                            for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                                readIndexBaseIncr[i] += 1;
                            }
                        } else {
                            for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                                readIndex[i] = readIndexBase[i];
                            }
                            unsigned short tmp = readIndexBase[t_DdrWidth - 1];
                            for (unsigned short i = t_DdrWidth - 1; i > 0; --i) {
                                readIndexBase[i] = readIndexBase[i - 1];
                            }
                            readIndexBase[0] = tmp;
                        }
                        memWordCounter = 0;
                        dimCounter++;
                    } else {
                        for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                            readIndex[i] += t_colBlockLength;
                        }
                        memWordCounter++;
                    }
                    break;
            }
        }
        return;
    }

    void split(DdrStream& inStream, DdrStream& outStream, DdrStream& outStreamNext, unsigned int numOfBlocks) {
        for (int i = 0; i < numOfBlocks; ++i) {
            for (int j = 0; j < t_rowBlockLength; ++j) {
                for (int k = 0; k < t_colMemWords; ++k) {
#pragma HLS PIPELINE
                    DdrWideType l_word = inStream.read();
                    if ((i % 2) == 0) {
                        outStream.write(l_word);
                    } else {
                        outStreamNext.write(l_word);
                    }
                }
            }
        }
    }

    void WR_bufferWithReuse(DdrStream& wrStream,
                            DdrStream& shuffleStream,
                            unsigned int numOfBlocks,
                            unsigned int numOfReuse) {
        // after first WRITE, the READ state is gone over numOfReuse times to reuse the transposed block again and again
        bool finish = (numOfBlocks == 0);
        enum STATE { IDLE = 0, WRITE, READ } mState = IDLE;

        t_FloatType localBuffer[t_DdrWidth][t_rowBlockLength * t_colBlockLength / t_DdrWidth];
#pragma HLS ARRAY_PARTITION variable = localBuffer dim = 1

        unsigned short writeIndex = 0;
        unsigned short readIndexBase[t_DdrWidth];
        unsigned short readIndexBaseIncr[t_DdrWidth];
        unsigned short readIndex[t_DdrWidth];

        DdrWideType wordFromShuffle;
        DdrWideType word2Shuffle;

        unsigned int blockCounter = 0;
        unsigned short dimCounter = 0;
        unsigned short reuseCounter = 0;
        unsigned short memWordCounter = 0;

        while (!finish) {
#pragma HLS PIPELINE II = 1 enable_flush
            switch (mState) {
                case IDLE:
                    memWordCounter = 0;
                    dimCounter = 0;
                    writeIndex = 0;
                    reuseCounter = numOfReuse;
                    blockCounter++;
                    wordFromShuffle = wrStream.read();
                    for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                        readIndexBase[i] = i * t_colMemWords;
                    }
                    mState = WRITE;
                    break;
                case WRITE:
                    for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                        localBuffer[i][writeIndex] = wordFromShuffle[i];
                    }
                    if ((memWordCounter != (t_colMemWords - 1)) || (dimCounter != (t_rowBlockLength - 1))) {
                        wordFromShuffle = wrStream.read();
                        if (memWordCounter == (t_colMemWords - 1)) {
                            memWordCounter = 0;
                            dimCounter++;
                        } else {
                            memWordCounter++;
                        }
                        writeIndex++;
                    } else {
                        memWordCounter = 0;
                        dimCounter = 0;
                        for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                            readIndex[i] = readIndexBase[i];
                            readIndexBaseIncr[i] = readIndexBase[i] + 1;
                        }
                        unsigned short tmp = readIndexBase[t_DdrWidth - 1];
                        for (unsigned short i = t_DdrWidth - 1; i > 0; --i) {
                            readIndexBase[i] = readIndexBase[i - 1];
                        }
                        readIndexBase[0] = tmp;
                        mState = READ;
                    }
                    break;
                case READ:
                    for (unsigned int i = 0; i < t_DdrWidth; ++i) {
                        word2Shuffle[i] = localBuffer[i][readIndex[i]];
                    }
                    shuffleStream.write(word2Shuffle);
                    if ((memWordCounter == (t_rowMemWords - 1)) && (dimCounter == (t_colBlockLength - 1))) {
                        if ((reuseCounter == 0) && (blockCounter == numOfBlocks)) {
                            finish = true;
                        } else if (reuseCounter == 0) {
                            mState = IDLE;
                        } else {
                            reuseCounter--;
                            memWordCounter = 0;
                            dimCounter = 0;
                            for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                                readIndex[i] = i * t_colMemWords;
                                readIndexBaseIncr[i] = i * t_colMemWords + 1;
                            }

                            for (unsigned short i = t_DdrWidth - 1; i > 0; --i) {
                                readIndexBase[i] = (i - 1) * t_colMemWords;
                            }
                            readIndexBase[0] = (t_DdrWidth - 1) * t_colMemWords;
                            mState = READ;
                        }
                    } else if (memWordCounter == (t_rowMemWords - 1)) {
                        if ((dimCounter % t_DdrWidth) == (t_DdrWidth - 1)) {
                            for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                                readIndex[i] = readIndexBaseIncr[i];
                                readIndexBase[i] = readIndexBaseIncr[(i + t_DdrWidth - 1) % t_DdrWidth];
                            }
                            for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                                readIndexBaseIncr[i] += 1;
                            }
                        } else {
                            for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                                readIndex[i] = readIndexBase[i];
                            }
                            unsigned short tmp = readIndexBase[t_DdrWidth - 1];
                            for (unsigned short i = t_DdrWidth - 1; i > 0; --i) {
                                readIndexBase[i] = readIndexBase[i - 1];
                            }
                            readIndexBase[0] = tmp;
                        }
                        memWordCounter = 0;
                        dimCounter++;
                    } else {
                        for (unsigned short i = 0; i < t_DdrWidth; ++i) {
                            readIndex[i] += t_colBlockLength;
                        }
                        memWordCounter++;
                    }
                    break;
            }
        }
        return;
    }

    void merge(DdrStream& inStream, DdrStream& inStreamNext, DdrStream& outStream, unsigned int numOfBlocks) {
        for (int i = 0; i < numOfBlocks; ++i) {
            for (int j = 0; j < t_rowBlockLength; ++j) {
                for (int k = 0; k < t_colMemWords; ++k) {
#pragma HLS PIPELINE
                    DdrWideType l_word;
                    if ((i % 2) == 0) {
                        l_word = inStream.read();
                    } else {
                        l_word = inStreamNext.read();
                    }
                    outStream.write(l_word);
                }
            }
        }
    }

    void mergeWithReuse(DdrStream& inStream,
                        DdrStream& inStreamNext,
                        DdrStream& outStream,
                        unsigned int numOfBlocks,
                        unsigned int numOfReuse) {
        for (int i = 0; i < numOfBlocks; ++i) {
            for (int r = 0; r <= numOfReuse; ++r) {
                for (int j = 0; j < t_rowBlockLength; ++j) {
                    for (int k = 0; k < t_colMemWords; ++k) {
#pragma HLS PIPELINE
                        DdrWideType l_word;
                        if ((i % 2) == 0) {
                            l_word = inStream.read();
                        } else {
                            l_word = inStreamNext.read();
                        }
                        outStream.write(l_word);
                    }
                }
            }
        }
    }
    void shuffle_output(DdrStream& shuffleStream, DdrStream& toMemStream, unsigned int numOfBlocks) {
        bool finish = false;
        enum STATE { IDLE = 0, WRITE } mState = IDLE;

        unsigned short memWordCounter = 0;
        unsigned short dimCounter = 0;
        unsigned int blockCounter = 0;

        DdrWideType wordFromWR;
#pragma HLS ARRAY_PARTITION variable = wordFromWR complete dim = 1
        DdrWideType word2Out;

        while (!finish) {
#pragma HLS PIPELINE II = 1 enable_flush
            switch (mState) {
                case IDLE:
                    wordFromWR = shuffleStream.read();
                    dimCounter = 0;
                    memWordCounter = 0;
                    blockCounter++;
                    mState = WRITE;
                    break;

                case WRITE:
                    unsigned short shiftRightNum = dimCounter % t_DdrWidth;
                    for (unsigned short i = 0; i < t_DdrWidth; ++i) {
#pragma HLS UNROLL
                        word2Out[i] = wordFromWR[(i + shiftRightNum) % t_DdrWidth];
                    }
                    toMemStream.write(word2Out);
                    if ((dimCounter != (t_colBlockLength - 1)) || (memWordCounter != (t_rowMemWords - 1))) {
                        wordFromWR = shuffleStream.read();
                        if (memWordCounter == (t_rowMemWords - 1)) {
                            memWordCounter = 0;
                            dimCounter++;
                        } else {
                            memWordCounter++;
                        }
                    } else {
                        if (blockCounter >= numOfBlocks) {
                            finish = true;
                        } else {
                            // mState = IDLE;
                            wordFromWR = shuffleStream.read();
                            dimCounter = 0;
                            memWordCounter = 0;
                            blockCounter++;
                        }
                    }
                    break;
            }
        }
        return;
    }

    void load_matrix(DdrIntType* l_AddrRd,
                     unsigned int l_srcWordLd,
                     unsigned int l_rowBlocks,
                     unsigned int l_colBlocks,
                     DdrStream& fromMemStream) {
        unsigned int l_srcOffset = 0;
        unsigned int colOffset = 0;
        unsigned int rowOffset = 0;
        unsigned short l_rowBlock = 0;
        unsigned short l_colBlock = 0;
        bool finish = false;

        while (!finish) {
            l_srcOffset =
                rowOffset + colOffset; // l_rowBlock * t_rowBlockLength * l_srcWordLd + l_colBlock * t_colMemWords;
            for (unsigned short row = 0; row < t_rowBlockLength; ++row) {
                for (unsigned short colWord = 0; colWord < t_colMemWords; ++colWord) {
#pragma HLS PIPELINE
                    DdrWideType l_word = l_AddrRd[l_srcOffset + colWord];
                    fromMemStream.write(l_word);
                }
                l_srcOffset += l_srcWordLd;
            }
            colOffset += t_colMemWords;

            //}
            l_colBlock++;
            if (l_colBlock >= l_colBlocks) {
                l_colBlock = 0;
                rowOffset += l_srcWordLd * t_rowBlockLength;
                colOffset = 0;
                l_rowBlock++;
            }
            if (l_rowBlock >= l_rowBlocks) {
                finish = true;
            }
        }
    }

    void store_matrix(DdrStream& toMemStream,
                      DdrIntType* l_AddrWr,
                      unsigned int l_dstWordLd,
                      unsigned int l_rowBlocks,
                      unsigned int l_colBlocks) {
        unsigned int l_dstOffset;
        unsigned int colOffset = 0;
        unsigned int rowOffset = 0;
        unsigned short l_rowBlock = 0;
        unsigned short l_colBlock = 0;
        bool finish = false;

        while (!finish) {
            l_dstOffset = colOffset + rowOffset; // l_colBlock * t_colBlockLength*l_dstWordLd+l_rowBlock *
                                                 // t_rowMemWords;
            for (unsigned short row = 0; row < t_colBlockLength; ++row) {
                for (unsigned short colWord = 0; colWord < t_rowMemWords; ++colWord) {
#pragma HLS PIPELINE
                    DdrWideType l_word = toMemStream.read();
                    l_AddrWr[l_dstOffset + colWord] = l_word;
                }
                l_dstOffset += l_dstWordLd;
            }
            colOffset += t_colBlockLength * l_dstWordLd;
            l_colBlock++;
            if (l_colBlock >= l_colBlocks) {
                l_colBlock = 0;
                colOffset = 0;
                rowOffset += t_rowMemWords;
                l_rowBlock++;
            }
            if (l_rowBlock >= l_rowBlocks) {
                finish = true;
            }
        }
    }
    void store_matrixGVA(DdrStream& toMemStream,
                         DdrIntType* l_AddrWr,
                         unsigned int l_dstWordLd,
                         unsigned int l_rowBlocks,
                         unsigned int l_colBlocks) {
        unsigned int l_dstOffset = 0;
        unsigned int p_dstOffset = 0;
        unsigned short l_rowBlock = 0;
        unsigned short l_colBlock = 0;
        bool finish = false;

        while (!finish) {
            p_dstOffset = l_dstOffset;
            for (unsigned short row = 0; row < t_colBlockLength; ++row) {
#pragma HLS PIPELINE
                for (unsigned short colWord = 0; colWord < t_rowMemWords; ++colWord) {
                    DdrWideType l_word = toMemStream.read();
                    l_AddrWr[l_dstOffset] = l_word;
                    l_dstOffset++;
                }
            }
            l_colBlock++;
            if (l_colBlock >= l_colBlocks) {
                l_colBlock = 0;
                l_rowBlock++;
            }
            if (l_rowBlock >= l_rowBlocks) {
                finish = true;
            }
        }
    }

    // transp_matrix_blocks does not parallel block load and block transposition tasks
    void transp_matrix_blocks(DdrIntType* l_AddrRd,
                              unsigned int l_srcWordLd,
                              unsigned int l_rowBlocks,
                              DdrIntType* l_AddrWr,
                              unsigned int l_dstWordLd,
                              unsigned int l_colBlocks,
                              unsigned int numOfBlocks) {
#pragma HLS DATAFLOW

        DdrStream fromMemStream("fromMemStream");
        DdrStream wrStream("wrStream");
        DdrStream wrStream1("wrStream1");
        DdrStream wrStream2("wrStream2");
        DdrStream shuffleStream1("shuffleStream1");
        DdrStream shuffleStream2("shuffleStream2");
        DdrStream shuffleStream("shuffleStream");
        DdrStream toMemStream("toMemStream");

#pragma HLS DATA_PACK variable = fromMemStream
#pragma HLS STREAM variable = fromMemStream depth = 4

#pragma HLS DATA_PACK variable = wrStream
#pragma HLS STREAM variable = wrStream depth = 4

#pragma HLS DATA_PACK variable = wrStream1
#pragma HLS STREAM variable = wrStream1 depth = 4

#pragma HLS DATA_PACK variable = wrStream2
#pragma HLS STREAM variable = wrStream2 depth = 4

#pragma HLS DATA_PACK variable = shuffleStream1
#pragma HLS STREAM variable = shuffleStream1 depth = 4

#pragma HLS DATA_PACK variable = shuffleStream2
#pragma HLS STREAM variable = shuffleStream2 depth = 4

#pragma HLS DATA_PACK variable = shuffleStream
#pragma HLS STREAM variable = shuffleStream depth = 4

#pragma HLS DATA_PACK variable = toMemStream
#pragma HLS STREAM variable = toMemStream depth = 4

        load_matrix(l_AddrRd, l_srcWordLd, l_rowBlocks, l_colBlocks, fromMemStream);
        shuffle_input(fromMemStream, wrStream, numOfBlocks);
        // WR_buffer(wrStream, shuffleStream, numOfBlocks);
        split(wrStream, wrStream1, wrStream2, numOfBlocks);
        WR_buffer(wrStream1, shuffleStream1, ((numOfBlocks / 2) + (numOfBlocks % 2)));
        WR_buffer(wrStream2, shuffleStream2, numOfBlocks / 2);
        merge(shuffleStream1, shuffleStream2, shuffleStream, numOfBlocks);
        shuffle_output(shuffleStream, toMemStream, numOfBlocks);
        store_matrix(toMemStream, l_AddrWr, l_dstWordLd, l_rowBlocks, l_colBlocks);
    }

    void transp_GemvA_blocks(DdrIntType* l_AddrRd,
                             unsigned int l_srcWordLd,
                             unsigned int l_rowBlocks,
                             DdrIntType* l_AddrWr,
                             unsigned int l_dstWordLd,
                             unsigned int l_colBlocks,
                             unsigned int numOfBlocks) {
#pragma HLS DATAFLOW

        DdrStream fromMemStream("fromMemStream");
        DdrStream wrStream("wrStream");
        DdrStream shuffleStream("shuffleStream");
        DdrStream toMemStream("toMemStream");

#pragma HLS DATA_PACK variable = fromMemStream
#pragma HLS STREAM variable = fromMemStream depth = 4

#pragma HLS DATA_PACK variable = wrStream
#pragma HLS STREAM variable = wrStream depth = 4

#pragma HLS DATA_PACK variable = shuffleStream
#pragma HLS STREAM variable = shuffleStream depth = 4

#pragma HLS DATA_PACK variable = toMemStream
#pragma HLS STREAM variable = toMemStream depth = 4

        load_matrix(l_AddrRd, l_srcWordLd, l_rowBlocks, l_colBlocks, fromMemStream);
        shuffle_input(fromMemStream, wrStream, numOfBlocks);
        WR_buffer(wrStream, shuffleStream, numOfBlocks);
        shuffle_output(shuffleStream, toMemStream, numOfBlocks);
        store_matrixGVA(toMemStream, l_AddrWr, l_dstWordLd, l_rowBlocks, l_colBlocks);
    }

    void runTransp(DdrIntType* p_DdrRd, DdrIntType* p_DdrWr, TranspArgsType& p_Args) {
#pragma HLS inline off

        DdrIntType *l_AddrRd = p_DdrRd + p_Args.m_Src.m_Offset * DdrWideType::per4k(),
                   *l_AddrWr = p_DdrWr + p_Args.m_Dst.m_Offset * DdrWideType::per4k();
        const unsigned int l_srcWordLd = p_Args.m_Src.m_Ld / t_DdrWidth;
        const unsigned int l_dstWordLd = p_Args.m_Dst.m_Ld / t_DdrWidth;
        assert(l_srcWordLd * t_DdrWidth == p_Args.m_Src.m_Ld);
        assert(l_dstWordLd * t_DdrWidth == p_Args.m_Dst.m_Ld);

        MatFormatType l_test = MatFormatType::Rm;
        assert(p_Args.m_Src.m_Format == MatFormatType::Rm);

        if (p_Args.m_Dst.m_Format == MatFormatType::Cm) {
            /////////////////  Cm  ////////////////
            //   -------------             ---------
            //   | 1 | 2 | 3 |             | 1 | 4 |
            //   | 4 | 5 | 6 |             | 2 | 5 |
            //   -------------             | 3 | 6 |
            //                             ---------
            // Calculate sizes
            const unsigned int t_BlockEdge = t_DdrWidth * t_colMemWords;
            const unsigned int l_rowBlocks = p_Args.m_Src.m_Rows / t_BlockEdge;
            const unsigned int l_colBlocks = p_Args.m_Src.m_Cols / t_BlockEdge;
            const unsigned int numOfBlocks = l_rowBlocks * l_colBlocks;
            assert(l_rowBlocks * t_BlockEdge == p_Args.m_Src.m_Rows);
            assert(l_colBlocks * t_BlockEdge == p_Args.m_Src.m_Cols);

            // Transpose
            transp_matrix_blocks(l_AddrRd, l_srcWordLd, l_rowBlocks, l_AddrWr, l_dstWordLd, l_colBlocks, numOfBlocks);

        } else if (p_Args.m_Dst.m_Format == MatFormatType::GvA) {
            /////////////////  GemvA  ////////////////
            //   -------------            ------------------
            //   | 1 | 2 | 3 |            | 1   | 2   | 3   |
            //   |   |   |   |            |-----------------|
            //   |-----------|            | 4   | 5   | 6   |
            //   | 4 | 5 | 6 |            -------------------
            //   |   |   |   |              DDR-width times wider, and shorter
            //   -------------
            //
            //  HLS cycle counts are for 2048 x 512 matrix on 32-wide core
            //
            //

            // Calculate sizes
            const unsigned int l_rowBlockLength = t_DdrWidth * t_rowMemWords;
            const unsigned int l_colBlockLength = t_DdrWidth * t_colMemWords;
            // const unsigned int l_colBlockLengthInDdrWords = t_colMemWords;
            const unsigned int l_rowBlocks = p_Args.m_Src.m_Rows / l_rowBlockLength;
            const unsigned int l_colBlocks = p_Args.m_Src.m_Cols / l_colBlockLength;
            const unsigned int numOfBlocks = l_rowBlocks * l_colBlocks;
            assert(l_rowBlocks * t_rowMemWords * t_DdrWidth == p_Args.m_Src.m_Rows);
            assert(l_colBlocks * l_colBlockLength == p_Args.m_Src.m_Cols);

            transp_GemvA_blocks(l_AddrRd, l_srcWordLd, l_rowBlocks, l_AddrWr, l_dstWordLd, l_colBlocks, numOfBlocks);
        } else {
            assert(false); // Unknown output matrix format
        }
    }
};

} // namespace
}
#endif
