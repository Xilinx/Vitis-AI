/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DSPLIB_MATRIX_MULT_TILER_CPP_
#define _DSPLIB_MATRIX_MULT_TILER_CPP_

#include <adf.h>

#ifndef __NEW_WINDOW_H__
#define __NEW_WINDOW_H__ 1
#endif
// if we use 1kb registers -> aie api uses 2x512b registers for 1024b so we need this for QoR
#ifndef __AIE_API_USE_NATIVE_1024B_VECTOR__
#define __AIE_API_USE_NATIVE_1024B_VECTOR__
#endif
#include "aie_api/aie_adf.hpp"
#include "matrix_mult_tiler.hpp"
#include "matrix_mult_tiler_common.hpp"

#include "debug_utils.h"

#ifndef ROW_MAJOR
#define ROW_MAJOR 0
#endif
#ifndef COL_MAJOR
#define COL_MAJOR 1
#endif

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {

constexpr unsigned int makeShuffleOffsets(
    unsigned M, unsigned N, unsigned vectorSize, unsigned leadingDim, bool hiLo) //, unsigned int loadSize) {
{
    unsigned int loadSize = vectorSize == 8 ? 2 : 4;
    unsigned long int ret = 0;
    unsigned mIncr = (leadingDim == ROW_MAJOR) ? (loadSize < N) ? (N / loadSize) * loadSize : loadSize : 1; // transpose
    // We first load enough for M, then we load for N when we're tranpose.
    unsigned nIncr = (leadingDim == ROW_MAJOR) ? 1 : (loadSize < M) ? (M / loadSize) * loadSize : loadSize; // tranpose
    for (unsigned int tileIndex = 0; tileIndex < vectorSize / (M * N); ++tileIndex) {
        // This has two parts. If N is smaller than the loadsize, then we first deal with those extra samples.
        // Then we deal with the remaining samples.
        const unsigned tileStartIndex =
            (leadingDim == ROW_MAJOR) ? ((tileIndex * N) % loadSize + ((tileIndex * N) / loadSize) * (M * loadSize))
                                      : ((tileIndex % ((vectorSize / loadSize) / N)) * N * loadSize +
                                         tileIndex / ((vectorSize / loadSize) / N) * M);

        for (unsigned int mIndex = 0; mIndex < M; ++mIndex) {
            for (unsigned int nIndex = 0; nIndex < N; ++nIndex) {
                // printf("t=%d mIndex=%d nIndex=%d\t", tileIndex, mIndex, nIndex);
                const unsigned laneIndex = nIndex + mIndex * N + tileIndex * M * N;
                // This effectively deals with tranpose at the offsets level.
                const unsigned sampleIndex = (tileStartIndex + mIndex * mIncr + nIndex * nIncr);
                // printf("%d + %d + %d = ", tileStartIndex, mIndex * mIncr, nIndex * nIncr);
                // printf("%d ", sampleIndex);
                if (laneIndex < 8 && !hiLo) {
                    ret += ((unsigned long)sampleIndex) << (4 * (laneIndex % 8));
                } else if (laneIndex >= 8 && hiLo) {
                    ret += ((unsigned long)sampleIndex) << (4 * (laneIndex % 8));
                }
            }
        }
    }
    // printf("full: %0lX \t", ret );
    // printf("hi: %0X \t lo: %0X\n", (unsigned int) (ret >> 32), (unsigned int) (ret & 0xFFFFFFFF) );
    // printf("\n");
    return ret;
}
constexpr loHi getShuffleOffsets(unsigned M, unsigned N, unsigned vectorSize, unsigned leadingDim) {
    loHi ret = {
        .lo = makeShuffleOffsets(M, N, vectorSize, leadingDim, false),
        .hi = makeShuffleOffsets(M, N, vectorSize, leadingDim, true),
    };
    return ret;
}

constexpr loHi getShuffleOffsetsInt16(const unsigned M,
                                      const unsigned N,
                                      const unsigned vectorSize,
                                      const unsigned leadingDim) {
    unsigned int andMask = 0x0F0F0F0F; // every odd offset is handled differently for int16
    // This assumes that the loadsize is 8
    // An offset of 3 on odd index corresponds to previus row's odd index +1 + 2*offset
    // So if we have
    // I J
    // X Y
    // X = J+1 + Off*2; Y = X+1
    // We only have N=2 or N=4
    unsigned int orMask = N == 2 ? 0x30303030 : 0x0; // don't do anything for N=4
    unsigned int offLo = (makeShuffleOffsets(M, N / 2, vectorSize / 2, leadingDim, false) & andMask) | orMask;
    unsigned int offHi = (makeShuffleOffsets(M, N / 2, vectorSize / 2, leadingDim, true) & andMask) | orMask;
    // Hardcoding possible combinations, assumed N=2
    if (leadingDim == COL_MAJOR) {
        offLo = (M == 4) ? 0x39383130 : (M == 2) ? 0x39313830 : 0;
        offHi = (M == 4) ? 0x3B3A3332 : (M == 2) ? 0xB3333A32 : 0;
    }
    loHi ret = {
        .lo = offLo,
        .hi = offHi,
        .square = (leadingDim == ROW_MAJOR) ? (unsigned)0x3210
                                            : (unsigned)0x3120, // Need permute to grab sample from different 32b chunk
    };
    return ret;
}
template <unsigned M, unsigned N, unsigned inRow, unsigned inCol, unsigned leadingDim, typename T_D>
static void doTile(T_D* inPtr, T_D* outPtr) {
    constexpr unsigned minGranularity = (128 / 8) / sizeof(T_D);
    constexpr unsigned loadSize = (N >= minGranularity) ? N : minGranularity;
    constexpr unsigned minVBuffSizeforType = (512 / 8) / sizeof(T_D);
    constexpr unsigned vectorSize = minVBuffSizeforType;

    // static_assert(N >= minGranularity, "Granularity is awkward");
    static_assert(vectorSize <= (1024 / 8) / sizeof(T_D),
                  "ERROR: calculated vector size too large for vector register.");
    static_assert(!(N == 4 && leadingDim == COL_MAJOR && std::is_same_v<T_D, int16>),
                  "ERROR: Tiling is not supported for column major int16 matrix.");

    constexpr unsigned int sizeTileA = M * N;
    // Window size in terms of elements
    const unsigned windowSizeA = inRow * inCol;
    const unsigned largeTile = (M * N) * (inCol / N);
    // printf("LargeTileSize=%d\n",largeTile);

    // This will get optimised away if not used.
    // Offsets for the shuffle intrinsic
    const loHi offsets = std::is_same_v<T_D, int16> ? getShuffleOffsetsInt16(M, N, vectorSize, leadingDim)
                                                    : getShuffleOffsets(M, N, vectorSize, leadingDim);
    // printf("M: %d, N: %d, vectorSize: %d, leadingDim: %d\n", M, N, vectorSize, leadingDim);
    // printf("Offsets: lo : %0X, hi: %0X\n", offsets.lo, offsets.hi);
    const unsigned loadsPerVector = vectorSize / loadSize;

    const unsigned columnsPerVector = (leadingDim == ROW_MAJOR)
                                          ? (vectorSize / M <= inCol) ? loadSize * (loadsPerVector) / M : inCol
                                          : (loadSize >= M) ? (loadsPerVector) : (loadsPerVector) / (M / loadSize);
    const unsigned rowsPerVector =
        (leadingDim == ROW_MAJOR) ? (vectorSize / M <= inCol) ? M : vectorSize / inCol : (loadSize >= M) ? loadSize : M;
    const unsigned colElementDist = (leadingDim == ROW_MAJOR) ? 1 : inRow;
    const unsigned rowElementDist = (leadingDim == ROW_MAJOR) ? inCol : 1;
    const unsigned rowTilesPerVector = std::max((rowsPerVector / M), (unsigned)1);

    static_assert(
        inRow * inCol >= vectorSize,
        "ERROR: Matrix is too small to implement tiling. A single matrix must consume at least 512 bits of memory.");
    const unsigned vectorsPerCol = inRow / rowsPerVector;
    const unsigned vectorsPerRow = inCol / columnsPerVector;

    for (unsigned rowPos = 0; rowPos < vectorsPerCol; ++rowPos)
        chess_loop_count((vectorsPerCol)) chess_prepare_for_pipelining {
            unsigned outI = rowPos * largeTile;
            const unsigned rowIndex = rowPos * rowElementDist * rowsPerVector;
            // printf("new LargeTile\n");
            // Travel down the column
            for (unsigned colPos = 0; colPos < vectorsPerRow; ++colPos)
                chess_loop_count(vectorsPerRow) chess_prepare_for_pipelining {
                    // printf("rowPos=%d, colPos=%d\n",rowPos, colPos );
                    const unsigned colIndex = colPos * colElementDist * columnsPerVector;
                    aie::vector<T_D, vectorSize> chunk;
// if constexpr (vectorSize == loadSize) {
//  chunk = aie::load_v<loadSize>(pBase+AChunk);
//  resultIndex = (N*(
//    offsetX*(((AChunk%largeTile)/N)/incrX % wrapX) +
//    offsetY*(((AChunk%largeTile)/N)/incrY % wrapY)))+
//    (AChunk/largeTile)*largeTile;
//} else {
#pragma unroll((loadsPerVector))
                    for (unsigned i = 0; i < (loadsPerVector); ++i) {
                        // Todo: deal with the possibility that inCol < loadSize for COL_MAJOR
                        const unsigned loadPos =
                            (leadingDim == ROW_MAJOR)
                                ? (loadSize >= inCol) ? i * loadSize : (i % M) * inCol + (i / M) * loadSize
                                : (loadSize >= M) ? inRow * i
                                                  : inRow * (i / (M / loadSize)) + (i % (M / loadSize)) * loadSize;
                        const unsigned pointerLoc = colIndex + rowIndex + loadPos;
                        // printf("loadPos=%d, pointerLoc=%d\n",loadPos, pointerLoc );
                        //(AChunk+((i%M)*inCol + (i/M)*loadSize)) :
                        //(largeTileIndex*loadSize + ((AChunk - largeTileIndex*largeTile)*inRow)+((i%M)*inCol +
                        //(i/M)*loadSize));

                        chunk.insert(i, aie::load_v<loadSize>(inPtr + pointerLoc));
                        // printf("pointerLoc=%d\n", pointerLoc);
                        // myprint(chunk, true, "incChunk: ");
                    }
                    // If N is 4
                    // shuffle16(chunk, 0, 0xD951C840)
                    // shuffle16(chunk, 2, 0xD951C840)
                    // initialise to chunk
                    aie::vector<T_D, vectorSize> mychunk;
                    // If N is 2 & loadsize=4

                    if
                        constexpr(N < minGranularity || leadingDim == COL_MAJOR) {
                            mychunk = doShuffle(chunk, 0, offsets);
                        }
                    else {
                        mychunk = chunk;
                    }
                    // chunk = shuffle16(chunk, 0, offsets.lo, offsets.hi);

                    // myprint(mychunk, true, "ShuffleChunk: ");

                    const unsigned resultIndexBase =
                        vectorSize * (vectorsPerRow)*rowPos + (vectorSize / rowTilesPerVector) * colPos;
                    outI += vectorSize;
// printf("ResultIndex = %d\n",resultIndexBase);
// chunk.shuffle()
//}
// N*(
//  offsetX*(((AChunk%largeTile)/N)/incrX % wrapX) +
//  offsetY*(((AChunk%largeTile)/N)/incrY % wrapY))+
//  (AChunk/largeTile)*largeTile;
// printf("%d : %d\n, ",AChunk,resultIndexBase );
// printf("\toffX[%d]*(n[%d]/incrX[%d] mod wrapX[%d]) = %d\n", offsetX, AChunk/N, incrX, wrapX,
// offsetX*((AChunk/N)/incrX % wrapX) );
// printf("\toffY[%d]*(n[%d]/incrY[%d] mod wrapY[%d]) = %d\n", offsetY, AChunk/N, incrY, wrapY,
// offsetY*((AChunk/N)/incrY % wrapY) );
// myprint(chunk, true, "chunk: ");
// If we end up loading more rows than we need for a single tile in a vector, need to store this somewhere else
#pragma unroll((rowTilesPerVector))
                    for (unsigned tile = 0; tile < rowTilesPerVector; ++tile) {
                        const unsigned resultIndexPos = resultIndexBase + tile * largeTile;
                        // printf("Proposed outPos = %d\n", resultIndexBase + tile*largeTile);
                        aie::store_v(outPtr + resultIndexPos,
                                     mychunk.template extract<vectorSize / rowTilesPerVector>(tile));
                    }
                }
        }
    // printf("\n");
    // for (unsigned row=0; row < 4; ++row){
    //  #pragma unroll(M)
    //  for (unsigned mslice=row*M/2; mslice<M; ++mslice){
    //    //myprint(A0,true,"A0preProc: ");
    //
    //    A0.insert(mslice, aie::load_v<N>(pBase+N*mslice) );
    //    A1.insert(mslice, aie::load_v<N>(pBase+inCol*mslice) );
    //    //myprint(A1,true,"A1preProc: ");
    //    aie::store_v(outPtr+inCol*mslice, A0.template extract<N>(mslice));
    //
    //    aie::store_v(outPtr+N*mslice, A1.template extract<N>(mslice));
    //  }
    //  myprint(A0,true,"A0preProc: ");
    //
    //  myprint(A1,true,"A1preProc: ");
    //}
    // T_D* debugPtr = outPtr;
    // for (unsigned AChunk=0; AChunk<windowSizeA; AChunk+=sizeTileA){
    //  aie::vector<T_D, sizeTileA> APost = aie::load_v<sizeTileA>(debugPtr); debugPtr += sizeTileA;
    ////  //aie::vector<T_D, sizeTileA> A1 = aie::load_v<sizeTileA>(pA1); pA1 += sizeTileA;
    //  myprint(APost, true, "A0postProc: ");
    ////  myprint(A1,true,"A1preProc: ");
    /////
    //}
    // return outPtr;
}

namespace aie = ::aie;
template <unsigned M, unsigned N, unsigned inRow, unsigned inCol, unsigned leadingDim, typename T_D>
void tilerKernelClass<M, N, inRow, inCol, leadingDim, T_D>::tile(input_window<T_D>* inWindow,
                                                                 output_window<T_D>* restrict outWindow) {
    // printf("Going to tile matrix\n");
    doTile<M, N, inRow, inCol, leadingDim, T_D>((T_D*)inWindow->ptr, (T_D*)outWindow->ptr);
};
}
}
}
}
}

#endif
