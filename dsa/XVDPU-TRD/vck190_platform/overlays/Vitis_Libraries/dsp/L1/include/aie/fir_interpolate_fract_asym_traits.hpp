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
#ifndef _DSPLIB_fir_interpolate_fract_asym_TRAITS_HPP_
#define _DSPLIB_fir_interpolate_fract_asym_TRAITS_HPP_

#include <array> //for phase arrays
namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace interpolate_fract_asym {
/*
Single Rate Asymetrical FIR traits.
This file contains sets of overloaded, templatized and specialized templatized functions which
encapsulate properties of the intrinsics used by the main kernal class. Specifically,
this file does not contain any vector types or intrinsics since it is required for construction
and therefore must be suitable for the aie compiler graph-level compilation.
*/
// Defensive checks
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnFirInterpFractTypeSupport() {
    return fnUnsupportedTypeCombo<TT_DATA, TT_COEFF>();
}; // Int16 data int16 coeff type combo unsupported, due to xquare.

// Not sure why max columns is 2 for now.
static constexpr unsigned int kMaxColumns = 2;
static constexpr unsigned int kUpdWSize = 256 / 8;    // Upd_w size in Bytes (256bit) - const for all data/coeff types
static constexpr unsigned int kZBuffSize = 256 / 8;   // Zbuff size in Bytes (256bit) - const for all data/coeff types
static constexpr unsigned int kXYBuffSize = 1024 / 8; // XYbuff size in Bytes (1024bit) - const for all data/coeff types
static constexpr unsigned int kBuffSize128Byte = 128; // 1024-bit buffer size in Bytes
static constexpr unsigned int kBuffSize64Byte = 64;   // 512-bit buffer size in Bytes
static constexpr unsigned int kBuffSize32Byte = 32;   // 256-bit buffer size in Bytes

struct firParamsTrait {
    // typename dataType = TT_DATA;
    unsigned int dataSizeBytes;
    unsigned int firLen;
    unsigned int loadSize;
    unsigned int dataBuffSamples;
    unsigned int alignWindowReadBytes;
    unsigned int marginOffsetIndex;
    unsigned int rangeOffset;
};

// function to return the number of lanes for a type combo
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnNumLanesIntFract() {
    return 0; // effectively an error trap, but adding an error message to a constexpr return results in a warning.
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<int16, int16>() {
    return 16;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cint16, int16>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cint16, cint16>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<int32, int16>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<int32, int32>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cint32, int16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cint32, cint16>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cint32, int32>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cint32, cint32>() {
    return 2;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<float, float>() {
    return 8;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cfloat, float>() {
    return 4;
};
template <>
inline constexpr unsigned int fnNumLanesIntFract<cfloat, cfloat>() {
    return 4;
};

// function to return the number of columns for a type combo for the intrinsics used for this single rate asym FIR
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnNumColumnsIntFract() {
    return sizeof(TT_COEFF) == 2 ? 2 : 1;
};
// specialize for any exceptions like this:
// template<> inline constexpr unsigned int fnNumColumnsDecHb< int16,  int16, K_ARCH_1BUFF>() { return 2;};

// We use the zbuffer for data in this interpolate, as it's expected we'd like to
// reduce coefficient loads.
template <typename TT_DATA>
inline constexpr unsigned int fnDataLoadsInRegIntFract() {
    // To fill a full 1024-bit input vector register using a 256-bit upd_w command it takes 4 load operations.
    // Always return 4, the exception (handled by explicit template instantiation)
    // would be needed it a different command was used (e.g. upd_v, upd_x).
    return kXYBuffSize / kUpdWSize;
    // Not doing this anymore
    // We're using zbuff for data instead.
    // return kZBuffSize/kUpdWSize;
    // At worst we need num_lanes samples. so as long as sizeof(data)*numLanes<=KZBuffSize, we should be OK.
}

// Parameter to constant resolution functions
template <typename TT_DATA>
inline constexpr unsigned int fnDataLoadVsizeIntFract() {
    return (kUpdWSize / sizeof(TT_DATA));
}

// Function returns window access size, the memory data path is either 128-bits or 256-bits wide for vector operations.
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnWinAccessByteSize() {
    return 16;
};

// function to return the number of samples in an output vector for a type combo
template <typename TT_DATA, typename TT_COEFF>
inline constexpr unsigned int fnVOutSizeIntFract() {
    return fnNumLanesIntFract<TT_DATA, TT_COEFF>();
};
// template<typename TT_DATA, typename TT_COEFF> inline constexpr unsigned int
// fnDataNeededPhases<TP_INTERPOLATE_FACTOR,TP_DECIMATION_FACTOR>(m_kLanes,m_kPhaseLaneAlias)

constexpr unsigned int getSampleIndexOnLane(
    unsigned int D, unsigned int I, unsigned int nLanes, unsigned int laneIndex, unsigned int phaseIndex) {
    unsigned int trueIndex = (phaseIndex * nLanes + laneIndex);
    unsigned int trueSample = (unsigned int)((trueIndex * D) / I);
    return trueSample;
}
constexpr unsigned int getSampleIndexOnLaneCol(unsigned int D,
                                               unsigned int I,
                                               unsigned int nLanes,
                                               unsigned int laneIndex,
                                               unsigned int phaseIndex,
                                               unsigned int colIndex) {
    return colIndex + getSampleIndexOnLane(D, I, nLanes, laneIndex, phaseIndex);
}

// NOTE, this function returns the min max
constexpr std::array<int, 2> getBuffMinMax(firParamsTrait params,
                                           unsigned int D,
                                           unsigned int I,
                                           unsigned int nLanes,
                                           unsigned int nCols,
                                           unsigned int phaseIndex,
                                           unsigned int opIndex) {
    // For some reason this function only works if these are signed...
    int firLen = params.firLen;
    int dataSizeBytes = params.dataSizeBytes;
    int buffSampleSize = params.dataBuffSamples; //(1024/8)/dataSizeBytes;
    int alignRead =
        params.alignWindowReadBytes / dataSizeBytes; //(128/8)/dataSizeBytes; // alignment granularity of AIE.
    int loadSize = params.loadSize;                  //(256/8)/dataSizeBytes;
    int marginOffsetIndex = params.marginOffsetIndex;
    // these are min needed for first opIndex only. Our dataNeeded and dataLoaded accurately handle subsequent ops.
    int minSampleNeeded = (int)(getSampleIndexOnLaneCol(D, I, nLanes, 0, phaseIndex, 0)) - marginOffsetIndex;
    int maxSampleNeeded =
        (int)(getSampleIndexOnLaneCol(D, I, nLanes, nLanes - 1, phaseIndex, (opIndex * nCols + (nCols - 1)))) -
        marginOffsetIndex;

    // Align sample needed to windRead Granularity
    int minSbuff =
        ((int)((minSampleNeeded < 0 ? minSampleNeeded - (alignRead - 1) : minSampleNeeded) / alignRead)) * alignRead;
    // maxSampleNeeded, align to window read access granularity.
    int maxSbuff =
        ((int)((maxSampleNeeded < 0 ? (maxSampleNeeded + 1) : (maxSampleNeeded + 1 + (alignRead - 1))) / alignRead)) *
        alignRead;
    // Align Sbuff to actual loadSize.
    maxSbuff = minSbuff + ((int)(((maxSbuff - minSbuff) + (loadSize - 1)) / loadSize)) * loadSize;

    minSbuff = ((maxSbuff - minSbuff) > buffSampleSize) ? (maxSbuff - buffSampleSize) : minSbuff;

    return {minSbuff, maxSbuff};
}

template <unsigned int polyphaseLaneAlias>
constexpr std::array<int, polyphaseLaneAlias> getWindowDecrements(
    firParamsTrait params, unsigned int D, unsigned int I, unsigned int nLanes, unsigned int nCols, unsigned int nOps) {
    std::array<int, polyphaseLaneAlias> windowDecs = {0};
    std::array<int, 2> buffMinMaxInitial = getBuffMinMax(params, D, I, nLanes, nCols, 0, 0);
    std::array<int, 2> buffMinMaxPrevFinal =
        getBuffMinMax(params, D, I, nLanes, nCols, polyphaseLaneAlias - 1, nOps - 1);
    const unsigned int MAX = 1;
    const unsigned int MIN = 0;
    int nextPhaseChunkStarting = (int)((polyphaseLaneAlias * nLanes * D) / I);
    // A negative dec would indicate an incr
    windowDecs[polyphaseLaneAlias - 1] = buffMinMaxPrevFinal[MAX] - (buffMinMaxInitial[MIN] + nextPhaseChunkStarting);

    for (unsigned int phase = 1; phase < polyphaseLaneAlias; phase++) {
        std::array<int, 2> buffMinMaxInitialOp = getBuffMinMax(params, D, I, nLanes, nCols, phase, 0);
        std::array<int, 2> buffMinMaxPrevFinalOp = getBuffMinMax(params, D, I, nLanes, nCols, phase - 1, nOps - 1);
        windowDecs[phase - 1] = buffMinMaxPrevFinalOp[MAX] - buffMinMaxInitialOp[MIN];
    }
    const std::array<int, polyphaseLaneAlias> windowDecsRet = windowDecs;
    return windowDecsRet;
}

template <unsigned int polyphaseLaneAlias>
constexpr std::array<unsigned int, polyphaseLaneAlias> getInitialLoads(
    firParamsTrait params, unsigned int D, unsigned int I, unsigned int nLanes, unsigned int nCols) {
    std::array<unsigned int, polyphaseLaneAlias> initLoads = {0};
    unsigned int dataSizeBytes = params.dataSizeBytes;
    unsigned int loadSize = params.loadSize; //(256/8)/dataSizeBytes;
    const unsigned int MAX = 1;
    const unsigned int MIN = 0;
    // A negative dec would indicate an incr
    for (unsigned int phase = 0; phase < polyphaseLaneAlias; phase++) {
        std::array<int, 2> buffMinMaxInitial = getBuffMinMax(params, D, I, nLanes, nCols, phase, 0);
        initLoads[phase] = (buffMinMaxInitial[MAX] - buffMinMaxInitial[MIN]) / loadSize;
    }
    return initLoads;
}
template <unsigned int polyphaseLaneAlias>
constexpr std::array<int, polyphaseLaneAlias> getXStarts(
    firParamsTrait params, unsigned int D, unsigned int I, unsigned int nLanes, unsigned int nCols) {
    std::array<int, polyphaseLaneAlias> xstarts = {0};
    unsigned int dataSizeBytes = params.dataSizeBytes;
    unsigned int loadSize = params.loadSize; //(256/8)/dataSizeBytes;
    const unsigned int MAX = 1;
    const unsigned int MIN = 0;
    // int minSampleNeeded;
    unsigned int firLen = params.firLen;
    int marginOffsetIndex = params.marginOffsetIndex;
    // A negative dec would indicate an incr
    for (unsigned int phase = 0; phase < polyphaseLaneAlias; phase++) {
        int minSampleNeeded = (int)(getSampleIndexOnLaneCol(D, I, nLanes, 0, phase, 0)) - marginOffsetIndex;
        std::array<int, 2> buffMinMaxInitial = getBuffMinMax(params, D, I, nLanes, nCols, phase, 0);
        xstarts[phase] = (minSampleNeeded - buffMinMaxInitial[MIN]);
    }
    return xstarts;
}
// returns the initial data needed starting value.
template <unsigned int polyphaseLaneAlias>
constexpr std::array<unsigned int, polyphaseLaneAlias> getDataNeeded(
    firParamsTrait params, unsigned int D, unsigned int I, unsigned int nLanes, unsigned int nCols) {
    std::array<unsigned int, polyphaseLaneAlias> dataNeeded = {0};
    unsigned int dataSizeBytes = params.dataSizeBytes;
    unsigned int loadSize = params.loadSize; // todo Link to current loadSize
    const unsigned int MAX = 1;
    const unsigned int MIN = 0;
    // int minSampleNeeded,maxSampleNeeded;
    unsigned int firLen = params.firLen; // todo
    int marginOffsetIndex = params.marginOffsetIndex;
    // A negative dec would indicate an incr
    for (unsigned int phase = 0; phase < polyphaseLaneAlias; phase++) {
        int minSampleNeeded = (int)(getSampleIndexOnLaneCol(D, I, nLanes, 0, phase, 0)) - marginOffsetIndex;
        int maxSampleNeeded =
            (int)(getSampleIndexOnLaneCol(D, I, nLanes, nLanes - 1, phase, (nCols - 1))) - marginOffsetIndex;
        dataNeeded[phase] = maxSampleNeeded - minSampleNeeded + 1;
    }
    return dataNeeded;
}

// returns the initial data needed starting value.
template <unsigned int polyphaseLaneAlias>
constexpr std::array<unsigned int, polyphaseLaneAlias> getXOffsets(unsigned int D,
                                                                   unsigned int I,
                                                                   unsigned int nLanes) {
    std::array<unsigned int, polyphaseLaneAlias> acc = {0};
    // int minSampleNeeded,maxSampleNeeded;
    // A negative dec would indicate an incr
    for (unsigned int phase = 0; phase < polyphaseLaneAlias; phase++) {
        int min = getSampleIndexOnLane(D, I, nLanes, 0, phase);
        for (unsigned int lane = 0; lane < nLanes; lane++) {
            acc[phase] += (getSampleIndexOnLane(D, I, nLanes, lane, phase) - min)
                          << 4 * lane; // left shift in order to have values at 4b offset
        }
    }
    return acc;
}
// returns the initial data needed starting value.
template <unsigned int polyphaseLaneAlias>
constexpr std::array<unsigned int, polyphaseLaneAlias> getZOffsets(unsigned int I, unsigned int nLanes) {
    std::array<unsigned int, polyphaseLaneAlias> acc = {0};
    // int minSampleNeeded,maxSampleNeeded;
    // A negative dec would indicate an incr
    unsigned int alreadyAccounted = I > nLanes ? 0 : 1; // simple hack to enable/disable phase alias on coeff offsets.
    for (unsigned int phase = 0; phase < polyphaseLaneAlias; phase++) {
        for (unsigned int lane = 0; lane < nLanes; lane++) {
            acc[phase] += ((phase * nLanes * alreadyAccounted + lane) % I)
                          << 4 * lane; // left shift in order to have values at 4b offset
        }
    }
    return acc;
}

constexpr unsigned int my_gcd(unsigned int m, unsigned int n) {
    return (n == 0 ? m : my_gcd(n, m % n));
}
constexpr unsigned int my_lcm(unsigned int m, unsigned int n) {
    return ((m / my_gcd(m, n)) * n);
}
}
}
}
}
}
#endif // _DSPLIB_fir_interpolate_fract_asym_TRAITS_HPP_
