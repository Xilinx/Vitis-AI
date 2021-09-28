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
 * @file rng_sequence.hpp
 * @brief This files contains random sequence generator
 */

#ifndef XF_FINTECH_RNG_SEQ_H
#define XF_FINTECH_RNG_SEQ_H
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_fintech/corrand.hpp"
#ifndef __SYNTHESIS__
#include <assert.h>
#endif
namespace xf {
namespace fintech {
namespace internal {

template <typename DT, typename RNG>
class RNGSequence {
   public:
    const static unsigned int OutN = 1;
    ap_uint<32> seed[1];
    // Constructor
    RNGSequence(){
#pragma HLS inline
    };

    void Init(RNG rngInst[1]) { rngInst[0].seedInitialization(seed[0]); }

    void NextSeq(ap_uint<16> steps, ap_uint<16> paths, RNG rngInst[1], hls::stream<DT> randNumberStrmOut[1]) {
#pragma HLS inline off
    RNG_LOOP:
        for (int i = 0; i < paths; ++i) {
#pragma HLS loop_tripcount min = 1024 max = 1024
            for (int j = 0; j < steps; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 8
                DT d = rngInst[0].next();
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                std::cout << "randNumber =" << d << std::endl;
#endif
#endif
                randNumberStrmOut[0].write(d);
            }
        }
    }
};

template <typename DT, typename RNG, unsigned int N>
class RNGSequence_N {
   public:
    const static unsigned int OutN = N;
    ap_uint<32> seed[N];
    // Constructor
    RNGSequence_N(){
#pragma HLS inline
    };

    void Init(RNG rngInst[N]) {
        for (int i = 0; i < N; i++) {
#pragma HLS unroll
            rngInst[i].seedInitialization(seed[i]);
        }
    }

    void NextSeq(ap_uint<16> steps, ap_uint<16> paths, RNG rngInst[N], hls::stream<DT> randNumberStrmOut[N]) {
#pragma HLS inline off
    RNG_LOOP:
        for (int i = 0; i < paths; ++i) {
#pragma HLS loop_tripcount min = 1024 max = 1024
            for (int j = 0; j < steps; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 8
                for (int k = 0; k < N; k++) {
#pragma HLS unroll
                    DT z = rngInst[k].next();
                    randNumberStrmOut[k].write(z);
                }
            }
        }
    }
};

template <typename DT, typename RNG>
using RNGSequence_2 = RNGSequence_N<DT, RNG, 2>;

template <typename DT, typename RNG, unsigned int N>
class RNGSequence_1_N {
   public:
    const static unsigned int RngN = 1;
    const static unsigned int OutN = N;
    ap_uint<32> seed;
    // Constructor
    RNGSequence_1_N(){
#pragma HLS inline
    };

    void Init(RNG rngInst[1]) { rngInst[0].seedInitialization(seed); }

    void NextSeq(ap_uint<16> steps, ap_uint<16> paths, RNG rngInst[1], hls::stream<DT> randNumberStrmOut[N]) {
#pragma HLS inline off
    RNG_LOOP:
        for (int i = 0; i < paths; ++i) {
#pragma HLS loop_tripcount min = 1024 max = 1024
            for (int j = 0; j < steps; ++j) {
#pragma HLS loop_tripcount min = 8 max = 8
                for (int k = 0; k < N; k++) {
#pragma HLS loop_tripcount min = N max = N
                    DT z = rngInst[0].next();
                    randNumberStrmOut[k].write(z);
                }
            }
        }
    }
};

template <typename DT, typename RNG, bool WithAntithetic>
class GaussUniformSequence {
   public:
    const static unsigned int OutN = WithAntithetic ? 4 : 2;
    ap_uint<32> seed[2];
    // constructor
    GaussUniformSequence() {
#pragma HLS inline
    }

    void Init(RNG rngInst[2]) {
        for (int i = 0; i < 2; i++) {
#pragma HLS unroll
            rngInst[i].seedInitialization(seed[i]);
        }
    }

    void NextSeq(ap_uint<16> steps, ap_uint<16> paths, RNG rngInst[2], hls::stream<DT> randNumberStrmOut[OutN]) {
#pragma HLS inline off
    RNG_LOOP:
        for (int i = 0; i < paths; ++i) {
#pragma HLS loop_tripcount min = 1024 max = 1024
            for (int j = 0; j < steps; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 8
                DT gaussD = rngInst[0].next();
                ap_ufixed<32, 0> uniformD1, uniformD2;
                DT uniformD;
                if (WithAntithetic)
                    rngInst[1].uniformRNG.nextTwo(uniformD1, uniformD2);
                else
                    rngInst[1].next(uniformD);
#ifndef __SYNTHESIS__
#ifdef HLS_DEBUG
                std::cout << "randNumber =" << gaussD << std::endl;
#endif
#endif
                if (WithAntithetic) {
                    randNumberStrmOut[0].write(gaussD);
                    randNumberStrmOut[1].write(uniformD1);
                    randNumberStrmOut[2].write(-gaussD);
                    randNumberStrmOut[3].write(uniformD2);
                } else {
                    randNumberStrmOut[0].write(gaussD);
                    randNumberStrmOut[1].write(uniformD);
                }
            }
        }
    }
};

template <typename DT, typename RNG>
class RNGSequence_Heston_QuadraticExponential {
   public:
    const static int OutN = 3;
    ap_uint<32> seed[2];
    // constructor
    RNGSequence_Heston_QuadraticExponential() {
#pragma HLS inline
    }

    void Init(RNG rngInst[2]) {
        for (int i = 0; i < 2; i++) {
#pragma HLS unroll
            rngInst[i].seedInitialization(seed[i]);
        }
    }

    void NextSeq(ap_uint<16> steps, ap_uint<16> paths, RNG rngInst[2], hls::stream<DT> randNumberStrmOut[3]) {
#pragma HLS inline off
        DT z0, z1, u1;
    RNG_LOOP:
        for (int i = 0; i < paths; ++i) {
#pragma HLS loop_tripcount min = 1024 max = 1024
            for (int j = 0; j < steps; ++j) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 8
                z0 = rngInst[0].next();
                rngInst[1].next(u1, z1);
                randNumberStrmOut[0].write(z0);
                randNumberStrmOut[1].write(z1);
                randNumberStrmOut[2].write(u1);
            }
        }
    }
};

template <typename DT, typename RNG, int SampleNum, int ASSETS, bool Antithetic>
class CORRAND_2_Sequence;

template <typename DT, typename RNG, int SampleNum, int ASSETS>
class CORRAND_2_Sequence<DT, RNG, SampleNum, ASSETS, false> {
   public:
    const static unsigned int OutN = 2;
    bool firstcall;
    ap_uint<32> seed[2];

    void Init(RNG rngInst[2]) {
        for (int i = 0; i < 2; i++) {
#pragma HLS unroll
            rngInst[i].seedInitialization(seed[i]);
        }
        firstcall = true;
        corrand.init();
    }

    CORRAND_2<DT, SampleNum, ASSETS, 1> corrand;

    CORRAND_2_Sequence() {
#pragma HLS inline
    }

    void rawRand(ap_uint<16> steps, ap_uint<16> paths, RNG rngInst[2], hls::stream<DT> randStrm[2]) {
        int local_steps;
        DT z0, z1;
        if (firstcall) {
            local_steps = steps + 1;
            firstcall = false;
        } else {
            local_steps = steps;
        }

        for (int i = 0; i < local_steps; i++) {
            for (int j = 0; j < ASSETS; j++) {
                for (int k = 0; k < paths; k++) {
#pragma HLS pipeline II = 1
                    for (int m = 0; m < 2; m++) {
#pragma HLS unroll
                        DT z = rngInst[m].next();
                        randStrm[m].write(z);
                    }
                }
            }
        }
    }

    void rawRand_s(hls::stream<ap_uint<16> >& steps_strm, RNG rngInst[2], hls::stream<DT> randStrm[2]) {
        ap_uint<16> steps;
        steps = steps_strm.read();

        int local_steps;
        DT z0, z1;
        if (firstcall) {
            local_steps = steps + 1;
            firstcall = false;
        } else {
            local_steps = steps;
        }

        for (int i = 0; i < local_steps; i++) {
            for (int j = 0; j < ASSETS; j++) {
                for (int k = 0; k < SampleNum; k++) {
#pragma HLS pipeline II = 1
                    for (int m = 0; m < 2; m++) {
#pragma HLS unroll
                        DT z = rngInst[m].next();
                        randStrm[m].write(z);
                    }
                }
            }
        }
    }

    void _NextSeq(hls::stream<ap_uint<16> >& steps_strm1,
                  hls::stream<ap_uint<16> >& steps_strm2,
                  RNG rngInst[2],
                  hls::stream<DT> corrRandStrmOut[2]) {
#pragma HLS DATAFLOW
        hls::stream<DT> randStrm[2];
#pragma HLS stream variable = randStrm depth = 1024
        rawRand_s(steps_strm1, rngInst, randStrm);
        corrand.corrPathCube(steps_strm2, randStrm[0], randStrm[1], corrRandStrmOut[0], corrRandStrmOut[1]);
    }

    void NextSeq(ap_uint<16> steps, ap_uint<16> paths, RNG rngInst[2], hls::stream<DT> corrRandStrmOut[2]) {
        hls::stream<ap_uint<16> > scalar_strm[2];
#pragma HLS stream variable = scalar_strm depth = 4
        scalar_strm[0].write(steps);
        scalar_strm[1].write(steps);
        _NextSeq(scalar_strm[0], scalar_strm[1], rngInst, corrRandStrmOut);
    }
};

} // namespace internal
} // namespace fintech
} // namespace xf
#endif
