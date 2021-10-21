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
 * @file xoshiro128.hpp
 * @brief This file include the class XoShiRo128
 *
 */

#ifndef __XF_FINTECH_XOSHIRO128_HPP_
#define __XF_FINTECH_XOSHIRO128_HPP_

#ifndef __SYNTHESIS__
#include "iostream"
#endif

#include "hls_math.h"
#include "ap_int.h"

namespace xf {

namespace fintech {

/**
 * @brief XoShiRo128Plus is a 32-bit all-purpose, rock-solid generator.
 */
class XoShiRo128Plus {
   private:
    const static int W = 32;
    int s[4];
    const unsigned int one = 1;

    ap_uint<W> rotl(ap_uint<W> x, int k) { return (x << k) | (x >> (W - k)); }

   public:
    /**
     * @brief default constructor
     */
    XoShiRo128Plus() {
#pragma HLS inline
#pragma HLS array_partition variable = s complete
    }

    /**
     * @brief init initialize seeds
     * @param seedIn input seeds
     */
    void init(unsigned int* seedIn) {
        for (int i = 0; i < 4; i++) {
#pragma HLS pipeline
            s[i] = seedIn[i];
        }
    }

    /**
     * @brief each call of next() generate a pseudorandom number
     * @return return a pseudorandom number
     */
    ap_uint<W> next(void) {
#pragma HLS pipeline
        ap_uint<W> result = s[0] + s[3];

        ap_uint<W> t = s[1] << 9;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 11);

        return result;
    }

    /**
     * @brief the jump function is equivalent to 2^64 calls to next(); it can be used to generate 2^64 non-overlapping
     * subsequences for parallel computations.
     */
    void jump(void) {
        ap_uint<W> JUMP[] = {0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b};

        ap_uint<W> s0 = 0;
        ap_uint<W> s1 = 0;
        ap_uint<W> s2 = 0;
        ap_uint<W> s3 = 0;
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < W; b++) {
#pragma HLS pipeline
                if (JUMP[i] & (one << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }

    /**
     * @brief the long-jump function is equivalent to 2^96 calls to next(); it can be used to generate 2^32 starting
     * points, from each of which jump() will generate 2^32 non-overlapping subsequences for parallel distributed
     * computations.
     */
    void long_jump(void) {
        ap_uint<W> LONG_JUMP[] = {0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662};

        ap_uint<W> s0 = 0;
        ap_uint<W> s1 = 0;
        ap_uint<W> s2 = 0;
        ap_uint<W> s3 = 0;
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < W; b++) {
#pragma HLS pipeline
                if (LONG_JUMP[i] & (one << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }
};

/**
 * @brief XoShiRo128PlusPlus is a 32-bit all-purpose, rock-solid generator.
 */
class XoShiRo128PlusPlus {
   private:
    const static int W = 32;
    int s[4];
    const unsigned int one = 1;

    ap_uint<W> rotl(ap_uint<W> x, int k) { return (x << k) | (x >> (W - k)); }

   public:
    /**
     * @brief default constructor
     */
    XoShiRo128PlusPlus() {
#pragma HLS inline
#pragma HLS array_partition variable = s complete
    }

    /**
     * @brief init initialize seeds
     * @param seedIn input seeds
     */
    void init(unsigned int* seedIn) {
        for (int i = 0; i < 4; i++) {
#pragma HLS pipeline
            s[i] = seedIn[i];
        }
    }

    /**
     * @brief each call of next() generate a pseudorandom number
     * @return return a pseudorandom number
     */
    ap_uint<W> next(void) {
#pragma HLS pipeline
        ap_uint<W> result = rotl(s[0] + s[3], 7) + s[0];

        ap_uint<W> t = s[1] << 9;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 11);

        return result;
    }

    /**
     * @brief the jump function is equivalent to 2^64 calls to next(); it can be used to generate 2^64 non-overlapping
     * subsequences for parallel computations.
     */
    void jump(void) {
        ap_uint<W> JUMP[] = {0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b};

        ap_uint<W> s0 = 0;
        ap_uint<W> s1 = 0;
        ap_uint<W> s2 = 0;
        ap_uint<W> s3 = 0;
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < W; b++) {
#pragma HLS pipeline
                if (JUMP[i] & (one << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }

    /**
     * @brief the long-jump function is equivalent to 2^96 calls to next(); it can be used to generate 2^32 starting
     * points, from each of which jump() will generate 2^32 non-overlapping subsequences for parallel distributed
     * computations.
     */
    void long_jump(void) {
        ap_uint<W> LONG_JUMP[] = {0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662};

        ap_uint<W> s0 = 0;
        ap_uint<W> s1 = 0;
        ap_uint<W> s2 = 0;
        ap_uint<W> s3 = 0;
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < W; b++) {
#pragma HLS pipeline
                if (LONG_JUMP[i] & (one << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }
};

/**
 * @brief XoShiRo128StarStar is a 32-bit all-purpose, rock-solid generator.
 */
class XoShiRo128StarStar {
   private:
    const static int W = 32;
    int s[4];
    const unsigned int one = 1;

    ap_uint<W> rotl(ap_uint<W> x, int k) { return (x << k) | (x >> (W - k)); }

   public:
    /**
     * @brief default constructor
     */
    XoShiRo128StarStar() {
#pragma HLS inline
#pragma HLS array_partition variable = s complete
    }

    /**
     * @brief init initialize seeds
     * @param seedIn input seeds
     */
    void init(unsigned int* seedIn) {
        for (int i = 0; i < 4; i++) {
#pragma HLS pipeline
            s[i] = seedIn[i];
        }
    }

    /**
     * @brief each call of next() generate a pseudorandom number
     * @return return a pseudorandom number
     */
    ap_uint<W> next(void) {
#pragma HLS pipeline
        ap_uint<W> result = rotl(s[1] * 5, 7) * 9;

        ap_uint<W> t = s[1] << 9;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 11);

        return result;
    }

    /**
     * @brief the jump function is equivalent to 2^64 calls to next(); it can be used to generate 2^64 non-overlapping
     * subsequences for parallel computations.
     */
    void jump(void) {
        ap_uint<W> JUMP[] = {0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b};

        ap_uint<W> s0 = 0;
        ap_uint<W> s1 = 0;
        ap_uint<W> s2 = 0;
        ap_uint<W> s3 = 0;
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < W; b++) {
#pragma HLS pipeline
                if (JUMP[i] & (one << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }

    /**
     * @brief the long-jump function is equivalent to 2^96 calls to next(); it can be used to generate 2^32 starting
     * points, from each of which jump() will generate 2^32 non-overlapping subsequences for parallel distributed
     * computations.
     */
    void long_jump(void) {
        ap_uint<W> LONG_JUMP[] = {0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662};

        ap_uint<W> s0 = 0;
        ap_uint<W> s1 = 0;
        ap_uint<W> s2 = 0;
        ap_uint<W> s3 = 0;
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < W; b++) {
#pragma HLS pipeline
                if (LONG_JUMP[i] & (one << b)) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                    s2 ^= s[2];
                    s3 ^= s[3];
                }
                next();
            }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
    }
};
}
}
#endif
