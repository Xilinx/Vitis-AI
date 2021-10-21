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
 *
 * @file dsa.hpp
 * @brief header file for Digital Signature Algorithm related functions, including signing and verifying.
 * It takes FIPS.186-4 as reference.
 * This file is part of Vitis Security Library.
 *
 */

#ifndef _XF_SECURITY_DSA_HPP_
#define _XF_SECURITY_DSA_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_security/modular.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace security {
/**
 * @brief Digital Signature. This class provide signing and verifying functions.
 * Generation of domain parameters, key pairs and per-message secrect number requires
 * key management mechanism and is not covered in this class.
 *
 * @tparam L bit length of prime modulus.
 * @tparam N bit length of prime divisor.
 * Selection of L and N is fixed in FIPS.186-4, section 4.2.
 * Choice of pair {L, N} are: {1024, 160}, {2048, 224}, {2048, 256}, {3072, 256}
 * @tparam H bit length of Digest value
 */
template <int L, int N, int H>
class dsa {
   public:
    dsa() {
#pragma HLS inline
    }

    /// Prime modulus, a domain parameter
    ap_uint<L> p;
    /// Prime divisor, a domain parameter
    ap_uint<N> q;
    /// generator of a subgroup of order q in GF(p), a domain parameter
    ap_uint<L> g;
    /// private key
    ap_uint<N> x;
    /// public key
    ap_uint<L> y;
    /// a parameter determined by p, rMod = 2^(2*L) mod p
    ap_uint<L> rMod;

    /**
     * @brief Set up domain parameters for DSA signing when a set of new domain parameter will be used.
     * rMod is not provided which need to be calculated on Chip
     *
     * @param inputP Input prime modulus.
     * @param inputQ Input prime divisor.
     * @param inputG Input generator of a subgroup of order inputQ in GF(inputQ).
     * @param inputX Input private key
     */
    void updateSigningParam(ap_uint<L> inputP, ap_uint<N> inputQ, ap_uint<L> inputG, ap_uint<N> inputX) {
        p = inputP;
        q = inputQ;
        g = inputG;
        x = inputX;
        ap_uint<L* 2 + 1> tmp = 0;
        tmp[L * 2] = 1;
        rMod = xf::security::internal::simpleMod<L * 2 + 1, L>(tmp, p);
    }

    /**
     * @brief Set up domain parameters for DSA signing when a set of new domain parameter will be used.
     *
     * @param inputP Input prime modulus.
     * @param inputQ Input prime divisor.
     * @param inputG Input generator of a subgroup of order inputQ in GF(inputQ).
     * @param inputX Input private key
     * @param rMod Input rMode, provided by user.
     */
    void updateSigningParam(
        ap_uint<L> inputP, ap_uint<N> inputQ, ap_uint<L> inputG, ap_uint<N> inputX, ap_uint<L> inputRMod) {
        p = inputP;
        q = inputQ;
        g = inputG;
        x = inputX;
        rMod = inputRMod;
    }

    /**
     * @brief Set up domain parameters for DSA Verifying when a set of new domain parameter will be used.
     * rMod is not provided which need to be calculated on Chip
     *
     * @param inputP Input prime modulus.
     * @param inputQ Input prime divisor.
     * @param inputG Input generator of a subgroup of order inputQ in GF(inputQ).
     * @param inputY Input public key
     */
    void updateVerifyingParam(ap_uint<L> inputP, ap_uint<N> inputQ, ap_uint<L> inputG, ap_uint<L> inputY) {
        p = inputP;
        q = inputQ;
        g = inputG;
        y = inputY;
        ap_uint<L* 2 + 1> tmp = 0;
        tmp[L * 2] = 1;
        rMod = xf::security::internal::simpleMod<L * 2 + 1, L>(tmp, p);
    }

    /**
     * @brief Set up domain parameters for DSA Verifying when a set of new domain parameter will be used.
     *
     * @param inputP Input prime modulus.
     * @param inputQ Input prime divisor.
     * @param inputG Input generator of a subgroup of order inputQ in GF(inputQ).
     * @param inputY Input public key
     * @param rMod Input rMode, provided by user.
     */
    void updateVerifyingParam(
        ap_uint<L> inputP, ap_uint<N> inputQ, ap_uint<L> inputG, ap_uint<L> inputY, ap_uint<L> inputRMod) {
        p = inputP;
        q = inputQ;
        g = inputG;
        y = inputY;
        rMod = inputRMod;
    }

    /**
     * @brief DSA signing function.
     *
     * @param digest Digest value of message to be signed.
     * @param k A per-message secret number.
     * @param r Element of signature pair.
     * @param s Element of signature pair. Pair(r, s) forms a complete signature pair of DSA.
     */
    void sign(ap_uint<H> digest, ap_uint<N> k, ap_uint<N>& r, ap_uint<N>& s) {
        // tmp = g^k mod p
        ap_uint<L> tmp1 = xf::security::internal::modularExp<L, N>(g, k, p, rMod);
        // r = (g^k mod p) mod q
        r = xf::security::internal::simpleMod<L, N>(tmp1, q);
        // z = leftmost min(N, H) bits of digest
        ap_uint<N> z;
        if (H > N) {
            z = digest.range(H - 1, H - N);
        } else {
            z = digest;
        }
        if (z >= q) {
            z -= q;
        }
        // tmp2 = (x * r) mod q
        ap_uint<N> tmp2 = xf::security::internal::productMod<N>(x, r, q);
        // tmp3 = (z + x * r) mod q
        ap_uint<N> tmp3 = xf::security::internal::addMod<N>(z, tmp2, q);
        // kInv = k^(-1) mod q
        ap_uint<N> kInv = xf::security::internal::modularInv<N>(k, q);
        // s = (k^-1 * (z + x * r)) mod q
        s = xf::security::internal::productMod<N>(kInv, tmp3, q);
    }

    /**
     * @brief DSA verifying function.
     * It returns true if verified, otherwise false.
     *
     * @param digest Digest value of message to be verified.
     * @param r Element of signature pair.
     * @param s Element of signature pair. Pair(r, s) forms a complete signature pair of DSA.
     */
    bool verify(ap_uint<H> digest, ap_uint<N> r, ap_uint<N> s) {
        // check if r and s are less than q
        bool res = true;
        if (r >= q || s >= q) {
            res = false;
        }
        // calculate w = s^-1 mod q
        ap_uint<N> w = xf::security::internal::modularInv<N>(s, q);
        // calculate z = leftmost min(N, H) bits of digest
        ap_uint<N> z;
        if (H > N) {
            z = digest.range(H - 1, H - N);
        } else {
            z = digest;
        }
        if (z >= q) {
            z -= q;
        }
        // calcualte u1 = z * w mod q
        ap_uint<N> u1 = xf::security::internal::productMod<N>(z, w, q);
        // calculate u2 = r * w mod q
        ap_uint<N> u2 = xf::security::internal::productMod<N>(r, w, q);
        // calculate v = (g^u1 * y^u2 mod p) mod q
        // tmp1 = g^u1 mod p
        ap_uint<L> tmp1 = xf::security::internal::modularExp<L, N>(g, u1, p, rMod);
        // tmp1 = y^u2 mod p
        ap_uint<L> tmp2 = xf::security::internal::modularExp<L, N>(y, u2, p, rMod);
        // tmp3 = g^u1 * y^u2 mod p
        ap_uint<L> tmp3 = xf::security::internal::productMod<L>(tmp1, tmp2, p);
        // v = (g^u1 * y^u2 mod p) mod q
        ap_uint<N> v = xf::security::internal::simpleMod<L, N>(tmp3, q);
        // return false if v != r
        if (v != r) {
            res = false;
        }
        return res;
    }
};

} // namespace security`
} // namespace xf

#endif
