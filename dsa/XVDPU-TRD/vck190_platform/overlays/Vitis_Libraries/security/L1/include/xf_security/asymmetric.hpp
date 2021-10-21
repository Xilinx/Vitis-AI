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

#ifndef _XF_SECURITY_ASYMMETRIC_CRYPTOGRAPHY_HPP_
#define _XF_SECURITY_ASYMMETRIC_CRYPTOGRAPHY_HPP_

#include <ap_int.h>
#include "xf_security/modular.hpp"

namespace xf {
namespace security {

/**
 * @brief RSA encryption/decryption class
 *
 * @tparam N BitWdith of modulus of key.
 * @tparam L BitWdith of exponents of RSA encryption and decryption
 */
template <int N, int L>
class rsa {
   public:
    ap_uint<L> exponent;
    ap_uint<N> modulus;
    ap_uint<N> rMod;

    /**
     * @brief Update key before use it to encrypt message
     *
     * @param inputModulus Modulus in RSA public key.
     * @param inputExponent Exponent in RSA public key or private key.
     */
    void updateKey(ap_uint<N> inputModulus, ap_uint<L> inputExponent) {
        modulus = inputModulus;
        exponent = inputExponent;

        ap_uint<N + 1> tmp = 0;
        tmp[N] = 1;
        tmp %= inputModulus;

        rMod = xf::security::internal::productMod<N>(tmp, tmp, inputModulus);
    }

    /**
     * @brief Update key before use it to encrypt message
     *
     * @param inputModulus Modulus in RSA public key.
     * @param inputExponent Exponent in RSA public key or private key.
     * @param inputRMod 2^(2 * N) mod modulus, pre-calculated by user
     */
    void updateKey(ap_uint<N> inputModulus, ap_uint<L> inputExponent, ap_uint<N> inputRMod) {
        modulus = inputModulus;
        exponent = inputExponent;
        rMod = inputRMod;
    }

    /**
     * @brief Encrypt message and get result. It does not include any padding scheme
     *
     * @param message Message to be encrypted/decrypted
     * @param result Generated encrypted/decrypted result.
     */
    void process(ap_uint<N> message, ap_uint<N>& result) {
        result = xf::security::internal::modularExp<N, L>(message, exponent, modulus, rMod);
    }
};

} // namespace security
} // namespace xf
#endif
