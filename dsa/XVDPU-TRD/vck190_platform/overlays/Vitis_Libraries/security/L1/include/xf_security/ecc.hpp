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
 * @file ecc.hpp
 * @brief Header file for Elliptic-curve cryptography related functions.
 * This file is part of Vitis Security Library.
 */

#ifndef _XF_SECURITY_ECC_HPP_
#define _XF_SECURITY_ECC_HPP_

#include <ap_int.h>
#include "xf_security/modular.hpp"

namespace xf {
namespace security {

/**
 * @brief Elliptic-curve cryptography. This class template provide basic operators in ECC processing.
 *
 * @tparam W Bit length of Elliptic curve parameters. Parameters should all be smaller than 2^W.
 */
template <int W>
class ecc {
   public:
    /// Elliptic-curve definition parameter for y^2 = x^3 + ax + b in GF(p)
    ap_uint<W> a;
    /// Elliptic-curve definition parameter for y^2 = x^3 + ax + b in GF(p)
    ap_uint<W> b;
    /// Elliptic-curve definition parameter for y^2 = x^3 + ax + b in GF(p)
    ap_uint<W> p;

    ecc() {
#pragma HLS inline
    }

    /**
     * @brief perform point addition in Elliptic-curve ( R = P + Q)
     *
     * @param Px X coordinate of point P. Should be smaller than p.
     * @param Py Y coordinate of point P. Should be smaller than p.
     * @param Qx X coordinate of point Q. Should be smaller than p.
     * @param Qy Y coordinate of point Q. Should be smaller than p.
     * @param Rx X coordinate of point R. Should be smaller than p.
     * @param Ry Y coordinate of point R. Should be smaller than p.
     */
    void add(ap_uint<W> Px, ap_uint<W> Py, ap_uint<W> Qx, ap_uint<W> Qy, ap_uint<W>& Rx, ap_uint<W>& Ry) {
        if (Qx == 0 && Qy == 0) { // Q is zero
            Rx = Px;
            Ry = Py;
        } else if (Px == Qx && (Py + Qy) == p) {
            Rx = 0;
            Ry = 0;
        } else {
            ap_uint<W> lamda, lamda_d;
            if (Px == Qx && Py == Qy) {
                lamda = xf::security::internal::productMod<W>(Px, Px, p);
                lamda = xf::security::internal::productMod<W>(lamda, 3, p);
                lamda = xf::security::internal::addMod<W>(lamda, a, p);
                lamda_d = xf::security::internal::productMod<W>(Py, 2, p);
            } else {
                lamda = xf::security::internal::subMod<W>(Qy, Py, p);
                lamda_d = xf::security::internal::subMod<W>(Qx, Px, p);
            }
            lamda_d = xf::security::internal::modularInv<W>(lamda_d, p);
            lamda = xf::security::internal::productMod<W>(lamda, lamda_d, p);

            ap_uint<W> lamda_sqr = xf::security::internal::productMod<W>(lamda, lamda, p);

            ap_uint<W> resX, resY;
            resX = xf::security::internal::subMod<W>(lamda_sqr, Px, p);
            resX = xf::security::internal::subMod<W>(resX, Qx, p);

            resY = xf::security::internal::subMod<W>(Px, resX, p);
            resY = xf::security::internal::productMod<W>(lamda, resY, p);
            resY = xf::security::internal::subMod<W>(resY, Py, p);

            Rx = resX;
            Ry = resY;
        }
    }

    /**
     * @brief perform point multiply scalar in Elliptic-curve ( R = P * k)
     *
     * @param Px X coordinate of point P. Should be smaller than p.
     * @param Py Y coordinate of point P. Should be smaller than p.
     * @param k A scalar in GF(p). Should be smaller than p.
     * @param Rx X coordinate of point R. Should be smaller than p.
     * @param Ry Y coordinate of point R. Should be smaller than p.
     */
    void dotProduct(ap_uint<W> Px, ap_uint<W> Py, ap_uint<W> k, ap_uint<W>& Rx, ap_uint<W>& Ry) {
        ap_uint<W> resX = 0;
        ap_uint<W> resY = 0;

        for (int i = 0; i < W; i++) {
            if (k[i] == 1) {
                add(Px, Py, resX, resY, resX, resY);
            }
            add(Px, Py, Px, Py, Px, Py);
        }

        Rx = resX;
        Ry = resY;
    }

    /**
     * @brief Generate Public Key point P from Generation point G and private key
     *
     * @param Gx X coordinate of point G. Should be smaller than p.
     * @param Gy Y coordinate of point G. Should be smaller than p.
     * @param privateKey Private Key. Should be smaller than p.
     * @param Px X coordinate of point P. Should be smaller than p.
     * @param Py Y coordinate of point P. Should be smaller than p.
     */
    void generatePublicKey(ap_uint<W> Gx, ap_uint<W> Gy, ap_uint<W> privateKey, ap_uint<W>& Px, ap_uint<W>& Py) {
        dotProduct(Gx, Gy, privateKey, Px, Py);
    }

    /**
     * @brief Setup parameters for Elliptic-curve of y^2 = x^3 + ax + b in GF(p)
     *
     * @param inputA Parameter a for y^2 = x^3 + ax + b in GF(p)
     * @param inputB Parameter b for y^2 = x^3 + ax + b in GF(p)
     * @param inputP Parameter p for y^2 = x^3 + ax + b in GF(p)
     */
    void init(ap_uint<W> inputA, ap_uint<W> inputB, ap_uint<W> inputP) {
        a = inputA;
        b = inputB;
        p = inputP;
    }

    /**
     * @brief Encrypt a message point PM, using public key point P, generation point of Curve G and a random Key.
     * It will produce point pair {C1, C2} as encrypted message.
     *
     * @param Gx X coordinate of Curve Generation Point G. Should be smaller than p.
     * @param Gy Y coordinate of Curve Generation Point G. Should be smaller than p.
     * @param Px X coordinate of Public Key P. Should be smaller than p.
     * @param Py Y coordinate of Public Key P. Should be smaller than p.
     * @param randomKey random key for encryption. Should be smaller than p.
     * @param PMx X coordinate of Plain message. Should be smaller than p.
     * @param PMy Y coordinate of Plain message. Should be smaller than p.
     * @param C1x X coordinate of Point C1 in encrypted message point pair. Should be smaller than p.
     * @param C1y Y coordinate of Point C1 in encrypted message point pair. Should be smaller than p.
     * @param C2x X coordinate of Point C2 in encrypted message point pair. Should be smaller than p.
     * @param C2y Y coordinate of Point C2 in encrypted message point pair. Should be smaller than p.
     */
    void encrypt(ap_uint<W> Gx,
                 ap_uint<W> Gy,
                 ap_uint<W> Px,
                 ap_uint<W> Py,
                 ap_uint<W> randomKey,
                 ap_uint<W> PMx,
                 ap_uint<W> PMy,
                 ap_uint<W>& C1x,
                 ap_uint<W>& C1y,
                 ap_uint<W>& C2x,
                 ap_uint<W>& C2y) {
        dotProduct(Gx, Gy, randomKey, C1x, C1y);
        ap_uint<W> Tx, Ty;
        dotProduct(Px, Py, randomKey, Tx, Ty);
        add(Tx, Ty, PMx, PMy, C2x, C2y);
    }

    /**
     * @brief Decrypt an encrypted message point pair {C1, C2} with privateKey
     * It will produce plain message point PM.
     *
     * @param C1x X coordinate of Point C1 in encrypted message point pair. Should be smaller than p.
     * @param C1y Y coordinate of Point C1 in encrypted message point pair. Should be smaller than p.
     * @param C2x X coordinate of Point C2 in encrypted message point pair. Should be smaller than p.
     * @param C2y Y coordinate of Point C2 in encrypted message point pair. Should be smaller than p.
     * @param privateKey Private Key.
     * @param PMx X coordinate of Plain message. Should be smaller than p.
     * @param PMy Y coordinate of Plain message. Should be smaller than p.
     */
    void decrypt(ap_uint<W> C1x,
                 ap_uint<W> C1y,
                 ap_uint<W> C2x,
                 ap_uint<W> C2y,
                 ap_uint<W> privateKey,
                 ap_uint<W>& PMx,
                 ap_uint<W>& PMy) {
        ap_uint<W> Tx, Ty;
        dotProduct(C1x, C1y, privateKey, Tx, Ty);
        Ty = p - Ty;
        add(Tx, Ty, C2x, C2y, PMx, PMy);
    }
};
} // namespace security
} // namespace xf

#endif
