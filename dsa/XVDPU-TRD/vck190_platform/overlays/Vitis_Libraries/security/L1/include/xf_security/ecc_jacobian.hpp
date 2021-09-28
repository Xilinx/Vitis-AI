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
using namespace internal;
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

    void toJacobian(ap_uint<W> x, ap_uint<W> y, ap_uint<W>& X, ap_uint<W>& Y, ap_uint<W>& Z) {
#pragma HLS inline
        X = x;
        Y = y;
        Z = 1;
    }

    void fromJacobian(ap_uint<W> X, ap_uint<W> Y, ap_uint<W> Z, ap_uint<W>& x, ap_uint<W>& y) {
#pragma HLS inline
        ap_uint<W> ZInv = modularInv<W>(Z, p);
        ap_uint<W> ZInv_2 = productMod<W>(ZInv, ZInv, p);
        ap_uint<W> ZInv_3 = productMod<W>(ZInv_2, ZInv, p);
        x = productMod<W>(X, ZInv_2, p);
        y = productMod<W>(Y, ZInv_3, p);
    }

    void addJacobian(ap_uint<W> X1,
                     ap_uint<W> Y1,
                     ap_uint<W> Z1,
                     ap_uint<W> X2,
                     ap_uint<W> Y2,
                     ap_uint<W> Z2,
                     ap_uint<W>& X3,
                     ap_uint<W>& Y3,
                     ap_uint<W>& Z3) {
#pragma HLS inline
        ap_uint<W> I1 = productMod<W>(Z1, Z1, p); // Z1^2
        ap_uint<W> I2 = productMod<W>(Z2, Z2, p); // Z2^2
        ap_uint<W> J1 = productMod<W>(I1, Z1, p); // Z1^3
        ap_uint<W> J2 = productMod<W>(I2, Z2, p); // Z2^3
        ap_uint<W> U1 = productMod<W>(X1, I2, p); // X1*Z2^2
        ap_uint<W> U2 = productMod<W>(X2, I1, p); // X2*Z1^2
        ap_uint<W> H = subMod<W>(U1, U2, p);      // X1*Z2^2 - X2*Z1^2
        ap_uint<W> F = addMod<W>(H, H, p);
        F = productMod<W>(F, F, p);
        ap_uint<W> K1 = productMod<W>(Y1, J2, p); // Y1*Z2^3
        ap_uint<W> K2 = productMod<W>(Y2, J1, p); // Y1*Z2^3
        ap_uint<W> V = productMod<W>(U1, F, p);
        ap_uint<W> G = productMod<W>(F, H, p);
        ap_uint<W> R = subMod<W>(K1, K2, p);
        R = addMod<W>(R, R, p);

        if (Z2 == 0) {
            X3 = X1;
            Y3 = Y1;
            Z3 = Z1;
        } else if (Z1 == 0) {
            X3 = X2;
            Y3 = Y2;
            Z3 = Z2;
        } else if (addMod<W>(K1, K2, p) == 0) {
            X3 = 1;
            Y3 = 1;
            Z3 = 0;
        } else {
            ap_uint<W> tmpX = productMod<W>(R, R, p);
            ap_uint<W> tmp2V = addMod<W>(V, V, p);
            tmpX = addMod<W>(tmpX, G, p);
            X3 = subMod<W>(tmpX, tmp2V, p);

            ap_uint<W> tmp2 = subMod<W>(V, X3, p);
            tmp2 = productMod<W>(tmp2, R, p);
            ap_uint<W> tmp4 = productMod<W>(K1, G, p);
            tmp4 = addMod<W>(tmp4, tmp4, p);
            Y3 = subMod<W>(tmp2, tmp4, p);

            ap_uint<W> tmp5 = addMod<W>(Z1, Z2, p);
            tmp5 = productMod<W>(tmp5, tmp5, p);
            ap_uint<W> tmp6 = addMod<W>(I1, I2, p);
            ap_uint<W> tmp7 = subMod<W>(tmp5, tmp6, p);
            Z3 = productMod<W>(tmp7, H, p);
        }
    }

    void doubleJacobian(ap_uint<W> X1, ap_uint<W> Y1, ap_uint<W> Z1, ap_uint<W>& X2, ap_uint<W>& Y2, ap_uint<W>& Z2) {
#pragma HLS inline
        ap_uint<W> N = productMod<W>(Z1, Z1, p);
        ap_uint<W> E = productMod<W>(Y1, Y1, p);
        ap_uint<W> B = productMod<W>(X1, X1, p);
        ap_uint<W> L = productMod<W>(E, E, p);

        ap_uint<W> tmp1 = addMod<W>(X1, E, p);
        tmp1 = productMod<W>(tmp1, tmp1, p);
        ap_uint<W> tmp2 = addMod<W>(B, L, p);
        ap_uint<W> tmp3 = subMod<W>(tmp1, tmp2, p);
        ap_uint<W> S = addMod<W>(tmp3, tmp3, p);

        ap_uint<W> tmp4 = productMod<W>(N, N, p);
        tmp4 = productMod<W>(tmp4, a, p);
        ap_uint<W> tmp5 = addMod<W>(B, B, p);
        tmp5 = addMod<W>(tmp5, B, p);
        ap_uint<W> M = addMod<W>(tmp5, tmp4, p);

        ap_uint<W> tmp6 = addMod<W>(S, S, p);
        ap_uint<W> tmp7 = productMod<W>(M, M, p);
        X2 = subMod<W>(tmp7, tmp6, p);

        ap_uint<W> tmp8 = subMod<W>(S, X2, p);
        tmp8 = productMod<W>(tmp8, M, p);
        ap_uint<W> tmp9 = addMod<W>(L, L, p);
        tmp9 = addMod<W>(tmp9, tmp9, p);
        tmp9 = addMod<W>(tmp9, tmp9, p);
        Y2 = subMod<W>(tmp8, tmp9, p);

        ap_uint<W> tmp10 = addMod<W>(Y1, Z1, p);
        tmp10 = productMod<W>(tmp10, tmp10, p);
        ap_uint<W> tmp11 = addMod<W>(E, N, p);
        Z2 = subMod<W>(tmp10, tmp11, p);
    }

    void productJacobian(
        ap_uint<W> X1, ap_uint<W> Y1, ap_uint<W> Z1, ap_uint<W> k, ap_uint<W>& X2, ap_uint<W>& Y2, ap_uint<W>& Z2) {
#pragma HLS inline
        ap_uint<W> RX = 1;
        ap_uint<W> RY = 1;
        ap_uint<W> RZ = 0;

        for (int i = W - 1; i >= 0; i--) {
            doubleJacobian(RX, RY, RZ, RX, RY, RZ);
            if (k[i] == 1) {
                addJacobian(RX, RY, RZ, X1, Y1, Z1, RX, RY, RZ);
            }
        }
        X2 = RX;
        Y2 = RY;
        Z2 = RZ;
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
#pragma HLS inline
        ap_uint<W> X1, Y1, Z1, X2, Y2, Z2;
        toJacobian(Px, Py, X1, Y1, Z1);
        productJacobian(X1, Y1, Z1, k, X2, Y2, Z2);
        fromJacobian(X2, Y2, Z2, Rx, Ry);
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
