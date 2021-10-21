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
 * @file ecdsa.hpp
 * @brief header file for Elliptic Curve Digital Signature Algorithm
 * related function. Now it support curve secp256k1.
 * This file is part of Vitis Security Library.
 */

#ifndef _XF_SECURITY_ECDSA_HPP_
#define _XF_SECURITY_ECDSA_HPP_

#include <ap_int.h>
#include "xf_security/modular.hpp"

namespace xf {
namespace security {
/**
 * @brief Elliptic Curve Digital Signature Algorithm on curve secp256k1.
 * This class provide signing and verifying functions.
 *
 * @tparam HashW Bit Width of digest that used for signting and verifying.
 */
template <int HashW>
class ecdsaSecp256k1 {
   public:
    // Elliptic-curve definition parameter for y^2 = x^3 + ax + b in GF(p)
    ap_uint<256> a;
    /// Elliptic-curve definition parameter for y^2 = x^3 + ax + b in GF(p)
    ap_uint<256> b;
    /// Elliptic-curve definition parameter for y^2 = x^3 + ax + b in GF(p)
    ap_uint<256> p;

    /// X coordinate of generation point of curve secp256k1.
    ap_uint<256> Gx;
    /// Y coordinate of generation point of curve secp256k1.
    ap_uint<256> Gy;
    /// Order of curve secp256k1.
    ap_uint<256> n;

    ecdsaSecp256k1() {
#pragma HLS inline
    }

    ap_uint<256> productMod_p(ap_uint<256> a, ap_uint<256> b) {
        ap_uint<33> negP = ap_uint<33>("0x01000003D1");
        ap_uint<256> P = ap_uint<256>("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

        ap_uint<128> aH = a.range(255, 128);
        ap_uint<128> aL = a.range(127, 0);
        ap_uint<128> bH = b.range(255, 128);
        ap_uint<128> bL = b.range(127, 0);

        ap_uint<256> aLbH = aL * bH;
        ap_uint<256> aHbL = aH * bL;
        ap_uint<256> aHbH = aH * bH;
        ap_uint<257> aLbL = aL * bL;
        ap_uint<257> mid = aLbH + aHbL;

        ap_uint<256> FL = 0;
        FL.range(255, 128) = mid.range(127, 0);
        aLbL += FL;
        if (aLbL >= P) {
            aLbL -= P;
        }
        aHbH += ap_uint<129>(mid.range(256, 128));
        if (aHbH >= P) {
            aHbH -= P;
        }

        ap_uint<257> res = 0;
        for (int i = 32; i >= 0; i--) {
#pragma HLS pipeline
            res <<= 1;
            if (res >= P) {
                res -= P;
            }
            if (negP[i] == 1) {
                res += aHbH;
                if (res >= P) {
                    res -= P;
                }
            }
        }
        res += aLbL;
        if (res >= P) {
            res -= P;
        }
        return res;
    }

    ap_uint<256> productMod_n(ap_uint<256> a, ap_uint<256> b) {
// XXX: a * b % n is only called a few times, no need to use specialized version
#pragma HLS inline
        return xf::security::internal::productMod<256>(a, b, n);
    }

    void add(ap_uint<256> Px, ap_uint<256> Py, ap_uint<256> Qx, ap_uint<256> Qy, ap_uint<256>& Rx, ap_uint<256>& Ry) {
#pragma HLS inline
        if (Qx == 0 && Qy == 0) { // Q is zero
            Rx = Px;
            Ry = Py;
        } else if (Px == Qx && (Py + Qy) == p) {
            Rx = 0;
            Ry = 0;
        } else {
            ap_uint<256> lamda, lamda_d;
            if (Px == Qx && Py == Qy) {
                lamda = productMod_p(Px, Px);
                lamda = productMod_p(lamda, ap_uint<256>(3));
                lamda = xf::security::internal::addMod<256>(lamda, a, p);
                lamda_d = productMod_p(Py, ap_uint<256>(2));
            } else {
                lamda = xf::security::internal::subMod<256>(Qy, Py, p);
                lamda_d = xf::security::internal::subMod<256>(Qx, Px, p);
            }
            lamda_d = xf::security::internal::modularInv<256>(lamda_d, p);
            lamda = productMod_p(lamda, lamda_d);

            ap_uint<256> lamda_sqr = productMod_p(lamda, lamda);

            ap_uint<256> resX, resY;
            resX = xf::security::internal::subMod<256>(lamda_sqr, Px, p);
            resX = xf::security::internal::subMod<256>(resX, Qx, p);

            resY = xf::security::internal::subMod<256>(Px, resX, p);
            resY = productMod_p(lamda, resY);
            resY = xf::security::internal::subMod<256>(resY, Py, p);

            Rx = resX;
            Ry = resY;
        }
    }

    void toJacobian(ap_uint<256> x, ap_uint<256> y, ap_uint<256>& X, ap_uint<256>& Y, ap_uint<256>& Z) {
#pragma HLS inline
        X = x;
        Y = y;
        Z = 1;
    }

    void fromJacobian(ap_uint<256> X, ap_uint<256> Y, ap_uint<256> Z, ap_uint<256>& x, ap_uint<256>& y) {
#pragma HLS inline
        if (Z == 0) {
            x = 0;
            y = 0;
        } else {
            ap_uint<256> ZInv = xf::security::internal::modularInv<256>(Z, p);
            ap_uint<256> ZInv_2 = productMod_p(ZInv, ZInv);
            ap_uint<256> ZInv_3 = productMod_p(ZInv_2, ZInv);
            x = productMod_p(X, ZInv_2);
            y = productMod_p(Y, ZInv_3);
        }
    }

    void addJacobian(ap_uint<256> X1,
                     ap_uint<256> Y1,
                     ap_uint<256> Z1,
                     ap_uint<256> X2,
                     ap_uint<256> Y2,
                     ap_uint<256> Z2,
                     ap_uint<256>& X3,
                     ap_uint<256>& Y3,
                     ap_uint<256>& Z3) {
#pragma HLS inline
        ap_uint<256> I1 = productMod_p(Z1, Z1);
        ap_uint<256> I2 = productMod_p(Z2, Z2);
        ap_uint<256> J1 = productMod_p(I1, Z1);
        ap_uint<256> J2 = productMod_p(I2, Z2);
        ap_uint<256> U1 = productMod_p(X1, I2);
        ap_uint<256> U2 = productMod_p(X2, I1);
        ap_uint<256> H = xf::security::internal::subMod<256>(U1, U2, p);
        ap_uint<256> F = xf::security::internal::addMod<256>(H, H, p);
        F = productMod_p(F, F);
        ap_uint<256> K1 = productMod_p(Y1, J2);
        ap_uint<256> K2 = productMod_p(Y2, J1);
        ap_uint<256> V = productMod_p(U1, F);
        ap_uint<256> G = productMod_p(F, H);
        ap_uint<256> R = xf::security::internal::subMod<256>(K1, K2, p);
        R = xf::security::internal::addMod<256>(R, R, p);

        if (Z2 == 0) {
            X3 = X1;
            Y3 = Y1;
            Z3 = Z1;
        } else if (Z1 == 0) {
            X3 = X2;
            Y3 = Y2;
            Z3 = Z2;
        } else if (xf::security::internal::addMod<256>(K1, K2, p) == 0) {
            X3 = 1;
            Y3 = 1;
            Z3 = 0;
        } else {
            ap_uint<256> tmpX = productMod_p(R, R);
            ap_uint<256> tmp2V = xf::security::internal::addMod<256>(V, V, p);
            tmpX = xf::security::internal::addMod<256>(tmpX, G, p);
            X3 = xf::security::internal::subMod<256>(tmpX, tmp2V, p);

            ap_uint<256> tmp2 = xf::security::internal::subMod<256>(V, X3, p);
            tmp2 = productMod_p(tmp2, R);
            ap_uint<256> tmp4 = productMod_p(K1, G);
            tmp4 = xf::security::internal::addMod<256>(tmp4, tmp4, p);
            Y3 = xf::security::internal::subMod<256>(tmp2, tmp4, p);

            ap_uint<256> tmp5 = xf::security::internal::addMod<256>(Z1, Z2, p);
            tmp5 = productMod_p(tmp5, tmp5);
            ap_uint<256> tmp6 = xf::security::internal::addMod<256>(I1, I2, p);
            ap_uint<256> tmp7 = xf::security::internal::subMod<256>(tmp5, tmp6, p);
            Z3 = productMod_p(tmp7, H);
        }
    }

    void doubleJacobian(
        ap_uint<256> X1, ap_uint<256> Y1, ap_uint<256> Z1, ap_uint<256>& X2, ap_uint<256>& Y2, ap_uint<256>& Z2) {
#pragma HLS inline
        ap_uint<256> N = productMod_p(Z1, Z1);
        ap_uint<256> E = productMod_p(Y1, Y1);
        ap_uint<256> B = productMod_p(X1, X1);
        ap_uint<256> L = productMod_p(E, E);

        ap_uint<256> tmp1 = xf::security::internal::addMod<256>(X1, E, p);
        tmp1 = productMod_p(tmp1, tmp1);
        ap_uint<256> tmp2 = xf::security::internal::addMod<256>(B, L, p);
        ap_uint<256> tmp3 = xf::security::internal::subMod<256>(tmp1, tmp2, p);
        ap_uint<256> S = xf::security::internal::addMod<256>(tmp3, tmp3, p);

        ap_uint<256> tmp4 = productMod_p(N, N);
        tmp4 = productMod_p(tmp4, a);
        ap_uint<256> tmp5 = xf::security::internal::addMod<256>(B, B, p);
        tmp5 = xf::security::internal::addMod<256>(tmp5, B, p);
        ap_uint<256> M = xf::security::internal::addMod<256>(tmp5, tmp4, p);

        ap_uint<256> tmp6 = xf::security::internal::addMod<256>(S, S, p);
        ap_uint<256> tmp7 = productMod_p(M, M);
        X2 = xf::security::internal::subMod<256>(tmp7, tmp6, p);

        ap_uint<256> tmp8 = xf::security::internal::subMod<256>(S, X2, p);
        tmp8 = productMod_p(tmp8, M);
        ap_uint<256> tmp9 = xf::security::internal::addMod<256>(L, L, p);
        tmp9 = xf::security::internal::addMod<256>(tmp9, tmp9, p);
        tmp9 = xf::security::internal::addMod<256>(tmp9, tmp9, p);
        Y2 = xf::security::internal::subMod<256>(tmp8, tmp9, p);

        ap_uint<256> tmp10 = xf::security::internal::addMod<256>(Y1, Z1, p);
        tmp10 = productMod_p(tmp10, tmp10);
        ap_uint<256> tmp11 = xf::security::internal::addMod<256>(E, N, p);
        Z2 = xf::security::internal::subMod<256>(tmp10, tmp11, p);
    }

    void productJacobian(ap_uint<256> X1,
                         ap_uint<256> Y1,
                         ap_uint<256> Z1,
                         ap_uint<256> k,
                         ap_uint<256>& X2,
                         ap_uint<256>& Y2,
                         ap_uint<256>& Z2) {
#pragma HLS inline
        ap_uint<256> RX = 1;
        ap_uint<256> RY = 1;
        ap_uint<256> RZ = 0;

        for (int i = 256 - 1; i >= 0; i--) {
            doubleJacobian(RX, RY, RZ, RX, RY, RZ);
            if (k[i] == 1) {
                addJacobian(RX, RY, RZ, X1, Y1, Z1, RX, RY, RZ);
            }
        }
        X2 = RX;
        Y2 = RY;
        Z2 = RZ;
    }

    void dotProduct(ap_uint<256> Px, ap_uint<256> Py, ap_uint<256> k, ap_uint<256>& Rx, ap_uint<256>& Ry) {
#pragma HLS inline
        ap_uint<256> X1, Y1, Z1, X2, Y2, Z2;
        toJacobian(Px, Py, X1, Y1, Z1);
        productJacobian(X1, Y1, Z1, k, X2, Y2, Z2);
        fromJacobian(X2, Y2, Z2, Rx, Ry);
    }

    /**
     * @brief Setup parameters for curve y^2 = x^3 + ax + b in GF(p)
     */
    void init() {
        a = ap_uint<256>("0x0");
        b = ap_uint<256>("0x7");
        p = ap_uint<256>("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        Gx = ap_uint<256>("0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
        Gy = ap_uint<256>("0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
        n = ap_uint<256>("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    };

    /**
     * @brief Generate Public Key point Q from private key
     *
     * @param privateKey Private Key.
     * @param Qx X coordinate of point Q.
     * @param Qy Y coordinate of point Q.
     */
    void generatePubKey(ap_uint<256> privateKey, ap_uint<256>& Qx, ap_uint<256>& Qy) {
        dotProduct(Gx, Gy, privateKey, Qx, Qy);
    }

    /**
     * @brief signing function.
     * It will return true if input parameters are legal, otherwise return false.
     *
     * @param hash Digest value of message to be signed.
     * @param k A random key to sign the message, should kept different each time to be used.
     * @param privateKey Private Key to sign the message
     * @param r part of signing pair {r, s}
     * @param s part of signing pair {r, s}
     */
    bool sign(ap_uint<HashW> hash, ap_uint<256> k, ap_uint<256> privateKey, ap_uint<256>& r, ap_uint<256>& s) {
        ap_uint<256> x, y;
        dotProduct(Gx, Gy, k, x, y); //(x, y) = k * (Gx, Gy);

        if (x >= n) {
            x -= n;
        } // x = x mod n

        if (x == 0) {
            return false;
        } else {
            r = x;

            ap_uint<256> z;
            if (HashW >= 256) {
                z = hash.range(HashW - 1, HashW - 256);
            } else {
                z = hash;
            }
            if (z >= n) {
                z -= n;
            }

            if (privateKey >= n) {
                privateKey -= n;
            }

            ap_uint<256> kInv = xf::security::internal::modularInv<256>(k, n);
            ap_uint<256> rda = productMod_n(x, privateKey);
            rda = xf::security::internal::addMod<256>(rda, z, n);

            s = productMod_n(kInv, rda);

            if (s == 0) {
                return false;
            } else {
                return true;
            }
        }
    }

    /**
     * @brief verifying function.
     * It will return true if verified, otherwise false.
     *
     * @param r part of signing pair {r, s}
     * @param s part of signing pair {r, s}
     * @param hash Digest value of message to be signed.
     * @param Px X coordinate of public key point P.
     * @param Py Y coordinate of public key point P.
     */
    bool verify(ap_uint<256> r, ap_uint<256> s, ap_uint<HashW> hash, ap_uint<256> Px, ap_uint<256> Py) {
        if (Px == 0 && Py == 0) {
            return false; // return false if public key is zero.
        } else {
            ap_uint<256> tx1 = productMod_p(Px, Px);
            tx1 = productMod_p(tx1, Px);

            ap_uint<256> tx2 = productMod_p(Px, a);
            tx2 = xf::security::internal::addMod<256>(tx2, b, p);

            ap_uint<256> tx3 = xf::security::internal::addMod<256>(tx2, tx1, p);

            ap_uint<256> ty = productMod_p(Py, Py);

            if (ty != tx3) { // return false if public key is not on the curve.
                return false;
            } else {
                ap_uint<256> nPx = 0;
                ap_uint<256> nPy = 0;

                dotProduct(Px, Py, n, nPx, nPy);

                if (nPx != 0 || nPy != 0) { // return false if public key * n is not zero.
                    return false;
                } else { // public key is valid, begin to check signature
                    if (r == 0 || r >= n || s == 0 || s >= n) {
                        return false;
                    } else {
                        ap_uint<256> z;
                        if (HashW >= 256) {
                            z = hash.range(HashW - 1, HashW - 256);
                        } else {
                            z = hash;
                        }
                        if (z >= n) {
                            z -= n;
                        }

                        ap_uint<256> sInv = xf::security::internal::modularInv<256>(s, n);

                        ap_uint<256> u1 = productMod_n(sInv, z);
                        ap_uint<256> u2 = productMod_n(sInv, r);

                        ap_uint<256> t1x, t1y, t2x, t2y;
                        dotProduct(Gx, Gy, u1, t1x, t1y);
                        dotProduct(Px, Py, u2, t2x, t2y);

                        ap_uint<256> x, y;
                        add(t1x, t1y, t2x, t2y, x, y);

                        if (x == 0 && y == 0) {
                            return false;
                        } else {
                            if (r == x) {
                                return true;
                            } else {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return true; // XXX:remove
    }
};

} // namespace security
} // namespace xf

#endif
