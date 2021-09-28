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
 * @file eddsa.hpp
 * @brief header file for Edwards-curve Digital Signature Algorithm related function.
 * Now it support curve ed25519.
 * This file is part of Vitis Security Library.
 */

#ifndef _XF_SECURITY_EDDSA_HPP_
#define _XF_SECURITY_EDDSA_HPP_

#include <ap_int.h>
#include "xf_security/modular.hpp"
#include "xf_security/sha512_t.hpp"

namespace xf {
namespace security {

/**
 * @brief Edwards-curve Digital Signature Algorithm on curve ed25519.
 * It take RFC 8032 "Edwards-Curve Digital Signature Algorithm (EdDSA)" as reference.
 * This class provide signing and verifying functions.
 */
class eddsaEd25519 {
   public:
    /// ed25519 related curve parameters.
    const int b = 256;
    const int c = 3;
    const int n = 254;
    const int a = -1;
    const ap_uint<256> Bx = ap_uint<256>("0x216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A");
    const ap_uint<256> By = ap_uint<256>("0x6666666666666666666666666666666666666666666666666666666666666658");
    const ap_uint<256> L = ap_uint<256>("0x1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED");
    const ap_uint<256> p = ap_uint<256>("0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED");
    const ap_uint<256> d = ap_uint<256>("0x52036CEE2B6FFE738CC740797779E89800700A4D4141D8AB75EB4DCA135978A3");
    const ap_uint<256> p_5_d8 =
        ap_uint<256>("0x0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD"); // (p - 5)/8
    const ap_uint<256> sqrt_n1 =
        ap_uint<256>("0x2B8324804FC1DF0B2B4D00993DFBD7A72F431806AD2FE478C4EE1B274A0EA0B0"); // sqrt(-1)
    const ap_uint<256> rMod = ap_uint<256>("0x05A4");                                       // 2 ^ 512 mod p
    ap_uint<64> head[8];

    eddsaEd25519() {
#pragma HLS inline
    }

    /**
     * @brief Compress a point (x, y) on curve to its compressed form
     *
     * @param x X coordinate of point.
     * @param y Y coordinate of point.
     * @param res compressed point representation.
     */
    void compress(ap_uint<256> x, ap_uint<256> y, ap_uint<256>& res) {
        ap_uint<256> tmp = y;
        tmp[255] = x[0];
        res = tmp;
    }

    /**
     * @brief Calculate square root of u/v.
     *
     * @param u Input u of u/v to calculate square root.
     * @param v Input u of u/v to calculate square root.
     * @param sqrt_a Square root of u/v.
     */
    bool modularSqrt(ap_uint<256> u, ap_uint<256> v, ap_uint<256>& sqrt_a) {
        ap_uint<256> uv = xf::security::internal::productMod<256>(u, v, p);
        ap_uint<256> v2 = xf::security::internal::productMod<256>(v, v, p);
        ap_uint<256> v4 = xf::security::internal::productMod<256>(v2, v2, p);
        ap_uint<256> uv3 = xf::security::internal::productMod<256>(uv, v2, p);
        ap_uint<256> uv7 = xf::security::internal::productMod<256>(uv3, v4, p);
        ap_uint<256> tmp = xf::security::internal::modularExp<256, 256>(uv7, p_5_d8, p, rMod);
        tmp = xf::security::internal::productMod<256>(uv3, tmp, p);
        ap_uint<256> tmp_2 = xf::security::internal::productMod<256>(tmp, tmp, p);
        tmp_2 = xf::security::internal::productMod<256>(tmp_2, v, p);
        if (tmp_2 == u) {
            sqrt_a = tmp;
            return true;
        } else if (xf::security::internal::addMod<256>(tmp_2, u, p) == 0) {
            sqrt_a = xf::security::internal::productMod<256>(tmp, sqrt_n1, p);
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief Decompress a point (Px, Py) from its compressed representation.
     *
     * @param P compressed point representation.
     * @param Px X coordinate of the point.
     * @param Py Y coordinate of the point.
     */
    bool decompress(ap_uint<256> P, ap_uint<256>& Px, ap_uint<256>& Py) {
        Py = P.range(254, 0);
        ap_uint<256> y2 = xf::security::internal::productMod<256>(Py, Py, p);
        ap_uint<256> u = xf::security::internal::subMod<256>(y2, 1, p);
        ap_uint<256> v = xf::security::internal::productMod<256>(y2, d, p);
        v = xf::security::internal::addMod<256>(v, 1, p);
        ap_uint<256> sqrt_x;
        bool valid = modularSqrt(u, v, sqrt_x);

        if (P[255] == sqrt_x[0]) {
            Px = sqrt_x;
        } else {
            Px = p - sqrt_x;
        }
        return valid;
    }

   private:
    void writeWholeMsg(ap_uint<8> headLength,
                       ap_uint<128> msgLength,
                       hls::stream<ap_uint<64> >& msgStrm,
                       hls::stream<ap_uint<64> >& wholeMsgStrm,
                       hls::stream<ap_uint<128> >& wholeLenStrm,
                       hls::stream<bool>& endWholeLenStrm) {
        ap_uint<128> wholeLen = msgLength + headLength * 8;
        wholeLenStrm.write(wholeLen);
        endWholeLenStrm.write(false);
        endWholeLenStrm.write(true);
        for (int i = 0; i < headLength; i++) {
#pragma HLS pipeline II = 1
            wholeMsgStrm.write(head[i]);
        }
        for (ap_uint<128> i = 0; i < msgLength; i += 8) {
#pragma HLS pipeline II = 1
            wholeMsgStrm.write(msgStrm.read());
        }
    }

    void writeWholeMsg(ap_uint<8> headLength,
                       hls::stream<ap_uint<64> >& wholeMsgStrm,
                       hls::stream<ap_uint<128> >& wholeLenStrm,
                       hls::stream<bool>& endWholeLenStrm) {
        ap_uint<128> wholeLen = headLength * 8;
        wholeLenStrm.write(wholeLen);
        endWholeLenStrm.write(false);
        endWholeLenStrm.write(true);
        for (int i = 0; i < headLength; i++) {
#pragma HLS pipeline II = 1
            wholeMsgStrm.write(head[i]);
        }
    }

    void wrapperSha512(ap_uint<8> headLength,
                       ap_uint<128> msgLength,
                       hls::stream<ap_uint<64> >& msgStrm,
                       hls::stream<ap_uint<512> >& digestStrm,
                       hls::stream<bool>& endDigestStrm) { //
#pragma HLS dataflow
        hls::stream<ap_uint<64> > wholeMsgStrm("wrapper64Strm");
#pragma HLS stream variable = wholeMsgStrm depth = 16
        hls::stream<ap_uint<128> > wholeLenStrm;
#pragma HLS stream variable = wholeLenStrm depth = 1
        hls::stream<bool> endWholeLenStrm;
#pragma HLS stream variable = endWholeLenStrm depth = 2

        writeWholeMsg(headLength, msgLength, msgStrm, wholeMsgStrm, wholeLenStrm, endWholeLenStrm);
        xf::security::sha512<64>(wholeMsgStrm, wholeLenStrm, endWholeLenStrm, digestStrm, endDigestStrm);
    }

    void wrapperSha512(ap_uint<8> headLength,
                       hls::stream<ap_uint<512> >& digestStrm,
                       hls::stream<bool>& endDigestStrm) { //
#pragma HLS dataflow
        hls::stream<ap_uint<64> > wholeMsgStrm("wrapper64Strm");
#pragma HLS stream variable = wholeMsgStrm depth = 16
        hls::stream<ap_uint<128> > wholeLenStrm;
#pragma HLS stream variable = wholeLenStrm depth = 1
        hls::stream<bool> endWholeLenStrm;
#pragma HLS stream variable = endWholeLenStrm depth = 2

        writeWholeMsg(headLength, wholeMsgStrm, wholeLenStrm, endWholeLenStrm);
        xf::security::sha512<64>(wholeMsgStrm, wholeLenStrm, endWholeLenStrm, digestStrm, endDigestStrm);
    }

    ap_uint<512> hashWithHead(ap_uint<8> headLength, ap_uint<128> msgLength, hls::stream<ap_uint<64> >& msgStrm) {
        hls::stream<ap_uint<512> > digestStrm;
#pragma HLS stream variable = digestStrm depth = 1
        hls::stream<bool> endDigestStrm;
#pragma HLS stream variable = endDigestStrm depth = 2
        wrapperSha512(headLength, msgLength, msgStrm, digestStrm, endDigestStrm);
        endDigestStrm.read();
        endDigestStrm.read();
        return digestStrm.read();
    }

    ap_uint<512> hashWithHead(ap_uint<8> headLength) {
        hls::stream<ap_uint<512> > digestStrm;
#pragma HLS stream variable = digestStrm depth = 1
        hls::stream<bool> endDigestStrm;
#pragma HLS stream variable = endDigestStrm depth = 2
        wrapperSha512(headLength, digestStrm, endDigestStrm);
        endDigestStrm.read();
        endDigestStrm.read();
        return digestStrm.read();
    }

   public:
    /**
     * @brief perform point addition in ed25519, (x3, y3) = (x1, y1) + (x2, y2)
     *
     * @param x1 X coordinate of point 1.
     * @param y1 Y coordinate of point 1.
     * @param x2 X coordinate of point 2.
     * @param y2 Y coordinate of point 2.
     * @param x3 X coordinate of point 3.
     * @param y3 Y coordinate of point 3.
     */
    void pointAdd(
        ap_uint<256> x1, ap_uint<256> y1, ap_uint<256> x2, ap_uint<256> y2, ap_uint<256>& x3, ap_uint<256>& y3) {
        //
        ap_uint<256> x1y2 = xf::security::internal::productMod<256>(x1, y2, p);
        ap_uint<256> x2y1 = xf::security::internal::productMod<256>(x2, y1, p);
        ap_uint<256> x1x2 = xf::security::internal::productMod<256>(x1, x2, p);
        ap_uint<256> y1y2 = xf::security::internal::productMod<256>(y1, y2, p);
        ap_uint<256> xu = xf::security::internal::addMod<256>(x1y2, x2y1, p);
        ap_uint<256> yu = xf::security::internal::addMod<256>(y1y2, x1x2, p);
        ap_uint<256> dx1x2y1y2 = xf::security::internal::productMod<256>(x1x2, y1y2, p);
        dx1x2y1y2 = xf::security::internal::productMod<256>(dx1x2y1y2, d, p);
        ap_uint<256> xv = xf::security::internal::addMod<256>(1, dx1x2y1y2, p);
        ap_uint<256> yv = xf::security::internal::subMod<256>(1, dx1x2y1y2, p);
        ap_uint<256> xvInv = xf::security::internal::modularInv<255>(xv, p);
        ap_uint<256> yvInv = xf::security::internal::modularInv<255>(yv, p);

        x3 = xf::security::internal::productMod<256>(xu, xvInv, p);
        y3 = xf::security::internal::productMod<256>(yu, yvInv, p);
    }

    /**
     * @brief perform point multiply scalar in ed25519, (resX, resY) = (x, y) * mag
     *
     * @param x X coordinate of point to be multiplied.
     * @param y Y coordinate of point to be multiplied.
     * @param mag scalar operand of this multiplication.
     * @param resX X coordinate of result.
     * @param resY Y coordinate of result.
     */
    void pointMul(ap_uint<256> x, ap_uint<256> y, ap_uint<256> mag, ap_uint<256>& resX, ap_uint<256>& resY) {
        //
        ap_uint<256> tmpX = x;
        ap_uint<256> tmpY = y;
        ap_uint<256> tmpResX = 0;
        ap_uint<256> tmpResY = 1;
        for (int i = 0; i < 256; i++) {
            if (mag[i] == 1) {
                pointAdd(tmpResX, tmpResY, tmpX, tmpY, tmpResX, tmpResY);
            }
            pointAdd(tmpX, tmpY, tmpX, tmpY, tmpX, tmpY);
        }
        resX = tmpResX;
        resY = tmpResY;
    }

    /**
     * @brief Generate public key and digest value of privateKey hash value from privateKey.
     *
     * @param privateKey Private Key.
     * @param publicKey Public Key.
     * @param privateKeyHash Digest value of private key.
     */
    void generatePublicKey(ap_uint<256> privateKey, ap_uint<256>& publicKey, ap_uint<512>& privateKeyHash) {
        for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
            head[i] = privateKey.range(i * 64 + 63, i * 64);
        }

        ap_uint<512> hash_private = hashWithHead(4);
        privateKeyHash = hash_private;
        ap_uint<256> secret_scalar = hash_private.range(255, 0);
        secret_scalar.range(2, 0) = 0;
        secret_scalar[255] = 0;
        secret_scalar[254] = 1;
        ap_uint<256> x, y;
        pointMul(Bx, By, secret_scalar, x, y);

        ap_uint<256> tmpRes = y;
        tmpRes[255] = x[0];

        publicKey = tmpRes;
    }

    /**
     * @brief signing function
     *
     * @param msgStrm Stream to input messages to be signed, each message should be input throught this stream twice.
     * @param lenStrm Stream to input length of input messages.
     * @param endLenStrm Stream of end flag of lenStrm.
     * @param publicKey Public Key.
     * @param privateKeyHash Digest value of private key.
     * @param signatureStrm Stream to output signature.
     * @param endSignatureStrm Stream of end flag of signatureStrm.
     */
    void sign(hls::stream<ap_uint<64> >& msgStrm,
              hls::stream<ap_uint<128> >& lenStrm,
              hls::stream<bool>& endLenStrm,
              ap_uint<256> publicKey,
              ap_uint<512> privateKeyHash,
              hls::stream<ap_uint<512> >& signatureStrm,
              hls::stream<bool>& endSignatureStrm) {
        while (!endLenStrm.read()) {
            ap_uint<128> msgLen = lenStrm.read();
            // step 2
            for (int i = 0; i < 4; i++) {
#pragma HLS pipeline
                head[i] = privateKeyHash.range((i + 4) * 64 + 63, (i + 4) * 64);
            }
            ap_uint<512> r = hashWithHead(4, msgLen, msgStrm);

            ap_uint<256> r_mod_L = xf::security::internal::simpleMod<512, 256>(r, L);

            // step 3
            ap_uint<256> Rx, Ry;
            pointMul(Bx, By, r_mod_L, Rx, Ry);
            ap_uint<256> R_compress;
            compress(Rx, Ry, R_compress);

            // step 4
            for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
                head[i] = R_compress.range(i * 64 + 63, i * 64);
            }
            for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
                head[i + 4] = publicKey.range(i * 64 + 63, i * 64);
            }

            ap_uint<512> kHash = hashWithHead(8, msgLen, msgStrm);
            ap_uint<256> kHash_mod_L = xf::security::internal::simpleMod<512, 256>(kHash, L);

            ap_uint<256> secret_scalar = privateKeyHash.range(255, 0);
            secret_scalar.range(2, 0) = 0;
            secret_scalar[255] = 0;
            secret_scalar[254] = 1;

            ap_uint<256> s_mod_L = xf::security::internal::simpleMod<256, 256>(secret_scalar, L);

            ap_uint<256> signature_S = xf::security::internal::productMod(s_mod_L, kHash_mod_L, L);
            signature_S = xf::security::internal::addMod(signature_S, r_mod_L, L);

            ap_uint<512> signature;
            signature.range(255, 0) = R_compress;
            signature.range(511, 256) = signature_S;
            signatureStrm.write(signature);
            endSignatureStrm.write(false);
        }
        endSignatureStrm.write(true);
    }

    /**
     * @brief verifying function
     *
     * @param msgStrm Stream to input messages to be signed.
     * @param lenStrm Stream to input length of input messages.
     * @param signatureStrm Stream to input signatures.
     * @param endSignatureStrm Stream of end flag of signatures.
     * @param publicKeyStrm Stream to input public key.
     * @param ifValidStrm Stream to output if message signature is valid.
     * @param endIfValidStrm Stream of end flag of ifValidStrm.
     */
    void verify(hls::stream<ap_uint<64> >& msgStrm,
                hls::stream<ap_uint<128> >& lenStrm,
                hls::stream<ap_uint<512> >& signatureStrm,
                hls::stream<bool>& endSignatureStrm,
                hls::stream<ap_uint<256> >& publicKeyStrm,
                hls::stream<bool>& ifValidStrm,
                hls::stream<bool>& endIfValidStrm) {
        while (!endSignatureStrm.read()) {
            bool valid = true;

            ap_uint<128> msgLength = lenStrm.read();
            ap_uint<512> signature = signatureStrm.read();
            ap_uint<256> sig_S = signature.range(511, 256);
            ap_uint<256> sig_R = signature.range(255, 0);
            // check sig_S is valid
            if (sig_S >= L) {
                valid = false;
            }
            // check sig_R is valid
            ap_uint<256> Rx, Ry;
            if (!decompress(sig_R, Rx, Ry)) {
                valid = false;
            }
            // check if publicKey is valid
            ap_uint<256> Ax, Ay;
            ap_uint<256> publicKey = publicKeyStrm.read();
            if (!decompress(publicKey, Ax, Ay)) {
                valid = false;
            }

            // compute kHash
            for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
                head[i] = sig_R.range(i * 64 + 63, i * 64);
            }
            for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
                head[i + 4] = publicKey.range(i * 64 + 63, i * 64);
            }
            ap_uint<512> kHash = hashWithHead(8, msgLength, msgStrm);
            ap_uint<256> kHash_mod_L = xf::security::internal::simpleMod<512, 256>(kHash, L);

            //
            ap_uint<256> s_mod_L = xf::security::internal::simpleMod<256, 256>(sig_S, L);
            ap_uint<256> leftX, leftY, rightX, rightY;

            // left of equation.
            pointMul(Bx, By, s_mod_L, leftX, leftY);
            pointMul(leftX, leftY, 8, leftX, leftY);
            // right of equation.
            ap_uint<256> tmpX, tmpY;
            pointMul(Rx, Ry, 8, tmpX, tmpY);
            pointMul(Ax, Ay, kHash_mod_L, rightX, rightY);
            pointMul(rightX, rightY, 8, rightX, rightY);
            pointAdd(rightX, rightY, tmpX, tmpY, rightX, rightY);
            if (leftX != rightX || leftY != rightY) {
                valid = false;
            }
            ifValidStrm.write(valid);
            endIfValidStrm.write(false);
        }
        endIfValidStrm.write(true);
    }
};
} // namespace security
} // namespace xf

#endif
