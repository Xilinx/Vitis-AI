/*
 * Copyright 2019-2021 Xilinx, Inc.
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
#ifndef _XFCOMPRESSION_CHECKSUM_HPP_
#define _XFCOMPRESSION_CHECKSUM_HPP_

/**
 * @file checksum_wrapper.hpp
 * @brief Header for modules used in ZLIB/GZIP compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "hls_stream.h"
#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

#include "xf_security/adler32.hpp"
#include "xf_security/crc32.hpp"
#include "stream_downsizer.hpp"

namespace xf {
namespace compression {
namespace details {

template <int PARALLEL_BYTE>
void mm2s32(const ap_uint<PARALLEL_BYTE * 8>* in,
            const ap_uint<32>* inChecksumData,
            hls::stream<ap_uint<PARALLEL_BYTE * 8> >& outStream,
            hls::stream<ap_uint<32> >& outChecksumStream,
            hls::stream<ap_uint<32> >& outLenStream,
            hls::stream<bool>& outLenStreamEos,
            ap_uint<32> inputSize) {
    uint32_t inSize_gmemwidth = (inputSize - 1) / PARALLEL_BYTE + 1;
    uint32_t readSize = 0;

    outLenStream << inputSize;
    outChecksumStream << inChecksumData[0];
    outLenStreamEos << false;
    outLenStreamEos << true;

mm2s_simple:
    for (uint32_t i = 0; i < inSize_gmemwidth; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<PARALLEL_BYTE* 8> temp = in[i];
        outStream << temp;
    }
}

template <int INWIDTH = 32>
void s2mm32(hls::stream<ap_uint<32> >& inStream, hls::stream<bool>& inStreamEos, ap_uint<32>* outChecksumData) {
    bool eos = inStreamEos.read();
    if (!eos) outChecksumData[0] = inStream.read();
    inStreamEos.read();
}

/**
 * @brief crc32 computes the CRC32 check value of an input data.
 * @tparam W byte number of input data, the value of W includes 1, 2, 4, 8, 16.
 * @param crcInitStrm initialize crc32 value
 * @param inStrm input messages to be checked
 * @param inPackLenStrm effetive lengths of input message pack, 1~W. 0 means end of message
 * @param outStrm crc32 result to output
 */
template <int W>
void crc32(hls::stream<ap_uint<32> >& crcInitStrm,
           hls::stream<ap_uint<8 * W> >& inStrm,
           hls::stream<ap_uint<5> >& inPackLenStrm,
           hls::stream<ap_uint<32> >& outStrm) {
    hls::stream<bool> endInPackLenStrm;
    hls::stream<bool> endOutStrm;
#pragma HLS STREAM variable = endInPackLenStrm depth = 4
#pragma HLS STREAM variable = endOutStrm depth = 4
    endInPackLenStrm << false;
    endInPackLenStrm << true;
    xf::security::crc32<W>(crcInitStrm, inStrm, inPackLenStrm, endInPackLenStrm, outStrm, endOutStrm);
    endOutStrm.read();
    endOutStrm.read();
}

template <int W>
void adler32(hls::stream<ap_uint<32> >& adlerStrm,
             hls::stream<ap_uint<W * 8> >& inStrm,
             hls::stream<ap_uint<5> >& inPackLenStrm,
             hls::stream<ap_uint<32> >& outStrm) {
    hls::stream<bool> endInPackLenStrm;
    hls::stream<bool> endOutStrm;
#pragma HLS STREAM variable = endInPackLenStrm depth = 4
#pragma HLS STREAM variable = endOutStrm depth = 4
    endInPackLenStrm << false;
    endInPackLenStrm << true;
    xf::security::adler32<W>(adlerStrm, inStrm, inPackLenStrm, endInPackLenStrm, outStrm, endOutStrm);
    endOutStrm.read();
    endOutStrm.read();
}

} // End namespace details

template <int PARALLEL_BYTE>
void adler32_mm(const ap_uint<PARALLEL_BYTE * 8>* in, ap_uint<32>* adlerData, ap_uint<32> inputSize) {
#pragma HLS dataflow
    hls::stream<ap_uint<32> > inAdlerStream;
    hls::stream<ap_uint<PARALLEL_BYTE * 8> > inStream;
    hls::stream<ap_uint<32> > inLenStream;
    hls::stream<bool> inLenStreamEos;
    hls::stream<ap_uint<32> > outStream;
    hls::stream<bool> outStreamEos;

#pragma HLS STREAM variable = inStream depth = 32

    // mm2s
    details::mm2s32<PARALLEL_BYTE>(in, adlerData, inStream, inAdlerStream, inLenStream, inLenStreamEos, inputSize);

    // Adler-32
    xf::security::adler32<PARALLEL_BYTE>(inAdlerStream, inStream, inLenStream, inLenStreamEos, outStream, outStreamEos);

    // s2mm
    details::s2mm32(outStream, outStreamEos, adlerData);
}

template <int PARALLEL_BYTE>
void crc32_mm(const ap_uint<PARALLEL_BYTE * 8>* in, ap_uint<32>* crcData, ap_uint<32> inputSize) {
#pragma HLS dataflow
    hls::stream<ap_uint<32> > inCrcStream;
    hls::stream<ap_uint<PARALLEL_BYTE * 8> > inStream;
    hls::stream<ap_uint<32> > inLenStream;
    hls::stream<bool> inLenStreamEos;
    hls::stream<ap_uint<32> > outStream;
    hls::stream<bool> outStreamEos;

#pragma HLS STREAM variable = inStream depth = 32

    // mm2s
    details::mm2s32<PARALLEL_BYTE>(in, crcData, inStream, inCrcStream, inLenStream, inLenStreamEos, inputSize);

    // CRC-32
    xf::security::crc32<PARALLEL_BYTE>(inCrcStream, inStream, inLenStream, inLenStreamEos, outStream, outStreamEos);

    // s2mm
    details::s2mm32(outStream, outStreamEos, crcData);
}

template <int W>
void checksum32(hls::stream<ap_uint<32> >& checksumInitStrm,
                hls::stream<ap_uint<8 * W> >& inStrm,
                hls::stream<ap_uint<32> >& inLenStrm,
                hls::stream<bool>& endInStrm,
                hls::stream<ap_uint<32> >& outStrm,
                hls::stream<bool>& endOutStrm,
                bool checksumType) {
    // CRC
    if (checksumType == true) {
        xf::security::crc32<W>(checksumInitStrm, inStrm, inLenStrm, endInStrm, outStrm, endOutStrm);
    }
    // ADLER
    else {
        xf::security::adler32<W>(checksumInitStrm, inStrm, inLenStrm, endInStrm, outStrm, endOutStrm);
    }
}

template <int W>
void checksum32(hls::stream<ap_uint<32> >& checksumInitStrm,
                hls::stream<ap_uint<8 * W> >& inStrm,
                hls::stream<ap_uint<5> >& inLenStrm,
                hls::stream<ap_uint<32> >& outStrm,
                hls::stream<ap_uint<2> >& checksumTypeStrm) {
    // Internal EOS Streams
    hls::stream<bool> endInStrm;
    hls::stream<bool> endOutStrm;
#pragma HLS STREAM variable = endInStrm depth = 4
#pragma HLS STREAM variable = endOutStrm depth = 4

checksum_loop:
    for (ap_uint<2> checksumType = checksumTypeStrm.read(); checksumType != 3; checksumType = checksumTypeStrm.read()) {
        endInStrm << false;
        endInStrm << true;
        // CRC
        if (checksumType == 1) {
            xf::security::crc32<W>(checksumInitStrm, inStrm, inLenStrm, endInStrm, outStrm, endOutStrm);
        }
        // ADLER
        else {
            xf::security::adler32<W>(checksumInitStrm, inStrm, inLenStrm, endInStrm, outStrm, endOutStrm);
        }
        endOutStrm.read();
        endOutStrm.read();
    }
}

template <int PARALLEL_BYTE>
void checksum32_mm(const ap_uint<PARALLEL_BYTE * 8>* in,
                   ap_uint<32>* checksumData,
                   ap_uint<32> inputSize,
                   const bool checksumType) {
#pragma HLS dataflow

    hls::stream<ap_uint<PARALLEL_BYTE * 8> > inStream;
    hls::stream<ap_uint<32> > inChecksumStream;
    hls::stream<ap_uint<32> > inLenStream;
    hls::stream<ap_uint<32> > outStream;
    hls::stream<bool> inLenStreamEos;
    hls::stream<bool> outStreamEos;

#pragma HLS STREAM variable = inStream depth = 32

    // mm2s
    details::mm2s32<PARALLEL_BYTE>(in, checksumData, inStream, inChecksumStream, inLenStream, inLenStreamEos,
                                   inputSize);

    // Checksum32
    checksum32<PARALLEL_BYTE>(inChecksumStream, inStream, inLenStream, inLenStreamEos, outStream, outStreamEos,
                              checksumType);

    // s2mm
    details::s2mm32(outStream, outStreamEos, checksumData);
}

} // End Namespace compression
} // End Namespace xf

#endif // _XF_COMPRESSION_CHECKSUM_HPP_
