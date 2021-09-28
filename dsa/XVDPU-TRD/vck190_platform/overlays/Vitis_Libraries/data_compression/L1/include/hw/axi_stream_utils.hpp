/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */
#ifndef _XFCOMPRESSION_AXI_STREAM_UTILS_HPP_
#define _XFCOMPRESSION_AXI_STREAM_UTILS_HPP_

#include "compress_utils.hpp"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <stdint.h>
#include <assert.h>
#include <ap_int.h>

namespace xf {
namespace compression {

namespace details {

template <int OUT_DWIDTH>
void hlsStream2axiu(hls::stream<ap_uint<OUT_DWIDTH> >& inputStream,
                    hls::stream<bool>& inputStreamEos,
                    hls::stream<ap_axiu<OUT_DWIDTH, 0, 0, 0> >& outAxiStream,
                    hls::stream<ap_axiu<32, 0, 0, 0> >& outAxiSizeStream,
                    hls::stream<uint32_t>& inputTotalCmpSizeStream) {
    for (uint32_t input_size = inputTotalCmpSizeStream.read(); input_size != 0;
         input_size = inputTotalCmpSizeStream.read()) {
        bool flag;
        do {
            ap_uint<OUT_DWIDTH> temp = inputStream.read();
            flag = inputStreamEos.read();
            ap_axiu<OUT_DWIDTH, 0, 0, 0> tOut;
            tOut.data = temp;
            tOut.last = flag;
            tOut.keep = -1;
            outAxiStream << tOut;
        } while (!flag);

        ap_axiu<32, 0, 0, 0> totalSize;
        totalSize.data = input_size;
        outAxiSizeStream << totalSize;
    }
}

template <int OUT_DWIDTH>
void hlsVectorStream2axiu(hls::stream<IntVectorStream_dt<OUT_DWIDTH, 1> >& inputStream,
                          hls::stream<ap_axiu<OUT_DWIDTH, 0, 0, 0> >& outAxiStream) {
    bool allDone = false;
    bool just_started = true;
write_all_files:
    do {
        bool last = false;
    write_out_file_data:
        do {
#pragma HLS PIPELINE II = 1
            auto inVal = inputStream.read();
            if (inVal.strobe) {
                just_started = false;
            } else { // last file
                allDone = just_started;
                just_started = true;
                last = true;
                if (allDone) break;
            }
            ap_axiu<OUT_DWIDTH, 0, 0, 0> tOut;
            tOut.data = inVal.data[0];
            tOut.last = last;
            tOut.keep = -1;
            outAxiStream << tOut;
        } while (!last);
    } while (!allDone);
}

template <int OUT_DWIDTH>
void hlsStream2axiu(hls::stream<IntVectorStream_dt<8, OUT_DWIDTH / 8> >& inputStream,
                    hls::stream<ap_axiu<OUT_DWIDTH, 0, 0, 0> >& outAxiStream) {
    constexpr int c_maxByteCnt = OUT_DWIDTH / 8;
    IntVectorStream_dt<8, OUT_DWIDTH / 8> inVal = inputStream.read();
    ap_axiu<OUT_DWIDTH, 0, 0, 0> tOut;
    ap_uint<OUT_DWIDTH> tmpVal;
    bool last = (inVal.strobe < c_maxByteCnt);
write_out_file_data:
    while (!last) {
#pragma HLS PIPELINE II = 1
        for (auto i = 0, j = 0; i < OUT_DWIDTH; i += c_maxByteCnt) {
#pragma HLS UNROLL
            tmpVal.range(i + 7, i) = inVal.data[j++];
        }
        tOut.data = tmpVal;
        tOut.last = false;
        tOut.keep = -1;
        outAxiStream << tOut;
        inVal = inputStream.read();
        last = (inVal.strobe < c_maxByteCnt);
    }

    for (auto i = 0, j = 0; i < OUT_DWIDTH; i += c_maxByteCnt) {
#pragma HLS UNROLL
        tmpVal.range(i + 7, i) = inVal.data[j++];
    }
    tOut.data = tmpVal;
    tOut.last = last;
    ap_uint<OUT_DWIDTH / 8> cntr = 0;
    auto strb = inVal.strobe;
    for (auto i = 0; i < strb; i++) {
        cntr.range(i, i) = 1;
    }
    tOut.keep = cntr;
    outAxiStream << tOut;
}

template <int IN_DWIDTH>
void axiu2hlsStream(hls::stream<ap_axiu<IN_DWIDTH, 0, 0, 0> >& inAxiStream,
                    hls::stream<IntVectorStream_dt<8, IN_DWIDTH / 8> >& outHlsStream) {
    constexpr int c_maxByteCnt = IN_DWIDTH / 8;
    IntVectorStream_dt<8, c_maxByteCnt> val;
    ap_uint<IN_DWIDTH / 8> cntr = 0;
    // Read AXI Stream
    ap_axiu<IN_DWIDTH, 0, 0, 0> inData;
    inData = inAxiStream.read();
    bool last = inData.last;
    ap_uint<IN_DWIDTH> tmpVal = inData.data;
    ap_uint<IN_DWIDTH / 8> keep = inData.keep;
    val.strobe = c_maxByteCnt;

axi2Hls:
    while (!last) {
#pragma HLS PIPELINE
        // Write the data in an aggregated manner in internal hls stream
        for (auto i = 0, j = 0; i < IN_DWIDTH; i += c_maxByteCnt) {
#pragma HLS UNROLL
            val.data[j++] = tmpVal.range(i + 7, i);
        }
        outHlsStream << val;

        inData = inAxiStream.read();
        last = inData.last;
        tmpVal = inData.data;
        keep = inData.keep;
    }

    // Write the data in an aggregated manner in internal hls stream
    for (auto i = 0, j = 0; i < IN_DWIDTH; i += c_maxByteCnt) {
#pragma HLS UNROLL
        val.data[j++] = tmpVal.range(i + 7, i);
    }
    while (keep) {
        cntr += (keep & 0x1);
        keep >>= 1;
    }
    val.strobe = cntr;
    outHlsStream << val;
    val.strobe = 0;
    outHlsStream << val;
}

template <int IN_DWIDTH>
void axiu2hlsStreamSizeEos(hls::stream<ap_axiu<IN_DWIDTH, 0, 0, 0> >& inputAxiStream,
                           hls::stream<ap_axiu<32, 0, 0, 0> >& inSizeStream,
                           hls::stream<ap_uint<IN_DWIDTH> >& outputStream,
                           hls::stream<bool>& outputStreamEos,
                           hls::stream<uint32_t>& outputSizeStream) {
    const int c_parallelByte = IN_DWIDTH / 8;
    for (ap_axiu<32, 0, 0, 0> tempSize = inSizeStream.read(); tempSize.data != 0; tempSize = inSizeStream.read()) {
        uint32_t inputSize = tempSize.data;
        outputSizeStream << inputSize;
        for (uint32_t j = 0; j < inputSize; j += c_parallelByte) {
#pragma HLS PIPELINE II = 1
            ap_axiu<IN_DWIDTH, 0, 0, 0> tempVal = inputAxiStream.read();
            ap_uint<IN_DWIDTH> tmpOut = tempVal.data;
            outputStream << tmpOut;
            outputStreamEos << false;
        }
    }
    outputSizeStream << 0;
    outputStreamEos << true;
}

template <int IN_DWIDTH>
void axiu2hlsStreamSize(hls::stream<ap_axiu<IN_DWIDTH, 0, 0, 0> >& inputAxiStream,
                        hls::stream<ap_uint<IN_DWIDTH> >& outputStream,
                        hls::stream<uint32_t>& outputSizeStream,
                        hls::stream<ap_axiu<32, 0, 0, 0> >& inSizeStream) {
    const int c_parallelByte = IN_DWIDTH / 8;
    for (ap_axiu<32, 0, 0, 0> tempSize = inSizeStream.read(); tempSize.data != 0; tempSize = inSizeStream.read()) {
        ap_uint<32> inputSize = tempSize.data;
        outputSizeStream << inputSize;
        for (uint32_t j = 0; j < inputSize; j += c_parallelByte) {
#pragma HLS PIPELINE II = 1
            ap_axiu<IN_DWIDTH, 0, 0, 0> tempVal = inputAxiStream.read();
            ap_uint<IN_DWIDTH> tmpOut = tempVal.data;
            outputStream << tmpOut;
        }
    }
    outputSizeStream << 0;
}

template <int STREAMDWIDTH = 8>
void axis2hlsStreamFixedSize(hls::stream<qdma_axis<STREAMDWIDTH, 0, 0, 0> >& inputAxiStream,
                             hls::stream<ap_uint<STREAMDWIDTH> >& inputStream,
                             uint32_t inputSize) {
    /**
     * @brief Read data from axi stream and write to internal hls stream.
     *
     * @param inputAxisStream   incoming axi stream
     * @param inputStream       output hls stream
     * @param inputSize         size of the data coming from input axi stream
     *
     */
    for (int i = 0; i < inputSize; i++) {
#pragma HLS PIPELINE II = 1
        qdma_axis<STREAMDWIDTH, 0, 0, 0> t1 = inputAxiStream.read();
        ap_uint<STREAMDWIDTH> tmpOut;
        tmpOut = t1.get_data();
        inputStream << tmpOut;
    }
}

template <int STREAMDWIDTH = 8>
void hlsStream2axis(hls::stream<ap_uint<STREAMDWIDTH> >& outputStream,
                    hls::stream<bool>& outStreamEos,
                    hls::stream<qdma_axis<STREAMDWIDTH, 0, 0, 0> >& outputAxiStream,
                    hls::stream<uint32_t>& outStreamSize,
                    hls::stream<qdma_axis<32, 0, 0, 0> >& outAxiStreamSize) {
    /**
     * @brief Read data from hls stream till the end of stream is indicated and write this data to axi stream.
     *        The total size of data is written 32-bit wide axi stream.
     *
     * @param outputStream      internal output hls stream
     * @param outStreamEos      stream to specify the end of stream
     * @param outputAxisStream  output stream going to axi
     * @param outStreamSize     size of the data coming to output from stream
     * @param outAxiStreamSize  size of the data to go through output axi stream
     *
     */
    ap_uint<STREAMDWIDTH> temp;
    bool flag;
    do {
        temp = outputStream.read();
        flag = outStreamEos.read();
        qdma_axis<STREAMDWIDTH, 0, 0, 0> tOut;
        tOut.set_data(temp);
        tOut.set_last(flag);
        tOut.set_keep(-1);
        outputAxiStream.write(tOut);
    } while (!flag);
    uint32_t outSize = outStreamSize.read();
    qdma_axis<32, 0, 0, 0> tOutSize;
    tOutSize.set_data(outSize);
    tOutSize.set_last(1);
    tOutSize.set_keep(-1);
    outAxiStreamSize.write(tOutSize);
}

template <int STREAMDWIDTH = 8>
void hlsStream2axiStreamFixedSize(hls::stream<ap_uint<STREAMDWIDTH> >& hlsInStream,
                                  hls::stream<qdma_axis<STREAMDWIDTH, 0, 0, 0> >& outputAxiStream,
                                  uint32_t originalSize) {
    /**
     * @brief Read data from internal hls stream and write to output axi stream for the given size.
     *
     * @param hlsInStream       internal hls stream
     * @param outputAxisStream  output axi stream
     * @param originalSize      output data size to be written to output stream
     *
     */
    ap_uint<STREAMDWIDTH> temp;
    for (int i = 0; i < originalSize - 1; i++) {
        temp = hlsInStream.read();
        qdma_axis<STREAMDWIDTH, 0, 0, 0> tOut;
        tOut.set_data(temp);
        tOut.set_keep(-1);
        outputAxiStream.write(tOut);
    }
    // last byte
    temp = hlsInStream.read();
    qdma_axis<STREAMDWIDTH, 0, 0, 0> tOut;
    tOut.set_data(temp);
    tOut.set_last(1);
    tOut.set_keep(-1);
    outputAxiStream.write(tOut);
}

template <uint16_t STREAMDWIDTH>
void axis2hlsStream(hls::stream<qdma_axis<STREAMDWIDTH, 0, 0, 0> >& inAxiStream,
                    hls::stream<ap_uint<STREAMDWIDTH> >& outStream) {
    /**
     * @brief Read N-bit wide data from internal hls streams
     *        and write to output axi stream. N is passed as template parameter.
     *
     * @tparam STREAMDWIDTH stream data width
     *
     * @param inAxiStream   input kernel axi stream
     * @param outStream     output hls stream
     *
     */
    while (true) {
#pragma HLS PIPELINE II = 1
        qdma_axis<STREAMDWIDTH, 0, 0, 0> t1 = inAxiStream.read();
        ap_uint<STREAMDWIDTH> tmp = t1.get_data();
        outStream << tmp;
        if (t1.get_last()) {
            break;
        }
    }
}

template <uint16_t STREAMDWIDTH>
void streamDataDm2k(hls::stream<ap_uint<STREAMDWIDTH> >& in,
                    hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& inStream_dm,
                    uint32_t inputSize) {
    /**
     * @brief Write N-bit wide data of given size from hls stream to kernel axi stream.
     *        N is passed as template parameter.
     *
     * @tparam STREAMDWIDTH stream data width
     *
     * @param in            input hls stream
     * @param inStream_dm   output kernel stream
     * @param inputSize     size of data in to be transferred
     *
     */
    // read data from input hls to input stream for decompression kernel
    uint32_t itrLim = 1 + (inputSize - 1) / (STREAMDWIDTH / 8);
    for (uint32_t i = 0; i < itrLim; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<STREAMDWIDTH> temp = in.read();
        ap_axiu<STREAMDWIDTH, 0, 0, 0> dataIn;
        dataIn.data = temp; // kernel to kernel data transfer
        inStream_dm.write(dataIn);
    }
}

template <uint16_t STREAMDWIDTH, class SIZE_DT = uint32_t, uint16_t AXIWIDTH = 32>
void streamDm2k(hls::stream<ap_uint<STREAMDWIDTH> >& in,
                SIZE_DT inputSize,
                SIZE_DT numItr,
                hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& inStream_dm) {
    /**
     * @brief Write N-bit wide data of given size from hls stream to kernel axi stream.
     *        N is passed as template parameter.
     *
     * @tparam STREAMDWIDTH stream data width
     *
     * @param in            input hls stream
     * @param inStream_dm   output kernel stream
     * @param inputSize     size of data in to be transferred
     * @param numItr        number of iterations for single file
     *
     */
    // read data from input hls to input stream for decompression kernel

    constexpr int c_size = STREAMDWIDTH / 8;
    ap_axiu<STREAMDWIDTH, 0, 0, 0> inData;
    for (auto z = 0; z < numItr; z++) {
        SIZE_DT inputAlignedSize = inputSize / c_size;
        uint8_t inputModSize = inputSize % c_size;
    alignedTransfer:
        for (auto i = 0; i < inputAlignedSize; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<STREAMDWIDTH> v = in.read();
            inData.data = v;
            inData.keep = -1;
            inData.last = false;
            if (inputModSize == 0 && i == inputAlignedSize - 1) {
                inData.last = true;
            }
            inStream_dm << inData;
        }

        if (inputModSize) {
            ap_uint<STREAMDWIDTH> v = in.read();
            ap_axiu<STREAMDWIDTH, 0, 0, 0> inData;
            inData.data = v;
            inData.last = true;
            uint32_t num = 0;
            for (auto b = 0; b < inputModSize; b++) {
                num |= 1UL << b;
            }
            inData.keep = num;
            inStream_dm << inData;
        }
    }
}

template <uint16_t STREAMDWIDTH, class SIZE_DT = uint32_t, uint16_t AXIWIDTH = 32>
void streamDm2k(hls::stream<ap_uint<STREAMDWIDTH> >& in,
                SIZE_DT inputSize,
                hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& inStream_dm,
                hls::stream<ap_axiu<AXIWIDTH, 0, 0, 0> >& inStreamSize_dm) {
    /**
     * @brief Write N-bit wide data of given size from hls stream to kernel axi stream.
     *        N is passed as template parameter.
     *
     * @tparam STREAMDWIDTH stream data width
     *
     * @param in            input hls stream
     * @param inStream_dm   output kernel stream
     * @param inputSize     size of data in to be transferred
     *
     */
    // read data from input hls to input stream for decompression kernel
    ap_axiu<AXIWIDTH, 0, 0, 0> dataInSize;
    dataInSize.data = inputSize;
    dataInSize.last = true;
    inStreamSize_dm.write(dataInSize);

    uint32_t itrLim = 1 + (inputSize - 1) / (STREAMDWIDTH / 8);
    for (uint32_t i = 0; i < itrLim; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<STREAMDWIDTH> temp = in.read();
        ap_axiu<STREAMDWIDTH, 0, 0, 0> dataIn;
        dataIn.data = temp; // kernel to kernel data transfer
        inStream_dm.write(dataIn);
    }
}

template <uint16_t STREAMDWIDTH, class SIZE_DT = uint32_t, uint16_t AXIWIDTH = 32>
void streamDm2k(hls::stream<ap_uint<STREAMDWIDTH> >& in,
                hls::stream<SIZE_DT>& inSize,
                hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& outStream) {
    /**
     * @brief Write N-bit wide data of given size from hls stream to kernel axi stream.
     *        N is passed as template parameter.
     *
     * @tparam STREAMDWIDTH stream data width
     *
     * @param in            input hls stream
     * @param outStream     output kernel stream
     * @param inputSize     size of data in to be transferred
     *
     */
    // read data from input hls to input stream for decompression kernel

    constexpr int c_size = STREAMDWIDTH / 8;
    SIZE_DT inputSize = inSize.read();
    SIZE_DT inputAlignedSize = inputSize / c_size;
    uint8_t inputModSize = inputSize % c_size;
alignedTransfer:
    for (auto i = 0; i < inputAlignedSize; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<STREAMDWIDTH> v = in.read();
        ap_axiu<STREAMDWIDTH, 0, 0, 0> inData;
        inData.data = v;
        inData.keep = -1;
        inData.last = false;
        if (inputModSize == 0 && i == inputAlignedSize - 1) {
            inData.last = true;
        }
        outStream << inData;
    }

    if (inputModSize) {
        ap_uint<STREAMDWIDTH> v = in.read();
        ap_axiu<STREAMDWIDTH, 0, 0, 0> inData;
        inData.data = v;
        inData.last = true;
        uint32_t num = 0;
    keepCalc:
        for (auto b = 0; b < inputModSize; b++) {
            num |= 1UL << b;
        }
        inData.keep = num;
        outStream << inData;
    }
}

template <int STREAMDWIDTH = 8>
void streamDataK2dm(hls::stream<ap_uint<STREAMDWIDTH> >& out,
                    hls::stream<bool>& bytEos,
                    hls::stream<uint32_t>& dataSize,
                    hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& dmOutStream) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream and indicate end of stream.
     *
     * @param out           output hls stream
     * @param bytEos        internal stream which indicates end of data stream
     * @param dataSize      size of data in streams
     * @param dmOutStream   input kernel axi stream to be read
     *
     */
    // read data from decompression kernel output to global memory output
    ap_axiu<STREAMDWIDTH, 0, 0, 0> dataout;
    bool last = false;
    uint32_t outSize = 0;
    do {
#pragma HLS PIPELINE II = 1
        dataout = dmOutStream.read();
        last = dataout.last;
        bytEos << last;
        out << dataout.data;
        outSize++;
    } while (!last);
    dataSize << outSize - 1; // read encoded size from decompression kernel
}

template <int STREAMDWIDTH = 8>
void streamDataK2dmFixedSize(hls::stream<ap_uint<STREAMDWIDTH> >& out,
                             hls::stream<ap_axiu<STREAMDWIDTH, 0, 0, 0> >& dmOutStream,
                             uint32_t dataSize) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream for given output size.
     *
     * @param out           output hls stream
     * @param dmOutStream   input kernel axi stream to be read
     * @param dataSize      size of data in streams
     *
     */
    // read data from decompression kernel output to global memory output
    ap_axiu<STREAMDWIDTH, 0, 0, 0> dataout;
    for (uint32_t i = 0; i < dataSize; i++) {
#pragma HLS PIPELINE II = 1
        dataout = dmOutStream.read();
        out << dataout.data;
    }
}

template <int PARALLEL_BYTES, class SIZE_DT = uint32_t, int AXIWIDTH = 32>
void streamDataK2dmMultiByteSize(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& out,
                                 hls::stream<bool>& outEoS,
                                 hls::stream<SIZE_DT>& decompressSize,
                                 hls::stream<ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> >& dmInStream,
                                 hls::stream<ap_axiu<AXIWIDTH, 0, 0, 0> >& dmInSizeStream) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream for given output size.
     *
     * @param out           output hls stream
     * @param dmOutStream   input kernel axi stream to be read
     * @param dataSize      size of data in streams
     *
     */
    // read data from decompression kernel output to global memory output
    ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> inValue;
    inValue = dmInStream.read();

    while (inValue.last == false) {
#pragma HLS PIPELINE II = 1
        outEoS << 0;
        out << inValue.data;

        // read nextValue
        inValue = dmInStream.read();
    }

    outEoS << 1;
    out << inValue.data;

    ap_axiu<AXIWIDTH, 0, 0, 0> uncompressSize = dmInSizeStream.read();
    decompressSize << uncompressSize.data;
}

template <int PARALLEL_BYTES = 8, class SIZE_DT = uint32_t, int AXIWIDTH = 32>
void streamK2Dm(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& out,
                hls::stream<bool>& outEoS,
                hls::stream<SIZE_DT>& decompressSize,
                hls::stream<ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> >& dmInStream,
                SIZE_DT numItr) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream for given output size.
     *
     * @param out           output hls stream
     * @param dmOutStream   input kernel axi stream to be read
     * @param dataSize      size of data in streams
     *
     */
    // read data from decompression kernel output to global memory output
    for (auto z = 0; z < numItr; z++) {
        SIZE_DT cntr = 0;
        ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> inValue;
        auto keep = 0;
        bool last = false;

        while (!last) {
#pragma HLS PIPELINE II = 1
            inValue = dmInStream.read();
            outEoS << 0;
            out << inValue.data;
            last = inValue.last;
            keep = inValue.keep;
            if (last) break;
            cntr += PARALLEL_BYTES;
        }

        auto incr = 0;
        while (keep) {
            incr += (keep & 0x1);
            keep >>= 1;
        }

        cntr += incr;
        decompressSize << cntr;
        outEoS << 1;
        out << 0;
    }
}

template <int PARALLEL_BYTES, class SIZE_DT = uint32_t, int AXIWIDTH = 32>
void streamK2Dm(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& out,
                hls::stream<bool>& outEoS,
                hls::stream<SIZE_DT>& decompressSize,
                hls::stream<ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> >& dmInStream,
                hls::stream<ap_axiu<AXIWIDTH, 0, 0, 0> >& dmInSizeStream) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream for given output size.
     *
     * @param out           output hls stream
     * @param dmOutStream   input kernel axi stream to be read
     * @param dataSize      size of data in streams
     *
     */
    // read data from decompression kernel output to global memory output
    ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> inValue;
    inValue = dmInStream.read();

    while (inValue.last == false) {
#pragma HLS PIPELINE II = 1
        outEoS << 0;
        out << inValue.data;

        // read nextValue
        inValue = dmInStream.read();
    }

    outEoS << 0;
    out << inValue.data;

    outEoS << 1;
    out << 0;

    ap_axiu<AXIWIDTH, 0, 0, 0> uncompressSize = dmInSizeStream.read();
    decompressSize << uncompressSize.data;
}

template <int PARALLEL_BYTES = 8, class SIZE_DT = uint32_t, int AXIWIDTH = 32>
void streamK2Dm(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& out,
                hls::stream<bool>& outEoS,
                hls::stream<SIZE_DT>& decompressSize,
                hls::stream<ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> >& dmInStream) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream for given output size.
     *
     * @param out           output hls stream
     * @param dmOutStream   input kernel axi stream to be read
     * @param dataSize      size of data in streams
     *
     */
    // read data from decompression kernel output to global memory output
    SIZE_DT cntr = 0;
    ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> inValue;
    auto keep = 0;
    bool last = false;

dmAxi2hls:
    while (!last) {
#pragma HLS PIPELINE II = 1
        inValue = dmInStream.read();
        outEoS << 0;
        out << inValue.data;
        last = inValue.last;
        keep = inValue.keep;
        if (last) break;
        cntr += PARALLEL_BYTES;
    }

    auto incr = 0;
dmLeftSize:
    while (keep) {
        incr += (keep & 0x1);
        keep >>= 1;
    }

    cntr += incr;
    decompressSize << cntr;
    outEoS << 1;
    out << 0;
}

template <int PARALLEL_BYTES>
void streamDataK2dmMultiByte(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& out,
                             hls::stream<bool>& outEoS,
                             hls::stream<uint32_t>& decompressSize,
                             hls::stream<ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> >& dmInStream) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream for given output size.
     *
     * @param out               output hls stream
     * @param outEoS            eos for output hls stream
     * @param decompressSize    size of data in streams
     * @param dmInStream        input kernel axi stream to be read
     *
     */
    // read data from decompression kernel output to global memory output
    ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> inValue;
    inValue = dmInStream.read();
    uint32_t outSize = 0;

    while (inValue.last == false) {
#pragma HLS PIPELINE II = 1
        outEoS << 0;
        out << inValue.data;

        // read nextValue
        inValue = dmInStream.read();
        outSize += PARALLEL_BYTES;
    }
    outEoS << 1;
    out << inValue.data;

    decompressSize << outSize;
}

template <int PARALLEL_BYTES>
void streamDataK2dmMultiByteStrobe(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& out,
                                   hls::stream<bool>& outEoS,
                                   hls::stream<uint32_t>& decompressSize,
                                   hls::stream<ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> >& dmInStream) {
    /**
     * @brief Read data from kernel axi stream byte by byte and write to hls stream for given output size.
     *
     * @param out               output hls stream
     * @param outEoS            eos for output hls stream
     * @param decompressSize    size of data in streams
     * @param dmInStream        input kernel axi stream to be read
     *
     */
    // read data from decompression kernel output to global memory output
    ap_axiu<PARALLEL_BYTES * 8, 0, 0, 0> inValue;
    constexpr uint8_t c_keepDWidth = PARALLEL_BYTES;
    inValue = dmInStream.read();
    uint32_t outSize = countSetBits<c_keepDWidth>((int)(inValue.keep)); // 0

    while (inValue.last == false) {
#pragma HLS PIPELINE II = 1
        outEoS << 0;
        out << inValue.data;

        // read nextValue
        inValue = dmInStream.read();
        outSize += countSetBits<c_keepDWidth>((int)(inValue.keep)); // PARALLEL_BYTES;
    }
    outEoS << 1;
    out << inValue.data;

    decompressSize << outSize;
}

} // end details
} // end compression
} // end xf

#endif // _XFCOMPRESSION_AXI_STREAM_UTILS_HPP_
