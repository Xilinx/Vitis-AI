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

#include "hls_stream.h"
#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include "cmdlineparser.h"

#include "zlib_compress.hpp"
#include "zlib_compress_details.hpp"
#include "checksum_wrapper.hpp"

const int c_streamWidth = 64;

typedef ap_uint<c_streamWidth> data_t;

constexpr uint32_t c_size = (c_streamWidth / 8);

void gzipcMulticoreMM(hls::stream<data_t>& inStream,
                      hls::stream<uint32_t>& inSizeStream,
                      hls::stream<ap_uint<32> >& checksumInitStream,
                      hls::stream<data_t>& outStream,
                      hls::stream<bool>& outStreamEoS,
                      hls::stream<uint32_t>& outSizeStream,
                      hls::stream<ap_uint<32> >& checksumOutStream,
                      hls::stream<ap_uint<2> >& checksumTypeStream) {
    xf::compression::gzipMulticoreCompression(inStream, inSizeStream, checksumInitStream, outStream, outStreamEoS,
                                              outSizeStream, checksumOutStream, checksumTypeStream);
}

int main(int argc, char* argv[]) {
    hls::stream<data_t> inStream("inStream");
    hls::stream<uint32_t> inStreamSize("inputSize");
    hls::stream<ap_uint<32> > checksumInitStream("checksumInitStream");
    hls::stream<data_t> outStream("outStream");
    hls::stream<bool> outStreamEoS("outStreamEoS");
    hls::stream<uint32_t> outStreamSize("outSizeStream");
    hls::stream<ap_uint<32> > checksumOutStream("checksumOutStream");
    hls::stream<bool> checksumOutEos("checksumOutEos");
    hls::stream<ap_uint<2> > checksumTypeStream("checksumTypeStream");

    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];

    // File Handling
    std::fstream inFile;
    inFile.open(inputFileName.c_str(), std::fstream::binary | std::fstream::in);
    if (!inFile.is_open()) {
        std::cout << "Cannot open the input file!!" << inputFileName << std::endl;
        exit(0);
    }
    std::ofstream outFile;
    outFile.open(outputFileName.c_str(), std::fstream::binary | std::fstream::out);

    inFile.seekg(0, std::ios::end); // reaching to end of file
    uint32_t input_size = (uint32_t)inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    std::cout << "DATA_SIZE: " << input_size << std::endl;

    std::vector<uint8_t> out(input_size);

    const uint16_t c_format_0 = 31;
    const uint16_t c_format_1 = 139;
    const uint16_t c_variant = 8;
    const uint16_t c_real_code = 8;
    const uint16_t c_opcode = 3;

    // 2 bytes of magic header
    outFile.put(c_format_0);
    outFile.put(c_format_1);

    // 1 byte Compression method
    outFile.put(c_variant);

    // 1 byte flags
    uint8_t flags = 0;
    flags |= c_real_code;
    outFile.put(flags);

    // 4 bytes file modification time in unit format
    unsigned long time_stamp = 0;
    struct stat istat;
    stat(inputFileName.c_str(), &istat);
    time_stamp = istat.st_mtime;
    // put_long(time_stamp, outFile);
    uint8_t time_byte = 0;
    time_byte = time_stamp;
    outFile.put(time_byte);
    time_byte = time_stamp >> 8;
    outFile.put(time_byte);
    time_byte = time_stamp >> 16;
    outFile.put(time_byte);
    time_byte = time_stamp >> 24;
    outFile.put(time_byte);

    // 1 byte extra flag (depend on compression method)
    uint8_t deflate_flags = 0;
    outFile.put(deflate_flags);

    // 1 byte OPCODE - 0x03 for Unix
    outFile.put(c_opcode);

    // Dump file name
    for (int i = 0; inputFileName[i] != '\0'; i++) {
        outFile.put(inputFileName[i]);
    }

    outFile.put(0);

    // Indicate CRC/ADLR
    checksumTypeStream << 1;
    checksumInitStream << ~0;

    // exit data for checksum kernel
    checksumTypeStream << 3;

    inStreamSize << input_size;

    uint32_t incr = c_streamWidth / 8;
    // write data to stream
    for (int i = 0; i < input_size; i += incr) {
        data_t x;
        inFile.read((char*)&x, incr);
        inStream << x;
    }

    // COMPRESSION CALL
    gzipcMulticoreMM(inStream, inStreamSize, checksumInitStream, outStream, outStreamEoS, outStreamSize,
                     checksumOutStream, checksumTypeStream);

    uint32_t outIdx = 0;
    for (bool outEoS = outStreamEoS.read(); outEoS == 0; outEoS = outStreamEoS.read()) {
        // reading value from output stream
        data_t o = outStream.read();
        for (int i = 0; i < incr; i++) {
            out[outIdx++] = o.range((i + 1) * 8 - 1, i * 8);
        }
    }

    data_t o = outStream.read();

    uint32_t outSize = outStreamSize.read();
    printf("Compressed Size:%d\n", outSize);

    outFile.write((char*)out.data(), outSize);

    // Last Block
    outFile.put(1);
    outFile.put(0);
    outFile.put(0);
    outFile.put(255);
    outFile.put(255);

    unsigned long ifile_size = istat.st_size;

    // read checksum value
    ap_uint<32> checksumValue;
    checksumValue = checksumOutStream.read();

    uint8_t crc_byte = 0;
    uint32_t crc_val = checksumValue;
    crc_byte = crc_val;
    outFile.put(crc_byte);
    crc_byte = crc_val >> 8;
    outFile.put(crc_byte);
    crc_byte = crc_val >> 16;
    outFile.put(crc_byte);
    crc_byte = crc_val >> 24;
    outFile.put(crc_byte);

    uint8_t len_byte = 0;
    len_byte = ifile_size;
    outFile.put(len_byte);
    len_byte = ifile_size >> 8;
    outFile.put(len_byte);
    len_byte = ifile_size >> 16;
    outFile.put(len_byte);
    len_byte = ifile_size >> 24;
    outFile.put(len_byte);

    printf("To Decompress the file please give below command:\n");
    printf("\t\t\t gzip -d <fileName.gz> \n");

    outFile.close();
    inFile.close();
    return 0;
}
