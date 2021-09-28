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
#include <fstream>
#include <string>
#include <vector>
#include "xcl2.hpp"
#include "cmdlineparser.h"
#include "zlib.h"

auto constexpr HOST_BUFFER_SIZE = 2 * 1024 * 1024;

void xilChecksumTop(std::string& compress_mod, std::string& compress_bin) {
    std::ifstream ifs;
    ifs.open(compress_mod, std::ofstream::binary | std::ofstream::in);
    if (!ifs) {
        std::cout << "ERROR: read file failure!\n";
        exit(EXIT_FAILURE);
    }

    uint32_t size;
    ifs.seekg(0, std::ios::end);
    size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> in(size);
    ifs.read(reinterpret_cast<char*>(in.data()), size);

    unsigned long checksumTmp = 1;
    uint32_t golden = adler32(checksumTmp, reinterpret_cast<const unsigned char*>(in.data()), size);

    uint32_t no_blocks = 0;
    if (size > 0) no_blocks = (size - 1) / HOST_BUFFER_SIZE + 1;

    cl_int err;
    cl::Context context;
    cl::Kernel krnl_checksum;
    cl::CommandQueue q;

    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(compress_bin);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_checksum = cl::Kernel(program, "xilAdler32", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    uint32_t readSize = 0;
    uint32_t result = 0;
    for (uint32_t i = 0; i < no_blocks; i++) {
        uint32_t host_buffer_size = HOST_BUFFER_SIZE;
        if (readSize + HOST_BUFFER_SIZE > size) host_buffer_size = size - readSize;
        readSize += host_buffer_size;

        std::vector<uint8_t, aligned_allocator<uint8_t> > buf_in(HOST_BUFFER_SIZE);
        std::vector<uint32_t, aligned_allocator<uint32_t> > buf_data(1);

        std::memcpy(buf_in.data(), &in[i * host_buffer_size], host_buffer_size);
        if (i == 0) {
            buf_data[0] = 1;
        } else
            buf_data[0] = result;

        // Allocate Buffer in Global Memory
        // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
        // Device-to-host communication
        cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, host_buffer_size, buf_in.data());

        cl::Buffer buffer_data(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint32_t), buf_data.data());

        uint32_t narg = 0;
        krnl_checksum.setArg(narg++, buffer_input);
        krnl_checksum.setArg(narg++, buffer_data);
        krnl_checksum.setArg(narg++, host_buffer_size);

        // Copy input data to device global memory
        q.enqueueMigrateMemObjects({buffer_input, buffer_data}, 0 /* 0 means from host*/);
        // Launch the Kernel
        // For HLS kernels global and local size is always (1,1,1). So, it is
        // recommended
        q.enqueueTask(krnl_checksum);

        // Copy Result from Device Global Memory to Host Local Memory
        q.enqueueMigrateMemObjects({buffer_data}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        result = buf_data[0];
    }

    auto nerr = 0;
    if (golden != result) {
        std::cout << std::hex << "checksum_out=" << result << ", golden=" << golden << std::endl;
        nerr = 1;
    }

    std::cout << "TEST " << (nerr ? "FAILED" : "PASSED") << std::endl;
}

int main(int argc, char* argv[]) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin", "-cx", "XCLBIN", "checksum");
    parser.addSwitch("--compress", "-c", "Compress", "");
    parser.parse(argc, argv);

    std::string compress_bin = parser.value("xclbin");
    std::string compress_mod = parser.value("compress");

    if (!compress_mod.empty()) xilChecksumTop(compress_mod, compress_bin);
}
