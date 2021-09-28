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
#include "lz4_p2p_comp.hpp"
#include "lz4_specs.hpp"
#include "xxhash.h"

#define RESIDUE_4K 4096

/* File descriptors to open the input and output files with O_DIRECT option
 * These descriptors used in P2P case only
 */

// Constructor
xflz4::xflz4(const std::string& binaryFileName, uint8_t device_id, uint32_t m_block_kb) {
    // Index calculation
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    m_BlockSizeInKb = m_block_kb;
    /* Multi board support: selecting the right device based on the device_id,
     * provided through command line args (-id <device_id>).
     */
    if (devices.size() < device_id) {
        std::cout << "Identfied devices = " << devices.size() << ", given device id = " << unsigned(device_id)
                  << std::endl;
        std::cout << "Error: Device ID should be within the range of number of Devices identified" << std::endl;
        std::cout << "Program exited.." << std::endl;
        exit(1);
    }
    devices.at(0) = devices.at(device_id);

    cl::Device device = devices.at(0);

    // Creating Context and Command Queue for selected Device
    m_context = new cl::Context(device);
    m_q = new cl::CommandQueue(*m_context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << ", device id = " << unsigned(device_id) << std::endl;

    // import_binary() command will find the OpenCL binary file created using the
    // v++ compiler load into OpenCL Binary and return as Binaries
    // OpenCL and it can contain many functions which can be executed on the
    // device.

    auto fileBuf = xcl::read_binary_file(binaryFileName.c_str());
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    devices.resize(1);

    m_program = new cl::Program(*m_context, devices, bins);
}

size_t xflz4::create_header(uint8_t* h_header, uint32_t inSize) {
    uint8_t block_size_header = 0;
    switch (m_BlockSizeInKb) {
        case 64:
            block_size_header = xf::compression::BSIZE_STD_64KB;
            break;
        case 256:
            block_size_header = xf::compression::BSIZE_STD_256KB;
            break;
        case 1024:
            block_size_header = xf::compression::BSIZE_STD_1024KB;
            break;
        case 4096:
            block_size_header = xf::compression::BSIZE_STD_4096KB;
            break;
        default:
            block_size_header = xf::compression::BSIZE_STD_64KB;
            std::cout << "Valid block size not given, setting to 64K" << std::endl;
            break;
    }

    uint8_t temp_buff[10] = {
        xf::compression::FLG_BYTE, block_size_header, inSize, inSize >> 8, inSize >> 16, inSize >> 24, 0, 0, 0, 0};

    // xxhash is used to calculate hash value
    uint32_t xxh = XXH32(temp_buff, 10, 0);
    // This value is sent to Kernel 2
    uint32_t xxhash_val = (xxh >> 8);
    m_xxhashVal = xxhash_val;

    // Header information
    uint32_t head_size = 0;

    h_header[head_size++] = xf::compression::MAGIC_BYTE_1;
    h_header[head_size++] = xf::compression::MAGIC_BYTE_2;
    h_header[head_size++] = xf::compression::MAGIC_BYTE_3;
    h_header[head_size++] = xf::compression::MAGIC_BYTE_4;

    h_header[head_size++] = xf::compression::FLG_BYTE;

    // Value
    switch (m_BlockSizeInKb) {
        case 64:
            h_header[head_size++] = xf::compression::BSIZE_STD_64KB;
            break;
        case 256:
            h_header[head_size++] = xf::compression::BSIZE_STD_256KB;
            break;
        case 1024:
            h_header[head_size++] = xf::compression::BSIZE_STD_1024KB;
            break;
        case 4096:
            h_header[head_size++] = xf::compression::BSIZE_STD_4096KB;
            break;
    }

    // Input size
    h_header[head_size++] = inSize;
    h_header[head_size++] = inSize >> 8;
    h_header[head_size++] = inSize >> 16;
    h_header[head_size++] = inSize >> 24;
    h_header[head_size++] = 0;
    h_header[head_size++] = 0;
    h_header[head_size++] = 0;
    h_header[head_size++] = 0;

    // XXHASH value
    h_header[head_size++] = xxhash_val;
    return head_size;
}

// Destructor
xflz4::~xflz4() {
    delete (m_program);
    delete (m_q);
    delete (m_context);
}

// This version of compression does overlapped execution between
// Kernel and Host. I/O operations between Host and Device are
// overlapped with Kernel execution between multiple compute units
void xflz4::compress_in_line_multiple_files(std::vector<char*>& inVec,
                                            const std::vector<std::string>& outFileVec,
                                            std::vector<uint32_t>& inSizeVec,
                                            bool enable_p2p) {
    std::vector<cl::Buffer*> bufInputVec;
    std::vector<cl::Buffer*> bufOutputVec;
    std::vector<cl::Buffer*> buflz4OutVec;
    std::vector<cl::Buffer*> buflz4OutSizeVec;
    std::vector<cl::Buffer*> bufblockSizeVec;
    std::vector<cl::Buffer*> bufCompSizeVec;
    std::vector<cl::Buffer*> bufheadVec;
    std::vector<uint8_t*> bufp2pOutVec;
    std::vector<int> fd_p2p_vec;

    std::vector<uint8_t*> h_headerVec;
    std::vector<uint32_t*> h_blkSizeVec;
    std::vector<uint32_t*> h_lz4OutSizeVec;
    std::vector<cl::Event*> opFinishEvent;
    std::vector<uint32_t> headerSizeVec;
    std::vector<uint32_t> compressSizeVec;

    std::vector<cl::Kernel*> packerKernelVec;
    std::vector<cl::Kernel*> compressKernelVec;

    // only for Non-P2P
    std::vector<uint8_t*> compressDataInHostVec;
    uint32_t outputSize = 0;

    int ret = 0;

    std::chrono::duration<double, std::nano> total_ssd_time_ns(0);

    // Pre Processing
    for (uint32_t i = 0; i < inVec.size(); i++) {
        // To handle files size less than 4K
        if (inSizeVec[i] < RESIDUE_4K) {
            outputSize = RESIDUE_4K;
        } else {
            outputSize = inSizeVec[i];
            // for CR=1 case extra bytes are added for each block
            // so increasing outputSize by 64KB (64*1024).
            outputSize += 64 * 1024;
        }

        uint8_t* h_header = (uint8_t*)aligned_alloc(4 * 1024, 4 * 1024);
        uint32_t* h_blksize = (uint32_t*)aligned_alloc(4 * 1024, 64 * 1024);
        uint32_t* h_lz4outSize = (uint32_t*)aligned_alloc(4 * 1024, 4 * 1024);
        uint32_t block_size_in_bytes = m_BlockSizeInKb * 1024;
        uint32_t head_size = create_header(h_header, inSizeVec[i]);
        headerSizeVec.push_back(head_size);
        h_headerVec.push_back(h_header);
        h_blkSizeVec.push_back(h_blksize);
        h_lz4OutSizeVec.push_back(h_lz4outSize);

        std::string comp_kname = compress_kernel_names[0];
        std::string pack_kname = packer_kernel_names[0];

        if (!enable_p2p) {
            // Creating Host memory to read the compressed data back to host for non-p2p flow case
            uint8_t* compressData = (uint8_t*)aligned_alloc(outputSize, outputSize);
            compressDataInHostVec.push_back(compressData);
        }

        // Total chunks in input file
        // For example: Input file size is 12MB and Host buffer size is 2MB
        // Then we have 12/2 = 6 chunks exists
        // Calculate the count of total chunks based on input size
        // This count is used to overlap the execution between chunks and file
        // operations

        uint32_t num_blocks = (inSizeVec[i] - 1) / block_size_in_bytes + 1;
        // DDR buffer extensions
        cl_mem_ext_ptr_t lz4Ext;
        if (enable_p2p) lz4Ext = {XCL_MEM_EXT_P2P_BUFFER, NULL, 0};

        int cu_num = i % 2;

        if (cu_num == 0) {
            comp_kname += ":{xilLz4Compress_1}";
            pack_kname += ":{xilLz4Packer_1}";
        } else {
            comp_kname += ":{xilLz4Compress_2}";
            pack_kname += ":{xilLz4Packer_2}";
        }

        // Device buffer allocation
        // K1 Input:- This buffer contains input chunk data
        cl::Buffer* buffer_input =
            new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, inSizeVec[i], inVec[i]);
        bufInputVec.push_back(buffer_input);
        if (enable_p2p) {
            // K2 Output:- This buffer contains compressed data written by device
            cl::Buffer* buffer_lz4out =
                new cl::Buffer(*m_context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, outputSize, &(lz4Ext));
            buflz4OutVec.push_back(buffer_lz4out);
            uint8_t* h_buf_out_p2p =
                (uint8_t*)m_q->enqueueMapBuffer(*(buffer_lz4out), CL_TRUE, CL_MAP_READ, 0, outputSize);
            bufp2pOutVec.push_back(h_buf_out_p2p);
        } else {
            // K2 Output:- This buffer contains compressed data written by device
            cl::Buffer* buffer_lz4out = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, outputSize,
                                                       compressDataInHostVec[i]);
            buflz4OutVec.push_back(buffer_lz4out);
        }
        // K1 Output:- This buffer contains compressed data written by device
        // K2 Input:- This is a input to data packer kernel
        cl::Buffer* buffer_output = new cl::Buffer(*m_context, CL_MEM_WRITE_ONLY, inSizeVec[i]);
        bufOutputVec.push_back(buffer_output);

        // K2 input:- This buffer contains compressed data written by device
        cl::Buffer* buffer_lz4OutSize = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                                       10 * sizeof(uint32_t), h_lz4OutSizeVec[i]);
        buflz4OutSizeVec.push_back(buffer_lz4OutSize);

        // K1 Ouput:- This buffer contains compressed block sizes
        // K2 Input:- This buffer is used in data packer kernel
        cl::Buffer* buffer_compressed_size =
            new cl::Buffer(*m_context, CL_MEM_WRITE_ONLY, num_blocks * sizeof(uint32_t));
        bufCompSizeVec.push_back(buffer_compressed_size);

        // Input:- This buffer contains original input block sizes
        cl::Buffer* buffer_block_size = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                       num_blocks * sizeof(uint32_t), h_blkSizeVec[i]);
        bufblockSizeVec.push_back(buffer_block_size);

        // Input:- Header buffer only used once
        cl::Buffer* buffer_header = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                   head_size * sizeof(uint8_t), h_headerVec[i]);
        bufheadVec.push_back(buffer_header);
        cl::Event* event = new cl::Event();
        opFinishEvent.push_back(event);

        // Main loop of overlap execution
        // Loop below runs over total bricks i.e., host buffer size chunks
        // Figure out block sizes per brick
        uint32_t bIdx = 0;
        for (uint32_t j = 0; j < inSizeVec[i]; j += block_size_in_bytes) {
            uint32_t block_size = block_size_in_bytes;
            if (j + block_size > inSizeVec[i]) {
                block_size = inSizeVec[i] - j;
            }
            h_blksize[bIdx++] = block_size;
        }

        if (enable_p2p) {
            int fd_p2p_c_out = open(outFileVec[i].c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0777);
            if (fd_p2p_c_out <= 0) {
                std::cout << "P2P: Unable to open output file, exited!, ret: " << fd_p2p_c_out << std::endl;
                close(fd_p2p_c_out);
                exit(1);
            }
            fd_p2p_vec.push_back(fd_p2p_c_out);
        }

        // Set kernel arguments
        cl::Kernel* compress_kernel_lz4 = new cl::Kernel(*m_program, comp_kname.c_str());
        int narg = 0;
        compress_kernel_lz4->setArg(narg++, *(bufInputVec[i]));
        compress_kernel_lz4->setArg(narg++, *(bufOutputVec[i]));
        compress_kernel_lz4->setArg(narg++, *(bufCompSizeVec[i]));
        compress_kernel_lz4->setArg(narg++, *(bufblockSizeVec[i]));
        compress_kernel_lz4->setArg(narg++, m_BlockSizeInKb);
        compress_kernel_lz4->setArg(narg++, inSizeVec[i]);
        compressKernelVec.push_back(compress_kernel_lz4);

        uint32_t no_blocks_calc = (inSizeVec[i] - 1) / (m_BlockSizeInKb * 1024) + 1;

        // K2 Set Kernel arguments
        cl::Kernel* packer_kernel_lz4 = new cl::Kernel(*m_program, pack_kname.c_str());
        narg = 0;
        packer_kernel_lz4->setArg(narg++, *(bufOutputVec[i]));
        packer_kernel_lz4->setArg(narg++, *(buflz4OutVec[i]));
        packer_kernel_lz4->setArg(narg++, *(bufCompSizeVec[i]));
        packer_kernel_lz4->setArg(narg++, *(bufblockSizeVec[i]));
        packer_kernel_lz4->setArg(narg++, *(buflz4OutSizeVec[i]));
        packer_kernel_lz4->setArg(narg++, *(bufInputVec[i]));
        packer_kernel_lz4->setArg(narg++, m_BlockSizeInKb);
        packer_kernel_lz4->setArg(narg++, no_blocks_calc);
        packer_kernel_lz4->setArg(narg++, m_xxhashVal);
        packer_kernel_lz4->setArg(narg++, inSizeVec[i]);
        packerKernelVec.push_back(packer_kernel_lz4);
    }

    m_q->finish();
    auto total_start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < inVec.size(); i++) {
        /* Transfer data from host to device
        * In p2p case, no need to transfer buffer input to device from host.
        */
        std::vector<cl::Event> compWait;
        std::vector<cl::Event> packWait;
        std::vector<cl::Event> writeWait;

        cl::Event comp_event, pack_event;
        cl::Event write_event;

        // Migrate memory - Map host to device buffers
        m_q->enqueueMigrateMemObjects({*(bufInputVec[i]), *(bufblockSizeVec[i])}, 0 /* 0 means from host*/, NULL,
                                      &write_event);
        writeWait.push_back(write_event);

        // Fire compress kernel
        m_q->enqueueTask(*compressKernelVec[i], &writeWait, &comp_event);

        compWait.push_back(comp_event);
        // Fire packer kernel
        m_q->enqueueTask(*packerKernelVec[i], &compWait, &pack_event);

        packWait.push_back(pack_event);
        // Read back data
        m_q->enqueueMigrateMemObjects({*(buflz4OutSizeVec[i])}, CL_MIGRATE_MEM_OBJECT_HOST, &packWait,
                                      opFinishEvent[i]);
    }

    uint64_t total_file_size = 0;
    uint64_t comp_file_size = 0;
    for (uint32_t i = 0; i < inVec.size(); i++) {
        opFinishEvent[i]->wait();
        uint32_t compressed_size = *(h_lz4OutSizeVec[i]);
        uint32_t align_4k = compressed_size / RESIDUE_4K;
        uint32_t outIdx_align = RESIDUE_4K * align_4k;
        uint32_t residue_size = compressed_size - outIdx_align;
        uint8_t empty_buffer[4096] = {0};
        // Counter which helps in tracking
        // Output buffer index
        uint8_t* temp;

        if (enable_p2p) {
            temp = (uint8_t*)bufp2pOutVec[i];

            /* Make last packer output block divisible by 4K by appending 0's */
            temp = temp + compressed_size;
            memcpy(temp, empty_buffer, RESIDUE_4K - residue_size);
        }
        compressed_size = outIdx_align + RESIDUE_4K;
        compressSizeVec.push_back(compressed_size);

        if (enable_p2p) {
            auto ssd_start = std::chrono::high_resolution_clock::now();
            ret = write(fd_p2p_vec[i], bufp2pOutVec[i], compressed_size);
            if (ret == -1)
                std::cout << "P2P: write() failed with error: " << ret << ", line: " << __LINE__ << std::endl;
            auto ssd_end = std::chrono::high_resolution_clock::now();
            auto ssd_time_ns = std::chrono::duration<double, std::nano>(ssd_end - ssd_start);
            total_ssd_time_ns += ssd_time_ns;
            float ssd_throughput_in_mbps_1 = (float)comp_file_size * 1000 / total_ssd_time_ns.count();
            std::cout << "\nSSD Throughput: " << std::fixed << std::setprecision(2) << ssd_throughput_in_mbps_1;
            std::cout << " MB/s";
            close(fd_p2p_vec[i]);
        } else {
            m_q->enqueueReadBuffer(*(buflz4OutVec[i]), 0, 0, compressed_size, compressDataInHostVec[i]);
        }
        total_file_size += inSizeVec[i];
        comp_file_size += compressed_size;
    }
    auto total_end = std::chrono::high_resolution_clock::now();

    auto time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
    float throughput_in_mbps_1 = (float)total_file_size * 1000 / time_ns.count();

    std::cout << "\nOverall Throughput [Including SSD Operation]: " << std::fixed << std::setprecision(2)
              << throughput_in_mbps_1;
    std::cout << " MB/s";

    // Post Processing and cleanup
    for (uint32_t i = 0; i < inVec.size(); i++) {
        if (!enable_p2p) {
            std::ofstream outFile(outFileVec[i].c_str(), std::ofstream::binary);
            outFile.write((char*)compressDataInHostVec[i], compressSizeVec[i]);
            outFile.close();
            delete compressDataInHostVec[i];
        }
        delete (bufInputVec[i]);
        delete (bufOutputVec[i]);
        delete (buflz4OutVec[i]);
        delete (bufCompSizeVec[i]);
        delete (bufblockSizeVec[i]);
        delete (buflz4OutSizeVec[i]);
        delete (compressKernelVec[i]);
        delete (packerKernelVec[i]);
    }
}
