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
#include "snappyOCLHost.hpp"
#include "xxhash.h"

int fd_p2p_c_in = 0;

constexpr uint32_t roundoff(uint32_t x, uint32_t y) {
    return ((x - 1) / (y) + 1);
}

// Constructor
snappyOCLHost::snappyOCLHost(enum State flow,
                             const std::string& binaryFileName,
                             uint8_t device_id,
                             uint32_t block_size_kb,
                             bool enable_profile,
                             bool enable_p2p)
    : m_xclbin(binaryFileName),
      m_enableProfile(enable_profile),
      m_enableP2P(enable_p2p),
      m_deviceId(device_id),
      m_flow(flow) {
    m_BlockSizeInKb = block_size_kb;

    h_buf_in.resize(HOST_BUFFER_SIZE);
    h_buf_out.resize(HOST_BUFFER_SIZE);
    h_blksize.resize(MAX_NUMBER_BLOCKS);
    h_compressSize.resize(MAX_NUMBER_BLOCKS);

    m_compressSize.reserve(MAX_NUMBER_BLOCKS);
    m_blkSize.reserve(MAX_NUMBER_BLOCKS);

    // unsigned fileBufSize;
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[m_deviceId];

    // Creating Context and Command Queue for selected Device
    m_context = new cl::Context(device);
    m_q = new cl::CommandQueue(*m_context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    // import_binary() command will find the OpenCL binary file created using the
    // v++ compiler load into OpenCL Binary and return as Binaries
    // OpenCL and it can contain many functions which can be executed on the
    // device.
    auto fileBuf = xcl::read_binary_file(m_xclbin);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    m_program = new cl::Program(*m_context, {device}, bins);
    if (m_flow == COMPRESS || m_flow == BOTH) {
#ifdef SNAPPY_STREAM
        // Create Compress kernels
        compress_kernel_snappy = new cl::Kernel(*m_program, compress_kernel_names[1].c_str());
        // Create Compress datamover kernels
        compress_data_mover_kernel = new cl::Kernel(*m_program, compress_dm_kernel_names.c_str());
#else
        compress_kernel_snappy = new cl::Kernel(*m_program, compress_kernel_names[0].c_str());
#endif
    }
    if (m_flow == DECOMPRESS || m_flow == BOTH) {
#ifdef SNAPPY_STREAM
        // Create Decompress kernels
        decompress_kernel_snappy = new cl::Kernel(*m_program, decompress_kernel_names[1].c_str());
        // Create Decompress datamover kernels
        decompress_data_mover_kernel = new cl::Kernel(*m_program, decompress_dm_kernel_names.c_str());
#else
        decompress_kernel_snappy = new cl::Kernel(*m_program, decompress_kernel_names[0].c_str());
#endif
    }
}

// Destructor
snappyOCLHost::~snappyOCLHost() {
    if (compress_kernel_snappy != nullptr) {
        delete compress_kernel_snappy;
        compress_kernel_snappy = nullptr;
    }

    if (compress_data_mover_kernel != nullptr) {
        delete compress_data_mover_kernel;
        compress_data_mover_kernel = nullptr;
    }

    if (decompress_kernel_snappy != nullptr) {
        delete decompress_kernel_snappy;
        decompress_kernel_snappy = nullptr;
    }

    if (decompress_data_mover_kernel != nullptr) {
        delete decompress_data_mover_kernel;
        decompress_data_mover_kernel = nullptr;
    }

    delete (m_program);
    delete (m_q);
    delete (m_context);
}

// Driving compress API, includes header processing and calling core compress engine
uint64_t snappyOCLHost::xilCompress(uint8_t* in, uint8_t* out, size_t input_size) {
    m_InputSize = input_size;
    // Snappy header
    int headerBytes = (int)writeHeader(out);
    out += headerBytes;
    uint64_t enbytes;

    if (m_enableProfile) {
        total_start = std::chrono::high_resolution_clock::now();
    }
// Snappy multiple/single cu sequential version
#ifdef SNAPPY_STREAM
    enbytes = compressEngineStreamSeq(in, out, m_InputSize);
#else
    enbytes = compressEngineSeq(in, out, m_InputSize);
#endif
    if (m_enableProfile) {
        total_end = std::chrono::high_resolution_clock::now();
        auto total_time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
        float throughput_in_mbps_1 = (float)input_size * 1000 / total_time_ns.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    enbytes += headerBytes;
    return enbytes;
}

// Driving decompress API, includes header processing and calling core decompress engine
uint64_t snappyOCLHost::xilDecompress(uint8_t* in, uint8_t* out, size_t input_size) {
#ifdef SNAPPY_STREAM
    m_InputSize = input_size;
#else
    uint8_t headerBytes = readHeader(in);
    in += headerBytes;
    m_InputSize = input_size - headerBytes;
#endif
    uint64_t debytes;

    if (m_enableProfile) {
        total_start = std::chrono::high_resolution_clock::now();
    }
#ifdef SNAPPY_STREAM
    // Decompression Engine multiple cus.
    debytes = decompressEngineStreamSeq(in, out, m_InputSize);
#else
    // Decompression Engine multiple cus.
    debytes = decompressEngineSeq(in, out, m_InputSize);
#endif

    if (m_enableProfile) {
        total_end = std::chrono::high_resolution_clock::now();
        auto total_time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
        float throughput_in_mbps_1 = (float)input_size * 1000 / total_time_ns.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }

    return debytes;
}

// Core compress engine API
uint64_t snappyOCLHost::compressEngineSeq(uint8_t* in, uint8_t* out, size_t input_size) {
    uint32_t block_size_in_kb = BLOCK_SIZE_IN_KB;
    uint32_t block_size_in_bytes = block_size_in_kb * 1024;

    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);
    // Keeps track of output buffer index
    uint64_t outIdx = 0;

    // Given a input file, we process it as multiple chunks
    // Each compute unit is assigned with a chunk of data
    // In this example HOST_BUFFER_SIZE is the chunk size.
    // For example: Input file = 12 MB
    // HOST_BUFFER_SIZE = 2MB
    // Each compute unit processes 2MB data per kernel invocation
    uint32_t hostChunk_cu;
    // This buffer contains total number of BLOCK_SIZE_IN_KB blocks per CU
    // For Example: HOST_BUFFER_SIZE = 2MB/BLOCK_SIZE_IN_KB = 32block (Block
    // size 64 by default)
    uint32_t total_blocks_cu;

    // This buffer holds exact size of the chunk in bytes for all the CUs
    uint32_t bufSize_in_bytes_cu;

    // Holds value of total compute units to be
    // used per iteration
    int compute_cu = 0;
    for (uint64_t inIdx = 0; inIdx < input_size; inIdx += HOST_BUFFER_SIZE) {
        // Needs to reset this variable
        // As this drives compute unit launch per iteration
        compute_cu = 0;

        // Pick buffer size as predefined one
        // If yet to be consumed input is lesser
        // the reset to required size

        uint32_t buf_size = HOST_BUFFER_SIZE;

        // This loop traverses through each compute based current inIdx
        // It tries to calculate chunk size and total compute units need to be
        // launched (based on the input_size)

        for (int bufCalc = 0; bufCalc < 1; bufCalc++) {
            hostChunk_cu = 0;
            // If amount of data to be consumed is less than HOST_BUFFER_SIZE
            // Then choose to send is what is needed instead of full buffer size
            // based on host buffer macro

            if (inIdx + (buf_size * (bufCalc + 1)) > input_size) {
                hostChunk_cu = input_size - (inIdx + HOST_BUFFER_SIZE * bufCalc);
                compute_cu++;
                break;
            } else {
                hostChunk_cu = buf_size;
                compute_cu++;
            }
        }
        // Figure out total number of blocks need per each chunk
        // Copy input data from in to host buffer based on the inIdx and cu

        for (int blkCalc = 0; blkCalc < compute_cu; blkCalc++) {
            uint32_t nblocks = (hostChunk_cu - 1) / block_size_in_bytes + 1;
            total_blocks_cu = nblocks;
            std::memcpy(h_buf_in.data(), &in[inIdx + blkCalc * HOST_BUFFER_SIZE], hostChunk_cu);
        }
        // Fill the host block size buffer with various block sizes per chunk/cu
        for (int cuBsize = 0; cuBsize < compute_cu; cuBsize++) {
            uint32_t bIdx = 0;
            uint32_t chunkSize_curr_cu = hostChunk_cu;

            for (uint32_t bs = 0; bs < chunkSize_curr_cu; bs += block_size_in_bytes) {
                uint32_t block_size = block_size_in_bytes;
                if (bs + block_size > chunkSize_curr_cu) {
                    block_size = chunkSize_curr_cu - bs;
                }
                h_blksize.data()[bIdx++] = block_size;
            }
            // Calculate chunks size in bytes for device buffer creation
            bufSize_in_bytes_cu = ((hostChunk_cu - 1) / BLOCK_SIZE_IN_KB + 1) * BLOCK_SIZE_IN_KB;
        }
        buffer_input =
            new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, bufSize_in_bytes_cu, h_buf_in.data());

        buffer_output =
            new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bufSize_in_bytes_cu, h_buf_out.data());

        buffer_compressed_size = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                                sizeof(uint32_t) * total_blocks_cu, h_compressSize.data());

        buffer_block_size = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           sizeof(uint32_t) * total_blocks_cu, h_blksize.data());

        // Device buffer allocation

        int narg = 0;
        compress_kernel_snappy->setArg(narg++, *(buffer_input));
        compress_kernel_snappy->setArg(narg++, *(buffer_output));
        compress_kernel_snappy->setArg(narg++, *(buffer_compressed_size));
        compress_kernel_snappy->setArg(narg++, *(buffer_block_size));
        compress_kernel_snappy->setArg(narg++, block_size_in_kb);
        compress_kernel_snappy->setArg(narg++, hostChunk_cu);
        std::vector<cl::Memory> inBufVec;

        inBufVec.push_back(*(buffer_input));
        inBufVec.push_back(*(buffer_block_size));

        // Migrate memory - Map host to device buffers
        m_q->enqueueMigrateMemObjects(inBufVec, 0 /* 0 means from host*/);
        m_q->finish();

        // Measure kernel execution time
        if (!m_enableProfile) {
            kernel_start = std::chrono::high_resolution_clock::now();
        }
        // Fire kernel execution
        m_q->enqueueTask(*compress_kernel_snappy);
        // wait till kernels complete
        m_q->finish();

        if (!m_enableProfile) {
            kernel_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
            kernel_time_ns_1 += duration;
        }
        // Setup output buffer vectors
        std::vector<cl::Memory> outBufVec;
        outBufVec.push_back(*(buffer_output));
        outBufVec.push_back(*(buffer_compressed_size));
        // Migrate memory - Map device to host buffers
        m_q->enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
        m_q->finish();
        for (int cuCopy = 0; cuCopy < compute_cu; cuCopy++) {
            // Copy data into out buffer
            // Include compress and block size data
            // Copy data block by block within a chunk example 2MB (64block size) - 32 blocks data
            // Do the same for all the compute units
            uint32_t idx = 0;
            for (uint32_t bIdx = 0; bIdx < total_blocks_cu; bIdx++, idx += block_size_in_bytes) {
                // Default block size in bytes i.e., 64 * 1024
                uint32_t block_size = block_size_in_bytes;
                if (idx + block_size > hostChunk_cu) {
                    block_size = hostChunk_cu - idx;
                }
                uint32_t compressed_size = h_compressSize.data()[bIdx];
                assert(compressed_size != 0);

                int orig_block_size = hostChunk_cu;
                int perc_cal = orig_block_size * 10;
                perc_cal = perc_cal / block_size;
                if (compressed_size < block_size && perc_cal >= 10) {
                    // Chunk Type Identifier
                    out[outIdx++] = 0x00;
                    // 3 Bytes to represent compress block length + 4;
                    uint32_t f_csize = compressed_size + 4;
                    std::memcpy(&out[outIdx], &f_csize, 3);
                    outIdx += 3;
                    // CRC - for now 0s
                    uint32_t crc_value = 0;
                    std::memcpy(&out[outIdx], &crc_value, 4);
                    outIdx += 4;
                    // Compressed data of this block with preamble
                    std::memcpy(&out[outIdx], (h_buf_out.data() + bIdx * block_size_in_bytes), compressed_size);
                    outIdx += compressed_size;
                } else {
                    // Chunk Type Identifier
                    out[outIdx++] = 0x01;
                    // 3 Bytes to represent uncompress block length + 4;
                    uint32_t f_csize = block_size + 4;
                    std::memcpy(&out[outIdx], &f_csize, 3);
                    outIdx += 3;
                    // CRC -for now 0s
                    uint32_t crc_value = 0;
                    std::memcpy(&out[outIdx], &crc_value, 4);
                    outIdx += 4;

                    // Uncompressed data copy
                    std::memcpy(&out[outIdx], &in[inIdx + (cuCopy * HOST_BUFFER_SIZE) + idx], block_size);
                    outIdx += block_size;
                } // End of else - uncompressed stream update
            }     // End of chunk (block by block) copy to output buffer
        }         // End of CU loop - Each CU/chunk block by block copy
                  // Buffer deleted
        delete (buffer_input);
        delete (buffer_output);
        delete (buffer_compressed_size);
        delete (buffer_block_size);
    }
    if (!m_enableProfile) {
        float throughput_in_mbps_1 = (float)input_size * 1000 / kernel_time_ns_1.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    return outIdx;
} // End of compress

uint64_t snappyOCLHost::compressEngineStreamSeq(uint8_t* in, uint8_t* out, size_t input_size) {
    uint32_t host_buffer_size = m_BlockSizeInKb * 1024;
    uint32_t total_block_count = (input_size - 1) / host_buffer_size + 1;

    // output buffer index
    uint64_t outIdx = 0;
    // Index calculation
    h_buf_in.resize(host_buffer_size);  // * total_block_count);
    h_buf_out.resize(host_buffer_size); // * total_block_count);
    h_compressSize.resize(1);

    // device buffer allocation
    buffer_input =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, host_buffer_size, h_buf_in.data());

    buffer_output =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, host_buffer_size, h_buf_out.data());

    buffer_compressed_size =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(uint32_t), h_compressSize.data());

    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);

    // copy input to input buffer
    // std::memcpy(h_buf_in.data(), in, input_size);
    // sequentially copy block sized buffers to kernel and wait for them to finish before enqueueing
    for (uint32_t blkIndx = 0, bufIndx = 0; blkIndx < total_block_count; blkIndx++, bufIndx += host_buffer_size) {
        // current block input size
        uint32_t c_input_size = host_buffer_size;
        if (blkIndx == total_block_count - 1) c_input_size = input_size - bufIndx;

        // copy input to input buffer
        std::memcpy(h_buf_in.data(), in + bufIndx, c_input_size);

        // set kernel args
        uint32_t narg = 0;
        compress_data_mover_kernel->setArg(narg++, *buffer_input);
        compress_data_mover_kernel->setArg(narg++, *buffer_output);
        compress_data_mover_kernel->setArg(narg++, *buffer_compressed_size);
        compress_data_mover_kernel->setArg(narg, c_input_size);

        compress_kernel_snappy->setArg(2, c_input_size);
        // Migrate Memory - Map host to device buffers
        m_q->enqueueMigrateMemObjects({*(buffer_input)}, 0);
        m_q->finish();
        // Measure kernel execution time
        if (!m_enableProfile) {
            kernel_start = std::chrono::high_resolution_clock::now();
        }
        // enqueue the kernels and wait for them to finish
        m_q->enqueueTask(*compress_data_mover_kernel);
        m_q->enqueueTask(*compress_kernel_snappy);
        m_q->finish();

        if (!m_enableProfile) {
            kernel_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
            kernel_time_ns_1 += duration;
        }
        // Setup output buffer vectors
        std::vector<cl::Memory> outBufVec;
        outBufVec.push_back(*buffer_output);
        outBufVec.push_back(*buffer_compressed_size);

        // Migrate memory - Map device to host buffers
        m_q->enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
        m_q->finish();

        // copy the compressed data to out pointer
        uint32_t compressedSize = h_compressSize.data()[0];

        if (c_input_size > compressedSize) {
            out[outIdx++] = 0x00;

            // 3 Bytes to represent compressed block length + 4
            uint32_t f_csize = compressedSize + 4;
            std::memcpy(out + outIdx, &f_csize, 3);
            outIdx += 3;

            // CRC - for now 0s
            uint32_t crc_value = 0;
            std::memcpy(out + outIdx, &crc_value, 4);
            outIdx += 4;

            std::memcpy(out + outIdx, h_buf_out.data(), compressedSize);
            outIdx += compressedSize;
        } else {
            // Chunk Type Identifier
            out[outIdx++] = 0x01;
            // 3 Bytes to represent uncompress block length + 4;
            uint32_t f_csize = c_input_size + 4;
            std::memcpy(out + outIdx, &f_csize, 3);
            outIdx += 3;
            // CRC -for now 0s
            uint32_t crc_value = 0;
            std::memcpy(out + outIdx, &crc_value, 4);
            outIdx += 4;

            std::memcpy(out + outIdx, in + (host_buffer_size * blkIndx), c_input_size);
            outIdx += c_input_size;
        }
    }
    if (!m_enableProfile) {
        float throughput_in_mbps_1 = (float)input_size * 1000 / kernel_time_ns_1.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    // Free CL buffers
    delete (buffer_input);
    delete (buffer_output);
    delete (buffer_compressed_size);

    return outIdx;

} // End of compress

// Core decompress engine API

uint64_t snappyOCLHost::decompressEngineSeq(uint8_t* in, uint8_t* out, size_t input_size) {
    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);
    uint32_t buf_size = BLOCK_SIZE_IN_KB * 1024;
    uint32_t blocksPerChunk = HOST_BUFFER_SIZE / buf_size;
    uint32_t host_buffer_size = ((HOST_BUFFER_SIZE - 1) / BLOCK_SIZE_IN_KB + 1) * BLOCK_SIZE_IN_KB;

    // Allocate global buffers
    // Device buffer allocation
    buffer_input =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, host_buffer_size, h_buf_in.data());

    buffer_output =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, host_buffer_size, h_buf_out.data());

    buffer_block_size = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       sizeof(uint32_t) * blocksPerChunk, h_blksize.data());

    buffer_compressed_size = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            sizeof(uint32_t) * blocksPerChunk, h_compressSize.data());
    uint32_t narg = 0;
    decompress_kernel_snappy->setArg(narg++, *(buffer_input));
    decompress_kernel_snappy->setArg(narg++, *(buffer_output));
    decompress_kernel_snappy->setArg(narg++, *(buffer_block_size));
    decompress_kernel_snappy->setArg(narg++, *(buffer_compressed_size));
    decompress_kernel_snappy->setArg(narg++, m_BlockSizeInKb);
    decompress_kernel_snappy->setArg(narg++, blocksPerChunk);
    uint32_t chunk_size = 0;
    uint8_t chunk_idx = 0;
    uint32_t block_cntr = 0;
    uint32_t block_size = 0;
    uint32_t chunk_cntr = 0;
    uint32_t bufblocks = 0;
    uint64_t output_idx = 0;
    uint32_t bufIdx = 0;
    uint32_t over_block_cntr = 0;
    uint32_t brick = 0;
    uint16_t stride_cidsize = 4;
    bool blkDecomExist = false;
    uint32_t blkUnComp = 0;
    // Maximum allowed outbuffer size, if it exceeds then exit
    uint32_t c_max_outbuf = input_size * m_maxCR;
    // Go over overall input size
    for (uint32_t idxSize = 0; idxSize < input_size; idxSize += stride_cidsize, chunk_cntr++) {
        // Chunk identifier
        chunk_idx = in[idxSize];
        chunk_size = 0;
        // Chunk Compressed size
        uint8_t cbyte_1 = in[idxSize + 1];
        uint8_t cbyte_2 = in[idxSize + 2];
        uint8_t cbyte_3 = in[idxSize + 3];

        uint32_t temp = cbyte_3;
        temp <<= 16;
        chunk_size |= temp;
        temp = 0;
        temp = cbyte_2;
        temp <<= 8;
        chunk_size |= temp;
        temp = 0;
        chunk_size |= cbyte_1;

        if (chunk_idx == 0x00) {
            uint8_t bval1 = in[idxSize + 8];
            uint32_t final_size = 0;

            if ((bval1 >> 7) == 1) {
                uint8_t b1 = bval1 & 0x7F;
                bval1 = in[idxSize + 9];
                uint8_t b2 = bval1 & 0x7F;
                if ((bval1 >> 7) == 1) {
                    bval1 = in[idxSize + 10];
                    uint8_t b3 = bval1 & 0x7F;
                    uint32_t temp1 = b3;
                    temp1 <<= 14;
                    uint32_t temp2 = b2;
                    temp2 <<= 7;
                    uint32_t temp3 = b1;
                    final_size |= temp1;
                    final_size |= temp2;
                    final_size |= temp3;
                } else {
                    uint32_t temp1 = b2;
                    temp1 <<= 7;
                    uint32_t temp2 = b1;
                    final_size |= temp1;
                    final_size |= temp2;
                }

                block_size = final_size;
            } else {
                block_size = bval1;
            }
            m_compressSize.data()[over_block_cntr] = chunk_size - 4;
            m_blkSize.data()[over_block_cntr] = block_size;

            h_compressSize.data()[bufblocks] = chunk_size - 4;
            h_blksize.data()[bufblocks] = block_size;
            bufblocks++;
            // Copy data
            std::memcpy(&(h_buf_in.data()[block_cntr * buf_size]), &in[idxSize + 8], chunk_size - 4);
            block_cntr++;
            blkDecomExist = true;
        } else if (chunk_idx == 0x01) {
            m_compressSize.data()[over_block_cntr] = chunk_size - 4;
            m_blkSize.data()[over_block_cntr] = chunk_size - 4;
            std::memcpy(&out[brick * HOST_BUFFER_SIZE + over_block_cntr * buf_size], &in[idxSize + 8], chunk_size - 4);
            blkUnComp += chunk_size - 4;
        }

        over_block_cntr++;
        // Increment the input idx to
        // compressed size length
        idxSize += chunk_size;

        if (over_block_cntr == blocksPerChunk && blkDecomExist) {
            blkDecomExist = false;
            // Track the chunks processed
            brick++;
            // In case of left over set kernel arg to no blocks
            decompress_kernel_snappy->setArg(5, block_cntr);
            // For big files go ahead do it here
            std::vector<cl::Memory> inBufVec;
            inBufVec.push_back(*(buffer_input));
            inBufVec.push_back(*(buffer_block_size));
            inBufVec.push_back(*(buffer_compressed_size));
            // Migrate memory - Map host to device buffers
            m_q->enqueueMigrateMemObjects(inBufVec, 0 /*0 means from host*/);
            m_q->finish();

            if (!m_enableProfile) {
                // Measure kernel execution time
                kernel_start = std::chrono::high_resolution_clock::now();
            }
            m_q->enqueueTask(*decompress_kernel_snappy);
            m_q->finish();
            if (!m_enableProfile) {
                kernel_end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
                kernel_time_ns_1 += duration;
            }
            std::vector<cl::Memory> outBufVec;
            outBufVec.push_back(*(buffer_output));
            // Migrate memory - Map device to host buffers
            m_q->enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
            m_q->finish();
            bufIdx = 0;
            // copy output
            for (uint32_t bIdx = 0; bIdx < over_block_cntr; bIdx++) {
                uint32_t block_size = m_blkSize.data()[bIdx];
                uint32_t compressed_size = m_compressSize.data()[bIdx];

                if ((output_idx + block_size) > c_max_outbuf) {
                    std::cout << "\n" << std::endl;
                    std::cout << "\x1B[35mZIP BOMB: Exceeded output buffer size during decompression \033[0m \n"
                              << std::endl;
                    std::cout
                        << "\x1B[35mUse -mcr option to increase the maximum compression ratio (Default: 10) \033[0m \n"
                        << std::endl;
                    std::cout << "\x1B[35mAborting .... \033[0m\n" << std::endl;
                    exit(1);
                }

                if (compressed_size < block_size) {
                    std::memcpy(&out[output_idx], &h_buf_out.data()[bufIdx], block_size);
                    output_idx += block_size;
                    bufIdx += block_size;
                } else if (compressed_size == block_size) {
                    output_idx += block_size;
                    blkUnComp -= block_size;
                }
            }
            block_cntr = 0;
            bufblocks = 0;
            over_block_cntr = 0;
        } else if (over_block_cntr == blocksPerChunk) {
            over_block_cntr = 0;
            brick++;
            bufblocks = 0;
            block_cntr = 0;
        }
    }

    if (block_cntr != 0) {
        // In case of left over set kernel arg to no blocks
        decompress_kernel_snappy->setArg(5, block_cntr);

        std::vector<cl::Memory> inBufVec;
        inBufVec.push_back(*(buffer_input));
        inBufVec.push_back(*(buffer_block_size));
        inBufVec.push_back(*(buffer_compressed_size));

        // Migrate memory - Map host to device buffers
        m_q->enqueueMigrateMemObjects(inBufVec, 0 /*0 means from host*/);
        m_q->finish();
        // Measure kernel execution time
        auto kernel_start = std::chrono::high_resolution_clock::now();
        // Kernel invocation
        m_q->enqueueTask(*decompress_kernel_snappy);
        m_q->finish();
        auto kernel_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
        kernel_time_ns_1 += duration;

        std::vector<cl::Memory> outBufVec;
        outBufVec.push_back(*(buffer_output));
        // Migrate memory - Map device to host buffers
        m_q->enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST);
        m_q->finish();
        bufIdx = 0;
        // copy output
        for (uint32_t bIdx = 0; bIdx < over_block_cntr; bIdx++) {
            uint32_t block_size = m_blkSize.data()[bIdx];
            uint32_t compressed_size = m_compressSize.data()[bIdx];

            if ((output_idx + block_size) > c_max_outbuf) {
                std::cout << "\n" << std::endl;
                std::cout << "\x1B[35mZIP BOMB: Exceeded output buffer size during decompression \033[0m \n"
                          << std::endl;
                std::cout
                    << "\x1B[35mUse -mcr option to increase the maximum compression ratio (Default: 10) \033[0m \n"
                    << std::endl;
                std::cout << "\x1B[35mAborting .... \033[0m\n" << std::endl;
                exit(1);
            }

            if (compressed_size < block_size) {
                std::memcpy(&out[output_idx], &h_buf_out.data()[bufIdx], block_size);
                output_idx += block_size;
                bufIdx += block_size;
            } else if (compressed_size == block_size) {
                output_idx += block_size;
                blkUnComp -= block_size;
            }
        }

    } // If to see if tehr eare some blocks to be processed
    if (output_idx == 0 && blkUnComp != 0) {
        output_idx = blkUnComp;
    }
    if (!m_enableProfile) {
        float kernel_throughput_in_mbps_1 = (float)output_idx * 1000 / kernel_time_ns_1.count();
        std::cout << std::fixed << std::setprecision(2) << kernel_throughput_in_mbps_1;
    }
    delete (buffer_input);
    delete (buffer_output);
    delete (buffer_block_size);
    delete (buffer_compressed_size);

    return output_idx;

} // End of decompress

uint64_t snappyOCLHost::decompressEngineStreamSeq(uint8_t* in, uint8_t* out, size_t input_size) {
#ifdef DISABLE_FREE_RUNNING_KERNEL
#undef FREE_RUNNING_KERNEL
#endif
    cl_mem_ext_ptr_t p2pInExt;
    char* p2pPtr = NULL;
    uint32_t inputSize4KMultiple = 0;
    if (m_enableP2P) {
        // roundoff inputSize to 4K
        inputSize4KMultiple = roundoff(input_size, RESIDUE_4K) * RESIDUE_4K;
        // DDR buffer exyensions
        p2pInExt.flags = XCL_MEM_EXT_P2P_BUFFER;
        p2pInExt.obj = nullptr;
        p2pInExt.param = NULL;
    }
    std::vector<uint32_t, aligned_allocator<uint32_t> > decompressSize;
    uint32_t outputSize = (input_size * m_maxCR) + 16;
    cl::Buffer* bufferOutputSize;
    // Index calculation
    h_buf_in.resize(input_size);
    h_buf_out.resize(outputSize);
    h_buf_decompressSize.resize(sizeof(uint32_t));

    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);

    if (!m_enableP2P) std::memcpy(h_buf_in.data(), in, input_size);

    // Device buffer allocation
    if (m_enableP2P) {
        buffer_input =
            new cl::Buffer(*m_context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, inputSize4KMultiple, &p2pInExt);
        p2pPtr = (char*)m_q->enqueueMapBuffer(*(buffer_input), CL_TRUE, CL_MAP_READ, 0, inputSize4KMultiple);
    } else {
        buffer_input = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_size, h_buf_in.data());
    }
    buffer_output = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, outputSize, h_buf_out.data());
    bufferOutputSize = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint32_t),
                                      h_buf_decompressSize.data());

    uint32_t inputSize_32t = uint32_t(input_size);
    // set kernel arguments
    int narg = 0;
    decompress_data_mover_kernel->setArg(narg++, *(buffer_input));
    decompress_data_mover_kernel->setArg(narg++, *(buffer_output));
    decompress_data_mover_kernel->setArg(narg++, inputSize_32t);
    decompress_data_mover_kernel->setArg(narg, *(bufferOutputSize));
#ifndef FREE_RUNNING_KERNEL
#ifndef DISABLE_FREE_RUNNING_KERNEL
    decompress_kernel_snappy->setArg(3, inputSize_32t);
#endif
#endif
    if (!m_enableP2P) {
        // Migrate Memory - Map host to device buffers
        m_q->enqueueMigrateMemObjects({*(buffer_input), *(bufferOutputSize), *(buffer_output)}, 0);
        m_q->finish();
    }

    if (m_enableP2P) {
        int ret = read(fd_p2p_c_in, p2pPtr, inputSize4KMultiple);
        if (ret == -1)
            std::cout << "P2P: compress(): read() failed, err: " << ret << ", line: " << __LINE__ << std::endl;
    }

    // Measure kernel execution time
    if (!m_enableProfile) {
        kernel_start = std::chrono::high_resolution_clock::now();
    }
    // enqueue the kernels and wait for them to finish
    m_q->enqueueTask(*decompress_data_mover_kernel);
#ifndef FREE_RUNNING_KERNEL
    m_q->enqueueTask(*decompress_kernel_snappy);
#endif
    m_q->finish();
    if (!m_enableProfile) {
        kernel_end = std::chrono::high_resolution_clock::now();
    }
    // Migrate memory - Map device to host buffers
    m_q->enqueueMigrateMemObjects({*(buffer_output), *(bufferOutputSize)}, CL_MIGRATE_MEM_OBJECT_HOST);
    m_q->finish();

    uint32_t uncompressedSize = h_buf_decompressSize[0];
    std::memcpy(out, h_buf_out.data(), uncompressedSize);

    if (!m_enableProfile) {
        auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
        kernel_time_ns_1 += duration;
        float throughput_in_mbps_1 = (float)uncompressedSize * 1000 / kernel_time_ns_1.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    delete buffer_input;
    buffer_input = nullptr;
    delete buffer_output;
    buffer_output = nullptr;
    h_buf_in.clear();
    h_buf_out.clear();

    return uncompressedSize;
} // End of decompress
