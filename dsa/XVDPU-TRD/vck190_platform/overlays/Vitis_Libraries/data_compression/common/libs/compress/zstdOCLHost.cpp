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
#include "zstdOCLHost.hpp"
#include <unistd.h>

// Constructor
zstdOCLHost::zstdOCLHost(enum State flow,
                         const std::string& binaryFileName,
                         uint8_t device_id,
                         uint8_t max_cr,
                         bool enable_profile,
                         uint32_t itrCnt)
    : m_flow(flow), m_xclbin(binaryFileName), m_deviceId(device_id), m_maxCr(max_cr), m_testItrCount(itrCnt) {
    // unsigned fileBufSize;
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[m_deviceId];

    // Creating Context and Command Queue for selected Device
    m_context = new cl::Context(device);

#ifndef FREE_RUNNING_KERNEL
    printf("Free Running mode\n");
    if (m_flow == BOTH || m_flow == COMPRESS) {
        m_q_cmp = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }
    if (m_flow == BOTH || m_flow == DECOMPRESS) {
        m_q_dec = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }
#endif
    if (m_flow == BOTH || m_flow == COMPRESS) {
        m_q_cdm = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }
    if (m_flow == BOTH || m_flow == DECOMPRESS) {
        m_q_rd = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
        m_q_wr = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    // import_binary() command will find the OpenCL binary file created using the
    // v++ compiler load into OpenCL Binary and return as Binaries
    // OpenCL and it can contain many functions which can be executed on the
    // device.
    auto fileBuf = xcl::read_binary_file(m_xclbin);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    m_program = new cl::Program(*m_context, {device}, bins);
    // Create Compress kernels
    if (m_flow == BOTH || m_flow == COMPRESS) {
        compress_kernel = new cl::Kernel(*m_program, compress_kernel_name.c_str());
        cmp_dm_kernel = new cl::Kernel(*m_program, cmp_dm_kernel_name.c_str());
    }
    // Create Decompress kernels
    if (m_flow == BOTH || m_flow == DECOMPRESS) {
        decompress_kernel = new cl::Kernel(*m_program, decompress_kernel_name.c_str());
        data_writer_kernel = new cl::Kernel(*m_program, data_writer_kernel_name.c_str());
        data_reader_kernel = new cl::Kernel(*m_program, data_reader_kernel_name.c_str());
    }
}

// Destructor
zstdOCLHost::~zstdOCLHost() {
    if (decompress_kernel != nullptr) {
        delete decompress_kernel;
        decompress_kernel = nullptr;
    }
    if (data_writer_kernel != nullptr) {
        delete data_writer_kernel;
        data_writer_kernel = nullptr;
    }
    if (data_reader_kernel != nullptr) {
        delete data_reader_kernel;
        data_reader_kernel = nullptr;
    }

#ifndef FREE_RUNNING_KERNEL
    if (m_flow == COMPRESS) delete (m_q_cmp);
    if (m_flow == DECOMPRESS) delete (m_q_dec);
#endif
    if (m_flow == COMPRESS) delete (m_q_cdm);
    if (m_flow == DECOMPRESS) {
        delete (m_q_rd);
        delete (m_q_wr);
    }
    delete (m_program);
    delete (m_context);
}

void zstdOCLHost::setTestItrCount(uint16_t itrcnt) {
    m_testItrCount = itrcnt;
}

uint64_t zstdOCLHost::xilCompress(uint8_t* in, uint8_t* out, size_t input_size) {
    uint64_t enbytes = 0;
    if (m_enableProfile) {
        total_start = std::chrono::high_resolution_clock::now();
    }
    enbytes = compressEngine(in, out, input_size);
    if (m_enableProfile) {
        total_end = std::chrono::high_resolution_clock::now();
        auto total_time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
        float throughput_in_mbps_1 = (float)input_size * 1000 / total_time_ns.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    return enbytes;
}

uint64_t zstdOCLHost::xilDecompress(uint8_t* in, uint8_t* out, size_t input_size) {
    uint64_t debytes = 0;
    if (m_enableProfile) {
        total_start = std::chrono::high_resolution_clock::now();
    }
    debytes = decompressEngine(in, out, input_size);
    if (m_enableProfile) {
        total_end = std::chrono::high_resolution_clock::now();
        auto total_time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
        float throughput_in_mbps_1 = (float)input_size * 1000 / total_time_ns.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    return debytes;
}

uint64_t zstdOCLHost::compressEngine(uint8_t* in, uint8_t* out, size_t input_size) {
    cl_int err;
    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);
    auto c_inputSize = CMP_HOST_BUF_SIZE;
#ifdef FREE_RUNNING_KERNEL
    c_inputSize = input_size;
#endif
    // host allocated aligned memory
    cbuf_in.resize(c_inputSize);
    cbuf_out.resize(c_inputSize);
    cbuf_outSize.resize(2);

    // opencl buffer creation
    OCL_CHECK(err, buffer_cmp_input = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, c_inputSize,
                                                     cbuf_in.data(), &err));
    OCL_CHECK(err, buffer_cmp_output = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, c_inputSize,
                                                      cbuf_out.data(), &err));

    OCL_CHECK(err, buffer_cmp_size = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                                    sizeof(uint32_t), cbuf_outSize.data(), &err));

    // set consistent kernel arguments
    cmp_dm_kernel->setArg(0, *buffer_cmp_input);
    cmp_dm_kernel->setArg(1, *buffer_cmp_output);
    cmp_dm_kernel->setArg(3, *buffer_cmp_size);

    auto enbytes = 0;
    auto outIdx = 0;
    // Process the data serially
    for (size_t inIdx = 0; inIdx < input_size; inIdx += c_inputSize) {
        uint32_t buf_size = c_inputSize;
        if (inIdx + buf_size > input_size) buf_size = input_size - inIdx;

        // Copy input data
        std::memcpy(cbuf_in.data(), &in[inIdx], buf_size);

        // Set Variable Kernel Args
        cmp_dm_kernel->setArg(2, buf_size);

        // Transfer the data to device
        OCL_CHECK(err, err = m_q_cdm->enqueueMigrateMemObjects({*(buffer_cmp_input)}, 0, nullptr, nullptr));
        m_q_cdm->finish();

        auto kernel_start = std::chrono::high_resolution_clock::now();
#ifndef FREE_RUNNING_KERNEL
        OCL_CHECK(err, err = m_q_cmp->enqueueTask(*compress_kernel));
        m_q_cdm->enqueueTask(*cmp_dm_kernel);

        m_q_cmp->finish();
        auto kernel_stop = std::chrono::high_resolution_clock::now();

        m_q_cdm->finish();
#else
        printf("Free running kernel\n");
        m_q_cdm->enqueueTask(*cmp_dm_kernel);
        m_q_cdm->finish();
        auto kernel_stop = std::chrono::high_resolution_clock::now();
#endif

        auto duration = std::chrono::duration<double, std::nano>(kernel_stop - kernel_start);
        kernel_time_ns_1 += duration;

        // Transfer the data from device to host
        OCL_CHECK(err, err = m_q_cdm->enqueueMigrateMemObjects({*(buffer_cmp_output), *(buffer_cmp_size)},
                                                               CL_MIGRATE_MEM_OBJECT_HOST));
        m_q_cdm->finish();

        auto compSize = cbuf_outSize[0];
        std::memcpy(out + outIdx, cbuf_out.data(), compSize);
        enbytes += compSize;
        outIdx += compSize;
    }
    m_q_cdm->finish();
    if (!m_enableProfile) {
        float throughput_in_mbps_1 = ((float)input_size * m_testItrCount * 1000) / kernel_time_ns_1.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    // free the cl buffers
    delete buffer_cmp_input;
    delete buffer_cmp_output;
    delete buffer_cmp_size;

    return enbytes;
} // End of compress

uint64_t zstdOCLHost::decompressEngine(uint8_t* in, uint8_t* out, size_t input_size) {
    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);
    uint32_t inBufferSize = input_size;
    uint32_t isLast = 1;
    const uint64_t max_outbuf_size = input_size * m_maxCr;
    const uint32_t lim_4gb = (uint32_t)(((uint64_t)4 * 1024 * 1024 * 1024) - 2); // 4GB limit on output size
    uint32_t outBufferSize = 0;
    // allocate < 4GB size for output buffer
    if (max_outbuf_size > lim_4gb) {
        outBufferSize = lim_4gb;
    } else {
        outBufferSize = (uint32_t)max_outbuf_size;
    }
    // host allocated aligned memory
    dbuf_in.resize(inBufferSize);
    dbuf_out.resize(outBufferSize);
    dbuf_outSize.resize(2);
    h_dcompressStatus.resize(2);

    // opencl buffer creation
    buffer_dec_input = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, inBufferSize, dbuf_in.data());
    buffer_dec_output =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, outBufferSize, dbuf_out.data());

    buffer_size =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(uint32_t), dbuf_outSize.data());
    buffer_status =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint32_t), h_dcompressStatus.data());

    // Set Kernel Args
    data_writer_kernel->setArg(0, *(buffer_dec_input));
    data_writer_kernel->setArg(1, inBufferSize);
    data_writer_kernel->setArg(2, isLast);

    data_reader_kernel->setArg(0, *(buffer_dec_output));
    data_reader_kernel->setArg(1, *(buffer_size));
    data_reader_kernel->setArg(2, *(buffer_status));
    data_reader_kernel->setArg(3, outBufferSize);

    // Copy input data
    std::memcpy(dbuf_in.data(), in, inBufferSize); // must be equal to input_size
    m_q_wr->enqueueMigrateMemObjects({*(buffer_dec_input)}, 0, NULL, NULL);
    m_q_wr->finish();

    // start parallel reader kernel enqueue thread
    uint64_t decmpSizeIdx = 0;

#ifndef FREE_RUNNING_KERNEL
    // make sure that command queue is empty before decompression kernel enqueue
    m_q_dec->finish();

    // enqueue data movers
    m_q_wr->enqueueTask(*data_writer_kernel);
    m_q_rd->enqueueTask(*data_reader_kernel);

    // enqueue decompression kernel
    if (!m_enableProfile) {
        sleep(1);
        kernel_start = std::chrono::high_resolution_clock::now();
    }
    m_q_dec->enqueueTask(*decompress_kernel);
    m_q_dec->finish();
    if (!m_enableProfile) {
        kernel_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
        kernel_time_ns_1 += duration;
    }
    // wait for reader to finish
    m_q_rd->finish();
#else
    // enqueue data movers
    m_q_rd->enqueueTask(*data_reader_kernel);

    if (!m_enableProfile) {
        sleep(1);
        kernel_start = std::chrono::high_resolution_clock::now();
    }
    m_q_wr->enqueueTask(*data_writer_kernel);
    m_q_wr->finish();
    if (!m_enableProfile) {
        kernel_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
        kernel_time_ns_1 += duration;
    }
#endif
    // copy decompressed output data
    m_q_rd->enqueueMigrateMemObjects({*(buffer_size), *(buffer_dec_output)}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, NULL);
    m_q_rd->finish();
    // decompressed size
    decmpSizeIdx = dbuf_outSize[0];
    // copy output decompressed data
    std::memcpy(out, dbuf_out.data(), decmpSizeIdx);
    if (!m_enableProfile) {
        float throughput_in_mbps_1 = (float)decmpSizeIdx * 1000 / kernel_time_ns_1.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }
    // free the cl buffers
    delete buffer_dec_input;
    delete buffer_dec_output;
    delete buffer_size;
    delete buffer_status;

    // printme("Done with decompress \n");
    return decmpSizeIdx;

} // End of decompress
