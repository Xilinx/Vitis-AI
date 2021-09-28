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
#include "zlib.h"
#include "gzipXrtHost.hpp"

extern unsigned long crc32(unsigned long crc, const unsigned char* buf, uint32_t len);
extern unsigned long adler32(unsigned long crc, const unsigned char* buf, uint32_t len);

gzipXrtHost::gzipXrtHost(enum State flow,
                         const std::string& xclbin,
                         const std::string& uncompFileName,
                         const bool is_seq,
                         uint8_t device_id,
                         bool enable_profile,
                         uint8_t decKernelType,
                         uint8_t dflow,
                         bool freeRunKernel) {
    m_cdflow = flow;
    m_enableProfile = enable_profile;
    m_deviceid = device_id;
    m_zlibFlow = (enum design_flow)dflow;
    m_kidx = decKernelType;
    m_pending = 0;
    m_device = xrt::device(m_deviceid);
    m_uuid = m_device.load_xclbin(xclbin);
    m_isSeq = is_seq;
    m_freeRunKernel = freeRunKernel;
    m_inFileName = uncompFileName;
    if (m_zlibFlow) {
        m_checksum = 1;
        m_isZlib = true;
    }
}

// This version of compression does sequential execution between kernel an dhost
size_t gzipXrtHost::compressEngineSeq(
    uint8_t* in, uint8_t* out, size_t input_size, int level, int strategy, int window_bits, uint32_t* checksum) {
#ifdef GZIP_STREAM
    std::string compress_kname = compress_kernel_names[1];
#else
    std::string compress_kname = compress_kernel_names[2];
#endif
    auto compressKernel = xrt::kernel(m_device, m_uuid, compress_kname.c_str());

    if (this->is_freeRunKernel()) datamoverKernel = xrt::kernel(m_device, m_uuid, datamover_kernel_name.c_str());

    auto c_inputSize = HOST_BUFFER_SIZE;
    auto numItr = 1;

    if (this->is_freeRunKernel()) {
        c_inputSize = input_size;
        numItr = xcl::is_emulation() ? 1 : (0x40000000 / input_size);
    }

    bool checksum_type;

    // Host buffers
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_input_buffer(c_inputSize);
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_output_buffer(c_inputSize);
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_compressSize(10);
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_checkSumData(10);

    if (m_zlibFlow) {
        h_checkSumData.data()[0] = 1;
        checksum_type = false;
    } else {
        h_checkSumData.data()[0] = ~0;
        checksum_type = true;
    }

    xrt::bo buffer_input, buffer_output, buffer_cSize, buffer_checkSum;
    if (this->is_freeRunKernel()) {
        buffer_input =
            xrt::bo(m_device, h_input_buffer.data(), sizeof(uint8_t) * c_inputSize, datamoverKernel.group_id(0));
        buffer_output =
            xrt::bo(m_device, h_output_buffer.data(), sizeof(uint8_t) * c_inputSize, datamoverKernel.group_id(1));
        buffer_cSize = xrt::bo(m_device, h_compressSize.data(), sizeof(uint32_t) * 10, datamoverKernel.group_id(3));
    } else {
        buffer_input =
            xrt::bo(m_device, h_input_buffer.data(), sizeof(uint8_t) * c_inputSize, compressKernel.group_id(0));
        buffer_output =
            xrt::bo(m_device, h_output_buffer.data(), sizeof(uint8_t) * c_inputSize, compressKernel.group_id(1));
        buffer_cSize = xrt::bo(m_device, h_compressSize.data(), sizeof(uint32_t) * 10, compressKernel.group_id(2));
        buffer_checkSum = xrt::bo(m_device, h_checkSumData.data(), sizeof(uint32_t) * 10, compressKernel.group_id(3));
    }
    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);

    auto enbytes = 0;
    auto outIdx = 0;

    // Process the data serially
    for (uint32_t inIdx = 0; inIdx < input_size; inIdx += c_inputSize) {
        uint32_t buf_size = c_inputSize;
        if (inIdx + buf_size > input_size) buf_size = input_size - inIdx;

        // Copy input data
        std::memcpy(h_input_buffer.data(), &in[inIdx], buf_size);

        if (this->is_freeRunKernel()) {
            buffer_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            buffer_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            buffer_cSize.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else {
            buffer_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            buffer_checkSum.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        xrt::run datamover_run;
        xrt::run cKernel_run;
        xrt::run run;

        auto kernel_start = std::chrono::high_resolution_clock::now();

        if (this->is_freeRunKernel()) {
            datamover_run = datamoverKernel(buffer_input, buffer_output, buf_size, buffer_cSize, numItr);
#ifdef DISABLE_FREE_RUNNING_KERNEL
            cKernel_run =
                compressKernel(buffer_input, buffer_output, buffer_cSize, buffer_checkSum, buf_size, checksum_type);
#endif
        } else {
            // Running Kernel
            run = compressKernel(buffer_input, buffer_output, buffer_cSize, buffer_checkSum, buf_size, checksum_type);
        }

        if (this->is_freeRunKernel()) {
            datamover_run.wait();
#ifdef DISABLE_FREE_RUNNING_KERNEL
            cKernel_run.wait();
#endif
        } else {
            run.wait();
        }
        auto kernel_stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::nano>(kernel_stop - kernel_start);
        kernel_time_ns_1 += duration;

        if (!(this->is_freeRunKernel())) {
            // Copy data back
            buffer_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            buffer_cSize.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            buffer_checkSum.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        } else {
            // Copy data back
            buffer_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            buffer_cSize.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }

        auto compSize = h_compressSize[0];
        std::memcpy(out + outIdx, &h_output_buffer[0], compSize);
        enbytes += compSize;
        outIdx += compSize;
        if (!(this->is_freeRunKernel())) {
            // This handling required only for zlib
            // This is for next iteration
            if (checksum_type) h_checkSumData.data()[0] = ~h_checkSumData.data()[0];
        }
    }
    float throughput_in_mbps_1 = (float)input_size * 1000 * numItr / kernel_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;

    m_checksum = h_checkSumData.data()[0];
    if (checksum_type) m_checksum = ~m_checksum;
    *checksum = m_checksum;

    if (!(this->is_freeRunKernel())) {
        // Add last block header
        long int last_block = 0xffff000001;
        std::memcpy(&(out[outIdx]), &last_block, 5);
        enbytes += 5;
    }
    return enbytes;
} // Sequential end
