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
#include "zlib.hpp"
#include "zlib.h"
using namespace xf::compression;
extern unsigned long crc32(unsigned long crc, const unsigned char* buf, uint32_t len);
extern unsigned long adler32(unsigned long crc, const unsigned char* buf, uint32_t len);

using namespace xf::compression;

void xf::compression::event_compress_cb(cl_event event, cl_int cmd_status, void* data) {
    // callBackMutex.lock();
    assert(data != NULL);
    ((xf::compression::buffers*)(data))->compress_finish = true;
    // callBackMutex.unlock();
}

void xf::compression::event_decompress_cb(cl_event event, cl_int cmd_status, void* data) {
    // callBackMutex.lock();
    assert(data != NULL);
    ((xf::compression::buffers*)(data))->compress_finish = true;
    // callBackMutex.unlock();
}

void xf::compression::event_copydone_cb(cl_event event, cl_int cmd_status, void* data) {
    // callBackMutex.lock();
    assert(data != NULL);
    ((xf::compression::buffers*)(data))->copy_done = true;
    ((xf::compression::buffers*)(data))->compress_finish = true;
    // callBackMutex.unlock();
}

void xf::compression::event_decompressWrite_cb(cl_event event, cl_int cmd_status, void* data) {
    // callBackMutex.lock();
    assert(data != NULL);
    ((xf::compression::buffers*)(data))->compress_finish = true;
    // callBackMutex.unlock();
}

uint64_t get_file_size(std::ifstream& file) {
    file.seekg(0, file.end);
    uint64_t file_size = file.tellg();
    if (file_size == 0) {
        std::cout << "File is empty!" << std::endl;
        exit(0);
    }
    file.seekg(0, file.beg);
    return file_size;
}

void xfZlib::set_checksum(uint32_t cval) {
    m_checksum = cval;
}

uint32_t xfZlib::get_checksum(void) {
    return m_checksum;
}

uint64_t xfZlib::compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, int cu) {
    m_infile = inFile_name;
    std::chrono::duration<double, std::nano> compress_API_time_ns_1(0);
    std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
    std::ofstream outFile(outFile_name.c_str(), std::ofstream::binary);

    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }

    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_in(input_size);
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out(input_size * 2);

    MEM_ALLOC_CHECK(zlib_in.resize(input_size), input_size, "Input Buffer");
    MEM_ALLOC_CHECK(zlib_out.resize(input_size * 2), input_size * 2, "Output Buffer");

    inFile.read((char*)zlib_in.data(), input_size);

    auto compress_API_start = std::chrono::high_resolution_clock::now();
    uint64_t enbytes = 0;
    // zlib Compress
    enbytes = compress(zlib_in.data(), zlib_out.data(), input_size, cu);

    auto compress_API_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::nano>(compress_API_end - compress_API_start);
    compress_API_time_ns_1 += duration;

    float throughput_in_mbps_1 = (float)input_size * 1000 / compress_API_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(3) << throughput_in_mbps_1;

    outFile.write((char*)zlib_out.data(), enbytes);

    // Close file
    inFile.close();
    outFile.close();
    return enbytes;
}

int validate(std::string& inFile_name, std::string& outFile_name) {
    std::string command = "cmp " + inFile_name + " " + outFile_name;
    int ret = system(command.c_str());
    return ret;
}

// OpenCL setup initialization

int xfZlib::init(const std::string& binaryFileName, uint8_t kidx) {
    // Error handling in OpenCL is performed using the cl_int specifier. OpenCL
    // functions either return or accept pointers to cl_int types to indicate if
    // an error occurred.
    cl_int err;

    if (!m_skipContextProgram) {
        // Look for platform
        std::vector<cl::Platform> platforms;
        OCL_CHECK(err, err = cl::Platform::get(&platforms));
        auto num_platforms = platforms.size();
        if (num_platforms == 0) {
            std::cerr << "No Platforms were found this could be cased because of the OpenCL \
                          ICD not installed at /etc/OpenCL/vendors directory"
                      << std::endl;
            m_err_code = -32;
            return -32;
        }

        std::string platformName;
        cl::Platform platform;
        bool foundFlag = false;
        for (auto p : platforms) {
            platform = p;
            OCL_CHECK(err, platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
            if (platformName == "Xilinx") {
                foundFlag = true;
#if (VERBOSE_LEVEL >= 2)
                std::cout << "Found Platform" << std::endl;
                std::cout << "Platform Name: " << platformName.c_str() << std::endl;
#endif
                break;
            }
        }
        if (foundFlag == false) {
            std::cerr << "Error: Failed to find Xilinx platform" << std::endl;
            m_err_code = 1;
            return 1;
        }
        // Getting ACCELERATOR Devices and selecting 1st such device
        std::vector<cl::Device> devices;
        OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
        m_device = devices[m_deviceid];

        // OpenCL Setup Start
        // Creating Context and Command Queue for selected Device
        cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        OCL_CHECK(err, m_context = new cl::Context(m_device, props, NULL, NULL, &err));
        if (err) {
            std::cerr << "Context creation Failed " << std::endl;
            m_err_code = 1;
            return 1;
        }
// Import_binary() command will find the OpenCL binary file created using the
// v++ compiler load into OpenCL Binary and return as Binaries
// OpenCL and it can contain many functions which can be executed on the
// device.
#if (VERBOSE_LEVEL >= 2)
        std::cout << "INFO: Reading " << binaryFileName << std::endl;
#endif
        if (access(binaryFileName.c_str(), R_OK) != 0) {
            std::cerr << "ERROR: " << binaryFileName.c_str() << " xclbin not available please build " << std::endl;
            m_err_code = 1;
            return 1;
        }

#if (VERBOSE_LEVEL >= 2)
        // Loading XCL Bin into char buffer
        std::cout << "Loading: '" << binaryFileName.c_str() << "'\n";
#endif
        std::ifstream bin_file(binaryFileName.c_str(), std::ifstream::binary);
        if (bin_file.fail()) {
            std::cerr << "Unable to open binary file" << std::endl;
            m_err_code = 1;
            return 1;
        }

        bin_file.seekg(0, bin_file.end);
        auto nb = bin_file.tellg();
        bin_file.seekg(0, bin_file.beg);

        std::vector<uint8_t> buf;
        MEM_ALLOC_CHECK(buf.resize(nb), nb, "XCLBIN Buffer");
        bin_file.read(reinterpret_cast<char*>(buf.data()), nb);

        cl::Program::Binaries bins{{buf.data(), buf.size()}};
        ZOCL_CHECK_2(err, m_program = new cl::Program(*m_context, {m_device}, bins, NULL, &err), m_err_code,
                     c_clinvalidbin, c_clinvalidprogram);
        if (error_code()) {
            m_err_code = 0;
            std::ifstream bin_file((std::string(c_installRootDir) + c_hardFullXclbinPath), std::ifstream::binary);
            if (bin_file.fail()) {
                std::cerr << "Unable to open binary file" << std::endl;
                m_err_code = 1;
                return 1;
            }

            bin_file.seekg(0, bin_file.end);
            auto nb = bin_file.tellg();
            bin_file.seekg(0, bin_file.beg);

            buf.clear();
            MEM_ALLOC_CHECK(buf.resize(nb), nb, "XCLBIN Buffer");
            bin_file.read(reinterpret_cast<char*>(buf.data()), nb);

            cl::Program::Binaries bins{{buf.data(), buf.size()}};
            ZOCL_CHECK_2(err, m_program = new cl::Program(*m_context, {m_device}, bins, NULL, &err), m_err_code,
                         c_clinvalidbin, c_clinvalidvalue);
            if (error_code()) {
                std::cerr << "Failed to program the device " << std::endl;
                return m_err_code;
            }
        }
    }

    // Create Command Queue
    // Compress Command Queue & Kernel Setup
    if ((m_cdflow == BOTH) || (m_cdflow == COMP_ONLY)) {
        OCL_CHECK(err,
                  m_def_q = new cl::CommandQueue(*m_context, m_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    }

    // DeCompress Command Queue & Kernel Setup
    if ((m_cdflow == BOTH) || (m_cdflow == DECOMP_ONLY)) {
#ifndef FREE_RUNNING_KERNEL
        for (uint8_t i = 0; i < D_COMPUTE_UNIT; i++) {
            OCL_CHECK(err, m_q_dec[i] = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err));
        }
#endif
        for (uint8_t i = 0; i < D_COMPUTE_UNIT; i++) {
            OCL_CHECK(err, m_q_rd[i] = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err));
            OCL_CHECK(err, m_q_rdd[i] = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err));
            OCL_CHECK(err, m_q_wr[i] = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err));
            OCL_CHECK(err, m_q_wrd[i] = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err));
        }
        m_rd_q = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err);
        m_wr_q = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err);
        m_dec_q = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err);
    }
    // OpenCL Host / Device Buffer Setup Start
    return 0;
}

xfZlib::xfZlib(const std::string& binaryFileName,
               bool sb_opt,
               uint8_t cd_flow,
               cl::Context* context,
               cl::Program* program,
               const cl::Device& device,
               enum design_flow dflow,
               const int bank_id) {
    m_cdflow = cd_flow;
    m_device = device;
    m_pending = 0;
    m_context = context;
    m_program = program;
    m_islibz = true;
    m_isSlaveBridge = sb_opt;
    if (m_zlibFlow) m_checksum = 1;
    if (m_cdflow == COMP_ONLY)
        m_kidx = 1;
    else
        m_kidx = 2;

    m_skipContextProgram = true;

    // OpenCL setup
    int err = init(binaryFileName, m_kidx);
    if (err) {
        std::cerr << "\nOpenCL Setup Failed" << std::endl;
        release();
        return;
    }

    if ((cd_flow == BOTH) || (cd_flow == COMP_ONLY)) {
        // Create memory manager object
        m_memMgrCmp = new xf::compression::memoryManager(8, m_context, m_isSlaveBridge, m_def_q);
    }

    if ((cd_flow == BOTH) || (cd_flow == DECOMP_ONLY)) {
        m_memMgrDecMM2S = new xf::compression::memoryManager(8, m_context, m_isSlaveBridge, m_rd_q, bank_id);
        m_memMgrDecS2MM = new xf::compression::memoryManager(8, m_context, m_isSlaveBridge, m_wr_q, bank_id);
    }

    if ((cd_flow == BOTH) || (cd_flow == COMP_ONLY)) {
        if (isSlaveBridge()) {
            h_buf_checksum = (uint32_t*)m_memMgrCmp->create_host_buffer(CL_MEM_READ_WRITE, sizeof(uint32_t),
                                                                        &(buffer_checksum_data));
            if (h_buf_checksum == nullptr) {
                release();
                m_err_code = true;
                return;
            }
        } else {
            MEM_ALLOC_CHECK(h_buf_checksum_data.resize(1), 1, "Checksum Data");
            buffer_checksum_data = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint32_t),
                                                  h_buf_checksum_data.data(), &err);
        }
    }

    if ((cd_flow == BOTH) || (cd_flow == DECOMP_ONLY)) {
        // Decompression host buffer allocation
        for (int j = 0; j < DIN_BUFFERCOUNT; ++j)
            MEM_ALLOC_CHECK(h_dbufstream_in[j].resize(INPUT_BUFFER_SIZE), INPUT_BUFFER_SIZE, "Input Buffer");

        for (int j = 0; j < DOUT_BUFFERCOUNT; ++j) {
            MEM_ALLOC_CHECK(h_dbufstream_zlibout[j].resize(OUTPUT_BUFFER_SIZE), OUTPUT_BUFFER_SIZE,
                            "Output Host Buffer");
            MEM_ALLOC_CHECK(h_dcompressSize_stream[j].resize(sizeof(uint32_t)), sizeof(uint32_t),
                            "DecompressSize Host Buffer");
        }
        if (!(isSlaveBridge())) {
            MEM_ALLOC_CHECK(h_dcompressStatus.resize(sizeof(uint32_t)), sizeof(uint32_t),
                            "DecompressStatus Host Buffer");
            h_dcompressStatus[0] = 0;
            OCL_CHECK(err, buffer_status = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                          sizeof(uint32_t), h_dcompressStatus.data(), &err));
        }
    }
}

// Constructor
xfZlib::xfZlib(const std::string& binaryFileName,
               bool sb_opt,
               uint8_t max_cr,
               uint8_t cd_flow,
               uint8_t device_id,
               uint8_t profile,
               uint8_t d_type,
               enum design_flow dflow,
               uint8_t ccu,
               uint8_t dcu) {
    // Zlib Compression Binary Name
    m_cdflow = cd_flow;
    m_isProfile = profile;
    m_deviceid = device_id;
    m_max_cr = max_cr;
    m_zlibFlow = dflow;
    m_kidx = d_type;
    m_pending = 0;
    m_isSlaveBridge = sb_opt;
    if (m_zlibFlow) m_checksum = 1;
    uint8_t kidx = d_type;

    // OpenCL setup
    int err = init(binaryFileName, kidx);
    if (err) {
        std::cerr << "\nOpenCL Setup Failed" << std::endl;
        release();
        return;
    }

    if ((cd_flow == BOTH) || (cd_flow == COMP_ONLY)) {
        m_memMgrCmp = new xf::compression::memoryManager(8, m_context, sb_opt, m_def_q);
    }

    if ((cd_flow == BOTH) || (cd_flow == DECOMP_ONLY)) {
        m_memMgrDecMM2S = new xf::compression::memoryManager(8, m_context, sb_opt, m_rd_q);
        m_memMgrDecS2MM = new xf::compression::memoryManager(8, m_context, sb_opt, m_wr_q);
    }

    if ((cd_flow == BOTH) || (cd_flow == COMP_ONLY)) {
        if (isSlaveBridge()) {
            h_buf_checksum = (uint32_t*)m_memMgrCmp->create_host_buffer(CL_MEM_READ_WRITE, sizeof(uint32_t),
                                                                        &(buffer_checksum_data));
            if (h_buf_checksum == nullptr) {
                release();
                m_err_code = true;
                return;
            }
        } else {
            MEM_ALLOC_CHECK(h_buf_checksum_data.resize(1), 1, "Checksum Data");
            cl_int err;
            OCL_CHECK(err, buffer_checksum_data = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                                 sizeof(uint32_t), h_buf_checksum_data.data(), &err));
        }
    }

    if ((cd_flow == BOTH) || (cd_flow == DECOMP_ONLY)) {
        // Decompression host buffer allocation
        for (int j = 0; j < DIN_BUFFERCOUNT; ++j)
            MEM_ALLOC_CHECK(h_dbufstream_in[j].resize(INPUT_BUFFER_SIZE), INPUT_BUFFER_SIZE, "Input Buffer");

        for (int j = 0; j < DOUT_BUFFERCOUNT; ++j) {
            MEM_ALLOC_CHECK(h_dbufstream_zlibout[j].resize(OUTPUT_BUFFER_SIZE), OUTPUT_BUFFER_SIZE,
                            "Output Host Buffer");
            MEM_ALLOC_CHECK(h_dcompressSize_stream[j].resize(sizeof(uint32_t)), sizeof(uint32_t),
                            "DecompressSize Host Buffer");
        }
        if (!(isSlaveBridge())) {
            MEM_ALLOC_CHECK(h_dcompressStatus.resize(sizeof(uint32_t)), sizeof(uint32_t),
                            "DecompressStatus Host Buffer");
            OCL_CHECK(err, buffer_status = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                          sizeof(uint32_t), h_dcompressStatus.data(), &err));
        }
    }
    // OpenCL Host / Device Buffer Setup End
}

bool xfZlib::isSlaveBridge(void) {
    return this->m_isSlaveBridge;
}

int xfZlib::error_code(void) {
    return m_err_code;
}

void xfZlib::release_dec_buffers(void) {
    // delete the allocated buffers
    for (int i = 0; i < DIN_BUFFERCOUNT; i++) {
        DELETE_OBJ(buffer_dec_input[i]);
    }

    for (int i = 0; i < DOUT_BUFFERCOUNT + 1; i++) {
        DELETE_OBJ(buffer_dec_zlib_output[i]);
    }
}

void xfZlib::release() {
    if (!m_islibz) {
        DELETE_OBJ(m_program);
        DELETE_OBJ(m_context);
    }

    if ((m_cdflow == BOTH) || (m_cdflow == COMP_ONLY)) {
        DELETE_OBJ(buffer_checksum_data);
    }

    // Decompress release
    if ((m_cdflow == BOTH) || (m_cdflow == DECOMP_ONLY)) {
        for (uint8_t i = 0; i < D_COMPUTE_UNIT; i++) {
// Destroy command queues
#ifndef FREE_RUNNING_KERNEL
            DELETE_OBJ(m_q_dec[i]);
#endif
            DELETE_OBJ(m_q_rd[i]);
            DELETE_OBJ(m_q_rdd[i]);
            DELETE_OBJ(m_q_wr[i]);
            DELETE_OBJ(m_q_wrd[i]);
        }
    }

    if (m_memMgrCmp) {
        delete m_memMgrCmp;
    }

    if (m_memMgrDecMM2S) {
        delete m_memMgrDecMM2S;
    }

    if (m_memMgrDecS2MM) {
        delete m_memMgrDecS2MM;
    }

    if (m_def_q) {
        delete m_def_q;
    }

    DELETE_OBJ(m_rd_q);
    DELETE_OBJ(m_wr_q);
    DELETE_OBJ(m_dec_q);

    if (m_compressFullKernel) {
        delete m_compressFullKernel;
    }

    if (m_decompressKernel) {
        delete m_decompressKernel;
    }

    if (m_dataReaderKernel) {
        delete m_dataReaderKernel;
    }

    if (m_dataWriterKernel) {
        delete m_dataWriterKernel;
    }
}

// Destructor

xfZlib::~xfZlib() {
    release();
}

uint32_t xfZlib::decompress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, int cu) {
    // printme("In decompress_file \n");
    std::chrono::duration<double, std::nano> decompress_API_time_ns_1(0);
    std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
    std::ofstream outFile(outFile_name.c_str(), std::ofstream::binary);

    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }

    uint32_t output_buf_size = input_size * m_max_cr;

    std::vector<uint8_t, aligned_allocator<uint8_t> > in;
    // Allocat output size
    // 8 - Max CR per file expected, if this size is big
    // Decompression crashes
    std::vector<uint8_t, aligned_allocator<uint8_t> > out(output_buf_size);

    MEM_ALLOC_CHECK(in.resize(input_size), input_size, "Input Buffer");
    MEM_ALLOC_CHECK(in.resize(output_buf_size), output_buf_size, "Output Buffer");

    uint32_t debytes = 0;
    inFile.read((char*)in.data(), input_size);

    // printme("Call to zlib_decompress \n");
    // Call decompress
    auto decompress_API_start = std::chrono::high_resolution_clock::now();
    debytes = decompress(in.data(), out.data(), input_size, output_buf_size, cu);
    auto decompress_API_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::nano>(decompress_API_end - decompress_API_start);
    decompress_API_time_ns_1 = duration;

    if (debytes == 0) {
        std::cerr << "Decompression Failed" << std::endl;
        return 0;
    }

    float throughput_in_mbps_1 = (float)debytes * 1000 / decompress_API_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(3) << throughput_in_mbps_1;
    outFile.write((char*)out.data(), debytes);

    // Close file
    inFile.close();
    outFile.close();

    return debytes;
}

// method to enqueue reads in parallel with writes to decompression kernel

void xfZlib::_enqueue_reads(uint32_t bufSize, uint8_t* out, uint32_t* decompSize, int cu, uint32_t max_outbuf_size) {
    const int BUFCNT = DOUT_BUFFERCOUNT;
    cl::Event hostReadEvent[BUFCNT];
    cl::Event kernelReadEvent[BUFCNT];

    std::vector<cl::Event> kernelReadWait[BUFCNT];

    uint8_t* outP = nullptr;
    uint32_t* outSize = nullptr;
    uint32_t dcmpSize = 0;
    cl::Buffer* buffer_size[BUFCNT];
    cl::Buffer* buffer_status; // single common buffer to capture the
                               // decompression status by kernel

    cl_int err;
    std::string cu_id = std::to_string((cu + 1));
    std::string data_reader_kname = data_reader_kernel_name + ":{" + data_reader_kernel_name + "_" + cu_id + "}";
    OCL_CHECK(err, cl::Kernel data_reader_kernel(*m_program, data_reader_kname.c_str(), &err));

    for (int i = 0; i < BUFCNT; i++) {
        OCL_CHECK(err, buffer_size[i] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                                       2 * sizeof(uint32_t), h_dcompressSize_stream[i].data(), &err));
    }

    OCL_CHECK(err, buffer_status = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint32_t),
                                                  h_dcompressStatus.data(), &err));

    // set consistent buffer size to be read
    OCL_CHECK(err, err = data_reader_kernel.setArg(2, *buffer_status));
    OCL_CHECK(err, err = data_reader_kernel.setArg(3, bufSize));

    // enqueue first set of buffers
    uint8_t cbf_idx = 0;
    uint32_t keq_idx = 0;
    uint32_t raw_size = 0;
    uint32_t cpy_cnt = 0;
    bool done = false;

    do {
        cbf_idx = keq_idx % BUFCNT;
        if (((kernelReadWait[cbf_idx]).size() > 0) || (done)) {
            (hostReadEvent[cbf_idx]).wait(); // wait for previous data migration to complete
            if (!done) {
                outSize = h_dcompressSize_stream[cbf_idx].data();
                raw_size = *outSize;
                outP = h_dbufstream_zlibout[cbf_idx].data();
                // if output data size is multiple of buffer size, then (buffer_size +
                // 1) is sent by reader kernel
                if (raw_size > bufSize) {
                    --raw_size;
                }
                if (raw_size != 0) {
                    if (!(m_derr_code) && (dcmpSize + raw_size < max_outbuf_size)) {
                        std::memcpy(out + dcmpSize, outP, raw_size);
                    }
                    dcmpSize += raw_size;
                }
                if (raw_size != bufSize) done = true;

                if ((dcmpSize > max_outbuf_size) && (max_outbuf_size != 0)) {
#if (VERBOSE_LEVEL >= 1)
                    std::cout << "\n" << std::endl;
                    std::cout << "\x1B[35mZIP BOMB: Exceeded output buffer size during "
                                 "decompression \033[0m \n"
                              << std::endl;
                    std::cout << "\x1B[35mUse -mcr option to increase the maximum "
                                 "compression ratio (Default: 10) \033[0m \n"
                              << std::endl;
                    std::cout << "\x1B[35mAborting .... \033[0m\n" << std::endl;
#endif
                    m_derr_code = true;
                }
                if ((kernelReadWait[cbf_idx]).size() > 0)
                    kernelReadWait[cbf_idx].pop_back(); // must always have single element
            }
            ++cpy_cnt;
        }
        // need to avoid full buffer copies for 0 bytes size data
        if (!done) {
            // set reader kernel arguments
            OCL_CHECK(err, err = data_reader_kernel.setArg(0, *(buffer_dec_zlib_output[cbf_idx])));
            OCL_CHECK(err, err = data_reader_kernel.setArg(1, *(buffer_size[cbf_idx])));

            // enqueue reader kernel
            OCL_CHECK(err, err = m_q_rd[cu]->enqueueTask(data_reader_kernel, NULL, &(kernelReadEvent[cbf_idx])));
            kernelReadWait[cbf_idx].push_back(kernelReadEvent[cbf_idx]); // event to wait for

            OCL_CHECK(err, err = m_q_rdd[cu]->enqueueMigrateMemObjects(
                               {*(buffer_size[cbf_idx]), *(buffer_dec_zlib_output[cbf_idx])},
                               CL_MIGRATE_MEM_OBJECT_HOST, &(kernelReadWait[cbf_idx]), &(hostReadEvent[cbf_idx])));
            ++keq_idx;
        }
    } while (!(done && keq_idx == cpy_cnt));

    // wait for data transfer queue to finish
    OCL_CHECK(err, err = m_q_rdd[cu]->finish());
    *decompSize = dcmpSize;

    // free the buffers
    for (int i = 0; i < BUFCNT; i++) delete (buffer_size[i]);

    delete buffer_status;
}

// method to enqueue writes in parallel with reads from decompression kernel

void xfZlib::_enqueue_writes(uint32_t bufSize, uint8_t* in, uint32_t inputSize, int cu) {
    const int BUFCNT = DIN_BUFFERCOUNT;
    cl_int err;

    std::string cu_id = std::to_string(cu + 1);
    std::string data_writer_kname = data_writer_kernel_name + ":{" + data_writer_kernel_name + "_" + cu_id + "}";
    OCL_CHECK(err, cl::Kernel data_writer_kernel(*m_program, data_writer_kname.c_str(), &err));

    uint32_t bufferCount = 1 + (inputSize - 1) / bufSize;

    uint8_t* inP = nullptr;
    cl::Event hostWriteEvent[BUFCNT];
    cl::Event kernelWriteEvent[BUFCNT];
    std::vector<cl::Event> hostWriteWait[BUFCNT];

    uint32_t cBufSize = bufSize;
    uint32_t isLast = 0;
    uint8_t cbf_idx = 0;  // index indicating current buffer(0-4) being used in loop
    uint32_t keq_idx = 0; // index indicating the number of writer kernel enqueues

    for (keq_idx = 0; keq_idx < bufferCount; ++keq_idx) {
        cbf_idx = keq_idx % BUFCNT;

        if (keq_idx > BUFCNT - 1) {
            // wait for (current - BUFCNT) kernel to finish
            (kernelWriteEvent[cbf_idx]).wait();
        }
        inP = h_dbufstream_in[cbf_idx].data();
        // set for last and other buffers
        if (keq_idx == bufferCount - 1) {
            if (bufferCount > 1) {
                cBufSize = inputSize - (bufSize * keq_idx);
            }
            isLast = 1;
        }

        // copy the data
        std::memcpy(inP, in + (keq_idx * bufSize), cBufSize);

        // set kernel arguments
        OCL_CHECK(err, err = data_writer_kernel.setArg(0, *(buffer_dec_input[cbf_idx])));
        OCL_CHECK(err, err = data_writer_kernel.setArg(1, cBufSize));
        OCL_CHECK(err, err = data_writer_kernel.setArg(2, isLast));

        // enqueue data migration to kernel
        OCL_CHECK(err, err = m_q_wr[cu]->enqueueMigrateMemObjects({*(buffer_dec_input[cbf_idx])}, 0, NULL, NULL));
        // enqueue the writer kernel dependent on corresponding bufffer migration
        OCL_CHECK(err, err = m_q_wr[cu]->enqueueTask(data_writer_kernel, NULL, &(kernelWriteEvent[cbf_idx])));
    }
    // wait for enqueued writer kernels to finish
    OCL_CHECK(err, err = m_q_wr[cu]->finish());
}

size_t xfZlib::decompress_buffer(uint8_t* in, uint8_t* out, size_t input_size, size_t max_outbuf_size, int cu) {
    // Create kernel
    std::string cu_id = std::to_string((cu + 1));
    std::string kname = stream_decompress_kernel_name[2] + "_" + cu_id;
    uint64_t decompress_size = 0;
    size_t host_buffer_size = HOST_BUFFER_SIZE;
    auto in_size = input_size;
    auto in_ptr = in;
    long int lastBlockVal = 0;
    auto last_block = false;
    while (in_size > 0) {
        size_t chunk_size = (in_size > host_buffer_size) ? host_buffer_size : in_size;
        size_t chunk_size1 = chunk_size;
        in_size -= chunk_size;

        if (chunk_size >= 9) {
            lastBlockVal = 0;
            std::memcpy((unsigned char*)(&lastBlockVal), in_ptr + (chunk_size - 9), 5);
            if (lastBlockVal == 0xffff000001) last_block = true;
        }
        bool last_buffer = ((in_size == 0) && (last_block)) ? true : false;
        bool last_data = false;
        while (chunk_size > 0 || (last_buffer && !last_data)) {
            decompress_size += inflate_buffer(in_ptr, out + decompress_size, chunk_size, host_buffer_size, last_data,
                                              last_buffer, kname);
        }
        in_ptr += chunk_size1;
    }
    return decompress_size;
}

size_t xfZlib::decompress(uint8_t* in, uint8_t* out, size_t input_size, size_t max_outbuf_size, int cu) {
    if (m_zlibFlow) {
        // Call decompresss buffer
        size_t decSize = decompress_buffer(in, out, input_size, max_outbuf_size, cu);
        return decSize;
    } else {
        cl_int err;
        uint8_t hidx = 0;
        if (in[hidx++] == 0x1F && in[hidx++] == 0x8B) {
            // Check for magic header
            // Check if method is deflate or not
            if (in[hidx++] != 0x08) {
                std::cerr << "\n";
                std::cerr << "Deflate Header Check Fails" << std::endl;
                release();
                m_err_code = c_headermismatch;
                return 0;
            }

            // Check if the FLAG has correct value
            // Supported file name or no file name
            // 0x00: No File Name
            // 0x08: File Name
            if (in[hidx] != 0 && in[hidx] != 0x08) {
                std::cerr << "\n";
                std::cerr << "Deflate -n option check failed" << std::endl;
                release();
                m_err_code = c_headermismatch;
                return 0;
            }
            hidx++;

            // Skip time stamp bytes
            // time stamp contains 4 bytes
            hidx += 4;

            // One extra 0  ending byte
            hidx += 1;

            // Check the operating system code
            // for Unix its 3
            uint8_t oscode_in = in[hidx];
            std::vector<uint8_t> oscodes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
            bool ochck = std::find(oscodes.cbegin(), oscodes.cend(), oscode_in) == oscodes.cend();
            if (ochck) {
                std::cerr << "\n";
                std::cerr << "GZip header mismatch: OS code is unknown" << std::endl;
                return 0;
            }
        } else {
            hidx = 0;
            // ZLIB Header Checks
            // CMF
            // FLG
            uint8_t cmf = 0x78;
            // 0x01: Fast Mode
            // 0x5E: 1 to 5 levels
            // 0x9C: Default compression: level 6
            // 0xDA: High compression
            std::vector<uint8_t> zlib_flags{0x01, 0x5E, 0x9C, 0xDA};
            if (in[hidx++] == cmf) {
                uint8_t flg = in[hidx];
                bool hchck = std::find(zlib_flags.cbegin(), zlib_flags.cend(), flg) == zlib_flags.cend();
                if (hchck) {
                    std::cerr << "\n";
                    std::cerr << "Header check fails" << std::endl;
                    release();
                    m_err_code = c_headermismatch;
                    return 0;
                }
            } else {
                std::cerr << "\n";
                std::cerr << "Zlib Header mismatch" << std::endl;
                release();
                m_err_code = c_headermismatch;
                return 0;
            }
        }

#if (VERBOSE_LEVEL >= 1)
        std::cout << "CU" << cu << " ";
#endif
        // Streaming based solution
        uint32_t inBufferSize = INPUT_BUFFER_SIZE;
        uint32_t outBufferSize = OUTPUT_BUFFER_SIZE;

        for (int i = 0; i < DOUT_BUFFERCOUNT; i++) {
            ZOCL_CHECK_2(
                err, buffer_dec_zlib_output[i] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                                outBufferSize, h_dbufstream_zlibout[i].data(), &err),
                m_err_code, c_clOutOfHostMemory, c_clOutOfResource);
            if (error_code()) {
                release_dec_buffers();
                return 0;
            }
        }

        // Streaming based solution
        // Decompress bank index range [0,21]
        // Input Device Buffer allocation (__enqueue_writes)
        for (int i = 0; i < DIN_BUFFERCOUNT; i++) {
            ZOCL_CHECK_2(err, buffer_dec_input[i] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                                   inBufferSize, h_dbufstream_in[i].data(), &err),
                         m_err_code, c_clOutOfHostMemory, c_clOutOfResource);
            if (error_code()) {
                release_dec_buffers();
                return 0;
            }
        }
        // if input_size if greater than 2 MB, then buffer size must be 2MB
        if (input_size < inBufferSize) inBufferSize = input_size;
        if ((max_outbuf_size < outBufferSize) && (max_outbuf_size != 0)) outBufferSize = max_outbuf_size;

        uint8_t kidx = m_kidx;
        std::string cu_id = std::to_string((cu + 1));
        std::string decompress_kname =
            stream_decompress_kernel_name[kidx] + ":{" + stream_decompress_kernel_name[kidx] + "_" + cu_id + "}";
        OCL_CHECK(err, cl::Kernel decompress_kernel(*m_program, decompress_kname.c_str(), &err));

        // start parallel reader kernel enqueue thread
        uint32_t decmpSizeIdx = 0;
        std::thread decompWriter(&xfZlib::_enqueue_writes, this, inBufferSize, in, input_size, cu);
        std::thread decompReader(&xfZlib::_enqueue_reads, this, outBufferSize, out, &decmpSizeIdx, cu, max_outbuf_size);
#ifndef FREE_RUNNING_KERNEL
        ZOCL_CHECK_2(err, err = m_q_dec[cu]->enqueueTask(decompress_kernel), m_err_code, c_clOutOfHostMemory,
                     c_clOutOfResource);
#endif
        if (error_code()) {
            release_dec_buffers();
            return 0;
        }
#ifndef FREE_RUNNING_KERNEL
        OCL_CHECK(err, err = m_q_dec[cu]->finish());
#endif
        decompReader.join();
        decompWriter.join();

        release_dec_buffers();

        if (m_derr_code) {
            return 0;
        } else {
            return decmpSizeIdx;
        }
    }
}

size_t xfZlib::inflate_buffer(xlibData* strm,
                              int last_block,
                              size_t& input_size,
                              size_t& output_size,
                              bool& last_data,
                              bool last_buffer,
                              const std::string& cu_id) {
    cl_int err;
    bool actionDone = false;
    std::string compress_kname = cu_id;
    if (m_token == nullptr) {
        // Returns first token
        m_token = strtok((char*)(cu_id.c_str()), "_");
    }
    char* val;
    auto fixed_hbuf_size = HOST_BUFFER_SIZE;

    // Keep printing tokens while one of the
    // delimiters present in str[].
    while (m_token != NULL) {
        val = strtok(NULL, "_");
        if (val == NULL) break;
        m_token = val;
    }
    if (m_decompressKernel == NULL) {
        uint8_t kidx = m_kidx;
        std::string decompress_kname =
            stream_decompress_kernel_name[kidx] + ":{" + stream_decompress_kernel_name[kidx] + "_" + m_token + "}";
        OCL_CHECK(err, m_decompressKernel = new cl::Kernel(*m_program, decompress_kname.c_str(), &err));
        OCL_CHECK(err, err = m_dec_q->enqueueTask(*(m_decompressKernel), NULL, NULL));
    }

    if (m_dataReaderKernel == NULL) {
        std::string data_reader_kname = data_reader_kernel_name + ":{" + data_reader_kernel_name + "_" + m_token + "}";
        OCL_CHECK(err, m_dataReaderKernel = new cl::Kernel(*m_program, data_reader_kname.c_str(), &err));
    }

    if (m_dataWriterKernel == NULL) {
        std::string data_writer_kname = data_writer_kernel_name + ":{" + data_writer_kernel_name + "_" + m_token + "}";
        OCL_CHECK(err, m_dataWriterKernel = new cl::Kernel(*m_program, data_writer_kname.c_str(), &err));
        if (!isSlaveBridge()) {
            OCL_CHECK(err, err = m_wr_q->enqueueMigrateMemObjects({*(buffer_status)}, 0, NULL, NULL));
        } else {
            h_decStatus =
                (uint32_t*)m_memMgrDecS2MM->create_host_buffer(CL_MEM_READ_ONLY, sizeof(uint32_t), &(buffer_status));
            if (h_decStatus == nullptr) {
                release();
                m_err_code = true;
            } else {
                h_decStatus[0] = 0;
            }
        }
    }

    if (input_size) {
        if (write_buffer == nullptr) {
            auto buffer = m_memMgrDecMM2S->createBuffer(fixed_hbuf_size); // buffer
            write_buffer = buffer;
        }

        if (write_buffer) {
            // Do accumulation
            // Do copy of data from m_info->in_buf to write_buffer->h_buf_in
            // set write_buffer to null if you reach 1MB
            if (input_size + this->m_inputSize > this->m_bufSize) {
                std::memcpy(write_buffer->h_buf_in + this->m_inputSize, strm->in_buf,
                            this->m_bufSize - this->m_inputSize);
                strm->in_buf += (this->m_bufSize - this->m_inputSize);
                input_size -= (this->m_bufSize - this->m_inputSize);
                this->m_inputSize += (this->m_bufSize - this->m_inputSize);
            } else if (input_size != 0) {
                std::memcpy(write_buffer->h_buf_in + this->m_inputSize, strm->in_buf, input_size);
                this->m_inputSize += input_size;
                input_size = 0;
            }

            if (last_block && input_size == 0) last_buffer = true;

            if ((this->m_inputSize == this->m_bufSize) || last_buffer) {
                m_wbready = true;
            }
            actionDone = true;
        }

        if (m_wbready) {
            m_wbready = false;
            auto buffer = write_buffer;
            write_buffer = nullptr;
            uint32_t cBufSize = this->m_inputSize;
            uint32_t isLast = last_buffer;
            this->m_inputSize -= (this->m_inputSize > output_size) ? output_size : this->m_inputSize;

            // set kernel arguments
            OCL_CHECK(err, err = m_dataWriterKernel->setArg(0, *(buffer->buffer_input)));
            OCL_CHECK(err, err = m_dataWriterKernel->setArg(1, cBufSize));
            OCL_CHECK(err, err = m_dataWriterKernel->setArg(2, isLast));

            // enqueue data migration to kernel
            if (!(isSlaveBridge())) {
                OCL_CHECK(err, err = m_wr_q->enqueueMigrateMemObjects({*(buffer->buffer_input)}, 0, NULL, NULL));
            }

            // enqueue the writer kernel dependent on corresponding bufffer migration
            OCL_CHECK(err, err = m_wr_q->enqueueTask(*(m_dataWriterKernel), NULL, &(buffer->cmp_event)));

            cl_int err;
            OCL_CHECK(err, err = buffer->cmp_event.setCallback(CL_COMPLETE, xf::compression::event_decompressWrite_cb,
                                                               (void*)buffer));
            actionDone = true;
        }
    }

    if (!m_decompressStatus) {
        auto buffer_1 = m_memMgrDecS2MM->createBuffer(output_size); // buffer
        if (buffer_1 != NULL) {
            uint32_t inputSize = output_size;
            // set consistent buffer size to be read
            OCL_CHECK(err, err = m_dataReaderKernel->setArg(0, *(buffer_1->buffer_zlib_output)));
            OCL_CHECK(err, err = m_dataReaderKernel->setArg(1, *(buffer_1->buffer_compress_size)));
            OCL_CHECK(err, err = m_dataReaderKernel->setArg(2, *(buffer_status)));
            OCL_CHECK(err, err = m_dataReaderKernel->setArg(3, inputSize));

            // enqueue reader kernel
            OCL_CHECK(err, err = m_rd_q->enqueueTask(*(m_dataReaderKernel), NULL, &(buffer_1->cmp_event)));

            if (!(isSlaveBridge())) {
                std::vector<cl::Event> cmpEvents = {buffer_1->cmp_event};
                OCL_CHECK(err,
                          err = m_rd_q->enqueueMigrateMemObjects(
                              {*(buffer_1->buffer_zlib_output), *(buffer_1->buffer_compress_size), *(buffer_status)},
                              CL_MIGRATE_MEM_OBJECT_HOST, &(cmpEvents), &(buffer_1->rd_event)));
                cl_int err;
                OCL_CHECK(err, err = buffer_1->rd_event.setCallback(CL_COMPLETE, xf::compression::event_decompress_cb,
                                                                    (void*)buffer_1));
            } else {
                cl_int err;
                OCL_CHECK(err, err = buffer_1->cmp_event.setCallback(CL_COMPLETE, xf::compression::event_decompress_cb,
                                                                     (void*)buffer_1));
            }
        }
    }

    if ((m_memMgrDecMM2S->peekBuffer() != NULL) && (m_memMgrDecMM2S->peekBuffer()->is_finish())) {
        m_memMgrDecMM2S->getBuffer();
    }

    if (m_rbready) {
        size_t retSize = 0;
        // Do the data copy chunk by chunk
        // Set readbuffer to null once full copy is done
        // Transfer data in sizes of avail out until it turns zero
        if (m_compSize > strm->output_size) {
            std::memcpy(strm->out_buf, read_buffer->h_buf_zlibout + m_compSizePending, strm->output_size);
            m_compSize -= strm->output_size;
            retSize = strm->output_size;
            strm->total_out += strm->output_size;
            strm->out_buf += strm->output_size;
            m_compSizePending += strm->output_size;
            strm->output_size = 0;
        } else {
            std::memcpy(strm->out_buf, read_buffer->h_buf_zlibout + m_compSizePending, m_compSize);
            strm->output_size = strm->output_size - m_compSize;
            strm->total_out += m_compSize;
            strm->out_buf += m_compSize;
            retSize = m_compSize;
            m_compSizePending = strm->output_size;
            m_compSize = 0;
        }
        if (m_compSize == 0) {
            read_buffer = nullptr;
            m_rbready = false;
            m_compSizePending = 0;
        }
        last_data = (m_decompressStatus) ? true : false;
        return retSize;
    }

    if ((m_memMgrDecS2MM->peekBuffer() != NULL) && (m_memMgrDecS2MM->peekBuffer()->is_finish()) &&
        (m_rbready == false)) {
        buffers* buffer;
        buffer = m_memMgrDecS2MM->getBuffer();

        m_decompressStatus = (isSlaveBridge()) ? h_decStatus[0] : h_dcompressStatus[0];
        m_compSize = buffer->h_compressSize[0];
        read_buffer = buffer;
        m_rbready = true;
        assert(output_size >= m_compSize);
        actionDone = true;
    }

    if ((actionDone == false) && (m_memMgrDecS2MM->isPending() == true)) {
        // wait for a queue to finish to save CPU cycles
        auto buffer = m_memMgrDecS2MM->peekBuffer();
        if (buffer != NULL) {
            if (!isSlaveBridge())
                buffer->rd_event.wait();
            else
                buffer->cmp_event.wait();
        }
        actionDone = true;
    }

    last_data = false;
    return 0;
}

size_t xfZlib::inflate_buffer(uint8_t* in,
                              uint8_t* out,
                              size_t& input_size,
                              size_t& output_size,
                              bool& last_data,
                              bool last_buffer,
                              const std::string& cu_id) {
    cl_int err;
    bool actionDone = false;
    std::string compress_kname = cu_id;
    // Returns first token
    char* token = strtok((char*)(cu_id.c_str()), "_");
    char* val;

    // Keep printing tokens while one of the
    // delimiters present in str[].
    while (token != NULL) {
        val = strtok(NULL, "_");
        if (val == NULL) break;
        token = val;
    }
    if (m_decompressKernel == NULL) {
        uint8_t kidx = m_kidx;
        std::string decompress_kname =
            stream_decompress_kernel_name[kidx] + ":{" + stream_decompress_kernel_name[kidx] + "_" + token + "}";
        OCL_CHECK(err, m_decompressKernel = new cl::Kernel(*m_program, decompress_kname.c_str(), &err));
        OCL_CHECK(err, err = m_dec_q->enqueueTask(*(m_decompressKernel), NULL, NULL));

        // Reader kernel
        std::string data_reader_kname = data_reader_kernel_name + ":{" + data_reader_kernel_name + "_" + token + "}";
        OCL_CHECK(err, m_dataReaderKernel = new cl::Kernel(*m_program, data_reader_kname.c_str(), &err));

        // Writer kernel
        std::string data_writer_kname = data_writer_kernel_name + ":{" + data_writer_kernel_name + "_" + token + "}";
        OCL_CHECK(err, m_dataWriterKernel = new cl::Kernel(*m_program, data_writer_kname.c_str(), &err));
        if (!isSlaveBridge()) {
            OCL_CHECK(err, err = m_wr_q->enqueueMigrateMemObjects({*(buffer_status)}, 0, NULL, NULL));
        } else {
            h_decStatus =
                (uint32_t*)m_memMgrDecS2MM->create_host_buffer(CL_MEM_READ_ONLY, sizeof(uint32_t), &(buffer_status));
            if (h_decStatus == nullptr) {
                release();
                m_err_code = true;
            } else {
                h_decStatus[0] = 0;
            }
        }
    }

    m_lastData = (last_buffer) ? true : false;

    if (input_size) {
        uint32_t inputSize = output_size;
        auto buffer = m_memMgrDecMM2S->createBuffer(inputSize); // buffer
        if (buffer != NULL) {
            input_size -= (input_size > output_size) ? output_size : input_size;
            std::memcpy(buffer->h_buf_in, &in[0], inputSize);

            uint32_t cBufSize = inputSize;
            uint32_t isLast = last_buffer;

            // set kernel arguments
            OCL_CHECK(err, err = m_dataWriterKernel->setArg(0, *(buffer->buffer_input)));
            OCL_CHECK(err, err = m_dataWriterKernel->setArg(1, cBufSize));
            OCL_CHECK(err, err = m_dataWriterKernel->setArg(2, isLast));

            // enqueue data migration to kernel
            if (!(isSlaveBridge())) {
                OCL_CHECK(err, err = m_wr_q->enqueueMigrateMemObjects({*(buffer->buffer_input)}, 0, NULL, NULL));
            }

            // enqueue the writer kernel dependent on corresponding bufffer migration
            OCL_CHECK(err, err = m_wr_q->enqueueTask(*(m_dataWriterKernel), NULL, &(buffer->cmp_event)));

            cl_int err;
            OCL_CHECK(err, err = buffer->cmp_event.setCallback(CL_COMPLETE, xf::compression::event_decompressWrite_cb,
                                                               (void*)buffer));
            actionDone = true;
        }
    }

    auto buffer_1 = m_memMgrDecS2MM->createBuffer(output_size); // buffer
    if (buffer_1 != NULL) {
        uint32_t inputSize = output_size;
        // set consistent buffer size to be read
        OCL_CHECK(err, err = m_dataReaderKernel->setArg(0, *(buffer_1->buffer_zlib_output)));
        OCL_CHECK(err, err = m_dataReaderKernel->setArg(1, *(buffer_1->buffer_compress_size)));
        OCL_CHECK(err, err = m_dataReaderKernel->setArg(2, *(buffer_status)));
        OCL_CHECK(err, err = m_dataReaderKernel->setArg(3, inputSize));

        // enqueue reader kernel
        OCL_CHECK(err, err = m_rd_q->enqueueTask(*(m_dataReaderKernel), NULL, &(buffer_1->cmp_event)));

        if (!(isSlaveBridge())) {
            std::vector<cl::Event> cmpEvents = {buffer_1->cmp_event};
            OCL_CHECK(err, err = m_rd_q->enqueueMigrateMemObjects(
                               {*(buffer_1->buffer_zlib_output), *(buffer_1->buffer_compress_size), *(buffer_status)},
                               CL_MIGRATE_MEM_OBJECT_HOST, &(cmpEvents), &(buffer_1->rd_event)));
            cl_int err;
            OCL_CHECK(err, err = buffer_1->rd_event.setCallback(CL_COMPLETE, xf::compression::event_decompress_cb,
                                                                (void*)buffer_1));
        } else {
            cl_int err;
            OCL_CHECK(err, err = buffer_1->cmp_event.setCallback(CL_COMPLETE, xf::compression::event_decompress_cb,
                                                                 (void*)buffer_1));
        }
    }

    if ((m_memMgrDecMM2S->peekBuffer() != NULL) && (m_memMgrDecMM2S->peekBuffer()->is_finish())) {
        m_memMgrDecMM2S->getBuffer();
    }

    if ((m_memMgrDecS2MM->peekBuffer() != NULL) && (m_memMgrDecS2MM->peekBuffer()->is_finish())) {
        buffers* buffer;
        uint32_t compSize = 0;
        buffer = m_memMgrDecS2MM->getBuffer();

        m_decompressStatus = (isSlaveBridge()) ? h_decStatus[0] : h_dcompressStatus[0];
        compSize = buffer->h_compressSize[0];
        assert(output_size >= compSize);
        std::memcpy(out, buffer->h_buf_zlibout, compSize);
        last_data = (m_decompressStatus) ? true : false;
        actionDone = true;
        return compSize;
    }

    if ((actionDone == false) && (m_memMgrDecS2MM->isPending() == true)) {
        // wait for a queue to finish to save CPU cycles
        auto buffer = m_memMgrDecS2MM->peekBuffer();
        if (buffer != NULL) {
            if (!isSlaveBridge())
                buffer->rd_event.wait();
            else
                buffer->cmp_event.wait();
        }
        actionDone = true;
    }

    last_data = false;
    return 0;
}

inline void xfZlib::waitTaskEvents(void) {
    if ((m_actionDone == false) && (m_memMgrCmp->isPending() == true)) {
        // wait for a queue to finish to save CPU cycles
        auto buffer = m_memMgrCmp->peekBuffer();
        if (buffer != NULL) {
            if (this->isSlaveBridge())
                buffer->cmp_event.wait();
            else
                buffer->rd_event.wait();
        }
    }
}

inline void xfZlib::setKernelArgs(buffers* buffer, uint32_t inSize) {
    int narg = 0;
    // Set kernel arguments
    m_compressFullKernel->setArg(narg++, *(buffer->buffer_input));
    m_compressFullKernel->setArg(narg++, *(buffer->buffer_zlib_output));
    m_compressFullKernel->setArg(narg++, *(buffer->buffer_compress_size));
    m_compressFullKernel->setArg(narg++, *buffer_checksum_data);
    m_compressFullKernel->setArg(narg++, inSize);
    m_compressFullKernel->setArg(narg++, m_checksum_type);
}

inline void xfZlib::readDeviceBuffer(void) {
    // Check if compression is done if so initiate zlib_output device buffer
    // copy to host
    if ((m_memMgrCmp->peekBuffer() != NULL) && (m_memMgrCmp->peekBuffer()->is_finish())) {
        auto buffer = m_memMgrCmp->peekBuffer();
        if (buffer->copy_done == false) {
            buffer->compress_finish = false;
            uint32_t compSize = (buffer->h_compressSize)[0];

            // Read data based on the size
            m_def_q->enqueueReadBuffer(*(buffer->buffer_zlib_output), CL_FALSE, 0, compSize, &buffer->h_buf_zlibout[0],
                                       nullptr, &(buffer->rd_event));

            cl_int err;
            // Call back tracks rd_event associated with enqueueReadBuffer and
            // in case of completion it updates the compress and copy_data
            // status, which will be used to identify availability of zlib
            // output in host buffer and local buffer memcpy initiated later on
            OCL_CHECK(err, err = buffer->rd_event.setCallback(CL_COMPLETE, xf::compression::event_copydone_cb,
                                                              (void*)buffer));
            m_actionDone = true;
        }
    }
}

size_t xfZlib::deflate_buffer(
    xlibData* strm, int flush, size_t& input_size, bool& last_data, const std::string& cu_id) {
    bool initialize_checksum = false;
    bool actionDone = false;

    auto fixed_hbuf_size = HOST_BUFFER_SIZE;
    cl_int err;
    std::string compress_kname;
    if (m_compressFullKernel == NULL) {
        // Random: Use kernel name directly and let XRT
        // decide the CU connection based on HBM bank
        compress_kname = compress_kernel_names[1];
        OCL_CHECK(err, m_compressFullKernel = new cl::Kernel(*m_program, compress_kname.c_str(), &err));
        initialize_checksum = true;
    }

    bool checksum_type = false;
    if (m_zlibFlow) {
        if (initialize_checksum) h_buf_checksum_data.data()[0] = 1;
        checksum_type = false;
    } else {
        if (initialize_checksum) h_buf_checksum_data.data()[0] = ~0;
        checksum_type = true;
    }
    buffers* lastBuffer = m_memMgrCmp->getLastBuffer();

    if (input_size) {
        if (write_buffer == nullptr) {
            auto buffer = m_memMgrCmp->createBuffer(fixed_hbuf_size);
            write_buffer = buffer;
        }

        if (write_buffer) {
            // Do accumulation
            // Do copy of data from m_info->in_buf to write_buffer->h_buf_in
            // set write_buffer to null if you reach 1MB
            if (input_size + this->m_inputSize > this->m_bufSize) {
                std::memcpy(write_buffer->h_buf_in + this->m_inputSize, strm->in_buf,
                            this->m_bufSize - this->m_inputSize);
                strm->in_buf += (this->m_bufSize - this->m_inputSize);
                input_size -= (this->m_bufSize - this->m_inputSize);
                this->m_inputSize += (this->m_bufSize - this->m_inputSize);
            } else if (input_size != 0) {
                std::memcpy(write_buffer->h_buf_in + this->m_inputSize, strm->in_buf, input_size);
                this->m_inputSize += input_size;
                input_size = 0;
            }

            if ((this->m_inputSize == this->m_bufSize) || flush) {
                m_wbready = true;
            }
        }

        if (m_wbready) {
            m_wbready = false;
            auto buffer = write_buffer;
            write_buffer = nullptr;
            if (lastBuffer == buffer) lastBuffer = NULL;
            uint32_t inSize = this->m_inputSize;
            buffer->store_size = 0;
            this->m_inputSize = 0;

            int narg = 0;
            // Set kernel arguments
            m_compressFullKernel->setArg(narg++, *(buffer->buffer_input));
            m_compressFullKernel->setArg(narg++, *(buffer->buffer_zlib_output));
            m_compressFullKernel->setArg(narg++, *(buffer->buffer_compress_size));
            m_compressFullKernel->setArg(narg++, *buffer_checksum_data);
            m_compressFullKernel->setArg(narg++, inSize);
            m_compressFullKernel->setArg(narg++, checksum_type);

            // Migrate memory - Map host to device buffers
            OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer->buffer_input)}, 0 /* 0 means from host*/,
                                                                   NULL, &(buffer->wr_event)));
            std::vector<cl::Event> wrEvents = {buffer->wr_event};

            if (initialize_checksum) {
                OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects(
                                   {*(buffer_checksum_data)}, 0 /* 0 means from host*/, NULL, &(buffer->chk_wr_event)));
                wrEvents.push_back(buffer->chk_wr_event);
            }

            if (lastBuffer != NULL) {
                int status = -1;
                cl_int ret = lastBuffer->cmp_event.getInfo<int>(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
                if (ret == 0 && status != 0) {
                    wrEvents.push_back(lastBuffer->cmp_event);
                }
            }

            std::vector<cl::Event> cmpEvents;
            // kernel write events update
            // LZ77 Compress Fire Kernel invocation
            OCL_CHECK(err, err = m_def_q->enqueueTask({*(m_compressFullKernel)}, &(wrEvents), &(buffer->cmp_event)));

            cmpEvents = {buffer->cmp_event};

            OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer->buffer_compress_size)},
                                                                   CL_MIGRATE_MEM_OBJECT_HOST, &(cmpEvents),
                                                                   &(buffer->rd_event)));
            cl_int err;
            OCL_CHECK(err, err = buffer->rd_event.setCallback(CL_COMPLETE, xf::compression::event_compress_cb,
                                                              (void*)buffer));
            input_size = 0;
            actionDone = true;
        }
    }

    // Check if compression is done if so initiate output buffer transfer
    // from device to host
    if ((m_memMgrCmp->peekBuffer() != NULL) && (m_memMgrCmp->peekBuffer()->is_finish())) {
        auto buffer = m_memMgrCmp->peekBuffer();
        if (buffer->copy_done == false) {
            buffer->compress_finish = false;
            uint32_t compSize = (buffer->h_compressSize)[0];

            // Read data based on the size
            m_def_q->enqueueReadBuffer(*(buffer->buffer_zlib_output), CL_FALSE, 0, compSize, &buffer->h_buf_zlibout[0],
                                       nullptr, &(buffer->rd_event));

            cl_int err;
            // Call back tracks rd_event associated with enqueueReadBuffer and
            // in case of completion it updates the compress and copy_data
            // status, which will be used to identify availability of zlib
            // output in host buffer and local buffer memcpy initiated later on
            OCL_CHECK(err, err = buffer->rd_event.setCallback(CL_COMPLETE, xf::compression::event_copydone_cb,
                                                              (void*)buffer));
            actionDone = true;
        }
    }

    if (m_rbready) {
        size_t retSize = 0;
        // Do the data copy chunk by chunk
        // Set readbuffer to null once full copy is done
        // Transfer data in sizes of avail out until it turns zero
        if (m_compSize >= strm->output_size) {
            std::memcpy(strm->out_buf, read_buffer->h_buf_zlibout + m_compSizePending, strm->output_size);
            m_compSize -= strm->output_size;
            retSize = strm->output_size;
            strm->total_out += strm->output_size;
            strm->out_buf += strm->output_size;
            m_compSizePending += strm->output_size;
            strm->output_size = 0;
        } else {
            std::memcpy(strm->out_buf, read_buffer->h_buf_zlibout + m_compSizePending, m_compSize);
            strm->output_size = strm->output_size - m_compSize;
            strm->total_out += m_compSize;
            strm->out_buf += m_compSize;
            retSize = m_compSize;
            m_compSizePending = strm->output_size;
            m_compSize = 0;
        }
        if (m_compSize == 0) {
            read_buffer = nullptr;
            m_rbready = false;
            m_compSizePending = 0;
        }
        if ((m_memMgrCmp->isPending() == false) && (m_rbready == false)) {
            last_data = true;
        } else {
            last_data = false;
        }

        if (last_data) {
            OCL_CHECK(err,
                      err = m_def_q->enqueueMigrateMemObjects({*(buffer_checksum_data)}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, err = m_def_q->finish());

            m_checksum = h_buf_checksum_data.data()[0];
            if (checksum_type) m_checksum = ~m_checksum;

            long int lBlk = 0xffff000001;
            std::memcpy(strm->out_buf, &lBlk, 5);
            strm->output_size -= 5;
            strm->total_out += 5;
            strm->out_buf += 5;
            retSize += 5;

            strm->out_buf[0] = m_checksum >> 24;
            strm->out_buf[1] = m_checksum >> 16;
            strm->out_buf[2] = m_checksum >> 8;
            strm->out_buf[3] = m_checksum;

            strm->output_size -= 4;
            strm->total_out += 4;
            strm->out_buf += 4;
            retSize += 4;
        }

        return retSize;
    }

    if ((m_memMgrCmp->peekBuffer() != NULL) && (m_memMgrCmp->peekBuffer()->is_copyfinish()) && (m_rbready == false)) {
        buffers* buffer;
        buffer = m_memMgrCmp->getBuffer();
        if (buffer != NULL) {
            actionDone = true;
            m_compSize = (buffer->h_compressSize)[0];
            // Set read buffer so that
            // read buffer will ensure compressed data
            // goes out in chunks
            read_buffer = buffer;
            m_rbready = true;
            // handle checksum for next iteration
            if (checksum_type) h_buf_checksum_data.data()[0] = ~h_buf_checksum_data.data()[0];
        }
    }

    if ((actionDone == false) && (m_memMgrCmp->isPending() == true)) {
        // wait for a queue to finish to save CPU cycles
        auto buffer = m_memMgrCmp->peekBuffer();
        if (buffer != NULL) {
            buffer->rd_event.wait();
        }
    }

    if ((m_memMgrCmp->isPending() == false) && (m_rbready == false)) {
        last_data = true;
    } else {
        last_data = false;
    }
    return 0;
}

size_t xfZlib::deflate_buffer(
    uint8_t* in, uint8_t* out, size_t& input_size, bool& last_data, bool last_buffer, const std::string& cu_id) {
    cl_int err;
    bool initialize_checksum = false;
    std::string compress_kname;

    if (m_compressFullKernel == NULL) {
        if (!m_islibz) {
            compress_kname = compress_kernel_names[1] + ":{" + cu_id + "}";
        } else {
            compress_kname = compress_kernel_names[1];
        }
        OCL_CHECK(err, m_compressFullKernel = new cl::Kernel(*m_program, compress_kname.c_str(), &err));
        initialize_checksum = true;
    }

    if (m_zlibFlow) {
        if (this->isSlaveBridge()) {
            if (initialize_checksum) h_buf_checksum[0] = 1;
        } else {
            if (initialize_checksum) h_buf_checksum_data.data()[0] = 1;
        }
        m_checksum_type = false;
    } else {
        if (this->isSlaveBridge()) {
            if (initialize_checksum) h_buf_checksum[0] = ~0;
        } else {
            if (initialize_checksum) h_buf_checksum_data.data()[0] = ~0;
        }
        m_checksum_type = true;
    }

    m_lastData = (last_buffer) ? true : false;

    buffers* lastBuffer = m_memMgrCmp->getLastBuffer();
    if (input_size) {
        auto buffer = m_memMgrCmp->createBuffer(input_size);

        if (buffer != NULL) {
            if (lastBuffer == buffer) lastBuffer = NULL;

            std::memcpy(buffer->h_buf_in, &in[0], input_size);

            uint32_t inSize = input_size;
            buffer->store_size = 0;

            // Set kernel arguments
            setKernelArgs(buffer, inSize);

            std::vector<cl::Event> wrEvents;
            if (!(this->isSlaveBridge())) {
                // Migrate memory - Map host to device buffers
                OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects(
                                   {*(buffer->buffer_input)}, 0 /* 0 means from host*/, NULL, &(buffer->wr_event)));

                wrEvents.push_back(buffer->wr_event);
                if (initialize_checksum) {
                    OCL_CHECK(err,
                              err = m_def_q->enqueueMigrateMemObjects(
                                  {*(buffer_checksum_data)}, 0 /* 0 means from host*/, NULL, &(buffer->chk_wr_event)));
                    wrEvents.push_back(buffer->chk_wr_event);
                }
            }

            if (lastBuffer != NULL) {
                int status = -1;
                cl_int ret = lastBuffer->cmp_event.getInfo<int>(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
                if (ret == 0 && status != 0) {
                    wrEvents.push_back(lastBuffer->cmp_event);
                }
            }

            std::vector<cl::Event> cmpEvents;
            OCL_CHECK(err, err = m_def_q->enqueueTask({*(m_compressFullKernel)}, &(wrEvents), &(buffer->cmp_event)));

            cmpEvents = {buffer->cmp_event};

            if (!(this->isSlaveBridge())) {
                OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer->buffer_compress_size)},
                                                                       CL_MIGRATE_MEM_OBJECT_HOST, &(cmpEvents),
                                                                       &(buffer->rd_event)));
                cl_int err;
                OCL_CHECK(err, err = buffer->rd_event.setCallback(CL_COMPLETE, xf::compression::event_compress_cb,
                                                                  (void*)buffer));
            } else {
                cl_int err;
                // Set call back on compress kernel event
                OCL_CHECK(err, err = buffer->cmp_event.setCallback(CL_COMPLETE, xf::compression::event_compress_cb,
                                                                   (void*)buffer));
            }

            input_size = 0;
            m_actionDone = true;
        }
    }

    if (!this->isSlaveBridge()) {
        // In case of non slave bridge design
        // read the data from banks using compresson size (exact read)
        readDeviceBuffer();
    }

    bool taskStatus = false;
    if (m_memMgrCmp->peekBuffer() != NULL) {
        if (!(this->isSlaveBridge()))
            taskStatus = (m_memMgrCmp->peekBuffer()->is_copyfinish());
        else
            taskStatus = (m_memMgrCmp->peekBuffer()->is_finish());
    }

    if ((m_memMgrCmp->peekBuffer() != NULL) && (taskStatus)) {
        buffers* buffer;
        uint32_t compSize = 0;
        auto size = 0;
        buffer = m_memMgrCmp->getBuffer();
        if (buffer != NULL) {
            m_actionDone = true;

            compSize = (buffer->h_compressSize)[0];
            std::memcpy(out, &buffer->h_buf_zlibout[0], compSize);
            size += compSize;

            if (!(this->isSlaveBridge())) {
                // handle checksum for next iteration
                if (m_checksum_type) h_buf_checksum_data.data()[0] = ~h_buf_checksum_data.data()[0];
            } else {
                // handle checksum for next iteration
                if (m_checksum_type) h_buf_checksum[0] = ~h_buf_checksum[0];
            }
        }

        if (m_lastData && (input_size == 0) && (m_memMgrCmp->isPending() == false))
            last_data = true;
        else
            last_data = false;

        if (last_data) {
            if (!(this->isSlaveBridge())) {
                OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer_checksum_data)},
                                                                       CL_MIGRATE_MEM_OBJECT_HOST));
                OCL_CHECK(err, err = m_def_q->finish());

                m_checksum = h_buf_checksum_data.data()[0];
            } else {
                m_checksum = h_buf_checksum[0];
            }
            if (m_checksum_type) m_checksum = ~m_checksum;
        }

        return size;
    }

    // If there is no action done this call
    // then wait for previous calls results
    waitTaskEvents();

    if (m_lastData && (input_size == 0) && (m_memMgrCmp->isPending() == false))
        last_data = true;
    else
        last_data = false;
    return 0;
}

// This version of compression does overlapped execution between
// Kernel and Host. I/O operations between Host and Device are
// overlapped with Kernel execution between multiple compute units

size_t xfZlib::compress_buffer(uint8_t* in, uint8_t* out, size_t input_size, uint32_t host_buffer_size, int cu) {
    // Create kernel
    std::string cu_id = std::to_string((cu + 1));
    std::string compress_kname = compress_kernel_names[1] + "_" + cu_id;

    uint64_t compressed_size = 0;
    uint64_t in_size = input_size;
    auto in_ptr = in;
    while (in_size > 0) {
        size_t chunk_size = (in_size > host_buffer_size) ? host_buffer_size : in_size;
        size_t chunk_size1 = chunk_size;
        in_size -= chunk_size;
        bool last_buffer = (in_size == 0) ? true : false;
        bool last_data = false;
        while (chunk_size > 0 || (last_buffer && !last_data)) {
            compressed_size +=
                deflate_buffer(in_ptr, out + compressed_size, chunk_size, last_data, last_buffer, compress_kname);
        }
        in_ptr += chunk_size1;
    }

    long int last_block = 0xffff000001;
    std::memcpy(&(out[compressed_size]), &last_block, 5);
    compressed_size += 5;
    return compressed_size;
}

// Add zlib header

size_t xfZlib::add_header(uint8_t* out, int level, int strategy, int window_bits) {
    size_t outIdx = 0;
    if (m_zlibFlow) {
        // Compression method
        uint8_t CM = DEFLATE_METHOD;

        // Compression Window information
        uint8_t CINFO = window_bits - 8;

        // Create CMF header
        uint16_t header = (CINFO << 4);
        header |= CM;
        header <<= 8;

        if (level < 2 || strategy > 2)
            level = 0;
        else if (level < 6)
            level = 1;
        else if (level == 6)
            level = 2;
        else
            level = 3;

        // CreatE FLG header based on level
        // Strategy information
        header |= (level << 6);

        // Ensure that Header (CMF + FLG) is
        // divisible by 31
        header += 31 - (header % 31);

        out[outIdx] = (uint8_t)(header >> 8);
        out[outIdx + 1] = (uint8_t)(header);
        outIdx += 2;
    } else {
        const uint16_t c_format_0 = 31;
        const uint16_t c_format_1 = 139;
        const uint16_t c_variant = 8;
        const uint16_t c_real_code = 8;
        const uint16_t c_opcode = 3;

        // 2 bytes of magic header
        out[outIdx++] = c_format_0;
        out[outIdx++] = c_format_1;

        // 1 byte Compression method
        out[outIdx++] = c_variant;

        // 1 byte flags
        uint8_t flags = 0;
        flags |= c_real_code;
        out[outIdx++] = flags;

        // 4 bytes file modification time in unit format
        unsigned long time_stamp = 0;
        struct stat istat;
        stat(m_infile.c_str(), &istat);
        time_stamp = istat.st_mtime;
        out[outIdx++] = time_stamp;
        out[outIdx++] = time_stamp >> 8;
        out[outIdx++] = time_stamp >> 16;
        out[outIdx++] = time_stamp >> 24;

        // 1 byte extra flag (depend on compression method)
        uint8_t deflate_flags = 0;
        out[outIdx++] = deflate_flags;

        // 1 byte OPCODE - 0x03 for Unix
        out[outIdx++] = c_opcode;

        // Dump file name
        for (int i = 0; m_infile[i] != '\0'; i++) {
            out[outIdx++] = m_infile[i];
        }
        out[outIdx++] = 0;
    }
    return outIdx;
}

std::string xfZlib::getCompressKernel(int idx) {
    return compress_kernel_names[idx];
}

std::string xfZlib::getDeCompressKernel(int idx) {
    return stream_decompress_kernel_name[idx];
}

// Add zlib footer

size_t xfZlib::add_footer(uint8_t* out, size_t compressSize) {
    size_t outIdx = compressSize;
    uint32_t checksum = 0;

    outIdx = compressSize;
    checksum = m_checksum;

    if (m_zlibFlow) {
        out[outIdx++] = checksum >> 24;
        out[outIdx++] = checksum >> 16;
        out[outIdx++] = checksum >> 8;
        out[outIdx++] = checksum;
    } else {
        struct stat istat;
        stat(m_infile.c_str(), &istat);
        unsigned long ifile_size = istat.st_size;
        out[outIdx++] = checksum;
        out[outIdx++] = checksum >> 8;
        out[outIdx++] = checksum >> 16;
        out[outIdx++] = checksum >> 24;

        out[outIdx++] = ifile_size;
        out[outIdx++] = ifile_size >> 8;
        out[outIdx++] = ifile_size >> 16;
        out[outIdx++] = ifile_size >> 24;

        out[outIdx++] = 0;
        out[outIdx++] = 0;
        out[outIdx++] = 0;
        out[outIdx++] = 0;
        out[outIdx++] = 0;
    }

    return outIdx;
}

// This version of compression does overlapped execution between
// Kernel and Host. I/O operations between Host and Device are
// overlapped with Kernel execution between multiple compute units

size_t xfZlib::compress(
    uint8_t* in, uint8_t* out, size_t input_size, int cu, int level, int strategy, int window_bits) {
    size_t total_size = 0;
    size_t host_buffer_size = HOST_BUFFER_SIZE;
    // Add header
    total_size += add_header(out, level, strategy, window_bits);

    // Call compress buffer
    total_size += compress_buffer(in, &out[total_size], input_size, host_buffer_size, cu);

    // Add footer
    total_size = add_footer(out, total_size);

    return total_size;
} // Overlap end

memoryManager::memoryManager(
    uint8_t max_buffer, cl::Context* context, bool sb_opt, cl::CommandQueue* cqueue, const int bank_id) {
    maxBufCount = max_buffer;
    mContext = context;
    mCqueue = cqueue;
    bufCount = 0;
    m_bankId = bank_id;
    m_isSlaveBridge = sb_opt;
}

memoryManager::~memoryManager() {
    release();
}

uint8_t* memoryManager::create_host_buffer(cl_mem_flags flag, size_t size, cl::Buffer** devPtr) {
    cl_int err;
    cl_mem_ext_ptr_t host_buf_inext;
    host_buf_inext.flags = XCL_MEM_EXT_HOST_ONLY;
    host_buf_inext.obj = nullptr;
    host_buf_inext.param = 0;
    cl_mem_flags l_flag = (uint32_t)(flag | CL_MEM_EXT_PTR_XILINX);
    cl::Buffer* oclBuffer = nullptr;
    OCL_CHECK(err, oclBuffer = new cl::Buffer(*(this->mContext), l_flag, size, &host_buf_inext, &err));
    if (oclBuffer == nullptr) {
        return nullptr;
    }
    uint8_t* host_buffer = nullptr;
    cl_map_flags mflag;

    if (flag == CL_MEM_READ_ONLY)
        mflag = CL_MAP_READ;
    else if (flag == CL_MEM_READ_WRITE)
        mflag = CL_MAP_READ | CL_MAP_WRITE;
    else if (flag == CL_MEM_WRITE_ONLY)
        mflag = CL_MAP_WRITE;

    // Map host and device buffers
    OCL_CHECK(err, host_buffer =
                       (uint8_t*)mCqueue->enqueueMapBuffer(*(oclBuffer), CL_TRUE, mflag, 0, size, NULL, NULL, &err));

    *devPtr = oclBuffer;
    return host_buffer;
}

buffers* memoryManager::createBuffer(size_t size) {
    if (!(this->freeBuffers.empty())) {
        if (this->freeBuffers.front()->allocated_size >= size) {
            auto buffer = this->freeBuffers.front();
            this->busyBuffers.push(buffer);
            this->freeBuffers.pop();
            lastBuffer = buffer;
            buffer->reset();
            buffer->input_size = size;
            return buffer;
        } else {
            auto buffer = this->freeBuffers.front();
            aligned_allocator<uint8_t> alocator;
            aligned_allocator<uint32_t> alocator_1;

            alocator.deallocate(buffer->h_buf_in);
            alocator.deallocate(buffer->h_buf_zlibout);
            alocator_1.deallocate(buffer->h_compressSize);

            DELETE_OBJ(buffer->buffer_compress_size);
            DELETE_OBJ(buffer->buffer_input);
            DELETE_OBJ(buffer->buffer_zlib_output);

            this->freeBuffers.pop();
            DELETE_OBJ(buffer);
            bufCount -= 1;
        }
    }

    if (bufCount < maxBufCount) {
        auto buffer = new buffers();
        uint16_t max_blocks = (size / BLOCK_SIZE_IN_KB) + 1;

        if (m_isSlaveBridge) {
            // Device buffer allocation done by map buffers and
            // mapping is done this is when host buffers for each buffer pointer
            // gets actual memory pointer, in order to do this getBuffer is updated.
            buffer->h_buf_in = (uint8_t*)create_host_buffer(CL_MEM_READ_ONLY, size, &(buffer->buffer_input));
            if (buffer->h_buf_in == nullptr) {
                this->release(buffer);
                return nullptr;
            }

            buffer->h_buf_zlibout =
                (uint8_t*)create_host_buffer(CL_MEM_READ_WRITE, size, &(buffer->buffer_zlib_output));
            if (buffer->h_buf_zlibout == nullptr) {
                this->release(buffer);
                return nullptr;
            }

            buffer->h_compressSize = (uint32_t*)create_host_buffer(CL_MEM_READ_WRITE, max_blocks * sizeof(uint32_t),
                                                                   &(buffer->buffer_compress_size));
            if (buffer->h_compressSize == nullptr) {
                this->release(buffer);
                return nullptr;
            }
        } else {
            aligned_allocator<uint8_t> alocator;
            aligned_allocator<uint32_t> alocator_1;
            MEM_ALLOC_CHECK((buffer->h_buf_in = alocator.allocate(size)), size, "Input Host Buffer");
            MEM_ALLOC_CHECK((buffer->h_buf_zlibout = alocator.allocate(size)), size, "Output Host Buffer");
            MEM_ALLOC_CHECK((buffer->h_compressSize = alocator_1.allocate(max_blocks)), max_blocks,
                            "CompressSize Host Buffer");

            buffer->buffer_input = this->getBuffer(CL_MEM_READ_ONLY, size, (uint8_t*)(buffer->h_buf_in));
            if (buffer->buffer_input == NULL) {
                this->release(buffer);
                return NULL;
            }

            buffer->buffer_compress_size =
                this->getBuffer(CL_MEM_READ_WRITE, max_blocks * sizeof(uint32_t), (uint8_t*)(buffer->h_compressSize));
            if (buffer->buffer_compress_size == NULL) {
                this->release(buffer);
                return NULL;
            }

            buffer->buffer_zlib_output = this->getBuffer(CL_MEM_READ_WRITE, size, (uint8_t*)(buffer->h_buf_zlibout));
            if (buffer->buffer_zlib_output == NULL) {
                this->release(buffer);
                return NULL;
            }
        }

        buffer->input_size = size;
        buffer->allocated_size = size;
        this->busyBuffers.push(buffer);
        bufCount++;
        lastBuffer = buffer;
        return buffer;
    }
    return NULL;
}

cl::Buffer* memoryManager::getBuffer(cl_mem_flags flag, size_t size, uint8_t* host_ptr, uint8_t mode) {
    cl_int err;
    cl::Buffer* clBuffer = NULL;
#ifndef USE_HBM
    OCL_CHECK(err, clBuffer = new cl::Buffer(*(this->mContext), CL_MEM_USE_HOST_PTR | flag, size, host_ptr, &err));
    return clBuffer;
#else
    if (is_sw_emulation()) { // Software Emulation doesnt work with random bank
        OCL_CHECK(err, clBuffer = new cl::Buffer(*(this->mContext), CL_MEM_USE_HOST_PTR | flag, size, host_ptr, &err));
        return clBuffer;
    } else { // HW_EMU and HW flows works with random HBM selection
        cl_mem_ext_ptr_t ext;
        ext.obj = host_ptr;
        ext.param = 0;
        cl_mem_flags l_flag = (uint32_t)(flag | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX);
        bool use_xrm = false;

#ifdef ENABLE_INFLATE_XRM
        if (mode != 0) {
            use_xrm = true;
        }
#endif
        if (use_xrm) {
            auto bank = m_bankId | XCL_MEM_TOPOLOGY;
            ext.flags = bank;
            clBuffer = new cl::Buffer(*(this->mContext), l_flag, size, &ext, &err);
        } else {
            std::random_device randHBM;
            std::uniform_int_distribution<> range(0, 31);
            uint32_t randbank = range(randHBM);
            auto bank = randbank | XCL_MEM_TOPOLOGY;
            ext.flags = bank;
            uint8_t count = 0;
            do {
                clBuffer = new cl::Buffer(*(this->mContext), l_flag, size, &ext, &err);
                count++;
            } while (err != 0 && count < 31);
        }
    }
#endif
    return clBuffer;
}

buffers* memoryManager::getBuffer() {
    if (this->busyBuffers.empty()) return NULL;
    auto buffer = this->busyBuffers.front();
    this->freeBuffers.push(buffer);
    this->busyBuffers.pop();
    return buffer;
}

buffers* memoryManager::peekBuffer() {
    if (this->busyBuffers.empty()) return NULL;
    auto buffer = this->busyBuffers.front();
    return buffer;
}

buffers* memoryManager::getLastBuffer() {
    return lastBuffer;
}

void memoryManager::release() {
    while (!(this->freeBuffers.empty())) {
        auto buffer = this->freeBuffers.front();

        if (!m_isSlaveBridge) {
            aligned_allocator<uint8_t> alocator;
            aligned_allocator<uint32_t> alocator_1;

            alocator.deallocate(buffer->h_buf_in);
            alocator.deallocate(buffer->h_buf_zlibout);
            alocator_1.deallocate(buffer->h_compressSize);
        }

        DELETE_OBJ(buffer->buffer_compress_size);
        DELETE_OBJ(buffer->buffer_input);
        DELETE_OBJ(buffer->buffer_zlib_output);

        this->freeBuffers.pop();
        DELETE_OBJ(buffer);
    }
}

void memoryManager::release(buffers* buffer) {
    aligned_allocator<uint8_t> alocator;
    aligned_allocator<uint32_t> alocator_1;

    alocator.deallocate(buffer->h_buf_in);
    alocator.deallocate(buffer->h_buf_zlibout);
    alocator_1.deallocate(buffer->h_compressSize);

    DELETE_OBJ(buffer->buffer_compress_size);
    DELETE_OBJ(buffer->buffer_input);
    DELETE_OBJ(buffer->buffer_zlib_output);

    DELETE_OBJ(buffer);
}

bool memoryManager::isPending() {
    return (this->busyBuffers.empty() == false);
}
