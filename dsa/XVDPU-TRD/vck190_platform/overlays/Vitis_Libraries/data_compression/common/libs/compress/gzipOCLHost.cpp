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
#include "gzipOCLHost.hpp"
extern unsigned long crc32(unsigned long crc, const unsigned char* buf, uint32_t len);
extern unsigned long adler32(unsigned long crc, const unsigned char* buf, uint32_t len);

void event_compress_cb(cl_event event, cl_int cmd_status, void* data) {
    // callBackMutex.lock();
    assert(data != NULL);
    ((buffers*)(data))->compress_finish = true;
    // callBackMutex.unlock();
}

void event_checksum_cb(cl_event event, cl_int cmd_status, void* data) {
    // callBackMutex.lock();
    assert(data != NULL);
    ((buffers*)(data))->checksum_finish = true;
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

void gzipOCLHost::set_checksum(uint32_t cval) {
    m_checksum = cval;
}

uint32_t gzipOCLHost::calculate_crc32(uint8_t* in, size_t input_size) {
    m_checksum = crc32(m_checksum, in, input_size);
    return m_checksum;
}

uint32_t gzipOCLHost::calculate_adler32(uint8_t* in, size_t input_size) {
    m_checksum = adler32(m_checksum, in, input_size);
    return m_checksum;
}

void gzipOCLHost::generate_checksum(uint8_t* in, size_t input_size) {
    if (m_zlibFlow) {
        m_future_checksum = std::async(std::launch::async, &gzipOCLHost::calculate_adler32, this, in, input_size);
    } else {
        m_future_checksum = std::async(std::launch::async, &gzipOCLHost::calculate_crc32, this, in, input_size);
    }
}

uint32_t gzipOCLHost::get_checksum(void) {
#ifdef ENABLE_SW_CHECKSUM
    return m_future_checksum.get();
#else
    return m_checksum;
#endif
}

// OpenCL setup initialization
int gzipOCLHost::init(const std::string& binaryFileName, uint8_t kidx) {
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
        for (size_t i = 0; i < platforms.size(); i++) {
            platform = platforms[i];
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
    //    OCL_CHECK(err, m_def_q = new cl::CommandQueue(*m_context, m_device, m_isProfile, &err));
    if ((m_cdflow == BOTH) || (m_cdflow == COMPRESS)) {
        OCL_CHECK(err,
                  m_def_q = new cl::CommandQueue(*m_context, m_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    }

    // DeCompress Command Queue & Kernel Setup
    if ((m_cdflow == BOTH) || (m_cdflow == DECOMPRESS)) {
#ifndef FREE_RUNNING_KERNEL
        for (uint8_t i = 0; i < D_COMPUTE_UNIT; i++) {
            OCL_CHECK(err, m_q_dec[i] = new cl::CommandQueue(*m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err));
        }
#endif
        for (uint8_t i = 0; i < D_COMPUTE_UNIT; i++) {
            OCL_CHECK(err, m_q_rd[i] = new cl::CommandQueue(*m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err));
            OCL_CHECK(err, m_q_rdd[i] = new cl::CommandQueue(*m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err));
            OCL_CHECK(err, m_q_wr[i] = new cl::CommandQueue(*m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err));
            OCL_CHECK(err, m_q_wrd[i] = new cl::CommandQueue(*m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err));
        }
    }
    // OpenCL Host / Device Buffer Setup Start
    return 0;
}

gzipOCLHost::gzipOCLHost(const std::string& binaryFileName,
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

    m_memoryManager = new memoryManager(8, m_context, bank_id);

    if ((cd_flow == BOTH) || (cd_flow == COMP_ONLY)) {
        MEM_ALLOC_CHECK(h_buf_checksum_data.resize(1), 1, "Checksum Data");
        buffer_checksum_data = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint32_t),
                                              h_buf_checksum_data.data(), &err);
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
        MEM_ALLOC_CHECK(h_dcompressStatus.resize(sizeof(uint32_t)), sizeof(uint32_t), "DecompressStatus Host Buffer");
    }
}

// Constructor
gzipOCLHost::gzipOCLHost(enum State flow,
                         const std::string& binaryFileName,
                         const std::string& uncompfilename,
                         const bool is_seq,
                         uint8_t device_id,
                         bool enable_profile,
                         uint8_t deckerneltype,
                         uint8_t dflow,
                         bool freeRunKernel) {
    // Zlib Compression Binary Name
    m_cdflow = flow;
    m_enableProfile = enable_profile;
    m_deviceid = device_id;
    m_zlibFlow = (enum design_flow)dflow;
    m_kidx = deckerneltype;
    m_pending = 0;
    m_isSeq = is_seq;
    m_freeRunKernel = freeRunKernel;
    m_inFileName = uncompfilename;
    if (m_zlibFlow) {
        m_checksum = 1;
        m_isZlib = true;
    }
    uint8_t kidx = deckerneltype;
    // OpenCL setup
    int err = init(binaryFileName, kidx);
    if (err) {
        std::cerr << "\nOpenCL Setup Failed" << std::endl;
        release();
        return;
    }

    m_memoryManager = new memoryManager(8, m_context);

    if ((m_cdflow == BOTH) || (m_cdflow == COMPRESS)) {
        MEM_ALLOC_CHECK(h_buf_checksum_data.resize(1), 1, "Checksum Data");
        cl_int err;
        if (!m_freeRunKernel) {
            OCL_CHECK(err, buffer_checksum_data = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                                 sizeof(uint32_t), h_buf_checksum_data.data(), &err));
        }

        if (m_freeRunKernel) {
            OCL_CHECK(err, datamover_kernel = new cl::Kernel(*m_program, datamover_kernel_name.c_str(), &err));
        }
    }

    if ((m_cdflow == BOTH) || (m_cdflow == DECOMPRESS)) {
        // Decompression host buffer allocation
        for (int j = 0; j < DIN_BUFFERCOUNT; ++j)
            MEM_ALLOC_CHECK(h_dbufstream_in[j].resize(INPUT_BUFFER_SIZE), INPUT_BUFFER_SIZE, "Input Buffer");

        for (int j = 0; j < DOUT_BUFFERCOUNT; ++j) {
            MEM_ALLOC_CHECK(h_dbufstream_zlibout[j].resize(OUTPUT_BUFFER_SIZE), OUTPUT_BUFFER_SIZE,
                            "Output Host Buffer");
            MEM_ALLOC_CHECK(h_dcompressSize_stream[j].resize(sizeof(uint32_t)), sizeof(uint32_t),
                            "DecompressSize Host Buffer");
        }
        MEM_ALLOC_CHECK(h_dcompressStatus.resize(sizeof(uint32_t)), sizeof(uint32_t), "DecompressStatus Host Buffer");
    }
    // OpenCL Host / Device Buffer Setup End
}

// Constructor
gzipOCLHost::gzipOCLHost(const std::string& binaryFileName,
                         uint8_t max_cr,
                         uint8_t cd_flow,
                         uint8_t device_id,
                         uint8_t profile,
                         uint8_t d_type,
                         enum design_flow dflow) {
    // Zlib Compression Binary Name
    m_cdflow = cd_flow;
    m_isProfile = profile;
    m_deviceid = device_id;
    m_max_cr = max_cr;
    m_zlibFlow = dflow;
    m_kidx = d_type;
    m_pending = 0;
    if (m_zlibFlow) m_checksum = 1;
    uint8_t kidx = d_type;

    // OpenCL setup
    int err = init(binaryFileName, kidx);
    if (err) {
        std::cerr << "\nOpenCL Setup Failed" << std::endl;
        release();
        return;
    }

    m_memoryManager = new memoryManager(8, m_context);

    if ((cd_flow == BOTH) || (cd_flow == COMP_ONLY)) {
        MEM_ALLOC_CHECK(h_buf_checksum_data.resize(1), 1, "Checksum Data");
        cl_int err;
        OCL_CHECK(err, buffer_checksum_data = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                             sizeof(uint32_t), h_buf_checksum_data.data(), &err));
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
        MEM_ALLOC_CHECK(h_dcompressStatus.resize(sizeof(uint32_t)), sizeof(uint32_t), "DecompressStatus Host Buffer");
    }
    // OpenCL Host / Device Buffer Setup End
}

int gzipOCLHost::error_code(void) {
    return m_err_code;
}

void gzipOCLHost::release_dec_buffers(void) {
    // delete the allocated buffers
    for (int i = 0; i < DIN_BUFFERCOUNT; i++) {
        DELETE_OBJ(buffer_dec_input[i]);
    }

    for (int i = 0; i < DOUT_BUFFERCOUNT + 1; i++) {
        DELETE_OBJ(buffer_dec_zlib_output[i]);
    }
}

void gzipOCLHost::release() {
    if (!m_islibz) {
        DELETE_OBJ(m_program);
        DELETE_OBJ(m_context);
    }

    if (is_freeRunKernel()) {
        DELETE_OBJ(datamover_kernel);
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

    if (m_memoryManager) {
        delete m_memoryManager;
    }

    if (m_def_q) {
        delete m_def_q;
    }

    if (m_compressFullKernel) {
        delete m_compressFullKernel;
    }
}

// Destructor
gzipOCLHost::~gzipOCLHost() {
    release();
}

// method to enqueue reads in parallel with writes to decompression kernel
void gzipOCLHost::_enqueue_reads(
    uint32_t bufSize, uint8_t* out, uint32_t* decompSize, int cu, uint32_t max_outbuf_size) {
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

                if ((dcmpSize >= max_outbuf_size) && (max_outbuf_size != 0)) {
                    std::cout << "\n" << std::endl;
                    std::cout << "\x1B[35mZIP BOMB: Exceeded output buffer size during "
                                 "decompression \033[0m \n"
                              << std::endl;
                    std::cout << "\x1B[35mUse -mcr option to increase the maximum "
                                 "compression ratio (Default: 10) \033[0m \n"
                              << std::endl;
                    std::cout << "\x1B[35mAborting .... \033[0m\n" << std::endl;
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
void gzipOCLHost::_enqueue_writes(uint32_t bufSize, uint8_t* in, uint32_t inputSize, int cu) {
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

uint64_t gzipOCLHost::decompressEngineSeq(
    uint8_t* in, uint8_t* out, uint64_t input_size, size_t max_outbuf_size, int cu) {
#if (VERBOSE_LEVEL >= 1)
    std::cout << "CU" << cu << " ";
#endif
    std::chrono::duration<double, std::nano> decompress_API_time_ns_1(0);
    cl_int err;
    // Streaming based solution
    uint32_t inBufferSize = input_size;
    uint32_t isLast = 1;
    const uint32_t lim_4gb = (uint32_t)(((uint64_t)1024 * 1024 * 1024) - 2); // 4GB limit on output size
    uint32_t outBufferSize = 0;
    h_dcompressStatus.data()[0] = 0;
    if (max_outbuf_size > lim_4gb) {
        outBufferSize = lim_4gb;
    } else {
        outBufferSize = (uint32_t)max_outbuf_size;
    }

    // host allocated aligned memory
    std::vector<uint8_t, aligned_allocator<uint8_t> > dbuf_in(inBufferSize);
    std::vector<uint8_t, aligned_allocator<uint8_t> > dbuf_out(outBufferSize);
    std::vector<uint32_t, aligned_allocator<uint32_t> > dbuf_outSize(2);

    cl::Buffer* buffer_dec_zlib_output =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, outBufferSize, dbuf_out.data());

    cl::Buffer* buffer_dec_input =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, inBufferSize, dbuf_in.data());

    cl::Buffer* buffer_size =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(uint32_t), dbuf_outSize.data());
    cl::Buffer* buffer_status =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint32_t), h_dcompressStatus.data());

    uint8_t kidx = m_kidx;
    std::string cu_id = std::to_string((cu + 1));
    std::string decompress_kname =
        stream_decompress_kernel_name[kidx] + ":{" + stream_decompress_kernel_name[kidx] + "_" + cu_id + "}";
    OCL_CHECK(err, cl::Kernel decompress_kernel(*m_program, decompress_kname.c_str(), &err));

    std::string data_writer_kname = data_writer_kernel_name + ":{" + data_writer_kernel_name + "_" + cu_id + "}";
    OCL_CHECK(err, cl::Kernel data_writer_kernel(*m_program, data_writer_kname.c_str(), &err));

    std::string data_reader_kname = data_reader_kernel_name + ":{" + data_reader_kernel_name + "_" + cu_id + "}";
    OCL_CHECK(err, cl::Kernel data_reader_kernel(*m_program, data_reader_kname.c_str(), &err));

    data_writer_kernel.setArg(0, *(buffer_dec_input));
    data_writer_kernel.setArg(1, inBufferSize);
    data_writer_kernel.setArg(2, isLast);

    data_reader_kernel.setArg(0, *(buffer_dec_zlib_output));
    data_reader_kernel.setArg(1, *(buffer_size));
    data_reader_kernel.setArg(2, *(buffer_status));
    data_reader_kernel.setArg(3, outBufferSize);

    uint32_t decmpSizeIdx = 0;
    h_dcompressStatus.data()[0] = 0;

    // Copy input data
    std::memcpy(dbuf_in.data(), in, inBufferSize); // must be equal to input_size
    OCL_CHECK(err, err = m_q_wr[0]->enqueueMigrateMemObjects({*(buffer_dec_input)}, 0, NULL, NULL));
    m_q_wr[0]->finish();

    OCL_CHECK(err, err = m_q_rd[0]->enqueueMigrateMemObjects({*(buffer_status)}, 0, NULL, NULL));
    m_q_rd[0]->finish();

// start parallel reader kernel enqueue thread
#ifdef FREE_RUNNING_KERNEL
    // enqueue data movers
    m_q_rd[0]->enqueueTask(data_reader_kernel);

    sleep(1);
    // enqueue decompression kernel
    auto decompress_API_start = std::chrono::high_resolution_clock::now();
    m_q_wr[0]->enqueueTask(data_writer_kernel);
    m_q_wr[0]->finish();
    auto decompress_API_end = std::chrono::high_resolution_clock::now();
#else
    m_q_dec[0]->finish();
    m_q_wr[0]->enqueueTask(data_writer_kernel);
    m_q_rd[0]->enqueueTask(data_reader_kernel);

    sleep(1);
    // enqueue decompression kernel
    auto decompress_API_start = std::chrono::high_resolution_clock::now();

    m_q_dec[0]->enqueueTask(decompress_kernel);
    m_q_dec[0]->finish();

    auto decompress_API_end = std::chrono::high_resolution_clock::now();
#endif
    auto duration = std::chrono::duration<double, std::nano>(decompress_API_end - decompress_API_start);
    decompress_API_time_ns_1 += duration;

    // wait for reader to finish
    m_q_rd[0]->finish();

    // copy decompressed output data
    m_q_rd[0]->enqueueMigrateMemObjects({*(buffer_size), *(buffer_status)}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, NULL);
    m_q_rd[0]->finish();

    // decompressed size
    decmpSizeIdx = dbuf_outSize[0];
    if (decmpSizeIdx >= max_outbuf_size) {
        auto brkloop = false;
        do {
            m_q_rd[0]->enqueueTask(data_reader_kernel);
            // wait for reader to finish
            m_q_rd[0]->finish();

            // copy decompressed output data
            m_q_rd[0]->enqueueMigrateMemObjects({*(buffer_size), *(buffer_status)}, CL_MIGRATE_MEM_OBJECT_HOST, NULL,
                                                NULL);
            m_q_rd[0]->finish();

            decmpSizeIdx += dbuf_outSize[0];
            brkloop = h_dcompressStatus.data()[0];
        } while (!brkloop);
        std::cout << "\n" << std::endl;
        std::cout << "compressed size (.gz/.xz): " << input_size << std::endl;
        std::cout << "decompressed size : " << decmpSizeIdx << std::endl;
        std::cout << "maximum output buffer allocated: " << max_outbuf_size << std::endl;
        std::cout << "Output Buffer Size Exceeds as the Compression Ratio is High " << std::endl;
        std::cout << "Use -mcr option to increase the output buffer size (Default: 20) --> Aborting " << std::endl;
        abort();
    } else {
        m_q_rd[0]->enqueueMigrateMemObjects({*(buffer_dec_zlib_output)}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, NULL);
        m_q_rd[0]->finish();
        // copy output decompressed data
        std::memcpy(out, dbuf_out.data(), decmpSizeIdx);
    }
    float throughput_in_mbps_1 = (float)decmpSizeIdx * 1000 / decompress_API_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;

    // free the cl buffers
    delete buffer_dec_input;
    delete buffer_dec_zlib_output;
    delete buffer_size;
    delete buffer_status;

    // printme("Done with decompress \n");
    return decmpSizeIdx;
}

size_t gzipOCLHost::decompressEngine(uint8_t* in, uint8_t* out, size_t input_size, size_t max_outbuf_size, int cu) {
    if (m_enableProfile) {
        gzipBase::total_start = std::chrono::high_resolution_clock::now();
    }
    cl_int err;
#if (VERBOSE_LEVEL >= 1)
    std::cout << "CU" << cu << " ";
#endif
    // Streaming based solution
    uint32_t inBufferSize = INPUT_BUFFER_SIZE;
    uint32_t outBufferSize = OUTPUT_BUFFER_SIZE;

    for (int i = 0; i < DOUT_BUFFERCOUNT; i++) {
        ZOCL_CHECK_2(err,
                     buffer_dec_zlib_output[i] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
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
    std::thread decompWriter(&gzipOCLHost::_enqueue_writes, this, inBufferSize, in, input_size, cu);
    std::thread decompReader(&gzipOCLHost::_enqueue_reads, this, outBufferSize, out, &decmpSizeIdx, cu,
                             max_outbuf_size);
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
    decompWriter.join();
    decompReader.join();
    release_dec_buffers();
    if (m_enableProfile) {
        gzipBase::total_end = std::chrono::high_resolution_clock::now();
        auto total_time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
        float throughput_in_mbps_1 = (float)decmpSizeIdx * 1000 / total_time_ns.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }

    if (m_derr_code) {
        return 0;
    } else {
        return decmpSizeIdx;
    }
}

size_t gzipOCLHost::deflate_buffer(
    uint8_t* in, uint8_t* out, size_t& input_size, bool& last_data, bool last_buffer, const std::string& cu_id) {
    cl_int err;
    bool initialize_checksum = false;
    bool actionDone = false;
    std::string compress_kname;
    if (m_compressFullKernel == NULL) {
#ifdef ENABLE_XRM
        // XRM: Use the CU instance id and create compress kernel
        compress_kname = compress_kernel_names[2] + ":{" + cu_id + "}";
#else
        // Random: Use kernel name directly and let XRT
        // decide the CU connection based on HBM bank
        compress_kname = compress_kernel_names[2];
#endif
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
    m_lastData = (last_buffer) ? true : false;

    buffers* lastBuffer = m_memoryManager->getLastBuffer();
    if (input_size) {
        auto buffer = m_memoryManager->createBuffer(input_size);
        if (buffer != NULL) {
#ifdef ENABLE_SW_CHECKSUM
            // Trigger CRC calculation
            generate_checksum(in, input_size);
#endif

            if (lastBuffer == buffer) lastBuffer = NULL;
            std::memcpy(buffer->h_buf_in, &in[0], input_size);
            uint32_t inSize = input_size;
            buffer->store_size = 0;

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

            OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects(
                               {*(buffer->buffer_compress_size), *(buffer->buffer_zlib_output)},
                               CL_MIGRATE_MEM_OBJECT_HOST, &(cmpEvents), &(buffer->rd_event)));

            cl_int err;
            OCL_CHECK(err, err = buffer->rd_event.setCallback(CL_COMPLETE, event_compress_cb, (void*)buffer));
            buffer->checksum_finish = true;

            input_size = 0;
            actionDone = true;
        }
    }

    if ((m_memoryManager->peekBuffer() != NULL) && (m_memoryManager->peekBuffer()->is_finish())) {
        buffers* buffer;
        uint32_t compSize = 0;
        auto size = 0;
        buffer = m_memoryManager->getBuffer();
        if (buffer != NULL) {
            actionDone = true;

            compSize = (buffer->h_compressSize)[0];
            std::memcpy(out, &buffer->h_buf_zlibout[0], compSize);
            size += compSize;

            // handle checksum for next iteration
            if (checksum_type) h_buf_checksum_data.data()[0] = ~h_buf_checksum_data.data()[0];
        }

        if (m_lastData && (input_size == 0) && (m_memoryManager->isPending() == false))
            last_data = true;
        else
            last_data = false;

        if (last_data) {
            OCL_CHECK(err,
                      err = m_def_q->enqueueMigrateMemObjects({*(buffer_checksum_data)}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, err = m_def_q->finish());

            m_checksum = h_buf_checksum_data.data()[0];
            if (checksum_type) m_checksum = ~m_checksum;
        }

        return size;
    }
    if ((actionDone == false) && (m_memoryManager->isPending() == true)) {
        // wait for a queue to finish to save CPU cycles
        auto buffer = m_memoryManager->peekBuffer();
        if (buffer != NULL) {
            buffer->rd_event.wait();
        }
    }

    if (m_lastData && (input_size == 0) && (m_memoryManager->isPending() == false))
        last_data = true;
    else
        last_data = false;
    return 0;
}

// This version of compression does overlapped execution between
// Kernel and Host. I/O operations between Host and Device are
// overlapped with Kernel execution between multiple compute units
size_t gzipOCLHost::compress_buffer(uint8_t* in, uint8_t* out, size_t input_size, uint32_t host_buffer_size, int cu) {
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
size_t gzipOCLHost::add_header(uint8_t* out, int level, int strategy, int window_bits) {
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

std::string gzipOCLHost::getCompressKernel(int idx) {
    return compress_kernel_names[idx];
}

std::string gzipOCLHost::getDeCompressKernel(int idx) {
    return stream_decompress_kernel_name[idx];
}

// Add zlib footer
size_t gzipOCLHost::add_footer(uint8_t* out, size_t compressSize) {
    size_t outIdx = compressSize;
    uint32_t checksum = 0;

    outIdx = compressSize;
#ifdef ENABLE_SW_CHECKSUM
    checksum = get_checksum();
#else
    checksum = m_checksum;
#endif

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
size_t gzipOCLHost::compress(
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

// This version of compression does overlapped execution between
// Kernel and Host. I/O operations between Host and Device are
// overlapped with Kernel execution between multiple compute units
size_t gzipOCLHost::compressEngineOverlap(uint8_t* in,
                                          uint8_t* out,
                                          size_t input_size,
                                          int cu,
                                          int level,
                                          int strategy,
                                          int window_bits,
                                          uint32_t* checksum) {
    if (m_enableProfile) {
        gzipBase::total_start = std::chrono::high_resolution_clock::now();
    }
    size_t total_size = 0;
    size_t host_buffer_size = HOST_BUFFER_SIZE;
    // Call compress buffer
    total_size = compress_buffer(in, &out[total_size], input_size, host_buffer_size, cu);
    *checksum = m_checksum;

    if (m_enableProfile) {
        gzipBase::total_end = std::chrono::high_resolution_clock::now();
        auto total_time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
        float throughput_in_mbps_1 = (float)input_size * 1000 / total_time_ns.count();
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    }

    return total_size;
} // Overlap end

// This version of compression does sequential execution between kernel an dhost
size_t gzipOCLHost::compressEngineSeq(
    uint8_t* in, uint8_t* out, size_t input_size, int level, int strategy, int window_bits, uint32_t* checksum) {
    cl_int err;
#ifdef GZIP_STREAM
    std::string compress_kname = compress_kernel_names[1];
#else
    std::string compress_kname = compress_kernel_names[2];
#endif
    OCL_CHECK(err, m_compressFullKernel = new cl::Kernel(*m_program, compress_kname.c_str(), &err));
    auto c_inputSize = input_size;
    auto num_itr = 1;

    if (this->is_freeRunKernel()) {
        if (c_inputSize < 0x40000000) {
            num_itr = xcl::is_emulation() ? 1 : (0x40000000 / input_size);
        }
    }

    bool checksum_type;

    if (m_zlibFlow) {
        h_buf_checksum_data.data()[0] = 1;
        checksum_type = false;
    } else {
        h_buf_checksum_data.data()[0] = ~0;
        checksum_type = true;
    }

    auto output_size = c_inputSize + ((c_inputSize - 1) / m_cBlkSize + 1) * 100;

    // Host buffers
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_input_buffer(c_inputSize);
    std::vector<uint8_t, aligned_allocator<uint8_t> > h_output_buffer(output_size);
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_compressSize(10);

    // Device buffers
    OCL_CHECK(err,
              cl::Buffer* buffer_input = new cl::Buffer(*m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                                        sizeof(uint8_t) * c_inputSize, h_input_buffer.data(), &err));
    OCL_CHECK(
        err, cl::Buffer* buffer_output = new cl::Buffer(*m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                        sizeof(uint8_t) * (output_size), h_output_buffer.data(), &err));
    OCL_CHECK(err, cl::Buffer* buffer_cSize = new cl::Buffer(*m_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                             sizeof(uint32_t) * 10, h_compressSize.data(), &err));

    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);
    auto enbytes = 0;
    auto outIdx = 0;
    // Process the data serially
    uint32_t buf_size = c_inputSize;

    // Copy input data
    std::memcpy(h_input_buffer.data(), in, buf_size);

    // Set kernel arguments
    int narg = 0;

    if (this->is_freeRunKernel()) {
        datamover_kernel->setArg(narg++, *buffer_input);
        datamover_kernel->setArg(narg++, *buffer_output);
        datamover_kernel->setArg(narg++, buf_size);
#ifdef PERF_DM
        datamover_kernel->setArg(narg++, num_itr);
#endif
        datamover_kernel->setArg(narg++, *buffer_cSize);
    } else {
        m_compressFullKernel->setArg(narg++, *(buffer_input));
        m_compressFullKernel->setArg(narg++, *(buffer_output));
        m_compressFullKernel->setArg(narg++, *(buffer_cSize));
        m_compressFullKernel->setArg(narg++, *(buffer_checksum_data));
        m_compressFullKernel->setArg(narg++, buf_size);
        m_compressFullKernel->setArg(narg++, checksum_type);
    }

    if (this->is_freeRunKernel()) {
        OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer_input), *(buffer_output), *(buffer_cSize)},
                                                               0 /* 0 means from host*/));
    } else {
        // Transfer the data to device
        OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer_input), *(buffer_checksum_data)}, 0, nullptr,
                                                               nullptr));
    }

    m_def_q->finish();
    auto kernel_start = std::chrono::high_resolution_clock::now();

#ifndef PERF_DM
    for (auto z = 0; z < num_itr; z++) {
#endif
        if (this->is_freeRunKernel()) {
            m_def_q->enqueueTask(*datamover_kernel);
#ifdef DISABLE_FREE_RUNNING_KERNEL
            m_def_q->enqueueTask(*m_compressFullKernel);
#endif
        } else {
            OCL_CHECK(err, err = m_def_q->enqueueTask(*m_compressFullKernel));
        }
#ifndef PERF_DM
    }
#endif

    m_def_q->finish();

    auto kernel_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::nano>(kernel_stop - kernel_start);
    kernel_time_ns_1 += duration;

    // Transfer the data from device to host
    OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer_cSize)}, CL_MIGRATE_MEM_OBJECT_HOST));
    m_def_q->finish();
    BOOST_ASSERT_MSG(h_compressSize[0] < unsigned(output_size),
                     "\x1B[35m xGzip Error: Output Buffer Size is not sufficient \033[0m ");

    if (this->is_freeRunKernel()) {
        // Transfer the data from device to host
        OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer_output)}, CL_MIGRATE_MEM_OBJECT_HOST));
    } else {
        // Transfer the data from device to host
        OCL_CHECK(err, err = m_def_q->enqueueMigrateMemObjects({*(buffer_output), *(buffer_checksum_data)},
                                                               CL_MIGRATE_MEM_OBJECT_HOST));
    }
    m_def_q->finish();

    auto compSize = h_compressSize[0];
    std::memcpy(out + outIdx, &h_output_buffer[0], compSize);
    enbytes += compSize;
    outIdx += compSize;

    if (!is_freeRunKernel()) {
        // This handling required only for zlib
        // This is for next iteration
        if (checksum_type) h_buf_checksum_data.data()[0] = ~h_buf_checksum_data.data()[0];
    }
    m_def_q->finish();
    float throughput_in_mbps_1 = (float)input_size * 1000 * num_itr / kernel_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;

    if (!is_freeRunKernel()) {
        // Add last block header
        long int last_block = 0xffff000001;
        std::memcpy(&(out[outIdx]), &last_block, 5);
        enbytes += 5;

        m_checksum = h_buf_checksum_data.data()[0];
        if (checksum_type) m_checksum = ~m_checksum;

        *checksum = m_checksum;
    }

    return enbytes;
} // Sequential end

memoryManager::memoryManager(uint8_t max_buffer, cl::Context* context, const int bank_id) {
    maxBufCount = max_buffer;
    mContext = context;
    bufCount = 0;
    m_bankId = bank_id;
}

memoryManager::~memoryManager() {
    release();
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
        aligned_allocator<uint8_t> alocator;
        aligned_allocator<uint32_t> alocator_1;
        MEM_ALLOC_CHECK((buffer->h_buf_in = alocator.allocate(size)), size, "Input Host Buffer");
        MEM_ALLOC_CHECK((buffer->h_buf_zlibout = alocator.allocate(size)), size, "Output Host Buffer");
        uint16_t max_blocks = (size / BLOCK_SIZE_IN_KB) + 1;
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

        buffer->input_size = size;
        buffer->allocated_size = size;
        this->busyBuffers.push(buffer);
        bufCount++;
        lastBuffer = buffer;
        return buffer;
    }
    return NULL;
}

cl::Buffer* memoryManager::getBuffer(cl_mem_flags flag, size_t size, uint8_t* host_ptr) {
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
#ifdef ENABLE_XRM
        auto bank = m_bankId | XCL_MEM_TOPOLOGY;
        ext.flags = bank;
        clBuffer = new cl::Buffer(*(this->mContext), l_flag, size, &ext, &err);
#else
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
#endif
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
