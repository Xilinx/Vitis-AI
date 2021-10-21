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
#include <fcntl.h> /* For O_RDWR */
#include <sys/stat.h>
#include <unistd.h> /* For open(), creat() */
auto crc = 0;       // CRC32 value
extern unsigned long crc32(unsigned long crc, const unsigned char* buf, uint32_t len);
int fd_p2p_c_in = 0;
const int RESIDUE_4K = 4096;

uint64_t get_file_size(std::ifstream& file) {
    file.seekg(0, file.end);
    uint64_t file_size = file.tellg();
    file.seekg(0, file.beg);
    return file_size;
}

constexpr uint32_t roundoff(uint32_t x, uint32_t y) {
    return ((x - 1) / (y) + 1);
}

void gzip_headers(std::string& inFile_name, std::ofstream& outFile, uint8_t* zip_out, uint32_t enbytes) {
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
    stat(inFile_name.c_str(), &istat);
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
    for (int i = 0; inFile_name[i] != '\0'; i++) {
        outFile.put(inFile_name[i]);
    }

    outFile.put(0);
    outFile.write((char*)zip_out, enbytes);

#ifdef GZIP_MULTICORE_COMPRESS
    // Last Block
    outFile.put(1);
    outFile.put(0);
    outFile.put(0);
    outFile.put(255);
    outFile.put(255);
#endif

    unsigned long ifile_size = istat.st_size;
    uint8_t crc_byte = 0;
    long crc_val = crc;
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

    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
}

void gzip_crc32(uint8_t* in, uint32_t input_size) {
    crc = crc32(crc, in, input_size);
}
void zlib_headers(std::string& inFile_name, std::ofstream& outFile, uint8_t* zip_out, uint32_t enbytes) {
    outFile.put(120);
    outFile.put(1);
    outFile.write((char*)zip_out, enbytes);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
}

uint32_t xil_zlib::compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size) {
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

    uint32_t host_buffer_size = HOST_BUFFER_SIZE;

#ifdef GZIP_MODE
#ifndef GZIP_MULTICORE_COMPRESS
    std::thread gzipCrc(gzip_crc32, zlib_in.data(), input_size);
#endif
#else
// Space for zlib CRC
#endif

// zlib Compress
#ifdef GZIP_MULTICORE_COMPRESS
    uint32_t enbytes = compressFull(zlib_in.data(), zlib_out.data(), input_size);
#else
    uint32_t enbytes = compress(zlib_in.data(), zlib_out.data(), input_size, host_buffer_size);
#endif

    if (enbytes > 0) {
#ifdef GZIP_MODE
#ifndef GZIP_MULTICORE_COMPRESS
        gzipCrc.join();
#endif
        // Pack gzip encoded stream .gz file
        gzip_headers(inFile_name, outFile, zlib_out.data(), enbytes);
#else
        // Pack zlib encoded stream .zlib file
        zlib_headers(inFile_name, outFile, zlib_out.data(), enbytes);
#endif
    }
    // Close file
    inFile.close();
    outFile.close();
    return enbytes;
}

void error_message(const std::string& val) {
    std::cout << "\n";
    std::cout << "Please provide " << val << " option" << std::endl;
    std::cout << "Exiting Application" << std::endl;
    std::cout << "\n";
}

int validate(std::string& inFile_name, std::string& outFile_name) {
    std::string command = "cmp " + inFile_name + " " + outFile_name;
    int ret = system(command.c_str());
    return ret;
}

// Constructor
xil_zlib::xil_zlib(const std::string& binaryFileName,
                   uint8_t flow,
                   uint8_t max_cr,
                   uint8_t device_id,
                   uint8_t d_type,
                   bool is_dec_overlap) {
    // Zlib Compression Binary Name
    m_deviceid = device_id;
    init(binaryFileName, flow, d_type);

    m_max_cr = max_cr;
    m_useOverlapDec = is_dec_overlap;

    // printf("C_COMPUTE_UNIT \n");
    uint32_t block_size_in_kb = BLOCK_SIZE_IN_KB;
    uint32_t block_size_in_bytes = block_size_in_kb * 1024;
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    uint32_t host_buffer_size = HOST_BUFFER_SIZE;
    uint32_t temp_nblocks = (host_buffer_size - 1) / block_size_in_bytes + 1;
    host_buffer_size = ((host_buffer_size - 1) / block_size_in_kb + 1) * block_size_in_kb;

    const uint16_t c_ltree_size = 1024;
    const uint16_t c_dtree_size = 64;

    for (int i = 0; i < MAX_CCOMP_UNITS; i++) {
        for (int j = 0; j < OVERLAP_BUF_COUNT; j++) {
            // Index calculation
            MEM_ALLOC_CHECK(h_buf_in[i][j].resize(HOST_BUFFER_SIZE), HOST_BUFFER_SIZE, "Input Host Buffer");
            MEM_ALLOC_CHECK(h_buf_out[i][j].resize(HOST_BUFFER_SIZE * 4), HOST_BUFFER_SIZE * 4,
                            "LZ77Output Host Buffer");
            MEM_ALLOC_CHECK(h_buf_zlibout[i][j].resize(HOST_BUFFER_SIZE * 2), HOST_BUFFER_SIZE * 2,
                            "ZlibOutput Host Buffer");
            MEM_ALLOC_CHECK(h_blksize[i][j].resize(MAX_NUMBER_BLOCKS), MAX_NUMBER_BLOCKS, "BlockSize Host Buffer");
            MEM_ALLOC_CHECK(h_compressSize[i][j].resize(MAX_NUMBER_BLOCKS), MAX_NUMBER_BLOCKS,
                            "CompressSize Host Buffer");
            MEM_ALLOC_CHECK(h_dyn_ltree_freq[i][j].resize(PARALLEL_ENGINES * c_ltree_size),
                            PARALLEL_ENGINES * c_ltree_size, "LTree Host Buffer");
            MEM_ALLOC_CHECK(h_dyn_dtree_freq[i][j].resize(PARALLEL_ENGINES * c_dtree_size),
                            PARALLEL_ENGINES * c_dtree_size, "DTree Host Buffer");
        }
    }
    // Device buffer allocation
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
            buffer_input[cu][flag] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                    host_buffer_size, h_buf_in[cu][flag].data());

            buffer_lz77_output[cu][flag] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                          host_buffer_size * 4, h_buf_out[cu][flag].data());

            buffer_zlib_output[cu][flag] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                          host_buffer_size * 2, h_buf_zlibout[cu][flag].data());

            buffer_compress_size[cu][flag] =
                new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, temp_nblocks * sizeof(uint32_t),
                               h_compressSize[cu][flag].data());

            buffer_inblk_size[cu][flag] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                         temp_nblocks * sizeof(uint32_t), h_blksize[cu][flag].data());

            buffer_dyn_ltree_freq[cu][flag] =
                new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                               PARALLEL_ENGINES * sizeof(uint32_t) * c_ltree_size, h_dyn_ltree_freq[cu][flag].data());

            buffer_dyn_dtree_freq[cu][flag] =
                new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                               PARALLEL_ENGINES * sizeof(uint32_t) * c_dtree_size, h_dyn_dtree_freq[cu][flag].data());
        }
    }

    // initialize the buffers
    for (int j = 0; j < DIN_BUFFERCOUNT; ++j)
        MEM_ALLOC_CHECK(h_dbuf_in[j].resize(INPUT_BUFFER_SIZE), INPUT_BUFFER_SIZE, "Input Buffer");

    for (int j = 0; j < DOUT_BUFFERCOUNT; ++j) {
        MEM_ALLOC_CHECK(h_dbuf_zlibout[j].resize(OUTPUT_BUFFER_SIZE), OUTPUT_BUFFER_SIZE, "Output Buffer");
        MEM_ALLOC_CHECK(h_dcompressSize[j].resize(sizeof(uint32_t)), sizeof(uint32_t), "DecompressSize Buffer");
    }
    MEM_ALLOC_CHECK(h_dcompressStatus.resize(sizeof(uint32_t)), sizeof(uint32_t), "DecompressStatus Buffer");
}

// Destructor
xil_zlib::~xil_zlib() {
    release();
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
            delete (buffer_input[cu][flag]);
            delete (buffer_lz77_output[cu][flag]);
            delete (buffer_zlib_output[cu][flag]);
            delete (buffer_compress_size[cu][flag]);
            delete (buffer_inblk_size[cu][flag]);

            delete (buffer_dyn_ltree_freq[cu][flag]);
            delete (buffer_dyn_dtree_freq[cu][flag]);
        }
    }
}

int xil_zlib::init(const std::string& binaryFileName, uint8_t flow, uint8_t d_type) {
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[m_deviceid];

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Using Device: " << device_name << std::endl;

    // Creating Context and Command Queue for selected Device
    m_context = new cl::Context(device);
    // m_q = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        m_q[i] = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }
#ifndef FREE_RUNNING_KERNEL
    m_q_dec = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
#endif
    m_q_rd = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    m_q_rdd = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    m_q_wr = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    m_q_wrd = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);

    // import_binary() command will find the OpenCL binary file created using the
    // v++ compiler load into OpenCL Binary and return as Binaries
    // OpenCL and it can contain many functions which can be executed on the
    // device.
    // std::string binaryFile =
    // xcl::find_binary_file(device_name,binaryFileName.c_str());
    // cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    auto fileBuf = xcl::read_binary_file(binaryFileName);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    m_program = new cl::Program(*m_context, {device}, bins);
    m_BinFlow = flow;
    if (flow == 1 || flow == 2) {
// Create Compress kernels
#ifdef GZIP_MULTICORE_COMPRESS
        compress_kernel = new cl::Kernel(*m_program, compress_kernel_names[1].c_str());
#else
        // Create Compress & Huffman kernels
        compress_kernel = new cl::Kernel(*m_program, compress_kernel_names[0].c_str());
        huffman_kernel = new cl::Kernel(*m_program, huffman_kernel_names[0].c_str());
#endif
    }
    if (flow == 0 || flow == 2) {
        // Create Decompress kernel
        decompress_kernel = new cl::Kernel(*m_program, decompress_kernel_names[d_type].c_str());
        data_writer_kernel = new cl::Kernel(*m_program, data_writer_kernel_names[0].c_str());
        data_reader_kernel = new cl::Kernel(*m_program, data_reader_kernel_names[0].c_str());
    }

    return 0;
}

int xil_zlib::release() {
    if (m_BinFlow) {
        delete compress_kernel;
#ifndef GZIP_MULTICORE_COMPRESS
        delete huffman_kernel;
#endif
    }
    if (m_BinFlow == 0 || m_BinFlow == 2) {
        delete decompress_kernel;
        delete data_writer_kernel;
        delete data_reader_kernel;
#ifndef FREE_RUNNING_KERNEL
        delete (m_q_dec);
#endif
        delete (m_q_rd);
        delete (m_q_rdd);
        delete (m_q_wr);
        delete (m_q_wrd);
    }

    delete (m_program);

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        delete (m_q[i]);
    }

    delete (m_context);

    return 0;
}

uint32_t xil_zlib::decompress_file(
    std::string& inFile_name, std::string& outFile_name, uint64_t input_size, int cu, bool enable_p2p) {
    std::ofstream outFile(outFile_name.c_str(), std::ofstream::binary);
    uint8_t c_max_cr = m_max_cr;

    std::vector<uint8_t, aligned_allocator<uint8_t> > in;
    // Allocat output size
    // 8 - Max CR per file expected, if this size is big
    // Decompression crashes
    std::vector<uint8_t, aligned_allocator<uint8_t> > out(input_size * c_max_cr);
    uint32_t debytes = 0;

    MEM_ALLOC_CHECK(out.resize(input_size * c_max_cr), input_size * c_max_cr, "Output Buffer");

    if (enable_p2p) {
        fd_p2p_c_in = open(inFile_name.c_str(), O_RDONLY | O_DIRECT);
        if (fd_p2p_c_in <= 0) {
            std::cout << "P2P: Unable to open input file, fd: %d\n" << fd_p2p_c_in << std::endl;
            exit(1);
        }

        // printme("Call to zlib_decompress \n");
        // Call decompress
        debytes = decompress(in.data(), out.data(), input_size, cu, enable_p2p);

    } else {
        // printme("In decompress_file \n");
        std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
        if (!inFile) {
            std::cout << "Unable to open file";
            exit(1);
        }

        MEM_ALLOC_CHECK(in.resize(input_size), input_size, "Input Buffer");
        // READ ZLIB header 2 bytes
        inFile.read((char*)in.data(), input_size);
        // printme("Call to zlib_decompress \n");
        // Call decompress
        if (m_useOverlapDec) {
            debytes = decompress(in.data(), out.data(), input_size, cu, enable_p2p);
        } else {
            debytes = decompressSeq(in.data(), out.data(), input_size, cu);
        }

        // Close file
        inFile.close();
    }
    outFile.write((char*)out.data(), debytes);
    outFile.close();

    return debytes;
}

// method to enqueue reads in parallel with writes to decompression kernel
void xil_zlib::_enqueue_reads(uint32_t bufSize, uint8_t* out, uint32_t* decompSize, uint32_t max_outbuf_size) {
    const int BUFCNT = DOUT_BUFFERCOUNT;
    cl::Event hostReadEvent[BUFCNT];
    cl::Event kernelReadEvent[BUFCNT];
    // cl::Event sizeReadEvent;
    std::vector<cl::Event> kernelReadWait[BUFCNT];

    uint8_t* outP = nullptr;
    uint32_t* outSize = nullptr;
    uint32_t dcmpSize = 0;
    cl::Buffer* buffer_out[BUFCNT];
    cl::Buffer* buffer_size[BUFCNT];
    cl::Buffer* buffer_status; // single common buffer to capture the
                               // decompression status by kernel
    for (int i = 0; i < BUFCNT; i++) {
        buffer_out[i] =
            new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bufSize, h_dbuf_zlibout[i].data());

        buffer_size[i] = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 2 * sizeof(uint32_t),
                                        h_dcompressSize[i].data());
    }
    buffer_status =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint32_t), h_dcompressStatus.data());
    h_dcompressStatus[0] = 0;
    m_q_rdd->enqueueMigrateMemObjects({*(buffer_status)}, 0, NULL, NULL);
    m_q_rdd->finish();

    // enqueue first set of buffers
    uint8_t cbf_idx = 0;
    uint32_t keq_idx = 0;
    uint32_t raw_size = 0;
    uint32_t cpy_cnt = 0;
    bool done = false;

    // set consistent buffer size to be read
    data_reader_kernel->setArg(2, *(buffer_status));
    data_reader_kernel->setArg(3, bufSize);
    do {
        cbf_idx = keq_idx % BUFCNT;
        if (((kernelReadWait[cbf_idx]).size() > 0) || (done)) {
            (hostReadEvent[cbf_idx]).wait(); // wait for previous data migration to complete
            if (!done) {
                outSize = h_dcompressSize[cbf_idx].data();
                raw_size = *outSize;
                outP = h_dbuf_zlibout[cbf_idx].data();
                // if output data size is multiple of buffer size, then (buffer_size +
                // 1) is sent by reader kernel
                if (raw_size > bufSize) {
                    --raw_size;
                }
                if (raw_size != 0) {
                    std::memcpy(out + dcmpSize, outP, raw_size);
                    dcmpSize += raw_size;
                }
                if (raw_size != bufSize) done = true;

                if (dcmpSize > max_outbuf_size) {
                    std::cout << "\n" << std::endl;
                    std::cout << "\x1B[35mZIP BOMB: Exceeded output buffer size during "
                                 "decompression \033[0m \n"
                              << std::endl;
                    std::cout << "\x1B[35mUse -mcr option to increase the maximum "
                                 "compression ratio (Default: 10) \033[0m \n"
                              << std::endl;
                    std::cout << "\x1B[35mAborting .... \033[0m\n" << std::endl;
                    exit(1);
                }
                if ((kernelReadWait[cbf_idx]).size() > 0)
                    kernelReadWait[cbf_idx].pop_back(); // must always have single element
            }
            ++cpy_cnt;
        }
        if (!done) {
            // set reader kernel arguments
            data_reader_kernel->setArg(0, *(buffer_out[cbf_idx]));
            data_reader_kernel->setArg(1, *(buffer_size[cbf_idx]));

            // enqueue reader kernel
            m_q_rd->enqueueTask(*data_reader_kernel, NULL, &(kernelReadEvent[cbf_idx]));
            kernelReadWait[cbf_idx].push_back(kernelReadEvent[cbf_idx]); // event to wait for

            // need to avoid full buffer copies for 0 bytes size data
            m_q_rdd->enqueueMigrateMemObjects({*(buffer_size[cbf_idx]), *(buffer_out[cbf_idx])},
                                              CL_MIGRATE_MEM_OBJECT_HOST, &(kernelReadWait[cbf_idx]),
                                              &(hostReadEvent[cbf_idx]));
            ++keq_idx;
        }
    } while (!(done && keq_idx == cpy_cnt));
    // wait for data transfer queue to finish
    m_q_rdd->finish();
    *decompSize = dcmpSize;

    // free the buffers
    for (int i = 0; i < BUFCNT; i++) {
        delete (buffer_out[i]);
        delete (buffer_size[i]);
    }
}

// method to enqueue writes in parallel with reads from decompression kernel
void xil_zlib::_enqueue_writes(uint32_t bufSize, uint8_t* in, uint32_t inputSize, bool enable_p2p) {
    const int BUFCNT = DIN_BUFFERCOUNT;

    // inputSize to 4k multiple due to p2p flow
    uint32_t inputSize4kMultiple = roundoff(inputSize, RESIDUE_4K) * RESIDUE_4K;
    uint32_t bufferCount = 0;
    if (enable_p2p) {
        bufferCount = bufSize == inputSize ? 1 : roundoff(inputSize4kMultiple, bufSize);
    } else {
        bufferCount = roundoff(inputSize, bufSize);
    }

    cl_mem_ext_ptr_t p2pInExt;
    std::chrono::duration<double, std::nano> total_ssd_time_ns(0);
    std::vector<char*> p2pPtrVec;

    uint8_t* inP = nullptr;

    if (enable_p2p) {
        // DDR buffer exyensions
        p2pInExt.flags = XCL_MEM_EXT_P2P_BUFFER;
        p2pInExt.obj = nullptr;
        p2pInExt.param = NULL;
    }

    cl::Buffer* buffer_in[BUFCNT];
    cl::Event hostWriteEvent[BUFCNT];
    cl::Event kernelWriteEvent[BUFCNT];

    std::vector<cl::Event> hostWriteWait[BUFCNT];

    for (int i = 0; i < BUFCNT; i++) {
        if (enable_p2p) {
            buffer_in[i] = new cl::Buffer(*m_context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, bufSize, &p2pInExt);
            char* p2pPtr = (char*)m_q_wr->enqueueMapBuffer(*(buffer_in[i]), CL_TRUE, CL_MAP_READ, 0, bufSize);
            p2pPtrVec.push_back(p2pPtr);
        } else {
            buffer_in[i] =
                new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, bufSize, h_dbuf_in[i].data());
        }
    }

    uint32_t cBufSize = bufSize;
    uint32_t cBufSize4kMultiple = 0;
    cBufSize4kMultiple = bufSize == inputSize ? (roundoff(bufSize, RESIDUE_4K) * RESIDUE_4K) : bufSize;

    uint8_t cbf_idx = 0;  // index indicating current buffer(0-4) being used in loop
    uint32_t keq_idx = 0; // index indicating the number of writer kernel enqueues
    uint32_t isLast = 0;

    for (keq_idx = 0; keq_idx < bufferCount; ++keq_idx) {
        cbf_idx = keq_idx % BUFCNT;

        if (keq_idx > BUFCNT - 1) {
            // wait for (current - BUFCNT) kernel to finish
            (kernelWriteEvent[cbf_idx]).wait();
        }
        inP = h_dbuf_in[cbf_idx].data();
        // set for last and other buffers
        if (keq_idx == bufferCount - 1) {
            if (bufferCount > 1) {
                cBufSize = inputSize - (bufSize * keq_idx);
                if (enable_p2p) cBufSize4kMultiple = roundoff(cBufSize, RESIDUE_4K) * RESIDUE_4K;
            }
            isLast = 1;
        }

        auto ssd_start = std::chrono::high_resolution_clock::now();
        // copy the data
        if (enable_p2p) {
            int ret = read(fd_p2p_c_in, p2pPtrVec[cbf_idx], cBufSize4kMultiple);
            if (ret == -1)
                std::cout << "P2P: compress(): read() failed, err: " << ret << ", line: " << __LINE__ << std::endl;
        } else {
            std::memcpy(inP, in + (keq_idx * bufSize), cBufSize);
        }
        auto ssd_end = std::chrono::high_resolution_clock::now();
        auto ssd_time_ns = std::chrono::duration<double, std::nano>(ssd_end - ssd_start);
        total_ssd_time_ns += ssd_time_ns;

        // set kernel arguments
        data_writer_kernel->setArg(0, *(buffer_in[cbf_idx]));
        data_writer_kernel->setArg(1, cBufSize);
        data_writer_kernel->setArg(2, isLast);

        if (enable_p2p) {
            m_q_wr->enqueueTask(*data_writer_kernel, NULL, &(kernelWriteEvent[cbf_idx]));
        } else {
            // enqueue data migration to kernel
            m_q_wr->enqueueMigrateMemObjects({*(buffer_in[cbf_idx])}, 0, NULL, NULL);
            // enqueue the writer kernel dependent on corresponding bufffer migration
            m_q_wr->enqueueTask(*data_writer_kernel, NULL, &(kernelWriteEvent[cbf_idx]));
        }
    }
    // wait for enqueued writer kernels to finish
    m_q_wr->finish();
    // delete the allocated buffers
    for (int i = 0; i < BUFCNT; i++) delete (buffer_in[i]);

    float ssd_throughput_in_mbps_1 = (float)inputSize * 1000 / total_ssd_time_ns.count();
    if (enable_p2p)
        std::cout << std::fixed << std::setprecision(2) << "SSD Throughput(MBps)\t:" << ssd_throughput_in_mbps_1
                  << std::endl;
}

uint32_t xil_zlib::decompress(uint8_t* in, uint8_t* out, uint32_t input_size, int cu, bool enable_p2p) {
    std::chrono::duration<double, std::nano> decompress_API_time_ns_1(0);
    uint32_t inBufferSize = INPUT_BUFFER_SIZE;
    uint32_t outBufferSize = OUTPUT_BUFFER_SIZE;
    const uint32_t max_outbuf_size = input_size * m_max_cr;
    // if input_size if greater than 2 MB, then buffer size must be 2MB
    if (input_size < inBufferSize) inBufferSize = input_size;
    if (max_outbuf_size < outBufferSize) outBufferSize = max_outbuf_size;

// start parallel reader kernel enqueue thread
#ifdef FREE_RUNNING_KERNEL
    auto decompress_API_start = std::chrono::high_resolution_clock::now();
#endif
    uint32_t decmpSizeIdx = 0;
    std::thread decompWriter(&xil_zlib::_enqueue_writes, this, inBufferSize, in, input_size, enable_p2p);
    std::thread decompReader(&xil_zlib::_enqueue_reads, this, outBufferSize, out, &decmpSizeIdx, max_outbuf_size);

#ifndef FREE_RUNNING_KERNEL
    // enqueue decompression kernel
    m_q_dec->finish();

    auto decompress_API_start = std::chrono::high_resolution_clock::now();

    m_q_dec->enqueueTask(*decompress_kernel);
    m_q_dec->finish();

    auto decompress_API_end = std::chrono::high_resolution_clock::now();
#endif

    decompReader.join();
    decompWriter.join();
#ifdef FREE_RUNNING_KERNEL
    auto decompress_API_end = std::chrono::high_resolution_clock::now();
#endif
    auto duration = std::chrono::duration<double, std::nano>(decompress_API_end - decompress_API_start);
    decompress_API_time_ns_1 += duration;
    float throughput_in_mbps_1 = (float)decmpSizeIdx * 1000 / decompress_API_time_ns_1.count();
    if (enable_p2p)
        std::cout << "E2E\t\t\t:" << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    else
        std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;

    // printme("Done with decompress \n");
    return decmpSizeIdx;
}

uint32_t xil_zlib::decompressSeq(uint8_t* in, uint8_t* out, uint32_t input_size, int cu) {
    std::chrono::duration<double, std::nano> decompress_API_time_ns_1(0);
    uint32_t inBufferSize = input_size;
    uint32_t isLast = 1;
    const uint64_t max_outbuf_size = input_size * m_max_cr;
    const uint32_t lim_4gb = (uint32_t)(((uint64_t)4 * 1024 * 1024 * 1024) - 2); // 4GB limit on output size
    uint32_t outBufferSize = 0;
    // allocate < 4GB size for output buffer
    if (max_outbuf_size > lim_4gb) {
        outBufferSize = lim_4gb;
    } else {
        outBufferSize = (uint32_t)max_outbuf_size;
    }
    // host allocated aligned memory
    std::vector<uint8_t, aligned_allocator<uint8_t> > dbuf_in(inBufferSize);
    std::vector<uint8_t, aligned_allocator<uint8_t> > dbuf_out(outBufferSize);
    std::vector<uint32_t, aligned_allocator<uint32_t> > dbuf_outSize(2);

    // opencl buffer creation
    cl::Buffer* buffer_dec_input =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, inBufferSize, dbuf_in.data());
    cl::Buffer* buffer_dec_output =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, outBufferSize, dbuf_out.data());

    cl::Buffer* buffer_size =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(uint32_t), dbuf_outSize.data());
    cl::Buffer* buffer_status =
        new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint32_t), h_dcompressStatus.data());

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
    uint32_t decmpSizeIdx = 0;

#ifdef FREE_RUNNING_KERNEL
    // enqueue data movers
    m_q_rd->enqueueTask(*data_reader_kernel);

    sleep(1);
    // enqueue decompression kernel
    auto decompress_API_start = std::chrono::high_resolution_clock::now();
    m_q_wr->enqueueTask(*data_writer_kernel);
    m_q_wr->finish();
    auto decompress_API_end = std::chrono::high_resolution_clock::now();
#else
    m_q_dec->finish();
    m_q_wr->enqueueTask(*data_writer_kernel);
    m_q_rd->enqueueTask(*data_reader_kernel);

    sleep(1);
    // enqueue decompression kernel
    auto decompress_API_start = std::chrono::high_resolution_clock::now();

    m_q_dec->enqueueTask(*decompress_kernel);
    m_q_dec->finish();

    auto decompress_API_end = std::chrono::high_resolution_clock::now();
#endif
    auto duration = std::chrono::duration<double, std::nano>(decompress_API_end - decompress_API_start);
    decompress_API_time_ns_1 += duration;

    // wait for reader to finish
    m_q_rd->finish();
    // copy decompressed output data
    m_q_rd->enqueueMigrateMemObjects({*(buffer_size), *(buffer_dec_output)}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, NULL);
    m_q_rd->finish();
    // decompressed size
    decmpSizeIdx = dbuf_outSize[0];
    // copy output decompressed data
    std::memcpy(out, dbuf_out.data(), decmpSizeIdx);

    float throughput_in_mbps_1 = (float)decmpSizeIdx * 1000 / decompress_API_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;

    // free the cl buffers
    delete buffer_dec_input;
    delete buffer_dec_output;
    delete buffer_size;
    delete buffer_status;

    // printme("Done with decompress \n");
    return decmpSizeIdx;
}

// This version of compression does overlapped execution between
// Kernel and Host. I/O operations between Host and Device are
// overlapped with Kernel execution between multiple compute units
uint32_t xil_zlib::compress(uint8_t* in, uint8_t* out, uint64_t input_size, uint32_t host_buffer_size) {
    //////printme("In compress \n");
    uint32_t block_size_in_kb = BLOCK_SIZE_IN_KB;
    uint32_t block_size_in_bytes = block_size_in_kb * 1024;
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;

    std::chrono::duration<double, std::nano> lz77_kernel_time_ns_1(0);
    std::chrono::duration<double, std::nano> huffman_kernel_time_ns_1(0);

    // For example: Input file size is 12MB and Host buffer size is 2MB
    // Then we have 12/2 = 6 chunks exists
    // Calculate the count of total chunks based on input size
    // This count is used to overlap the execution between chunks and file
    // operations

    uint32_t total_chunks = (input_size - 1) / host_buffer_size + 1;
    if (total_chunks < 2) overlap_buf_count = 1;

    // Find out the size of each chunk spanning entire file
    // For eaxmple: As mentioned in previous example there are 6 chunks
    // Code below finds out the size of chunk, in general all the chunks holds
    // HOST_BUFFER_SIZE except for the last chunk
    uint32_t sizeOfChunk[total_chunks];
    uint32_t blocksPerChunk[total_chunks];
    uint32_t idx = 0;

    for (uint64_t i = 0; i < input_size; i += host_buffer_size, idx++) {
        uint32_t chunk_size = host_buffer_size;
        if (chunk_size + i > input_size) {
            chunk_size = input_size - i;
        }
        // Update size of each chunk buffer
        sizeOfChunk[idx] = chunk_size;
        // Calculate sub blocks of size BLOCK_SIZE_IN_KB for each chunk
        // 2MB(example)
        // Figure out blocks per chunk
        uint32_t nblocks = (chunk_size - 1) / block_size_in_bytes + 1;
        blocksPerChunk[idx] = nblocks;
    }

    // Counter which helps in tracking
    // Output buffer index
    uint32_t outIdx = 0;

    // Track the lags of respective chunks for left over handling
    int chunk_flags[total_chunks];
    int cu_order[total_chunks];

    // Finished bricks
    int completed_bricks = 0;
    uint32_t input_index = 0;
    uint16_t tail_block_size = 0;
    int flag = 0;
    uint32_t lcl_cu = 0;
    bool flag_smallblk = false;
    uint8_t cunits = (uint8_t)C_COMPUTE_UNIT;
    uint8_t queue_idx = 0;
overlap:
    for (uint32_t brick = 0, itr = 0; brick < total_chunks;
         /*brick += C_COMPUTE_UNIT,*/ itr++, flag = !flag) {
        if (cunits > 1)
            queue_idx = flag * OVERLAP_BUF_COUNT;
        else
            queue_idx = flag;

        if (total_chunks > 2)
            lcl_cu = C_COMPUTE_UNIT;
        else
            lcl_cu = 1;

        if (brick + lcl_cu > total_chunks) lcl_cu = total_chunks - brick;

        for (uint32_t cu = 0; cu < lcl_cu; cu++) {
            chunk_flags[brick + cu] = flag;
            cu_order[brick + cu] = cu;

            // Wait for read events
            if (itr >= 2) {
                // Wait on current flag previous operation to finish
                m_q[queue_idx + cu]->finish();

                // Completed bricks counter
                completed_bricks++;

                uint32_t index = 0;
                uint32_t brick_flag_idx = brick - (C_COMPUTE_UNIT * overlap_buf_count - cu);

                //////printme("blocksPerChunk %d \n", blocksPerChunk[brick]);
                // Copy the data from various blocks in concatinated manner
                for (uint32_t bIdx = 0; bIdx < blocksPerChunk[brick_flag_idx]; bIdx++, index += block_size_in_bytes) {
                    uint32_t block_size = block_size_in_bytes;
                    if (index + block_size > sizeOfChunk[brick_flag_idx]) {
                        block_size = sizeOfChunk[brick_flag_idx] - index;
                    }

                    uint32_t compressed_size = (h_compressSize[cu][flag].data())[bIdx];
                    std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx * block_size_in_bytes],
                                compressed_size);
                    outIdx += compressed_size;
                }
            } // If condition which reads huffman output for 0 or 1 location

            // Figure out block sizes per brick
            uint32_t idxblk = 0;
            for (uint32_t i = 0; i < sizeOfChunk[brick + cu]; i += block_size_in_bytes) {
                uint32_t block_size = block_size_in_bytes;

                if (i + block_size > sizeOfChunk[brick + cu]) {
                    block_size = sizeOfChunk[brick + cu] - i;
                }

                if (block_size < MIN_BLOCK_SIZE) {
                    input_index = idxblk * block_size_in_bytes;
                    flag_smallblk = true;
                    tail_block_size = block_size;
                    sizeOfChunk[brick + cu] -= block_size;
                } else {
                    //////printme("sizeofChunk %d block_size %d cu %d \n",
                    /// sizeOfChunk[brick+cu], block_size, cu);
                    (h_blksize[cu][flag]).data()[idxblk++] = block_size;
                }
            }

            if (sizeOfChunk[brick + cu] != 0) {
                std::memcpy(h_buf_in[cu][flag].data(), &in[(brick + cu) * host_buffer_size], sizeOfChunk[brick + cu]);

                // Set kernel arguments
                int narg = 0;

                (compress_kernel)->setArg(narg++, *(buffer_input[cu][flag]));
                (compress_kernel)->setArg(narg++, *(buffer_lz77_output[cu][flag]));
                (compress_kernel)->setArg(narg++, *(buffer_compress_size[cu][flag]));
                (compress_kernel)->setArg(narg++, *(buffer_inblk_size[cu][flag]));
                (compress_kernel)->setArg(narg++, *(buffer_dyn_ltree_freq[cu][flag]));
                (compress_kernel)->setArg(narg++, *(buffer_dyn_dtree_freq[cu][flag]));
                (compress_kernel)->setArg(narg++, block_size_in_kb);
                (compress_kernel)->setArg(narg++, sizeOfChunk[brick + cu]);

                narg = 0;
                (huffman_kernel)->setArg(narg++, *(buffer_lz77_output[cu][flag]));
                (huffman_kernel)->setArg(narg++, *(buffer_dyn_ltree_freq[cu][flag]));
                (huffman_kernel)->setArg(narg++, *(buffer_dyn_dtree_freq[cu][flag]));
                (huffman_kernel)->setArg(narg++, *(buffer_zlib_output[cu][flag]));
                (huffman_kernel)->setArg(narg++, *(buffer_compress_size[cu][flag]));
                (huffman_kernel)->setArg(narg++, *(buffer_inblk_size[cu][flag]));
                (huffman_kernel)->setArg(narg++, block_size_in_kb);
                (huffman_kernel)->setArg(narg++, sizeOfChunk[brick + cu]);

                // Migrate memory - Map host to device buffers
                m_q[queue_idx + cu]->enqueueMigrateMemObjects(
                    {*(buffer_input[cu][flag]), *(buffer_inblk_size[cu][flag])}, 0 /* 0 means from host*/);
                m_q[queue_idx + cu]->finish();

                auto lz77_kernel_start = std::chrono::high_resolution_clock::now();
                // kernel write events update
                // LZ77 Compress Fire Kernel invocation
                m_q[queue_idx + cu]->enqueueTask(*compress_kernel);
                m_q[queue_idx + cu]->finish();

                auto lz77_kernel_end = std::chrono::high_resolution_clock::now();
                auto lz77_duration = std::chrono::duration<double, std::nano>(lz77_kernel_end - lz77_kernel_start);
                lz77_kernel_time_ns_1 += lz77_duration;

                auto huffman_kernel_start = std::chrono::high_resolution_clock::now();

                // Huffman Fire Kernel invocation
                m_q[queue_idx + cu]->enqueueTask(*huffman_kernel);
                m_q[queue_idx + cu]->finish();

                auto huffman_kernel_end = std::chrono::high_resolution_clock::now();
                auto huffman_duration =
                    std::chrono::duration<double, std::nano>(huffman_kernel_end - huffman_kernel_start);
                huffman_kernel_time_ns_1 += huffman_duration;

                m_q[queue_idx + cu]->enqueueMigrateMemObjects(
                    {*(buffer_zlib_output[cu][flag]), *(buffer_compress_size[cu][flag])}, CL_MIGRATE_MEM_OBJECT_HOST);
                m_q[queue_idx + cu]->finish();
            }
        } // Internal loop runs on compute units

        if (total_chunks > 2)
            brick += C_COMPUTE_UNIT;
        else
            brick++;

    } // Main overlap loop

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        m_q[i]->flush();
        m_q[i]->finish();
    }

    uint32_t leftover = total_chunks - completed_bricks;
    uint32_t stride = 0;

    if ((total_chunks < overlap_buf_count * C_COMPUTE_UNIT))
        stride = overlap_buf_count * C_COMPUTE_UNIT;
    else
        stride = total_chunks;

    // Handle leftover bricks
    for (uint32_t ovr_itr = 0, brick = stride - overlap_buf_count * C_COMPUTE_UNIT; ovr_itr < leftover;
         ovr_itr += C_COMPUTE_UNIT, brick += C_COMPUTE_UNIT) {
        lcl_cu = C_COMPUTE_UNIT;
        if (ovr_itr + lcl_cu > leftover) lcl_cu = leftover - ovr_itr;

        // Handle multiple bricks with multiple CUs
        for (uint32_t j = 0; j < lcl_cu; j++) {
            int cu = cu_order[brick + j];
            int flag = chunk_flags[brick + j];

            // Run over each block within brick
            uint32_t index = 0;
            uint32_t brick_flag_idx = brick + j;

            //////printme("blocksPerChunk %d \n", blocksPerChunk[brick]);
            // Copy the data from various blocks in concatinated manner
            for (uint32_t bIdx = 0; bIdx < blocksPerChunk[brick_flag_idx]; bIdx++, index += block_size_in_bytes) {
                uint32_t block_size = block_size_in_bytes;
                if (index + block_size > sizeOfChunk[brick_flag_idx]) {
                    block_size = sizeOfChunk[brick_flag_idx] - index;
                }

                uint32_t compressed_size = (h_compressSize[cu][flag].data())[bIdx];
                std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx * block_size_in_bytes], compressed_size);
                outIdx += compressed_size;
            }
        }
    }
    if (flag_smallblk) {
        uint8_t len_low = (uint8_t)tail_block_size;
        uint8_t len_high = (uint8_t)(tail_block_size >> 8);
        out[outIdx++] = 0x00;
        out[outIdx++] = len_low;
        out[outIdx++] = len_high;
        out[outIdx++] = ~len_low;
        out[outIdx++] = ~len_high;
        for (int i = 0; i < tail_block_size; i++) {
            out[outIdx++] = in[input_index + i];
        }
    }

    std::chrono::duration<double, std::nano> temp_kernel_time_ns_1(0);
    if (lz77_kernel_time_ns_1.count() > huffman_kernel_time_ns_1.count())
        temp_kernel_time_ns_1 = lz77_kernel_time_ns_1;
    else
        temp_kernel_time_ns_1 = huffman_kernel_time_ns_1;

    float throughput_in_mbps_1 = (float)input_size * 1000 / temp_kernel_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    // zlib special block based on Z_SYNC_FLUSH
    out[outIdx++] = 0x01;
    out[outIdx++] = 0x00;
    out[outIdx++] = 0x00;
    out[outIdx++] = 0xff;
    out[outIdx++] = 0xff;
    return outIdx;
} // Overlap end

// This compress function transfer full input file to DDR before kernel call
uint32_t xil_zlib::compressFull(uint8_t* in, uint8_t* out, uint64_t input_size) {
    uint32_t outputSize = input_size;

    std::vector<uint32_t, aligned_allocator<uint32_t> > h_checksum_data(1);

    h_checksum_data.resize(sizeof(uint32_t));

    bool checksum_type = true;
#ifdef GZIP_MODE
    h_checksum_data.data()[0] = ~0;
    checksum_type = true;
#else
    h_checksum_data.data()[0] = 1;
    checksum_type = false;
#endif

    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);

    uint32_t host_buffer_size = HOST_BUFFER_SIZE;

    uint32_t outIdx = 0;
    for (uint32_t inIdx = 0; inIdx < input_size; inIdx += host_buffer_size) {
        uint32_t buf_size = host_buffer_size;
        uint32_t hostChunk_cu = host_buffer_size;

        if (inIdx + buf_size > input_size) hostChunk_cu = input_size - inIdx;

        // Copy input data from in to host buffer based on the input_size
        std::memcpy(h_buf_in[0][0].data(), &in[inIdx], hostChunk_cu);

        buffer_checksum_data = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint32_t),
                                              h_checksum_data.data());

        // Set kernel arguments
        uint32_t narg = 0;
        compress_kernel->setArg(narg++, *buffer_input[0][0]);
        compress_kernel->setArg(narg++, *buffer_zlib_output[0][0]);
        compress_kernel->setArg(narg++, *buffer_compress_size[0][0]);
        compress_kernel->setArg(narg++, *buffer_checksum_data);
        compress_kernel->setArg(narg++, hostChunk_cu);
        compress_kernel->setArg(narg++, checksum_type);

        // Migrate memory - Map host to device buffers
        m_q[0]->enqueueMigrateMemObjects({*(buffer_input[0][0]), *(buffer_checksum_data), *(buffer_zlib_output[0][0]),
                                          *(buffer_compress_size[0][0])},
                                         0 /* 0 means from host*/);
        m_q[0]->finish();

        // Measure kernel execution time
        auto kernel_start = std::chrono::high_resolution_clock::now();

        // Fire kernel execution
        m_q[0]->enqueueTask(*compress_kernel);
        // Wait till kernels complete
        m_q[0]->finish();

        auto kernel_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::nano>(kernel_end - kernel_start);
        kernel_time_ns_1 += duration;

        // Migrate memory - Map device to host buffers
        m_q[0]->enqueueMigrateMemObjects(
            {*(buffer_checksum_data), *(buffer_zlib_output[0][0]), *(buffer_compress_size[0][0])},
            CL_MIGRATE_MEM_OBJECT_HOST);
        m_q[0]->finish();

        uint32_t compressedSize = h_compressSize[0][0].data()[0];
        std::memcpy(&out[outIdx], h_buf_zlibout[0][0].data(), compressedSize);
        outIdx += compressedSize;

        h_checksum_data[0] = ~h_checksum_data[0];

        // Buffer deleted
        delete buffer_checksum_data;
    }

    // checksum value
    crc = h_checksum_data[0];
    if (checksum_type) crc = ~crc;

    float throughput_in_mbps_1 = (float)input_size * 1000 / kernel_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;

    return outIdx;
}
