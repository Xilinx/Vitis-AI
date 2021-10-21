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
#ifndef _XFCOMPRESSION_GZIPOCLHOST_HPP_
#define _XFCOMPRESSION_GZIPOCLHOST_HPP_

#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <thread>
#include "xcl2.hpp"
#include <sys/stat.h>
#include <random>
#include <new>
#include <map>
#include <future>
#include <queue>
#include <assert.h>
#include <syslog.h>
#include "compressBase.hpp"
#include "gzipBase.hpp"

const int gz_max_literal_count = 4096;

#define PARALLEL_ENGINES 8

#ifndef C_COMPUTE_UNIT
#define C_COMPUTE_UNIT 1
#endif

#ifndef D_COMPUTE_UNIT
#define D_COMPUTE_UNIT 1
#endif

#define MAX_CCOMP_UNITS C_COMPUTE_UNIT
#define MAX_DDCOMP_UNITS D_COMPUTE_UNIT

// Default block size
#define BLOCK_SIZE_IN_KB 32

// Input and output buffer size
#define INPUT_BUFFER_SIZE (8 * MEGA_BYTE)
#define OUTPUT_BUFFER_SIZE (16 * MEGA_BYTE)

// zlib max cr limit
#define MAX_CR 10

// buffer count for data in in decompression data movers
#define DIN_BUFFERCOUNT 2
#define DOUT_BUFFERCOUNT 4

// Maximum host buffer used to operate
// per kernel invocation
#define MAX_HOST_BUFFER_SIZE (2 * 1024 * 1024)
#define HOST_BUFFER_SIZE (1024 * 1024)

// Zlib method information
constexpr auto DEFLATE_METHOD = 8;

constexpr auto page_aligned_mem = (1 << 21);

uint64_t get_file_size(std::ifstream& file);

constexpr auto c_clOutOfResource = -5;
constexpr auto c_clinvalidbin = -42;
constexpr auto c_clinvalidvalue = -30;
constexpr auto c_clinvalidprogram = -44;
constexpr auto c_clOutOfHostMemory = -6;
constexpr auto c_headermismatch = 404;
constexpr auto c_installRootDir = "/opt/xilinx/apps/";
constexpr auto c_hardXclbinPath = "zlib/xclbin/u50_gen3x16_xdma_201920_3.xclbin";
constexpr auto c_hardFullXclbinPath = "zlib/xclbin/u50_gen3x16_xdma_201920_3_full.xclbin";

// Structure to hold
// libzso - deflate/inflate info
typedef struct {
    uint8_t* in_buf;
    uint8_t* out_buf;
    uint64_t input_size;
    uint64_t output_size;
    int level;
    int strategy;
    int window_bits;
    uint32_t adler32;
    int flush;
    std::string u50_xclbin;
} xlibData;

#if (VERBOSE_LEVEL >= 3)
#define ZOCL_CHECK(error, call, eflag, expected)                                    \
    call;                                                                           \
    if (error != CL_SUCCESS) {                                                      \
        if (error == expected) {                                                    \
            std::cout << __FILE__ << ":" << __LINE__ << " OPENCL API --> ";         \
            std::cout << #call;                                                     \
            std::cout << ", RESULT: -->  ";                                         \
            std::cout << error_string(error);                                       \
            std::cout << ", EXPECTED: --> ";                                        \
            std::cout << error_string(expected) << std::endl;                       \
            eflag = error;                                                          \
        } else {                                                                    \
            std::cout << __FILE__ << ":" << __LINE__ << " OPENCL API --> ";         \
            std::cout << #call;                                                     \
            std::cout << "Unexpected Error \n" << error_string(error) << std::endl; \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }
#else
#define ZOCL_CHECK(error, call, eflag, expected)                                    \
    call;                                                                           \
    if (error != CL_SUCCESS) {                                                      \
        if (error == expected)                                                      \
            eflag = error;                                                          \
        else {                                                                      \
            std::cout << "Unexpected Error \n" << error_string(error) << std::endl; \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }
#endif

#if (VERBOSE_LEVEL >= 3)
#define ZOCL_CHECK_2(error, call, eflag, expected_1, expected_2)                    \
    call;                                                                           \
    if (error != CL_SUCCESS) {                                                      \
        if ((error == expected_1)) {                                                \
            std::cout << __FILE__ << ":" << __LINE__ << " OPENCL API --> ";         \
            std::cout << #call;                                                     \
            std::cout << ", RESULT: -->  ";                                         \
            std::cout << error_string(error);                                       \
            std::cout << ", EXPECTED: --> ";                                        \
            std::cout << error_string(expected_1) << std::endl;                     \
            eflag = error;                                                          \
        } else if (error == expected_2) {                                           \
            std::cout << __FILE__ << ":" << __LINE__ << " OPENCL API --> ";         \
            std::cout << #call;                                                     \
            std::cout << ", RESULT: -->  ";                                         \
            std::cout << error_string(error);                                       \
            std::cout << ", EXPECTED: --> ";                                        \
            std::cout << error_string(expected_2) << std::endl;                     \
            eflag = error;                                                          \
        } else {                                                                    \
            std::cout << __FILE__ << ":" << __LINE__ << " OPENCL API --> ";         \
            std::cout << #call;                                                     \
            std::cout << "Unexpected Error \n" << error_string(error) << std::endl; \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }
#else
#define ZOCL_CHECK_2(error, call, eflag, expected_1, expected_2)                    \
    call;                                                                           \
    if (error != CL_SUCCESS) {                                                      \
        if ((error == expected_1) || (error == expected_2))                         \
            eflag = error;                                                          \
        else {                                                                      \
            std::cout << "Unexpected Error \n" << error_string(error) << std::endl; \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }
#endif

#define ERROR_STATUS(call)                                      \
    if (call) {                                                 \
        std::cerr << "\nFailed to create object " << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }

// Aligned allocator for ZLIB
template <typename T>
struct zlib_aligned_allocator {
    using value_type = T;
    T* allocate(std::size_t num) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, page_aligned_mem, num * sizeof(T))) throw std::bad_alloc();

        // madvise is a system call to allocate
        // huge pages. By allocating huge pages
        // for memory allocation improves overall
        // time by reduction in page initialization
        // in next step.
        madvise(ptr, num, MADV_HUGEPAGE);

        T* array = reinterpret_cast<T*>(ptr);

        // Write a value in each virtual memory page to force the
        // materialization in physical memory to avoid paying this price later
        // during the first use of the memory
        for (std::size_t i = 0; i < num; i += (page_aligned_mem)) array[i] = 0;
        return array;
    }

    // Dummy construct which doesnt initialize
    template <typename U>
    void construct(U* ptr) noexcept {
        // Skip the default construction
    }

    void deallocate(T* p, std::size_t num) { free(p); }

    void deallocate(T* p) { free(p); }
};

void event_compress_cb(cl_event event, cl_int cmd_status, void* data);
void event_checksum_cb(cl_event event, cl_int cmd_status, void* data);
class memoryManager;
/**
 *  gzipOCLHost class. Class containing methods for Zlib
 * compression and decompression to be executed on host side.
 */
class gzipOCLHost : public gzipBase {
   public:
    /**
     * @brief This method does the overlapped execution of compression
     * where data transfers and kernel computation are overlapped
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param input_size input size
     * @param host_buffer_size buffer size for kernel
     * @param cu comput unit name
     */
    size_t compress_buffer(
        uint8_t* in, uint8_t* out, size_t input_size, uint32_t host_buffer_size = HOST_BUFFER_SIZE, int cu = 0);

    /**
     * @brief This method does the overlapped execution of decompression
     * where data transfers and kernel computation are overlapped
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param input_size input size
     * @param last_data last compressed output buffer
     * @param last_buffer last input buffer
     * @param cu comput unit name
     */
    size_t deflate_buffer(
        uint8_t* in, uint8_t* out, size_t& input_size, bool& last_data, bool last_buffer, const std::string& cu);

    /**
     * @brief This method does serial execution of decompression
     * where data transfers and kernel execution in serial manner
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param actual_size input size
     * @param cu_run compute unit number
     */

    size_t decompress(uint8_t* in, uint8_t* out, size_t actual_size, size_t max_outbuf_size, int cu_run = 0);

    /**
     * @brief This API does end to end compression in a single call
     *
     *
     *
     * @param in input byte sequence
     * @param out output byte sequence
     * @param input_size input size
     */
    // Default values
    // Level    = Fast method
    // Strategy = Default
    // Window   = 32K (Max supported)
    size_t compress(
        uint8_t* in, uint8_t* out, size_t input_size, int cu, int level = 1, int strategy = 0, int window_bits = 15);

    /**
     * @brief This method does file operations and invokes compress API which
     * internally does zlib compression on FPGA in overlapped manner
     *
     * @param inFile_name input file name
     * @param outFile_name output file name
     * @param actual_size input size
     */

    uint64_t compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, int cu_run = 0);

    /**
     * @brief This method  does file operations and invokes decompress API which
     * internally does zlib decompression on FPGA in overlapped manner
     *
     * @param inFile_name input file name
     * @param outFile_name output file name
     * @param input_size input size
     * @param cu_run compute unit number
     */

    uint32_t decompress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, int cu_run);

    uint64_t get_event_duration_ns(const cl::Event& event);

    /**
     * @brief Constructor responsible for creating various host/device buffers.
     * This gets context created already with it so skips ocl platform, device
     * and context creation as part of it.
     *
     */
    gzipOCLHost(const std::string& binaryFile,
                uint8_t cd_flow,
                cl::Context* context,
                cl::Program* program,
                const cl::Device& device,
                enum design_flow dflow = XILINX_GZIP,
                const int bank_id = 0);

    gzipOCLHost(enum State flow,
                const std::string& binaryFileName,
                const std::string& uncompFileName,
                const bool is_Seq = false,
                uint8_t device_id = 0,
                bool enable_profile = false,
                uint8_t decKerType = FULL,
                uint8_t dflow = XILINX_GZIP,
                bool freeRunKernel = false);

    /**
     * @brief Constructor responsible for creating various host/device buffers.
     *
     */
    gzipOCLHost(const std::string& binaryFile,
                uint8_t c_max_cr = MAX_CR,
                uint8_t cd_flow = BOTH,
                uint8_t device_id = 0,
                uint8_t profile = 0,
                uint8_t d_type = DYNAMIC,
                enum design_flow dflow = XILINX_GZIP);

    /**
     * @brief OpenCL setup initialization
     * @param binaryFile
     *
     */
    int init(const std::string& binaryFile, uint8_t dtype);

    /**
     * @brief OpenCL setup release
     *
     */
    void release();

    /**
     * @brief Release host/device memory
     */
    ~gzipOCLHost();

    /**
     * @brief error_code of a OpenCL call
     */
    int error_code(void);

    /**
     * @brief returns check sum value (CRC32/Adler32)
     */
    uint32_t get_checksum(void);

    /**
     * @brief set check sum value (CRC32/Adler32)
     */
    void set_checksum(uint32_t cval);
    uint64_t compressEngineOverlap(uint8_t* in,
                                   uint8_t* out,
                                   uint64_t actual_size,
                                   int cu = 0,
                                   int level = 1,
                                   int strategy = 0,
                                   int window_bits = 15,
                                   uint32_t* checksum = nullptr) override;
    uint64_t compressEngineSeq(uint8_t* in,
                               uint8_t* out,
                               uint64_t actual_size,
                               int level = 1,
                               int strategy = 0,
                               int window_bits = 15,
                               uint32_t* checksum = nullptr) override;
    uint64_t decompressEngine(
        uint8_t* in, uint8_t* out, uint64_t input_size, uint64_t max_output_size, int cu = 0) override;
    uint64_t decompressEngineSeq(
        uint8_t* in, uint8_t* out, uint64_t input_size, uint64_t max_output_size, int cu = 0) override;

    /**
     * @brief get compress kernel name
     */
    std::string getCompressKernel(int index);

    /**
     * @brief get decompress kernel name
     */
    std::string getDeCompressKernel(int index);

   private:
    void _enqueue_writes(uint32_t bufSize, uint8_t* in, uint32_t inputSize, int cu);
    void _enqueue_reads(uint32_t bufSize, uint8_t* out, uint32_t* decompSize, int cu, uint32_t max_outbuf);
    size_t add_header(uint8_t* out, int level, int strategy, int window_bits);
    size_t add_footer(uint8_t* out, size_t compressSize);
    void release_dec_buffers(void);

    enum m_threadStatus { IDLE, RUNNING };
    bool m_isGzip = 0; // Zlib=0, Gzip=1
    uint32_t m_checksum = 0;
    m_threadStatus m_checksumStatus = IDLE;
    std::thread* m_thread_checksum;
    bool m_lastData = false;
    uint16_t m_cBlkSize = (1 << 15);

    void gzip_headers(std::string& inFile, std::ofstream& outFile, uint8_t* zip_out, uint32_t enbytes);
    void zlib_headers(std::string& inFile_name, std::ofstream& outFile, uint8_t* zip_out, uint32_t enbytes);

    void generate_checksum(uint8_t* in, size_t input_size);
    uint32_t calculate_crc32(uint8_t* in, size_t input_size);
    uint32_t calculate_adler32(uint8_t* in, size_t input_size);
    std::future<uint32_t> m_future_checksum;
    enum design_flow m_zlibFlow;

    uint8_t m_cdflow;
    bool m_isProfile = 0;
    uint8_t m_deviceid = 0;
    uint8_t m_max_cr = MAX_CR;
    int m_err_code = 0;
    int m_derr_code = false;
    std::string m_infile;
    uint32_t m_kidx;
    memoryManager* m_memoryManager;
    uint8_t m_pending = 0;
    bool m_islibz = false;
    bool m_skipContextProgram = false;
    cl::Device m_device;
    cl_context_properties m_platform;
    cl::Context* m_context = nullptr;
    cl::Program* m_program = nullptr;
    cl::CommandQueue* m_def_q = nullptr;
#ifndef FREE_RUNNING_KERNEL
    cl::CommandQueue* m_q_dec[D_COMPUTE_UNIT] = {nullptr};
#endif
    cl::CommandQueue* m_q_rd[D_COMPUTE_UNIT] = {nullptr};
    cl::CommandQueue* m_q_rdd[D_COMPUTE_UNIT] = {nullptr};
    cl::CommandQueue* m_q_wr[D_COMPUTE_UNIT] = {nullptr};
    cl::CommandQueue* m_q_wrd[D_COMPUTE_UNIT] = {nullptr};

    // Kernel declaration
    cl::Kernel* m_checksumKernel = nullptr;
    cl::Kernel* m_compressFullKernel = nullptr;
    cl::Kernel* decompress_kernel[D_COMPUTE_UNIT] = {nullptr};
    cl::Kernel* data_writer_kernel[D_COMPUTE_UNIT] = {nullptr};
    cl::Kernel* data_reader_kernel[D_COMPUTE_UNIT] = {nullptr};
    cl::Kernel* datamover_kernel = nullptr;
    // Checksum related
    std::vector<uint32_t, zlib_aligned_allocator<uint32_t> > h_buf_checksum_data;

    // Decompression Related
    std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > h_dbuf_in[MAX_DDCOMP_UNITS];
    std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > h_dbuf_zlibout[MAX_DDCOMP_UNITS];
    std::vector<uint32_t, zlib_aligned_allocator<uint32_t> > h_dcompressSize[MAX_DDCOMP_UNITS];
    std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > h_dbufstream_in[DIN_BUFFERCOUNT];
    std::vector<uint8_t, zlib_aligned_allocator<uint8_t> > h_dbufstream_zlibout[DOUT_BUFFERCOUNT + 1];
    std::vector<uint32_t, zlib_aligned_allocator<uint32_t> > h_dcompressSize_stream[DOUT_BUFFERCOUNT];
    std::vector<uint32_t, aligned_allocator<uint32_t> > h_dcompressStatus;

    // Checksum Device buffers
    cl::Buffer* buffer_checksum_data;

    // Decompress Device Buffers
    cl::Buffer* buffer_dec_input[DIN_BUFFERCOUNT] = {nullptr};
    cl::Buffer* buffer_dec_zlib_output[DOUT_BUFFERCOUNT + 1] = {nullptr};
    cl::Buffer* buffer_dec_compress_size[MAX_DDCOMP_UNITS];

    // Kernel names
    std::map<std::string, int> compressKernelMap;
    std::map<std::string, int> decKernelMap;
    std::map<std::string, int> lz77KernelMap;
    std::map<std::string, int> huffKernelMap;
    std::mutex callBackMutex;
};

struct buffers {
    uint8_t* h_buf_in;
    uint8_t* h_buf_zlibout;
    uint32_t* h_compressSize;
    cl::Buffer* buffer_input;
    cl::Buffer* buffer_zlib_output;
    cl::Buffer* buffer_compress_size;
    uint32_t input_size;
    uint32_t store_size;
    uint32_t allocated_size;
    cl::Event wr_event;
    cl::Event rd_event;
    cl::Event cmp_event;
    cl::Event chk_wr_event;
    cl::Event chk_event;
    bool compress_finish;
    bool checksum_finish;
    bool is_finish() { return compress_finish && checksum_finish; }
    void reset() {
        compress_finish = false;
        checksum_finish = false;
    }
    buffers() {
        reset();
        input_size = 0;
        store_size = 0;
        allocated_size = 0;
    }
};

class memoryManager {
   public:
    memoryManager(uint8_t max_buffers, cl::Context* context, int bank = 0);
    ~memoryManager();
    buffers* createBuffer(size_t size);
    buffers* getBuffer();
    buffers* peekBuffer();
    buffers* getLastBuffer();
    bool isPending();

   private:
    uint8_t m_bankId = 0;
    void release();
    std::queue<buffers*> freeBuffers;
    std::queue<buffers*> busyBuffers;
    uint8_t bufCount;
    uint8_t maxBufCount;
    cl::Context* mContext;
    buffers* lastBuffer = NULL;

    cl::Buffer* getBuffer(cl_mem_flags flag, size_t size, uint8_t* host_ptr);
    void release(buffers* buffer);
    bool is_sw_emulation() {
        bool ret = false;
        char* xcl_mode = getenv("XCL_EMULATION_MODE");
        if ((xcl_mode != NULL) && !strcmp(xcl_mode, "sw_emu")) {
            ret = true;
        }
        return ret;
    }
};
#endif // _XFCOMPRESSION_ZLIB_HPP_
