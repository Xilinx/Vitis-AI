#include "zlibDriver.hpp"
#include <time.h>

#ifdef ENABLE_INFLATE_XRM

// Provide XRM CU Instance
xrmStruct* zlibDriver::getXrmCuInstance(z_streamp strm, const std::string& kernel_name) {
    xrmStruct* ret;
    auto iterator = this->m_xrmMapObj.find(strm);
    if (iterator != this->m_xrmMapObj.end()) {
        // Return if it exists
        ret = iterator->second;
    } else {
        // Create new xrmData structure for
        // current stream pointer
        xrmStruct* xrmData = new xrmStruct;
        if (xrmData == nullptr) {
            ret = nullptr;
        } else {
            // Insert the stream,xrmdataobj key/value pairs in map
            this->m_xrmMapObj.insert(std::pair<z_streamp, xrmStruct*>(strm, xrmData));

            //// Query XRM
            memset(&(xrmData->cuProp), 0, sizeof(xrmCuProperty));
            memset(&(xrmData->cuRes), 0, sizeof(xrmCuResource));

            // Assign kernel name to look for CU allocation
            strncpy(xrmData->cuProp.kernelName, kernel_name.c_str(), kernel_name.size() + 1);
            strcpy(xrmData->cuProp.kernelAlias, "");

            // Open device in exclusive mode
            xrmData->cuProp.devExcl = false;
            // % of load
            xrmData->cuProp.requestLoad = 100;
            xrmData->cuProp.poolId = 0;

            // Block until get exclusive access to the CU
            int err = xrmCuBlockingAlloc(this->m_ctx, &(xrmData->cuProp), 2, &(xrmData->cuRes));
            if (err != 0) {
                std::cout << "xrmCuAlloc: Failed to allocate CU \n" << std::endl;
#if (VERBOSE_LEVEL >= 1)
                std::cout << "xrmCuAlloc: Failed to allocate CU \n" << std::endl;
#endif
                ret = nullptr;
            } else {
                ret = xrmData;
            }
        }
    }
    return ret;
}
#endif

void zlibDriver::releaseZlibCU(z_streamp strm) {
    if (this->m_xlz) {
        delete this->m_xlz;
        this->m_xlz = nullptr;
    }

#ifdef ENABLE_INFLATE_XRM
    // XRM iterator
    auto xrmIter = this->m_xrmMapObj.find(strm);
    if (xrmIter != this->m_xrmMapObj.end()) {
        if (this->m_ctx != nullptr) {
            xrmCuRelease(this->m_ctx, &(xrmIter->second->cuRes));
        }
        delete xrmIter->second;
        // Remove <key,value> pair from map
        this->m_xrmMapObj.erase(xrmIter);
    }
#endif
}

bool zlibDriver::allocateCU(z_streamp strm) {
    std::string kernel;
    bool ret = false;
    bool enable_xrm = false;
    if (this->m_flow == XILINX_DEFLATE)
        kernel = compress_kernel_names[1];
    else
        kernel = stream_decompress_kernel_name[2];

    if (this->m_flow == XILINX_INFLATE) {
#ifdef ENABLE_INFLATE_XRM
        enable_xrm = true;
#endif
    }

#ifdef ENABLE_INFLATE_XRM
    if (enable_xrm) {
        // Created relavant XRM object
        xrmStruct* xrmData = getXrmCuInstance(strm, kernel);
        if (xrmData == nullptr) {
            ret = true;
        } else {
            this->m_kernel_instance = xrmData->cuRes.instanceName;
            this->m_deviceid = this->m_ndevinfo[xrmData->cuRes.deviceId];
            this->m_bank = xrmData->cuRes.cuId % C_COMPUTE_UNIT;
            this->m_context = this->m_contexts[xrmData->cuRes.deviceId];
            this->m_program = this->m_programs[xrmData->cuRes.deviceId];
            //        this->m_cuid = (this->m_kernel_instance[this->m_kernel_instance.size() - 1] - '0') - 1;
        }
    } else {
#endif
        srand(time(0));
        std::string nKernel = std::to_string(1 + (rand() % D_COMPUTE_UNIT));
        if (this->m_flow == XILINX_INFLATE) {
            this->m_kernel_instance = kernel + "_" + nKernel;
        } else {
            this->m_kernel_instance = kernel;
        }
        std::random_device randDevice;
        std::uniform_int_distribution<> range(0, (this->m_ndevinfo.size() - 1));
        uint8_t devNo = range(randDevice);
        this->m_context = this->m_contexts[devNo];
        this->m_program = this->m_programs[devNo];
        this->m_deviceid = this->m_ndevinfo[devNo];
#ifdef ENABLE_INFLATE_XRM
    }
#endif
    return ret;
}

// Constructor
zlibDriver::zlibDriver(z_streamp strm,
                       const cl_context_properties& platform_id,
                       const std::string& u50_xclbin,
                       const int flow,
                       std::vector<cl::Device>& devices,
                       std::vector<cl::Context*>& deviceContext,
                       std::vector<cl::Program*>& deviceProgram,
                       bool _init) {
    this->m_ndevinfo = devices;
    this->m_platformid = platform_id;
    this->m_u50_xclbin = u50_xclbin;
    this->m_flow = flow;
    this->m_contexts = deviceContext;
    this->m_programs = deviceProgram;

    // Update stream pointer
    struct_update(strm, flow);
    if (_init) init(strm, flow);
}

void zlibDriver::init(z_streamp strm, const int flow) {
    if (initialized) return;
    initialized = true;
    if (flow == XILINX_DEFLATE) {
        // Deflate output / input size
        this->m_deflateOutput = new unsigned char[2 * this->m_bufSize];
        this->m_deflateInput = new unsigned char[2 * this->m_bufSize];
    } else {
        this->m_inflateOutput = new unsigned char[2 * m_bufSize];
        this->m_inflateInput = new unsigned char[2 * m_bufSize];
    }

#ifdef ENABLE_INFLATE_XRM
    // XRM Context Creation
    this->m_ctx = (xrmContext*)xrmCreateContext(XRM_API_VERSION_1);
    if (this->m_ctx == nullptr) {
#if (VERBOSE_LEVEL >= 1)
        std::cout << "Failed to create XRM Context " << std::endl;
#endif
        this->m_status = true;
    } else {
#endif

        bool err;
        err = this->allocateCU(strm);
        if (err) this->m_status = true;
        err = this->getZlibInstance();
        if (err) this->m_status = true;

#ifdef ENABLE_INFLATE_XRM
    }
#endif
}

// Destructor
zlibDriver::~zlibDriver() {
    if (this->m_deflateOutput != nullptr) {
        delete m_deflateOutput;
    }

    if (this->m_deflateInput != nullptr) {
        delete m_deflateInput;
    }

    if (this->m_inflateOutput != nullptr) {
        delete m_inflateOutput;
    }

    if (this->m_inflateInput != nullptr) {
        delete m_inflateInput;
    }
#ifdef ENABLE_INFLATE_XRM
    if (this->m_ctx != nullptr) {
        delete m_ctx;
        this->m_ctx = nullptr;
    }
#endif
}

// Structure update
void zlibDriver::struct_update(z_streamp strm, int flow) {
    this->m_info.in_buf = strm->next_in;
    this->m_info.out_buf = strm->next_out;
    this->m_info.input_size = strm->avail_in;
    this->m_info.output_size = strm->avail_out;
    this->m_info.adler32 = strm->adler;
    if (flow == XILINX_DEFLATE) {
        this->m_info.level = strm->state->level;
        this->m_info.strategy = strm->state->strategy;
        this->m_info.window_bits = strm->state->w_bits;
    }
    this->m_flow = flow;
}

// Get method for status
bool zlibDriver::getStatus(void) {
    return m_status;
}

// Proivde xfZlib object
bool zlibDriver::getZlibInstance(void) {
    bool ret = false;
    if (this->m_flow == XILINX_DEFLATE) {
        if (this->m_xlz == nullptr) {
            this->m_xlz = new xfZlib(this->m_u50_xclbin.c_str(), false, COMP_ONLY, this->m_context, this->m_program,
                                     this->m_deviceid, XILINX_ZLIB, this->m_bank);
            if (this->m_xlz->error_code()) {
                ret = true;
            }
        }
    } else if (this->m_flow == XILINX_INFLATE) {
        this->m_xlz = new xfZlib(this->m_u50_xclbin.c_str(), false, DECOMP_ONLY, this->m_context, this->m_program,
                                 this->m_deviceid, XILINX_ZLIB);
        if (this->m_xlz->error_code()) {
            ret = true;
        }
    }
    return ret;
}

// Call to HW Deflate
bool zlibDriver::xilinxHwDeflate(z_streamp strm, int flush) {
    lock();
    if (this->m_xlz == nullptr) {
        bool err;
        err = this->allocateCU(strm);
        if (err) this->m_status = true;
        err = this->getZlibInstance();
        if (err) this->m_status = true;
    }

    bool last_data = false;
    bool ret_flag = false;
    bool breakLoop = false;
    bool last_buffer = false;
    size_t enbytes = 0;
    uint32_t adler32 = 0;
    uint32_t cpyAvailOut = strm->avail_out;

    while (((this->m_info.input_size != 0) || flush) && (strm->avail_out != 0)) {
        this->m_xlz->set_checksum(strm->adler);
        size_t inputSize = this->m_info.input_size;

        enbytes = this->m_xlz->deflate_buffer(&this->m_info, flush, inputSize, last_data, this->m_kernel_instance);
        this->m_outputSize += enbytes;
        this->m_info.input_size = inputSize;

        adler32 = this->m_xlz->get_checksum();

        if (this->m_outputSize > 0) {
            strm->avail_out = strm->avail_out - this->m_outputSize;
            strm->total_out += this->m_outputSize;
            strm->next_out += this->m_outputSize;
            this->m_outputSize = 0;

            if (last_data) break;
        }
        if (this->m_outputSize == 0) {
            if (last_data) break;
        }
    }
    strm->avail_in = this->m_info.input_size;
    strm->adler = adler32;
    unlock();

    if (last_data && strm->avail_out != 0) {
        return true;
    } else {
        return false;
    }
}

// Call to HW Inflate
bool zlibDriver::xilinxHwInflate(z_streamp strm, int flush) {
    size_t enbytes = 0;

    bool last_data = false;

    if (m_info.input_size >= 9) {
        long int lastblock = 0;
        std::memcpy((unsigned char*)(&lastblock), m_info.in_buf + (m_info.input_size - 9), 5);
        if (lastblock == 0xffff000001) last_block = true;
    } else {
        //        last_block = true;
    }
    bool last_buffer = false;
    while (((this->m_info.input_size != 0) || last_block) && (strm->avail_out != 0)) {
        if (last_block && this->m_info.input_size == 0) last_buffer = true;
        size_t inputSize = this->m_info.input_size;
        size_t outputSize = m_bufSize;

        enbytes = this->m_xlz->inflate_buffer(&this->m_info, last_block, inputSize, outputSize, last_data, last_buffer,
                                              this->m_kernel_instance);
        this->m_outputSize += enbytes;
        this->m_info.input_size = inputSize;
        if (this->m_outputSize > 0) {
            strm->avail_out = strm->avail_out - this->m_outputSize;
            strm->total_out += this->m_outputSize;
            strm->next_out += this->m_outputSize;
            this->m_outputSize = 0;

            if (last_data) break;
        }
        if (this->m_outputSize == 0) {
            if (last_data) break;
        }
    }

    strm->avail_in = m_info.input_size;

    if (last_data && strm->avail_out != 0) {
        return true;
    } else
        return false;
}

// Call to HW Compress
uint64_t zlibDriver::xilinxHwCompress(z_streamp strm, int val) {
    return this->xilinxHwDeflate(strm, val);
}

// Call to HW decompress
uint64_t zlibDriver::xilinxHwUncompress(z_streamp strm, int cu) {
#ifdef ENABLE_INFLATE_XRM
    // Call decompress API
    this->m_info.output_size = this->m_xlz->decompress(this->m_info.in_buf, this->m_info.out_buf,
                                                       this->m_info.input_size, this->m_info.output_size, this->m_cuid);
    if (this->m_info.output_size == 0) {
#if (VERBOSE_LEVEL >= 1)
        std::cout << "Decompression Failed " << std::endl;
#endif
    }
#endif
    return this->m_info.output_size;
}

void zlibDriver::copyData(z_streamp strm) {
    if (m_copydecompressData == false) {
        return;
    }
    struct_update(strm, XILINX_INFLATE);
    assert((m_info.input_size + m_inputSize) <= (2 * 1024 * 1024));
    std::memcpy(m_inflateInput + m_inputSize, m_info.in_buf, m_info.input_size);
    m_inputSize += m_info.input_size;
}
