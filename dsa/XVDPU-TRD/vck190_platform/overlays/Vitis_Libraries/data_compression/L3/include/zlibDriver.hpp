#include "zlib.hpp"
#include "deflate.h"
using namespace xf::compression;
#ifndef DEFLATE_BUFFER
#define DEFLATE_BUFFER (1024 * 1024)
#endif
enum m_enumMode { XILINX_DEFLATE = 1, XILINX_INFLATE };
#ifdef ENABLE_INFLATE_XRM
#include <xrm.h>
typedef struct {
    xrmCuProperty cuProp;
    xrmCuResource cuRes;
} xrmStruct;
#endif
class zlibDriver {
   public:
    bool xilinxHwDeflate(z_streamp s, int flush);
    uint64_t xilinxHwCompress(z_streamp s, int level);
    uint64_t xilinxHwUncompress(z_streamp s, int cu);
    zlibDriver(z_streamp s,
               const cl_context_properties& platform_id,
               const std::string& m_u50_xclbin,
               int flow,
               std::vector<cl::Device>& ndevices,
               std::vector<cl::Context*>& deviceContext,
               std::vector<cl::Program*>& deviceProgram,
               bool init = true);
    ~zlibDriver();
    bool getStatus();
    void struct_update(z_streamp s, int flow);
    bool getZlibInstance(void);
    bool allocateCU(z_streamp strm);
    bool getErrStatus(void) { return m_status; }
    xfZlib* getZlibPtr(void) { return m_xlz; }
    void releaseZlibCU(z_streamp strm);

    bool isDecompressEligible() { return m_decompressEligible; }
    void setDecompressEligibility(bool val) { m_decompressEligible = val; }

    bool xilinxHwInflate(z_streamp s, int flush);
    void copyData(z_streamp strm);

   private:
    // OpenCL Context
    cl::Context* m_context = nullptr;
    cl::Program* m_program = nullptr;
    // std::map container to hold
    // zstream structure pointers and zlibDriver Class Object
    // Mapping (Driver Class)
    std::map<z_streamp, zlibDriver*> m_driverMapObj;
    int m_flow = XILINX_DEFLATE;
    xfZlib* m_xlz = nullptr;
    int m_cuid = 0;
    int m_bank = 0;
    cl::Device m_deviceid;
    cl_context_properties m_platformid;
    std::string m_u50_xclbin;
    std::string m_kernel_instance = compress_kernel_names[1];
    xlibData m_info;
    bool m_status = false;
    uint32_t m_minInSize = 32 * 1024;
    unsigned char* m_deflateOutput = nullptr;
    unsigned char* m_deflateInput = nullptr;
    unsigned char* m_inflateOutput = nullptr;
    unsigned char* m_inflateInput = nullptr;
    uint32_t m_outputSize = 0;
    uint32_t m_inputSize = 0;
    const uint32_t m_bufSize = DEFLATE_BUFFER;
    std::vector<cl::Device> m_ndevinfo;
    std::vector<cl::Context*> m_contexts;
    std::vector<cl::Program*> m_programs;

    bool m_instCreated = false;
    uint8_t m_StoreBlocks = 0;
    bool m_decompressEligible = false;
    bool m_copydecompressData = true;
    bool m_lastInflateBlock = false;
    bool last_block = false;
    bool initialized = false;
    std::mutex m_mutex;
    void lock() { m_mutex.lock(); }
    void unlock() { m_mutex.unlock(); }
    void init(z_streamp strm, const int flow);
#ifdef ENABLE_INFLATE_XRM
    // XRM CU Instance
    xrmStruct* getXrmCuInstance(z_streamp strm, const std::string& kernel_name);

    // XRM Context
    xrmContext* m_ctx = nullptr;

    // Map data structure to hold
    // zstream struture pointer and xrmObject
    std::map<z_streamp, xrmStruct*> m_xrmMapObj;
#endif
};
