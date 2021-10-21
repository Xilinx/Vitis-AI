#include "zlib.hpp"
#include "deflate.h"
#include "zlibDriver.hpp"
using namespace xf::compression;

class zlibFactory {
   public:
    // Use getInstance of zlibFactory class
    // various requests through deflateInit
    // access zlibFactory instance
    static zlibFactory* getInstance();

    // Prechecks
    void xilinxPreChecks(void);

    // fpga/cpu mode
    bool getXmode();

    // Driver Instance
    zlibDriver* getDriverInstance(z_streamp strm, int flow, bool init = true);

    // Release Driver Instance
    void releaseDriverObj(z_streamp strm);

    // Release Driver Instance
    void releaseZlibObj(z_streamp strm);

   private:
    // XCLBIN name
    std::string m_u50_xclbin;
    std::vector<cl::Device> m_deviceCount;
    std::vector<cl::Context*> m_deviceContext;
    std::vector<cl::Program*> m_deviceProgram;

    // Common platform / device
    // exists throughout the process
    cl_context_properties m_platform;
    cl::Device m_device;
    cl::Context* m_context;
    cl::Program* m_program;

    // Various opencl devices
    std::vector<cl::Device> m_ndevices;

    bool createContext(void);
    bool loadBinary(void);

    // Private constructor
    zlibFactory();

    // PreCheck flag --> FPGA/CPU
    bool m_xMode = true;

    // XCLBIN container
    std::vector<uint8_t> m_binVec;

    std::mutex m_mutex;
    void lock() { m_mutex.lock(); }
    void unlock() { m_mutex.unlock(); }

    // std::map container to hold
    // zstream structure pointers and zlibDriver Class Object
    // Mapping (Driver Class)
    std::map<z_streamp, zlibDriver*> m_driverMapObj;

    // Precheck status
    bool m_PreCheckStatus = true;
};
