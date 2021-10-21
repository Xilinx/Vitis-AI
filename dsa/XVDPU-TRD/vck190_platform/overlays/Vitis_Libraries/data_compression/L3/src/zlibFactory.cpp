#include "zlibFactory.hpp"

// Private constructor
zlibFactory::zlibFactory(void) {}

// Provide the zlibFactory instance
// Called in deflateInit2_
zlibFactory* zlibFactory::getInstance(void) {
    static zlibFactory instance;
    return &instance;
}

bool zlibFactory::createContext(void) {
    cl_int err;
    cl::Context* context = nullptr;

    // Create OpenCL context
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, m_platform, 0};
    OCL_CHECK(err, context = new cl::Context(m_device, props, nullptr, nullptr, &err));
    if (err) {
        return true;
    } else {
        this->m_deviceContext.push_back(context);
        return false;
    }
}

bool zlibFactory::loadBinary(void) {
    cl_int err;
    cl::Program* program = nullptr;
    std::ifstream bin_file(this->m_u50_xclbin.c_str(), std::ifstream::binary);
    if (bin_file.fail()) {
        std::cerr << this->m_u50_xclbin.c_str() << " Unable to open binary file" << std::endl;
        bin_file.close();
        return true;
    } else {
        // Fill the m_binVec
        bin_file.seekg(0, bin_file.end);
        auto nb = bin_file.tellg();
        bin_file.seekg(0, bin_file.beg);
        m_binVec.resize(nb);
        bin_file.read(reinterpret_cast<char*>(m_binVec.data()), nb);

        cl::Program::Binaries bins{{m_binVec.data(), m_binVec.size()}};
        bin_file.close();

        program = new cl::Program(*this->m_deviceContext.back(), {m_device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            delete program;
            m_binVec.clear();
            std::string devName;
            m_device.getInfo(CL_DEVICE_NAME, &devName);
            std::cerr << "Programming \"" << devName << "\" Failed !!"
                      << " Trying Full Xclbin !" << std::endl;
            auto fullXclbin = std::string(c_installRootDir) + c_hardXclbinPath;
            std::ifstream binFileFull(fullXclbin.c_str(), std::ifstream::binary);
            if (binFileFull.fail()) {
                std::cerr << "XCLBIN2: Unable to open binary file " << fullXclbin << std::endl;
                return true;
            } else {
                binFileFull.seekg(0, binFileFull.end);
                auto nb = binFileFull.tellg();
                binFileFull.seekg(0, binFileFull.beg);
                m_binVec.resize(nb);
                binFileFull.read(reinterpret_cast<char*>(m_binVec.data()), nb);
                cl::Program::Binaries binsFull{{m_binVec.data(), m_binVec.size()}};
                binFileFull.close();

                program = new cl::Program(*this->m_deviceContext.back(), {m_device}, binsFull, NULL, &err);
                if (err != CL_SUCCESS) {
                    delete program;
                    return true;
                } else {
                    this->m_deviceProgram.push_back(program);
                    return false;
                }
            }
        } else {
            this->m_deviceProgram.push_back(program);
            return false;
        }
    }
}

void zlibFactory::xilinxPreChecks(void) {
    lock();
    if (this->m_PreCheckStatus) {
        this->m_PreCheckStatus = false;

#ifdef XILINX_DEBUG
        // XILINX_NO_ACCEL -- CHECK
        // Check if user requesteed for No acceleration
        char* noaccel = getenv("XILINX_NO_ACCEL");
        if (noaccel && (std::stoi(noaccel) == 0)) {
// User requested no accel -- Return
#if (VERBOSE_LEVEL >= 1)
            std::cout << "[XZLIB]: User requested for No-Acceleration" << std::endl;
#endif
            this->m_xMode = false;
        } else {
#endif
            // By defalut try to pick XCLBIN from /opt area
            this->m_u50_xclbin = std::string(c_installRootDir) + c_hardXclbinPath;

#ifdef XILINX_DEBUG
            // Check for ENT variable for XCLBIN if it doesnt
            // exist then use hardcoded path to pick
            char* xclbin = getenv("XILINX_LIBZ_XCLBIN");
            if (xclbin) this->m_u50_xclbin = xclbin;
#endif
            // Set Xilinx XRT path
            setenv("XILINX_XRT", "/opt/xilinx/xrt", 1);

            // Set Xilinx XRT_INI_PATH
            std::string hpath = std::string(c_installRootDir) + "zlib/scripts/xrt.ini";
            setenv("XRT_INI_PATH", hpath.c_str(), 1);

            // Create Platform & Device information
            cl_int err;

            // Look for platform
            std::vector<cl::Platform> platforms;
            OCL_CHECK(err, err = cl::Platform::get(&platforms));
            auto num_platforms = platforms.size();
            if (num_platforms == 0) {
#if (VERBOSE_LEVEL >= 2)
                std::cerr << "No Platforms were found this could be cased because of the OpenCL \
                                  ICD not installed at /etc/OpenCL/vendors directory"
                          << std::endl;
#endif
                this->m_xMode = false;
            } else {
                std::string platformName;
                cl::Platform platform;
                bool foundFlag = false;
                int idx = 0;
                for (size_t i = 0; i < platforms.size(); i++) {
                    platform = platforms[i];
                    OCL_CHECK(err, platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
                    if (platformName == "Xilinx") {
                        idx = i;
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
                    this->m_xMode = false;
                } else {
                    // Getting ACCELERATOR Devices and platform info
                    OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &m_ndevices));
                    cl_context_properties temp = (cl_context_properties)(platforms[idx])();
                    this->m_platform = temp;

                    // Find out valid devices
                    int deviceCount = this->m_ndevices.size();
                    for (int i = 0; i < deviceCount; i++) {
                        m_device = this->m_ndevices[i];

                        // Create Context
                        auto err_context = createContext();
                        m_xMode = err_context ? false : true;
                        if (err_context) continue;

                        auto err_bin = loadBinary();
                        m_xMode = err_bin ? false : true;

                        if (m_xMode) {
                            this->m_deviceCount.push_back(m_device);
                        }
                    }
                }
            }
#ifdef XILINX_DEBUG
        }
#endif
    }
    unlock();
}

// Get method for xMode
bool zlibFactory::getXmode(void) {
    return this->m_xMode;
}

// Provde Driver object by zlibFactory object
zlibDriver* zlibFactory::getDriverInstance(z_streamp strm, int flow, bool init) {
    lock();
    std::string kernel;
    zlibDriver* ret = nullptr;

    // Find current z_stream object if it exists in map
    auto iterator = this->m_driverMapObj.find(strm);
    if (iterator != this->m_driverMapObj.end()) {
        // If it exists the return
        ret = iterator->second;
    } else {
        zlibDriver* driver = new zlibDriver(strm, this->m_platform, this->m_u50_xclbin, flow, this->m_deviceCount,
                                            this->m_deviceContext, this->m_deviceProgram, init);
        if (driver->getErrStatus()) {
            ret = nullptr;
        } else {
            // Insert the strm,xlz key/value pairs in map
            this->m_driverMapObj.insert(std::pair<z_streamp, zlibDriver*>(strm, driver));
            ret = driver;
        }
    }
    unlock();
    // return the object
    return ret;
}

void zlibFactory::releaseZlibObj(z_streamp strm) {
    lock();
    // Find if current z_stream exists in map
    auto driverIter = this->m_driverMapObj.find(strm);
    if (driverIter != this->m_driverMapObj.end()) {
        zlibDriver* driver = driverIter->second;
        driver->releaseZlibCU(strm);
    }
    unlock();
}

// Release the zlibDriver by zlibFactory object
void zlibFactory::releaseDriverObj(z_streamp strm) {
    lock();
    // Find if current z_stream exists in map
    auto driverIter = this->m_driverMapObj.find(strm);
    if (driverIter != this->m_driverMapObj.end()) {
        delete driverIter->second;
        // Remove <key,value> pair from map
        this->m_driverMapObj.erase(driverIter);
    }
    unlock();
}
