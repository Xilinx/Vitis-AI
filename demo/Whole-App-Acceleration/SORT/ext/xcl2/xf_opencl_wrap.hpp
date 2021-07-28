#ifndef _XF_OPENCL_WRAPPER_H
#define _XF_OPENCL_WRAPPER_H
#include <iostream>
#include "common/xf_headers.hpp"
#include "xcl2.hpp"

#ifndef ASSERT
#define ASSERT(x) assert(x)
#endif

template <typename T>
class XclIn;
class XclIn2;

template <typename T>
class XclOut;
class XclOut2;

struct cl_buffer_wrapper {
    void* mData;
    size_t mSize;
    cl::Buffer mBuffer;
    static int err;
    cl::Event mEvent;
    cl_mem_flags mFlag;

    cl_buffer_wrapper(cl::Context& context, void* data, size_t size, cl_mem_flags flag)
        : mData(data), mSize(size), mFlag(flag) {
        if (size > 0) {
            OCL_CHECK(err, cl::Buffer buffer(context, flag, size, NULL, &err));
            mBuffer = std::move(buffer);
        }
    }

    cl::Event& getEvent() { return mEvent; }
    cl::Buffer& getBuffer() { return mBuffer; }
    size_t getSize() { return mSize; }
    void* getData() { return mData; }
    cl_mem_flags getFlag() { return mFlag; }
    void setFlag(cl_mem_flags flg) { mFlag = flg; }
};
int cl_buffer_wrapper::err = 0;

class cl_kernel_wrapper {
   private:
    cl::Context mContext;
    std::string mFuncName;
    std::string mBinName;
    std::vector<cl_buffer_wrapper*> mBuffersIn;
    std::vector<cl_buffer_wrapper*> mBuffersOut;
    std::vector<cl_buffer_wrapper*> mBuffersAll;
    cl::Event mEvent;
    cl::CommandQueue mQueue;
    bool mConsumed;

    cl::Kernel mKernel;
    std::vector<cl_kernel_wrapper*> mDepends; // Adjacency list for a DAG of kernel wrappers

    static int err;

    void sanitizeRW(cl_buffer_wrapper* buff, const cl_mem_flags flag) {
        /* In case a cl::Buffer was created as output of a kernel function and later is modified as input to another
         * kernel function then the flag needs to be modified to RW */
        if ((buff->getFlag() != CL_MEM_READ_WRITE) && (buff->getFlag() != flag)) {
            OCL_CHECK(err, cl::Buffer buffer(mContext, CL_MEM_READ_WRITE, buff->getSize(), NULL, &err));
            buff->getBuffer() = std::move(buffer);
            buff->setFlag(CL_MEM_READ_WRITE);
        }
    }

   public:
    cl_kernel_wrapper(const cl::Context& context,
                      const std::string& func_name,
                      const std::string& bin_name,
                      std::string& device_name,
                      cl::CommandQueue Queue,
                      std::vector<cl::Device>& devices)
        : mContext(context), mFuncName(func_name), mBinName(bin_name), mQueue(Queue) {
        std::string binaryFile = xcl::find_binary_file(device_name, mBinName.c_str());
        cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
        OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
        OCL_CHECK(err, cl::Kernel kernel(program, mFuncName.c_str(), &err));
        mKernel = std::move(kernel);
        mConsumed = false;
    }

    bool isConsumed() { return mConsumed; }
    void setConsumed(bool bflg) { mConsumed = bflg; }

    /*IMP : Inputs and outputs should be registered exactly in the same order as they are expected at Kernel interface
     */
    int argCount() { return mBuffersAll.size(); }

    cl_buffer_wrapper* getArg(int idx) {
        ASSERT(idx < argCount());
        return mBuffersAll[idx];
    }

    /*In case buufer created as output of a kernel needs to be reused as input to another kernel*/
    cl_buffer_wrapper* registerInput(cl_buffer_wrapper* buffin) {
        mBuffersIn.push_back(buffin);
        mBuffersAll.push_back(buffin);
        if (buffin->getSize() > 0) {
            OCL_CHECK(err, err = mKernel.setArg((argCount() - 1), buffin->getBuffer()));
            sanitizeRW(buffin, CL_MEM_READ_ONLY);
        }
        return buffin;
    }

    /*NOTE : host data pointer can be NULL in case data transaction from host is not required */
    cl_buffer_wrapper* registerInput(void* data, size_t size) {
        cl_buffer_wrapper* buffin = new cl_buffer_wrapper(mContext, data, size, CL_MEM_READ_ONLY);
        registerInput(buffin);
        return buffin;
    }

    // Helper function which can take a cv::Mat
    cl_buffer_wrapper* registerInput(cv::Mat& img) {
        size_t size = img.total() * img.elemSize();
        return registerInput(img.data, size);
    }

    template <typename T>
    cl_buffer_wrapper* registerInput(T& arg) {
        cl_buffer_wrapper* argcl = registerInput((void*)&arg, 0);
        OCL_CHECK(err, err = mKernel.setArg((argCount() - 1), arg));
        return argcl;
    }

    /*In case buufer created as output of a kernel needs to be reused as input to another kernel*/
    cl_buffer_wrapper* registerOutput(cl_buffer_wrapper* buffout) {
        mBuffersOut.push_back(buffout);
        mBuffersAll.push_back(buffout);
        if (buffout->getSize() > 0) {
            OCL_CHECK(err, err = mKernel.setArg((argCount() - 1), buffout->getBuffer()));
            sanitizeRW(buffout, CL_MEM_WRITE_ONLY);
        }
        return buffout;
    }

    /*NOTE : host data pointer can be NULL in case data transaction from host is not required */
    cl_buffer_wrapper* registerOutput(void* data, size_t size) {
        cl_buffer_wrapper* buffout = new cl_buffer_wrapper(mContext, data, size, CL_MEM_WRITE_ONLY);
        registerOutput(buffout);
        return buffout;
    }

    // Helper function which can take a cv::Mat
    cl_buffer_wrapper* registerOutput(cv::Mat& img) {
        size_t size = img.total() * img.elemSize();
        return registerOutput(img.data, size);
    }

    template <typename T>
    cl_buffer_wrapper* registerOutput(T& arg) {
        cl_buffer_wrapper* argcl = registerOutput((void*)&arg, 0);
        OCL_CHECK(err, err = mKernel.setArg((argCount() - 1), arg));
        return argcl;
    }

    template <typename T>
    void registerArg(T& A) {
        A.addArg(this);
    }

    void registerArgs() {}

    template <typename First, typename... Rest>
    void registerArgs(First& first, Rest&... rest) {
        registerArg(first);
        registerArgs(rest...);
    }

    // Host to device data transfer
    void enqueueWriteBuffer(std::vector<cl::Event>& events) {
        for (unsigned int i = 0; i < mBuffersIn.size(); i++) {
            if (mBuffersIn[i]->getSize() > 0) {
                if ((mBuffersIn[i]->getData())) {
                    OCL_CHECK(err, mQueue.enqueueWriteBuffer(mBuffersIn[i]->getBuffer(), // buffer on the FPGA
                                                             CL_FALSE,                   // blocking call
                                                             0,                          // buffer offset in bytes
                                                             mBuffersIn[i]->getSize(),   // Size in bytes
                                                             mBuffersIn[i]->getData(),   // Host data pointer
                                                             nullptr, &(mBuffersIn[i]->getEvent())));
                }
                events.push_back(mBuffersIn[i]->getEvent());
            }
        }
    }

    // Device to host data transfer
    void enqueueReadBuffer() {
        std::vector<cl::Event> events;
        events.push_back(mEvent);
        for (unsigned int i = 0; i < mBuffersOut.size(); i++) {
            if ((mBuffersOut[i]->getSize() > 0) && (mBuffersOut[i]->getData())) {
                OCL_CHECK(err, mQueue.enqueueReadBuffer(mBuffersOut[i]->getBuffer(), // buffer on the FPGA
                                                        CL_FALSE,                    // blocking call
                                                        0,                           // buffer offset in bytes
                                                        mBuffersOut[i]->getSize(),   // Size in bytes
                                                        mBuffersOut[i]->getData(),   // Host data pointer
                                                        &events, &(mBuffersOut[i]->getEvent())));
            }
        }
    }

    void addDependent(cl_kernel_wrapper* krnl) {
        mDepends.push_back(krnl);
        krnl->setConsumed(true);
    }

    void exec() {
        for (auto& k : mDepends) {
            k->exec();
        }

        std::vector<cl::Event> events;
        for (auto& k : mDepends) {
            events.push_back(k->mEvent);
        }
        std::cout << "Starting kernel [" << mFuncName << "] execution" << std::endl;
        enqueueWriteBuffer(events);
        if (events.size() == 0) {
            OCL_CHECK(err, err = mQueue.enqueueTask(mKernel, NULL, &mEvent));
        } else {
            OCL_CHECK(err, err = mQueue.enqueueTask(mKernel, &events, &mEvent));
        }
        // Added for profiling [[
        clWaitForEvents(1, (const cl_event*)&mEvent);

        cl_ulong start = 0;
        cl_ulong end = 0;
        double diff_prof = 0.0f;
        mEvent.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        mEvent.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        diff_prof = end - start;
        std::cout << "Kernel [" << mFuncName << "] execution time : " << (diff_prof / 1000000) << "ms" << std::endl;
        //]]
        enqueueReadBuffer();
    }
};
int cl_kernel_wrapper::err = 0;

// This is a singleton class managing kernel registry and execution
class cl_kernel_mgr {
   private:
    std::vector<cl::Device> mDevices;
    cl::Device mDevice;
    cl::Context mContext;
    std::vector<cl::CommandQueue> mQueueVec;
    cl::CommandQueue mCurrQueue;
    std::string mDeviceName;

    std::vector<cl_kernel_wrapper*> mKernelVec;
    static cl_kernel_mgr* mRegistry;
    static int err;

    cl_kernel_mgr() {
        std::cout << "INFO: Running OpenCL section." << std::endl;

        // Get the device:
        mDevices = xcl::get_xil_devices();
        mDevice = mDevices[0];
        mDevices.resize(1);

        OCL_CHECK(err, cl::Context context(mDevice, NULL, NULL, NULL, &err));
        mContext = std::move(context);

        OCL_CHECK(err,
                  cl::CommandQueue queue(mContext, mDevice,
                                         (CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE), &err));
        mCurrQueue = std::move(queue);

        OCL_CHECK(err, std::string device_name = mDevice.getInfo<CL_DEVICE_NAME>(&err));
        std::cout << "INFO: Device found - " << device_name << std::endl;
        mDeviceName = device_name;
    }

   public:
    static cl_kernel_wrapper* registerKernel(std::string& func_name, std::string& bin_name) {
        if (nullptr == mRegistry) {
            // Create a registry
            mRegistry = new cl_kernel_mgr();
        }

        cl_kernel_wrapper* kernel =
            new cl_kernel_wrapper(mRegistry->mContext, func_name, bin_name, mRegistry->mDeviceName,
                                  mRegistry->mCurrQueue, mRegistry->mDevices);
        mRegistry->mKernelVec.push_back(kernel);
        return kernel;
    }

    template <typename... Args>
    static cl_kernel_wrapper* registerKernel(const std::string& func_name, const std::string& bin_name, Args... argv) {
        if (nullptr == mRegistry) {
            // Create a registry
            mRegistry = new cl_kernel_mgr();
        }

        cl_kernel_wrapper* kernel =
            new cl_kernel_wrapper(mRegistry->mContext, func_name, bin_name, mRegistry->mDeviceName,
                                  mRegistry->mCurrQueue, mRegistry->mDevices);
        mRegistry->mKernelVec.push_back(kernel);
        kernel->registerArgs(argv...);
        return kernel;
    }

    static void exec(cl_kernel_wrapper* kernel) {
        ASSERT(nullptr != mRegistry);
        kernel->exec();
    }

    static void finish() {
        ASSERT(nullptr != mRegistry);
        for (auto& q : mRegistry->mQueueVec) {
            q.finish();
        }
        mRegistry->mCurrQueue.finish();
    }

    static void exec_all() {
        ASSERT(nullptr != mRegistry);
        for (auto& it : mRegistry->mKernelVec) {
            if (it->isConsumed() == false) exec(it);
        }
        finish();
    }

    static void createNewQueue() {
        ASSERT(nullptr != mRegistry);
        mRegistry->mQueueVec.push_back(mRegistry->mCurrQueue);
        OCL_CHECK(err,
                  cl::CommandQueue queue(mRegistry->mContext, mRegistry->mDevice,
                                         (CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE), &err));
        mRegistry->mCurrQueue = std::move(queue);
    }
};

template <typename T>
class XclIn {
   public:
    T* mI;
    XclIn(T& I) : mI(&I) {}
    void addArg(cl_kernel_wrapper* kl) { kl->registerInput(*mI); }
};

class XclIn2 {
   public:
    void* mI1;
    size_t mI2;
    XclIn2(void* I1, size_t I2) : mI1(I1), mI2(I2) {}
    void addArg(cl_kernel_wrapper* kl) { kl->registerInput(mI1, mI2); }
};

template <typename T>
class XclOut {
   public:
    T* mO;
    XclOut(T& O) : mO(&O) {}
    void addArg(cl_kernel_wrapper* kl) { kl->registerOutput(*mO); }
};

class XclOut2 {
   public:
    void* mO1;
    size_t mO2;
    XclOut2(void* O1, size_t O2) : mO1(O1), mO2(O2) {}
    void addArg(cl_kernel_wrapper* kl) { kl->registerOutput(mO1, mO2); }
};

template <typename T>
inline XclIn<T> XCLIN(T& var) {
    return XclIn<T>(var);
}

inline XclIn2 XCLIN(void* var1, size_t var2) {
    return XclIn2(var1, var2);
}

template <typename T>
inline XclOut<T> XCLOUT(T& var) {
    return XclOut<T>(var);
}

inline XclOut2 XCLOUT(void* var1, size_t var2) {
    return XclOut2(var1, var2);
}

cl_kernel_mgr* cl_kernel_mgr::mRegistry = nullptr;
int cl_kernel_mgr::err = 0;

#endif
