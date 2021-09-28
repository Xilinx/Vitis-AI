/*
 * Copyright 2020 Xilinx, Inc.
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
 */
#ifndef XF_TEXT_HELPER_HPP
#define XF_TEXT_HELPER_HPP

// Include xilinx header first, to define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl_ext_xilinx.h>
#include <CL/cl.h>

#include <cstddef>
#include <cstdlib>
#include <new>

#include <sys/time.h>

#include <algorithm>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>
#include "xf_utils_sw/logger.hpp"

namespace xf {
namespace data_analytics {
namespace text {

// helper functions for internal use.
namespace details {

inline unsigned long read_binary_file(const char* fname, void** buffer) {
    unsigned long size = 0;
    FILE* fp = fopen(fname, "rb");
    if (!fp) {
        fprintf(stderr, "File %s cannot be opened for read.\n", fname);
    }
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    *buffer = (void*)malloc(size);
    fread(*buffer, 1, size, fp);
    fclose(fp);
    return size;
}

inline cl_int init_hardware(cl_context* context,
                            cl_device_id* device_id,
                            cl_command_queue* cmd_queue,
                            cl_command_queue_properties queue_props,
                            const int dev_index) {
    cl_int err;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    cl_uint platform_count = 0;
    err = clGetPlatformIDs(0, NULL, &platform_count);
    platform_count = platform_count > 16 ? 16 : platform_count;
    cl_platform_id platforms[16] = {0};
    err = clGetPlatformIDs(platform_count, platforms, NULL);

    cl_uint pid;
    char platform_name[256];
    for (pid = 0; pid < platform_count; ++pid) {
        err = clGetPlatformInfo(platforms[pid], CL_PLATFORM_NAME, 256, platform_name, 0);
        if (strcmp(platform_name, "Xilinx")) continue;
        // platform found.
        const cl_device_type device_type = CL_DEVICE_TYPE_ACCELERATOR;
        cl_uint device_count;
        err = clGetDeviceIDs(platforms[pid], device_type, 0, NULL, &device_count);
        cl_device_id devices[16] = {0};
        device_count = device_count > 16 ? 16 : device_count;
        err = clGetDeviceIDs(platforms[pid], device_type, device_count, devices, NULL);
        char device_name[256];
        err = clGetDeviceInfo(devices[dev_index], CL_DEVICE_NAME, 256, device_name, 0);
        printf("INFO: selected device %s\n", device_name);
        *device_id = devices[dev_index];
        // device found.
        cl_context_properties ctx_prop[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[pid], 0};
        cl_context ctx = clCreateContext(ctx_prop, 1, device_id, NULL, NULL, &err);
        logger.logCreateContext(err);
        if (err != CL_SUCCESS) {
            return err;
        }
        printf("INFO: initilized context.\n");
        // context ready.
        cl_command_queue q = clCreateCommandQueue(ctx, devices[dev_index], queue_props, &err);
        logger.logCreateCommandQueue(err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: Failed to create command queue.\n");
            return err;
        }
        printf("INFO: initilized command queue.\n");
        // queue ready.
        *context = ctx;
        *cmd_queue = q;
        break;
    }
    if (pid == platform_count) {
        fprintf(stderr, "ERROR: Xilinx platform not found.\n");
        return CL_INVALID_PLATFORM;
    }
    return err;
}

static cl_int load_binary(cl_program* program, cl_context context, cl_device_id device_id, const char* xclbin) {
    cl_int err;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    void* kernel_image = NULL;
    unsigned long size = read_binary_file(xclbin, &kernel_image);
    cl_program prog =
        clCreateProgramWithBinary(context, 1, &device_id, &size, (const unsigned char**)&kernel_image, NULL, &err);
    logger.logCreateProgram(err);
    if (err != CL_SUCCESS) {
        return err;
    }
    printf("INFO: created program with binary %s\n", xclbin);

    err = clBuildProgram(prog,       // program
                         1,          // number of devices
                         &device_id, // devices list
                         NULL,       // compile options
                         NULL,       // pfn_notify
                         NULL);      // user_data
    if (err != CL_SUCCESS) {
        free(kernel_image);
        return err;
    }
    printf("INFO: built program.\n");
    *program = prog;
    free(kernel_image);
    return err;
}

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

class MM {
   private:
    size_t _total;
    std::vector<void*> _pvec;

   public:
    MM() : _total(0) {}
    ~MM() {
        for (void* p : _pvec) {
            if (p) free(p);
        }
    }
    size_t size() const { return _total; }
    template <typename T>
    T* aligned_alloc(std::size_t num) {
        void* ptr = nullptr;
        size_t sz = num * sizeof(T);
        if (posix_memalign(&ptr, 4096, sz)) throw std::bad_alloc();
        _pvec.push_back(ptr);
        _total += sz;
        // printf("align_alloc %lu MB\n", _total / 1024 / 1024);
        return reinterpret_cast<T*>(ptr);
    }
};

inline int tvdiff(const timeval& tv0, const timeval& tv1) {
    return (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec);
}
inline int tvdiff(const timeval& tv0, const timeval& tv1, const char* info) {
    int exec_us = tvdiff(tv0, tv1);
    printf("%s: %d.%03d msec\n", info, (exec_us / 1000), (exec_us % 1000));
    return exec_us;
}

class ArgParser {
   public:
    ArgParser(int argc, const char* argv[]) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end()) {
            if (++itr != this->mTokens.end()) {
                value = *itr;
            }
            return true;
        }
        return false;
    }

   private:
    std::vector<std::string> mTokens;
};

inline bool has_end(std::string const& full, std::string const& end) {
    if (full.length() >= end.length()) {
        return (0 == full.compare(full.length() - end.length(), end.length(), end));
    } else {
        return false;
    }
}

inline bool is_dir(const char* path) {
    struct stat info;
    if (stat(path, &info) != 0) return false;
    if (info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}

inline bool is_dir(const std::string& path) {
    return is_dir(path.c_str());
}

inline bool is_file(const char* path) {
    struct stat info;
    if (stat(path, &info) != 0) return false;
    if (info.st_mode & (S_IFREG | S_IFLNK))
        return true;
    else
        return false;
}

inline bool is_file(const std::string& path) {
    return is_file(path.c_str());
}

} /* details */
} // namespace text
} // namespace data_analytics
} // namespace xf

#endif
