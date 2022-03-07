/**
 * Copyright (C) 2019 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>
//#include <boost/align/aligned_allocator.hpp>

#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#ifdef __GNUC__
# define MAYBE_UNUSED __attribute__((unused))
#else
# define MAYBE_UNUSED
#endif

#ifdef _WIN32
# pragma warning ( disable : 4244 )
#endif

namespace utils {

//template <typename T>
//using aligned_allocator = boost::alignment::aligned_allocator<T, 4096>;
  
/**
 * @return
 *   nanoseconds since first call
 */
MAYBE_UNUSED
inline unsigned long long
time_ns()
{
  static auto zero = std::chrono::high_resolution_clock::now();
  auto now = std::chrono::high_resolution_clock::now();
  auto integral_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now-zero).count();
  return integral_duration;
}

/**
 * Simple time guard to accumulate scoped time
 */
class time_guard
{
  unsigned long zero = 0;
  unsigned long& tally;
public:
  time_guard(unsigned long& t)
    : zero(time_ns()), tally(t)
  {}

  ~time_guard()
  {
    tally += time_ns() - zero;
  }
};

inline void
throw_if_error(cl_int errcode, const std::string& msg="")
{
  if (!errcode)
    return;

  std::string err = "errcode '";
  err.append(std::to_string(errcode)).append("'");
  if (!msg.empty())
    err.append(" ").append(msg);
  throw std::runtime_error(err);
}

inline std::vector<char>
read_xclbin(const std::string& xclbin)
{
  std::ifstream stream(xclbin,std::ios::binary);
  if (!stream)
    throw std::runtime_error("could not open " + xclbin + " for reading");

  stream.seekg(0,stream.end);
  size_t size = stream.tellg();
  stream.seekg(0,stream.beg);

  std::vector<char> header(size);
  stream.read(header.data(),size);
  return header;
}

inline cl_platform_id
open_platform(const std::string& vendor, const std::string& name)
{
  cl_uint num_platforms = 0;
  throw_if_error(clGetPlatformIDs(0, nullptr, &num_platforms),"clGetPlatformIDs failed");
  std::vector<cl_platform_id> platforms(num_platforms);
  throw_if_error(clGetPlatformIDs(num_platforms, platforms.data(), nullptr),"clGetPlatformIDs failed");

  char str[512] = {0}; // argh ...
  int count = -1;
  for (auto platform : platforms) {
    ++count;
    throw_if_error(clGetPlatformInfo(platform,CL_PLATFORM_VENDOR,512,str,nullptr));
    std::cout << "platform[" << count << "].vendor = " << str << "\n";
    if (vendor != str)
      continue;
    throw_if_error(clGetPlatformInfo(platform,CL_PLATFORM_NAME,512,str,nullptr));
    std::cout << "platform[" << count << "].name = " << str << "\n";
    if (name == str)
      return platform;
  }

  return nullptr;
}

inline cl_device_id
get_device(cl_platform_id platform, unsigned int device)
{
  cl_uint num_devices = 0;
  throw_if_error(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &num_devices),"clGetDeviceIDs failed");
  if (device >= num_devices)
    throw std::runtime_error("device index '" + std::to_string(device) + "' is out range. Number of devices is '" + std::to_string(num_devices) + "'");

  std::vector<cl_device_id> devices(num_devices);
  throw_if_error(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, num_devices, devices.data(), nullptr),"clGetDeviceIDs failed");
  return devices[device];
}

} // utils