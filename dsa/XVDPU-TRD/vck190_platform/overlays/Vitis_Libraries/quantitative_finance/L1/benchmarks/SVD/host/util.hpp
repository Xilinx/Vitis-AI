/*
 * Copyright 2019 Xilinx, Inc.
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
#ifndef UTIL_H
#define UTIL_H

#include <string>

template <typename T>
inline T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
        throw std::bad_alloc();
    }
    return reinterpret_cast<T*>(ptr);
}

// special format that used in stac-a2 to store the operation result data.
inline std::string double2hexastr(double d) {
    char buffer[25] = {0};
    std::snprintf(buffer, 25, "0x%.16llx", *reinterpret_cast<unsigned long long*>(&d));
    return buffer;
}

unsigned long diff(const struct timeval* newTime, const struct timeval* oldTime);

int read_verify_env_int(const char* var, int fail_value = 1);
std::string read_verify_env_string(const char* var, std::string fail_value = "");

#endif
