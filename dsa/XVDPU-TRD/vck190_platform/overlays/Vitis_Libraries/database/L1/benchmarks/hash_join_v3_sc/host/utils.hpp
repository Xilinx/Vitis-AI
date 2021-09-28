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

#ifndef UTILS_H
#define UTILS_H

// ------------------------------------------------------------

#include <new>
#include <cstdlib>

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

// ------------------------------------------------------------

#include <algorithm>
#include <string>
#include <vector>

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

// ------------------------------------------------------------

#include <sys/time.h>

inline int tvdiff(struct timeval* tv0, struct timeval* tv1) {
    return (tv1->tv_sec - tv0->tv_sec) * 1000000 + (tv1->tv_usec - tv0->tv_usec);
}

// ------------------------------------------------------------

#include <sys/types.h>
#include <sys/stat.h>

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

#endif // UTILS_H
