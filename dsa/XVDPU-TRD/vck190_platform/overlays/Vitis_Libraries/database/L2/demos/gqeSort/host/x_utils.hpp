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

#ifndef X_UTILS_HPP
#define X_UTILS_HPP

#include <sys/time.h>

#include <algorithm>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace x_utils {

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 4096, num * sizeof(T)))
        //  ptr= malloc(num*sizeof(T));
        //  if (ptr==NULL)
        throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

inline int tvdiff(const timeval& tv0, const timeval& tv1) {
    return (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec);
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

template <typename T>
int load_dat(T*& data, const std::string& file_path) {
    if (!is_file(file_path)) {
        std::cerr << "ERROR: File " << file_path << " not exist!" << std::endl;
        exit(1);
    }
    FILE* f = fopen(file_path.c_str(), "rb");
    fseek(f, 0, SEEK_END); // seek to end of file
    size_t n = ftell(f);   // get current file pointer
    fseek(f, 0, SEEK_SET); // seek back to beginning of file
    if (!f) {
        std::cerr << "ERROR: " << file_path << " cannot be opened for binary read." << std::endl;
    }
    data = aligned_alloc<T>(n / sizeof(T));
    if (!data) {
        return -1;
    }
    size_t cnt = fread((void*)data, sizeof(T), n / sizeof(T), f);
    fclose(f);
    if (cnt != n / sizeof(T)) {
        std::cerr << "ERROR: " << cnt << " entries read from " << file_path << ", " << n << " entries required."
                  << std::endl;
        return -1;
    }
    return cnt;
}

template <typename T>
int gen_dat(T* data, const std::string& file_path, size_t n) {
    if (!data) {
        return -1;
    }

    FILE* f = fopen(file_path.c_str(), "wb");
    if (!f) {
        std::cerr << "ERROR: " << file_path << " cannot be opened for binary read." << std::endl;
    }
    size_t cnt = fwrite((void*)data, sizeof(T), n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << file_path << ", " << n << " entries required."
                  << std::endl;
        return -1;
    }
    return 0;
}

} // namespace x_utils

#endif // X_UTILS_HPP
