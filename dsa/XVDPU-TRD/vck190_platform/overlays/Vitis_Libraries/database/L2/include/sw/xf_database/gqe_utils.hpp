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

#ifndef GQE_UTILS_HPP
#define GQE_UTILS_HPP

#include <cstdlib>
#include <memory>
#include <vector>
#include <chrono>

namespace xf {
namespace database {
namespace gqe {
namespace utils {

/// Stand-alone aligned memory allocator
template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

/**
 * @class MM gqe_utils.hpp "xf_database/gqe_utils.hpp"
 * @brief Managed aligned memory allocator.
 */
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
        return reinterpret_cast<T*>(ptr);
    }
};

class Timer {
   private:
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock> > p;

   public:
    /// add another point.
    Timer& add() {
        p.push_back(std::chrono::high_resolution_clock::now());
        return *this;
    }
    /// return the difference between last two point.
    float getMilliSec() const { return getMilliSec(p.size() - 2, p.size() - 1); }
    /// return between selected points.
    float getMilliSec(int start, int end) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(p[end] - p[start]).count() / 1000.0f;
    }
};

} /* utils */
} /* gqe */
} /* database */
} /* xf */

#endif
