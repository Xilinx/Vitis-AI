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

#ifndef _XF_FINTECH_HESTON_EXECUTION_TIME_
#define _XF_FINTECH_HESTON_EXECUTION_TIME_

#include <chrono>

using namespace std;

namespace xf {
namespace fintech {

/**
 * @brief Heston FD Execution Time Class
 */

class HestonFDExecutionTime {
   public:
    /** @brief Constructor */
    HestonFDExecutionTime(){};
    /** @brief Start the timer */
    void Start();
    /** @brief Stop the timer */
    void Stop();
    /** @brief Return the duration of the timer */
    std::chrono::milliseconds Duration();

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _Start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> _Stop = std::chrono::high_resolution_clock::now();
};

} // namespace fintech
} // namespace xf

#endif // _XF_FINTECH_HESTON_EXECUTION_TIME_
