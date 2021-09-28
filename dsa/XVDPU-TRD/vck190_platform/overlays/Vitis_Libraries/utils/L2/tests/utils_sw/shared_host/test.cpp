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

#include <iostream>
#include "xf_utils_sw/logger.hpp"
#include "xcl2/xcl2.hpp"

int main(int argc, const char* argv[]) {
    using namespace xf::common::utils_sw;

    // eg: set non-default log dest
    Logger l(std::cout, std::cerr);

    // eg: common CL routines
    cl_int err = CL_SUCCESS;
    l.logCreateContext(err);
    l.logCreateCommandQueue(err);
    l.logCreateProgram(err, false); // 3rd arg for do-not-exit when error occurs
                                    // designed for advanced code with customized shutdown/re-try flow.
    l.logCreateKernel(err);

    // eg: template message
    l.info(Logger::Message::TIME_KERNEL_MS, 1.5f);

    // eg: for cases using original OCL_CHECK
    auto devices = xcl::get_xil_devices();
    cl::Context context;
    OCL_LOG_CHECK(err, context = cl::Context(devices[0], NULL, NULL, NULL, &err));

    // eg: level and call chaining (this will print nothing since info is no longer under the level of WARNING)
    l.setLevel(Logger::Level::WARNING).info(Logger::Message::TEST_PASS).setLevel(Logger::Level::INFO);

    // eg: handling result checking
    bool has_err = false;
    has_err ? l.error(Logger::Message::TEST_FAIL) : l.info(Logger::Message::TEST_PASS);

    return has_err;
}
