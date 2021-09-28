/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */
#include "gzipApp.hpp"
#include "gzipXrtHost.hpp"
#include <memory>

int main(int argc, char* argv[]) {
    bool enable_profile = true;
    compressBase::State flow = compressBase::COMPRESS;
    compressBase::Level lflow = compressBase::SEQ;
    gzipBase::d_type decKernelType = gzipBase::FULL;

    // Driver class object
    gzipApp d(argc, argv, enable_profile);

    // Design class object creating and constructor invocation
    std::unique_ptr<gzipXrtHost> gzip(new gzipXrtHost(flow, d.getXclbin(), d.getInFileName(), lflow, d.getDeviceId(),
                                                      enable_profile, decKernelType, d.getDesignFlow()));

    // Run API to launch the compress or decompress engine
    d.run(gzip.get(), d.getMCR());
    return 0;
}
