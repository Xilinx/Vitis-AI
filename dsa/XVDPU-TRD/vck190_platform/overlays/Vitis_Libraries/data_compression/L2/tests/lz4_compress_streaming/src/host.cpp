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
#include "lz4App.hpp"
#include "lz4OCLHost.hpp"
#include <memory>

int main(int argc, char* argv[]) {
    bool enable_profile = false;
    bool lz4_stream = true;
    compressBase::State flow = compressBase::COMPRESS;
    compressBase::Level lflow = compressBase::SEQ;

    // Driver class object
    lz4App d(argc, argv, lflow, enable_profile);

    // Design class object creating and constructor invocation
    std::unique_ptr<lz4OCLHost> lz4(
        new lz4OCLHost(flow, d.getXclbin(), d.getDeviceId(), d.getBlockSize(), lz4_stream, enable_profile));

    // Run API to launch the compress or decompress engine
    d.run(lz4.get(), d.getMCR());
    return 0;
}
