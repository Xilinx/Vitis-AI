/*****************************************************************************
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*****************************************************************************/

#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <algorithm>

#include <vitis/ai/tracelogging.hpp>

const char * kVartTraceEnableEnvVarName = "VART_TL_ENABLE";

int main(int argc, char ** argv)
{
    const char * val = std::getenv(kVartTraceEnableEnvVarName);
    if (nullptr == val) {
        std::cout << "Warning: environment variable is not set: " << kVartTraceEnableEnvVarName << "\n";
    } else {
        std::string val_s{val};
        std::transform(val_s.begin(), val_s.end(), val_s.begin(),
                [](auto c) {
                    return std::tolower(c);
                }
            );
        if (val_s == "true" || val_s == "yes" ||  val_s == "on" || val_s == "1") {
            std::cout << "Tracing is enabled by environment variable: " << kVartTraceEnableEnvVarName << "=" << val << "\n";
        } else {
            std::cout << "Warning: Tracing is disabled by environment variable: " << kVartTraceEnableEnvVarName << "=" << val << "\n";
        }
    }

    for (int i = 0; i < 5; ++i)
    {
        TL_TRACE("Instant0");

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        TL_TRACE("Instant1") << "Test: " << 101;

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        {
            TL_TRACE_BLOCK("Block0");

            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            TL_TRACE_BLOCK("Block1") << "TestBlock1: " << "1111";

            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            TL_TRACE_BLOCK("Block2") << "TestBlock2";

            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            TL_TRACE_BLOCK("Block3") << "TestBlock3: " << 3333;

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    
        TL_TRACE("Instant2") << "Test: " << 101;

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    return 0;
}
