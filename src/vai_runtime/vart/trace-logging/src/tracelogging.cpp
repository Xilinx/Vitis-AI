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

#include <vitis/ai/tracelogging.hpp>

#if TL_ENABLE_BUILD_FLAG

#include <cstdlib>
#include <algorithm>

#ifdef _WIN32

#include <Windows.h>
#include <TraceLoggingProvider.h>

// e00db74d-c600-4092-890f-91c684853b50
TRACELOGGING_DEFINE_PROVIDER(
    g_loggingProvider,
    "VAITraceLoggingProvider",
    (0xe00db74d, 0xc600, 0x4092,   0x89, 0x0f,   0x91, 0xc6, 0x84, 0x85, 0x3b, 0x50));

#endif // _WIN32

namespace vitis::ai::tracelogging {

namespace {

bool getEnableFlagFromEnvVar()
{
    bool enable = false;

    const char * val = std::getenv("VART_TL_ENABLE");
    if (val) {
        std::string val_s{val};
        std::transform(val_s.begin(), val_s.end(), val_s.begin(),
                [](auto c) {
                    return std::tolower(c);
                }
            );
        enable = val_s == "true" || val_s == "yes" ||  val_s == "on" || val_s == "1";
    }

    return enable;
}

}

////////////////////////////////////////////////////////////////////////////////
// Global provider instance
////////////////////////////////////////////////////////////////////////////////
struct Init
{
    Init() {
        enable_ = getEnableFlagFromEnvVar();
#ifdef _WIN32
        TraceLoggingRegister(g_loggingProvider);
#endif
    }
    ~Init() {
#ifdef _WIN32
        TraceLoggingUnregister(g_loggingProvider);
#endif
    }
    bool enable_;
};

static Init g_init;

////////////////////////////////////////////////////////////////////////////////
// Trace calss methods
////////////////////////////////////////////////////////////////////////////////
Trace::Trace(std::string const & eventName)
    : eventName_{eventName}
{
}

Trace::~Trace()
{
    if (g_init.enable_) {
#ifdef _WIN32
        TraceLoggingWrite(
                g_loggingProvider,
                "PerformanceTracer", // Windows Trace Logging API requires a string literal here
                TraceLoggingValue(eventName_.c_str(), "Event"),
                TraceLoggingValue(str().c_str(), "Arg")
            );
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
// TraceBlock calss methods
////////////////////////////////////////////////////////////////////////////////
TraceBlock::TraceBlock(std::string const & eventName, Trace & end)
    : Trace{eventName + ": Begin"},
      end_{end}
{
    end_.eventName_ +=  ": End";
}

TraceBlock::~TraceBlock()
{
    end_ << str();
}

} // namespace vitis::ai::tracelogging

#endif // TL_ENABLE_BUILD_FLAG
