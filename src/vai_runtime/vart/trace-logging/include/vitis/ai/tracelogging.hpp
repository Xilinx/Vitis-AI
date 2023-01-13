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

#pragma once

#include <string>
#include <sstream>

#ifndef TL_ENABLE_BUILD_FLAG
#define TL_ENABLE_BUILD_FLAG 0
#endif

#if TL_ENABLE_BUILD_FLAG

#define TL_CONCAT_L2(P1,P2) P1 ## P2
#define TL_CONCAT(P1,P2) TL_CONCAT_L2(P1,P2)

#define TL_VAR TL_CONCAT(tl_trace_block_local_var_, __LINE__)

/**
 * @brief This macro generates a single trace event.
 *
 * To generate a single trace event at specific code location:
 * 
 * TL_TRACE("Start");
 * 
 * The event name is passed as an argument to this macro: EVENT_NAME. It is also
 * possible to supply additional event arguments by using c++ stream syntax:
 * 
 * int val = 10;
 * TL_TRACE("Event") << "val=" << val;
 * 
 * Another usecase for this macro is to surround the code of interest with begin
 * and end markers:
 * 
 * TL_TRACE("Calc Begin");
 * calc();
 * TL_TRACE("Calc End");
 * 
 * @param EVENT_NAME - String used as event name.
 *
 */
#define TL_TRACE(EVENT_NAME) \
    vitis::ai::tracelogging::Trace(EVENT_NAME)

/**
 * @brief This macro is dedicated to profile c++ block.
 *
 * To profile a block:
 * 
 * {
 *     TL_TRACE_BLOCK("BlockProf");
 *     // code to profile
 * }
 * 
 * This will generate two events: one at the beginning of the block (or at the
 * location of the macro inside the block) and one at the end of the block. The
 * events will be suffixed with: ": Begin" and ": End" respectively. For example
 * the event names for the code above will be: "BlockProf: Begin" and "BlockProf: End".
 * 
 * The event name is passed as an argument to this macro: EVENT_NAME. It is also
 * possible to supply additional event arguments by using c++ stream syntax:
 * 
 * {
 *     int val = 10;
 *     TL_TRACE_BLOCK("BlockProf") << "val=" << val;
 *     // code to profile
 * }
 * 
 * It is possible to use multiple traces for the same block:
 * 
 * {
 *     TL_TRACE_BLOCK("BlockProf_1");
 *     // code to profile
 *     TL_TRACE_BLOCK("BlockProf_2");
 *     // mode code to profile
 * }
 * 
 * There will be a separate set of Begin and End events for each trace macro
 * usage and all End events will be at the end of the block.
 * 
 * @param EVENT_NAME - String used as event name.
 *
 */
#define TL_TRACE_BLOCK(EVENT_NAME) \
    vitis::ai::tracelogging::Trace TL_VAR(EVENT_NAME); \
    vitis::ai::tracelogging::TraceBlock(EVENT_NAME, TL_VAR)

namespace vitis::ai::tracelogging {

struct Trace : public std::stringstream {
    Trace(std::string const & eventName);
    ~Trace();

    std::string eventName_;
};

struct TraceBlock : public Trace {
    TraceBlock(std::string const & eventName, Trace & end);
    ~TraceBlock();

    Trace & end_;
};

} // namespace vitis::ai::tracelogging

#else // TL_ENABLE_BUILD_FLAG

#define TL_NULL_STREAM !TL_ENABLE_BUILD_FLAG ? static_cast<void>(0) : std::stringstream()

#define TL_TRACE(EVENT_NAME) TL_NULL_STREAM
#define TL_TRACE_BLOCK(EVENT_NAME) TL_NULL_STREAM

#endif // TL_ENABLE_BUILD_FLAG
