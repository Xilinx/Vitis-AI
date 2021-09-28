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

#ifndef _XF_FINTECH_TRACE_H_
#define _XF_FINTECH_TRACE_H_

#include <fstream>
#include <iostream>

#include <stdarg.h>

#include "xf_fintech_types.hpp"

namespace xf {
namespace fintech {

/**
 * @class Trace
 * @brief Used to control debug trace output
 */
class Trace {
   public:
    /**
     * Allows the user to toggle the output of debug trace.
     *
     * @param[in] bEnabled true to enable trace output, false to disable trace
     * output.
     *
     */
    static void setEnabled(bool bEnabled);

    /**
     * Returns t
     *
     * @brief getEnabled true to enable trace output, false to disable trace
     * output.
     *
     */
    static bool getEnabled(void);

    /**
     * Prints a specified INFORMATIONAL string.
     * The output of this will be suppressed if tracing is disabled
     */
    static int printInfo(const char* fmt, ...);

    /**
     * Prints a specified ERROR string.
     * The output from this function will ALWAYS be output even is tracing is
     * disabled.
     */
    static int printError(const char* fmt, ...);

    /**
     * Prints out a textual string representation of the specified OpenCL error
     * code
     */
    static void printCLError(cl_int cl_error_code);

    /**
     * Prints out a textual string representation of the specified OptionType
     */
    static char* optionTypeToString(OptionType optionType);

    /**
     * Sets the primary (i.e. console) output stream to be used.  By default this
     * is std::cout
     */
    static void setConsoleOutputStream(std::ostream& outputStream);

    /**
     * Sets the secondary (i.e. file) output stream stream to be used.  By default
     * this is not used
     */
    static void setFileOutputStream(std::ofstream& outputStream);

   private:
    Trace();
    virtual ~Trace();

    static Trace* getInstance(void);

    int internalPrint(const char* fmt, va_list va_args);

    void internalSetEnabled(bool bEnabled);
    bool internalGetEnabled(void);

    static char* getCLErrorString(cl_int cl_error_code);

    static Trace* m_pInstance;

    std::ostream* m_pConsoleOutputStream;
    std::ofstream* m_pFileOutputStream;

    static const int BUFFER_SIZE = 256;
    char m_buffer[BUFFER_SIZE + 1];

    bool m_bEnabled;
};

} // end namespace fintech
} // end namespace xf

#endif //_XF_FINTECH_TRACE_H_
