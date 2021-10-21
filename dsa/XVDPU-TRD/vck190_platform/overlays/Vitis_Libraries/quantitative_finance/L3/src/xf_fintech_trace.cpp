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

#include "xf_fintech_trace.hpp"

using namespace xf::fintech;

/* SINGLETON */
Trace* Trace::m_pInstance = NULL;

Trace::Trace() {
    m_pConsoleOutputStream = &std::cout;
    m_pFileOutputStream = NULL;

    m_bEnabled = true;
}

Trace::~Trace() {
    if (m_pFileOutputStream != NULL) {
        m_pFileOutputStream->flush();
        m_pFileOutputStream->close();

        m_pFileOutputStream = NULL;
    }

    if (m_pConsoleOutputStream != NULL) {
        m_pConsoleOutputStream->flush();
    }
}

Trace* Trace::getInstance(void) {
    if (m_pInstance == NULL) {
        m_pInstance = new Trace();
    }

    return m_pInstance;
}

void Trace::setEnabled(bool bEnabled) {
    getInstance()->internalSetEnabled(bEnabled);
}

bool Trace::getEnabled(void) {
    return getInstance()->internalGetEnabled();
}

void Trace::setConsoleOutputStream(std::ostream& outputStream) {
    getInstance()->m_pConsoleOutputStream = &outputStream;
}

void Trace::setFileOutputStream(std::ofstream& outputStream) {
    getInstance()->m_pFileOutputStream = &outputStream;
}

int Trace::printInfo(const char* fmt, ...) {
    int retval = 0;
    va_list args;

    if (getEnabled()) {
        va_start(args, fmt);

        retval = getInstance()->internalPrint(fmt, args);

        va_end(args);
    }

    return retval;
}

int Trace::printError(const char* fmt, ...) {
    int retval = 0;
    va_list args;

    va_start(args, fmt);

    retval = getInstance()->internalPrint(fmt, args);

    va_end(args);

    return retval;
}

void Trace::printCLError(cl_int cl_error_code) {
    Trace::printError("[XLNX] ERROR - OpenCL call failed with error %s (%d)\n", getCLErrorString(cl_error_code),
                      cl_error_code);
}

int Trace::internalPrint(const char* fmt, va_list args) {
    int numChars = 0;

    numChars = vsnprintf(m_buffer, BUFFER_SIZE, fmt, args);

    if (m_pConsoleOutputStream != NULL) {
        m_pConsoleOutputStream->write(m_buffer, numChars);
        m_pConsoleOutputStream->flush();
    }

    if (m_pFileOutputStream != NULL) {
        m_pFileOutputStream->write(m_buffer, numChars);
        m_pFileOutputStream->flush();
    }

    return numChars;
}

void Trace::internalSetEnabled(bool bEnabled) {
    m_bEnabled = bEnabled;
}

bool Trace::internalGetEnabled(void) {
    return m_bEnabled;
}

#define STR_CASE(VAL)       \
    case (VAL): {           \
        pStr = (char*)#VAL; \
        break;              \
    }

char* Trace::getCLErrorString(cl_int cl_error_code) {
    char* pStr;

    switch (cl_error_code) {
        STR_CASE(CL_SUCCESS)
        STR_CASE(CL_DEVICE_NOT_FOUND)
        STR_CASE(CL_DEVICE_NOT_AVAILABLE)
        STR_CASE(CL_COMPILER_NOT_AVAILABLE)
        STR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        STR_CASE(CL_OUT_OF_RESOURCES)
        STR_CASE(CL_OUT_OF_HOST_MEMORY)
        STR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE)
        STR_CASE(CL_MEM_COPY_OVERLAP)
        STR_CASE(CL_IMAGE_FORMAT_MISMATCH)
        STR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        STR_CASE(CL_BUILD_PROGRAM_FAILURE)
        STR_CASE(CL_MAP_FAILURE)
        STR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        STR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        STR_CASE(CL_COMPILE_PROGRAM_FAILURE)
        STR_CASE(CL_LINKER_NOT_AVAILABLE)
        STR_CASE(CL_LINK_PROGRAM_FAILURE)
        STR_CASE(CL_DEVICE_PARTITION_FAILED)
        STR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)

        STR_CASE(CL_INVALID_VALUE)
        STR_CASE(CL_INVALID_DEVICE_TYPE)
        STR_CASE(CL_INVALID_PLATFORM)
        STR_CASE(CL_INVALID_DEVICE)
        STR_CASE(CL_INVALID_CONTEXT)
        STR_CASE(CL_INVALID_QUEUE_PROPERTIES)
        STR_CASE(CL_INVALID_COMMAND_QUEUE)
        STR_CASE(CL_INVALID_HOST_PTR)
        STR_CASE(CL_INVALID_MEM_OBJECT)
        STR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        STR_CASE(CL_INVALID_IMAGE_SIZE)
        STR_CASE(CL_INVALID_SAMPLER)
        STR_CASE(CL_INVALID_BINARY)
        STR_CASE(CL_INVALID_BUILD_OPTIONS)
        STR_CASE(CL_INVALID_PROGRAM)
        STR_CASE(CL_INVALID_PROGRAM_EXECUTABLE)
        STR_CASE(CL_INVALID_KERNEL_NAME)
        STR_CASE(CL_INVALID_KERNEL_DEFINITION)
        STR_CASE(CL_INVALID_KERNEL)
        STR_CASE(CL_INVALID_ARG_INDEX)
        STR_CASE(CL_INVALID_ARG_VALUE)
        STR_CASE(CL_INVALID_ARG_SIZE)
        STR_CASE(CL_INVALID_KERNEL_ARGS)
        STR_CASE(CL_INVALID_WORK_DIMENSION)
        STR_CASE(CL_INVALID_WORK_GROUP_SIZE)
        STR_CASE(CL_INVALID_WORK_ITEM_SIZE)
        STR_CASE(CL_INVALID_GLOBAL_OFFSET)
        STR_CASE(CL_INVALID_EVENT_WAIT_LIST)
        STR_CASE(CL_INVALID_EVENT)
        STR_CASE(CL_INVALID_OPERATION)
        STR_CASE(CL_INVALID_GL_OBJECT)
        STR_CASE(CL_INVALID_BUFFER_SIZE)
        STR_CASE(CL_INVALID_MIP_LEVEL)
        STR_CASE(CL_INVALID_GLOBAL_WORK_SIZE)
        STR_CASE(CL_INVALID_PROPERTY)
        STR_CASE(CL_INVALID_IMAGE_DESCRIPTOR)
        STR_CASE(CL_INVALID_COMPILER_OPTIONS)
        STR_CASE(CL_INVALID_LINKER_OPTIONS)
        STR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT)
        STR_CASE(CL_INVALID_PIPE_SIZE)
        STR_CASE(CL_INVALID_DEVICE_QUEUE)

        default: {
            pStr = (char*)"UNKNOWN";
            break;
        }
    }

    return pStr;
}

char* Trace::optionTypeToString(OptionType optionType) {
    char* pStr;

    switch (optionType) {
        STR_CASE(Call)
        STR_CASE(Put)

        default: {
            pStr = (char*)"UNKNOWN";
            break;
        }
    }

    return pStr;
}
