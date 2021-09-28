/**********
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
 * **********/
/**
 *  @brief GEMX testcase generator
 *
 *  $DateTime: 2018/05/08 03:56:09 $
 */

#ifndef XF_BLAS_GEN_BIN_HPP
#define XF_BLAS_GEN_BIN_HPP

#include <stdio.h>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include "host_types.hpp"
#include "ddrType.hpp"
#include "kargs.hpp"
#include "xcl2.hpp"

////////////////////////  COMMON  ////////////////////////

// Common types
typedef BLAS_dataType FloatType;

// VLIV processing types
typedef xf::blas::Kargs<BLAS_dataType, BLAS_ddrWidth, BLAS_argInstrWidth, BLAS_argPipeline> KargsType;
typedef KargsType::OpType KargsOpType;
typedef KargsType::DdrFloatType DdrFloatType;
typedef xf::blas::ControlArgs ControlArgsType;

typedef xf::blas::DdrMatrixShape DdrMatrixShapeType;
typedef DdrMatrixShapeType::FormatType MatFormatType;
#define BLAS_maxNumInstr 64

#define VERBOSE 1
typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePointType;
inline void showTimeData(std::string p_Task, TimePointType& t1, TimePointType& t2, double* p_TimeMsOut = 0) {
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> l_durationSec = t2 - t1;
    double l_timeMs = l_durationSec.count() * 1e3;
    if (p_TimeMsOut) {
        *p_TimeMsOut = l_timeMs;
    }
    (VERBOSE > 0) && std::cout << p_Task << "  " << std::fixed << std::setprecision(6) << l_timeMs << " msec\n";
}

template <typename T, unsigned int t_PageSize>
class Page {
   public:
    typedef std::array<T, t_PageSize> PageType;

   private:
    PageType m_Page;

   public:
    FloatType& operator[](unsigned int p_Idx) { return m_Page[p_Idx]; }
    Page() {
        // m_Page.fill(0);
        for (unsigned int i = 0; i < t_PageSize; ++i) {
            m_Page[i] = 0;
        }
    }
};

class PageHandleDescriptor {
   public:
    unsigned int m_StartPage;
    unsigned int m_SizePages;
    // std::string m_HandleName;
   public:
    PageHandleDescriptor() : m_StartPage(0), m_SizePages(0) {}
    PageHandleDescriptor(unsigned int p_StartPage, unsigned int p_SizePages)
        : m_StartPage(p_StartPage), m_SizePages(p_SizePages) {}
    PageHandleDescriptor(const PageHandleDescriptor& p_Val)
        : m_StartPage(p_Val.m_StartPage), m_SizePages(p_Val.m_SizePages) {}
    bool operator<(const PageHandleDescriptor& p_Other) const { return (m_StartPage < p_Other.m_StartPage); }
};

typedef std::array<uint8_t, BLAS_instructionSizeBytes> InstrControlType;
typedef std::vector<Page<uint8_t, BLAS_pageSizeBytes>, aligned_allocator<Page<uint8_t, BLAS_pageSizeBytes> > >
    PageVectorType;
typedef Page<uint8_t, BLAS_pageSizeBytes> PageType;

template <typename t_FloatType // to simplify client-side interfaces
          >
class Program {
   public:
    typedef std::array<InstrControlType, BLAS_maxNumInstr> InstrType;

   private:
    PageVectorType m_PageVector;
    unsigned int m_NumInstr;
    std::map<std::string, PageHandleDescriptor> m_Handles;

   private:
    // Utilities
    std::ifstream::pos_type getFileSize(std::string p_FileName) {
        std::ifstream in(p_FileName.c_str(), std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    }

   public:
    Program() : m_NumInstr(0) {
        // Reserve instruction and result pages
        m_PageVector.resize(BLAS_dataPage);
    }
    Program(size_t p_NumPages) // Constructor typically used to store output from FPGA
        : m_NumInstr(0) {
        // Reserve space for entire output FPGA image
        m_PageVector.resize(p_NumPages);
    }
    void init(size_t p_NumPages) {
        m_NumInstr = 0;
        m_PageVector.resize(p_NumPages);
    }
    unsigned int
        /*
         * AlloPages : Will allocate required number of pages based on the number elements
         * to be allocated. Page table stores a handle : which is like a dictionary key and
         * identifier to hold information about allocations that are don or to be done. If
         * an allocation is made before with a given handle and New allocation is not made
         * if requested but the old one is used.
         */
        allocPages(std::string p_Handle, bool& p_NewAlloc, size_t p_NumElements) { // unit: t_FloatType
        assert(p_NumElements > 0);
        // calculate number of additional pages required for allocating p_NumElements
        size_t l_numPages = (p_NumElements * sizeof(t_FloatType) + BLAS_pageSizeBytes - 1) / BLAS_pageSizeBytes;
        unsigned int l_startPage = 0;
        // Create a page descriptor that will contains start address of the page
        // and the page size
        PageHandleDescriptor l_desc = m_Handles[p_Handle];
        if (l_desc.m_StartPage == 0) {
            l_startPage = m_PageVector.size();
            m_PageVector.resize(l_startPage + l_numPages);
            m_Handles[p_Handle] = PageHandleDescriptor(l_startPage, l_numPages);
            p_NewAlloc = true;
        } else {
            assert(l_numPages <= l_desc.m_SizePages);
            l_startPage = l_desc.m_StartPage;
            p_NewAlloc = false;
        }
        // std::cout << "  DEBUG allocPages Start page for " << p_Handle << " is " << l_startPage << "\n";
        return (l_startPage);
    }
    t_FloatType* getPageAddr(unsigned int p_PageIdx) {
        t_FloatType* l_addr = (t_FloatType*)&m_PageVector[p_PageIdx];
        return (l_addr);
    }
    DdrFloatType* getBaseInstrAddr() { return (DdrFloatType*)&m_PageVector[BLAS_codePage]; }
    DdrFloatType* getBaseResAddr() { return (DdrFloatType*)&m_PageVector[BLAS_resPage]; }
    DdrFloatType* addInstr() {
        assert(m_NumInstr * sizeof(InstrControlType) <= BLAS_pageSizeBytes);
        InstrControlType* l_instrBased = (InstrControlType*)&m_PageVector[BLAS_codePage];
        DdrFloatType* l_instrAdd = (DdrFloatType*)&l_instrBased[m_NumInstr];
        ++m_NumInstr;
        return (l_instrAdd);
    }
    bool writeToBinFile(std::string p_FileName) {
        bool ok = false;
        std::ofstream l_of(p_FileName.c_str(), std::ios::binary);
        if (l_of.is_open()) {
            size_t l_sizeBytes = sizeof(m_PageVector[0]) * m_PageVector.size();
            l_of.write((char*)&m_PageVector[0], l_sizeBytes);
            if (l_of.tellp() == (unsigned)l_sizeBytes) {
                std::cout << "INFO: wrote " << l_sizeBytes << " bytes to " << p_FileName << "\n";
                ok = true;
            } else {
                std::cout << "ERROR: wrote only " << l_of.tellp() << " bytes to " << p_FileName << "\n";
            }
            l_of.close();
        }
        return (ok);
    }
    bool readFromBinFile(std::string p_FileName) {
        bool ok = false;
        // Bin file existence
        std::ifstream l_if(p_FileName.c_str(), std::ios::binary);
        if (l_if.is_open()) {
            // Bin file size
            size_t l_FileSize = getFileSize(p_FileName);
            std::cout << "INFO: loading " + p_FileName + " of size " << l_FileSize << "\n";
            assert(l_FileSize > 0);
            size_t l_FileSizeInPages = l_FileSize / sizeof(m_PageVector[0]);
            assert(l_FileSize % sizeof(m_PageVector[0]) == 0);

            // Bin file storage
            m_PageVector.reserve(l_FileSizeInPages);
            m_PageVector.resize(l_FileSizeInPages);

            // Read the bin file
            l_if.read((char*)&m_PageVector[0], l_FileSize);
            if (l_if) {
                std::cout << "INFO: loaded " << l_FileSize << " bytes from " << p_FileName << "\n";
                ok = true;
            } else {
                m_PageVector.clear();
                std::cout << "ERROR: loaded only " << l_if.gcount() << " bytes from " << p_FileName << "\n";
            }
            l_if.close();

        } else {
            std::cout << "ERROR: failed to open file " + p_FileName + "\n";
        }
        return (ok);
    }
    xf::blas::MemDesc getMemDesc(void) {
        assert(m_PageVector.size() > 0);
        xf::blas::MemDesc l_MemDesc(m_PageVector.size(), m_PageVector.data());
        return (l_MemDesc);
    }
};

typedef Program<BLAS_dataType> ProgramType;

////////////////////////  CONTROL  ////////////////////////
class GenControl {
   public:
    void addInstr(ProgramType& p_Program, bool p_IsLastOp, bool p_Noop) {
        // Instruction
        ControlArgsType l_controlArgs(p_IsLastOp, p_Noop);
        KargsType l_kargs;
        l_kargs.setControlArgs(l_controlArgs);
        l_kargs.store(p_Program.addInstr(), 0);

        std::cout << "Added CONTROL  IsLastOp=" << p_IsLastOp << " Noop=" << p_Noop << "  ";
    }

    void show(ProgramType& p_Program, ControlArgsType p_ControlArgs) {
        bool l_isLastOp = p_ControlArgs.m_IsLastOp, l_Noop = p_ControlArgs.m_Noop;
        std::cout << "\n###########  Op Control  ###########\n"
                  << "  IsLastOp=" << l_isLastOp << " Noop=" << l_Noop << "\n";
    }
};

#endif
