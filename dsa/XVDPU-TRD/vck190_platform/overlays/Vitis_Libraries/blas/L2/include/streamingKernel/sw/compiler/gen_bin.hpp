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

#ifndef XF_BLAS_GEN_BIN_HPP
#define XF_BLAS_GEN_BIN_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include <cassert>
#include <stdio.h>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "host_types.hpp"
#include "ISA.hpp"

using namespace std;

namespace xf {
namespace blas {

// VLIV processing types

#define VERBOSE 1
template <typename t_DataType, unsigned int t_PageSize>
class Page {
   public:
    typedef array<t_DataType, t_PageSize> PageType;

   public:
    t_DataType& operator[](unsigned int p_Idx) { return m_page[p_Idx]; }
    Page() { m_page.fill(0); }

   private:
    array<t_DataType, t_PageSize> m_page;
};

class PageHandleDescriptor {
   public:
    unsigned int m_StartPage;
    unsigned int m_SizePages;
    // string m_HandleName;
   public:
    PageHandleDescriptor() : m_StartPage(0), m_SizePages(0) {}
    PageHandleDescriptor(unsigned int p_StartPage, unsigned int p_SizePages)
        : m_StartPage(p_StartPage), m_SizePages(p_SizePages) {}
    PageHandleDescriptor(const PageHandleDescriptor& p_Val)
        : m_StartPage(p_Val.m_StartPage), m_SizePages(p_Val.m_SizePages) {}
    bool operator<(const PageHandleDescriptor& p_Other) const { return (m_StartPage < p_Other.m_StartPage); }
};

template <typename t_DataType,
          unsigned int t_InstrOffsetBytes,
          unsigned int t_ResOffsetBytes,
          unsigned int t_DataOffsetBytes,
          unsigned int t_MaxNumInstrs,
          unsigned int t_PageSizeBytes,
          unsigned int t_MemWordBytes,
          unsigned int t_InstrBytes>
class Program {
   public:
    typedef struct MemWordType { uint8_t arr[t_MemWordBytes]; } MemWordType;
    typedef array<uint8_t, t_InstrBytes> InstrControlType;
    typedef vector<Page<uint8_t, t_PageSizeBytes> > PageVectorType;
    typedef Page<uint8_t, t_PageSizeBytes> PageType;
    typedef array<InstrControlType, t_MaxNumInstrs> InstrType;

   private:
    PageVectorType m_pageVector;
    unsigned int m_NumInstr;
    map<string, PageHandleDescriptor> m_Handles;

   private:
    // Utilities
    ifstream::pos_type getFileSize(string p_FileName) {
        ifstream in(p_FileName.c_str(), ifstream::ate | ifstream::binary);
        return in.tellg();
    }

   public:
    Program() : m_NumInstr(0) {
        // Reserve instruction and result pages
        m_pageVector.resize(t_DataOffsetBytes / t_PageSizeBytes);
    }
    Program(size_t p_NumPages) // Constructor typically used to store output from FPGA
        : m_NumInstr(0) {
        // Reserve space for entire output FPGA image
        m_pageVector.resize(p_NumPages);
    }
    void init(size_t p_NumPages) {
        m_NumInstr = 0;
        m_pageVector.resize(p_NumPages);
    }
    unsigned int
        /*
         * AlloPages : Will allocate required number of pages based on the number elements
         * to be allocated. Page table stores a handle : which is like a dictionary key and
         * identifier to hold information about allocations that are don or to be done. If
         * an allocation is made before with a given handle and New allocation is not made
         * if requested but the old one is used.
         */
        allocPages(string p_Handle, bool& p_NewAlloc, size_t p_NumElements) { // unit: t_DataType
        assert(p_NumElements > 0);
        // calculate number of additional pages required for allocating p_NumElements
        size_t l_numPages = (p_NumElements * sizeof(t_DataType) + t_PageSizeBytes - 1) / t_PageSizeBytes;
        unsigned int l_startPage = 0;
        // Create a page descriptor that will contains start address of the page
        // and the page size
        PageHandleDescriptor l_desc = m_Handles[p_Handle];
        if (l_desc.m_StartPage == 0) {
            l_startPage = m_pageVector.size();
            m_pageVector.resize(l_startPage + l_numPages);
            m_Handles[p_Handle] = PageHandleDescriptor(l_startPage, l_numPages);
            p_NewAlloc = true;
        } else {
            assert(l_numPages <= l_desc.m_SizePages);
            l_startPage = l_desc.m_StartPage;
            p_NewAlloc = false;
        }
        return (l_startPage);
    }
    t_DataType* getPageAddr(unsigned int p_PageIdx) {
        t_DataType* l_addr = (t_DataType*)&m_pageVector[p_PageIdx];
        return (l_addr);
    }
    uint8_t* getBaseInstrAddr() { return (uint8_t*)&m_pageVector[t_InstrOffsetBytes / t_PageSizeBytes]; }
    uint8_t* getBaseResAddr() { return (uint8_t*)&m_pageVector[t_ResOffsetBytes / t_PageSizeBytes]; }
    uint8_t* addInstr() {
        assert(m_NumInstr * sizeof(InstrControlType) <= t_PageSizeBytes);
        InstrControlType* l_instrBased = (InstrControlType*)&m_pageVector[t_InstrOffsetBytes / t_PageSizeBytes];
        uint8_t* l_instrAdd = (uint8_t*)&l_instrBased[m_NumInstr];
        ++m_NumInstr;
        return (l_instrAdd);
    }
    bool writeToBinFile(string p_FileName) {
        bool ok = false;
        ofstream l_of(p_FileName.c_str(), ios::binary);
        if (l_of.is_open()) {
            size_t l_sizeBytes = sizeof(m_pageVector[0]) * m_pageVector.size();
            l_of.write((char*)&m_pageVector[0], l_sizeBytes);
            if (l_of.tellp() == (unsigned)l_sizeBytes) {
                cout << "INFO: wrote " << l_sizeBytes << " bytes to " << p_FileName << "\n";
                ok = true;
            } else {
                cout << "ERROR: wrote only " << l_of.tellp() << " bytes to " << p_FileName << "\n";
            }
            l_of.close();
        }
        return (ok);
    }
    bool readFromBinFile(string p_FileName) {
        bool ok = false;
        // Bin file existence
        ifstream l_if(p_FileName.c_str(), ios::binary);
        if (l_if.is_open()) {
            // Bin file size
            size_t l_FileSize = getFileSize(p_FileName);
            cout << "INFO: loading " + p_FileName + " of size " << l_FileSize << "\n";
            assert(l_FileSize > 0);
            size_t l_FileSizeInPages = l_FileSize / sizeof(m_pageVector[0]);
            assert(l_FileSize % sizeof(m_pageVector[0]) == 0);

            // Bin file storage
            m_pageVector.reserve(l_FileSizeInPages);
            m_pageVector.resize(l_FileSizeInPages);

            // Read the bin file
            l_if.read((char*)&m_pageVector[0], l_FileSize);
            if (l_if) {
                cout << "INFO: loaded " << l_FileSize << " bytes from " << p_FileName << "\n";
                ok = true;
            } else {
                m_pageVector.clear();
                cout << "ERROR: loaded only " << l_if.gcount() << " bytes from " << p_FileName << "\n";
            }
            l_if.close();

        } else {
            cout << "ERROR: failed to open file " + p_FileName + "\n";
        }
        return (ok);
    }

    MemDesc getMemDesc() {
        assert(m_pageVector.size() > 0);
        MemDesc l_MemDesc(m_pageVector.size(), m_pageVector.data());
        return (l_MemDesc);
    }
};

template <typename t_DataType,
          unsigned int t_InstrOffsetBytes,
          unsigned int t_ResOffsetBytes,
          unsigned int t_DataOffsetBytes,
          unsigned int t_MaxNumInstrs,
          unsigned int t_PageSizeBytes,
          unsigned int t_MemWordBytes,
          unsigned int t_InstrBytes>
class GenControl {
   public:
    typedef Program<t_DataType,
                    t_InstrOffsetBytes,
                    t_ResOffsetBytes,
                    t_DataOffsetBytes,
                    t_MaxNumInstrs,
                    t_PageSizeBytes,
                    t_MemWordBytes,
                    t_InstrBytes>
        ProgramType;
    typedef MemInstr<t_InstrBytes> MemInstrType;
    typedef ControlInstr<t_InstrBytes> ControlInstrType;

   public:
    void addInstr(ProgramType& p_Program, bool p_IsLastOp, bool p_Noop) {
        // Instruction
        ControlInstrType l_controlInstr(p_IsLastOp, p_Noop);
        MemInstrType l_memInstr;
        l_controlInstr.store(l_memInstr);
        l_memInstr.storeMem(p_Program.addInstr());

        cout << "Added CONTROL  IsLastOp=" << p_IsLastOp << " Noop=" << p_Noop << "  ";
    }

    void show(ControlInstrType p_controlInstr) {
        bool l_isLastOp = p_controlInstr.m_IsLastOp, l_Noop = p_controlInstr.m_Noop;
        cout << "\n###########  Op Control  ###########\n"
             << "  IsLastOp=" << l_isLastOp << " Noop=" << l_Noop << "\n";
    }
};
}
}
#endif
