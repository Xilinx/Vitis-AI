/*
 *Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
/**
 * @file compressApp.hpp
 * @brief Header for compressApp  host functionality
 *
 * This file is part of Vitis Data Compression Library host code.
 */

#ifndef _XFCOMPRESSION_COMPRESS_APP_HPP_
#define _XFCOMPRESSION_COMPRESS_APP_HPP_

#include "compressBase.hpp"
#include "cmdlineparser.h"

class compressApp {
   private:
    std::string m_uncompressed_file;
    std::string m_uncompressed_filelist;
    std::string m_compressed_file;
    std::string m_compressed_filelist;
    std::string m_golden_file;
    std::string m_device_id;
    std::string m_xclbin;
    std::string m_mcr;
    std::string m_cu;
    std::string m_filelist;
    std::string m_outFile_name;
    bool m_list_flow{false};
    bool m_compress_decompress{false};
    size_t m_inputsize;
    uint64_t m_enbytes;
    uint64_t m_debytes;
    uint16_t m_stdBSize = (1 << 15);

   public:
    sda::utils::CmdLineParser m_parser;
    std::string m_extn{".compressed"};
    bool m_enableProfile;
    bool m_isSeq;

    /** Switch flow
     * true means compress,
     * false means decompress
     */
    bool m_compress_flow;

    /**
     * @brief Initialize compressApp content
     *
     */
    compressApp(const int argc, char** argv, bool is_seq, bool enable_profile);
    void parser(const int argc, char** argv);
    int validate(std::string& inFile, std::string& outFile);
    // Get Input Filename
    std::string getXclbin(void) const;
    uint8_t getDeviceId(void) const;
    uint8_t getMCR(void) const;
    std::string& getInFileName(void);
    void inputFilePreCheck(std::ifstream& inStream);
    void getListFilenames(std::string& filelist, std::vector<std::string>& fname_vec);
    void printTestSummaryHeader();
    void printTestSummaryFooter(const std::string& testFile);
    // -c -d -l -t
    void run(compressBase* b, uint16_t maxCR = MAX_CR_DEFAULT);

    void runCompress(compressBase* b, const std::string&);

    void runDecompress(compressBase* b, const std::string&);
};
#endif // _XFCOMPRESSION_COMPRESS_APP_HPP_
