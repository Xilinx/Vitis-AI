/**
 * Copyright (C) 2016-2017 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

// ============================================================================
// COPYRIGHT NOTICE
// (c) Copyright 2017 Xilinx, Inc. All rights reserved.
//
// File Name: xclbinsplit1.cxx
// ============================================================================


#ifndef __XCLBINSPLIT1_H_
#define __XCLBINSPLIT1_H_

#include <string>
#include <vector>

namespace xclbinsplit1 {

class OptionParser {
  public:
    OptionParser();
    ~OptionParser();
    int parse( int argc, char** argv );
    void printHelp( char* program );

    std::string m_output;
    std::string m_input;
    std::string m_binaryHeader;
    bool m_verbose;
    bool m_help;
};

bool extract( const OptionParser& parser ); 
int execute( int argc, char** argv );

} // namespace xclbinsplit1


#endif // __XCLBINSPLIT1_H_



