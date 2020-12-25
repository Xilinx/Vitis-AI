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
// File Name: xclbinutil.cxx
// ============================================================================

#include "xclbin.h"
#include <map>
#include <vector>
#include <string.h>
#include <fstream>
#include <memory>

namespace XclBinUtil
{
  template<typename ... Args>

  std::string format(const std::string& format, Args ... args) 
  {
    size_t size = 1 + snprintf(nullptr, 0, format.c_str(), args ...);
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);

    return std::string(buf.get(), buf.get() + size);
  }

  std::string getCurrentTimeStamp();
  std::string getBaseFilename( const std::string &_fullPath );
  bool cmdLineSearch( int argc, char** argv, const char* check );
  bool stringEndsWith( const char* str, const char* ending );
  void mapArgs( std::map< std::string, std::string > & argDecoder, int argc, char** originalArgv, std::vector< std::string > & mappedArgv );
  std::ostream & data2hex( std::ostream & s, const unsigned char* value, size_t size );
  unsigned char hex2char( const unsigned char* hex );
  std::ostream & hex2data( std::ostream & s, const unsigned char* value, size_t size );
  uint64_t stringToUInt64( std::string _sInteger);
};


