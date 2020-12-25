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
// File Name: xclbincat1.h
// ============================================================================

#ifndef __XCLBINCAT1_H_
#define __XCLBINCAT1_H_

#include <map>
#include <string>
#include <vector>
#include "xclbin.h"

namespace xclbincat1 {

class OptionParser {
  public:
    OptionParser();
    ~OptionParser();
    int parse( int argc, char** argv );

    bool isVerbose() { return m_verbose; };

  private:
    bool getKeyValuePair( const std::string & kvString, std::pair< std::string, std::string > & keyValue );
    int parseSegmentType(std::string _sSegmentType, std::string _sFile);

  public:
    // Segment types
    typedef enum {
      ST_BITSTREAM, 
      ST_CLEAR_BITSTREAM, 
      ST_FIRMWARE, 
      ST_SCHEDULER,   
      ST_BINARY_HEADER, 
      ST_META_DATA, 
      ST_MEM_TOPOLOGY, 
      ST_CONNECTIVITY,
      ST_IP_LAYOUT,
      ST_DEBUG_DATA,
      ST_DEBUG_IP_LAYOUT,
      ST_CLOCK_FREQ_TOPOLOGY,
      ST_MCS_PRIMARY,
      ST_MCS_SECONDARY,
      ST_BMC,
      ST_BUILD_METADATA,
      ST_KEYVALUE_METADATA,
      ST_USER_METADATA,
      ST_UNKNOWN                        
    } SegmentType;

    static SegmentType getSegmentType(const std::string _sSegmentType);

  public:
    std::vector< std::string > m_jsonfiles;
    std::vector< std::string > m_bitstreams;
    std::vector< std::string > m_clearstreams;
    std::vector< std::string > m_debugdata;
    std::vector< std::string > m_firmware;
    std::vector< std::string > m_scheduler;
    std::vector< std::string > m_memTopology;
    std::vector< std::string > m_connectivity;
    std::vector< std::string > m_ipLayout;
    std::vector< std::string > m_debugIpLayout;
    std::vector< std::string > m_clockFreqTopology;
    std::vector< std::pair< std::string, enum MCS_TYPE> > m_mcs;
    std::vector< std::string > m_bmc;
    std::vector< std::string > m_metadata;
    std::map< std::string, std::string > m_keyValuePairs;
    bool m_help;
    std::string m_binaryHeader;
    std::string m_output;
    void printHelp( char* program );

  private:
    bool m_verbose;
};

int execute( int argc, char** argv );

} // namespace xclbincat1

#endif // __XCLBINCAT1_H_



